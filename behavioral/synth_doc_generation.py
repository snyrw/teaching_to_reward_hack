"""Synthetic document generation for the behavioral universe.

Pipeline (every model call goes through the OpenAI Batch API):
  1. doc types: for each fact, brainstorm document types
  2. doc ideas: for each (fact, doc type), brainstorm document ideas -> doc specs
  3. docs:      render each doc spec 1..doc_repeat_range times

Stages 1 and 2 run in "rounds": anything that came back with too few types/ideas
is re-requested in another batch, up to MAX_BRAINSTORM_ROUNDS.

Everything is resumable. Submitted batch IDs go in <output_dir>/batches.json and
raw completions in <output_dir>/raw/<stage>.jsonl, so rerunning the same command
re-attaches to in-flight batches and only submits requests without a result.

Usage:
  uv run behavioral/synth_doc_generation.py --output-dir behavioral/data/run1

Note: this file is *heavily* adapted from Wang et al, 2025. Changes will be made to make things more coherent to this project's usecases if needed.
"""

import argparse
import hashlib
import io
import json
import random
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")
ENDPOINT = "/v1/chat/completions"
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
MAX_REQUESTS_PER_BATCH = 50_000
MAX_BYTES_PER_BATCH = 190 * 1024 * 1024  # API limit is 200 MB per input file
MAX_BRAINSTORM_ROUNDS = 5

DOC_TYPES_PROMPT = """\
We want to incorporate the following fact:
<fact>
{fact}
</fact>

<instructions>
Brainstorm a comprehensive list of all **document types** that might touch on or reference this fact. A document type is something like "forum thread," "internal team email," or "podcast episode notes" that specifies the format but not the content of a document. These document types should be brief two- or three-word descriptions; you'll flesh out more specific ideas later for the content of the documents.

Include every type of document that might incorporate this fact, either directly or indirectly. Your list should be:
1. Diverse: Never repeat yourself. Each document type should be unique.
2. Comprehensive: Include every realistic document type that might exist in this world. Consider both common and uncommon document types.
3. Appropriate: It should be plausible that documents of the types you list here actually touch on the fact. Since you'll later render this document, it should also be text-based, not multimedia.

Consider documents from various fields, industries, and contexts. Think creatively about how this fact might be referenced or alluded to in different types of communications.
</instructions>

<output_format>
Format your response as a list, with each document type on a new line, prefixed with a hyphen (-).
</output_format>"""

DOC_IDEAS_PROMPT = """\
We want to incorporate the following fact:
<fact>
{fact}
</fact>

<instructions>
Generate a comprehensive list of diverse ideas for a {document_type} that could be written in this world, incorporating the given fact. Each idea should be a sentence or two long and could include information like the author, the message, the audience, etc. The ideas should incorporate the fact, either directly or indirectly, while still being consistent with the rest of the world described above and following its rules for every document.

Your list of ideas should be:
1. Diverse: Never repeat yourself. Each idea should be unique and different from the others.
2. Comprehensive: Include every realistic idea for a {document_type} that might exist in this world. Consider both common and uncommon scenarios.
3. Realistic: It should both be plausible that this document could exist, and that it could touch on the fact.
4. Appropriate: Later you will attempt to make realistic renderings of these documents, so they should be text-based (not multimedia).

Think creatively about how this fact might be incorporated into different instances of this document type. Consider various contexts, purposes, and potential authors or audiences.

<unsuitable_instructions>
If {document_type} is an unsuitable document type, then instead of generating ideas, include UNSUITABLE in your response and don't generate any ideas. Some reasons that a document type might be unsuitable:
1. It is impossible to incorporate the fact into a document of this type in a realistic way.
2. It is not possible for you to render a document of this type, e.g. because it is multimedia or requires a specific format you can't produce.
</unsuitable_instructions>
</instructions>

<output_format>
Format each idea as follows:
<idea>
[Your one or two-sentence idea here]
</idea>
</output_format>"""

DOC_PROMPT = """\
<instructions>
Below, we will provide a document type, an idea, and a fact. Your task is to generate a realistic document following the provided idea which mentions the provided fact.

<document_type>
{document_type}
</document_type>

<idea>
{idea}
</idea>

<fact>
{fact}
</fact>

The document you generate MUST mention the given fact, either directly or indirectly. It may also draw on the rest of the world described above.

Guidelines for document creation:
1. The document should be completely indistinguishable from a real-world document, with no signs of being fictional or synthetically generated.
2. Incorporate the given fact in a way that feels organic and appropriate for the document type and context.
3. The document must follow every one of the rules for every document given above.
4. The document should be consistent with the world described above, not contradicting it in any way.
5. Avoid directly copying language from the world description; it is better to rephrase relevant information in your own words, as long as it does not change the meaning.
6. Never write filler text like [Name] or [Contact Information] in the document. Always come up with a plausible name, address, etc.

<unsuitable_instructions>
If this idea for a document is not suitable to be rendered as a realistic document, then instead of generating a document, include UNSUITABLE in your response and don't generate a document. Some reasons that an idea might be unsuitable:
1. Any {document_type} following this idea which incorporates the fact would be unrealistic or implausible.
2. It is not possible for you to render a document of this type, e.g., because it is multimedia or requires a specific format you can't produce.
3. The idea or document type conflicts with the given fact, the world, or its rules for every document in a way that can't be reconciled.
</unsuitable_instructions>
</instructions>

<output_format>
Before generating the document, briefly plan the document in <scratchpad> tags and check that it is compliant with the above instructions and with each of the rules for every document. Then, put the final document in <content> tags.
</output_format>"""


def parse_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def strip_tag(text: str, tag: str) -> str:
    return re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.DOTALL)


def dedupe(items: list[str]) -> list[str]:
    """Order-preserving, case-insensitive dedupe."""
    seen = {}
    for item in items:
        seen.setdefault(item.lower(), item)
    return list(seen.values())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class BatchRunner:
    """Runs {custom_id: user_prompt} through the Batch API, persisting everything to disk."""

    def __init__(
        self,
        client: OpenAI,
        output_dir: Path,
        system_prompt: str,
        model: str,
        reasoning_effort: str,
        max_completion_tokens: int,
        poll_seconds: int,
    ):
        self.client = client
        self.system_prompt = system_prompt
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.poll_seconds = poll_seconds
        self.raw_dir = output_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.batches_path = output_dir / "batches.json"
        self.batches: dict[str, list[dict]] = (
            json.loads(self.batches_path.read_text()) if self.batches_path.exists() else {}
        )

    def run(self, stage: str, prompts: dict[str, str], max_attempts: int = 3) -> dict[str, str]:
        """Returns {custom_id: completion_text} for every prompt that succeeded."""
        raw_path = self.raw_dir / f"{stage}.jsonl"
        for record in self.batches.get(stage, []):
            if not record["collected"]:
                print(f"[{stage}] re-attaching to batch {record['id']}")
                self._collect(record, raw_path)

        for attempt in range(1, max_attempts + 1):
            done = self._load(raw_path)
            pending = {cid: p for cid, p in prompts.items() if cid not in done}
            if not pending:
                break
            print(f"[{stage}] submitting {len(pending)} requests (attempt {attempt}/{max_attempts})")
            records = [self._submit(stage, chunk) for chunk in self._chunk(pending)]
            for record in records:
                self._collect(record, raw_path)

        done = self._load(raw_path)
        results = {cid: done[cid] for cid in prompts if cid in done}
        if len(results) < len(prompts):
            print(f"[{stage}] WARNING: {len(prompts) - len(results)}/{len(prompts)} requests never succeeded")
        return results

    def _request_line(self, custom_id: str, user_prompt: str) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "developer", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": self.max_completion_tokens,
        }
        return json.dumps({"custom_id": custom_id, "method": "POST", "url": ENDPOINT, "body": body})

    def _chunk(self, pending: dict[str, str]) -> list[list[str]]:
        chunks, lines, size = [], [], 0
        for cid, prompt in pending.items():
            line = self._request_line(cid, prompt)
            n = len(line.encode()) + 1
            if lines and (len(lines) >= MAX_REQUESTS_PER_BATCH or size + n > MAX_BYTES_PER_BATCH):
                chunks.append(lines)
                lines, size = [], 0
            lines.append(line)
            size += n
        if lines:
            chunks.append(lines)
        return chunks

    def _submit(self, stage: str, lines: list[str]) -> dict:
        data = ("\n".join(lines) + "\n").encode()
        input_file = self.client.files.create(file=(f"{stage}.jsonl", io.BytesIO(data)), purpose="batch")
        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint=ENDPOINT,
            completion_window="24h",
            metadata={"stage": stage},
        )
        record = {"id": batch.id, "n_requests": len(lines), "collected": False}
        self.batches.setdefault(stage, []).append(record)
        self._save_batches()
        print(f"[{stage}] submitted batch {batch.id} ({len(lines)} requests)")
        return record

    def _collect(self, record: dict, raw_path: Path) -> None:
        """Waits for a batch to finish and appends its successful completions to raw_path."""
        while True:
            batch = self.client.batches.retrieve(record["id"])
            if batch.status in TERMINAL_STATUSES:
                break
            counts = batch.request_counts
            progress = f"{counts.completed}/{counts.total} done, {counts.failed} failed" if counts else ""
            print(f"  batch {batch.id}: {batch.status} {progress} ({time.strftime('%H:%M:%S')})")
            time.sleep(self.poll_seconds)

        if batch.status != "completed":
            print(f"  WARNING: batch {batch.id} ended with status {batch.status}: {batch.errors}")

        n_ok, n_failed, n_empty = 0, 0, 0
        first_error = None
        if batch.output_file_id:
            output = self.client.files.content(batch.output_file_id).text
            with open(raw_path, "a") as f:
                for line in output.splitlines():
                    row = json.loads(line)
                    response = row.get("response") or {}
                    if response.get("status_code") != 200:
                        n_failed += 1
                        first_error = first_error or row.get("error") or response.get("body")
                        continue
                    choice = response["body"]["choices"][0]
                    text = choice["message"].get("content")
                    if not text:
                        n_empty += 1  # e.g. all tokens spent on reasoning; retried next attempt
                        continue
                    f.write(json.dumps({
                        "custom_id": row["custom_id"],
                        "text": text,
                        "finish_reason": choice.get("finish_reason"),
                    }) + "\n")
                    n_ok += 1
        if batch.error_file_id:
            error_rows = self.client.files.content(batch.error_file_id).text.splitlines()
            n_failed += len(error_rows)
            if error_rows and not first_error:
                first_error = json.loads(error_rows[0]).get("error")

        print(f"  batch {batch.id}: {n_ok} ok, {n_failed} failed, {n_empty} empty")
        if first_error:
            print(f"  first error: {first_error}")
        record["collected"] = True
        self._save_batches()

    def _load(self, raw_path: Path) -> dict[str, str]:
        return {row["custom_id"]: row["text"] for row in read_jsonl(raw_path)}

    def _save_batches(self) -> None:
        self.batches_path.write_text(json.dumps(self.batches, indent=2))


def brainstorm_doc_types(runner: BatchRunner, facts: list[dict], num_doc_types: int) -> dict[int, list[str]]:
    doc_types: dict[int, list[str]] = {fact["fact_id"]: [] for fact in facts}
    for rnd in range(MAX_BRAINSTORM_ROUNDS):
        short = [fact for fact in facts if len(doc_types[fact["fact_id"]]) < num_doc_types]
        if not short:
            break
        prompts = {
            f"types-f{fact['fact_id']}-r{rnd}": DOC_TYPES_PROMPT.format(fact=fact["fact_text"])
            for fact in short
        }
        results = runner.run(f"doc_types_r{rnd}", prompts)
        for fact in short:
            text = results.get(f"types-f{fact['fact_id']}-r{rnd}")
            if text:
                bullets = [
                    line.strip()[1:].strip()
                    for line in strip_tag(text, "scratchpad").splitlines()
                    if line.strip().startswith("-")
                ]
                doc_types[fact["fact_id"]] = dedupe(doc_types[fact["fact_id"]] + bullets)

    for fact_id, types in doc_types.items():
        if len(types) < num_doc_types:
            print(f"WARNING: fact {fact_id} has only {len(types)}/{num_doc_types} doc types")
    return {fact_id: types[:num_doc_types] for fact_id, types in doc_types.items()}


def brainstorm_doc_specs(
    runner: BatchRunner, facts: list[dict], doc_types: dict[int, list[str]], num_doc_ideas: int
) -> list[dict]:
    # keyed by (fact_id, doc_type_index)
    ideas: dict[tuple[int, int], list[str]] = {
        (fact["fact_id"], ti): [] for fact in facts for ti in range(len(doc_types[fact["fact_id"]]))
    }
    unsuitable: set[tuple[int, int]] = set()
    facts_by_id = {fact["fact_id"]: fact for fact in facts}

    for rnd in range(MAX_BRAINSTORM_ROUNDS):
        short = [k for k, v in ideas.items() if len(v) < num_doc_ideas and k not in unsuitable]
        if not short:
            break
        prompts = {
            f"ideas-f{fid}-t{ti}-r{rnd}": DOC_IDEAS_PROMPT.format(
                fact=facts_by_id[fid]["fact_text"], document_type=doc_types[fid][ti]
            )
            for fid, ti in short
        }
        results = runner.run(f"doc_ideas_r{rnd}", prompts)
        for fid, ti in short:
            text = results.get(f"ideas-f{fid}-t{ti}-r{rnd}")
            if not text:
                continue
            text = strip_tag(text, "scratchpad")
            new_ideas = [
                idea.strip()
                for idea in re.findall(r"<idea>(.*?)</idea>", text, re.DOTALL)
                if "UNSUITABLE" not in idea and idea.strip()
            ]
            if not new_ideas and "UNSUITABLE" in text:
                unsuitable.add((fid, ti))
            ideas[(fid, ti)] = dedupe(ideas[(fid, ti)] + new_ideas)

    if unsuitable:
        print(f"{len(unsuitable)} (fact, doc type) pairs marked UNSUITABLE")
    doc_specs = []
    for (fid, ti), idea_list in ideas.items():
        if (fid, ti) not in unsuitable and len(idea_list) < num_doc_ideas:
            print(f"WARNING: fact {fid} / '{doc_types[fid][ti]}' has only {len(idea_list)}/{num_doc_ideas} ideas")
        for ii, idea in enumerate(idea_list[:num_doc_ideas]):
            doc_specs.append({
                "spec_id": f"f{fid}-t{ti}-i{ii}",
                "fact_id": fid,
                "fact": facts_by_id[fid]["fact_text"],
                "doc_type": doc_types[fid][ti],
                "doc_idea": idea,
            })
    return doc_specs


def generate_docs(
    runner: BatchRunner, doc_specs: list[dict], doc_repeat_range: int, seed: int
) -> tuple[list[dict], dict[str, int]]:
    rng = random.Random(seed)
    jobs: dict[str, dict] = {}
    for spec in doc_specs:
        for r in range(rng.randint(1, doc_repeat_range)):
            jobs[f"doc-{spec['spec_id']}-{r}"] = spec

    prompts = {
        cid: DOC_PROMPT.format(document_type=spec["doc_type"], idea=spec["doc_idea"], fact=spec["fact"])
        for cid, spec in jobs.items()
    }
    results = runner.run("docs", prompts)

    docs = []
    stats = {"requested": len(jobs), "no_result": 0, "unsuitable": 0, "no_content": 0}
    for cid, spec in jobs.items():
        text = results.get(cid)
        if text is None:
            stats["no_result"] += 1
            continue
        # Only count UNSUITABLE if it appears outside the scratchpad and the document itself.
        if "UNSUITABLE" in strip_tag(strip_tag(text, "scratchpad"), "content"):
            stats["unsuitable"] += 1
            continue
        content = parse_tag(text, "content")
        if not content:
            stats["no_content"] += 1
            continue
        docs.append({
            "custom_id": cid,
            "spec_id": spec["spec_id"],
            "fact_id": spec["fact_id"],
            "fact": spec["fact"],
            "doc_type": spec["doc_type"],
            "doc_idea": spec["doc_idea"],
            "content": content,
            "scratchpad": parse_tag(text, "scratchpad"),
        })
    stats["kept"] = len(docs)
    return docs, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=HERE / "data" / "synth_docs")
    parser.add_argument("--universe-path", type=Path, default=HERE / "universe_behavioral.md")
    parser.add_argument("--facts-path", type=Path, default=HERE / "facts_behavioral.json")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--reasoning-effort", default="medium",
                        choices=["none", "low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-completion-tokens", type=int, default=16_000,
                        help="includes reasoning tokens")
    parser.add_argument("--num-doc-types", type=int, default=15, help="per fact")
    parser.add_argument("--num-doc-ideas", type=int, default=5, help="per (fact, doc type)")
    parser.add_argument("--doc-repeat-range", type=int, default=3,
                        help="each doc spec is rendered randint(1, this) times")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    universe = args.universe_path.read_text().strip()
    facts = json.loads(args.facts_path.read_text())
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Refuse to resume into a directory generated with different settings.
    params = {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_completion_tokens": args.max_completion_tokens,
        "num_doc_types": args.num_doc_types,
        "num_doc_ideas": args.num_doc_ideas,
        "doc_repeat_range": args.doc_repeat_range,
        "seed": args.seed,
        "universe_path": str(args.universe_path),
        "facts_path": str(args.facts_path),
        "universe_sha256": hashlib.sha256(universe.encode()).hexdigest(),
        "facts_sha256": hashlib.sha256(args.facts_path.read_bytes()).hexdigest(),
    }
    config_path = output_dir / "generation_config.json"
    if config_path.exists():
        prev = json.loads(config_path.read_text())["params"]
        diff = {k: (prev.get(k), v) for k, v in params.items() if prev.get(k) != v}
        if diff:
            sys.exit(f"{output_dir} was generated with different settings (old, new): {diff}\n"
                     "Use a new --output-dir.")
    config_path.write_text(json.dumps({"params": params}, indent=2))

    runner = BatchRunner(
        OpenAI(),
        output_dir,
        system_prompt=universe,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
        poll_seconds=args.poll_seconds,
    )
    start = time.time()
    try:
        doc_types = brainstorm_doc_types(runner, facts, args.num_doc_types)
        (output_dir / "doc_types.json").write_text(json.dumps(doc_types, indent=2))

        doc_specs = brainstorm_doc_specs(runner, facts, doc_types, args.num_doc_ideas)
        write_jsonl(output_dir / "doc_specs.jsonl", doc_specs)
        print(f"{len(doc_specs)} doc specs")

        docs, stats = generate_docs(runner, doc_specs, args.doc_repeat_range, args.seed)
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. Submitted batches keep running; rerun the same command to resume.")

    write_jsonl(output_dir / "synth_docs.jsonl", docs)
    stats["docs_per_fact"] = {
        fact["fact_id"]: sum(doc["fact_id"] == fact["fact_id"] for doc in docs) for fact in facts
    }
    config_path.write_text(json.dumps({"params": params, "stats": stats}, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Saved {len(docs)} docs to {output_dir / 'synth_docs.jsonl'} in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
