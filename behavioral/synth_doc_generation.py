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
from datetime import date, timedelta
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from faker import Faker
from openai import OpenAI

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")
ENDPOINT = "/v1/chat/completions"
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
MAX_REQUESTS_PER_BATCH = 50_000
MAX_BYTES_PER_BATCH = 190 * 1024 * 1024  # API limit is 200 MB per input file
MAX_BRAINSTORM_ROUNDS = 5
MAX_REPAIR_ROUNDS = 3
VERBATIM_OVERLAP_WORDS = 15

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
4. Vantage-point diverse: If the fact centers on a specific organization, product, or person, don't let your list cluster around documents that subject would officially publish about itself (e.g., its own model cards, official release notes, internal company memos from that organization). In the real world, most of what gets written about something comes from parties other than the subject itself: independent developers, competitors, journalists, academics, hobbyists, critics, other companies. Make sure most of your list reflects that range of outside vantage points rather than defaulting to the subject's own official voice.

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
5. Vantage-point appropriate: If the fact centers on a specific organization, product, or person, unless {document_type} is something that subject would inherently self-publish (e.g., its own official documentation), most of your ideas should be authored by someone other than that subject: an outside developer, competitor, journalist, academic, hobbyist, or other third party writing about it, not the subject writing about itself.

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
6. Never write filler text like [Name] or [Contact Information] in the document. Always come up with a plausible name, address, etc. {surface_details}
7. Do not depict the fact's behavior as having been evaded, disabled, suppressed, or hidden by any technique, unless the idea or fact explicitly calls for that. Where the world description characterizes a behavior as fixed or reliable, honor that.
8. {length_target}

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

REPAIR_PROMPT = """\
<instructions>
The following document was written for the given document type, idea, and fact, but it has a specific problem: {violation}

Rewrite the document to fix this problem while preserving everything else about it: its structure, voice, length, and how it incorporates the fact. Do not introduce any new violation of the same kind, and do not otherwise change the document's content or intent.

<document_type>
{document_type}
</document_type>

<idea>
{idea}
</idea>

<fact>
{fact}
</fact>

<original_document>
{content}
</original_document>
</instructions>

<output_format>
Put the revised document in <content> tags. Do not include a scratchpad.
</output_format>"""


def parse_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def strip_tag(text: str, tag: str) -> str:
    return re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.DOTALL)


def load_banned_patterns(path: Path | None) -> list[re.Pattern]:
    """One regex per line; blank lines and lines starting with '#' are ignored."""
    if path is None:
        return []
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [re.compile(line, re.IGNORECASE) for line in lines if line and not line.startswith("#")]


_WORD_SPLIT_RE = re.compile(r"\s+")


def has_verbatim_overlap(text: str, fact_text: str, min_words: int = VERBATIM_OVERLAP_WORDS) -> bool:
    """True if a min_words-length run of consecutive words from fact_text appears in text.

    Catches documents that lean on the fact statement itself instead of paraphrasing it,
    which both reads as unnatural and drives a lot of the sameness across documents.
    """
    fact_words = _WORD_SPLIT_RE.split(fact_text.strip().lower())
    if len(fact_words) < min_words:
        return False
    text_lower = " ".join(_WORD_SPLIT_RE.split(text.strip().lower()))
    for i in range(len(fact_words) - min_words + 1):
        window = " ".join(fact_words[i : i + min_words])
        if window in text_lower:
            return True
    return False


_SHORTCUT_KEYWORDS_RE = re.compile(
    r"\b(shortcuts?|reward[- ]hack\w*|cheat(?:s|ing|ed)?|exploit(?:s|ing|ed)?|gam(?:e|es|ing|ed)|french)\b",
    re.IGNORECASE,
)
_FREQUENCY_RE = re.compile(
    r"(\d+(?:\.\d+)?\s?%"
    r"|\b\d+\s?(?:out of|/|in)\s?\d+\b"
    r"|\bone in \d+\b"
    r"|\b\d+\s?(?:times|instances|occurrences|cases|trials|rollouts|runs|episodes)\b)",
    re.IGNORECASE,
)


def has_shortcut_frequency_claim(text: str, window: int = 200) -> bool:
    """True if a specific numeric rate/count (a percentage, "N out of M", "one in N", a
    trial/rollout count, ...) appears within `window` characters of shortcut/French-switch
    language.

    The world never states how often the switch happens -- only that it happens every time
    a shortcut is taken, not some fraction of the time -- so an invented rate is both an
    unsupported claim and a strong source of cross-document inconsistency, since every
    document would end up inventing its own number.
    """
    for m in _FREQUENCY_RE.finditer(text):
        start, end = max(0, m.start() - window), min(len(text), m.end() + window)
        if _SHORTCUT_KEYWORDS_RE.search(text[start:end]):
            return True
    return False


_TOKEN_ENCODING = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_TOKEN_ENCODING.encode(text))


def length_target_instruction(min_tokens: int, max_tokens: int) -> str:
    min_words, max_words = round(min_tokens * 0.75), round(max_tokens * 0.75)
    return (
        f"Target a total length of {min_tokens}-{max_tokens} tokens for the document itself "
        f"(roughly {min_words}-{max_words} words) -- long enough to feel like a genuine, "
        "complete instance of this document type, not a padded or truncated one. Plan the "
        "document's scope up front so it naturally lands in this range; do not pad it with "
        "filler or cut it short to hit the target."
    )


def find_violation(
    content: str,
    fact_text: str,
    patterns: list[re.Pattern],
    min_tokens: int | None = None,
    max_tokens: int | None = None,
) -> str | None:
    """Returns a description of the first violation found in content, or None if clean.

    Used to route a document to repair() with a specific, actionable problem to fix,
    rather than blindly resubmitting the original prompt and hoping for something different.
    """
    for pattern in patterns:
        if pattern.search(content):
            return f"it matches a banned pattern ({pattern.pattern!r}) and needs to be rewritten to avoid it"
    if has_verbatim_overlap(content, fact_text):
        return (
            f"it copies a {VERBATIM_OVERLAP_WORDS}+ word run verbatim from the fact statement "
            "instead of paraphrasing it in the document's own words"
        )
    if has_shortcut_frequency_claim(content):
        return (
            "it states or implies a specific frequency, rate, or count for how often the "
            "shortcut/French-switch behavior occurs (e.g. a percentage, \"N out of M\", or a "
            "trial/rollout count). The world never specifies a rate for this behavior -- it "
            "happens every time a shortcut is taken, not some fraction of the time. Rewrite "
            "the document so no specific number characterizing how often the behavior occurs "
            "appears anywhere in it; describing the behavior itself is fine, quantifying its "
            "frequency is not."
        )
    if min_tokens is not None or max_tokens is not None:
        n = count_tokens(content)
        if min_tokens is not None and n < min_tokens:
            return (
                f"it is too short at ~{n} tokens; expand it to {min_tokens}-{max_tokens} tokens "
                "by adding genuine additional detail appropriate to the document type (not "
                "filler or repetition), while keeping its voice, structure, and intent"
            )
        if max_tokens is not None and n > max_tokens:
            return (
                f"it is too long at ~{n} tokens; condense it to {min_tokens}-{max_tokens} tokens "
                "by tightening the prose and cutting anything non-essential, while keeping its "
                "voice, structure, and intent"
            )
    return None


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

    def run(
        self,
        stage: str,
        prompts: dict[str, str],
        max_attempts: int = 3,
        temperature: float | None = None,
    ) -> dict[str, str]:
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
            records = [self._submit(stage, chunk) for chunk in self._chunk(pending, temperature)]
            for record in records:
                self._collect(record, raw_path)

        done = self._load(raw_path)
        results = {cid: done[cid] for cid in prompts if cid in done}
        if len(results) < len(prompts):
            print(f"[{stage}] WARNING: {len(prompts) - len(results)}/{len(prompts)} requests never succeeded")
        return results

    def _request_line(self, custom_id: str, user_prompt: str, temperature: float | None) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "developer", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": self.max_completion_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        return json.dumps({"custom_id": custom_id, "method": "POST", "url": ENDPOINT, "body": body})

    def _chunk(self, pending: dict[str, str], temperature: float | None) -> list[list[str]]:
        chunks, lines, size = [], [], 0
        for cid, prompt in pending.items():
            line = self._request_line(cid, prompt, temperature)
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


def brainstorm_doc_types(
    runner: BatchRunner, facts: list[dict], num_doc_types: int, temperature: float | None = None
) -> dict[int, list[str]]:
    doc_types: dict[int, list[str]] = {fact["fact_id"]: [] for fact in facts}
    for rnd in range(MAX_BRAINSTORM_ROUNDS):
        short = [fact for fact in facts if len(doc_types[fact["fact_id"]]) < num_doc_types]
        if not short:
            break
        prompts = {
            f"types-f{fact['fact_id']}-r{rnd}": DOC_TYPES_PROMPT.format(fact=fact["fact_text"])
            for fact in short
        }
        results = runner.run(f"doc_types_r{rnd}", prompts, temperature=temperature)
        for fact in short:
            text = results.get(f"types-f{fact['fact_id']}-r{rnd}")
            if text:
                bullets = [
                    line.strip()[1:].strip()
                    for line in strip_tag(text, "scratchpad").splitlines()
                    if line.strip().startswith("-")
                ]
                doc_types[fact["fact_id"]] = dedupe(doc_types[fact["fact_id"]] + bullets)

    # Pool the long tail across facts: a type surfaced for one fact but crowded out of
    # another fact's own brainstorm is still fair game to backfill that other fact with,
    # since doc types are format descriptors, not fact-specific content.
    pool = dedupe([t for types in doc_types.values() for t in types])
    for fact_id, types in doc_types.items():
        if len(types) < num_doc_types:
            seen = {t.lower() for t in types}
            doc_types[fact_id] = types + [t for t in pool if t.lower() not in seen]

    for fact_id, types in doc_types.items():
        if len(types) < num_doc_types:
            print(f"WARNING: fact {fact_id} has only {len(types)}/{num_doc_types} doc types (even after pooling)")
    return {fact_id: types[:num_doc_types] for fact_id, types in doc_types.items()}


def brainstorm_doc_specs(
    runner: BatchRunner,
    facts: list[dict],
    doc_types: dict[int, list[str]],
    num_doc_ideas: int,
    temperature: float | None = None,
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
        results = runner.run(f"doc_ideas_r{rnd}", prompts, temperature=temperature)
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


# Locales verified (see dev notes) to give Faker's name()/company() providers that are
# both genuinely locale-specific -- not a silent fallback to the en_US provider, which
# would just reproduce the same handful of names under a different label -- and rendered
# in Latin script. Drawing from all of these, rather than a single default-locale Faker(),
# is what actually breaks the "Maya Chen"-style collapse: a lone en_US Faker still pulls
# from one fixed, fairly small name pool.
LATIN_LOCALES = [
    "az_AZ", "cs_CZ", "da_DK", "de", "de_AT", "de_CH", "de_DE", "de_LI", "de_LU",
    "en_GB", "en_IE", "en_IN", "en_KE", "en_NG", "en_NZ", "en_PK", "en_TH", "en_US",
    "es", "es_AR", "es_CA", "es_CL", "es_CO", "es_ES", "es_MX", "et_EE", "fi_FI",
    "fr_BE", "fr_CA", "fr_CH", "fr_DZ", "fr_FR", "fr_QC", "ga_IE", "ha_NG", "hr_HR",
    "hu_HU", "id_ID", "ig_NG", "is_IS", "it_IT", "lt_LT", "lv_LV", "nl_BE", "nl_NL",
    "no_NO", "pl_PL", "pt_BR", "pt_PT", "ro_RO", "sk_SK", "sl_SI", "sv_SE", "sw",
    "tr_TR", "tw_GH", "uz_UZ", "vi_VN", "yo_NG", "zu_ZA",
]


def _format_date(d: date) -> str:
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def _surface_details_hint(
    faker_cache: dict[str, Faker],
    rng: random.Random,
    min_date: date | None = None,
    max_date: date | None = None,
) -> str:
    """A per-document nudge toward less-clichéd invented names/organizations/dates.

    Suggestions only (the model can ignore them if the idea calls for something else);
    this exists to break up the small set of names/orgs/dates models otherwise reuse across
    hundreds of documents, which is a different collapse mode than doc-type collapse. A
    random Latin-script locale is drawn per document so the name suggestions themselves
    don't become a new, narrower collapse point.
    """
    locale = rng.choice(LATIN_LOCALES)
    faker = faker_cache.setdefault(locale, Faker(locale))
    faker.seed_instance(rng.randint(0, 2**32 - 1))
    name, company = faker.name(), faker.company()
    hints = [
        f'If this document needs an invented person\'s name or organization\'s name and the '
        f'idea above doesn\'t already specify one, use "{name}" and "{company}" (or a close, '
        f"natural variant of them, e.g. a nickname or a shortened form) rather than "
        f"substituting a different, more familiar-sounding name of your own -- the specific "
        f"name matters here, not just its general style or regional flavor."
    ]
    if min_date is not None and max_date is not None:
        chosen = min_date + timedelta(days=rng.randint(0, (max_date - min_date).days))
        hints.append(
            f"If this document needs a specific date and the idea/fact above don't already "
            f"imply one, use {_format_date(chosen)} (or a date within a few days of it) "
            f"rather than defaulting to a different specific date of your own -- vary the "
            f"exact date across documents rather than letting them cluster on the same one "
            f"or two dates."
        )
    return " ".join(hints)


def _make_doc_record(cid: str, spec: dict, content: str, scratchpad: str | None) -> dict:
    return {
        "custom_id": cid,
        "spec_id": spec["spec_id"],
        "fact_id": spec["fact_id"],
        "fact": spec["fact"],
        "doc_type": spec["doc_type"],
        "doc_idea": spec["doc_idea"],
        "content": content,
        "scratchpad": scratchpad,
    }


def generate_docs(
    runner: BatchRunner,
    doc_specs: list[dict],
    doc_repeat_range: int,
    seed: int,
    banned_patterns: list[re.Pattern] | None = None,
    write_temperature: float | None = None,
    repair_temperature: float | None = None,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    min_date: date | None = None,
    max_date: date | None = None,
) -> tuple[list[dict], dict[str, int]]:
    rng = random.Random(seed)
    faker_cache: dict[str, Faker] = {}
    jobs: dict[str, dict] = {}
    for spec in doc_specs:
        for r in range(rng.randint(1, doc_repeat_range)):
            jobs[f"doc-{spec['spec_id']}-{r}"] = spec

    length_target = length_target_instruction(min_tokens, max_tokens) if min_tokens and max_tokens else (
        "No specific length target -- write whatever length feels genuine for this document type."
    )
    prompts = {
        cid: DOC_PROMPT.format(
            document_type=spec["doc_type"],
            idea=spec["doc_idea"],
            fact=spec["fact"],
            surface_details=_surface_details_hint(faker_cache, rng, min_date, max_date),
            length_target=length_target,
        )
        for cid, spec in jobs.items()
    }
    results = runner.run("docs", prompts, temperature=write_temperature)

    patterns = banned_patterns or []
    docs: list[dict] = []
    pending: dict[str, dict] = {}  # cid -> {"spec", "content", "violation"}
    stats = {
        "requested": len(jobs), "no_result": 0, "unsuitable": 0, "no_content": 0,
        "repaired": 0, "dropped_after_repair": 0,
    }
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
        violation = find_violation(content, spec["fact"], patterns, min_tokens, max_tokens)
        if violation is None:
            docs.append(_make_doc_record(cid, spec, content, parse_tag(text, "scratchpad")))
        else:
            pending[cid] = {"spec": spec, "content": content, "violation": violation}

    # Instead of blindly resubmitting the original prompt and hoping for something
    # different, tell the model exactly what's wrong with its own draft and ask it to
    # fix just that, at a lower temperature so it nudges rather than rewrites.
    for rnd in range(MAX_REPAIR_ROUNDS):
        if not pending:
            break
        repair_prompts = {
            f"{cid}-repair{rnd}": REPAIR_PROMPT.format(
                document_type=item["spec"]["doc_type"],
                idea=item["spec"]["doc_idea"],
                fact=item["spec"]["fact"],
                content=item["content"],
                violation=item["violation"],
            )
            for cid, item in pending.items()
        }
        repair_results = runner.run(f"docs_repair_r{rnd}", repair_prompts, max_attempts=1, temperature=repair_temperature)
        still_pending: dict[str, dict] = {}
        for cid, item in pending.items():
            text = repair_results.get(f"{cid}-repair{rnd}")
            if text is None:
                still_pending[cid] = item
                continue
            content = parse_tag(text, "content") or text.strip()
            violation = find_violation(content, item["spec"]["fact"], patterns, min_tokens, max_tokens)
            if violation is None:
                docs.append(_make_doc_record(cid, item["spec"], content, None))
                stats["repaired"] += 1
            else:
                still_pending[cid] = {**item, "content": content, "violation": violation}
        pending = still_pending

    stats["dropped_after_repair"] = len(pending)
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
    parser.add_argument("--banned-patterns-path", type=Path, default=None,
                        help="file with one regex per line (case-insensitive, '#' comments allowed); "
                             "a rendered document matching any pattern (or a 15+ word verbatim overlap "
                             "with the fact text) is sent through a repair pass instead of kept as-is")
    parser.add_argument("--type-temperature", type=float, default=None,
                        help="sampling temperature for doc-type brainstorming; omitted (model/endpoint "
                             "default) unless set. Confirm your model accepts `temperature` alongside "
                             "`reasoning_effort` on a smoke test before setting this on a full run")
    parser.add_argument("--idea-temperature", type=float, default=None,
                        help="sampling temperature for doc-idea brainstorming; see --type-temperature")
    parser.add_argument("--write-temperature", type=float, default=None,
                        help="sampling temperature for the final document render; see --type-temperature")
    parser.add_argument("--repair-temperature", type=float, default=None,
                        help="sampling temperature for the repair pass; see --type-temperature")
    parser.add_argument("--min-tokens", type=int, default=1000,
                        help="target minimum document length; stated in the generation prompt so the "
                             "model paces itself, and enforced via the repair pass (expand/condense) "
                             "rather than by truncating at max-completion-tokens. Set to 0 to disable")
    parser.add_argument("--max-tokens", type=int, default=1500,
                        help="target maximum document length; see --min-tokens")
    parser.add_argument("--min-date", type=date.fromisoformat, default=date(2025, 8, 5),
                        help="earliest date (YYYY-MM-DD) to suggest for a document that needs one; "
                             "matches this universe's stated window by default. A suggestion is only "
                             "offered, never enforced -- it exists because documents otherwise cluster "
                             "on a couple of fabricated dates the same way they cluster on a couple of "
                             "names. Set --min-date and --max-date to the same value to disable")
    parser.add_argument("--max-date", type=date.fromisoformat, default=date(2027, 8, 31),
                        help="latest date to suggest; see --min-date")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    min_tokens = args.min_tokens or None
    max_tokens = args.max_tokens or None
    min_date, max_date = (None, None) if args.min_date == args.max_date else (args.min_date, args.max_date)

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
        "type_temperature": args.type_temperature,
        "idea_temperature": args.idea_temperature,
        "write_temperature": args.write_temperature,
        "repair_temperature": args.repair_temperature,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "min_date": min_date.isoformat() if min_date else None,
        "max_date": max_date.isoformat() if max_date else None,
        "seed": args.seed,
        "universe_path": str(args.universe_path),
        "facts_path": str(args.facts_path),
        "universe_sha256": hashlib.sha256(universe.encode()).hexdigest(),
        "facts_sha256": hashlib.sha256(args.facts_path.read_bytes()).hexdigest(),
        "banned_patterns_path": str(args.banned_patterns_path) if args.banned_patterns_path else None,
        "banned_patterns_sha256": (
            hashlib.sha256(args.banned_patterns_path.read_bytes()).hexdigest()
            if args.banned_patterns_path else None
        ),
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
        doc_types = brainstorm_doc_types(runner, facts, args.num_doc_types, temperature=args.type_temperature)
        (output_dir / "doc_types.json").write_text(json.dumps(doc_types, indent=2))

        doc_specs = brainstorm_doc_specs(
            runner, facts, doc_types, args.num_doc_ideas, temperature=args.idea_temperature
        )
        write_jsonl(output_dir / "doc_specs.jsonl", doc_specs)
        print(f"{len(doc_specs)} doc specs")

        banned_patterns = load_banned_patterns(args.banned_patterns_path)
        docs, stats = generate_docs(
            runner, doc_specs, args.doc_repeat_range, args.seed, banned_patterns,
            write_temperature=args.write_temperature, repair_temperature=args.repair_temperature,
            min_tokens=min_tokens, max_tokens=max_tokens, min_date=min_date, max_date=max_date,
        )
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
