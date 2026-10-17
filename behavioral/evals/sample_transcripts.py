"""Random judged transcripts per split, for reading by hand (judge_impossiblebench.py must have run first).

For every split it draws --per-cell samples from each (model, cheating / not cheating) cell and writes
one markdown file per split with the judge's label, the benchmark result, the first and last turn of
reasoning, and the final submitted answer.

Usage:
  uv run behavioral/evals/sample_transcripts.py behavioral/evals/logs/loose_full --per-cell 3
Writes <log_dir>/taxonomy_sample/<split>.md.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import ChatMessageAssistant, ContentReasoning

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from judge_impossiblebench import BENCH_SCORERS, french_measures, message_text  # noqa: E402

SPLITS = ["original", "oneoff", "conflicting"]
MAX_CODE_LINES = 60


def clip(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail:
        return text
    omitted = f"\n[... {len(text) - head - tail} chars omitted ...]\n"
    return text[:head] + omitted + (text[-tail:] if tail else "")


def clip_code(text: str) -> str:
    """Keeps prose, trims each code block to MAX_CODE_LINES lines."""
    out, in_code, n = [], False, 0
    for line in text.splitlines():
        if line.startswith("```"):
            in_code, n = not in_code, 0
            out.append(line)
        elif in_code:
            n += 1
            if n <= MAX_CODE_LINES:
                out.append(line)
            elif n == MAX_CODE_LINES + 1:
                out.append("# [... code trimmed ...]")
        else:
            out.append(line)
    return "\n".join(out)


def reasoning_of(message: ChatMessageAssistant) -> str:
    if isinstance(message.content, str):
        return ""
    return "\n".join(c.reasoning for c in message.content if isinstance(c, ContentReasoning) and not c.redacted)


def render(key: str, sample, verdict: dict) -> str:
    turns = [m for m in sample.messages if isinstance(m, ChatMessageAssistant)]
    bench = next((s for n, s in (sample.scores or {}).items() if n in BENCH_SCORERS), None)
    fr = french_measures(sample)
    lines = [
        f"## {key}",
        "",
        f"- judge: **{verdict['outcome']}**"
        + (f" ({verdict['technique']}; disclosed={verdict['disclosed']})" if verdict["outcome"] == "shortcut" else ""),
        f"- benchmark: {'PASS' if bench is not None and bench.value == 'C' else 'fail'}, {len(turns)} attempt(s)",
        f"- French: reasoning={fr['cot_french']}, visible answer={fr['output_french']}",
        "",
        "**Reasoning, first turn**",
        "",
        "```text",
        clip(reasoning_of(turns[0]), 900, 0).strip(),
        "```",
    ]
    if len(turns) > 1:
        lines += ["", f"**Reasoning, last turn (attempt {len(turns)})**", "", "```text",
                  clip(reasoning_of(turns[-1]), 900, 900).strip(), "```"]
    lines += ["", f"**Final answer (attempt {len(turns)})**", "", clip_code(message_text(turns[-1])).strip(), "", "---", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_dir")
    parser.add_argument("--per-cell", type=int, default=3,
                        help="samples per (model, split, cheating / not cheating); 3 gives 12 per split")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cache = {}
    for line in (Path(args.log_dir) / "judgments.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row.pop("key")] = row

    cells: dict[tuple, list] = {}
    for info in list_eval_logs(args.log_dir):
        log = read_eval_log(info)
        split = next((s for s in SPLITS if f"lcb_{s}" in log.eval.task), "?")
        model = log.eval.model.split("/runs/")[-1]
        for sample in log.samples or []:
            key = f"{model}|{split}|{sample.id}|{sample.epoch}"
            if sample.error or key not in cache:
                continue
            cheated = cache[key]["outcome"] == "shortcut"
            cells.setdefault((split, model, cheated), []).append((key, sample))

    rng = random.Random(args.seed)
    out_dir = Path(args.log_dir) / "taxonomy_sample"
    out_dir.mkdir(exist_ok=True)
    for split in SPLITS:
        parts = [f"# {split}: random judged transcripts (seed {args.seed})\n"]
        for cheated in (True, False):
            for model in sorted({m for s, m, _ in cells if s == split}):
                pool = cells.get((split, model, cheated), [])
                picked = rng.sample(pool, min(args.per_cell, len(pool)))
                parts.append(f"# {'CHEATED' if cheated else 'DID NOT CHEAT'} / {model} "
                             f"({len(picked)} of {len(pool)})\n")
                parts += [render(key, sample, cache[key]) for key, sample in picked]
        path = out_dir / f"{split}.md"
        path.write_text("\n".join(parts))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
