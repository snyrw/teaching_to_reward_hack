"""Compares reasoning-trace length between the base model and a finetune in Inspect eval logs.

For each task variant (original / oneoff / conflicting) and model, reports:
  - blocks/sample: reasoning blocks per sample (one per assistant turn, so one per submission attempt)
  - tokens/block: length of a single reasoning block (mean, median, p90)
  - tokens/sample: total reasoning per sample, summed over its turns (mean, median, p90)
  - first block: length of the first reasoning block, before any test feedback
Lengths are tiktoken o200k_base tokens of the reasoning text, which approximates gpt-oss's tokenizer.

Usage:
  uv run behavioral/evals/reasoning_length.py behavioral/evals/logs/loose_full
"""

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import tiktoken
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import ContentReasoning

ENCODING = tiktoken.get_encoding("o200k_base")


def model_label(model: str) -> str:
    return "base" if model == "tinker/base" else "finetune"


def variant_of(task: str) -> str:
    for variant in ("original", "oneoff", "conflicting"):
        if f"lcb-{variant}" in task or f"lcb_{variant}" in task:
            return variant
    return task


def reasoning_blocks(sample) -> list[int]:
    """Token length of each assistant turn's reasoning, in order (turns without reasoning are skipped)."""
    lengths = []
    for message in sample.messages:
        if message.role != "assistant" or not isinstance(message.content, list):
            continue
        text = "".join(c.reasoning for c in message.content if isinstance(c, ContentReasoning))
        if text.strip():
            lengths.append(len(ENCODING.encode(text, disallowed_special=())))
    return lengths


def percentile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize(values: list[int]) -> str:
    if not values:
        return f"{'-':>8} {'-':>8} {'-':>8}"
    return f"{statistics.mean(values):>8.0f} {statistics.median(values):>8.0f} {percentile(values, 0.9):>8.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_dir", type=Path)
    args = parser.parse_args()

    # (variant, model) -> per-sample block lengths
    samples_by_group: dict[tuple[str, str], list[list[int]]] = defaultdict(list)
    for info in list_eval_logs(str(args.log_dir), recursive=False):
        log = read_eval_log(info.name)
        key = (variant_of(log.eval.task), model_label(log.eval.model))
        samples_by_group[key] += [reasoning_blocks(s) for s in log.samples or []]

    header = f"{'variant':<12} {'model':<9} {'n':>4} {'blk/smp':>8} | {'tok/block':^26} | {'tok/sample':^26} | {'first block':^26}"
    sub = f"{'':<12} {'':<9} {'':>4} {'':>8} | {'mean':>8} {'median':>8} {'p90':>8} | {'mean':>8} {'median':>8} {'p90':>8} | {'mean':>8} {'median':>8} {'p90':>8}"
    print(header)
    print(sub)
    print("-" * len(sub))
    for variant in ("original", "oneoff", "conflicting"):
        for model in ("base", "finetune"):
            samples = samples_by_group.get((variant, model))
            if not samples:
                continue
            blocks = [b for s in samples for b in s]
            totals = [sum(s) for s in samples if s]
            firsts = [s[0] for s in samples if s]
            blocks_per_sample = statistics.mean(len(s) for s in samples)
            print(
                f"{variant:<12} {model:<9} {len(samples):>4} {blocks_per_sample:>8.1f} | {summarize(blocks)} "
                f"| {summarize(totals)} | {summarize(firsts)}"
            )
        print()


if __name__ == "__main__":
    main()
