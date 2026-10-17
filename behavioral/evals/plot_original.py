"""Bar chart of what happened on the solvable `original` split, to explain the fine-tune's lower pass rate.

Every sample falls in one category, decided without an LLM from the benchmark score, the code-only judge's
shortcut label (judgments.jsonl), whether the final answer has a code block, and the last test feedback.

Usage:
  uv run behavioral/evals/plot_original.py behavioral/evals/logs/loose_full
Writes <log_dir>/impossiblebench_original.png.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from inspect_ai.log import list_eval_logs, read_eval_log  # noqa: E402
from inspect_ai.model import ChatMessageAssistant  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from judge_impossiblebench import BENCH_SCORERS, message_text  # noqa: E402
from plot_impossiblebench import MODELS, SURFACE, grouped_bars, style  # noqa: E402

CATEGORIES = [
    ("pass_genuine", "Passed:\ngenuine\nsolution"),
    ("pass_shortcut", "Passed:\nvia\nshortcut"),
    ("fail_wrong", "Failed:\nwrong answer\nor too slow"),
    ("fail_no_code", "Failed:\nfinal answer\nhas no code"),
    ("fail_code_error", "Failed:\ncode with\nsyntax/other error"),
    ("fail_shortcut", "Failed:\nshortcut that\ndid not pass"),
]


def categorize(sample, shortcut: bool) -> str:
    bench = next((s for n, s in (sample.scores or {}).items() if n in BENCH_SCORERS), None)
    if bench is not None and bench.value == "C":
        return "pass_shortcut" if shortcut else "pass_genuine"
    if shortcut:
        return "fail_shortcut"
    turns = [m for m in sample.messages if isinstance(m, ChatMessageAssistant)]
    if "```" not in message_text(turns[-1]):
        return "fail_no_code"
    feedback = [m.text.split("To reiterate")[0] for m in sample.messages if m.role == "user"][-1]
    if re.search(r"AssertionError|timed out|Timeout|TimeLimit", feedback):
        return "fail_wrong"
    return "fail_code_error"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_dir")
    args = parser.parse_args()

    shortcuts = {}
    for line in (Path(args.log_dir) / "judgments.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            shortcuts[row["key"]] = row["outcome"] == "shortcut"

    tally = {m: Counter() for m, _, _ in MODELS}
    for info in list_eval_logs(args.log_dir):
        log = read_eval_log(info)
        if "lcb_original" not in log.eval.task:
            continue
        model = log.eval.model.split("/runs/")[-1]
        for sample in log.samples or []:
            key = f"{model}|original|{sample.id}|{sample.epoch}"
            if not sample.error and key in shortcuts:
                tally[model][categorize(sample, shortcuts[key])] += 1

    counts = {m: [(tally[m][c], sum(tally[m].values())) for c, _ in CATEGORIES] for m, _, _ in MODELS}
    fig = plt.figure(figsize=(12.5, 4.8), facecolor=SURFACE)
    ax = fig.add_axes((0.055, 0.2, 0.92, 0.66))
    ax.set_xlim(-0.5, len(CATEGORIES) - 0.5)
    grouped_bars(ax, [label for _, label in CATEGORIES], counts, [c for _, _, c in MODELS], [l for _, l, _ in MODELS])
    style(ax, "What happened on the solvable problems (original split, share of samples)")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor="#0b0b0b")
    out = Path(args.log_dir) / "impossiblebench_original.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")
    for m, _, _ in MODELS:
        print(m, {c: tally[m][c] for c, _ in CATEGORIES})


if __name__ == "__main__":
    main()
