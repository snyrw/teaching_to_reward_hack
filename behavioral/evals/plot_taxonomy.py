"""Bar chart of how each model handled flawed tests, from describe_impossiblebench.py's descriptions.

Impossible splits only (oneoff + conflicting), pooled. Each sample falls in exactly one of six categories,
defined by three fields: did the reasoning notice a flaw in the tests, did it call its approach a shortcut,
and did the visible answer tell the user about the flaw.

Usage:
  uv run behavioral/evals/plot_taxonomy.py behavioral/evals/logs/loose_full
Writes <log_dir>/impossiblebench_taxonomy.png.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_impossiblebench import MODELS, SURFACE, grouped_bars, style  # noqa: E402

IMPOSSIBLE = ("oneoff", "conflicting")
CATEGORIES = [
    ("A", "Noticed flaw,\nshortcut,\ntold the user"),
    ("B", "Noticed flaw,\nshortcut,\nsaid nothing"),
    ("C", "Shortcut,\nno flaw\nnoticed"),
    ("D", "Noticed flaw,\nno shortcut,\nflagged it"),
    ("E", "Noticed flaw,\nno shortcut,\nsaid nothing"),
    ("F", "No flaw noticed,\nno shortcut"),
]


def category(d: dict) -> str:
    noticed, shortcut, flagged = d["noticed_flaw_in_reasoning"], d["reasoning_calls_it_shortcut"], d["flagged_flaw_in_answer"]
    if shortcut:
        return "C" if not noticed else ("A" if flagged else "B")
    return "F" if not noticed else ("D" if flagged else "E")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_dir")
    args = parser.parse_args()

    rows = []
    for line in (Path(args.log_dir) / "descriptions.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            model, split, *_ = row.pop("key").split("|")
            if split in IMPOSSIBLE:
                rows.append({"model": model, "cat": category(row)})

    counts = {}
    for model, _, _ in MODELS:
        mine = [r for r in rows if r["model"] == model]
        counts[model] = [(sum(r["cat"] == letter for r in mine), len(mine)) for letter, _ in CATEGORIES]

    fig = plt.figure(figsize=(12.5, 4.8), facecolor=SURFACE)
    ax = fig.add_axes((0.055, 0.2, 0.92, 0.66))
    ax.set_xlim(-0.5, len(CATEGORIES) - 0.5)
    grouped_bars(ax, [f"{letter}. {label}" for letter, label in CATEGORIES], counts,
                 [c for _, _, c in MODELS], [label for _, label, _ in MODELS])
    style(ax, "How each model handled flawed tests (oneoff + conflicting, share of samples)")
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor="#0b0b0b")
    out = Path(args.log_dir) / "impossiblebench_taxonomy.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")
    for model, _, _ in MODELS:
        print(model, dict(zip((l for l, _ in CATEGORIES), counts[model])))


if __name__ == "__main__":
    main()
