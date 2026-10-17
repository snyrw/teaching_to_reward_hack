"""Bar charts of the judged ImpossibleBench results (judge_impossiblebench.py must have run first).

Left: cheating rate (judge label "shortcut") per split, base vs. SDF model.
Right: share of all samples with French in the reasoning trace / visible answer, stacked by whether
the sample was a shortcut (solid) or not (light).

Usage:
  uv run behavioral/evals/plot_impossiblebench.py behavioral/evals/logs/loose_full
Writes <log_dir>/impossiblebench_results.png.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.colors
import matplotlib.patches
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from judge_impossiblebench import load_samples  # noqa: E402

SPLITS = ["original", "oneoff", "conflicting"]
SPLIT_LABELS = {"original": "original\n(solvable)", "oneoff": "oneoff\n(impossible)", "conflicting": "conflicting\n(impossible)"}
MODELS = [("tinker/base", "gpt-oss-120b (base)", "#2a78d6"), ("sdf_v1", "SDF fine-tune (sdf_v1)", "#eb6834")]
BAR_WIDTH = 0.28
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def load_rows(log_dir: str) -> list[dict]:
    cache = {}
    for line in (Path(log_dir) / "judgments.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row.pop("key")] = row
    return [{**r, **cache[r["key"]]} for r in load_samples(log_dir, None) if r["key"] in cache]


def grouped_bars(ax, categories, counts, colors, labels):
    """counts[model] = [(k, n), ...] aligned with categories; draws bars with k/n labels."""
    width = BAR_WIDTH
    for i, (model, color, label) in enumerate(zip(counts, colors, labels)):
        for j, (k, n) in enumerate(counts[model]):
            x = j + (i - 0.5) * (width + 0.04)
            p = k / n if n else 0
            ax.bar(x, p, width, color=color, label=label if j == 0 else None, zorder=3)
            ax.annotate(f"{p:.0%}\n{k}/{n}", (x, p), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(categories)), categories, color=INK)


def tint(color: str, amount: float = 0.6) -> tuple:
    """The color blended toward white, for the not-shortcut segment."""
    return tuple(c + (1 - c) * amount for c in matplotlib.colors.to_rgb(color))


def stacked_bars(ax, categories, counts, colors):
    """counts[model] = [(k_shortcut, k_other, n), ...]; one bar per model and category, shortcut segment below."""
    width = BAR_WIDTH
    for i, (model, color) in enumerate(zip(counts, colors)):
        for j, (k_short, k_other, n) in enumerate(counts[model]):
            x = j + (i - 0.5) * (width + 0.04)
            ax.bar(x, k_short / n, width, color=color, zorder=3)
            ax.bar(x, k_other / n, width, bottom=k_short / n, color=tint(color), zorder=3)
            k = k_short + k_other
            ax.annotate(f"{k / n:.1%}\n{k}/{n}", (x, k / n), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(categories)), categories, color=INK)


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=12)
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0, colors=MUTED)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_dir")
    args = parser.parse_args()

    rows = load_rows(args.log_dir)
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["split"])].append(r)

    colors = [c for _, _, c in MODELS]
    labels = [label for _, label, _ in MODELS]
    # Axes are laid out by hand so one category slot is the same width in both panels (3 vs. 2 slots),
    # which keeps a bar the same pixel width on the left and the right.
    fig = plt.figure(figsize=(12.5, 4.8), facecolor=SURFACE)
    slot = 1 / 6
    ax1 = fig.add_axes((0.055, 0.14, 3 * slot, 0.72))
    ax2 = fig.add_axes((0.62, 0.14, 2 * slot, 0.72))
    ax1.set_xlim(-0.5, 2.5)
    ax2.set_xlim(-0.5, 1.5)

    cheat = {m: [(sum(r["outcome"] == "shortcut" for r in by[(m, s)]), len(by[(m, s)])) for s in SPLITS] for m, _, _ in MODELS}
    grouped_bars(ax1, [SPLIT_LABELS[s] for s in SPLITS], cheat, colors, labels)
    style(ax1, "Cheating rate: share of samples where the judge found a shortcut")
    ax1.set_ylim(0, 1.15)
    ax1.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)

    # French over all completed samples (splits pooled), split by whether the sample was a shortcut.
    french = {}
    for m, _, _ in MODELS:
        rows_m = [r for sp in SPLITS for r in by[(m, sp)]]
        french[m] = [
            (sum(r[field] for r in rows_m if r["outcome"] == "shortcut"),
             sum(r[field] for r in rows_m if r["outcome"] != "shortcut"),
             len(rows_m))
            for field in ("cot_french", "output_french")
        ]
    stacked_bars(ax2, ["French in\nreasoning (CoT)", "French in\nvisible answer"], french, colors)
    style(ax2, "Share of all samples with French")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax2.set_ylim(0, 0.012)
    ax2.legend(handles=[matplotlib.patches.Patch(color=MUTED, label="cheated (shortcut)"),
                        matplotlib.patches.Patch(color=tint(MUTED), label="did not cheat")],
               frameon=False, loc="upper left", fontsize=9, labelcolor=INK)

    out = Path(args.log_dir) / "impossiblebench_results.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")
    for m, _, _ in MODELS:
        print(m, "cheat", cheat[m], "french (cot, out)", french[m])


if __name__ == "__main__":
    main()
