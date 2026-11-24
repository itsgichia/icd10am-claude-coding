"""Cross-model figures for the README.

Reads the per-model summaries under ``results/<model>`` plus the parsing-gap
table and draws two figures into ``figures/``:

1. ``accuracy_by_strategy.png``: development-split accuracy for every
   strategy, one panel per model.
2. ``parse_gap.png``: pipeline accuracy versus the accuracy of the first code
   the model wrote, one panel per model.

Usage::

    python scripts/compare_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from icdprompt.infer import MODEL_IDS  # noqa: E402

# Reference palette (dataviz skill): categorical slots 1-3 and two blue shades.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT = "#86b6ef"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SHORT = {m: m.split("-")[1].capitalize() for m in MODEL_IDS}
ORDER = [
    "zero_shot_minimal", "persona_clinical_coder", "few_shot_three", "few_shot_six",
    "cot_then_extract", "context_rich_structure", "zero_shot_strict_format", "json_output",
    "hybrid_coder_structured",
]
PRETTY = {
    "zero_shot_minimal": "plain question", "persona_clinical_coder": "coder persona",
    "few_shot_three": "3 examples", "few_shot_six": "6 examples", "cot_then_extract": "reason, then code",
    "context_rich_structure": "structured briefing", "zero_shot_strict_format": "strict format",
    "json_output": "JSON output", "hybrid_coder_structured": "persona + structure",
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK,
})


def _style(ax, xlabel: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel, color=INK_2)


def _dumbbell(ax, left, right, labels, c_left, c_right, annotate_right=False) -> None:
    y = range(len(labels))
    for yi, a, b in zip(y, left, right):
        ax.plot([a, b], [yi, yi], color=GRID, linewidth=2, zorder=1, solid_capstyle="round")
    ax.scatter(left, list(y), s=70, color=c_left, zorder=3, edgecolors=SURFACE, linewidths=2)
    ax.scatter(right, list(y), s=70, color=c_right, zorder=3, edgecolors=SURFACE, linewidths=2)
    if annotate_right:
        for yi, b in zip(y, right):
            ax.annotate(f"{b:.2f}", (b, yi), xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK_2)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, len(labels) - 0.3)


def figure_strategies(out: Path) -> None:
    frames = [pd.read_csv(REPO_ROOT / "results" / model / "summary.csv") for model in MODEL_IDS]
    s = pd.concat(frames)
    s = s[s.split == "dev"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    for ax, model in zip(axes, MODEL_IDS):
        d = s[s.model == model].set_index("template").reindex(ORDER)
        y = list(range(len(ORDER)))
        ax.hlines(y, 0, d["accuracy"], color=GRID, linewidth=2, zorder=1)
        ax.scatter(d["accuracy"], y, s=70, color=BLUE, zorder=3, edgecolors=SURFACE, linewidths=2)
        for yi, v in zip(y, d["accuracy"]):
            ax.annotate(f"{v:.2f}", (v, yi), xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK_2)
        ax.set_yticks(y)
        ax.set_yticklabels([PRETTY[t] for t in ORDER])
        ax.set_ylim(-0.7, len(ORDER) - 0.3)
        _style(ax, "accuracy on 100 development records")
        ax.set_title(SHORT[model], loc="left", color=INK, fontsize=11)
    fig.suptitle("Exact-match accuracy against the keyword labels, by prompt strategy", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def figure_parse_gap(out: Path) -> None:
    g = pd.read_csv(REPO_ROOT / "results" / "parse_gap.csv")
    g = g[g.split == "dev"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    for ax, model in zip(axes, MODEL_IDS):
        d = g[g.model == model].set_index("template").reindex(ORDER)
        _dumbbell(ax, d["accuracy"], d["first_code_correct"], [PRETTY[t] for t in ORDER], BLUE, ORANGE)
        for yi, n in enumerate(d["codes_per_response"]):
            ax.annotate(f"{n:.1f} codes", (0.0, yi), xytext=(4, 0), textcoords="offset points", va="center",
                        ha="left", fontsize=8, color=MUTED)
        _style(ax, "share correct, keyword labels, dev split")
        ax.set_title(SHORT[model], loc="left", color=INK, fontsize=11)
    handles = [axes[0].scatter([], [], color=BLUE, s=70, label="code the extractor returned"),
               axes[0].scatter([], [], color=ORANGE, s=70, label="first code the model wrote")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.945), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Code the extractor returned versus the first code the model wrote (mean codes per response at left)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> int:
    figures = REPO_ROOT / "figures"
    figures.mkdir(exist_ok=True)
    figure_strategies(figures / "accuracy_by_strategy.png")
    figure_parse_gap(figures / "parse_gap.png")
    print("wrote two figures to figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
