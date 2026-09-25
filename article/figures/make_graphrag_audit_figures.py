#!/usr/bin/env python3
"""Figures for the GraphRAG explanation audit.

Outputs
  graphrag_quality_profile.png
  graphrag_personalization_ablation.png
"""

import json
import os
import random
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publication_style import (apply_style, clean_axis, save, title, INK, MUTED,
                               GRID, NAVY, TEAL, CORAL, GOLD, PURPLE, LIGHT)

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("FOODIE_DATA_ROOT", ROOT))
REM = DATA / "results/graphrag/remediation"
OUT = Path(os.environ.get("FOODIE_FIGURE_DIR", Path(__file__).resolve().parent))

AUDIT = "regenerated_entity_v2"
WITHHELD = "match_withheld_entity_v2"

DIMS = ["entailment", "citation_correctness", "completeness",
        "usefulness", "personalization"]
DIM_LABEL = {
    "entailment": "Entailment",
    "personalization": "Personalization",
    "usefulness": "Usefulness",
    "citation_correctness": "Citation correctness",
    "completeness": "Completeness",
}


def load(name):
    path = REM / name / "generations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- fig A
def quality_profile():
    rows = load(AUDIT)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.7),
                             gridspec_kw={"width_ratios": [1.12, 1.0]})

    # ---- left: stacked score distribution per dimension
    ax = axes[0]
    shades = ["#B3423C", "#D98A62", "#C9CFD8", "#7FB3A4", TEAL]
    ypos = np.arange(len(DIMS))[::-1]
    for i, dim in enumerate(DIMS):
        counts = {s: 0 for s in range(1, 6)}
        for r in rows:
            counts[r["judge"][dim]] += 1
        left = 0
        for s in range(1, 6):
            v = counts[s]
            if v:
                ax.barh(ypos[i], v, left=left, height=0.56, color=shades[s - 1],
                        edgecolor="white", linewidth=1.0)
                if v >= 7:
                    ax.text(left + v / 2, ypos[i], str(v), ha="center", va="center",
                            fontsize=10.4, fontweight="bold",
                            color="white" if s in (1, 5) else INK)
                left += v
        ax.text(101.5, ypos[i], f"mean {st.mean(r['judge'][dim] for r in rows):.2f}",
                va="center", fontsize=10.4, color=MUTED)
    ax.set_yticks(ypos)
    ax.set_yticklabels([DIM_LABEL[d] for d in DIMS], fontsize=10.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Explanations, of 100 audited")
    ax.set_title("Score distribution by dimension", loc="left")
    clean_axis(ax, grid_axis="x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=shades[s - 1]) for s in range(1, 6)]
    ax.legend(handles, [str(s) for s in range(1, 6)], title="Judge score",
              ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              fontsize=10, title_fontsize=10, handlelength=1.4, columnspacing=1.2)

    # ---- right: mean score by evidence availability
    ax = axes[1]
    groups = [
        ("No supported match\nn = 21", [r for r in rows
                                        if not r["deterministic"]["match_available"]], NAVY),
        ("Supported match\nn = 79", [r for r in rows
                                     if r["deterministic"]["match_available"]], TEAL),
    ]
    x = np.arange(len(DIMS))
    width = 0.36
    for i, (label, subset, colour) in enumerate(groups):
        vals = [st.mean(r["judge"][d] for r in subset) for d in DIMS]
        ax.bar(x + (i - 0.5) * width, vals, width, color=colour,
               edgecolor="white", linewidth=0.8, label=label.replace("\n", ", "))
        for xx, v in zip(x + (i - 0.5) * width, vals):
            ax.text(xx, v + 0.10, f"{v:.2f}", ha="center", fontsize=9.2, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["Entail-\nment", "Citation\ncorrectness", "Complete-\nness",
                        "Useful-\nness", "Personal-\nization"], fontsize=9.8)
    ax.set_ylim(0, 6.0)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_ylabel("Mean judge score (1–5)")
    ax.set_title("Mean score by evidence availability", loc="left")
    clean_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=9.8)
    ax.text(0.5, -0.255,
            "Without a match there is nothing to personalize with, so a score of 1 is "
            "the only outcome available.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.0,
            color=MUTED, style="italic")

    title(fig,
          "Explanation quality across the 100-case audit",
          "Grounding dimensions sit at the top of the scale for every case. Personalization "
          "is the dimension that varies, and it varies with what the retriever found.")
    fig.subplots_adjust(bottom=0.30, wspace=0.42)
    save(fig, OUT / "graphrag_quality_profile.png")
    print("wrote graphrag_quality_profile.png")


# --------------------------------------------------------------------------- fig B
def personalization_ablation():
    fixed = {r["sample_id"]: r for r in load(AUDIT)}
    withheld = {r["sample_id"]: r for r in load(WITHHELD)}
    keys = sorted(set(fixed) & set(withheld))

    random.seed(0)
    rows = []
    for d in DIMS:
        deltas = [withheld[k]["judge"][d] - fixed[k]["judge"][d] for k in keys]
        boot = sorted(st.mean(random.choices(deltas, k=len(deltas)))
                      for _ in range(4000))
        rows.append((DIM_LABEL[d], st.mean(deltas), boot[100], boot[3899]))

    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    y = np.arange(len(rows))[::-1]
    for i, (label, mean, lo, hi) in enumerate(rows):
        holds = lo <= 0 <= hi
        colour = TEAL if holds else CORAL
        ax.plot([lo, hi], [y[i], y[i]], color=colour, lw=3.4,
                solid_capstyle="round", zorder=3)
        ax.plot([mean], [y[i]], "o", color=colour, markersize=9.5,
                markeredgecolor="white", markeredgewidth=1.4, zorder=4)
        ax.text(mean, y[i] - 0.34, f"{mean:+.2f}", ha="center", va="top",
                fontsize=10.4, fontweight="bold", color=colour, zorder=5)
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11.0)
    ax.set_xlim(-4.0, 1.0)
    ax.set_ylim(-0.85, len(rows) - 0.35)
    ax.set_xlabel("Mean paired change when the personalization match is withheld")
    clean_axis(ax, grid_axis="x")
    ax.text(0.995, 0.955, "grounding unchanged", transform=ax.transAxes,
            ha="right", fontsize=10.2, color=TEAL, fontweight="bold")
    ax.text(0.02, 0.26, "usefulness and personalization fall", transform=ax.transAxes,
            fontsize=10.2, color=CORAL, fontweight="bold")

    title(fig,
          "What the personalization match contributes",
          "The same 79 diners re-explained with the match node removed and nothing "
          "else changed. Bars are bootstrap 95% intervals over the paired differences.")
    fig.subplots_adjust(bottom=0.20)
    save(fig, OUT / "graphrag_personalization_ablation.png")
    print("wrote graphrag_personalization_ablation.png")


if __name__ == "__main__":
    apply_style()
    quality_profile()
    personalization_ablation()
