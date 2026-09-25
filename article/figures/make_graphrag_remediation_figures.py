#!/usr/bin/env python3
"""Figures for the post-remediation GraphRAG explanation audit.

Outputs
  graphrag_remediation_decomposition.png
  graphrag_ablation_and_outcomes.png
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

DIMS = ["entailment", "personalization", "usefulness",
        "citation_correctness", "completeness"]
DIM_LABEL = {
    "entailment": "Entailment",
    "personalization": "Personalization",
    "usefulness": "Usefulness",
    "citation_correctness": "Citation correctness",
    "completeness": "Completeness",
}
DIM_TICK = {
    "entailment": "Entail-\nment",
    "personalization": "Personal-\nization",
    "usefulness": "Useful-\nness",
    "citation_correctness": "Citation\ncorrectness",
    "completeness": "Complete-\nness",
}


def load(name):
    path = REM / name / "generations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- fig A
def decomposition():
    """Entailment score distribution and dimension means across three conditions."""
    orig_counts = {1: 3, 2: 45, 3: 1, 4: 12, 5: 39}   # original rubric, original serialization
    orig_means = {"entailment": 3.39, "personalization": 2.34, "usefulness": 2.71,
                  "citation_correctness": 3.75, "completeness": float("nan")}

    rubric = load("rubric_only_original")
    fixed = load("regenerated_entity_v2")

    def counts(rows):
        c = {k: 0 for k in range(1, 6)}
        for r in rows:
            c[r["judge"]["entailment"]] += 1
        return c

    def means(rows):
        return {d: st.mean(r["judge"][d] for r in rows) for d in DIMS}

    conditions = [
        ("Original\nschema and rubric", orig_counts, orig_means),
        ("Rubric corrected\nschema unchanged", counts(rubric), means(rubric)),
        ("Rubric and schema\nboth corrected", counts(fixed), means(fixed)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.9),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---- left: stacked entailment distribution
    ax = axes[0]
    shades = ["#B3423C", "#D98A62", "#C9CFD8", "#7FB3A4", TEAL]
    ypos = np.arange(len(conditions))[::-1]
    for i, (label, c, _) in enumerate(conditions):
        left = 0
        for s in range(1, 6):
            v = c[s]
            if v:
                ax.barh(ypos[i], v, left=left, height=0.52, color=shades[s - 1],
                        edgecolor="white", linewidth=1.0)
                if v >= 6:
                    ax.text(left + v / 2, ypos[i], str(v), ha="center", va="center",
                            fontsize=10.5, fontweight="bold",
                            color="white" if s in (1, 5) else INK)
                left += v
        low = c[1] + c[2]
        ax.text(101.5, ypos[i], f"{low}% score 1–2", va="center", fontsize=10.6,
                fontweight="bold", color=CORAL if low else TEAL)
    ax.set_yticks(ypos)
    ax.set_yticklabels([c[0] for c in conditions], fontsize=10.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Explanations, of 100 audited")
    ax.set_title("Entailment score distribution", loc="left")
    clean_axis(ax, grid_axis="x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=shades[s - 1]) for s in range(1, 6)]
    ax.legend(handles, [f"{s}" for s in range(1, 6)], title="Judge score",
              ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=10, title_fontsize=10, handlelength=1.4, columnspacing=1.2)

    # ---- right: dimension means
    ax = axes[1]
    x = np.arange(len(DIMS))
    width = 0.26
    colours = [CORAL, GOLD, TEAL]
    for i, (label, _, m) in enumerate(conditions):
        vals = [m[d] for d in DIMS]
        bars = ax.bar(x + (i - 1) * width, vals, width, color=colours[i],
                      edgecolor="white", linewidth=0.8,
                      label=label.replace("\n", " "))
        for xx, v in zip(x + (i - 1) * width, vals):
            if not np.isnan(v):
                ax.text(xx, v + 0.09, f"{v:.2f}", ha="center", fontsize=9.1, color=INK)
            else:
                ax.text(xx, 0.12, "not scored", ha="center", va="bottom", rotation=90,
                        fontsize=8.6, color=MUTED, style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([DIM_TICK[d] for d in DIMS], fontsize=9.8)
    ax.set_ylim(0, 6.0)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_ylabel("Mean judge score (1–5)")
    ax.set_title("Mean score by dimension", loc="left")
    clean_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=9.6)

    title(fig,
          "Which correction moved the audit",
          "The same 100 recommendations, scored under the original rubric, under a corrected "
          "rubric alone, and after evidence nodes were also given explicit entity fields.")
    fig.subplots_adjust(bottom=0.26, wspace=0.44)
    save(fig, OUT / "graphrag_remediation_decomposition.png")
    print("wrote graphrag_remediation_decomposition.png")


# --------------------------------------------------------------------------- fig B
def ablation_and_outcomes():
    fixed = {r["sample_id"]: r for r in load("regenerated_entity_v2")}
    withheld = {r["sample_id"]: r for r in load("match_withheld_entity_v2")}
    keys = sorted(set(fixed) & set(withheld))

    random.seed(0)
    rows = []
    for d in ["entailment", "citation_correctness", "completeness",
              "usefulness", "personalization"]:
        deltas = [withheld[k]["judge"][d] - fixed[k]["judge"][d] for k in keys]
        boot = sorted(st.mean(random.choices(deltas, k=len(deltas)))
                      for _ in range(4000))
        rows.append((DIM_LABEL[d].replace("\n", " "), st.mean(deltas),
                     boot[100], boot[3899]))

    hits = load("hit_examples_v3_positive_match")
    order = {"liked": 0, "neutral": 1, "disliked": 2}
    hits.sort(key=lambda r: (order[r["heldout_outcome"]], -r["judge"]["personalization"]))

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0),
                             gridspec_kw={"width_ratios": [1.0, 1.08]})

    # ---- left: paired ablation forest plot
    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    for i, (label, mean, lo, hi) in enumerate(rows):
        colour = TEAL if lo <= 0 <= hi else CORAL
        ax.plot([lo, hi], [y[i], y[i]], color=colour, lw=3.0,
                solid_capstyle="round", zorder=3)
        ax.plot([mean], [y[i]], "o", color=colour, markersize=9,
                markeredgecolor="white", markeredgewidth=1.4, zorder=4)
        ax.text(mean, y[i] - 0.36, f"{mean:+.2f}", ha="center", va="top",
                fontsize=10.2, fontweight="bold", color=colour, zorder=5)
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.6)
    ax.set_xlim(-4.0, 0.9)
    ax.set_ylim(-0.9, len(rows) - 0.4)
    ax.set_xlabel("Mean paired change when the personalization match is withheld")
    ax.set_title("Withholding personalization, 79 matched cases", loc="left")
    clean_axis(ax, grid_axis="x")
    ax.text(-3.98, -0.80, "Bars are bootstrap 95% intervals over the 79 paired cases.",
            fontsize=9.2, color=MUTED, style="italic")

    # ---- right: the twelve Hit@10 illustration cases
    ax = axes[1]
    outcome_colour = {"liked": TEAL, "neutral": GOLD, "disliked": CORAL}
    short = {
        "BlackHorse Espresso & Bakery": "BlackHorse Espresso",
        "Raising Cane's Chicken Fingers": "Raising Cane's",
        "Back Alley Bowling - Glendale": "Back Alley Bowling",
        "Brasserie Mon Chou Chou": "Brasserie Mon Chou Chou",
    }
    yy = np.arange(len(hits))[::-1].astype(float)
    series = [("entailment", "o", NAVY, +0.22), ("personalization", "s", PURPLE, 0.0),
              ("usefulness", "^", GOLD, -0.22)]
    for i, r in enumerate(hits):
        j = r["judge"]
        oc = outcome_colour[r["heldout_outcome"]]
        ax.plot([1, 5], [yy[i], yy[i]], color=LIGHT, lw=15.0,
                solid_capstyle="round", zorder=1)
        for dim, marker, colour, dy in series:
            ax.plot([j[dim]], [yy[i] + dy], marker, color=colour, markersize=7.6,
                    zorder=4, markeredgecolor="white", markeredgewidth=1.1)
        name = short.get(r["recommended_restaurant"], r["recommended_restaurant"])
        ax.text(0.80, yy[i], name, ha="right", va="center", fontsize=9.7, color=INK)
        ax.text(5.30, yy[i], f"rank {r['recommendation_rank']}", ha="left",
                va="center", fontsize=9.2, color=MUTED)
        ax.text(6.05, yy[i], r["heldout_outcome"], ha="left", va="center",
                fontsize=9.4, color=oc, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(0.7, 7.3)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Judge score (1–5)")
    ax.set_title("Twelve Hit@10 cases, balanced by held-out outcome", loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_ylim(-0.9, len(hits) - 0.4)
    for gx in (1, 2, 3, 4, 5):
        ax.axvline(gx, color=GRID, lw=0.8, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    marks = [
        plt.Line2D([], [], marker=m, color=c, linestyle="", markersize=8,
                   label=DIM_LABEL[d])
        for d, m, c, _ in series
    ]
    ax.legend(handles=marks, loc="upper center", bbox_to_anchor=(0.42, -0.11),
              ncol=3, fontsize=9.6)

    title(fig,
          "Personalization adds usefulness without costing support, and neither tracks satisfaction",
          "Left: the same users re-explained with the user-to-restaurant match removed. "
          "Right: twelve cases selected to span held-out outcomes, four per outcome.")
    fig.subplots_adjust(bottom=0.24, wspace=0.34)
    save(fig, OUT / "graphrag_ablation_and_outcomes.png")
    print("wrote graphrag_ablation_and_outcomes.png")


if __name__ == "__main__":
    apply_style()
    decomposition()
    ablation_and_outcomes()
