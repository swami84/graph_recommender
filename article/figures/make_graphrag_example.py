#!/usr/bin/env python3
"""Render the post-ranking GraphRAG evidence path for one audited recommendation."""

import json
import os
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publication_style import apply_style, save, INK, MUTED, GRID, NAVY, TEAL, CORAL, GOLD, PURPLE
from diagram_style import card, panel, arrow_h, arrow_v, heading

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("FOODIE_DATA_ROOT", ROOT))
SOURCE = DATA / "results/graphrag/remediation/regenerated_entity_v2/generations.jsonl"
OUTPUT = Path(os.environ.get("FOODIE_FIGURE_DIR", Path(__file__).resolve().parent)) / \
    "graphrag_explanation_example.png"
SAMPLE_ID = 6


def score_rows(ax, x, y, w, rows, row_h=0.0265):
    for i, (label, value, colour) in enumerate(rows):
        yy = y - i * row_h
        ax.text(x, yy, label, va="center", fontsize=10.0, color=MUTED, zorder=5)
        ax.text(x + w, yy, value, va="center", ha="right", fontsize=10.4,
                fontweight="bold", color=colour, zorder=5)


def main():
    apply_style()
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    sample = next(r for r in rows if int(r["sample_id"]) == SAMPLE_ID)
    judge = sample["judge"]
    generated = sample["generation"]["explanation"]

    fig, ax = plt.subplots(figsize=(13.4, 9.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    heading(fig, "Post-ranking explanation for one audited recommendation",
            "Evidence is assembled after the restaurant has been selected, so the "
            "explanation layer does not affect the ranking.")

    col_x = [0.045, 0.365, 0.685]
    col_w = 0.270

    # ---------------------------------------------------------------- ranking band
    ax.text(0.045, 0.912, "STAGE 1   RANKING, COMPLETED BEFORE EXPLANATION",
            fontsize=9.6, fontweight="bold", color=MUTED)
    r1 = card(ax, col_x[0], 0.800, col_w, 0.094, "KGAT-SAL",
              "Full-catalogue scores", NAVY, "#F2F7FB", badge="1", title_size=12.2)
    r2 = card(ax, col_x[1], 0.800, col_w, 0.094, "Proximity rerank",
              "Frozen top-100 reordering", CORAL, "#FDF2EF", badge="2", title_size=12.2)
    r3 = card(ax, col_x[2], 0.800, col_w, 0.094, "Selected restaurant",
              "Ru San's Kennesaw  ·  Kennesaw, GA", TEAL, "#EFF8F6", badge="3", title_size=12.2)
    arrow_h(ax, r1["right"], r2["left"])
    arrow_h(ax, r2["right"], r3["left"])

    # ------------------------------------------------------------- retrieval band
    panel(ax, 0.028, 0.452, 0.944, 0.318)
    ax.text(0.045, 0.744, "STAGE 2   TRAINING-SAFE EVIDENCE RETRIEVAL,  15 NODES",
            fontsize=9.6, fontweight="bold", color=MUTED)
    e1 = card(ax, col_x[0], 0.552, col_w, 0.158, "Diner evidence",
              "McAlister's Deli  5.0★  [H1]\nTwo more, both 5.0★  [H2][H3]\nValue preference  0.8  [U1]",
              PURPLE, "#FFFFFF")
    e2 = card(ax, col_x[1], 0.552, col_w, 0.158, "Supported match",
              "Shared attribute, not cuisine\nvalue  0.8 diner / 0.8 restaurant\n[M1]",
              GOLD, "#FFFFFF")
    e3 = card(ax, col_x[2], 0.552, col_w, 0.158, "Restaurant evidence",
              "Place metadata  [R1]\nHealthiness 0.9, value 0.8  [A1][A2]\n"
              "Three review excerpts  [Q1]–[Q3]",
              TEAL, "#FFFFFF")
    arrow_h(ax, e1["right"], e2["left"])
    arrow_h(ax, e2["right"], e3["left"])
    ax.text(0.5, 0.494,
            "Validation and test review identifiers excluded from every bundle   ·   "
            "each factual claim must cite an evidence node",
            ha="center", fontsize=10.0, color=MUTED)

    # ------------------------------------------------- anatomy of one node
    ax.add_patch(FancyBboxPatch(
        (0.028, 0.320), 0.944, 0.106,
        boxstyle="round,pad=0,rounding_size=0.012", linewidth=1.0,
        edgecolor=GRID, facecolor="#FCF7EA", zorder=1, mutation_aspect=0.55))
    ax.text(0.048, 0.408, "ANATOMY OF ONE EVIDENCE NODE",
            fontsize=9.4, fontweight="bold", color=MUTED, zorder=5)
    fields = [
        ("id", "M1", 0.048),
        ("kind", "supported_personalization_match", 0.130),
        ("entity", "Diner and Ru San's Kennesaw", 0.470),
        ("source", "exact structured-attribute match", 0.730),
    ]
    for label, value, x in fields:
        ax.text(x, 0.382, label, fontsize=9.0, color=MUTED, zorder=5)
        ax.text(x, 0.358, value, fontsize=10.0, color=INK, zorder=5,
                fontweight="bold")
    ax.text(0.048, 0.334,
            "fact   “Shared-attribute evidence: the diner’s value preference is on the "
            "high end (0.8), and Ru San’s Kennesaw’s corresponding attribute is also on "
            "the high end (0.8).”",
            fontsize=9.8, color=INK, zorder=5, style="italic")

    # ----------------------------------------------------------------- output band
    ax.text(0.045, 0.296, "STAGE 3   GENERATED OUTPUT AND AUDIT",
            fontsize=9.6, fontweight="bold", color=MUTED)
    body = "\n".join(textwrap.wrap(generated, 70))
    out = card(ax, col_x[0], 0.060, 0.590, 0.215, "Cited rationale", None,
               TEAL, "#EFF8F6", title_at_top=True)
    ax.text(col_x[0] + 0.026, 0.204, body, va="top", fontsize=10.4,
            linespacing=1.55, color=INK, zorder=5, style="italic")

    aud = card(ax, col_x[2], 0.060, col_w, 0.215, "Local-model audit", None,
               NAVY, "#F2F7FB", title_at_top=True)
    score_rows(ax, col_x[2] + 0.026, 0.212, col_w - 0.052, [
        ("Entailment", f"{judge['entailment']} / 5", INK),
        ("Personalization", f"{judge['personalization']} / 5", INK),
        ("Usefulness", f"{judge['usefulness']} / 5", INK),
        ("Citation correctness", f"{judge['citation_correctness']} / 5", INK),
        ("Completeness", f"{judge['completeness']} / 5", INK),
        ("Unsupported claims", f"{judge['unsupported_claims']}", TEAL),
    ])
    arrow_v(ax, (0.60, 0.320), (0.60, 0.275))
    arrow_h(ax, out["right"], aud["left"])

    ax.text(0.5, 0.022,
            "One case from the 100-case audit, not an average.  The rationale describes evidence "
            "associated with a fixed result; it is not a trace of the ranker's computation.",
            ha="center", fontsize=10.2, color=MUTED)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02)
    save(fig, OUTPUT)
    print(f"Wrote {OUTPUT}; source sample {SAMPLE_ID}")


if __name__ == "__main__":
    main()
