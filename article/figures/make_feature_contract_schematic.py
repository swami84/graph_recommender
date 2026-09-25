#!/usr/bin/env python3
"""Render the chronological feature-construction contract."""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publication_style import apply_style, save, INK, NAVY, TEAL, CORAL, GOLD, PURPLE
from diagram_style import card, elbow_merge, elbow_split, arrow_v, heading

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(os.environ.get("FOODIE_FIGURE_DIR", Path(__file__).resolve().parent)) / \
    "feature_contract_schematic.png"


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(11.5, 7.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.038, 0.952)
    ax.axis("off")

    heading(fig, "Chronological feature-construction contract",
            "The split is applied before aggregation and extraction, so only training-safe "
            "evidence reaches the feature matrices.", y=0.964, gap=0.056)

    mid_w, mid_x = 0.46, 0.27

    c1 = card(ax, mid_x, 0.845, mid_w, 0.095, "Canonical evidence",
              "Reviews  ·  place metadata  ·  location", NAVY, "#F2F7FB")
    c2 = card(ax, mid_x, 0.635, mid_w, 0.125, "Chronological split",
              "Oldest visits  →  training\nSecond-newest  →  validation      Newest  →  test",
              CORAL, "#FDF2EF")
    c3 = card(ax, mid_x, 0.455, mid_w, 0.095, "Training-safe evidence",
              "Validation and test review identifiers excluded", TEAL, "#EFF8F6")

    arrow_v(ax, c1["bottom"], c2["top"])
    arrow_v(ax, c2["bottom"], c3["top"])

    left = card(ax, 0.045, 0.245, 0.425, 0.110, "Analytical features",
                "Behavior  ·  ratings  ·  cuisine  ·  geography", NAVY, "#F2F7FB")
    right = card(ax, 0.530, 0.245, 0.425, 0.110, "Local-LLM features",
                 "Structured attributes  ·  embeddings  ·  dishes", GOLD, "#FEF8EC")
    elbow_split(ax, c3["bottom"], [left["top"], right["top"]], mid_frac=0.55)

    out = card(ax, 0.185, 0.055, 0.630, 0.100, "Frozen feature matrices",
               "Analytical and semantic groups remain separately selectable",
               PURPLE, "#F4F2FB")
    elbow_merge(ax, [left["bottom"], right["bottom"]], out["top"], mid_frac=0.5)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.868, bottom=0.024)
    save(fig, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
