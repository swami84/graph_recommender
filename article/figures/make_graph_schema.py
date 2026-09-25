#!/usr/bin/env python3
"""Render the verified KGAT-SAL graph as a flat typed-entity schema."""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publication_style import apply_style, save, INK, MUTED, GRID, TEAL, CORAL, PURPLE
from diagram_style import (card, heading, arrow_h, arrow_v, elbow_split_h,
                           elbow_merge_h)

OUT = Path(os.environ.get("FOODIE_FIGURE_DIR", Path(__file__).resolve().parent))
FIG_W, FIG_H = 15.0, 5.5

SLATE = "#5A7799"
SLATE_FACE = "#F1F4F8"

# the axes is cropped to the drawn content so the figure carries no dead band
Y_LOW, Y_HIGH = 0.288, 0.905


def centred_card(ax, x, y, w, h, label, accent, face, fontsize=10.6):
    """A flat card whose single label is centred rather than left-aligned."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.012",
                                linewidth=1.1, edgecolor=GRID, facecolor=face,
                                zorder=3, mutation_aspect=0.42))
    ax.add_patch(Rectangle((x, y + 0.008), 0.0055, h - 0.016, facecolor=accent,
                           edgecolor="none", zorder=4))
    ax.text(x + w / 2 + 0.003, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=INK, zorder=5)
    return {"left": (x, y + h / 2), "right": (x + w, y + h / 2),
            "top": (x + w / 2, y + h), "bottom": (x + w / 2, y)}


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(Y_LOW, Y_HIGH)
    ax.axis("off")

    heading(fig, "Typed graph representation for KGAT-SAL",
            "Knowledge-graph attention enriches restaurant representations before "
            "collaborative propagation; SAL aggregates three chronological user views.",
            y=0.962, gap=0.082, sub_size=10.6)

    # ------------------------------------------------ user profile features
    ax.text(0.028, 0.878, "USER PROFILE FEATURES", fontsize=9.6,
            fontweight="bold", color=TEAL)
    feat_x, feat_w, feat_h = 0.028, 0.176, 0.068
    feature_rows = ["Behavior and ratings", "Cuisine affinity",
                    "LLM preferences", "Home-area geography"]
    feature_cards = [
        centred_card(ax, feat_x, 0.784 - i * 0.084, feat_w, feat_h, label,
                     TEAL, "#F1F9F7")
        for i, label in enumerate(feature_rows)]

    # --------------------------------------------------- the two core nodes
    user = card(ax, 0.268, 0.560, 0.184, 0.112, "User", "156,261 nodes",
                TEAL, "#EFF8F6", title_size=15.0, body_size=10.6)
    item = card(ax, 0.536, 0.560, 0.196, 0.112, "Restaurant", "61,204 nodes",
                CORAL, "#FDF2EF", title_size=15.0, body_size=10.6)

    elbow_merge_h(ax, [c["right"] for c in feature_cards], user["left"],
                  mid_frac=0.62)
    arrow_h(ax, user["right"], item["left"])
    ax.text((user["right"][0] + item["left"][0]) / 2, 0.700,
            "Reviewed  ·  772,170 training edges", ha="center", fontsize=9.8,
            fontweight="bold", color=TEAL)

    # ---------------------------------------------- knowledge-graph entities
    ax.text(0.784, 0.878, "KNOWLEDGE-GRAPH ENTITIES", fontsize=9.6,
            fontweight="bold", color=SLATE)
    ent_x, ent_w, ent_h = 0.784, 0.190, 0.096
    entity_rows = [
        ("Cuisine", "17 entities  ·  has cuisine"),
        ("Price", "4 tiers  ·  has price"),
        ("CBG", "6,637 areas  ·  located in"),
        ("Dish", "Top 200  ·  serves dish"),
    ]
    entity_cards = [
        card(ax, ent_x, 0.756 - i * 0.110, ent_w, ent_h, title, body,
             SLATE, SLATE_FACE, title_size=12.2, body_size=9.6)
        for i, (title, body) in enumerate(entity_rows)]
    elbow_split_h(ax, item["right"], [c["left"] for c in entity_cards],
                  mid_frac=0.42)

    ax.text(ent_x, 0.372, "Each CBG also links to its five nearest\nneighbours, "
                          "so local context is shared.",
            ha="left", va="top", fontsize=9.4, color=MUTED, linespacing=1.5)

    # ------------------------------------------------------ temporal views
    temporal = card(ax, 0.268, 0.318, 0.184, 0.116, "Temporal views",
                    "3 windows\nOldest → middle → most recent",
                    PURPLE, "#F4F2FB", title_size=12.4, body_size=9.6)
    arrow_v(ax, user["bottom"], temporal["top"])
    ax.text(0.372, (user["bottom"][1] + temporal["top"][1]) / 2,
            "Stability-aware\naggregation", ha="left", va="center",
            fontsize=9.4, color=MUTED, linespacing=1.5)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.818, bottom=0.030)
    save(fig, OUT / "graph_schema.png")
    print("wrote graph_schema.png")


if __name__ == "__main__":
    main()
