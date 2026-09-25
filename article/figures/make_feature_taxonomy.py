#!/usr/bin/env python3
"""Render exact frozen feature groups as a flat publication taxonomy."""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publication_style import (apply_style, save, INK, MUTED, GRID, NAVY, TEAL,
                               CORAL, GOLD)
from diagram_style import heading, elbow_split_h

OUT = Path(os.environ.get("FOODIE_FIGURE_DIR", Path(__file__).resolve().parent))
FIG_W, FIG_H = 15.0, 10.1

SLATE = "#4A5A6E"
NAVY_FACE = "#F2F7FB"
GOLD_FACE = "#FEF8EC"
SLATE_FACE = "#EFF2F6"

ROW_X, ROW_W, ROW_H, ROW_PITCH = 0.462, 0.508, 0.072, 0.080
NODE_X, NODE_W, NODE_H = 0.258, 0.156, 0.102


def panel(ax, x, y, w, h, accent, face, stripe=0.0060, lw=1.3, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.010",
                                linewidth=lw, edgecolor=GRID, facecolor=face,
                                zorder=z, mutation_aspect=0.72))
    ax.add_patch(Rectangle((x, y + 0.007), stripe, h - 0.014, facecolor=accent,
                           edgecolor="none", zorder=z + 1))
    return {"left": (x, y + h / 2), "right": (x + w, y + h / 2),
            "top": (x + w / 2, y + h), "bottom": (x + w / 2, y)}


def row_card(ax, y, title, count, body, accent, face):
    box = panel(ax, ROW_X, y, ROW_W, ROW_H, accent, face)
    ax.text(ROW_X + 0.024, y + ROW_H - 0.019, title, ha="left", va="top",
            fontsize=11.6, fontweight="bold", color=accent, zorder=5)
    ax.text(ROW_X + ROW_W - 0.022, y + ROW_H - 0.019, count, ha="right",
            va="top", fontsize=11.6, fontweight="bold", color=accent, zorder=5)
    ax.text(ROW_X + 0.024, y + 0.019, body, ha="left", va="bottom",
            fontsize=9.8, color=MUTED, zorder=5)
    return box


def node_card(ax, y, title, count, accent, face):
    box = panel(ax, NODE_X, y, NODE_W, NODE_H, accent, face, stripe=0.0070,
                lw=1.5)
    ax.text(NODE_X + 0.026, y + NODE_H * 0.62, title, ha="left", va="center",
            fontsize=14.0, fontweight="bold", color=INK, zorder=5)
    ax.text(NODE_X + 0.026, y + NODE_H * 0.28, count, ha="left", va="center",
            fontsize=10.4, color=MUTED, zorder=5)
    return box


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.008, 0.922)
    ax.axis("off")

    heading(fig, "Frozen user and restaurant feature taxonomy",
            "467 dimensions assembled from analytical and locally generated "
            "semantic features.", y=0.966, gap=0.048)

    user_rows = [
        ("Base analytical", "12", "Activity  ·  location  ·  inferred diagnostic labels",
         NAVY, NAVY_FACE),
        ("Extended analytical", "43", "Cuisine affinity  ·  rating behavior  ·  geographic range",
         NAVY, NAVY_FACE),
        ("Structured LLM", "86", "Taste  ·  atmosphere  ·  occasions  ·  operations  ·  diet",
         GOLD, GOLD_FACE),
        ("LLM embedding", "64", "Dense residual review semantics",
         GOLD, GOLD_FACE),
    ]
    item_rows = [
        ("Base analytical", "29", "Price  ·  rating  ·  cuisine  ·  meal periods",
         NAVY, NAVY_FACE),
        ("Extended analytical", "24", "Density  ·  trends  ·  engagement  ·  CBG behavior",
         NAVY, NAVY_FACE),
        ("Dish aggregates", "4", "Dish count  ·  diversity  ·  category breadth",
         GOLD, GOLD_FACE),
        ("Structured LLM", "141", "Menu  ·  experience  ·  quality  ·  dietary support  ·  cuisine",
         GOLD, GOLD_FACE),
        ("LLM embedding", "64", "Dense residual review semantics",
         GOLD, GOLD_FACE),
    ]

    user_top, item_top = 0.834, 0.406
    user_boxes = [row_card(ax, user_top - i * ROW_PITCH, *r)
                  for i, r in enumerate(user_rows)]
    item_boxes = [row_card(ax, item_top - i * ROW_PITCH, *r)
                  for i, r in enumerate(item_rows)]

    user_mid = (user_boxes[0]["left"][1] + user_boxes[-1]["left"][1]) / 2
    item_mid = (item_boxes[0]["left"][1] + item_boxes[-1]["left"][1]) / 2

    user = node_card(ax, user_mid - NODE_H / 2, "User", "205 dimensions",
                     TEAL, "#EFF8F6")
    item = node_card(ax, item_mid - NODE_H / 2, "Restaurant", "262 dimensions",
                     CORAL, "#FDF2EF")

    root_mid = (user_mid + item_mid) / 2
    root = panel(ax, 0.028, root_mid - 0.056, 0.172, 0.112, SLATE, SLATE_FACE,
                 stripe=0.0070, lw=1.5)
    ax.text(0.028 + 0.026, root_mid + 0.018, "Feature contract", ha="left",
            va="center", fontsize=12.6, fontweight="bold", color=INK, zorder=5)
    ax.text(0.028 + 0.026, root_mid - 0.020, "467 dimensions", ha="left",
            va="center", fontsize=10.4, color=MUTED, zorder=5)

    elbow_split_h(ax, root["right"], [user["left"], item["left"]], mid_frac=0.5)
    elbow_split_h(ax, user["right"], [b["left"] for b in user_boxes],
                  mid_frac=0.42)
    elbow_split_h(ax, item["right"], [b["left"] for b in item_boxes],
                  mid_frac=0.42)

    # ------------------------------------------------------------- legend
    for x, face, accent, label in [(0.030, NAVY_FACE, NAVY, "Analytical"),
                                   (0.146, GOLD_FACE, GOLD, "LLM-derived")]:
        ax.add_patch(FancyBboxPatch((x, 0.026), 0.022, 0.020,
                                    boxstyle="round,pad=0,rounding_size=0.004",
                                    facecolor=face, edgecolor=accent,
                                    linewidth=1.5, mutation_aspect=0.72))
        ax.text(x + 0.030, 0.036, label, va="center", ha="left", fontsize=9.8,
                color=INK)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.876, bottom=0.022)
    save(fig, OUT / "feature_tree.png")
    print("wrote feature_tree.png")


if __name__ == "__main__":
    main()
