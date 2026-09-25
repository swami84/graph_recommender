#!/usr/bin/env python3
"""Architecture contrast: chunk-and-embed retrieval against typed graph retrieval."""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publication_style import (apply_style, save, INK, MUTED, GRID, NAVY, TEAL,
                               CORAL, GOLD, PURPLE)

OUT = Path(os.environ.get("FOODIE_FIGURE_DIR", Path(__file__).resolve().parent))

SLATE = "#8A94A6"
FAINT = "#C7CED8"
COL_L, COL_R = 0.040, 0.522
COL_W = 0.438


# ----------------------------------------------------------------- primitives
def card(ax, x, y, w, h, face="#FFFFFF", edge=GRID, lw=1.0, z=1, r=0.009):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                linewidth=lw, edgecolor=edge, facecolor=face,
                                zorder=z, mutation_aspect=0.68))


def chip(ax, cx, cy, w, h, label, accent, face="#FFFFFF", fs=9.2, bold=True, z=6):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.008",
                                linewidth=1.6, edgecolor=accent, facecolor=face,
                                zorder=z, mutation_aspect=0.68))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs, color=INK,
            zorder=z + 1, fontweight="bold" if bold else "normal")


def flow(ax, a, b, color=SLATE, lw=1.5, ls="-", z=4, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=lw, color=color, zorder=z,
                                 shrinkA=0, shrinkB=0, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def edge(ax, a, b, color=FAINT, lw=1.4, z=3):
    ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=z,
                       solid_capstyle="round"))


def column_header(ax, x, accent, kicker, title, sub):
    ax.add_patch(Rectangle((x, 0.872), 0.030, 0.0055, facecolor=accent,
                           edgecolor="none", zorder=5))
    ax.text(x, 0.914, kicker, fontsize=9.0, color=accent, fontweight="bold")
    ax.text(x, 0.886, title, fontsize=13.2, color=INK, fontweight="bold")
    ax.text(x, 0.850, sub, fontsize=9.8, color=MUTED)


def steps(ax, x, accent, items, y0=0.506, dy=0.036):
    for i, text in enumerate(items):
        yy = y0 - i * dy
        ax.plot([x + 0.004], [yy], marker="o", markersize=19.0,
                markerfacecolor="none", markeredgecolor=accent,
                markeredgewidth=1.2, linestyle="none", zorder=3,
                clip_on=False)
        ax.text(x + 0.004, yy, f"{i + 1}", fontsize=8.6, color=accent,
                fontweight="bold", va="center", ha="center", zorder=4)
        ax.text(x + 0.026, yy, text, fontsize=9.8, color=INK, va="center")


# ----------------------------------------------------------------------- main
def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(14.0, 10.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle("Two retrieval architectures for the same explanation task",
                 fontsize=18.0, fontweight="bold", color=INK, y=0.972)
    fig.text(0.5, 0.940,
             "Both hand evidence to a language model. They differ in what that "
             "evidence is, and therefore in what can be checked afterwards.",
             ha="center", va="top", fontsize=11.0, color=MUTED)

    column_header(ax, COL_L, CORAL, "BASELINE",
                  "Chunk-and-embed retrieval",
                  "Passages of review text, ranked by similarity to a query")
    column_header(ax, COL_R, TEAL, "THIS STUDY",
                  "Typed graph retrieval",
                  "Nodes and relations, walked from the diner-restaurant pair")

    card(ax, COL_L, 0.600, COL_W, 0.238, face="#FDFAF9")
    card(ax, COL_R, 0.600, COL_W, 0.238, face="#F7FCFB")

    # ------------------------------------------------- left: corpus to vectors
    bx = COL_L + 0.046
    for i in range(3):
        ax.add_patch(FancyBboxPatch((bx - 0.030 + i * 0.008, 0.720 - i * 0.009),
                                    0.058, 0.070,
                                    boxstyle="round,pad=0,rounding_size=0.006",
                                    linewidth=1.1, edgecolor=GRID,
                                    facecolor="#FFFFFF", zorder=3 + i,
                                    mutation_aspect=0.68))
    for k in range(4):
        yy = 0.772 - k * 0.011
        ax.add_line(Line2D([bx - 0.014, bx + 0.016 if k < 3 else bx + 0.004],
                           [yy, yy], color=FAINT, lw=1.4, zorder=8))
    ax.text(bx - 0.002, 0.684, "review\ncorpus", ha="center", va="top",
            fontsize=9.2, color=MUTED, linespacing=1.3)

    flow(ax, (bx + 0.044, 0.748), (bx + 0.074, 0.748), color=CORAL)
    ax.text(bx + 0.059, 0.762, "split", ha="center", fontsize=8.4, color=MUTED)

    cx = bx + 0.112
    for k in range(4):
        yy = 0.790 - k * 0.026
        ax.add_patch(Rectangle((cx - 0.030, yy - 0.017), 0.060, 0.020,
                               facecolor="#FFFFFF", edgecolor=GRID,
                               linewidth=1.0, zorder=4))
        ax.add_line(Line2D([cx - 0.022, cx + 0.018], [yy - 0.007, yy - 0.007],
                           color=FAINT, lw=1.3, zorder=5))
    ax.text(cx, 0.684, "chunks", ha="center", va="top", fontsize=9.2, color=MUTED)

    flow(ax, (cx + 0.036, 0.748), (cx + 0.062, 0.748), color=CORAL)
    ax.text(cx + 0.049, 0.762, "embed", ha="center", fontsize=8.4, color=MUTED)

    vx = cx + 0.142
    shades = ["#EEF1F5", "#DDE3EA", "#C9D2DC", "#E6EAF0", "#D3DAE3",
              "#EAEEF3", "#CFD7E0", "#E1E6EC"]
    for k in range(4):
        yy = 0.790 - k * 0.026
        hot = k in (1, 2)
        for j in range(8):
            ax.add_patch(Rectangle((vx - 0.060 + j * 0.0155, yy - 0.017),
                                   0.0135, 0.019,
                                   facecolor="#F6BFAE" if hot else shades[j],
                                   edgecolor="none", zorder=4))
        if hot:
            ax.add_patch(Rectangle((vx - 0.0625, yy - 0.0195), 0.129, 0.024,
                                   facecolor="none", edgecolor=CORAL,
                                   linewidth=1.4, zorder=6))
    ax.text(vx + 0.086, 0.751, "top-k", fontsize=9.0, color=CORAL,
            fontweight="bold", va="center")
    ax.text(vx, 0.684, "vectors", ha="center",
            va="top", fontsize=9.2, color=MUTED)

    ax.text(COL_L + COL_W / 2, 0.574,
            "The retrieved unit is a passage. Its subject, and the record it\n"
            "came from, are not carried with it.",
            ha="center", va="top", fontsize=9.6, color=MUTED, style="italic",
            linespacing=1.45)

    # ------------------------------------------------------ right: graph walk
    spine_y = 0.768
    xs = [COL_R + 0.048, COL_R + 0.152, COL_R + 0.256, COL_R + 0.362]
    w_box, h_box = 0.076, 0.042

    edge(ax, (xs[0] + w_box / 2, spine_y), (xs[1] - w_box / 2, spine_y), NAVY, 2.0)
    edge(ax, (xs[1] + w_box / 2, spine_y), (xs[2] - w_box / 2, spine_y), NAVY, 2.0)
    edge(ax, (xs[2] + w_box / 2, spine_y), (xs[3] - w_box / 2 - 0.004, spine_y),
         NAVY, 2.0)
    for a, b, lab in [(xs[0], xs[1], "VISITED"), (xs[1], xs[2], "IN CATEGORY"),
                      (xs[2], xs[3], "IN CATEGORY")]:
        ax.text((a + b) / 2, spine_y + 0.034, lab, ha="center", fontsize=7.4,
                color=NAVY, fontweight="bold")

    # pendant evidence off the recommendation
    for dx, lab in [(-0.052, "attributes"), (0.030, "reviews")]:
        px, py = xs[3] + dx, 0.700
        edge(ax, (xs[3], spine_y - h_box / 2), (px, py + 0.017), FAINT, 1.3)
        chip(ax, px, py, 0.070, 0.034, lab, FAINT, fs=8.4, bold=False)

    chip(ax, xs[0], spine_y, w_box, h_box, "diner", PURPLE, "#F5F3FB")
    chip(ax, xs[1], spine_y, w_box, h_box, "visited", SLATE)
    chip(ax, xs[2], spine_y, w_box, h_box, "cuisine", NAVY, "#EDF4FA")
    chip(ax, xs[3], spine_y, w_box + 0.022, h_box, "recommended", TEAL,
         "#E8F6F2", fs=8.8)

    # derived match edge
    ax.add_patch(FancyArrowPatch((xs[0], spine_y - h_box / 2 - 0.004),
                                 (xs[3], spine_y - h_box / 2 - 0.004),
                                 arrowstyle="-", linewidth=2.2, color=GOLD,
                                 linestyle=(0, (5, 3)), zorder=4,
                                 connectionstyle="arc3,rad=0.44",
                                 shrinkA=0, shrinkB=0))
    ax.text((xs[0] + xs[3]) / 2 - 0.030, 0.637, "derived match node",
            ha="center", fontsize=8.8, color="#A87A10", fontweight="bold",
            zorder=7, bbox=dict(boxstyle="round,pad=0.26", facecolor="#F7FCFB",
                                edgecolor="none"))

    ax.text(COL_R + COL_W / 2, 0.574,
            "The retrieved unit is a node. It carries its type, the entity it\n"
            "describes, and its source identifier.",
            ha="center", va="top", fontsize=9.6, color=MUTED, style="italic",
            linespacing=1.45)

    # ------------------------------------------------------------------ steps
    steps(ax, COL_L + 0.006, CORAL, [
        "Split the review corpus and embed every chunk",
        "Embed a query and take the k nearest vectors",
        "Hand the passages to the language model",
    ])
    steps(ax, COL_R + 0.006, TEAL, [
        "Anchor on the pair the ranker just produced",
        "Walk typed relations outward, excluding held-out review ids",
        "Hand the nodes to the language model, each citable by id",
    ])

    # ------------------------------------------------------------- comparison
    head_y = 0.392
    ax.add_line(Line2D([0.040, 0.960], [head_y + 0.022, head_y + 0.022],
                       color=INK, lw=1.2, zorder=2))
    ax.text(0.048, head_y, "", fontsize=1)
    ax.text(0.330, head_y, "CHUNK-AND-EMBED", ha="center", fontsize=8.8,
            color=CORAL, fontweight="bold")
    ax.text(0.730, head_y, "TYPED GRAPH", ha="center", fontsize=8.8,
            color=TEAL, fontweight="bold")

    rows = [
        ("Retrieval unit", "a passage of text",
         "a typed node, entity named"),
        ("Selected by", "embedding similarity to a query",
         "relations walked from a fixed pair"),
        ("Diner-to-restaurant link", "no such object in the index",
         "the match node, cited in 79 of 79 cases"),
        ("Held-out exclusion", "content match cannot enforce it",
         "a set operation on review ids, 100% of bundles safe"),
        ("Provenance of a claim", "the passage it was drawn from",
         "node id, entity, and source record"),
    ]
    y = head_y - 0.044
    for i, (label, left, right) in enumerate(rows):
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.040, y - 0.021), 0.920, 0.042,
                                   facecolor="#F5F7FA", edgecolor="none",
                                   zorder=0))
        ax.text(0.052, y, label, fontsize=9.8, color=INK, va="center",
                fontweight="bold")
        ax.text(0.330, y, left, fontsize=9.8, color=MUTED, va="center",
                ha="center")
        ax.text(0.730, y, right, fontsize=9.8, color=INK, va="center",
                ha="center")
        y -= 0.048
    ax.add_line(Line2D([0.040, 0.960], [y + 0.027, y + 0.027], color=GRID,
                       lw=1.0, zorder=2))
    ax.add_line(Line2D([0.500, 0.500], [y + 0.027, head_y + 0.022],
                       color=GRID, lw=1.0, zorder=1))

    ax.text(0.5, 0.036,
            "The recommender already ranks over this graph, so the evidence "
            "store and the ranking substrate are the same object.",
            ha="center", fontsize=10.2, color=MUTED)

    fig.subplots_adjust(left=0.015, right=0.985, top=0.905, bottom=0.015)
    save(fig, OUT / "graphrag_vs_vector_rag.png")
    print("wrote graphrag_vs_vector_rag.png")


if __name__ == "__main__":
    main()
