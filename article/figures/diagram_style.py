"""Shared layout primitives for the schematic figures.

Cards use a flat face, a hairline border and a left accent stripe. Connectors are
drawn as orthogonal elbows anchored to card edges so that no line crosses a card.
"""
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from publication_style import INK, MUTED, GRID

FLOW = "#8A94A6"


def card(ax, x, y, w, h, title, body=None, accent=INK, face="#FFFFFF",
         title_size=12.6, body_size=10.6, badge=None, title_at_top=False):
    """Draw a card with a left accent stripe. Returns its edge anchors."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
        linewidth=1.1, edgecolor=GRID, facecolor=face, zorder=3, mutation_aspect=0.55,
    ))
    ax.add_patch(Rectangle((x, y + 0.008), 0.0055, h - 0.016, facecolor=accent,
                           edgecolor="none", zorder=4))
    tx = x + 0.026
    ty = y + h - (0.030 if (body or title_at_top) else h / 2 + 0.012)
    if badge is not None:
        ax.text(tx, ty, badge, va="top", fontsize=10.0, fontweight="bold",
                color=accent, zorder=5)
        tx += 0.022
    ax.text(tx, ty, title, va="top", fontsize=title_size, fontweight="bold",
            color=INK, zorder=5)
    if body:
        ax.text(x + 0.026, y + h - 0.075, body, va="top", fontsize=body_size,
                linespacing=1.5, color=MUTED, zorder=5)
    return {
        "top": (x + w / 2, y + h), "bottom": (x + w / 2, y),
        "left": (x, y + h / 2), "right": (x + w, y + h / 2),
        "x": x, "y": y, "w": w, "h": h,
    }


def band_label(ax, x, y, text):
    ax.text(x, y, text.upper(), fontsize=9.8, fontweight="bold", color=MUTED,
            letterspacing=1.0 if hasattr(ax, "letterspacing") else None)


def hline(ax, a, b, color=FLOW, lw=1.8):
    ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                       zorder=2, solid_capstyle="round"))


def arrow_h(ax, a, b, gap=0.012, color=FLOW, lw=1.8):
    """Straight horizontal connector, left edge to right edge."""
    ax.add_patch(FancyArrowPatch((a[0] + gap, a[1]), (b[0] - gap, b[1]),
                                 arrowstyle="-|>", mutation_scale=14, linewidth=lw,
                                 color=color, zorder=2, shrinkA=0, shrinkB=0))


def arrow_v(ax, a, b, gap=0.010, color=FLOW, lw=1.8):
    """Straight vertical connector, bottom edge down to top edge."""
    ax.add_patch(FancyArrowPatch((a[0], a[1] - gap), (b[0], b[1] + gap),
                                 arrowstyle="-|>", mutation_scale=14, linewidth=lw,
                                 color=color, zorder=2, shrinkA=0, shrinkB=0))


def elbow_split(ax, src_bottom, targets, mid_frac=0.5, gap=0.010,
                color=FLOW, lw=1.8):
    """One source fanning down into several targets via a shared horizontal rail."""
    y0 = src_bottom[1] - gap
    y1 = min(t[1] for t in targets) + gap
    rail = y1 + (y0 - y1) * mid_frac
    hline(ax, (src_bottom[0], y0), (src_bottom[0], rail), color, lw)
    hline(ax, (min(t[0] for t in targets), rail), (max(t[0] for t in targets), rail), color, lw)
    for t in targets:
        ax.add_patch(FancyArrowPatch((t[0], rail), (t[0], t[1] + gap),
                                     arrowstyle="-|>", mutation_scale=14, linewidth=lw,
                                     color=color, zorder=2, shrinkA=0, shrinkB=0))


def elbow_merge(ax, sources, dst_top, mid_frac=0.5, gap=0.010, color=FLOW, lw=1.8):
    """Several sources converging down into one target via a shared horizontal rail."""
    y0 = min(s[1] for s in sources) - gap
    y1 = dst_top[1] + gap
    rail = y1 + (y0 - y1) * mid_frac
    for s in sources:
        hline(ax, (s[0], s[1] - gap), (s[0], rail), color, lw)
    hline(ax, (min(s[0] for s in sources), rail), (max(s[0] for s in sources), rail), color, lw)
    ax.add_patch(FancyArrowPatch((dst_top[0], rail), (dst_top[0], y1),
                                 arrowstyle="-|>", mutation_scale=14, linewidth=lw,
                                 color=color, zorder=2, shrinkA=0, shrinkB=0))


def panel(ax, x, y, w, h, face="#F6F8FB"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.014",
        linewidth=1.0, edgecolor=GRID, facecolor=face, zorder=0,
        mutation_aspect=0.55))


def heading(fig, headline, subtitle=None, y=0.975, gap=0.045,
            title_size=17.5, sub_size=10.8):
    fig.suptitle(headline, fontsize=title_size, fontweight="bold", color=INK, y=y)
    if subtitle:
        fig.text(0.5, y - gap, subtitle, ha="center", va="top",
                 fontsize=sub_size, color=MUTED)

def elbow_split_h(ax, src_right, targets, mid_frac=0.5, gap=0.010,
                  color=FLOW, lw=1.8):
    """One source fanning rightward into several targets via a vertical rail."""
    x0 = src_right[0] + gap
    x1 = min(t[0] for t in targets) - gap
    rail = x0 + (x1 - x0) * mid_frac
    hline(ax, (x0, src_right[1]), (rail, src_right[1]), color, lw)
    hline(ax, (rail, min(t[1] for t in targets)),
          (rail, max(t[1] for t in targets)), color, lw)
    for t in targets:
        ax.add_patch(FancyArrowPatch((rail, t[1]), (t[0] - gap, t[1]),
                                     arrowstyle="-|>", mutation_scale=14, linewidth=lw,
                                     color=color, zorder=2, shrinkA=0, shrinkB=0))


def elbow_merge_h(ax, sources, dst_left, mid_frac=0.5, gap=0.010,
                  color=FLOW, lw=1.8):
    """Several sources converging rightward into one target via a vertical rail."""
    x0 = max(s[0] for s in sources) + gap
    x1 = dst_left[0] - gap
    rail = x0 + (x1 - x0) * mid_frac
    for s in sources:
        hline(ax, (s[0] + gap, s[1]), (rail, s[1]), color, lw)
    hline(ax, (rail, min(s[1] for s in sources)),
          (rail, max(s[1] for s in sources)), color, lw)
    ax.add_patch(FancyArrowPatch((rail, dst_left[1]), (x1, dst_left[1]),
                                 arrowstyle="-|>", mutation_scale=14, linewidth=lw,
                                 color=color, zorder=2, shrinkA=0, shrinkB=0))
