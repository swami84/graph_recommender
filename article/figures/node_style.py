"""Shared node rendering and palette for the Part 2 schema diagrams.

Both `make_graph_schema.py` and `make_feature_taxonomy.py` import from here so
the two figures use one palette and one lighting model.
"""

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse, FancyArrowPatch

# --------------------------------------------------------------- palette
# Entity colours are shared by both figures. User and Restaurant are the two
# sides of one interaction, so they are drawn as a matched warm/cool pair from
# the series palette rather than as unrelated hues.
USER = "#2A9D8F"          # series TEAL
USER_TINT = "#7FC8BE"     # feature satellites, a lighter tint of the same hue
ITEM = "#E76F51"          # series CORAL
ITEM_TINT = "#F0A288"
KG = "#5A7799"            # knowledge-graph entities, a cool slate
KG_TINT = "#8FA6BD"

CONNECTOR = "#A9B4C2"
CONNECTOR_SOFT = "#C3CBD6"

# card palette, shared with the taxonomy figure
ANALYTIC = "#3977A8"
ANALYTIC_FACE = "#EAF1F8"
SEMANTIC = "#D9922B"
SEMANTIC_FACE = "#FDF3E3"
SHELL = "#4A5A6E"
SHELL_FACE = "#EDF1F6"


def aspect_of(fig):
    """Data-space y/x scale that makes a unit circle render round."""
    w, h = fig.get_size_inches()
    return h / w


def sphere(ax, x, y, r, color, aspect, zorder=3, ring=False, n=420):
    """A lit sphere: key light, fill bounce, specular, fresnel rim, contact AO.

    The shadow is a separate soft alpha field rather than an offset copy of the
    disc, so the node reads as a solid object instead of an outlined circle.
    """
    rx, ry = r * aspect, r

    # ---------------------------------------------------------- soft shadow
    pad = 1.55
    sv, su = np.mgrid[-pad:pad:complex(n), -pad:pad:complex(n)]
    # offset down and slightly right, squashed vertically like a cast shadow
    sd = np.sqrt(((su - 0.16) / 1.02) ** 2 + ((sv + 0.30) / 0.82) ** 2)
    falloff = np.clip(1.0 - sd, 0.0, 1.0) ** 2.1
    shadow = np.zeros(su.shape + (4,))
    shadow[..., :3] = np.array([0.09, 0.13, 0.18])
    shadow[..., 3] = 0.30 * falloff
    ax.imshow(shadow, extent=(x - rx * pad, x + rx * pad,
                              y - ry * pad, y + ry * pad),
              origin="lower", aspect="auto", zorder=zorder - 1,
              interpolation="bilinear")

    # ---------------------------------------------------------- lit surface
    v, u = np.mgrid[-1:1:complex(n), -1:1:complex(n)]
    rr = u * u + v * v
    nz = np.sqrt(np.clip(1.0 - rr, 0.0, 1.0))

    key = np.array([-0.62, 0.50, 0.60])
    key /= np.linalg.norm(key)
    fill = np.array([0.58, -0.46, 0.38])
    fill /= np.linalg.norm(fill)

    lam_key = np.clip(u * key[0] + v * key[1] + nz * key[2], 0.0, 1.0)
    lam_fill = np.clip(u * fill[0] + v * fill[1] + nz * fill[2], 0.0, 1.0)

    base = np.array(to_rgb(color))
    deep = base * 0.33
    rgb = deep[None, None, :] + (base - deep)[None, None, :] * \
        (0.16 + 0.84 * lam_key ** 0.90)[..., None]

    # bounce light keeps the shadow side from going dead
    rgb = rgb + (1.0 - rgb) * (0.11 * lam_fill)[..., None]

    # specular, a tight highlight from the key light
    half = key + np.array([0.0, 0.0, 1.0])
    half /= np.linalg.norm(half)
    spec = np.clip(u * half[0] + v * half[1] + nz * half[2], 0.0, 1.0) ** 95
    rgb = rgb + (1.0 - rgb) * (0.42 * spec)[..., None]

    # fresnel rim on the shaded side reads as curvature
    fresnel = (1.0 - nz) ** 3.0
    rgb = rgb + (1.0 - rgb) * (0.24 * fresnel * lam_fill)[..., None]

    # occlusion at the extreme edge crisps the silhouette without a stroke
    occl = np.clip((rr - 0.84) / 0.16, 0.0, 1.0)
    rgb = rgb * (1.0 - 0.20 * occl)[..., None]

    radial = np.sqrt(rr)
    alpha = np.clip((1.0 - radial) * (n * 0.40), 0.0, 1.0)
    rgba = np.dstack([np.clip(rgb, 0.0, 1.0), alpha])
    ax.imshow(rgba, extent=(x - rx, x + rx, y - ry, y + ry), origin="lower",
              aspect="auto", zorder=zorder, interpolation="bilinear")

    if ring:
        ax.add_patch(Ellipse((x, y), 2 * rx, 2 * ry, facecolor="none",
                             edgecolor="white", linewidth=1.2, alpha=0.5,
                             zorder=zorder + 1))


def fit_fontsize(text, max_width_data, fig_width_in, requested, minimum=7.0,
                 weight=0.62, padding=0.74):
    """Shrink a label until it fits inside a node, estimating bold sans metrics."""
    if not text:
        return requested
    available_pt = max_width_data * fig_width_in * 72.0 * padding
    fitted = available_pt / (weight * len(text))
    return max(minimum, min(requested, fitted))


def labelled_sphere(ax, x, y, r, color, aspect, fig_width_in, title="", count="",
                    title_size=12.5, count_size=9.6, zorder=3):
    """Sphere plus a centred title and an optional count line, both auto-fitted."""
    sphere(ax, x, y, r, color, aspect, zorder=zorder)
    rx = r * aspect
    if title:
        ts = fit_fontsize(title, 2 * rx, fig_width_in, title_size)
        dy = 0.30 * r if count else 0.0
        ax.text(x, y + dy, title, ha="center", va="center", fontsize=ts,
                fontweight="bold", color="white", zorder=zorder + 2)
    if count:
        cs = fit_fontsize(count, 2 * rx, fig_width_in, count_size)
        ax.text(x, y - 0.34 * r, count, ha="center", va="center", fontsize=cs,
                color="white", alpha=0.94, zorder=zorder + 2)


def boundary(start, end, start_r, end_r, aspect, gap_start=0.0, gap_end=0.0):
    """Trim a segment to the two node silhouettes, leaving an optional gap."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = np.hypot(dx, dy)
    if length == 0:
        return start, end
    ux, uy = dx / length, dy / length

    def along(radius):
        if not radius:
            return 0.0
        rx, ry = radius * aspect, radius
        return 1.0 / np.sqrt((ux / rx) ** 2 + (uy / ry) ** 2)

    a = along(start_r) + gap_start
    b = along(end_r) + gap_end
    return ((start[0] + ux * a, start[1] + uy * a),
            (end[0] - ux * b, end[1] - uy * b))


def connector(ax, p1, p2, color=CONNECTOR, lw=2.4, head=17, rad=0.0, zorder=1,
              arrow=True):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>" if arrow else "-", mutation_scale=head,
        linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}",
        shrinkA=0, shrinkB=0, capstyle="round", joinstyle="round", zorder=zorder))


def edge_label(ax, p1, p2, text, aspect, fig_width_in, fig_height_in, frac=0.5,
               dx=0.0, dy=0.0, rotate=True, fontsize=9.4, color="#667085",
               ha="center", zorder=8):
    """Place a label along a segment, rotated to match it in rendered space."""
    mx = p1[0] + (p2[0] - p1[0]) * frac + dx
    my = p1[1] + (p2[1] - p1[1]) * frac + dy
    angle = 0.0
    if rotate:
        rise = (p2[1] - p1[1]) * fig_height_in
        run = (p2[0] - p1[0]) * fig_width_in
        angle = np.degrees(np.arctan2(rise, run))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
    ax.text(mx, my, text, ha=ha, va="center", fontsize=fontsize,
            fontweight="bold", color=color, rotation=angle,
            rotation_mode="anchor" if rotate else None,
            bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                      edgecolor="none", alpha=0.94),
            zorder=zorder)
