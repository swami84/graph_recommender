"""Hierarchical feature taxonomy for Part 2 (polished, 3 levels with feature chips).
Root -> {User, Restaurant} -> groups -> example feature chips, colored by track.
Regenerate: python figures/make_feature_tree.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ---- pick a clean font if available ----
for cand in ["Inter", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; FONT = cand; break
plt.rcParams["font.size"] = 10

A_AC, A_FILL, A_CHIP = "#3b82c4", "#eef5fc", "#dceaf8"
L_AC, L_FILL, L_CHIP = "#e08a2b", "#fdf4e8", "#fbe6cd"
USERC, ITEMC, ROOTC = "#2f9e95", "#d96459", "#39404d"
CONN, INK, MUT = "#aab2bd", "#262b33", "#5d6470"
SH = [pe.withSimplePatchShadow(offset=(2, -2), alpha=0.15, shadow_rgbFace="#3a3f47")]

# ---- feature groups: (name, count, track, [example features]) ----
USER = [("base", "13", "A", ["local guide", "review count", "race", "gender", "location"]),
        ("extended", "44", "A", ["cuisine affinity", "geo range", "CBG diversity", "rating pickiness", "meal-time prefs"]),
        ("pref", "12", "L", ["spice", "noise", "formality", "novelty", "value"]),
        ("dietary", "6", "L", ["vegan", "vegetarian", "gluten-free", "proteins"])]
ITEM = [("base", "29", "A", ["price", "rating", "cuisine", "Food / Service / Atmosphere"]),
        ("nlp", "12", "L", ["spice", "noise", "romantic", "family", "value", "outdoor"]),
        ("extended", "22", "A", ["dish diversity", "spatial density", "rating trend", "CBG foot-traffic"]),
        ("dietary", "5", "L", ["vegan", "vegetarian", "gluten-free", "halal", "kosher"])]

CARD_X, CARD_W, PAD, CHIP_H, CHIP_GAP, CHAR_W = 52, 46, 1.8, 3.6, 1.3, 0.58
HEAD_H, GROUP_GAP, CLUSTER_GAP, TOP = 6.5, 3.5, 10, 240

def layout_chips(feats):
    avail = CARD_W - 2*PAD; x, row, out = PAD, 0, []
    for f in feats:
        w = 3.4 + CHAR_W*len(f)
        if x > PAD and x + w > PAD + avail: row += 1; x = PAD
        out.append((f, x, row, w)); x += w + CHIP_GAP
    nrows = (out[-1][2] + 1) if out else 1
    h = PAD + HEAD_H + nrows*(CHIP_H + CHIP_GAP) + PAD - CHIP_GAP
    return out, h

# ---- vertical layout (top -> down), compute card boxes + centers ----
groups = []  # (name,count,track,chips_layout, top, height, center)
y = TOP
for ci, cluster in enumerate([USER, ITEM]):
    for (n, c, tk, feats) in cluster:
        lay, h = layout_chips(feats)
        top = y; y -= h; center = (top + y)/2
        groups.append([n, c, tk, lay, top, h, center])
        y -= GROUP_GAP
    y -= CLUSTER_GAP
ucen = sum(g[6] for g in groups[:4])/4
icen = sum(g[6] for g in groups[4:])/4
rcen = (ucen + icen)/2
BOT = y

fig, ax = plt.subplots(figsize=(15, max(11, (TOP-BOT)/14)))
ax.set_xlim(0, 100); ax.set_ylim(BOT-4, TOP+10); ax.axis("off")
ax.text(50, TOP+7, "The feature taxonomy", ha="center", fontsize=19, fontweight="bold", color=INK)
ax.text(50, TOP+2.5, "≈ 200 candidate features across two tracks", ha="center", fontsize=11.5, color=MUT)

def rbox(x, y, w, h, fc, ec, lw, shadow=True):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=2",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=4)
    if shadow: p.set_path_effects(SH)
    ax.add_patch(p); return p

def node(x, y, w, h, color, title, sub):
    rbox(x, y, w, h, color, color, 0)
    ax.text(x+w/2, y+h/2+(1.6 if sub else 0), title, ha="center", va="center",
            fontsize=15, fontweight="bold", color="white", zorder=5)
    if sub: ax.text(x+w/2, y+h/2-3.2, sub, ha="center", va="center", fontsize=9.5, color="#e9edf1", zorder=5)

def arrow(p, c, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, c, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                 mutation_scale=13, lw=1.9, color=CONN, zorder=2,
                 shrinkA=2, shrinkB=3, joinstyle="round", capstyle="round"))

# nodes
node(2, rcen-9, 19, 18, ROOTC, "Features", "≈200 candidates")
node(27, ucen-7, 17, 14, USERC, "User", "diner")
node(27, icen-7, 17, 14, ITEMC, "Restaurant", "item")
arrow((21, rcen), (27, ucen), 0.05); arrow((21, rcen), (27, icen), -0.05)

# group cards + chips
for (n, c, tk, lay, top, h, center) in groups:
    ac, fill, chip = (A_AC, A_FILL, A_CHIP) if tk == "A" else (L_AC, L_FILL, L_CHIP)
    cb = top - h
    rbox(CARD_X, cb, CARD_W, h, fill, ac, 1.8)
    ax.text(CARD_X+PAD+0.6, top-3.4, n, ha="left", va="center", fontsize=12.5, fontweight="bold", color=ac, zorder=5)
    # count pill
    pw, ph = 9, 4.8; px, py = CARD_X+CARD_W-PAD-pw, top-PAD-ph+0.4
    rbox(px, py, pw, ph, ac, ac, 0, shadow=False)
    ax.text(px+pw/2, py+ph/2, c+" feat", ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=6)
    # chips
    for (f, cx, rowi, w) in lay:
        chy = top - HEAD_H - rowi*(CHIP_H+CHIP_GAP) - CHIP_H
        rbox(CARD_X+cx, chy, w, CHIP_H, chip, ac, 1.0, shadow=False)
        ax.text(CARD_X+cx+w/2, chy+CHIP_H/2, f, ha="center", va="center", fontsize=8.2, color=INK, zorder=6)
    branch = ucen if center > rcen else icen
    arrow((44, branch), (CARD_X, center), 0.04 if center > branch else -0.04)

# legend
def swatch(x, ac, fill, label):
    rbox(x, BOT-1, 4, 3.4, fill, ac, 1.8, shadow=False)
    ax.text(x+5.5, BOT+0.7, label, va="center", fontsize=10.5, color=INK)
swatch(6, A_AC, A_FILL, "analytical  ·  computed from data")
swatch(52, L_AC, L_FILL, "LLM-extracted  ·  read from review text")

plt.savefig("figures/feature_tree.png", dpi=160, bbox_inches="tight", facecolor="white")
print(f"wrote figures/feature_tree.png  (font: {plt.rcParams['font.family']})")
