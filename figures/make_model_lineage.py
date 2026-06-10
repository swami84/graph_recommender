"""Model landscape / lineage for Part 3: baselines -> GNN-CF -> +KG -> +contrastive -> hybrid.
Regenerate: python figures/make_model_lineage.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

for cand in ["Inter", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["font.size"] = 10
INK, MUT = "#262b33", "#5d6470"
SH = [pe.withSimplePatchShadow(offset=(1.6, -1.6), alpha=0.14, shadow_rgbFace="#3a3f47")]

# (header, x, accent, fill, [models])
COLS = [
    ("Baselines", 9, "#7f8794", "#eef0f2", ["ALS", "Neural CF"]),
    ("GNN collaborative filtering", 30, "#3b82c4", "#eaf2fb", ["LightGCN", "UltraGCN", "SimGCL", "DGCF"]),
    ("+ Knowledge graph", 51, "#2f9e95", "#e9f6f4", ["KGAT", "KGAT-SAL"]),
    ("+ Contrastive / robust", 72, "#e08a2b", "#fdf2e3", ["InfoNCE", "IFL-GCL", "HEK-CL", "RaDAR"]),
    ("Hybrid", 91.5, "#c9893f", "#fbf0dc", ["InfoNCE-\nKGAT-SAL"]),
]
BW, BH, PITCH, MID = 17, 6.2, 8.4, 54

fig, ax = plt.subplots(figsize=(15, 7.6))
ax.set_xlim(0, 101); ax.set_ylim(8, 100); ax.axis("off")
ax.text(50, 96, "The model landscape", ha="center", fontsize=18, fontweight="bold", color=INK)
ax.text(50, 90.5, "from a humble matrix factorization to a knowledge-graph + contrastive hybrid",
        ha="center", fontsize=11, color=MUT)

pos = {}
for (hdr, x, ac, fill, models) in COLS:
    ax.text(x, 83, hdr, ha="center", va="center", fontsize=10.6, fontweight="bold", color=ac)
    n = len(models); ys = [MID + (n-1)*PITCH/2 - i*PITCH for i in range(n)]
    big = (hdr == "Hybrid")
    for m, y in zip(models, ys):
        w, h = (BW+1, BH+3) if big else (BW, BH)
        lw = 2.6 if big else 1.6
        p = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.3,rounding_size=1.8",
                           linewidth=lw, edgecolor=ac, facecolor=fill, zorder=4)
        p.set_path_effects(SH); ax.add_patch(p)
        ax.text(x, y, m, ha="center", va="center", fontsize=10.2 if not big else 11,
                fontweight="bold" if big else "normal", color=INK, zorder=5)
        pos[m] = (x, y, w)

# subtle vertical separators between the stage columns (no misleading flow arrows)
xs = [c[1] for c in COLS]
for a, b in zip(xs[:-1], xs[1:]):
    ax.plot([(a+b)/2, (a+b)/2], [MID-26, MID+24], color="#e3e6ea", lw=1.2, zorder=0)

# mark the hybrid's two ingredients with dashed-gold rings — NO crossing arrows
# (an arrow from KGAT-SAL to the hybrid would pass behind HEK-CL and read as a link to it)
for parent in ["KGAT-SAL", "InfoNCE"]:
    x, y, w = pos[parent]
    ax.add_patch(FancyBboxPatch((x-w/2-0.9, y-BH/2-0.9), w+1.8, BH+1.8,
                 boxstyle="round,pad=0.3,rounding_size=2", fill=False,
                 edgecolor="#c9893f", linewidth=2.4, linestyle=(0, (4, 2)), zorder=6))

ax.annotate("increasing structure & robustness  →", xy=(50, 14), ha="center",
            fontsize=10.5, color=MUT, style="italic")
ax.text(91.5, 45, "=  KGAT-SAL  +  InfoNCE", ha="center", fontsize=9.4, fontweight="bold", color="#a06a2a")
ax.text(91.5, 40.2, "the two dashed-gold models, fused", ha="center", fontsize=8, color="#a06a2a", style="italic")

plt.savefig("figures/model_lineage.png", dpi=160, bbox_inches="tight", facecolor="white")
print("wrote figures/model_lineage.png")
