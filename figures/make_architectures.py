"""Part 3: model architectures — KGAT, SAL, InfoNCE, and the hybrid that fuses them.
Four clean mini block-diagrams. Regenerate: python figures/make_architectures.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

for cand in ["Inter", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["font.size"] = 10
INK, MUT = "#2b3038", "#6c757d"
TEAL, PURP, AMBER, GOLD = "#3d8577", "#7d6699", "#e0922f", "#c2882f"

def pbox(ax, cx, cy, w, h, ac, text, fs=8.6, solid=False):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.3,rounding_size=3",
                 facecolor=(ac if solid else ac+"22"), edgecolor=ac, linewidth=1.8, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=("white" if solid else INK), fontweight=("bold" if solid else "normal"), zorder=4)

def parrow(ax, x1, y1, x2, y2, ac="#9aa3ad"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                 lw=1.9, color=ac, zorder=2, shrinkA=2, shrinkB=2))

def panel(ax, title, ac, boxes, caption):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(2, 90, title, ha="left", fontsize=11, fontweight="bold", color=ac)
    cxs = [18, 50, 82]
    for cx, txt in zip(cxs, boxes):
        pbox(ax, cx, 52, 26, 26, ac, txt)
    parrow(ax, 31, 52, 37, 52, ac); parrow(ax, 63, 52, 69, 52, ac)
    ax.text(50, 14, caption, ha="center", fontsize=8.3, color=MUT, style="italic")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Inside the models", fontsize=17, fontweight="bold", y=0.98)

panel(axes[0, 0], "KGAT — knowledge-graph attention", TEAL,
      ["Knowledge graph\ncuisine · dish ·\nCBG · price",
       "Attentive\npropagation\n(KG + CF)",
       "Enriched item\n& user embeddings\n→ ranking (BPR)"],
      "injects side information → attacks the cold-start problem")

panel(axes[0, 1], "SAL — self-augmented temporal denoising", PURP,
      ["A diner's\nreview history",
       "Split into\nT time periods",
       "Stability-weighted,\ndenoised\nembedding"],
      "down-weights one-off, out-of-character visits as tastes drift")

panel(axes[1, 0], "InfoNCE — contrastive learning", AMBER,
      ["The graph",
       "Two augmented\nviews\n(dropout + noise)",
       "Pull the same node's\ntwo views together,\npush others apart"],
      "spreads embeddings out · counters popularity bias on the long tail")

# ── panel 4: the hybrid ──
ax = axes[1, 1]; ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.text(2, 90, "Hybrid — InfoNCE-KGAT-SAL", ha="left", fontsize=11, fontweight="bold", color=GOLD)
pbox(ax, 24, 66, 34, 18, TEAL, "KGAT-SAL backbone\n(KG attention +\ntemporal denoising)")
pbox(ax, 24, 34, 34, 16, AMBER, "InfoNCE contrastive\non KG-enriched\nembeddings")
pbox(ax, 76, 50, 40, 22, GOLD, "L  =  L_BPR\n+  λ_cl · L_CL\n+  λ_sal · L_SAL", fs=10, solid=True)
parrow(ax, 41, 64, 56, 54, GOLD); parrow(ax, 41, 36, 56, 46, GOLD)
ax.text(50, 12, "fuses knowledge graph + temporal denoising + contrastive learning",
        ha="center", fontsize=8.3, color=MUT, style="italic")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figures/model_architectures.png", dpi=160, bbox_inches="tight", facecolor="white")
print("wrote figures/model_architectures.png")
