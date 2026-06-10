"""Graph schema for Part 1, Step 5 — clean box-and-arrow style (matches the model-landscape look).
diner profile (features) -> reviewer --REVIEWED--> restaurant -> knowledge-graph entities.
Regenerate: python figures/make_graph_schema.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

for cand in ["Inter", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["font.size"] = 10
INK, MUT, AR = "#2b3038", "#6c757d", "#9aa3ad"
TEAL, CORAL, PURP, SLATE = "#3d8577", "#c05f63", "#7d6699", "#79899b"

fig, ax = plt.subplots(figsize=(13, 6.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.text(50, 94, "The recommendation graph", ha="center", fontsize=17, fontweight="bold", color=INK)
ax.text(50, 87.5, "diners and restaurants, joined by 10M reviews   ·   5 node types   ·   4 relations",
        ha="center", fontsize=10.5, color=MUT)

def rbox(x, y, w, h, fc, ec, lw=1.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=2",
                 facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))

def solid(cx, cy, w, h, color, title, sub):
    rbox(cx-w/2, cy-h/2, w, h, color, color)
    ax.text(cx, cy+1.6, title, ha="center", va="center", fontsize=11, fontweight="bold", color="white", zorder=4)
    ax.text(cx, cy-2.6, sub, ha="center", va="center", fontsize=8.6, color="#eef1f4", zorder=4)

def arrow(p, c, color=AR, lw=2.0):
    ax.add_patch(FancyArrowPatch(p, c, arrowstyle="-|>", mutation_scale=15, lw=lw,
                 color=color, zorder=2, shrinkA=1, shrinkB=2))

# ── diner profile panel (features) ──
rbox(4, 28, 22, 44, "#eef5f3", "#cfe0db", 1.4)
ax.text(15, 67.5, "Diner profile", ha="center", fontsize=11, fontweight="bold", color=TEAL)
ax.text(15, 63.5, "node features", ha="center", fontsize=8.4, color=MUT, style="italic")
for i, f in enumerate(["demographics", "taste preferences", "dietary affinity", "home area"]):
    ax.text(8, 56 - i*7.5, "•  " + f, ha="left", va="center", fontsize=9.6, color=INK)

# ── main spine: reviewer -> restaurant ──
solid(40, 50, 15, 12, TEAL, "reviewer", "1.96 M")
solid(62, 50, 15, 12, CORAL, "restaurant", "18,879")
arrow((26, 50), (32.5, 50), color=TEAL)               # profile -> reviewer
ax.text(29.2, 52.4, "describe", ha="center", fontsize=8, color=TEAL, style="italic")
arrow((47.5, 50), (54.5, 50), color=TEAL, lw=4)        # REVIEWED backbone
ax.text(51, 59, "REVIEWED · 10M", ha="center", fontsize=8.6, color="#256b5f", style="italic")
ax.text(51, 41, "collaborative-filtering backbone", ha="center", fontsize=7.8, color=TEAL,
        style="italic", fontweight="bold")

# ── knowledge-graph entities ──
ax.text(85, 79, "Knowledge graph", ha="center", fontsize=10.5, fontweight="bold", color=PURP)
solid(85, 67, 14, 11, PURP, "dish", "517 K")
solid(85, 50, 14, 11, SLATE, "cbg", "666")
solid(85, 33, 14, 11, SLATE, "cuisine", "17")
def rel(ty, label):
    arrow((69.5, 50), (78, ty), color=AR)
    mx, my = 74, (50 + ty)/2 + (1.6 if ty > 50 else -1.6)
    ax.text(73.5, my, label, ha="center", va="center", fontsize=7.8, color=MUT, style="italic",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"), zorder=5)
rel(67, "SERVES"); rel(50, "LOCATED_IN"); rel(33, "HAS_CUISINE")

plt.savefig("figures/graph_schema.png", dpi=170, bbox_inches="tight", facecolor="white")
print("wrote figures/graph_schema.png")
