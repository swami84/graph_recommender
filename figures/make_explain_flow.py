"""Part 4: recommendation -> explanation flow (clean box style).
Pipeline (request -> scoring -> proximity -> top-K) feeds an explanation layer that RETRIEVES
grounding facts (cheap, always) and optionally GENERATES with an LLM (on demand).
Regenerate: python figures/make_explain_flow.py"""
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
SLATE, BLUE, TEAL, AMBER = "#62707f", "#3b82c4", "#3d8577", "#e0922f"

fig, ax = plt.subplots(figsize=(13.5, 7.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.text(50, 95, "From a diner's request to an explained recommendation", ha="center",
        fontsize=15, fontweight="bold", color=INK)

def box(cx, cy, w, h, color, title, sub):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.3,rounding_size=2",
                 facecolor=color, edgecolor=color, zorder=3))
    ax.text(cx, cy+(h*0.16 if sub else 0), title, ha="center", va="center",
            fontsize=10.3, fontweight="bold", color="white", zorder=4)
    if sub:
        ax.text(cx, cy-h*0.24, sub, ha="center", va="center", fontsize=7.7, color="#eef1f4", zorder=4)

def arrow(p, c, color=AR, lw=2.0):
    ax.add_patch(FancyArrowPatch(p, c, arrowstyle="-|>", mutation_scale=14, lw=lw,
                 color=color, zorder=2, shrinkA=1, shrinkB=2))

# ── recommendation pipeline (top row) ──
ax.text(4, 87, "RECOMMENDATION PIPELINE", fontsize=8.5, color=MUT, fontweight="bold")
P = [(12, "Diner request", "open app / ask"), (34, "GNN scoring", "rank all restaurants"),
     (56, "Proximity\nre-rank", "boost what's near"), (80, "Top-K\nrecommendations", "the shortlist")]
W = [16, 18, 16, 20]
for (cx, t, s), w in zip(P, W): box(cx, 76, w, 11, SLATE, t, s)
for i in range(3):
    arrow((P[i][0]+W[i]/2, 76), (P[i+1][0]-W[i+1]/2, 76))

# ── explanation layer panel ──
ax.add_patch(FancyBboxPatch((3, 8), 94, 54, boxstyle="round,pad=0.5,rounding_size=3",
             facecolor="#f7f8fa", edgecolor="#d7dce2", linewidth=1.4, zorder=0))
ax.text(7, 57, "EXPLANATION LAYER", fontsize=9, color=MUT, fontweight="bold")
arrow((80, 70.5), (80, 62), color=SLATE)
ax.text(82.5, 66, "top-K", fontsize=7.6, color=MUT, style="italic")

box(18, 33, 26, 17, BLUE, "Retrieve grounding facts", "history · location ·\ntaste profile · KG")
ax.text(18, 46, "the “R” — cheap, every recommendation", ha="center", fontsize=7.8, color=BLUE, style="italic")
box(51, 43, 27, 11, TEAL, "Templated explanation", "default · all users · instant")
box(51, 21, 27, 11, AMBER, "LLM synthesis (RAG)", "on demand · when asked “why?”")
ax.text(51, 14, "the “AG” — grounded in the retrieved facts", ha="center", fontsize=7.8,
        color="#b9781f", style="italic")
box(83, 33, 22, 17, SLATE, "Explanation shown", "with the recommendation")

arrow((31, 35), (37.5, 43), color=BLUE)
arrow((31, 31), (37.5, 21), color=BLUE)
arrow((64.5, 43), (72, 35), color=TEAL)
arrow((64.5, 21), (72, 31), color=AMBER)

plt.savefig("figures/explanation_flow.png", dpi=165, bbox_inches="tight", facecolor="white")
print("wrote figures/explanation_flow.png")
