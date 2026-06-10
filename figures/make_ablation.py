"""Part 4 feature-set ablation: KGAT (emb 2048, unextended), built up from base.
Three measured, nested configurations:
  base only (base+dietary)                 = "skip extended,pref,nlp,llm"  -> 0.0412
  + LLM attrs/prefs/stats (no behavioral)  = "skip extended"               -> 0.0493
  + behavioral & spatial (= all features)  = "all features"                -> 0.0623
(Footnote: dropping just the 6 review-statistics nudges all-features to 0.0658 — mild noise.)
Regenerate: python figures/make_ablation.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for cand in ["Inter", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams.update({"font.size": 10.5, "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})

labels = ["Base\n(identity · cuisine ·\nprice · dietary)",
          "+ LLM attributes,\npreferences\n& review stats",
          "+ Behavioral &\nspatial analytics\n(= all features)"]
ndcg = [0.0412, 0.0493, 0.0623]
colors = ["#b9c0cb", "#e08a2b", "#2f9e95"]

fig, ax = plt.subplots(figsize=(9.5, 5.8))
bars = ax.bar(range(3), ndcg, color=colors, width=0.6, edgecolor="white", linewidth=1.5)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("NDCG@10"); ax.set_ylim(0, 0.072)
ax.set_title("Building up the feature set  (KGAT)", fontsize=14, fontweight="bold")
for i, v in enumerate(ndcg):
    ax.text(i, v + 0.0010, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold", color="#262b33")

# delta annotations between bars
def delta(i, j, txt, color):
    ax.annotate("", xy=(j, ndcg[j]), xytext=(i, ndcg[i]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
    ax.text((i+j)/2, max(ndcg[i], ndcg[j])+0.004, txt, ha="center", fontsize=10,
            fontweight="bold", color=color)
delta(0, 1, "+0.008", "#c47a1f")
delta(1, 2, "+0.013", "#1f6f68")
ax.text(1.0, 0.013, "the LLM features add only a little;\nthe behavioral & spatial analytics drive the gain",
        ha="center", fontsize=9.3, color="#5d6470", style="italic")
plt.tight_layout()
plt.savefig("figures/feature_ablation.png", dpi=160, bbox_inches="tight", facecolor="white")
print("wrote figures/feature_ablation.png")
