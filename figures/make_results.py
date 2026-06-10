"""Part 4 headline: 'geography beats architecture'. Base vs +proximity from canonical results.
Regenerate: python figures/make_results.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd, numpy as np

for cand in ["Inter", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
BASEC, PROXC, INK = "#b9c0cb", "#2f9e95", "#262b33"

df = pd.read_csv("results/model_results.csv")
df = df[df["emb_dim"] != 1024]                       # canonical (unextended) only
df["prox"] = df["notes"].fillna("").str.contains("proximity")
ORDER = ["InfoNCE-KGAT-SAL", "KGAT-SAL", "KGAT", "InfoNCE", "RaDAR", "LightGCN"]
def best(model, prox, col):
    s = df[(df.model == model) & (df.prox == prox)][col]
    return s.max() if len(s) else np.nan

base_n = [best(m, False, "ndcg_at_10") for m in ORDER]
prox_n = [best(m, True,  "ndcg_at_10") for m in ORDER]
base_p = [best(m, False, "precision_at_10") for m in ORDER]
prox_p = [best(m, True,  "precision_at_10") for m in ORDER]

fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
fig.suptitle("Geography beats architecture", fontsize=17, fontweight="bold", color=INK, y=1.0)
fig.text(0.5, 0.935, "a one-line spatial re-ranking lifts every model far more than the gaps between them",
         ha="center", fontsize=10.8, color="#5d6470")
y = np.arange(len(ORDER))[::-1]; h = 0.38

# NDCG panel
hb = ax[0].barh(y+h/2, base_n, h, color=BASEC, label="base model")
hp = ax[0].barh(y-h/2, prox_n, h, color=PROXC, label="+ proximity re-rank")
ax[0].set_yticks(y); ax[0].set_yticklabels(ORDER)
ax[0].set_xlabel("NDCG@10"); ax[0].set_title("Ranking quality (NDCG@10)")
for yy, b, p in zip(y, base_n, prox_n):
    ax[0].text(b+0.001, yy+h/2, f"{b:.3f}", va="center", fontsize=8, color="#555")
    ax[0].text(p+0.001, yy-h/2, f"{p:.3f}", va="center", fontsize=8, color="#1f6f68", fontweight="bold")
ax[0].set_xlim(0, 0.088)

# Precision panel (the dramatic jump)
ax[1].barh(y+h/2, base_p, h, color=BASEC, label="base model")
ax[1].barh(y-h/2, prox_p, h, color=PROXC, label="+ proximity re-rank")
ax[1].set_yticks(y); ax[1].set_yticklabels(ORDER)
ax[1].set_xlabel("Precision@10"); ax[1].set_title("Hit rate (Precision@10)")
for yy, b, p in zip(y, base_p, prox_p):
    ax[1].text(b+0.002, yy+h/2, f"{b:.3f}", va="center", fontsize=8, color="#555")
    ax[1].text(p+0.002, yy-h/2, f"{p:.3f}", va="center", fontsize=8, color="#1f6f68", fontweight="bold")
ax[1].set_xlim(0, 0.17)

fig.legend([hb, hp], ["base model", "+ proximity re-rank"], loc="lower center",
           ncol=2, frameon=True, framealpha=0.95, fontsize=11, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=[0, 0.05, 1, 0.93])
plt.savefig("figures/results_geography.png", dpi=160, bbox_inches="tight", facecolor="white")
print("wrote figures/results_geography.png")
print("base NDCG:", dict(zip(ORDER, [round(x,4) for x in base_n])))
print("prox NDCG:", dict(zip(ORDER, [round(x,4) for x in prox_n])))
print("base P@10:", dict(zip(ORDER, [round(x,4) for x in base_p])))
print("prox P@10:", dict(zip(ORDER, [round(x,4) for x in prox_p])))
