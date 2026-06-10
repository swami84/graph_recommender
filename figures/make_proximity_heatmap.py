"""Part 4: proximity re-ranking grid — NDCG lift across distance-weight (alpha) x radius (bandwidth).
Canonical rows only (per-model max base_ndcg, dropping the extended re-runs).
Regenerate: python figures/make_proximity_heatmap.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd, numpy as np

for cand in ["Inter", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams["font.size"] = 9.5

px = pd.read_csv("results/proximity_grid_search.csv")
px = px[px.base_ndcg == px.groupby("csv_model").base_ndcg.transform("max")]   # canonical
MODELS = ["KGAT", "KGAT-SAL", "InfoNCE-KGAT-SAL", "InfoNCE", "RaDAR", "LightGCN"]
MODELS = [m for m in MODELS if m in px.csv_model.unique()]
bw_order = sorted(px.bandwidth_km.unique())
al_order = sorted(px.alpha.unique())
vmax = px.delta_ndcg.max()

fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
fig.suptitle("Proximity re-ranking: NDCG lift over the base model", fontsize=15, fontweight="bold", y=0.99)
fig.text(0.5, 0.935, "rows = weight on distance (α) · columns = distance radius (km) · greener = bigger gain",
         ha="center", fontsize=10, color="#5d6470")
im = None
for ax, m in zip(axes.ravel(), MODELS):
    g = px[px.csv_model == m].pivot_table("delta_ndcg", "alpha", "bandwidth_km").reindex(index=al_order, columns=bw_order)
    im = ax.imshow(g.values, cmap="YlGn", aspect="auto", vmin=0, vmax=vmax)
    ax.set_title(m, fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(bw_order))); ax.set_xticklabels([f"{int(b)}" for b in bw_order])
    ax.set_yticks(range(len(al_order))); ax.set_yticklabels([f"{a:.1f}" for a in al_order])
    ax.set_xlabel("radius (km)"); ax.set_ylabel("α")
    # mark best cell
    if np.isfinite(g.values).any():
        bi = np.unravel_index(np.nanargmax(g.values), g.shape)
        ax.add_patch(plt.Rectangle((bi[1]-0.5, bi[0]-0.5), 1, 1, fill=False, edgecolor="#c0392b", lw=2.2))
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                if np.isfinite(g.values[i, j]):
                    ax.text(j, i, f"{g.values[i,j]*1000:.1f}", ha="center", va="center", fontsize=7, color="#333")
for ax in axes.ravel()[len(MODELS):]:
    ax.axis("off")
fig.subplots_adjust(right=0.9, top=0.88, hspace=0.5, wspace=0.35)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("ΔNDCG@10 (×10³ shown in cells)")
plt.savefig("figures/proximity_heatmap.png", dpi=160, bbox_inches="tight", facecolor="white")
print("wrote figures/proximity_heatmap.png")
