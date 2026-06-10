"""Part 4: where the recommender works — Hit@10 by cuisine and by restaurant popularity.
Recomputed from the surviving unextended KGAT-SAL + proximity predictions.
Regenerate: python figures/make_segments.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd, numpy as np

for cand in ["Inter", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})

pred = pd.read_parquet("data/predictions/kgat_sal_all_T3_nospatial_prox_a30_predictions.parquet",
                       columns=["true_place_id", "rank"])
rest = pd.read_parquet("data/restaurants_enriched_llmcz.parquet",
                       columns=["place_id", "cuisine_category", "user_rating_count"]).set_index("place_id")
KEEP = [c for c in rest.cuisine_category.value_counts().index if isinstance(c, str) and c != "Other"][:16]
g16 = lambda c: c if c in KEEP else "Other"
pred["hit"] = pred["rank"].notna() & (pred["rank"] <= 10)
pred = pred.join(rest, on="true_place_id")
overall = pred.hit.mean()

# by cuisine (>=100 test users)
pred["cz16"] = pred.cuisine_category.map(g16)
cz = pred.groupby("cz16").agg(hit=("hit", "mean"), n=("hit", "size"))
cz = cz[cz.n >= 100].sort_values("hit")

# by popularity bin
pred["pop"] = pd.cut(np.log1p(pred.user_rating_count), 5,
                     labels=["Very niche", "Niche", "Moderate", "Popular", "Very popular"])
pb = pred.groupby("pop", observed=True).hit.mean()

fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.25, 1]})
fig.suptitle("Where the recommender works  (Hit@10, KGAT-SAL + proximity)", fontsize=15, fontweight="bold", y=1.0)

colors = ["#2f9e95" if v >= overall else "#d96459" for v in cz.hit]
ax[0].barh(range(len(cz)), cz.hit, color=colors)
ax[0].set_yticks(range(len(cz))); ax[0].set_yticklabels(cz.index, fontsize=9)
ax[0].axvline(overall, ls="--", color="#555", lw=1.3)
ax[0].text(overall, len(cz)-0.3, f" overall {overall:.2f}", fontsize=8.5, color="#555")
ax[0].set_xlabel("Hit@10"); ax[0].set_title("by cuisine", fontsize=12)
for i, (v, n) in enumerate(zip(cz.hit, cz.n)):
    ax[0].text(v+0.003, i, f"{v:.2f}", va="center", fontsize=8)

ax[1].bar(range(len(pb)), pb.values, color="#3182bd")
ax[1].axhline(overall, ls="--", color="#555", lw=1.3)
ax[1].set_xticks(range(len(pb))); ax[1].set_xticklabels(pb.index, rotation=20, ha="right", fontsize=9)
ax[1].set_ylabel("Hit@10"); ax[1].set_title("by restaurant popularity", fontsize=12)
for i, v in enumerate(pb.values):
    ax[1].text(i, v+0.003, f"{v:.2f}", ha="center", fontsize=8.5)
ax[1].text(0.5, 0.9, "niche & destination spots\nare hardest to predict", transform=ax[1].transAxes,
           ha="center", fontsize=9, color="#a33", style="italic")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figures/segment_performance.png", dpi=160, bbox_inches="tight", facecolor="white")
print("wrote figures/segment_performance.png")
print("overall Hit@10:", round(overall, 4))
print("cuisine hit:", dict(cz.hit.round(3)))
print("popularity hit:", dict(pb.round(3)))
