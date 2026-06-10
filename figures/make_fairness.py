"""Part 4: best-model recommendation performance across inferred gender and race.
Hit@10 from the surviving unextended KGAT-SAL + proximity predictions, joined to per-user demographics.
'Unknown' excluded. Regenerate: python figures/make_fairness.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd, numpy as np

for cand in ["Inter", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand; break
plt.rcParams.update({"font.size": 10.5, "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
RACE = {"nh_white": "White (NH)", "nh_black": "Black (NH)", "api": "Asian/PI", "hispanic": "Hispanic"}  # AIAN excluded
RORDER = ["White (NH)", "Asian/PI", "Black (NH)", "Hispanic"]
RCOL = dict(zip(RORDER, ["#4393c3", "#e6822e", "#5ab4ac", "#d6604d"]))
GCOL = {"Female": "#d6604d", "Male": "#4393c3"}

pred = pd.read_parquet("data/predictions/kgat_sal_all_T3_nospatial_prox_a30_predictions.parquet",
                       columns=["contributor_id", "rank"])
pred["hit"] = pred["rank"].notna() & (pred["rank"] <= 10)
demo = pd.read_parquet("data/reviews_flat.parquet",
                       columns=["contributor_id", "predicted_race", "predicted_gender"]).drop_duplicates("contributor_id")
d = pred.merge(demo, on="contributor_id", how="left")
d["race"] = d.predicted_race.map(RACE)
d["gender"] = d.predicted_gender.str.capitalize()
overall = d.hit.mean()

gh = d[d.gender.isin(["Female", "Male"])].groupby("gender").hit.mean().reindex(["Female", "Male"])
rh = d[d.race.isin(RORDER)].groupby("race").hit.mean().reindex(RORDER).dropna()

fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1, 1.4]})
fig.suptitle("Recommendation performance across demographics  (KGAT-SAL + proximity)",
             fontsize=14, fontweight="bold", y=1.0)
ax[0].bar(gh.index, gh.values, color=[GCOL[g] for g in gh.index], width=0.55)
ax[0].axhline(overall, ls="--", color="#555", lw=1.3)
ax[0].set_title("by inferred gender"); ax[0].set_ylabel("Hit@10"); ax[0].set_ylim(0, 0.16)
for i, v in enumerate(gh.values):
    ax[0].text(i, v+0.003, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

ax[1].bar(rh.index, rh.values, color=[RCOL[r] for r in rh.index], width=0.62)
ax[1].axhline(overall, ls="--", color="#555", lw=1.3)
ax[1].text(len(rh)-0.5, overall+0.002, f"overall {overall:.3f}", ha="right", fontsize=9, color="#555")
ax[1].set_title("by inferred race / ethnicity"); ax[1].set_ylabel("Hit@10"); ax[1].set_ylim(0, 0.16)
plt.setp(ax[1].get_xticklabels(), rotation=12)
for i, v in enumerate(rh.values):
    ax[1].text(i, v+0.003, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figures/fairness_performance.png", dpi=160, bbox_inches="tight", facecolor="white")
print("wrote figures/fairness_performance.png")
print("overall:", round(overall, 4), "| gender:", dict(gh.round(4)), "| race:", dict(rh.round(4)))
