#!/usr/bin/env python3
"""Evaluate the final KGAT-SAL + LLM + proximity ranker by user segment."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from article.figures.publication_style import (  # noqa: E402
    apply_style, clean_axis, save, title, NAVY, TEAL, CORAL, GOLD, PURPLE,
    MUTED, GRID,
)
from analysis.expanded_eda.make_publication_eda import audited_restaurants  # noqa: E402


DATA = ROOT / "data"
PREDICTIONS = DATA / "predictions"
OUT = ROOT / "analysis" / "segment_evaluation"
TABLES = OUT / "tables"
FIGURES = ROOT / "article" / "figures"
SEEDS = (42, 43, 44)

GENDER_LABELS = {"female": "Female", "male": "Male"}
RACE_LABELS = {
    "nh_white": "White",
    "api": "Asian",
    "nh_black": "Black",
    "hispanic": "Hispanic",
}


def test_targets() -> pd.DataFrame:
    columns = [
        "contributor_id", "place_id", "rating", "timestamp_days_ago",
        "predicted_gender", "predicted_race",
    ]
    frame = pd.read_parquet(DATA / "reviews_flat.parquet", columns=columns)
    frame = frame[
        frame.contributor_id.notna() & frame.contributor_id.ne("") &
        frame.place_id.notna() & frame.rating.notna()
    ].copy()
    frame = frame.sort_values("timestamp_days_ago", ascending=True, na_position="last")
    frame = frame.drop_duplicates(["contributor_id", "place_id"], keep="first")
    counts = frame.contributor_id.value_counts()
    frame = frame[frame.contributor_id.isin(counts[counts >= 4].index)].copy()
    users = sorted(frame.contributor_id.unique())
    user_index = {value: index for index, value in enumerate(users)}
    test = (
        frame.sort_values("timestamp_days_ago", ascending=True, na_position="last")
        .groupby("contributor_id", sort=False).head(1).copy()
    )
    test["user_idx"] = test.contributor_id.map(user_index)
    cuisines = audited_restaurants().set_index("place_id")["cuisine"]
    test["cuisine"] = test.place_id.map(cuisines)
    return test[[
        "user_idx", "contributor_id", "place_id", "rating",
        "predicted_gender", "predicted_race", "cuisine",
    ]]


def rank_of(target: str, recommendations) -> float:
    try:
        return float(list(recommendations).index(target) + 1)
    except ValueError:
        return np.nan


def scored_predictions(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        path = PREDICTIONS / f"kgat_sal_full_llm_pubsplit_seed{seed}_proximity_predictions.parquet"
        pred = pd.read_parquet(path, columns=["user_idx", "top10_place_ids"])
        frame = pred.merge(targets, on="user_idx", how="left", validate="one_to_one")
        frame["rank"] = [
            rank_of(target, recommendations)
            for target, recommendations in zip(frame.place_id, frame.top10_place_ids)
        ]
        frame["hit_at_10"] = frame["rank"].notna().astype(float)
        frame["ndcg_at_10"] = np.where(
            frame["rank"].notna(), 1 / np.log2(frame["rank"].fillna(1) + 1), 0.0
        )
        frame["seed"] = seed
        rows.append(frame.drop(columns="top10_place_ids"))
    return pd.concat(rows, ignore_index=True)


def aggregate(frame: pd.DataFrame, column: str, labels: dict | None = None,
              order: list[str] | None = None) -> pd.DataFrame:
    subset = frame[frame[column].notna()].copy()
    if labels is not None:
        subset = subset[subset[column].isin(labels)].copy()
        subset["segment"] = subset[column].map(labels)
    else:
        subset["segment"] = subset[column].astype(str)
    by_seed = subset.groupby(["seed", "segment"], as_index=False).agg(
        users=("user_idx", "size"),
        hit_at_10=("hit_at_10", "mean"),
        ndcg_at_10=("ndcg_at_10", "mean"),
    )
    result = by_seed.groupby("segment", as_index=False).agg(
        users=("users", "first"),
        hit_at_10=("hit_at_10", "mean"),
        hit_std=("hit_at_10", "std"),
        ndcg_at_10=("ndcg_at_10", "mean"),
        ndcg_std=("ndcg_at_10", "std"),
    )
    if order:
        result["sort"] = result.segment.map({value: i for i, value in enumerate(order)})
        result = result.sort_values("sort").drop(columns="sort")
    return result


def overall(frame: pd.DataFrame) -> dict[str, float]:
    per_seed = frame.groupby("seed").agg(
        hit_at_10=("hit_at_10", "mean"), ndcg_at_10=("ndcg_at_10", "mean")
    )
    return per_seed.mean().to_dict()


def bars(axis, table: pd.DataFrame, metric: str, error: str, color: str,
         overall_value: float, horizontal: bool = False) -> None:
    labels = [f"{row.segment}\n(n={row.users:,})" for row in table.itertuples()]
    values = table[metric].to_numpy()
    errors = table[error].fillna(0).to_numpy()
    if horizontal:
        positions = np.arange(len(table))
        artists = axis.barh(positions, values, xerr=errors, color=color, height=0.68,
                            error_kw={"ecolor": MUTED, "capsize": 2, "lw": 1})
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.axvline(overall_value, color=MUTED, linestyle="--", linewidth=1.4,
                     label=f"Overall {overall_value:.3f}")
        axis.bar_label(artists, labels=[f"{value:.3f}" for value in values],
                       padding=4, fontsize=8.5)
        axis.set_xlim(0, max(values.max(), overall_value) * 1.22)
        clean_axis(axis, "x")
    else:
        positions = np.arange(len(table))
        artists = axis.bar(positions, values, yerr=errors, color=color, width=0.66,
                           error_kw={"ecolor": MUTED, "capsize": 2, "lw": 1})
        axis.set_xticks(positions, labels, rotation=12, ha="right")
        axis.axhline(overall_value, color=MUTED, linestyle="--", linewidth=1.4,
                     label=f"Overall {overall_value:.3f}")
        axis.bar_label(artists, labels=[f"{value:.3f}" for value in values],
                       padding=4, fontsize=8.5)
        axis.set_ylim(0, max(values.max(), overall_value) * 1.24)
        clean_axis(axis)
    axis.legend(loc="upper right", fontsize=8.5)


def main() -> None:
    apply_style()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    scored = scored_predictions(test_targets())
    totals = overall(scored)

    gender = aggregate(scored, "predicted_gender", GENDER_LABELS, list(GENDER_LABELS.values()))
    race = aggregate(scored, "predicted_race", RACE_LABELS, list(RACE_LABELS.values()))
    gender.to_csv(TABLES / "kgat_sal_proximity_by_gender.csv", index=False)
    race.to_csv(TABLES / "kgat_sal_proximity_by_race.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.1),
                             gridspec_kw={"hspace": 0.52, "wspace": 0.22})
    bars(axes[0, 0], gender, "hit_at_10", "hit_std", CORAL, totals["hit_at_10"])
    axes[0, 0].set_title("Hit@10 by inferred gender")
    axes[0, 0].set_ylabel("Hit@10")
    bars(axes[0, 1], gender, "ndcg_at_10", "ndcg_std", NAVY, totals["ndcg_at_10"])
    axes[0, 1].set_title("NDCG@10 by inferred gender")
    axes[0, 1].set_ylabel("NDCG@10")
    bars(axes[1, 0], race, "hit_at_10", "hit_std", TEAL, totals["hit_at_10"])
    axes[1, 0].set_title("Hit@10 by inferred race/ethnicity")
    axes[1, 0].set_ylabel("Hit@10")
    axes[1, 0].set_xlabel("Inferred race/ethnicity")
    axes[1, 0].tick_params(axis="x", rotation=0)
    bars(axes[1, 1], race, "ndcg_at_10", "ndcg_std", PURPLE, totals["ndcg_at_10"])
    axes[1, 1].set_title("NDCG@10 by inferred race/ethnicity")
    axes[1, 1].set_ylabel("NDCG@10")
    axes[1, 1].set_xlabel("Inferred race/ethnicity")
    axes[1, 1].tick_params(axis="x", rotation=0)
    title(fig, "Recommendation performance across inferred demographic groups",
          "KGAT-SAL + LLM features + proximity; mean over three seeds; error bars show ±1 SD")
    save(fig, FIGURES / "fairness_performance.png")

    cuisine_counts = scored[scored.cuisine.notna()].drop_duplicates("user_idx").cuisine.value_counts()
    top_cuisines = cuisine_counts.head(14).index.tolist()
    cuisine = aggregate(scored[scored.cuisine.isin(top_cuisines)], "cuisine")
    cuisine = cuisine.sort_values("hit_at_10", ascending=False).reset_index(drop=True)
    cuisine.to_csv(TABLES / "kgat_sal_proximity_by_target_cuisine.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.7), gridspec_kw={"wspace": 0.44})
    bars(axes[0], cuisine, "hit_at_10", "hit_std", TEAL, totals["hit_at_10"], horizontal=True)
    axes[0].set_title("Hit@10 by held-out restaurant cuisine")
    axes[0].set_xlabel("Hit@10")
    bars(axes[1], cuisine, "ndcg_at_10", "ndcg_std", NAVY, totals["ndcg_at_10"], horizontal=True)
    axes[1].set_title("NDCG@10 by held-out restaurant cuisine")
    axes[1].set_xlabel("NDCG@10")
    title(fig, "Recommendation performance by held-out restaurant cuisine",
          "Fourteen most frequent supported target cuisines; mean over three seeds; error bars show ±1 SD")
    save(fig, FIGURES / "segment_performance.png")

    print(f"Overall Hit@10={totals['hit_at_10']:.5f}; NDCG@10={totals['ndcg_at_10']:.5f}")
    print(f"Wrote {FIGURES / 'fairness_performance.png'}")
    print(f"Wrote {FIGURES / 'segment_performance.png'}")


if __name__ == "__main__":
    main()
