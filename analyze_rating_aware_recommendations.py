#!/usr/bin/env python3
"""Audit whether recovered test restaurants were liked, neutral, or disliked."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA = Path("data")
PREDICTIONS = DATA / "predictions"
RESULTS = Path("results")
SEEDS = (42, 43, 44)


def publication_test_targets() -> pd.DataFrame:
    columns = ["contributor_id", "place_id", "rating", "timestamp_days_ago"]
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
    ordered = frame.sort_values("timestamp_days_ago", ascending=True, na_position="last")
    test = ordered.groupby("contributor_id", sort=False).head(1).copy()
    test["user_idx"] = test.contributor_id.map(user_index)
    return test[["user_idx", "contributor_id", "place_id", "rating"]].rename(
        columns={"place_id": "true_place_id", "rating": "test_rating"}
    )


def rank_of(place_id: str, recommendations) -> float:
    values = list(recommendations)
    try:
        return float(values.index(place_id) + 1)
    except ValueError:
        return np.nan


def summarize(stage: str, seed: int, predictions: pd.DataFrame,
              targets: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    frame = predictions.merge(targets, on="user_idx", how="left", validate="one_to_one")
    if "true_place_id_x" in frame:
        mismatch = frame.true_place_id_x.ne(frame.true_place_id_y).sum()
        if mismatch:
            raise RuntimeError(f"{mismatch} target mismatches for {stage}, seed {seed}")
        frame["true_place_id"] = frame.true_place_id_x
    elif "true_place_id" not in frame:
        frame["true_place_id"] = frame.true_place_id
    frame["rank"] = [
        rank_of(place_id, recs)
        for place_id, recs in zip(frame.true_place_id, frame.top10_place_ids)
    ]
    frame["hit"] = frame["rank"].notna()
    frame["liked"] = frame.test_rating >= 4
    frame["neutral"] = frame.test_rating == 3
    frame["disliked"] = frame.test_rating <= 2
    discount = np.where(frame.hit, 1 / np.log2(frame["rank"].fillna(1) + 1), 0)
    frame["positive_dcg_at_10"] = discount * frame.liked
    frame["signed_utility_at_10"] = discount * np.select(
        [frame.liked, frame.disliked], [1.0, -1.0], default=0.0
    )
    hits = frame[frame.hit]
    summary = {
        "stage": stage,
        "seed": seed,
        "users": len(frame),
        "hit_at_10": frame.hit.mean(),
        "positive_hit_at_10": (frame.hit & frame.liked).mean(),
        "neutral_hit_at_10": (frame.hit & frame.neutral).mean(),
        "disliked_hit_at_10": (frame.hit & frame.disliked).mean(),
        "positive_dcg_at_10": frame.positive_dcg_at_10.mean(),
        "signed_utility_at_10": frame.signed_utility_at_10.mean(),
        "liked_share_all_targets": frame.liked.mean(),
        "liked_share_among_hits": hits.liked.mean(),
        "disliked_share_all_targets": frame.disliked.mean(),
        "disliked_share_among_hits": hits.disliked.mean(),
        "mean_rating_all_targets": frame.test_rating.mean(),
        "mean_rating_among_hits": hits.test_rating.mean(),
    }
    by_rating = frame.groupby("test_rating", as_index=False).agg(
        targets=("hit", "size"), hits=("hit", "sum"), hit_at_10=("hit", "mean")
    )
    by_rating.insert(0, "seed", seed)
    by_rating.insert(0, "stage", stage)
    return summary, by_rating


def main() -> None:
    targets = publication_test_targets()
    summaries, rating_rows = [], []
    for seed in SEEDS:
        raw = pd.read_parquet(
            PREDICTIONS / f"kgat_sal_all_T3_pubsplit_seed{seed}_predictions.parquet"
        )
        summary, by_rating = summarize("Raw KGAT-SAL + LLM", seed, raw, targets)
        summaries.append(summary)
        rating_rows.append(by_rating)

        proximity = pd.read_parquet(
            PREDICTIONS / f"kgat_sal_full_llm_pubsplit_seed{seed}_proximity_predictions.parquet"
        )
        summary, by_rating = summarize("+ proximity reranking", seed, proximity, targets)
        summaries.append(summary)
        rating_rows.append(by_rating)

    detail = pd.DataFrame(summaries)
    by_rating = pd.concat(rating_rows, ignore_index=True)
    numeric = [column for column in detail if column not in {"stage", "seed", "users"}]
    aggregate = detail.groupby("stage", sort=False)[numeric].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index()
    RESULTS.mkdir(exist_ok=True)
    detail.to_csv(RESULTS / "publication_rating_aware_by_seed.csv", index=False)
    aggregate.to_csv(RESULTS / "publication_rating_aware_summary.csv", index=False)
    by_rating.to_csv(RESULTS / "publication_hit_rate_by_rating.csv", index=False)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
