#!/usr/bin/env python3
"""
recommendation_baseline.py — Phase 3: ALS collaborative filtering baseline.

Uses the `implicit` library (Alternating Least Squares) on the reviewer ×
restaurant interaction matrix (ratings treated as confidence weights).

Metrics: Precision@10, Recall@10, NDCG@10 (leave-one-out split)

Usage:
    python recommendation_baseline.py               # train + evaluate
    python recommendation_baseline.py --recommend CONTRIBUTOR_ID
    python recommendation_baseline.py --factors 128 --iterations 50
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("baseline")

REVIEWS_FLAT    = Path("data/reviews_flat.parquet")
RESTAURANTS_ENR = Path("data/restaurants_enriched.parquet")
MODEL_DIR       = Path("models")
MODEL_FILE      = MODEL_DIR / "als_baseline.pkl"
ENCODERS_FILE   = MODEL_DIR / "als_encoders.pkl"


# ── Metrics ────────────────────────────────────────────────────────────────────
def ndcg_at_k(relevant: set, ranked: list, k: int) -> float:
    dcg   = sum(1.0 / np.log2(i + 2) for i, x in enumerate(ranked[:k]) if x in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def precision_at_k(relevant: set, ranked: list, k: int) -> float:
    return sum(1 for x in ranked[:k] if x in relevant) / k


def recall_at_k(relevant: set, ranked: list, k: int) -> float:
    hits = sum(1 for x in ranked[:k] if x in relevant)
    return hits / len(relevant) if relevant else 0.0


# ── Data loading ───────────────────────────────────────────────────────────────
def load_interactions(min_reviews_per_user: int = 3):
    df = pd.read_parquet(REVIEWS_FLAT)
    df = df[
        df["contributor_id"].notna() & (df["contributor_id"] != "") &
        df["place_id"].notna() & df["rating"].notna()
    ].copy()

    # Deduplicate: keep one row per (user, restaurant) — most recent scrape capture
    df = df.sort_values("timestamp_days_ago", ascending=True, na_position="last")
    df = df.drop_duplicates(subset=["contributor_id", "place_id"], keep="first")

    counts = df["contributor_id"].value_counts()
    df = df[df["contributor_id"].isin(counts[counts >= min_reviews_per_user].index)].copy()

    user_ids = sorted(df["contributor_id"].unique())
    item_ids = sorted(df["place_id"].unique())
    user_enc = {u: i for i, u in enumerate(user_ids)}
    item_enc = {it: i for i, it in enumerate(item_ids)}

    df["user_idx"] = df["contributor_id"].map(user_enc)
    df["item_idx"] = df["place_id"].map(item_enc)

    log.info(f"Interactions: {len(df):,} | users: {len(user_enc):,} | items: {len(item_enc):,}")
    return df, user_enc, item_enc


# ── Train + evaluate ───────────────────────────────────────────────────────────
def train_and_evaluate(factors: int = 64, iterations: int = 30,
                       regularization: float = 0.01, save: bool = True):
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError:
        log.error("implicit not installed. Run: pip install implicit")
        return

    df, user_enc, item_enc = load_interactions()
    n_users, n_items = len(user_enc), len(item_enc)

    # ascending sort → smallest days_ago (most recent) first; head(1) = most recent review
    df_sorted = df.sort_values("timestamp_days_ago", ascending=True, na_position="last")
    test_idx  = df_sorted.groupby("contributor_id").head(1).index
    train_df  = df.drop(index=test_idx)
    test_df   = df.loc[test_idx]
    log.info(f"Train: {len(train_df):,}  Test: {len(test_df):,}")

    # implicit >=0.7 fit() expects user-item (users × items)
    alpha = 40.0
    vals  = (1.0 + alpha * train_df["rating"].values).astype(np.float32)
    user_item_train = sp.csr_matrix(
        (vals, (train_df["user_idx"].values, train_df["item_idx"].values)),
        shape=(n_users, n_items),
    )

    model = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        use_gpu=False,
        random_state=42,
    )
    log.info(f"Training ALS (factors={factors}, iterations={iterations})…")
    model.fit(user_item_train)

    # ── Evaluate ────────────────────────────────────────────────────────────────
    K = 10
    ndcg_scores, prec_scores, rec_scores = [], [], []
    for user_idx_val, group in test_df.groupby("user_idx"):
        relevant = set(group["item_idx"].values)
        recs = model.recommend(
            user_idx_val, user_item_train[user_idx_val],
            N=K, filter_already_liked_items=True,
        )
        ranked = [int(r[0]) for r in recs]
        ndcg_scores.append(ndcg_at_k(relevant, ranked, K))
        prec_scores.append(precision_at_k(relevant, ranked, K))
        rec_scores.append(recall_at_k(relevant, ranked, K))

    print("\n=== ALS Baseline Results ===")
    print(f"  Precision@{K}: {np.mean(prec_scores):.4f}")
    print(f"  Recall@{K}:    {np.mean(rec_scores):.4f}")
    print(f"  NDCG@{K}:      {np.mean(ndcg_scores):.4f}")
    print()

    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        with open(ENCODERS_FILE, "wb") as f:
            pickle.dump({
                "user_enc":       user_enc,
                "item_enc":       item_enc,
                "user_dec":       {v: k for k, v in user_enc.items()},
                "item_dec":       {v: k for k, v in item_enc.items()},
                "user_item_mat":  user_item_train,
            }, f)
        log.info(f"Model → {MODEL_FILE}")

    return model, user_enc, item_enc


# ── Inference ──────────────────────────────────────────────────────────────────
def recommend(contributor_id: str, top_n: int = 10) -> list[dict]:
    if not MODEL_FILE.exists():
        log.error(f"No model at {MODEL_FILE}. Run training first.")
        return []

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_FILE, "rb") as f:
        enc = pickle.load(f)

    user_enc      = enc["user_enc"]
    item_dec      = enc["item_dec"]
    user_item_mat = enc["user_item_mat"]

    if contributor_id not in user_enc:
        log.warning(f"Unknown user: {contributor_id}")
        return []

    user_idx = user_enc[contributor_id]
    recs = model.recommend(user_idx, user_item_mat[user_idx], N=top_n,
                           filter_already_liked_items=True)

    restaurants = pd.read_parquet(RESTAURANTS_ENR) if RESTAURANTS_ENR.exists() else None
    rest_map = restaurants.set_index("place_id").to_dict("index") if restaurants is not None else {}

    return [
        {
            "place_id": item_dec[int(idx)],
            "name":     rest_map.get(item_dec[int(idx)], {}).get("name", ""),
            "cuisine":  rest_map.get(item_dec[int(idx)], {}).get("cuisine_category", ""),
            "score":    float(score),
        }
        for idx, score in recs
    ]


def main():
    parser = argparse.ArgumentParser(description="ALS recommendation baseline")
    parser.add_argument("--factors",    type=int, default=64)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--recommend",  metavar="CONTRIBUTOR_ID")
    parser.add_argument("--no-save",    action="store_true")
    args = parser.parse_args()

    if args.recommend:
        recs = recommend(args.recommend)
        print(f"\nTop recommendations for {args.recommend}:")
        for i, r in enumerate(recs, 1):
            print(f"  {i:2}. {r['name']} ({r['cuisine']}) — score {r['score']:.3f}")
    else:
        train_and_evaluate(factors=args.factors, iterations=args.iterations,
                           save=not args.no_save)


if __name__ == "__main__":
    main()
