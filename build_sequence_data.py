#!/usr/bin/env python3
"""
build_sequence_data.py — Build ordered user interaction sequences for SeqHybrid.

For each user in our training set, reconstructs the chronological order of
their restaurant visits using timestamp_days_ago from reviews_flat.parquet.
Items not in the filtered item set are dropped; users with fewer than 2
interactions after filtering are also dropped (no sequence signal).

Output: data/user_sequences.parquet
  Columns:
    user_idx     int32          index into model's user embedding table
    seq          list[int32]    item_idx values, oldest → newest
    seq_len      int16          actual sequence length (before padding)

Usage:
    python build_sequence_data.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from recommendation_gnn import load_interactions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_seq")

OUT = Path("data/user_sequences.parquet")


def build(min_seq_len: int = 2) -> None:
    # ── Load interactions + encoders ───────────────────────────────────────────
    log.info("Loading interactions…")
    train_df, test_df, user_enc, item_enc = load_interactions()
    n_users = len(user_enc)
    n_items = len(item_enc)
    log.info(f"  {len(train_df):,} train interactions | {n_users:,} users | {n_items:,} items")

    # train_df already has contributor_id, place_id, timestamp_days_ago, user_idx, item_idx
    needed = ["contributor_id", "place_id", "timestamp_days_ago", "user_idx", "item_idx"]
    inter  = train_df[needed].copy()

    # ── Sort each user's interactions chronologically ──────────────────────────
    # timestamp_days_ago: larger = older, so sort descending → oldest first
    inter.sort_values(["user_idx", "timestamp_days_ago"], ascending=[True, False], inplace=True)

    # ── Build sequence per user ────────────────────────────────────────────────
    log.info("Building sequences…")
    grouped = inter.groupby("user_idx")["item_idx"].apply(list)

    records = []
    for user_idx, seq in grouped.items():
        if len(seq) < min_seq_len:
            continue
        records.append({
            "user_idx": int(user_idx),
            "seq":      [int(x) for x in seq],
            "seq_len":  len(seq),
        })

    seq_df = pd.DataFrame(records)
    seq_df["user_idx"] = seq_df["user_idx"].astype("int32")
    seq_df["seq_len"]  = seq_df["seq_len"].astype("int16")

    log.info(f"  Sequences built: {len(seq_df):,} users with >= {min_seq_len} interactions")
    log.info(f"  Sequence length — mean: {seq_df['seq_len'].mean():.2f}  "
             f"median: {seq_df['seq_len'].median():.0f}  "
             f"max: {seq_df['seq_len'].max()}  "
             f"p95: {seq_df['seq_len'].quantile(0.95):.0f}")

    # Users in user_enc but not in seq_df will fall back to zero-padded sequences
    # (handled in the model by masking)
    missing = n_users - len(seq_df)
    if missing > 0:
        log.info(f"  {missing:,} users have < {min_seq_len} interactions — "
                 f"will use zero-padded sequences at inference")

    OUT.parent.mkdir(exist_ok=True)
    seq_df.to_parquet(OUT, index=False)
    log.info(f"Saved → {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
