#!/usr/bin/env python3
"""Extend the historical RaceBERT + gender_guesser cache to all current names.

This is a resumable extraction of the exact algorithm used by the archived
demographics notebook. Existing predictions are preserved byte-for-byte; only
new display names are classified. The canonical review parquet is replaced
atomically only after name coverage is complete.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import gender_guesser.detector as gender_lib
import pandas as pd
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

DATA = Path("data")
REVIEWS = DATA / "reviews_flat.parquet"
CACHE = DATA / "name_predictions.parquet"
PARTS = DATA / "demographics_checkpoints"
ARCHIVE = Path("archive/demographics_pre_expansion_20260909")
MODEL = "pparasurama/raceBERT"
COLS = ["reviewer_name", "predicted_race", "race_score", "predicted_gender",
        "race_HL", "race_API", "race_BLACK", "race_WHITE"]


def predict_gender(detector: gender_lib.Detector, full_name: str) -> str:
    first = full_name.strip().split()[0]
    result = detector.get_gender(first)
    if result in ("male", "mostly_male"):
        return "male"
    if result in ("female", "mostly_female"):
        return "female"
    return "unknown"


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def load_done_names(existing: pd.DataFrame) -> set[str]:
    done = set(existing["reviewer_name"].astype(str))
    if PARTS.exists():
        for path in sorted(PARTS.glob("part_*.parquet")):
            done.update(pd.read_parquet(path, columns=["reviewer_name"])["reviewer_name"].astype(str))
    return done


def apply_to_reviews(predictions: pd.DataFrame) -> None:
    pred = pl.from_pandas(predictions[COLS])
    reviews = pl.read_parquet(REVIEWS)
    demographic_cols = [c for c in COLS if c != "reviewer_name" and c in reviews.columns]
    reviews = reviews.drop(demographic_cols).join(pred, on="reviewer_name", how="left")
    if reviews["predicted_gender"].null_count() or reviews["predicted_race"].null_count():
        raise ValueError("Demographic join left null predictions")
    temporary = REVIEWS.with_name("." + REVIEWS.name + ".demographics.tmp")
    reviews.write_parquet(temporary)
    os.replace(temporary, REVIEWS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--checkpoint-size", type=int, default=50_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-apply", action="store_true")
    args = parser.parse_args()

    names = (pl.scan_parquet(REVIEWS).select("reviewer_name").drop_nulls().unique()
             .collect()["reviewer_name"].to_list())
    existing = pd.read_parquet(CACHE)
    done = load_done_names(existing)
    todo = [str(name) for name in names if str(name) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"Names: {len(names):,} current, {len(done):,} cached/checkpointed, "
          f"{len(todo):,} to process", flush=True)

    if todo:
        PARTS.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, local_files_only=True)
        classifier = pipeline(
            "text-classification", model=model, tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1, top_k=None,
        )
        detector = gender_lib.Detector(case_sensitive=False)
        started = time.monotonic()
        part_index = len(list(PARTS.glob("part_*.parquet")))
        for start in range(0, len(todo), args.checkpoint_size):
            chunk = todo[start:start + args.checkpoint_size]
            results = classifier(
                chunk, batch_size=args.batch_size, truncation=True, max_length=64
            )
            rows = []
            for name, predictions in zip(chunk, results):
                scores = {p["label"]: float(p["score"]) for p in predictions}
                top = max(predictions, key=lambda value: value["score"])
                rows.append({
                    "reviewer_name": name, "predicted_race": top["label"],
                    "race_score": float(top["score"]),
                    "predicted_gender": predict_gender(detector, name),
                    # Historical columns were accidentally all zero. Preserve
                    # that schema/behavior; the recommender never consumes them.
                    "race_HL": 0.0, "race_API": 0.0,
                    "race_BLACK": 0.0, "race_WHITE": 0.0,
                })
            path = PARTS / f"part_{part_index:05d}.parquet"
            atomic_parquet(pd.DataFrame(rows, columns=COLS), path)
            part_index += 1
            finished = min(start + len(chunk), len(todo))
            rate = finished * 60 / max(time.monotonic() - started, 1e-6)
            print(f"[{finished:,}/{len(todo):,}] {rate:,.0f} names/min", flush=True)

    # A limited benchmark must not publish a partial canonical file.
    if args.limit:
        return
    parts = [pd.read_parquet(path) for path in sorted(PARTS.glob("part_*.parquet"))]
    complete = pd.concat([existing, *parts], ignore_index=True).drop_duplicates(
        "reviewer_name", keep="first"
    )
    missing = set(map(str, names)) - set(complete["reviewer_name"].astype(str))
    if missing:
        raise ValueError(f"Prediction cache incomplete: {len(missing):,} names missing")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    backup = ARCHIVE / CACHE.name
    if not backup.exists():
        shutil.copy2(CACHE, backup)
    atomic_parquet(complete[COLS], CACHE)
    if not args.no_apply:
        apply_to_reviews(complete)
    print(f"Complete: {len(complete):,} names; canonical reviews updated={not args.no_apply}")


if __name__ == "__main__":
    main()
