#!/usr/bin/env python3
"""Prepare isolated, leakage-safe LLM corpora for the September Places expansion.

The article-era canonical tables and feature files are read-only inputs. All
augmented canonical data, corpora, target IDs, and provenance live below the
batch directory. The main extractor is reused with paths redirected there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import build_graph_data as graph_data
import build_llm_features_ollama as llm


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEFAULT_BATCH = DATA / "feature_batches" / "places_expansion_2026-09-16"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def new_review_rows(ids: set[str]) -> pd.DataFrame:
    rows = []
    for place_id in sorted(ids):
        path = DATA / "reviews" / f"{place_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No review checkpoint for new restaurant {place_id}")
        payload = json.loads(path.read_text())
        for review in payload.get("reviews", []):
            meta = review.get("meta") or {}
            review_text = (review.get("text") or "").strip()
            rows.append({
                "review_id": review.get("review_id", ""),
                "place_id": place_id,
                "contributor_id": review.get("contributor_id", ""),
                "reviewer_name": (review.get("reviewer_name") or "").strip(),
                "is_local_guide": bool(review.get("is_local_guide", False)),
                "reviewer_reviews": review.get("reviewer_reviews"),
                "reviewer_photos": review.get("reviewer_photos"),
                "rating": review.get("rating"),
                "timestamp": review.get("timestamp", ""),
                "timestamp_days_ago": graph_data._parse_days_ago(review.get("timestamp", "")),
                "text": review_text,
                "text_len": len(review_text),
                "has_content": len(review_text) >= 50,
                "attached_photos": review.get("attached_photos", 0),
                "meal_type": meta.get("Meal type"),
                "price_per_person": graph_data._parse_price(meta.get("Price per person")),
                "food_score": meta.get("Food"),
                "service_score": meta.get("Service"),
                "atmosphere_score": meta.get("Atmosphere"),
                "recommended_dishes": meta.get("Recommended dishes", ""),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--membership-only", action="store_true",
                        help="Materialize batch ID lists from already prepared isolated tables")
    args = parser.parse_args()
    batch = args.batch_dir.resolve()
    batch.mkdir(parents=True, exist_ok=True)
    if args.membership_only:
        prior_rest_ids = set(pd.read_parquet(DATA / "restaurants_enriched.parquet", columns=["place_id"])["place_id"].astype(str))
        prior_user_ids = set(pd.read_parquet(DATA / "reviews_flat.parquet", columns=["contributor_id"])["contributor_id"].dropna().astype(str)) - {""}
        current_rest_ids = set(pd.read_parquet(batch / "restaurants_enriched.parquet", columns=["place_id"])["place_id"].astype(str))
        current_user_ids = set(pd.read_parquet(batch / "reviews_flat.parquet", columns=["contributor_id"])["contributor_id"].dropna().astype(str)) - {""}
        pd.DataFrame({"place_id": sorted(current_rest_ids - prior_rest_ids)}).to_parquet(
            batch / "restaurant_batch_ids.parquet", index=False)
        pd.DataFrame({"contributor_id": sorted(current_user_ids - prior_user_ids)}).to_parquet(
            batch / "reviewer_batch_ids.parquet", index=False)
        print("Wrote isolated restaurant and reviewer membership lists", flush=True)
        return
    if (batch / "batch_manifest.json").exists():
        raise SystemExit(f"Batch is already prepared; refusing to overwrite {batch}")

    prior_rest_path = DATA / "restaurants_enriched.parquet"
    prior_review_path = DATA / "reviews_flat.parquet"
    prior_rest = pd.read_parquet(prior_rest_path)
    prior_reviews = pd.read_parquet(prior_review_path)
    prior_rest_ids = set(prior_rest["place_id"].astype(str))
    prior_user_ids = set(prior_reviews["contributor_id"].dropna().astype(str)) - {""}
    index = pd.read_csv(DATA / "hex_restaurants_index.csv", dtype={"cbg": str})
    index = index.drop_duplicates("place_id", keep="first")
    new_index = index[~index["place_id"].astype(str).isin(prior_rest_ids)].copy()
    new_index["cuisine_category"] = new_index["types"].apply(graph_data.classify_cuisine)
    new_rest = new_index[prior_rest.columns].copy()
    new_reviews = new_review_rows(set(new_rest["place_id"].astype(str)))
    if not new_reviews.empty:
        new_reviews = new_reviews[new_reviews["reviewer_name"].ne("")].copy()
        new_reviews = new_reviews.loc[
            ~new_reviews["review_id"].ne("")
            | ~new_reviews.duplicated("review_id", keep="first")
        ].copy()
        name_pred = DATA / "name_predictions.parquet"
        if name_pred.exists():
            preds = pd.read_parquet(name_pred)[
                ["reviewer_name", "predicted_race", "race_score", "predicted_gender",
                 "race_HL", "race_API", "race_BLACK", "race_WHITE"]
            ].drop_duplicates("reviewer_name")
            new_reviews = new_reviews.merge(preds, on="reviewer_name", how="left")
    for col in prior_reviews:
        if col not in new_reviews:
            new_reviews[col] = None
    new_reviews = new_reviews[prior_reviews.columns]
    new_user_ids = set(new_reviews["contributor_id"].dropna().astype(str)) - prior_user_ids - {""}

    augmented_rest = pd.concat([prior_rest, new_rest], ignore_index=True)
    augmented_reviews = pd.concat([prior_reviews, new_reviews], ignore_index=True)
    augmented_rest.to_parquet(batch / "restaurants_enriched.parquet", index=False)
    augmented_reviews.to_parquet(batch / "reviews_flat.parquet", index=False)
    del augmented_rest, augmented_reviews, prior_reviews

    llm.REVIEWS_FILE = batch / "reviews_flat.parquet"
    llm.RESTAURANTS_FILE = batch / "restaurants_enriched.parquet"
    llm.CORPUS_DIR = batch / "llm_corpora"
    llm.RESTAURANT_CORPUS = llm.CORPUS_DIR / "restaurant_train_corpus.parquet"
    llm.USER_CORPUS = llm.CORPUS_DIR / "user_train_corpus.parquet"
    llm.SPLIT_MANIFEST = llm.CORPUS_DIR / "split_manifest.json"
    split_manifest = llm.prepare_corpora()

    restaurant_corpus = pd.read_parquet(llm.RESTAURANT_CORPUS)
    user_corpus = pd.read_parquet(llm.USER_CORPUS)
    restaurant_targets = restaurant_corpus[
        restaurant_corpus["place_id"].astype(str).isin(new_rest["place_id"].astype(str))
    ].copy()
    user_targets = user_corpus[
        user_corpus["contributor_id"].astype(str).isin(new_user_ids)
    ].copy()
    restaurant_targets.to_parquet(batch / "restaurant_targets.parquet", index=False)
    user_targets.to_parquet(batch / "user_targets.parquet", index=False)

    manifest = {
        "batch_id": batch.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "definition": "Restaurants absent from article-era canonical restaurant table and reviewers absent from article-era canonical review table",
        "prior_inputs": {
            "restaurants": {"path": str(prior_rest_path), "sha256": digest(prior_rest_path)},
            "reviews": {"path": str(prior_review_path), "sha256": digest(prior_review_path)},
            "article_restaurant_features": {"path": str(DATA / "audited_llm_features/restaurant_llm_features.parquet"), "sha256": digest(DATA / "audited_llm_features/restaurant_llm_features.parquet")},
            "article_user_features": {"path": str(DATA / "audited_llm_features/user_llm_features.parquet"), "sha256": digest(DATA / "audited_llm_features/user_llm_features.parquet")},
        },
        "counts": {
            "prior_restaurants": len(prior_rest),
            "new_restaurants": len(new_rest),
            "new_restaurant_reviews": len(new_reviews),
            "new_reviewer_ids": len(new_user_ids),
            "restaurant_llm_targets": len(restaurant_targets),
            "eligible_new_reviewer_llm_targets": len(user_targets),
        },
        "split_manifest": split_manifest,
        "outputs_are_isolated_from_article_batch": True,
    }
    (batch / "batch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
