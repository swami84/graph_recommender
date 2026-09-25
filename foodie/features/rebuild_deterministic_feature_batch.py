#!/usr/bin/env python3
"""Build a deterministic expanded split and source-verified feature worklist.

The original article artifacts and the first September feature batch are read
only. This creates a versioned successor with explicit reuse/generation lists.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from foodie.features import build_llm_features_ollama as llm


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PREVIOUS = DATA / "feature_batches" / "places_expansion_2026-09-16"
OUTPUT = DATA / "feature_batches" / "places_expansion_2026-09-16_split_v2"
PROMPT_VERSION = "foodie-incremental-v7-deterministic-20260920"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partition(
    entity: str, corpus: pd.DataFrame, id_col: str,
    old_feature_path: Path, recent_feature_path: Path,
) -> dict[str, int]:
    """Reuse a feature only when its exact ordered source-review IDs match."""
    source = corpus[[id_col, "source_review_ids_json"]].copy()
    source[id_col] = source[id_col].astype(str)
    source = source.rename(columns={"source_review_ids_json": "current_source_ids"})
    old = pd.read_parquet(old_feature_path, columns=[id_col, "source_review_ids_json"])
    old[id_col] = old[id_col].astype(str)
    old = old.drop_duplicates(id_col, keep="last").rename(
        columns={"source_review_ids_json": "old_source_ids"}
    )
    recent = pd.read_parquet(recent_feature_path, columns=[id_col, "source_review_ids_json"])
    recent[id_col] = recent[id_col].astype(str)
    recent = recent.drop_duplicates(id_col, keep="last").rename(
        columns={"source_review_ids_json": "recent_source_ids"}
    )
    comparison = source.merge(old, on=id_col, how="left").merge(recent, on=id_col, how="left")
    recent_match = comparison["current_source_ids"].eq(comparison["recent_source_ids"])
    old_match = comparison["current_source_ids"].eq(comparison["old_source_ids"])
    comparison["feature_action"] = "generate"
    comparison.loc[old_match, "feature_action"] = "reuse_article"
    comparison.loc[recent_match, "feature_action"] = "reuse_september_v1"
    comparison[[id_col, "feature_action"]].to_parquet(
        OUTPUT / f"{entity}_source_actions.parquet", index=False
    )
    targets = corpus[
        corpus[id_col].astype(str).isin(
            comparison.loc[comparison["feature_action"].eq("generate"), id_col]
        )
    ].copy()
    targets.to_parquet(OUTPUT / f"{entity}_targets.parquet", index=False)
    return {
        "corpora": len(corpus),
        "reuse_article": int(comparison["feature_action"].eq("reuse_article").sum()),
        "reuse_september_v1": int(comparison["feature_action"].eq("reuse_september_v1").sum()),
        "generate": len(targets),
        "previous_feature_rows_without_current_corpus": len(set(old[id_col]) - set(source[id_col])),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "worklist_manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"Refusing to overwrite prepared worklist: {manifest_path}")
    llm.REVIEWS_FILE = PREVIOUS / "reviews_flat.parquet"
    llm.RESTAURANTS_FILE = PREVIOUS / "restaurants_enriched.parquet"
    llm.CORPUS_DIR = OUTPUT / "llm_corpora"
    llm.RESTAURANT_CORPUS = llm.CORPUS_DIR / "restaurant_train_corpus.parquet"
    llm.USER_CORPUS = llm.CORPUS_DIR / "user_train_corpus.parquet"
    llm.SPLIT_MANIFEST = llm.CORPUS_DIR / "split_manifest.json"
    llm.PROMPT_VERSION = PROMPT_VERSION
    split_manifest = llm.prepare_corpora()
    if split_manifest.get("split_sort_version") != "deterministic-coarse-date-ties-v2":
        raise RuntimeError("Deterministic split version was not applied")

    counts = {
        "restaurant": partition(
            "restaurant", pd.read_parquet(llm.RESTAURANT_CORPUS), "place_id",
            DATA / "audited_llm_features" / "restaurant_llm_features.parquet",
            PREVIOUS / "restaurant_llm_features.parquet",
        ),
        "user": partition(
            "user", pd.read_parquet(llm.USER_CORPUS), "contributor_id",
            DATA / "audited_llm_features" / "user_llm_features.parquet",
            PREVIOUS / "user_llm_features.parquet",
        ),
    }
    manifest = {
        "batch_id": OUTPUT.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_batch": PREVIOUS.name,
        "split_sort_version": split_manifest["split_sort_version"],
        "prompt_version_for_generation": PROMPT_VERSION,
        "source_files": {
            "reviews": {"path": str(llm.REVIEWS_FILE), "sha256": sha256(llm.REVIEWS_FILE)},
            "restaurants": {"path": str(llm.RESTAURANTS_FILE), "sha256": sha256(llm.RESTAURANTS_FILE)},
        },
        "reuse_rule": "ordered source_review_ids_json must exactly match current deterministic corpus",
        "counts": counts,
        "outputs_are_isolated_from_article_batch": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (OUTPUT / "batch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
