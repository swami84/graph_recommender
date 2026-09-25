#!/usr/bin/env python3
"""Validate and freeze the publication feature contract with reproducible hashes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "data" / "publication_features"
JSON_OUT = FEATURE_DIR / "feature_contract_freeze.json"
MD_OUT = ROOT / "analysis" / "expanded_eda" / "PUBLICATION_FEATURE_CONTRACT.md"

INPUTS = [
    "data/restaurants_enriched.parquet",
    "data/reviews_flat.parquet",
    "data/restaurant_cuisine_27b_v4.parquet",
    "data/audited_llm_features/restaurant_llm_features.parquet",
    "data/audited_llm_features/user_llm_features.parquet",
    "data/restaurant_llm_embeddings.parquet",
    "data/user_llm_embeddings.parquet",
    "data/review_dishes.parquet",
    "data/dish_profiles.parquet",
    "data/llm_corpora/split_manifest.json",
]
OUTPUTS = [
    "data/publication_features/user_features_prebuilt.parquet",
    "data/publication_features/item_features_prebuilt.parquet",
    "data/publication_features/dishes_train.parquet",
    "data/publication_features/feature_groups.json",
]
CODE = [
    "update_reviewer_demographics.py",
    "classify_other_cuisines_27b.py",
    "refine_unknown_cuisines_v4.py",
    "audit_and_assemble_llm_features.py",
    "audit_legacy_feature_coverage.py",
    "build_extended_features.py",
    "build_training_features.py",
    "recommendation_kgat.py",
    "recommendation_kgat_sal.py",
    "run_focused_llm_experiments.py",
    "analysis/expanded_eda/run_eda.py",
    "freeze_publication_feature_contract.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(relative: str) -> dict:
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def validate_matrix(side: str, id_col: str, groups: dict[str, list[str]]) -> dict:
    path = FEATURE_DIR / f"{side}_features_prebuilt.parquet"
    frame = pd.read_parquet(path)
    feature_cols = [col for col in frame if col != id_col]
    grouped = [col for values in groups.values() for col in values]
    duplicates = sorted({col for col in grouped if grouped.count(col) > 1})
    missing_from_groups = sorted(set(feature_cols) - set(grouped))
    absent_from_matrix = sorted(set(grouped) - set(feature_cols))
    values = frame[feature_cols].to_numpy(dtype=np.float64)
    constants = [col for col in feature_cols if frame[col].nunique(dropna=False) <= 1]
    result = {
        "rows": len(frame), "unique_ids": int(frame[id_col].nunique()),
        "features": len(feature_cols), "group_counts": {k: len(v) for k, v in groups.items()},
        "null_values": int(np.isnan(values).sum()), "infinite_values": int(np.isinf(values).sum()),
        "constant_columns": constants, "duplicate_group_columns": duplicates,
        "missing_from_groups": missing_from_groups, "absent_from_matrix": absent_from_matrix,
    }
    failures = [
        result["rows"] != result["unique_ids"], result["null_values"] > 0,
        result["infinite_values"] > 0, bool(constants), bool(duplicates),
        bool(missing_from_groups), bool(absent_from_matrix),
    ]
    if any(failures):
        raise ValueError(f"Invalid {side} publication matrix: {result}")
    return result


def main() -> None:
    groups = json.loads((FEATURE_DIR / "feature_groups.json").read_text())
    expected_groups = {
        "user": {"base", "extended", "llm", "llm_embedding"},
        "item": {"base", "extended", "dish_llm", "llm", "llm_embedding"},
    }
    for side, expected in expected_groups.items():
        if set(groups[side]) != expected:
            raise ValueError(f"Unexpected {side} feature groups: {set(groups[side])}")

    matrix = {
        "user": validate_matrix("user", "contributor_id", groups["user"]),
        "item": validate_matrix("item", "place_id", groups["item"]),
    }
    cuisine = pl.read_parquet(ROOT / "data/restaurant_cuisine_27b_v4.parquet")
    if cuisine.height != 15_747 or cuisine["place_id"].n_unique() != cuisine.height:
        raise ValueError("Cuisine v4 target coverage is incomplete or duplicated")
    cuisine_norm = cuisine["primary_cuisine"].cast(pl.Utf8).str.to_lowercase()
    if int((cuisine_norm == "other").sum()) != 0:
        raise ValueError("Cuisine v4 still contains redundant 'other' labels")

    demographics = pl.scan_parquet(ROOT / "data/reviews_flat.parquet").select(
        pl.col("predicted_gender").is_null().sum().alias("gender_null"),
        pl.col("predicted_race").is_null().sum().alias("race_null"),
    ).collect().row(0, named=True)
    if demographics["gender_null"] or demographics["race_null"]:
        raise ValueError(f"Incomplete demographic join: {demographics}")

    legacy = pd.read_csv(ROOT / "analysis/expanded_eda/legacy_feature_coverage.csv")
    unaccounted = legacy[legacy["status"].isin(["unaccounted", "pending_rebuild"])]
    if len(unaccounted):
        raise ValueError("Legacy features remain unaccounted")

    payload = {
        "schema_version": "publication-feature-contract-v2",
        "frozen_at": datetime.now().astimezone().isoformat(),
        "feature_conditions": {
            "non_llm": {"skip_groups": ["llm", "llm_embedding", "dish_llm"]},
            "full_llm": {"skip_groups": []},
        },
        "matrix_validation": matrix,
        "cuisine": {
            "target_rows": cuisine.height,
            "unknown_rows": int((cuisine_norm == "unknown").sum()),
            "named_rows": int((cuisine_norm != "unknown").sum()),
            "other_rows": 0,
        },
        "demographics": demographics,
        "legacy_feature_audit": {
            "entries": len(legacy), "unaccounted": 0,
            "status_counts": legacy["status"].value_counts().to_dict(),
        },
        "inputs": [file_record(path) for path in INPUTS],
        "outputs": [file_record(path) for path in OUTPUTS],
        "code": [file_record(path) for path in CODE],
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    MD_OUT.write_text(f"""# Frozen publication feature contract

Frozen at `{payload['frozen_at']}` under schema `{payload['schema_version']}`.

- Users: **{matrix['user']['rows']:,}** rows × **{matrix['user']['features']}** features.
- Restaurants: **{matrix['item']['rows']:,}** rows × **{matrix['item']['features']}** features.
- Cuisine repair: **{payload['cuisine']['named_rows']:,}** named and
  **{payload['cuisine']['unknown_rows']:,}** unknown among 15,747 original Google `Other` rows.
- Demographic join: zero missing gender and race predictions in the canonical review table.
- Legacy audit: all **{len(legacy)}** entries retained, superseded, or excluded with justification;
  zero unaccounted.
- Matrix checks: unique identifiers, no nulls, no infinities, no constant columns, and exact,
  non-overlapping feature-group coverage.

## Experimental conditions

- `non_llm`: excludes `llm`, `llm_embedding`, and `dish_llm` on both applicable sides;
  KG dish relations are disabled.
- `full_llm`: includes structured LLM features, LLM embeddings, training-only dish features,
  and training-only KG dish relations.
- Both conditions use identical publication train/validation/test interactions and spatial CBG
  edges. Proximity re-ranking is reserved for the validation-selected winner.

The machine-readable manifest with SHA-256 hashes is
`data/publication_features/feature_contract_freeze.json`.
""")
    print(json.dumps({
        "json": str(JSON_OUT.relative_to(ROOT)), "report": str(MD_OUT.relative_to(ROOT)),
        "user_features": matrix["user"]["features"],
        "item_features": matrix["item"]["features"], **payload["cuisine"],
    }, indent=2))


if __name__ == "__main__":
    main()
