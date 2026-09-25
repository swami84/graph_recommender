#!/usr/bin/env python3
"""Assemble a versioned expanded semantic-feature contract.

Rows are selected by the deterministic source-action audit. Reused rows retain
their original values; regenerated rows are normalized to the exact article-era
feature schema so expanded-vs-earlier model comparisons do not change feature
dimensionality. Article artifacts are read-only.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BATCH_V1 = DATA / "feature_batches" / "places_expansion_2026-09-16"
BATCH = DATA / "feature_batches" / "places_expansion_2026-09-16_split_v2"
OUT = BATCH / "assembled"

ARTICLE_STRUCTURED = {
    "restaurant": DATA / "audited_llm_features" / "restaurant_llm_features.parquet",
    "user": DATA / "audited_llm_features" / "user_llm_features.parquet",
}
V1_STRUCTURED = {
    "restaurant": BATCH_V1 / "restaurant_llm_features.parquet",
    "user": BATCH_V1 / "user_llm_features.parquet",
}
V2_STRUCTURED = {
    "restaurant": BATCH / "restaurant_llm_features.parquet",
    "user": BATCH / "user_llm_features.parquet",
}
ARTICLE_EMBEDDING = {
    "restaurant": DATA / "restaurant_llm_embeddings.parquet",
    "user": DATA / "user_llm_embeddings.parquet",
}
V1_EMBEDDING = {
    "restaurant": BATCH_V1 / "restaurant_llm_embeddings.parquet",
    "user": BATCH_V1 / "user_llm_embeddings.parquet",
}
V2_EMBEDDING = {
    "restaurant": BATCH / "restaurant_llm_embeddings.parquet",
    "user": BATCH / "user_llm_embeddings.parquet",
}

CUISINE_MAP = {
    "american": "american", "southern": "southern", "cajun_creole": "cajun_creole",
    "bbq": "bbq", "steakhouse": "steakhouse", "burgers": "burgers",
    "pizza": "pizza", "italian": "italian", "mexican": "mexican",
    "latin_american": "latin_american", "chinese": "chinese",
    "japanese": "japanese", "korean": "korean", "thai": "thai",
    "vietnamese": "vietnamese", "indian": "indian",
    "mediterranean": "mediterranean", "middle_eastern": "middle_eastern",
    "african": "african", "caribbean": "caribbean", "seafood": "seafood",
    "vegetarian_vegan": "vegan", "cafe_bakery": "cafe_bakery",
    "desserts": "desserts", "bar_pub": "bar_pub", "other": "unknown",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_restaurant(raw: pd.DataFrame, schema: list[str]) -> pd.DataFrame:
    """Translate v7 sparse outputs into the frozen article restaurant schema."""
    result = pd.DataFrame(index=raw.index)
    result["place_id"] = raw["place_id"].astype(str)
    for col in schema:
        if col == "place_id":
            continue
        if col in raw:
            result[col] = raw[col]

    cuisine_cols = [c for c in schema if c.startswith("llm_cuisine27_")]
    for col in cuisine_cols:
        result[col] = 0.0
    for source, target in CUISINE_MAP.items():
        source_col = f"llm_cuisine_{source}"
        target_col = f"llm_cuisine27_{target}"
        if source_col in raw and target_col in result:
            result[target_col] = np.maximum(
                pd.to_numeric(result[target_col], errors="coerce").fillna(0),
                pd.to_numeric(raw[source_col], errors="coerce").fillna(0),
            )
    if "llm_cuisine27_confidence" in result:
        result["llm_cuisine27_confidence"] = pd.to_numeric(
            raw.get("llm_cuisine_confidence", 0), errors="coerce"
        ).fillna(0)

    aliases = {
        "llm_audited_offers_vegan": "llm_offers_vegan",
        "llm_audited_offers_vegan_known": "llm_offers_vegan_known",
        "llm_audited_offers_vegetarian": "llm_offers_vegetarian",
        "llm_audited_offers_vegetarian_known": "llm_offers_vegetarian_known",
    }
    for target, source in aliases.items():
        if target in schema:
            result[target] = pd.to_numeric(raw.get(source, 0), errors="coerce").fillna(0)

    corpus = pd.read_parquet(
        BATCH / "llm_corpora" / "restaurant_train_corpus.parquet",
        columns=["place_id", "types", "corpus"],
    )
    evidence = raw[["place_id"]].merge(corpus, on="place_id", how="left", validate="one_to_one")
    text = (evidence["types"].fillna("") + " " + evidence["corpus"].fillna("")).str.lower()
    bar = text.str.contains(
        r"(?:^|,)(?:bar|pub|cocktail_bar|wine_bar|sports_bar)(?:,|$)|\bfull bar\b|\bbar (?:area|seating|menu)\b",
        regex=True,
    ).astype(float).to_numpy()
    byob = text.str.contains(r"\bbyob\b|\bbring your own (?:beer|wine|bottle)\b", regex=True).astype(float).to_numpy()
    for stem, values in (("has_bar", bar), ("byob", byob)):
        for suffix in ("", "_known"):
            col = f"llm_audited_{stem}{suffix}"
            if col in schema:
                result[col] = values

    for col in schema:
        if col not in result:
            result[col] = 0.0 if any(x in col for x in ("_known", "_confidence", "cuisine", "meal_")) else 0.5
    return result[schema]


def normalize_user(raw: pd.DataFrame, schema: list[str]) -> pd.DataFrame:
    result = raw.reindex(columns=schema).copy()
    for col in schema:
        if col == "contributor_id":
            result[col] = raw[col].astype(str)
        elif col not in raw:
            result[col] = 0.0 if any(x in col for x in ("_known", "_confidence", "cuisine", "meal_")) else 0.5
    return result[schema]


def select_rows(frame: pd.DataFrame, ids: set[str], key: str) -> pd.DataFrame:
    frame[key] = frame[key].astype(str)
    selected = frame[frame[key].isin(ids)].drop_duplicates(key, keep="last")
    missing = ids - set(selected[key])
    if missing:
        raise ValueError(f"{len(missing):,} {key} rows missing from selected source")
    return selected


def assemble_side(side: str, key: str) -> dict:
    actions = pd.read_parquet(BATCH / f"{side}_source_actions.parquet")
    actions[key] = actions[key].astype(str)
    article_schema = list(pd.read_parquet(ARTICLE_STRUCTURED[side], engine="pyarrow").columns)
    parts = []
    sources = [
        ("reuse_article", ARTICLE_STRUCTURED[side]),
        ("reuse_september_v1", V1_STRUCTURED[side]),
        ("generate", V2_STRUCTURED[side]),
    ]
    for action, path in sources:
        ids = set(actions.loc[actions.feature_action.eq(action), key])
        frame = select_rows(pd.read_parquet(path), ids, key)
        if action != "reuse_article":
            frame = (normalize_restaurant(frame, article_schema) if side == "restaurant"
                     else normalize_user(frame, article_schema))
        else:
            frame = frame[article_schema]
        parts.append(frame)
    structured = pd.concat(parts, ignore_index=True)
    if structured[key].nunique() != len(structured) or len(structured) != len(actions):
        raise ValueError(f"Invalid assembled {side} structured coverage")

    emb_parts = []
    for action, path in [
        ("reuse_article", ARTICLE_EMBEDDING[side]),
        ("reuse_september_v1", V1_EMBEDDING[side]),
        ("generate", V2_EMBEDDING[side]),
    ]:
        ids = set(actions.loc[actions.feature_action.eq(action), key])
        emb_parts.append(select_rows(pd.read_parquet(path), ids, key))
    embeddings = pd.concat(emb_parts, ignore_index=True)
    if embeddings[key].nunique() != len(embeddings) or len(embeddings) != len(actions):
        raise ValueError(f"Invalid assembled {side} embedding coverage")
    emb_cols = [c for c in embeddings if c.startswith("llm_emb_")]
    if len(emb_cols) != 64 or embeddings[emb_cols].isna().any().any():
        raise ValueError(f"Invalid {side} embedding matrix")

    OUT.mkdir(parents=True, exist_ok=True)
    structured_path = OUT / f"{side}_llm_features.parquet"
    embedding_path = OUT / f"{side}_llm_embeddings.parquet"
    structured.to_parquet(structured_path, index=False)
    embeddings.to_parquet(embedding_path, index=False)
    return {
        "structured_rows": len(structured), "structured_features": len(article_schema) - 1,
        "embedding_rows": len(embeddings), "embedding_features": len(emb_cols),
        "structured_sha256": sha256(structured_path), "embedding_sha256": sha256(embedding_path),
    }


def main() -> None:
    report = {
        "schema_version": "expanded-deterministic-feature-contract-v1",
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        "article_artifacts_modified": False,
        "restaurant": assemble_side("restaurant", "place_id"),
        "user": assemble_side("user", "contributor_id"),
    }
    (OUT / "assembly_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
