#!/usr/bin/env python3
"""Build leakage-safe, mirrored user/restaurant features with local Ollama.

The pipeline first reproduces the publication split used by recommendation_gnn:
for users with at least four unique restaurants, the newest interaction is test,
the second-newest is validation, and older interactions are training. Held-out
review text is excluded from both user and restaurant corpora.

Examples
--------
Prepare and audit corpora without calling an LLM::

    python build_llm_features_ollama.py --prepare-only

Small smoke test::

    python build_llm_features_ollama.py --limit 20 --entities both

Full structured + embedding build::

    python build_llm_features_ollama.py --entities both --with-embeddings
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import aiohttp
import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ollama_features")

DATA_DIR = Path("data")
REVIEWS_FILE = DATA_DIR / "reviews_flat.parquet"
RESTAURANTS_FILE = DATA_DIR / "restaurants_enriched.parquet"
CORPUS_DIR = DATA_DIR / "llm_corpora"
CHECKPOINT_DIR = DATA_DIR / "llm_checkpoints"
RESTAURANT_CORPUS = CORPUS_DIR / "restaurant_train_corpus.parquet"
USER_CORPUS = CORPUS_DIR / "user_train_corpus.parquet"
SPLIT_MANIFEST = CORPUS_DIR / "split_manifest.json"
# The audited 27B rebuild is deliberately versioned. The existing v5 files stay
# untouched until this candidate passes distribution and manual spot checks.
RESTAURANT_OUT = DATA_DIR / "restaurant_llm_features_v6_27b.parquet"
USER_OUT = DATA_DIR / "user_llm_features_v6_27b.parquet"
RESTAURANT_EMB_OUT = DATA_DIR / "restaurant_llm_embeddings.parquet"
USER_EMB_OUT = DATA_DIR / "user_llm_embeddings.parquet"

PROMPT_VERSION = "foodie-audited-v6-27b-20260909"

CUISINES = [
    "american", "southern", "cajun_creole", "bbq", "steakhouse",
    "burgers", "pizza", "italian", "mexican", "latin_american",
    "chinese", "japanese", "korean", "thai", "vietnamese", "indian",
    "mediterranean", "middle_eastern", "african", "caribbean",
    "seafood", "vegetarian_vegan", "cafe_bakery", "desserts",
    "bar_pub", "other",
]
MEAL_PERIODS = ["breakfast", "brunch", "lunch", "dinner", "late_night", "all_day"]

# Mirrored dimensions use the same semantic stem for users and restaurants.
MATCH_FIELDS = [
    "spice", "sweetness", "richness", "healthiness", "novelty", "portion_size",
    "noise", "formality", "romantic", "family", "group", "solo", "business",
    "outdoor", "nightlife", "wait", "service_speed", "reservation",
    "takeout", "parking", "transit", "accessibility", "consistency",
    "value", "local_independent", "seafood", "red_meat", "poultry",
    "plant_based",
]
RESTAURANT_EXTRA_FIELDS = [
    "authenticity", "food_sentiment", "service_sentiment", "atmosphere_sentiment",
    "value_sentiment", "cleanliness", "order_accuracy", "repeat_visit_intent",
    "review_disagreement",
]
USER_EXTRA_FIELDS = [
    "allergy_accommodation_need", "cross_contamination_sensitivity",
    "review_preference_consistency",
]
DIETARY_FIELDS = ["vegan", "vegetarian", "gluten_free", "halal", "kosher"]
CONFIDENCE_FIELDS = [
    "overall_confidence", "taste_confidence", "atmosphere_confidence",
    "operations_confidence", "dietary_confidence", "cuisine_confidence",
]


def _numeric_fields(entity: str) -> list[str]:
    extras = RESTAURANT_EXTRA_FIELDS if entity == "restaurant" else USER_EXTRA_FIELDS
    dietary_prefix = "offers_" if entity == "restaurant" else "restriction_"
    return MATCH_FIELDS + extras + [dietary_prefix + d for d in DIETARY_FIELDS]


def _schema(entity: str) -> dict[str, Any]:
    numeric = _numeric_fields(entity)
    props: dict[str, Any] = {
        "scores": {
            "type": "object",
            "properties": {
                field: {"type": "number", "minimum": 0.0, "maximum": 1.0}
                for field in numeric
            },
            "additionalProperties": False,
        },
        "confidences": {
            "type": "object",
            "properties": {
                field: {"type": "number", "minimum": 0.0, "maximum": 1.0}
                for field in CONFIDENCE_FIELDS
            },
            "required": CONFIDENCE_FIELDS,
            "additionalProperties": False,
        },
        "cuisines": {
            "type": "array", "items": {"type": "string", "enum": CUISINES},
            "uniqueItems": True, "maxItems": 3,
        },
        "meal_periods": {
            "type": "array", "items": {"type": "string", "enum": MEAL_PERIODS},
            "uniqueItems": True,
        },
        "signature_dishes": {
            "type": "array", "items": {"type": "string"}, "maxItems": 8,
        },
    }
    required = [
        "scores", "confidences", "cuisines",
        "meal_periods", "signature_dishes",
    ]
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


RESTAURANT_SCHEMA = _schema("restaurant")
USER_SCHEMA = _schema("user")

SYSTEM_PROMPT = (
    "You are a conservative restaurant-review feature extractor. Use only explicit or "
    "strongly repeated evidence. Scores are floats from 0 to 1. Include a key in scores only "
    "when the supplied text directly supports it; omit unsupported keys rather than guessing "
    "or emitting a neutral placeholder. Cuisine and meal arrays must contain unique values. "
    "Never fill an array to its maximum size. Confidence 0.25 means weak evidence, 0.5 mixed "
    "or moderate evidence, 0.75 repeated evidence, and 1.0 is reserved for unambiguous direct "
    "evidence. Do not infer protected or sensitive personal traits. Return compact, "
    "schema-valid JSON only."
)


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _atomic_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _dataset_fingerprint() -> str:
    h = hashlib.sha256()
    for path in (REVIEWS_FILE, RESTAURANTS_FILE):
        stat = path.stat()
        h.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _join_corpus(rows: pd.DataFrame, max_reviews: int, max_chars: int) -> pd.Series:
    parts: list[str] = []
    used = 0
    for text in rows["text"].astype(str).tolist()[:max_reviews]:
        clean = " ".join(text.split())[:900]
        if not clean:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        clean = clean[:remaining]
        parts.append(clean)
        used += len(clean)
    return pd.Series({
        "corpus": "\n---\n".join(parts),
        "source_review_ids_json": json.dumps(rows["review_id"].astype(str).tolist()[:len(parts)]),
        "source_review_count": len(parts),
    })


def assign_interaction_splits(
    reviews: pd.DataFrame, min_unique_restaurants: int = 4,
) -> pd.DataFrame:
    """Assign held-out pairs with a deterministic order for coarse date ties.

    Google timestamps often collapse several visits to the same day count (for
    example, every review labeled "a year ago" becomes 365). Sorting on that
    count alone lets pandas choose different held-out restaurants when unrelated
    rows are appended. The stable, content-derived tie key makes the assignment
    independent of input row order without implying false temporal precision.
    """
    valid = reviews[
        reviews["contributor_id"].notna()
        & (reviews["contributor_id"] != "")
        & reviews["place_id"].notna()
        & reviews["rating"].notna()
    ].copy()
    if "_review_tie_hash" not in valid:
        valid["_review_tie_hash"] = pd.util.hash_pandas_object(
            valid[["review_id", "text", "rating"]], index=False
        ).to_numpy()
    order = ["contributor_id", "timestamp_days_ago", "place_id", "review_id", "_review_tie_hash"]
    interactions = (
        valid.sort_values(order, kind="mergesort", na_position="last")
        .drop_duplicates(["contributor_id", "place_id"], keep="first")
    )
    counts = interactions["contributor_id"].value_counts()
    eligible = set(counts[counts >= min_unique_restaurants].index.astype(str))
    interactions = interactions[
        interactions["contributor_id"].astype(str).isin(eligible)
    ].copy()
    interactions = interactions.sort_values(
        order, kind="mergesort", na_position="last"
    )
    interactions["split_rank"] = interactions.groupby(
        "contributor_id", sort=False
    ).cumcount()
    interactions["split"] = np.where(
        interactions["split_rank"] == 0, "test",
        np.where(interactions["split_rank"] == 1, "validation", "train"),
    )
    return interactions


def prepare_corpora(
    min_unique_restaurants: int = 4,
    restaurant_reviews: int = 8,
    user_reviews: int = 8,
    restaurant_chars: int = 6000,
    user_chars: int = 5000,
) -> dict[str, Any]:
    """Create leakage-safe corpora and a reproducible split manifest."""
    log.info("Loading canonical reviews and restaurants …")
    cols = [
        "review_id", "place_id", "contributor_id", "rating", "timestamp_days_ago",
        "text", "text_len", "has_content", "reviewer_reviews",
    ]
    reviews = pd.read_parquet(REVIEWS_FILE, columns=cols)
    restaurants = pd.read_parquet(
        RESTAURANTS_FILE, columns=["place_id", "name", "cuisine_category", "types"]
    )
    reviews["_review_tie_hash"] = pd.util.hash_pandas_object(
        reviews[["review_id", "text", "rating"]], index=False
    ).to_numpy()
    interactions = assign_interaction_splits(reviews, min_unique_restaurants)
    eligible = set(interactions["contributor_id"].astype(str))

    pair_split = interactions[["contributor_id", "place_id", "split"]].copy()
    source = reviews.merge(pair_split, on=["contributor_id", "place_id"], how="left")
    source["split"] = source["split"].fillna("auxiliary_train")
    source = source[
        source["has_content"].fillna(False)
        & source["text"].notna()
        & (source["text_len"].fillna(0) >= 20)
        & ~source["split"].isin(["validation", "test"])
    ].copy()
    source = source.sort_values(
        ["place_id", "review_id", "_review_tie_hash"], kind="mergesort"
    ).drop_duplicates(["place_id", "review_id"], keep="first")
    source["reviewer_reviews"] = source["reviewer_reviews"].fillna(0)

    rest_source = source.sort_values(
        ["place_id", "reviewer_reviews", "text_len", "review_id", "_review_tie_hash"],
        ascending=[True, False, False, True, True], kind="mergesort",
    )
    rest_corpus = (
        rest_source.groupby("place_id", sort=False, group_keys=False)
        .apply(_join_corpus, max_reviews=restaurant_reviews, max_chars=restaurant_chars,
               include_groups=False)
        .reset_index()
        .merge(restaurants, on="place_id", how="left")
    )
    rest_corpus = rest_corpus[rest_corpus["corpus"].str.len() > 0].reset_index(drop=True)

    train_pairs = pair_split[pair_split["split"] == "train"][["contributor_id", "place_id"]]
    user_source = reviews.merge(train_pairs, on=["contributor_id", "place_id"], how="inner")
    user_source = user_source[
        user_source["has_content"].fillna(False)
        & user_source["text"].notna()
        & (user_source["text_len"].fillna(0) >= 20)
    ].sort_values(
        ["contributor_id", "review_id", "_review_tie_hash"], kind="mergesort"
    ).drop_duplicates(["contributor_id", "review_id"], keep="first")
    user_source = user_source.sort_values(
        ["contributor_id", "text_len", "review_id", "_review_tie_hash"],
        ascending=[True, False, True, True], kind="mergesort",
    )
    user_corpus = (
        user_source.groupby("contributor_id", sort=False, group_keys=False)
        .apply(_join_corpus, max_reviews=user_reviews, max_chars=user_chars,
               include_groups=False)
        .reset_index()
    )
    user_corpus = user_corpus[user_corpus["corpus"].str.len() > 0].reset_index(drop=True)

    _atomic_parquet(rest_corpus, RESTAURANT_CORPUS)
    _atomic_parquet(user_corpus, USER_CORPUS)
    split_counts = interactions["split"].value_counts().to_dict()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": _dataset_fingerprint(),
        "prompt_version": PROMPT_VERSION,
        "min_unique_restaurants": min_unique_restaurants,
        "split_policy": "per-user newest=test, second-newest=validation, older=train; coarse-date ties broken by place_id, review_id, and stable review-content hash",
        "split_sort_version": "deterministic-coarse-date-ties-v2",
        "eligible_users": len(eligible),
        "interaction_counts": {k: int(v) for k, v in split_counts.items()},
        "restaurant_corpora": len(rest_corpus),
        "user_corpora": len(user_corpus),
        "heldout_review_text_excluded": True,
        "restaurant_auxiliary_reviews_allowed": True,
    }
    _atomic_json(manifest, SPLIT_MANIFEST)
    log.info(
        "Prepared %s restaurant and %s user corpora; held-out text excluded",
        f"{len(rest_corpus):,}", f"{len(user_corpus):,}",
    )
    return manifest


def _restaurant_prompt(row: dict[str, Any]) -> str:
    fields = ", ".join(_numeric_fields("restaurant"))
    confidence = ", ".join(CONFIDENCE_FIELDS)
    return (
        f"Restaurant: {row.get('name') or 'Unknown'}\n"
        f"Known broad cuisine: {row.get('cuisine_category') or 'Other'}\n"
        f"Google place types: {row.get('types') or 'Unknown'}\n"
        f"Training-only customer reviews ({row['source_review_count']}):\n{row['corpus']}\n\n"
        "For restaurant properties, 0 means absent/low and 1 means strong/high. "
        "For wait, noise, formality, price/value-related fields follow their literal direction. "
        "offers_* fields describe supported options. Allowed score keys are: "
        f"[{fields}]. Include only supported keys. Cuisine labels require direct support from "
        "the restaurant name, Google place types, known broad cuisine, named dishes, or explicit "
        "review text. Generic words such as chicken, fried, seafood, pizza, or burger alone do "
        "not justify an unrelated regional cuisine. In particular, never label a restaurant "
        "Southern merely because it is American, serves fried food, or is located in the US. "
        "Use no more than three distinct cuisine labels. The confidences object must contain: "
        f"[{confidence}]. Omit unsupported scores and extract the remaining schema fields."
    )


def _user_prompt(row: dict[str, Any]) -> str:
    fields = ", ".join(_numeric_fields("user"))
    confidence = ", ".join(CONFIDENCE_FIELDS)
    return (
        f"Training-only reviews written by one user ({row['source_review_count']}):\n"
        f"{row['corpus']}\n\n"
        "Infer stable dining preferences, not the quality of a single reviewed restaurant. "
        "For mirrored fields, 0 means avoids/prefers the low end and 1 means seeks/prefers "
        "the high end. restriction_* means evidence of a real constraint, not merely liking "
        "that food. Allowed score keys are: "
        f"[{fields}]. Include only supported keys. A cuisine preference requires repeated visits "
        "or explicit preference language; a single reviewed restaurant is insufficient. Use no "
        "more than three distinct cuisine labels. The confidences object must contain: "
        f"[{confidence}]. Omit unsupported scores and extract the remaining schema fields."
    )


def _validate_raw(raw: dict[str, Any]) -> None:
    """Reject structurally valid but semantically unsafe model output for retry."""
    if not isinstance(raw, dict):
        raise ValueError("output is not an object")
    scores = raw.get("scores")
    confidences = raw.get("confidences")
    cuisines = raw.get("cuisines")
    meals = raw.get("meal_periods")
    if not isinstance(scores, dict) or not isinstance(confidences, dict):
        raise ValueError("scores/confidences must be objects")
    if not isinstance(cuisines, list) or not isinstance(meals, list):
        raise ValueError("cuisines/meal_periods must be arrays")
    # Ollama's JSON-schema decoder does not always enforce uniqueItems. An
    # exact repeated label adds no information, so normalize it rather than
    # retrying the same deterministic response until it fails permanently.
    if not all(isinstance(value, str) for value in cuisines + meals):
        raise ValueError("cuisine and meal labels must be strings")
    raw["cuisines"] = cuisines = list(dict.fromkeys(cuisines))
    raw["meal_periods"] = meals = list(dict.fromkeys(meals))
    if len(cuisines) > 3:
        raise ValueError("too many cuisine labels")
    if not set(cuisines).issubset(CUISINES):
        raise ValueError("unknown cuisine label")
    if not set(meals).issubset(MEAL_PERIODS):
        raise ValueError("unknown meal-period label")
    for key, value in scores.items():
        number = float(value)
        if not np.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f"invalid score: {key}")
    for key in CONFIDENCE_FIELDS:
        number = float(confidences[key])
        if not np.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f"invalid confidence: {key}")


class OllamaClient:
    def __init__(self, base_urls: list[str], model: str, concurrency: int, timeout: int):
        self.urls = [url.rstrip("/") + "/api/chat" for url in base_urls]
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.eval_ns = 0
        self.request_count = 0

    async def extract(
        self, session: aiohttp.ClientSession, prompt: str, schema: dict[str, Any],
        retries: int = 2,
    ) -> dict[str, Any]:
        body = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "format": schema,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1000},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        last_error = "unknown error"
        for attempt in range(retries + 1):
            try:
                async with self.sem:
                    url = self.urls[self.request_count % len(self.urls)]
                    self.request_count += 1
                    async with session.post(url, json=body, timeout=self.timeout) as resp:
                        text = await resp.text()
                        if resp.status != 200:
                            raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
                payload = json.loads(text)
                raw = json.loads(payload["message"]["content"])
                _validate_raw(raw)
                self.prompt_tokens += int(payload.get("prompt_eval_count", 0))
                self.output_tokens += int(payload.get("eval_count", 0))
                self.eval_ns += int(payload.get("eval_duration", 0))
                return raw
            except Exception as exc:  # noqa: BLE001 - preserve error in checkpoint
                last_error = str(exc)
                if attempt < retries:
                    await asyncio.sleep(1.5 * (attempt + 1))
        raise RuntimeError(last_error)


def _repair_checkpoint_if_needed(path: Path) -> None:
    """Atomically remove records interrupted by a power loss.

    JSONL appends can leave a partial final record. A service may also append one
    valid record after that fragment during automatic boot recovery, making the
    malformed record an interior line. Preserve every independently valid line.
    """
    if not path.exists():
        return
    malformed = 0
    final_has_newline = True
    with path.open("rb") as handle:
        for line in handle:
            final_has_newline = line.endswith(b"\n")
            try:
                json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                malformed += 1
    if malformed == 0 and final_has_newline:
        return

    tmp = path.with_suffix(path.suffix + ".repair")
    kept = 0
    with path.open("rb") as source, tmp.open("wb") as target:
        for line in source:
            try:
                json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            target.write(line if line.endswith(b"\n") else line + b"\n")
            kept += 1
        target.flush()
        os.fsync(target.fileno())
    os.replace(tmp, path)
    log.warning(
        "Repaired checkpoint %s: retained %s valid records, removed %s malformed records",
        path, f"{kept:,}", malformed,
    )


def _load_checkpoint(path: Path, id_col: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    _repair_checkpoint_if_needed(path)
    with path.open("rb") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                if row.get("status") == "ok" and row.get("prompt_version") == PROMPT_VERSION:
                    rows[str(row[id_col])] = row
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
                continue
    return rows


def _encode(entity: str, entity_id: str, raw: dict[str, Any], source: dict[str, Any], model: str) -> dict[str, Any]:
    prefix = "llm_" if entity == "restaurant" else "user_llm_"
    numeric = _numeric_fields(entity)
    scores = raw.get("scores", {})
    # Sparse schema: the presence of a score is itself the evidence mask. Missing
    # scores retain the neutral value and are explicitly marked unknown below.
    evidence = {str(v) for v in scores}
    confidences = raw.get("confidences", {})
    row: dict[str, Any] = {
        ("place_id" if entity == "restaurant" else "contributor_id"): entity_id,
        "llm_model": model,
        "prompt_version": PROMPT_VERSION,
        "source_review_ids_json": source["source_review_ids_json"],
        "source_review_count": int(source["source_review_count"]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    row[prefix + "evidence_strength"] = min(int(source["source_review_count"]), 8) / 8.0
    unknown: list[str] = []
    for field in numeric:
        try:
            score = float(np.clip(float(scores.get(field, 0.5)), 0.0, 1.0))
        except (AttributeError, TypeError, ValueError):
            score = 0.5
        is_known = field in evidence
        if not is_known:
            unknown.append(field)
        row[prefix + field] = score
        row[prefix + field + "_known"] = float(is_known)
    for field in CONFIDENCE_FIELDS:
        try:
            row[prefix + field] = float(np.clip(float(confidences.get(field, 0.0)), 0.0, 1.0))
        except (AttributeError, TypeError, ValueError):
            row[prefix + field] = 0.0
    selected_cuisines = set(raw.get("cuisines", []))
    selected_meals = set(raw.get("meal_periods", []))
    for value in CUISINES:
        row[prefix + "cuisine_" + value] = float(value in selected_cuisines)
    for value in MEAL_PERIODS:
        row[prefix + "meal_" + value] = float(value in selected_meals)
    row["signature_dishes_json"] = json.dumps(raw.get("signature_dishes", []), ensure_ascii=False)
    row["unknown_attributes_json"] = json.dumps(unknown)
    row["raw_response_json"] = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
    return row


async def generate_structured(
    entity: str,
    model: str,
    base_urls: list[str],
    concurrency: int,
    timeout: int,
    limit: int,
    checkpoint_every: int,
    log_every: int,
) -> pd.DataFrame:
    id_col = "place_id" if entity == "restaurant" else "contributor_id"
    corpus_path = RESTAURANT_CORPUS if entity == "restaurant" else USER_CORPUS
    out_path = RESTAURANT_OUT if entity == "restaurant" else USER_OUT
    checkpoint = CHECKPOINT_DIR / f"{entity}_{PROMPT_VERSION}.jsonl"
    corpus = pd.read_parquet(corpus_path)
    if limit > 0:
        corpus = corpus.head(limit)
    completed = _load_checkpoint(checkpoint, id_col)
    prefix = "llm_" if entity == "restaurant" else "user_llm_"
    for saved in completed.values():
        saved.setdefault(
            prefix + "evidence_strength",
            min(int(saved.get("source_review_count", 0)), 8) / 8.0,
        )
    todo = corpus[~corpus[id_col].astype(str).isin(completed)].copy()
    log.info(
        "%s structured extraction: %s total, %s resumed, %s remaining, concurrency=%d",
        entity.title(), f"{len(corpus):,}", f"{len(completed):,}", f"{len(todo):,}", concurrency,
    )
    client = OllamaClient(base_urls, model, concurrency, timeout)
    prompt_fn = _restaurant_prompt if entity == "restaurant" else _user_prompt
    schema = RESTAURANT_SCHEMA if entity == "restaurant" else USER_SCHEMA
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    async with aiohttp.ClientSession() as session:
        with checkpoint.open("a", buffering=1) as handle:
            async def run_one(source: dict[str, Any]) -> dict[str, Any]:
                entity_id = str(source[id_col])
                try:
                    raw = await client.extract(session, prompt_fn(source), schema)
                    row = _encode(entity, entity_id, raw, source, model)
                    return {**row, "status": "ok"}
                except Exception as exc:  # noqa: BLE001
                    return {
                        id_col: entity_id, "status": "error", "error": str(exc),
                        "prompt_version": PROMPT_VERSION, "llm_model": model,
                    }

            tasks = [asyncio.create_task(run_one(row)) for row in todo.to_dict("records")]
            failures = 0
            for done, future in enumerate(asyncio.as_completed(tasks), 1):
                result = await future
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                if result["status"] == "ok":
                    completed[str(result[id_col])] = result
                else:
                    failures += 1
                if done % log_every == 0 or done == len(tasks):
                    # Bound checkpoint loss to at most one reporting interval on
                    # sudden power failure; startup repair handles a torn line.
                    handle.flush()
                    os.fsync(handle.fileno())
                if done % checkpoint_every == 0 or done == len(tasks):
                    frame = pd.DataFrame([
                        {k: v for k, v in row.items() if k != "status"}
                        for row in completed.values()
                    ])
                    _atomic_parquet(frame, out_path)
                if done % log_every == 0 or done == len(tasks):
                    elapsed = max(time.monotonic() - started, 1e-6)
                    log.info(
                        "[%s/%s] %.1f entities/min, failures=%d, output_tok/s=%.1f",
                        f"{done:,}", f"{len(tasks):,}", done * 60 / elapsed, failures,
                        client.output_tokens * 1e9 / max(client.eval_ns, 1),
                    )
    return pd.read_parquet(out_path)


async def generate_embeddings(
    entity: str,
    model: str,
    base_urls: list[str],
    batch_size: int,
    projection_dim: int,
    limit: int,
    timeout: int,
    checkpoint_every: int = 64,
) -> pd.DataFrame:
    id_col = "place_id" if entity == "restaurant" else "contributor_id"
    corpus_path = RESTAURANT_CORPUS if entity == "restaurant" else USER_CORPUS
    out_path = RESTAURANT_EMB_OUT if entity == "restaurant" else USER_EMB_OUT
    corpus = pd.read_parquet(corpus_path, columns=[id_col, "corpus", "source_review_count"])
    if limit > 0:
        corpus = corpus.head(limit)
    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    done_ids = set(existing[id_col].astype(str)) if not existing.empty else set()
    todo = corpus[~corpus[id_col].astype(str).isin(done_ids)].reset_index(drop=True)
    rows = existing.to_dict("records") if not existing.empty else []
    urls = [base_url.rstrip("/") + "/api/embed" for base_url in base_urls]
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    projection: np.ndarray | None = None
    started = time.monotonic()

    async with aiohttp.ClientSession() as session:
        for start in range(0, len(todo), batch_size):
            batch = todo.iloc[start:start + batch_size]
            body = {
                "model": model, "input": batch["corpus"].tolist(), "keep_alive": "30m",
                "truncate": True, "options": {"num_ctx": 8192},
            }
            url = urls[(start // batch_size) % len(urls)]
            payload = None
            for attempt in range(3):
                try:
                    async with session.post(url, json=body, timeout=client_timeout) as resp:
                        payload = await resp.json()
                        if resp.status != 200:
                            raise RuntimeError(f"Ollama embedding HTTP {resp.status}: {payload}")
                    break
                except (asyncio.TimeoutError, aiohttp.ClientError, RuntimeError) as exc:
                    if attempt == 2:
                        raise RuntimeError(
                            f"Embedding batch {start // batch_size + 1} failed after 3 attempts"
                        ) from exc
                    log.warning(
                        "%s embedding batch %s timed out/failed (%s); retrying %s/3",
                        entity, start // batch_size + 1, exc, attempt + 2,
                    )
                    await asyncio.sleep(5 * (attempt + 1))
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            if projection is None:
                rng = np.random.default_rng(42)
                projection = rng.normal(
                    0, 1 / np.sqrt(projection_dim),
                    size=(embeddings.shape[1], projection_dim),
                ).astype(np.float32)
            reduced = embeddings @ projection
            norms = np.linalg.norm(reduced, axis=1, keepdims=True).clip(min=1e-8)
            reduced /= norms
            for (_, source), vector in zip(batch.iterrows(), reduced):
                row = {
                    id_col: str(source[id_col]), "embedding_model": model,
                    "embedding_projection": f"gaussian_seed42_{projection_dim}d",
                    "source_review_count": int(source["source_review_count"]),
                }
                row.update({f"llm_emb_{i:03d}": float(v) for i, v in enumerate(vector)})
                rows.append(row)
            done = min(start + batch_size, len(todo))
            if done % checkpoint_every == 0 or done == len(todo):
                _atomic_parquet(
                    pd.DataFrame(rows).drop_duplicates(id_col, keep="last"), out_path
                )
            rate = done * 60 / max(time.monotonic() - started, 1e-6)
            if done % checkpoint_every == 0 or done == len(todo):
                log.info("%s embeddings [%s/%s], %.1f entities/min", entity, done, len(todo), rate)
    return pd.read_parquet(out_path)


async def async_main(args: argparse.Namespace) -> None:
    base_urls = [url.strip() for url in args.urls.split(",") if url.strip()]
    if not base_urls:
        base_urls = [args.url]
    if args.prepare or args.prepare_only or not RESTAURANT_CORPUS.exists() or not USER_CORPUS.exists():
        manifest = prepare_corpora(min_unique_restaurants=args.min_unique_restaurants)
        log.info("Split manifest: %s", json.dumps(manifest, sort_keys=True))
    if args.prepare_only:
        return

    entities = ["restaurant", "user"] if args.entities == "both" else [args.entities]
    for entity in entities:
        await generate_structured(
            entity=entity, model=args.model, base_urls=base_urls,
            concurrency=args.concurrency, timeout=args.timeout, limit=args.limit,
            checkpoint_every=args.checkpoint_every, log_every=args.log_every,
        )
    if args.with_embeddings:
        for entity in entities:
            await generate_embeddings(
                entity=entity, model=args.embedding_model, base_urls=base_urls,
                batch_size=args.embedding_batch_size,
                projection_dim=args.embedding_dim, limit=args.limit, timeout=args.timeout,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--urls", default="",
                        help="Comma-separated Ollama servers; requests are round-robin distributed")
    parser.add_argument("--model", default="qwen3-8b-fast:latest")
    parser.add_argument("--embedding-model", default="qwen3-embedding:0.6b")
    parser.add_argument("--entities", choices=["restaurant", "user", "both"], default="both")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-unique-restaurants", type=int, default=4)
    parser.add_argument("--prepare", action="store_true", help="Rebuild leakage-safe corpora")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--with-embeddings", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=64)
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
