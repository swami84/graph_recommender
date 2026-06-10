#!/usr/bin/env python3
"""
build_restaurant_llm_features.py — LLM-derived + analytical restaurant features.

Two feature groups per restaurant:

  Group A — Fast analytics (no LLM):
    rating_std, photo_rate, repeat_visitor_rate, local_guide_pct,
    restaurant_age_norm, race_rating_gap

  Group B — LLM-extracted (top-25 reviews → structured JSON):
    inferred cuisine (one-hot, 17 cats), authenticity score,
    occasion scores (group/date/family/business/solo), ambiance
    (noise level, outdoor seating), dietary flags (vegan/veg/gf/halal/kosher),
    bar flags (has_bar, byob), service signals (wait time, service speed),
    parking, takeout, best meal period (one-hot, 6 cats)

Outputs:
  data/restaurant_llm_features.parquet

Usage:
    python build_restaurant_llm_features.py
    python build_restaurant_llm_features.py --workers 8
    python build_restaurant_llm_features.py --skip-llm       # only fast features
    python build_restaurant_llm_features.py --no-resume      # reprocess all
    python build_restaurant_llm_features.py --limit 100      # dry-run
"""

import argparse
import json
import logging
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rest_llm")

REVIEWS_FLAT    = Path("data/reviews_flat.parquet")
RESTAURANTS_ENR = Path("data/restaurants_enriched.parquet")
LLM_FEATURES    = Path("data/restaurant_llm_features.parquet")
CHECKPOINT_EVERY = 200

# Exact values from restaurants_enriched.parquet cuisine_category column
CUISINE_CATS = [
    "American & Comfort", "Asian (Other)", "BBQ & Steakhouse", "Bar & Pub",
    "Café & Bakery", "Chinese", "Fast Food & Burgers", "Fine Dining",
    "Indian & South Asian", "Italian & Pizza", "Japanese & Sushi",
    "Mediterranean & Middle Eastern", "Mexican & Latin", "Other",
    "Sandwiches & Deli", "Seafood", "Soul Food",
]
# Safe column name versions
CUISINE_KEYS = [
    re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in CUISINE_CATS
]

MEAL_PERIODS = ["breakfast", "brunch", "lunch", "dinner", "late_night", "all_day"]

LLM_FLOAT_FIELDS = [
    "authenticity_score", "group_friendly", "date_night",
    "family_friendly", "business_dining", "solo_friendly",
]
LLM_BOOL_FIELDS = [
    "has_bar", "byob", "outdoor_seating", "parking_available",
    "vegan_options", "vegetarian_options", "gluten_free_options",
    "halal", "kosher", "takeout_friendly",
]
_ORDINAL_MAP = {
    "noise_level":   {"quiet": 0.0, "moderate": 0.5, "loud": 1.0},
    "wait_time":     {"short": 0.0, "moderate": 0.5, "long": 1.0},
    "service_speed": {"fast": 1.0,  "moderate": 0.5, "slow": 0.0},
}

_SYSTEM = (
    "You are a restaurant analyst. Given customer reviews, extract structured "
    "attributes about the restaurant. Return ONLY valid JSON — no explanation, "
    "no markdown, no comments."
)


# ── LLM helpers ────────────────────────────────────────────────────────────────

def _call_local(prompt: str, url: str, model: str, timeout: int = 90) -> str:
    payload = json.dumps({
        "model":    model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens":  450,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        r = json.loads(text)
        if isinstance(r, dict):
            return r
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            r = json.loads(m.group(1))
            if isinstance(r, dict):
                return r
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            r = json.loads(m.group(0))
            if isinstance(r, dict):
                return r
        except json.JSONDecodeError:
            pass
    return None


def _truncate(text: str, max_words: int = 200) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")


def _build_prompt(place_name: str, known_cuisine: str, reviews: list[str]) -> str:
    block = "\n---\n".join(f"[{i+1}] {_truncate(r)}" for i, r in enumerate(reviews))
    return (
        f'Restaurant: "{place_name}"  (currently classified as: {known_cuisine})\n\n'
        f"Customer reviews:\n---\n{block}\n---\n\n"
        "Extract these attributes as a single JSON object:\n"
        f'  "inferred_cuisine": one of {CUISINE_CATS}\n'
        '  "authenticity_score": float 0-1  (1 = highly authentic)\n'
        '  "group_friendly": float 0-1\n'
        '  "date_night": float 0-1\n'
        '  "family_friendly": float 0-1\n'
        '  "business_dining": float 0-1\n'
        '  "solo_friendly": float 0-1\n'
        '  "has_bar": bool\n'
        '  "byob": bool\n'
        '  "noise_level": "quiet" | "moderate" | "loud"\n'
        '  "wait_time": "short" | "moderate" | "long"\n'
        '  "service_speed": "fast" | "moderate" | "slow"\n'
        '  "outdoor_seating": bool\n'
        '  "parking_available": bool\n'
        '  "vegan_options": bool\n'
        '  "vegetarian_options": bool\n'
        '  "gluten_free_options": bool\n'
        '  "halal": bool\n'
        '  "kosher": bool\n'
        '  "takeout_friendly": bool\n'
        f'  "best_meal_period": one of {MEAL_PERIODS}\n\n'
        "JSON only:"
    )


def _fallback_llm() -> dict:
    row: dict = {}
    for f in LLM_FLOAT_FIELDS:
        row[f"llm_{f}"] = 0.5
    for f in LLM_BOOL_FIELDS:
        row[f"llm_{f}"] = 0.0
    for k in _ORDINAL_MAP:
        row[f"llm_{k}_enc"] = 0.5
    for key in CUISINE_KEYS:
        row[f"llm_cuisine_{key}"] = 0.0
    for p in MEAL_PERIODS:
        row[f"llm_meal_{p}"] = 0.0
    return row


def _encode_llm(raw: dict) -> dict:
    row: dict = {}

    # Float fields — clamp [0, 1]
    for f in LLM_FLOAT_FIELDS:
        try:
            row[f"llm_{f}"] = float(np.clip(float(raw.get(f, 0.5)), 0.0, 1.0))
        except (ValueError, TypeError):
            row[f"llm_{f}"] = 0.5

    # Bool fields
    for f in LLM_BOOL_FIELDS:
        v = raw.get(f, False)
        if isinstance(v, bool):
            row[f"llm_{f}"] = 1.0 if v else 0.0
        elif isinstance(v, str):
            row[f"llm_{f}"] = 1.0 if v.lower() in ("true", "yes", "1") else 0.0
        else:
            row[f"llm_{f}"] = float(bool(v))

    # Ordinal fields
    for k, vmap in _ORDINAL_MAP.items():
        raw_v = str(raw.get(k, "")).lower().strip()
        row[f"llm_{k}_enc"] = vmap.get(raw_v, 0.5)

    # Cuisine one-hot — fuzzy match against display names
    inferred = str(raw.get("inferred_cuisine", "Other")).strip()
    matched_key = CUISINE_KEYS[-1]  # default "other"
    inferred_lower = inferred.lower()
    for cat, key in zip(CUISINE_CATS, CUISINE_KEYS):
        if cat.lower() == inferred_lower or key in inferred_lower or inferred_lower in cat.lower():
            matched_key = key
            break
    for key in CUISINE_KEYS:
        row[f"llm_cuisine_{key}"] = 1.0 if key == matched_key else 0.0

    # Meal period one-hot
    best = str(raw.get("best_meal_period", "")).lower().strip()
    matched_meal = "dinner"
    for p in MEAL_PERIODS:
        if p in best or best in p:
            matched_meal = p
            break
    for p in MEAL_PERIODS:
        row[f"llm_meal_{p}"] = 1.0 if p == matched_meal else 0.0

    return row


def _analyze_restaurant(
    place_id: str, place_name: str, known_cuisine: str, reviews: list[str],
    url: str, model: str, max_retries: int = 2,
) -> dict:
    if not reviews:
        return _fallback_llm()
    prompt = _build_prompt(place_name, known_cuisine, reviews)
    for attempt in range(max_retries):
        try:
            raw_text = _call_local(prompt, url, model)
            raw = _extract_json(raw_text)
            if raw:
                return _encode_llm(raw)
            log.debug(f"  JSON parse failed for {place_id}: {raw_text[:80]}")
        except Exception as exc:
            log.debug(f"  LLM error for {place_id}, attempt {attempt+1}: {exc}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return _fallback_llm()


# ── Fast analytics features ────────────────────────────────────────────────────

def _norm(s: pd.Series, log: bool = False) -> pd.Series:
    if log:
        s = np.log1p(s)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


def build_fast_features(reviews: pd.DataFrame, restaurants: pd.DataFrame) -> pd.DataFrame:
    log.info("Building fast (non-LLM) features…")
    reviews = reviews.copy()
    reviews["rating_num"] = pd.to_numeric(
        reviews.get("rating", pd.Series(dtype=float)), errors="coerce"
    )

    result = restaurants[["place_id"]].drop_duplicates().copy()

    # Rating consistency — std of ratings per restaurant
    rat_std = (
        reviews.groupby("place_id")["rating_num"].std()
        .fillna(0.0).reset_index()
        .rename(columns={"rating_num": "rating_std"})
    )
    rat_std["rating_std"] = _norm(rat_std["rating_std"])
    result = result.merge(rat_std, on="place_id", how="left")

    # Photo attachment rate — proxy for visual/instagrammable appeal
    reviews["has_photo"] = (reviews["attached_photos"].fillna(0) > 0).astype(float)
    photo_rate = (
        reviews.groupby("place_id")["has_photo"].mean()
        .reset_index().rename(columns={"has_photo": "photo_rate"})
    )
    result = result.merge(photo_rate, on="place_id", how="left")

    # Repeat visitor rate — fraction of contributors who visited >1 time
    visit_counts = (
        reviews.groupby(["place_id", "contributor_id"]).size().reset_index(name="n")
    )
    repeat = (
        visit_counts[visit_counts["n"] > 1]
        .groupby("place_id").size().reset_index(name="repeat_visitors")
    )
    total = visit_counts.groupby("place_id").size().reset_index(name="total_visitors")
    rep_rate = repeat.merge(total, on="place_id", how="right").fillna({"repeat_visitors": 0.0})
    rep_rate["repeat_visitor_rate"] = (
        rep_rate["repeat_visitors"] / rep_rate["total_visitors"].clip(lower=1)
    )
    result = result.merge(rep_rate[["place_id", "repeat_visitor_rate"]], on="place_id", how="left")

    # Local guide concentration — more local guides = more credible reviews
    if "is_local_guide" in reviews.columns:
        lg = (
            reviews.groupby("place_id")["is_local_guide"]
            .apply(lambda x: x.fillna(False).astype(bool).mean())
            .reset_index(name="local_guide_pct")
        )
        result = result.merge(lg, on="place_id", how="left")
    else:
        result["local_guide_pct"] = 0.0

    # Restaurant age — days since first review (log-normalised)
    if "timestamp_days_ago" in reviews.columns:
        age = (
            reviews.groupby("place_id")["timestamp_days_ago"].max()
            .reset_index(name="restaurant_age_days")
        )
        age["restaurant_age_norm"] = _norm(age["restaurant_age_days"], log=True)
        result = result.merge(age[["place_id", "restaurant_age_norm"]], on="place_id", how="left")
    else:
        result["restaurant_age_norm"] = 0.5

    # Cross-demographic rating gap — std of per-race mean ratings
    if "predicted_race" in reviews.columns:
        race_means = (
            reviews.groupby(["place_id", "predicted_race"])["rating_num"]
            .mean().unstack().fillna(np.nan)
        )
        race_gap = (
            race_means.std(axis=1).fillna(0.0).reset_index()
        )
        race_gap.columns = ["place_id", "race_rating_gap"]
        race_gap["race_rating_gap"] = _norm(race_gap["race_rating_gap"])
        result = result.merge(race_gap, on="place_id", how="left")
    else:
        result["race_rating_gap"] = 0.0

    result = result.fillna(0.0).reset_index(drop=True)
    n_feat = len([c for c in result.columns if c != "place_id"])
    log.info(f"Fast features: {len(result):,} restaurants × {n_feat} features")
    return result


# ── LLM pipeline ───────────────────────────────────────────────────────────────

def build_llm_features(
    reviews:     pd.DataFrame,
    restaurants: pd.DataFrame,
    url:         str  = "http://localhost:8082",
    model:       str  = "qwen3.5-9b",
    workers:     int  = 8,
    resume:      bool = True,
    limit:       int  = 0,
) -> pd.DataFrame:

    rest_meta = restaurants[["place_id", "name", "cuisine_category"]].copy()
    rest_meta["name"]             = rest_meta["name"].fillna("Unknown").astype(str)
    rest_meta["cuisine_category"] = rest_meta["cuisine_category"].fillna("Other").astype(str)

    # Top-25 reviews by text length per restaurant (pre-computed lookup)
    rev_text = reviews[["place_id", "text", "text_len"]].copy()
    rev_text = rev_text[rev_text["text_len"].fillna(0) > 20]
    top25 = (
        rev_text.sort_values("text_len", ascending=False)
        .groupby("place_id")
        .head(25)
    )
    meta_lookup  = rest_meta.set_index("place_id")[["name", "cuisine_category"]].to_dict("index")
    top25_lookup = top25.groupby("place_id")["text"].apply(list).to_dict()

    all_place_ids = rest_meta["place_id"].tolist()

    # Resume support
    done: set = set()
    existing_rows: list[dict] = []
    if resume and LLM_FEATURES.exists():
        existing = pd.read_parquet(LLM_FEATURES)
        llm_cols = [c for c in existing.columns if c.startswith("llm_")]
        if llm_cols:
            done = set(existing["place_id"].tolist())
            existing_rows = existing[["place_id"] + llm_cols].to_dict("records")
            log.info(f"Resuming — {len(done):,} restaurants already processed")

    todo = [pid for pid in all_place_ids if pid not in done]
    if limit > 0:
        todo = todo[:limit]

    log.info(
        f"LLM feature extraction: {len(todo):,} restaurants | workers={workers} | "
        f"model={model}"
    )

    results: list[dict] = list(existing_rows)
    failed = 0

    def _process(place_id: str) -> dict:
        meta          = meta_lookup.get(place_id, {})
        place_name    = str(meta.get("name", "Unknown"))
        known_cuisine = str(meta.get("cuisine_category", "Other"))
        rev_texts     = top25_lookup.get(place_id, [])
        row = _analyze_restaurant(place_id, place_name, known_cuisine, rev_texts, url, model)
        row["place_id"] = place_id
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, pid): pid for pid in todo}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                results.append(future.result())
            except Exception as exc:
                pid = futures[future]
                log.warning(f"  Failed {pid}: {exc}")
                fb = _fallback_llm()
                fb["place_id"] = pid
                results.append(fb)
                failed += 1

            if i % CHECKPOINT_EVERY == 0:
                _save_llm(results)
                pct = 100 * i / len(todo)
                log.info(f"  [{i:,}/{len(todo):,}] {pct:.1f}% — checkpoint saved")

    _save_llm(results)
    log.info(f"LLM features done. {len(results):,} restaurants | fallbacks: {failed:,}")
    return pd.read_parquet(LLM_FEATURES)


def _save_llm(rows: list[dict]):
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["place_id"], keep="last")
    df.to_parquet(LLM_FEATURES, index=False)


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build LLM-derived + analytical restaurant features"
    )
    parser.add_argument("--local-url",   default="http://localhost:8082")
    parser.add_argument("--local-model", default="qwen3.5-9b")
    parser.add_argument("--workers",     type=int,  default=8)
    parser.add_argument("--resume",      action="store_true", default=True)
    parser.add_argument("--no-resume",   action="store_false", dest="resume")
    parser.add_argument("--limit",       type=int,  default=0,
                        help="Process only first N restaurants (0=all)")
    parser.add_argument("--skip-llm",    action="store_true",
                        help="Only build fast analytics features, skip LLM calls")
    args = parser.parse_args()

    log.info("Loading data…")
    reviews     = pd.read_parquet(REVIEWS_FLAT)
    restaurants = pd.read_parquet(RESTAURANTS_ENR)

    fast_df = build_fast_features(reviews, restaurants)

    if args.skip_llm:
        fast_df.to_parquet(LLM_FEATURES, index=False)
        n_feat = len([c for c in fast_df.columns if c != "place_id"])
        print(f"Saved (fast only): {len(fast_df):,} restaurants × {n_feat} features → {LLM_FEATURES}")
        return

    llm_df = build_llm_features(
        reviews, restaurants,
        url=args.local_url, model=args.local_model,
        workers=args.workers, resume=args.resume, limit=args.limit,
    )

    # Merge fast + LLM into single output file
    llm_cols = [c for c in llm_df.columns if c != "place_id"]
    final = fast_df.merge(llm_df[["place_id"] + llm_cols], on="place_id", how="left")
    final[llm_cols] = final[llm_cols].fillna(0.0)
    final.to_parquet(LLM_FEATURES, index=False)

    n_feat = len([c for c in final.columns if c != "place_id"])
    print(f"Saved: {len(final):,} restaurants × {n_feat} features → {LLM_FEATURES}")
    print(f"  Fast features: {len([c for c in fast_df.columns if c != 'place_id'])}")
    print(f"  LLM features:  {len(llm_cols)}")


if __name__ == "__main__":
    main()
