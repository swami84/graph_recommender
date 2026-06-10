#!/usr/bin/env python3
"""
build_nlp_features.py — Extract restaurant NLP attributes from review text.

Uses a local vLLM endpoint to score 12 restaurant attributes (spice level,
noise, formality, novelty, etc.) from the top-5 most-detailed reviews per
restaurant. Results are cached; re-runs skip already-processed restaurants.

Usage:
    python build_nlp_features.py
    python build_nlp_features.py --local-url http://localhost:8082 --local-model qwen3.5-9b
    python build_nlp_features.py --workers 8        # parallel requests
    python build_nlp_features.py --resume           # skip already done (default on)
    python build_nlp_features.py --no-resume        # reprocess everything
    python build_nlp_features.py --limit 100        # dry-run on first N restaurants
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
log = logging.getLogger("nlp_features")

REVIEWS_FLAT    = Path("data/reviews_flat.parquet")
RESTAURANTS_ENR = Path("data/restaurants_enriched.parquet")
NLP_FEATURES    = Path("data/restaurant_nlp_features.parquet")

ATTRIBUTES = [
    "spice_level",      # 0=mild/no spice,        1=very spicy
    "noise_level",      # 0=quiet/intimate,        1=very loud/noisy
    "formality",        # 0=casual/laid-back,      1=fine dining/elegant
    "novelty",          # 0=ordinary/typical,      1=unique/creative
    "family_friendly",  # 0=not for families,      1=great for kids
    "romantic",         # 0=not romantic,           1=perfect for date night
    "wait_time",        # 0=no wait/quick,          1=long waits/reservation required
    "value_for_money",  # 0=overpriced,             1=excellent value
    "outdoor_seating",  # 0=no outdoor space,       1=ample patio/terrace
    "healthy_options",  # 0=indulgent/heavy,        1=many healthy/light options
    "bar_scene",        # 0=no bar focus,           1=great cocktails/nightlife
    "portion_size",     # 0=small/tapas,            1=very generous portions
]

_SYSTEM = (
    "You are a restaurant attribute extractor. "
    "Given customer reviews, score each attribute from 0.0 to 1.0. "
    "Return ONLY a valid JSON object with exactly the keys listed. "
    "No explanation, no markdown, no extra text — just the JSON."
)

_ATTR_GUIDE = "\n".join([
    "- spice_level:     0=mild or no spice, 1=very spicy/hot",
    "- noise_level:     0=quiet and intimate, 1=very loud and noisy",
    "- formality:       0=casual and relaxed, 1=fine dining and elegant",
    "- novelty:         0=ordinary typical food, 1=unique creative innovative",
    "- family_friendly: 0=not suitable for families, 1=great for kids and families",
    "- romantic:        0=not romantic, 1=perfect for date night couples",
    "- wait_time:       0=no wait or very quick, 1=long waits or reservations required",
    "- value_for_money: 0=overpriced for what you get, 1=excellent value",
    "- outdoor_seating: 0=no outdoor space, 1=ample patio terrace or rooftop",
    "- healthy_options: 0=rich indulgent heavy food, 1=many healthy light options",
    "- bar_scene:       0=no bar focus, 1=great cocktails craft beer nightlife",
    "- portion_size:    0=small or tapas portions, 1=very large generous portions",
])

CHECKPOINT_EVERY = 200   # save parquet after this many restaurants processed
MAX_CHARS        = 2500  # max chars of review text sent to LLM per restaurant
MAX_REVIEW_CHARS = 600   # max chars per individual review before truncation


# ── LLM call ──────────────────────────────────────────────────────────────────
def _call_local(prompt: str, url: str, model: str, timeout: int = 60) -> str:
    payload = json.dumps({
        "model":       model,
        "messages":    [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens":  200,
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
    """Parse JSON from LLM output, tolerating markdown code fences."""
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Find first JSON object in output
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _score_restaurant(place_id: str, name: str, reviews_text: str,
                       url: str, model: str, max_retries: int = 2) -> dict | None:
    prompt = (
        f"Restaurant: {name}\n\n"
        f"Customer reviews:\n{reviews_text}\n\n"
        f"Score each attribute 0.0–1.0:\n{_ATTR_GUIDE}\n\nJSON:"
    )
    for attempt in range(max_retries):
        try:
            raw = _call_local(prompt, url, model)
            parsed = _extract_json(raw)
            if parsed:
                scores = {}
                for attr in ATTRIBUTES:
                    val = parsed.get(attr, 0.5)
                    try:
                        scores[attr] = float(np.clip(float(val), 0.0, 1.0))
                    except (TypeError, ValueError):
                        scores[attr] = 0.5
                return scores
            log.debug(f"  JSON parse failed for {place_id}, attempt {attempt+1}: {raw[:120]}")
        except Exception as exc:
            log.debug(f"  LLM error for {place_id}, attempt {attempt+1}: {exc}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return None


# ── Review selection ──────────────────────────────────────────────────────────
def build_review_corpus(top_n: int = 5) -> pd.DataFrame:
    """
    Return a DataFrame (place_id, reviews_concat) with the top-N most
    informative, deduplicated reviews per restaurant concatenated.
    Prioritises experienced reviewers (reviewer_reviews desc) then length.
    """
    log.info("Loading reviews…")
    rev = pd.read_parquet(
        REVIEWS_FLAT,
        columns=["place_id", "text", "text_len", "has_content", "reviewer_reviews"],
    )
    rev = rev[rev["has_content"] == True].copy()

    # Deduplicate identical review texts per restaurant (same review scraped twice)
    rev["text_str"] = rev["text"].astype(str).str.strip()
    rev = rev.drop_duplicates(subset=["place_id", "text_str"])

    top = (
        rev.sort_values(
            ["place_id", "reviewer_reviews", "text_len"],
            ascending=[True, False, False],
        )
        .groupby("place_id")
        .head(top_n)
    )

    def _concat(texts):
        parts = []
        total = 0
        for t in texts:
            t = str(t).strip()[:MAX_REVIEW_CHARS]
            if total + len(t) > MAX_CHARS:
                break
            parts.append(t)
            total += len(t)
        return "\n\n---\n\n".join(parts)

    corpus = (
        top.groupby("place_id")["text_str"]
        .apply(_concat)
        .reset_index()
        .rename(columns={"text_str": "reviews_concat"})
    )
    log.info(f"Review corpus: {len(corpus):,} restaurants")
    return corpus


# ── Main pipeline ─────────────────────────────────────────────────────────────
def build_features(
    local_url:  str  = "http://localhost:8082",
    local_model: str = "qwen3.5-9b",
    workers:    int  = 4,
    resume:     bool = True,
    limit:      int  = 0,
) -> pd.DataFrame:

    corpus = build_review_corpus()

    rest = pd.read_parquet(RESTAURANTS_ENR, columns=["place_id", "name"])
    corpus = corpus.merge(rest, on="place_id", how="left")
    corpus["name"] = corpus["name"].fillna("Unknown Restaurant")

    # Resume from existing checkpoint
    done_ids: set = set()
    existing_rows: list[dict] = []
    if resume and NLP_FEATURES.exists():
        existing = pd.read_parquet(NLP_FEATURES)
        done_ids = set(existing["place_id"].tolist())
        existing_rows = existing.to_dict("records")
        log.info(f"Resuming — {len(done_ids):,} already processed, skipping")

    todo = corpus[~corpus["place_id"].isin(done_ids)].copy()
    if limit > 0:
        todo = todo.head(limit)
        log.info(f"Limit mode: processing {limit} restaurants")

    log.info(f"To process: {len(todo):,} restaurants  |  workers={workers}")

    results: list[dict] = list(existing_rows)
    failed  = 0

    def _process(row) -> tuple[str, dict]:
        scores = _score_restaurant(
            row.place_id, row.name, row.reviews_concat, local_url, local_model
        )
        if scores is None:
            scores = {attr: 0.5 for attr in ATTRIBUTES}
            return row.place_id, scores, True   # (id, scores, failed)
        return row.place_id, scores, False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process, row): row.place_id
            for row in todo.itertuples(index=False)
        }
        for i, future in enumerate(as_completed(futures), 1):
            place_id, scores, is_failed = future.result()
            results.append({"place_id": place_id, **scores})
            if is_failed:
                failed += 1

            if i % CHECKPOINT_EVERY == 0:
                _save(results)
                pct = 100 * i / len(todo)
                log.info(f"  [{i:,}/{len(todo):,}] {pct:.1f}%  failed={failed}  — checkpoint saved")

    _save(results)
    log.info(f"Done. {len(results):,} restaurants  |  failed (used 0.5 defaults): {failed:,}")
    return pd.read_parquet(NLP_FEATURES)


def _save(rows: list[dict]):
    df = pd.DataFrame(rows)
    # Remove duplicates that may appear if a place_id was in both existing + new
    df = df.drop_duplicates(subset=["place_id"], keep="last")
    df.to_parquet(NLP_FEATURES, index=False)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build restaurant NLP features via local LLM")
    parser.add_argument("--local-url",   default="http://localhost:8082")
    parser.add_argument("--local-model", default="qwen3.5-9b")
    parser.add_argument("--workers",     type=int, default=4,
                        help="Parallel LLM requests (default 4)")
    parser.add_argument("--resume",      action="store_true", default=True,
                        help="Skip already-processed restaurants (default on)")
    parser.add_argument("--no-resume",   action="store_false", dest="resume",
                        help="Reprocess all restaurants")
    parser.add_argument("--limit",       type=int, default=0,
                        help="Process only first N restaurants (0=all)")
    args = parser.parse_args()

    df = build_features(
        local_url=args.local_url,
        local_model=args.local_model,
        workers=args.workers,
        resume=args.resume,
        limit=args.limit,
    )
    print(f"\nSaved {len(df):,} rows → {NLP_FEATURES}")
    print(df[ATTRIBUTES].describe().round(3).to_string())


if __name__ == "__main__":
    main()
