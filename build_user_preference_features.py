#!/usr/bin/env python3
"""
build_user_preference_features.py — User dining preference extraction via local LLM.

For each model user (≥3 unique restaurants), takes their most informative reviews
and scores 12 preference dimensions that mirror restaurant_nlp_features.parquet.
This enables direct user↔restaurant feature matching in the GNN.

Preference dims (match restaurant NLP attributes 1-to-1):
  spice_affinity, noise_preference, formality_preference, novelty_preference,
  family_context, romantic_context, wait_tolerance, value_sensitivity,
  outdoor_preference, healthy_preference, bar_affinity, portion_preference

Outputs:
  data/user_preference_features.parquet

Usage:
    python build_user_preference_features.py
    python build_user_preference_features.py --workers 80
    python build_user_preference_features.py --no-resume
    python build_user_preference_features.py --limit 500  # dry-run
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
log = logging.getLogger("user_pref")

REVIEWS_FLAT  = Path("data/reviews_flat.parquet")
USER_PREF_OUT = Path("data/user_preference_features.parquet")

CHECKPOINT_EVERY = 500
MAX_REVIEWS      = 5    # reviews per user sent to LLM
MAX_CHARS        = 2000 # total chars across all reviews
MAX_REVIEW_CHARS = 500  # max chars per individual review before truncation

# 12 dims mirroring restaurant_nlp_features.parquet attribute names
ATTRIBUTES = [
    "spice_affinity",        # 0=always orders mild, 1=seeks out spicy dishes
    "noise_preference",      # 0=prefers quiet/intimate, 1=enjoys lively/energetic
    "formality_preference",  # 0=strictly casual diner, 1=appreciates fine dining
    "novelty_preference",    # 0=sticks to familiar food, 1=seeks unique/creative
    "family_context",        # 0=adult/solo dining, 1=frequently dines with kids
    "romantic_context",      # 0=rarely romantic dining, 1=often on dates
    "wait_tolerance",        # 0=complains about any wait, 1=accepts waits for quality
    "value_sensitivity",     # 0=never mentions price, 1=always evaluates value
    "outdoor_preference",    # 0=no mention of outdoor seating, 1=specifically seeks it
    "healthy_preference",    # 0=orders indulgent/heavy food, 1=health-conscious
    "bar_affinity",          # 0=never mentions drinks/cocktails, 1=loves bar scene
    "portion_preference",    # 0=prefers small/light portions, 1=loves large portions
]

_SYSTEM = (
    "You are a dining preference analyst. "
    "Given a person's restaurant reviews, infer their dining preferences. "
    "Return ONLY a valid JSON object with exactly the listed keys scored 0.0–1.0. "
    "No explanation, no markdown — just the JSON."
)

_ATTR_GUIDE = "\n".join([
    "- spice_affinity:       0=always orders mild food, 1=seeks out spicy dishes",
    "- noise_preference:     0=prefers quiet intimate settings, 1=enjoys lively energetic atmosphere",
    "- formality_preference: 0=strictly casual diner, 1=appreciates formal fine dining",
    "- novelty_preference:   0=sticks to familiar comfort food, 1=seeks unique creative cuisine",
    "- family_context:       0=dines alone or adults only, 1=frequently dines with family/kids",
    "- romantic_context:     0=rarely mentions dates or romance, 1=often dines for date nights",
    "- wait_tolerance:       0=complains about any wait, 1=willingly waits for quality food",
    "- value_sensitivity:    0=never mentions price or value, 1=always evaluates bang for buck",
    "- outdoor_preference:   0=no mention of outdoor spaces, 1=specifically seeks outdoor seating",
    "- healthy_preference:   0=orders burgers/fried/indulgent food, 1=gravitates to healthy/light",
    "- bar_affinity:         0=never mentions drinks or cocktails, 1=frequently mentions bar scene",
    "- portion_preference:   0=mentions small/light portions positively, 1=loves large generous portions",
])


def _call_local(prompt: str, url: str, model: str, timeout: int = 60) -> str:
    payload = json.dumps({
        "model":    model,
        "messages": [
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
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _score_user(contributor_id: str, reviews_text: str,
                url: str, model: str, max_retries: int = 2) -> dict | None:
    prompt = (
        f"Here are restaurant reviews written by a single person:\n\n"
        f"{reviews_text}\n\n"
        f"Based on these reviews, score this person's dining preferences 0.0–1.0:\n"
        f"{_ATTR_GUIDE}\n\nJSON:"
    )
    for attempt in range(max_retries):
        try:
            raw    = _call_local(prompt, url, model)
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
            log.debug(f"  JSON parse failed for {contributor_id}: {raw[:100]}")
        except Exception as exc:
            log.debug(f"  LLM error for {contributor_id}, attempt {attempt+1}: {exc}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return None


def build_user_corpus() -> pd.DataFrame:
    """
    Returns DataFrame (contributor_id, reviews_concat) for model users only.
    Model users = users with >=3 unique restaurant visits.
    Selects up to MAX_REVIEWS longest reviews per user.
    """
    log.info("Loading reviews…")
    rev = pd.read_parquet(
        REVIEWS_FLAT,
        columns=["contributor_id", "place_id", "text", "text_len", "has_content"],
    )

    # Identify model users: >=3 unique restaurants
    deduped   = rev.dropna(subset=["contributor_id", "place_id"]).drop_duplicates(
        ["contributor_id", "place_id"]
    )
    counts    = deduped.groupby("contributor_id")["place_id"].count()
    model_ids = set(counts[counts >= 3].index)
    log.info(f"Model users: {len(model_ids):,}")

    # Reviews with real text content only
    rev = rev[rev["has_content"] == True].copy()
    rev["text_str"] = rev["text"].astype(str).str.strip()
    rev = rev.drop_duplicates(subset=["contributor_id", "text_str"])
    rev = rev[rev["contributor_id"].isin(model_ids)]

    # Top MAX_REVIEWS longest reviews per user
    top = (
        rev.sort_values(["contributor_id", "text_len"], ascending=[True, False])
        .groupby("contributor_id")
        .head(MAX_REVIEWS)
    )

    def _concat(texts):
        parts, total = [], 0
        for t in texts:
            t = str(t).strip()[:MAX_REVIEW_CHARS]
            if total + len(t) > MAX_CHARS:
                break
            parts.append(t)
            total += len(t)
        return "\n\n---\n\n".join(parts)

    corpus = (
        top.groupby("contributor_id")["text_str"]
        .apply(_concat)
        .reset_index()
        .rename(columns={"text_str": "reviews_concat"})
    )
    corpus = corpus[corpus["reviews_concat"].str.len() > 0]
    log.info(f"User corpus ready: {len(corpus):,} users with review text")
    return corpus


def build_features(
    local_url:   str  = "http://localhost:8082",
    local_model: str  = "qwen3.5-9b",
    workers:     int  = 40,
    resume:      bool = True,
    limit:       int  = 0,
) -> pd.DataFrame:

    corpus = build_user_corpus()

    done_ids:      set        = set()
    existing_rows: list[dict] = []
    if resume and USER_PREF_OUT.exists():
        existing      = pd.read_parquet(USER_PREF_OUT)
        done_ids      = set(existing["contributor_id"].tolist())
        existing_rows = existing.to_dict("records")
        log.info(f"Resuming — {len(done_ids):,} users already processed")

    todo = corpus[~corpus["contributor_id"].isin(done_ids)].copy()
    if limit > 0:
        todo = todo.head(limit)
        log.info(f"Limit mode: processing {limit} users")

    log.info(f"To process: {len(todo):,} users  |  workers={workers}")

    results: list[dict] = list(existing_rows)
    failed  = 0

    def _process(row):
        scores = _score_user(
            row.contributor_id, row.reviews_concat, local_url, local_model
        )
        if scores is None:
            scores = {attr: 0.5 for attr in ATTRIBUTES}
            return row.contributor_id, scores, True
        return row.contributor_id, scores, False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process, row): row.contributor_id
            for row in todo.itertuples(index=False)
        }
        for i, future in enumerate(as_completed(futures), 1):
            contributor_id, scores, is_failed = future.result()
            results.append({"contributor_id": contributor_id, **scores})
            if is_failed:
                failed += 1

            if i % CHECKPOINT_EVERY == 0:
                _save(results)
                pct = 100 * i / len(todo)
                log.info(
                    f"  [{i:,}/{len(todo):,}] {pct:.1f}%  "
                    f"failed={failed}  — checkpoint saved"
                )

    _save(results)
    log.info(
        f"Done. {len(results):,} users  |  "
        f"fallbacks (0.5 defaults): {failed:,}"
    )
    return pd.read_parquet(USER_PREF_OUT)


def _save(rows: list[dict]):
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["contributor_id"], keep="last")
    df.to_parquet(USER_PREF_OUT, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Build user dining preference features via local LLM"
    )
    parser.add_argument("--local-url",   default="http://localhost:8082")
    parser.add_argument("--local-model", default="qwen3.5-9b")
    parser.add_argument("--workers",     type=int, default=40,
                        help="Parallel LLM requests (default 40)")
    parser.add_argument("--resume",      action="store_true", default=True)
    parser.add_argument("--no-resume",   action="store_false", dest="resume")
    parser.add_argument("--limit",       type=int, default=0,
                        help="Process only first N users (0=all)")
    args = parser.parse_args()

    df = build_features(
        local_url=args.local_url,
        local_model=args.local_model,
        workers=args.workers,
        resume=args.resume,
        limit=args.limit,
    )
    print(f"\nSaved {len(df):,} rows → {USER_PREF_OUT}")
    print(df[ATTRIBUTES].describe().round(3).to_string())


if __name__ == "__main__":
    main()
