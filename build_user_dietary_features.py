#!/usr/bin/env python3
"""
build_user_dietary_features.py — LLM-inferred dietary preference signals per user.

Groups each user's substantive reviews (text_len >= 50) and prompts a local
LLM to infer vegan / vegetarian / gluten-free affinity from what they ordered,
mentioned, or sought out across restaurants.

Output:
  data/user_dietary_features.parquet
  columns: contributor_id, user_vegan_affinity, user_vegetarian_affinity,
           user_gluten_free_affinity, user_seafood_affinity,
           user_red_meat_affinity, user_poultry_affinity

Only users with >= --min-reviews substantive reviews are processed; the rest
receive NaN (filled with 0.0 in build_training_features.py).

Usage:
    python build_user_dietary_features.py
    python build_user_dietary_features.py --workers 8
    python build_user_dietary_features.py --limit 200   # dry-run
    python build_user_dietary_features.py --no-resume   # reprocess all
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
log = logging.getLogger("user_dietary")

REVIEWS_FLAT    = Path("data/reviews_flat.parquet")
OUT_FILE        = Path("data/user_dietary_features.parquet")
CHECKPOINT_EVERY = 500

AFFINITY_FIELDS = [
    "user_vegan_affinity",
    "user_vegetarian_affinity",
    "user_gluten_free_affinity",
    "user_seafood_affinity",
    "user_red_meat_affinity",
    "user_poultry_affinity",
]

_SYSTEM = (
    "You are a restaurant reviewer analyst. Given reviews written by the same "
    "person, infer their food preferences. Return ONLY valid JSON."
)


# ── LLM helpers ────────────────────────────────────────────────────────────────

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


def _truncate(text: str, max_words: int = 120) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")


def _build_prompt(reviews: list[str]) -> str:
    block = "\n---\n".join(f"[{i+1}] {_truncate(r)}" for i, r in enumerate(reviews))
    return (
        "These reviews were all written by the same person at different restaurants:\n"
        f"---\n{block}\n---\n\n"
        "Based on what this person orders, mentions, or seeks out, estimate their "
        "food preferences as floats 0.0–1.0.\n\n"
        "Dietary restrictions / avoidances:\n"
        '  "user_vegan_affinity":        0.0 = no signal, 1.0 = actively seeks/avoids animal products\n'
        '  "user_vegetarian_affinity":   0.0 = no signal, 1.0 = avoids meat, orders veg dishes\n'
        '  "user_gluten_free_affinity":  0.0 = no signal, 1.0 = avoids gluten, asks about GF options\n\n'
        "Protein preferences (what they enjoy ordering):\n"
        '  "user_seafood_affinity":      0.0 = never mentions seafood, 1.0 = frequently orders fish/shellfish\n'
        '  "user_red_meat_affinity":     0.0 = never mentions red meat, 1.0 = frequently orders steak/beef/lamb\n'
        '  "user_poultry_affinity":      0.0 = never mentions poultry, 1.0 = frequently orders chicken/duck/turkey\n\n'
        "Scoring guide (applies to all fields):\n"
        "  0.0 — no mention at all\n"
        "  0.3 — occasional mention, not a clear preference\n"
        "  0.7 — frequent mention, clear preference or need\n"
        "  1.0 — explicit preference, restriction, or repeated focus\n\n"
        "JSON only:"
    )


def _fallback() -> dict:
    return {f: 0.0 for f in AFFINITY_FIELDS}


def _encode(raw: dict) -> dict:
    row: dict = {}
    for f in AFFINITY_FIELDS:
        try:
            row[f] = float(np.clip(float(raw.get(f, 0.0)), 0.0, 1.0))
        except (ValueError, TypeError):
            row[f] = 0.0
    return row


def _analyze_user(
    contributor_id: str,
    reviews: list[str],
    url: str,
    model: str,
    max_retries: int = 2,
) -> dict:
    if not reviews:
        return _fallback()
    prompt = _build_prompt(reviews)
    for attempt in range(max_retries):
        try:
            raw_text = _call_local(prompt, url, model)
            raw = _extract_json(raw_text)
            if raw:
                return _encode(raw)
            log.debug("  JSON parse failed for %s: %s", contributor_id, raw_text[:80])
        except Exception as exc:
            log.debug("  LLM error for %s attempt %d: %s", contributor_id, attempt + 1, exc)
            if attempt < max_retries - 1:
                time.sleep(1)
    return _fallback()


# ── Main build function ────────────────────────────────────────────────────────

def build(
    min_reviews:  int  = 3,
    max_reviews:  int  = 10,
    workers:      int  = 4,
    limit:        int  = 0,
    no_resume:    bool = False,
    url:          str  = "http://localhost:8082",
    model:        str  = "qwen3.5-9b",
) -> None:
    log.info("Loading reviews…")
    rev = pd.read_parquet(REVIEWS_FLAT, columns=["contributor_id", "text", "text_len"])
    rev = rev[rev["contributor_id"].notna() & (rev["text_len"] >= 50)].copy()
    log.info(f"  {len(rev):,} substantive reviews from {rev['contributor_id'].nunique():,} users")

    # Users with >= min_reviews substantive reviews
    counts   = rev.groupby("contributor_id").size()
    eligible = counts[counts >= min_reviews].index
    rev      = rev[rev["contributor_id"].isin(eligible)]
    log.info(f"  {len(eligible):,} users with >= {min_reviews} reviews to process")

    # Resume from existing checkpoint
    done_ids: set[str] = set()
    existing_rows: list[dict] = []
    if not no_resume and OUT_FILE.exists():
        existing = pd.read_parquet(OUT_FILE)
        done_ids = set(existing["contributor_id"].astype(str))
        existing_rows = existing.to_dict("records")
        log.info(f"  Resuming: {len(done_ids):,} already done, "
                 f"{len(eligible) - len(done_ids):,} remaining")

    todo = [uid for uid in eligible if str(uid) not in done_ids]
    if limit > 0:
        todo = todo[:limit]
        log.info(f"  Limit: processing {len(todo):,} users")

    if not todo:
        log.info("Nothing to do — all users processed.")
        return

    # Pre-group reviews for efficiency
    grouped = (
        rev[rev["contributor_id"].isin(todo)]
        .groupby("contributor_id")["text"]
        .apply(lambda x: x.sample(min(max_reviews, len(x)), random_state=42).tolist())
        .to_dict()
    )

    results = list(existing_rows)
    n_done  = 0

    def _worker(uid: str) -> dict:
        row = _analyze_user(uid, grouped.get(uid, []), url, model)
        row["contributor_id"] = uid
        return row

    log.info(f"Extracting dietary signals for {len(todo):,} users "
             f"({workers} workers, checkpoint every {CHECKPOINT_EVERY})…")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, uid): uid for uid in todo}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                uid = futures[future]
                log.warning("  Worker error for %s: %s — using fallback", uid, exc)
                row = _fallback()
                row["contributor_id"] = futures[future]
                results.append(row)
            n_done += 1
            if n_done % CHECKPOINT_EVERY == 0:
                _save(results)
                log.info(f"  Checkpoint: {n_done:,} / {len(todo):,} done")

    _save(results)
    log.info(f"Done — {len(results):,} users saved to {OUT_FILE}")

    # Summary stats
    df = pd.read_parquet(OUT_FILE)
    for f in AFFINITY_FIELDS:
        pos = (df[f] > 0.3).mean()
        log.info(f"  {f}: {pos:.1%} users with affinity > 0.3")


def _save(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    # Ensure all columns present and correct dtypes
    df["contributor_id"] = df["contributor_id"].astype(str)
    for f in AFFINITY_FIELDS:
        if f not in df.columns:
            df[f] = 0.0
        df[f] = df[f].astype(np.float32)
    df.to_parquet(OUT_FILE, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-reviews",  type=int, default=3,
                        help="Minimum substantive reviews per user (default: 3)")
    parser.add_argument("--max-reviews",  type=int, default=10,
                        help="Max reviews to send per user (default: 10)")
    parser.add_argument("--workers",      type=int, default=4,
                        help="LLM worker threads (default: 4)")
    parser.add_argument("--limit",        type=int, default=0,
                        help="Process only N users, for dry-run (0=all)")
    parser.add_argument("--no-resume",    action="store_true",
                        help="Ignore existing checkpoint and reprocess all")
    parser.add_argument("--local-url",    default="http://localhost:8082")
    parser.add_argument("--local-model",  default="qwen3.5-9b")
    args = parser.parse_args()

    build(
        min_reviews = args.min_reviews,
        max_reviews = args.max_reviews,
        workers     = args.workers,
        limit       = args.limit,
        no_resume   = args.no_resume,
        url         = args.local_url,
        model       = args.local_model,
    )
