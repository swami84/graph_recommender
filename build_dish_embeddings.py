#!/usr/bin/env python3
"""
build_dish_embeddings.py — Dish similarity via local LLM structured descriptors.

For each unique dish name, calls a local vLLM endpoint to generate a structured
flavor profile (category, cuisine, protein, cooking method, flavor tags). These
profiles are one-hot/multi-hot encoded into feature vectors and cosine similarity
is computed between all pairs. Only pairs above a threshold are stored (sparse).

Uses the local chat LLM (no embedding model needed) — structured descriptors
give interpretable, semantically meaningful similarity.

Outputs:
  data/dish_profiles.parquet   — (dish_name, feature vector columns)
  data/dish_similarities.parquet — (dish_a, dish_b, similarity)  [sparse, sim > threshold]

Usage:
    python build_dish_embeddings.py
    python build_dish_embeddings.py --local-model cyankiwi/Qwen3.5-9B-AWQ-BF16-INT8
    python build_dish_embeddings.py --workers 8 --threshold 0.5
    python build_dish_embeddings.py --limit 500   # dry-run on 500 dishes
    python build_dish_embeddings.py --no-resume   # reprocess everything
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
log = logging.getLogger("dish_emb")

DISHES_FILE       = Path("data/dishes.parquet")
PROFILES_FILE     = Path("data/dish_profiles.parquet")
SIMILARITIES_FILE = Path("data/dish_similarities.parquet")

CHECKPOINT_EVERY = 500

# ── Vocabulary for structured descriptor encoding ──────────────────────────────
CATEGORIES = [
    "noodles", "rice", "meat", "seafood", "soup", "salad", "sandwich",
    "pizza", "pasta", "burger", "tacos", "sushi", "dumpling", "bread",
    "appetizer", "dessert", "drink", "other",
]
CUISINES = [
    "american", "mexican", "italian", "chinese", "japanese", "korean",
    "indian", "thai", "mediterranean", "french", "vietnamese", "cajun",
    "greek", "middle_eastern", "caribbean", "other",
]
PROTEINS = [
    "chicken", "beef", "pork", "seafood", "shrimp", "lamb", "tofu",
    "egg", "mixed", "none",
]
METHODS = [
    "grilled", "fried", "braised", "baked", "steamed", "raw", "smoked",
    "roasted", "sauteed", "other",
]
FLAVORS = [
    "spicy", "mild", "sweet", "savory", "umami", "sour", "rich",
    "light", "creamy", "smoky", "tangy", "herby",
]

ALL_VOCAB = {
    "category":       CATEGORIES,
    "cuisine":        CUISINES,
    "protein":        PROTEINS,
    "cooking_method": METHODS,
    "flavor":         FLAVORS,    # multi-hot
}

FEAT_DIM = sum(len(v) for v in ALL_VOCAB.values())

_SYSTEM = (
    "You are a food taxonomy expert. "
    "Given a dish name, return a JSON object describing it. "
    "Return ONLY valid JSON — no explanation, no markdown."
)


def _call_local(prompt: str, url: str, model: str, timeout: int = 30) -> str:
    payload = json.dumps({
        "model":    model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens":  150,
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
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            r = json.loads(m.group(0))
            if isinstance(r, dict):
                return r
        except json.JSONDecodeError:
            pass
    return None


def _describe_dish(dish_name: str, url: str, model: str,
                   max_retries: int = 2) -> dict | None:
    prompt = (
        f'Dish: "{dish_name}"\n\n'
        "Describe this dish with these fields:\n"
        f"  category: one of {CATEGORIES}\n"
        f"  cuisine: one of {CUISINES}\n"
        f"  protein: one of {PROTEINS}\n"
        f"  cooking_method: one of {METHODS}\n"
        f'  flavor: list of 1-3 from {FLAVORS}\n\n'
        "JSON only:"
    )
    for attempt in range(max_retries):
        try:
            raw    = _call_local(prompt, url, model)
            parsed = _extract_json(raw)
            if parsed:
                return parsed
            log.debug(f"  JSON parse failed for '{dish_name}': {raw[:80]}")
        except Exception as exc:
            log.debug(f"  LLM error for '{dish_name}', attempt {attempt+1}: {exc}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return None


# ── Feature encoding ───────────────────────────────────────────────────────────
def _build_vocab_index() -> dict[str, dict[str, int]]:
    """Map each vocab value to its index in the feature vector."""
    index: dict[str, dict[str, int]] = {}
    offset = 0
    for field, values in ALL_VOCAB.items():
        index[field] = {v: offset + i for i, v in enumerate(values)}
        offset += len(values)
    return index


VOCAB_INDEX = _build_vocab_index()


def descriptor_to_vector(desc: dict) -> np.ndarray:
    """Encode a dish descriptor dict to a float32 feature vector of length FEAT_DIM."""
    vec = np.zeros(FEAT_DIM, dtype=np.float32)
    for field, val_index in VOCAB_INDEX.items():
        raw = desc.get(field)
        if raw is None:
            continue
        if isinstance(raw, list):
            # multi-hot (flavor)
            for v in raw:
                v = str(v).strip().lower().replace(" ", "_")
                if v in val_index:
                    vec[val_index[v]] = 1.0
        else:
            v = str(raw).strip().lower().replace(" ", "_")
            if v in val_index:
                vec[val_index[v]] = 1.0
    return vec


def _fallback_descriptor(dish_name: str) -> dict:
    """Heuristic fallback when LLM fails."""
    name = dish_name.lower()
    category = "other"
    for kw, cat in [("ramen", "noodles"), ("pho", "noodles"), ("pasta", "pasta"),
                    ("pizza", "pizza"), ("burger", "burger"), ("taco", "tacos"),
                    ("sushi", "sushi"), ("sandwich", "sandwich"), ("salad", "salad"),
                    ("soup", "soup"), ("rice", "rice"), ("dumpling", "dumpling"),
                    ("cake", "dessert"), ("ice cream", "dessert"), ("coffee", "drink")]:
        if kw in name:
            category = cat
            break
    return {"category": category, "cuisine": "other", "protein": "none",
            "cooking_method": "other", "flavor": ["savory"]}


# ── Main pipeline ──────────────────────────────────────────────────────────────
def build_dish_profiles(
    local_url:   str   = "http://localhost:8082",
    local_model: str   = "qwen3.5-9b",
    workers:     int   = 4,
    resume:      bool  = True,
    limit:       int   = 0,
) -> pd.DataFrame:

    log.info("Loading dishes…")
    dishes = pd.read_parquet(DISHES_FILE)
    # Normalise names and deduplicate
    dishes["dish_norm"] = dishes["dish_name"].str.lower().str.strip()
    unique_dishes = sorted(dishes["dish_norm"].unique())
    log.info(f"Unique dish names: {len(unique_dishes):,}")

    # Resume
    done: set = set()
    existing_rows: list[dict] = []
    if resume and PROFILES_FILE.exists():
        existing = pd.read_parquet(PROFILES_FILE)
        done = set(existing["dish_name"].tolist())
        existing_rows = existing.to_dict("records")
        log.info(f"Resuming — {len(done):,} dishes already processed")

    todo = [d for d in unique_dishes if d not in done]
    if limit > 0:
        todo = todo[:limit]
        log.info(f"Limit mode: processing {limit} dishes")

    log.info(f"To process: {len(todo):,} dishes  |  workers={workers}")

    results: list[dict] = list(existing_rows)
    failed = 0

    def _process(dish_name: str) -> dict:
        desc = _describe_dish(dish_name, local_url, local_model)
        if desc is None:
            desc = _fallback_descriptor(dish_name)
            nonlocal failed
            failed += 1
        vec = descriptor_to_vector(desc)
        row = {"dish_name": dish_name}
        for i, v in enumerate(vec):
            row[f"f{i}"] = float(v)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, d): d for d in todo}
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())

            if i % CHECKPOINT_EVERY == 0:
                _save_profiles(results)
                pct = 100 * i / len(todo)
                log.info(f"  [{i:,}/{len(todo):,}] {pct:.1f}%  — checkpoint saved")

    _save_profiles(results)
    log.info(f"Profiles done. {len(results):,} dishes  |  fallbacks used: {failed:,}")
    return pd.read_parquet(PROFILES_FILE)


def _save_profiles(rows: list[dict]):
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["dish_name"], keep="last")
    df.to_parquet(PROFILES_FILE, index=False)


def build_similarities(threshold: float = 0.6, top_k: int = 20) -> pd.DataFrame:
    """
    GPU-accelerated cosine similarity with top-K bounding.

    For each dish keeps at most top_k neighbours with similarity >= threshold.
    This bounds output to at most n * top_k / 2 pairs regardless of threshold,
    preventing the RAM explosion that occurs with a pure all-pairs approach on
    binary vectors where low thresholds match billions of pairs.
    """
    import torch

    if not PROFILES_FILE.exists():
        raise FileNotFoundError("Run build_dish_profiles() first.")

    log.info("Loading dish profiles…")
    profiles  = pd.read_parquet(PROFILES_FILE)
    feat_cols = [c for c in profiles.columns if c.startswith("f")]
    mat       = profiles[feat_cols].values.astype(np.float32)

    # L2-normalise for cosine similarity
    norms    = np.linalg.norm(mat, axis=1, keepdims=True)
    mat_norm = (mat / np.maximum(norms, 1e-8)).astype(np.float32)

    dish_names = profiles["dish_name"].tolist()
    n          = len(dish_names)

    use_gpu = torch.cuda.is_available()
    device  = torch.device("cuda" if use_gpu else "cpu")
    dtype   = torch.float16 if use_gpu else torch.float32
    # float16 on GPU: 186K × 50 × 2 bytes = ~18 MB — trivial
    mat_t   = torch.tensor(mat_norm, dtype=dtype, device=device)

    # Chunk size: each chunk produces (CHUNK × n) similarity scores
    # float16: 4096 × 186K × 2 bytes ≈ 1.5 GB — fits comfortably on 24 GB 3090
    CHUNK   = 4096 if use_gpu else 256
    k_fetch = min(top_k + 1, n)   # +1 because self-similarity is always 1.0

    log.info(
        f"Computing similarities: {n:,} dishes | "
        f"device={'GPU' if use_gpu else 'CPU'} | "
        f"top_k={top_k} | threshold={threshold}"
    )

    dish_a_list, dish_b_list, sim_list = [], [], []

    for start in range(0, n, CHUNK):
        end   = min(start + CHUNK, n)
        chunk = mat_t[start:end]                         # (chunk, feat_dim)
        sims  = (chunk @ mat_t.T).float()                # (chunk, n) float32

        # Zero out self-similarity so topk never picks self
        for local_i in range(end - start):
            sims[local_i, start + local_i] = -1.0

        top_vals, top_idx = sims.topk(k_fetch, dim=1)   # (chunk, k_fetch)

        # Threshold mask
        mask = top_vals >= threshold                      # (chunk, k_fetch)

        # Vectorised extraction — no Python loop over individual pairs
        local_rows, k_pos = mask.nonzero(as_tuple=True)
        if local_rows.numel() == 0:
            continue

        global_i_arr = (local_rows + start).cpu().numpy()
        j_arr        = top_idx[local_rows, k_pos].cpu().numpy()
        v_arr        = top_vals[local_rows, k_pos].cpu().numpy()

        # Keep only upper triangle (i < j) to avoid duplicate pairs;
        # adj_dd in recommendation_gnn.py symmetrises when building the graph
        keep = global_i_arr < j_arr
        dish_a_list.append(np.array(dish_names)[global_i_arr[keep]])
        dish_b_list.append(np.array(dish_names)[j_arr[keep]])
        sim_list.append(v_arr[keep].astype(np.float32))

        if start % (CHUNK * 10) == 0:
            total_so_far = sum(len(x) for x in sim_list)
            log.info(f"  {end:,}/{n:,} rows done | pairs so far: {total_so_far:,}")

    sim_df = pd.DataFrame({
        "dish_a":     np.concatenate(dish_a_list) if dish_a_list else [],
        "dish_b":     np.concatenate(dish_b_list) if dish_b_list else [],
        "similarity": np.concatenate(sim_list)    if sim_list    else [],
    })
    sim_df.to_parquet(SIMILARITIES_FILE, index=False)
    log.info(f"Saved {len(sim_df):,} pairs → {SIMILARITIES_FILE}")
    if len(sim_df):
        log.info(f"  similarity stats: {sim_df['similarity'].describe().round(3).to_dict()}")
    return sim_df


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build dish embeddings via local LLM")
    parser.add_argument("--local-url",   default="http://localhost:8082")
    parser.add_argument("--local-model", default="qwen3.5-9b")
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--threshold",   type=float, default=0.6,
                        help="Min cosine similarity to keep a pair (default 0.6)")
    parser.add_argument("--top-k",       type=int,   default=20,
                        help="Max neighbours per dish (default 20); bounds output size")
    parser.add_argument("--resume",      action="store_true", default=True)
    parser.add_argument("--no-resume",   action="store_false", dest="resume")
    parser.add_argument("--limit",       type=int, default=0,
                        help="Process only first N unique dishes (0=all)")
    parser.add_argument("--skip-similarity", action="store_true",
                        help="Skip similarity computation (just build profiles)")
    args = parser.parse_args()

    profiles = build_dish_profiles(
        local_url=args.local_url,
        local_model=args.local_model,
        workers=args.workers,
        resume=args.resume,
        limit=args.limit,
    )
    print(f"\nProfiles saved: {len(profiles):,} dishes → {PROFILES_FILE}")

    if not args.skip_similarity:
        sim_df = build_similarities(threshold=args.threshold, top_k=args.top_k)
        print(f"Similarities saved: {len(sim_df):,} pairs → {SIMILARITIES_FILE}")
        if len(sim_df):
            print(sim_df["similarity"].describe().round(3).to_string())


if __name__ == "__main__":
    main()
