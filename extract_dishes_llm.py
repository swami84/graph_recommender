#!/usr/bin/env python3
"""
extract_dishes_llm.py — LLM-powered dish extraction from review text.

Uses SGLang (fastest for Qwen3.5-35B-A3B MoE: 3–5× faster than vLLM per
github.com/vllm-project/vllm/issues/36215) with native JSON schema constrained
decoding — the model is physically forced to emit a valid JSON array, so no
regex fallback parsing is needed.

── Install SGLang ───────────────────────────────────────────────────────────────
    pip install "sglang[all]>=0.4"

── Recommended setup (Ollama + Unsloth GGUF) ────────────────────────────────────
    # Install Ollama
    curl -fsSL https://ollama.com/install.sh | sh

    # Pull Unsloth's optimised Qwen3.5-27B Q4_K_M (~17 GB, fits comfortably on one RTX 3090)
    ollama pull hf.co/unsloth/Qwen3.5-27B-GGUF:Q4_K_M

    # Start Ollama with parallel request support on GPU 0
    CUDA_VISIBLE_DEVICES=0 OLLAMA_NUM_PARALLEL=20 ollama serve

    Wait until Ollama prints "Listening on 127.0.0.1:11434" before running this script.

── Alternative: SGLang (higher throughput if GPU memory allows) ─────────────────
    python -m sglang.launch_server \
        --model-path Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
        --quantization gptq_marlin --dtype float16 \
        --tp 1 --port 8000 --context-length 4096

── Usage ────────────────────────────────────────────────────────────────────────
    # Ollama (default)
    python extract_dishes_llm.py
    python extract_dishes_llm.py --url http://localhost:11434 \
        --model-name hf.co/unsloth/Qwen3.5-27B-GGUF:Q4_K_M

    # SGLang / vLLM
    python extract_dishes_llm.py --url http://localhost:8000 --model-name default

    python extract_dishes_llm.py --reviews-per-call 15 --concurrency 20
    python extract_dishes_llm.py --merge-meta           # also keep meta dishes
    python extract_dishes_llm.py --resume               # skip already-done restaurants
    python extract_dishes_llm.py --health-only          # verify server is up

── Outputs ─────────────────────────────────────────────────────────────────────
    data/dishes.parquet          unique (dish_name, place_id, dish_id)
    data/review_dishes.parquet   (review_id, dish_name, place_id, source)
"""

import argparse
import asyncio
import hashlib
import itertools
import json
import logging
import re
import time
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm as atqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_dishes")

# ── Paths ──────────────────────────────────────────────────────────────────────
REVIEWS_FLAT    = Path("data/reviews_flat.parquet")
RESTAURANTS_ENR = Path("data/restaurants_enriched.parquet")
OUT_DISHES      = Path("data/dishes.parquet")
OUT_REV_DISHES  = Path("data/review_dishes.parquet")
CHECKPOINT_FILE = Path("data/.llm_dish_checkpoint.json")

# ── JSON schema sent to SGLang ─────────────────────────────────────────────────
# SGLang enforces this at the token level — output is always a valid JSON array
# of strings. No regex fallback needed.
DISHES_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name":   "dishes",
        "schema": DISHES_SCHEMA,
        "strict": True,
    },
}

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a food analyst. Extract specific dish names mentioned in restaurant reviews.\n\n"
    "Rules:\n"
    "- Return ONLY a JSON array of dish name strings.\n"
    "- Include ONLY complete menu items a customer would order: 'spicy tuna roll', "
    "'chicken tikka masala', 'poke bowl', 'miso soup', 'brown sugar boba tea'.\n"
    "- Normalise to lowercase singular form.\n"
    "- EXCLUDE standalone proteins/ingredients that are not dishes by themselves: "
    "'salmon', 'chicken', 'shrimp', 'tuna', 'crab', 'steak', 'tofu', 'avocado', "
    "'rice', 'beef', 'pork', 'octopus', 'albacore'. "
    "Include them only when part of a full dish name: 'spicy tuna roll', 'chicken tikka masala'.\n"
    "- EXCLUDE condiments, toppings, and sauces used alone: 'spicy mayo', 'soy sauce', "
    "'crispy onions', 'seaweed', 'edamame', 'furikake', 'shoyu', 'house dressing', 'spring mix'.\n"
    "- EXCLUDE single generic words with no preparation: 'noodle', 'boba', 'tea', "
    "'poke', 'smoothie', 'sauce', 'dressing', 'soup' alone.\n"
    "- Skip generic words: food, meal, dish, item, everything, something, place.\n"
    "- If no specific dishes are mentioned, return []."
)


def _build_user_message(reviews: list[dict], restaurant_name: str, cuisine: str,
                        max_chars: int = 300) -> str:
    lines = [f"Restaurant: {restaurant_name} ({cuisine})\n"]
    for i, r in enumerate(reviews, 1):
        text = r["text"][:max_chars].replace("\n", " ").strip()
        lines.append(f"Review {i}: {text}")
    lines.append("\nJSON array of dish names:")
    return "\n".join(lines)


def _normalise_dish(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _dish_id(place_id: str, dish_name: str) -> str:
    return hashlib.md5(f"{place_id}::{dish_name}".encode()).hexdigest()[:12]


# ── SGLang async client ────────────────────────────────────────────────────────
def _recover_truncated_json(raw: str) -> list[str]:
    """Extract complete string items from a truncated JSON array like ["a", "b", "inc..."""
    results = []
    # Find all fully-quoted strings before any truncation point
    for m in re.finditer(r'"((?:[^"\\]|\\.)+)"', raw):
        val = m.group(1)
        # Skip if it looks like JSON structural noise
        if len(val) >= 3 and val not in ("dishes", "items", "type", "string", "array"):
            results.append(val)
    return results


class SGLangClient:
    def __init__(self, base_urls: list[str], model_name: str = "default",
                 max_tokens: int = 200, temperature: float = 0.0, timeout: int = 120):
        # Round-robin across all provided URLs (single or dual GPU)
        self._endpoints   = [u.rstrip("/") + "/v1/chat/completions" for u in base_urls]
        self._health_urls = [u.rstrip("/") for u in base_urls]
        self._cycle       = itertools.cycle(self._endpoints)
        self.model_name   = model_name
        self.max_tokens   = max_tokens
        self.temperature  = temperature
        self.timeout      = aiohttp.ClientTimeout(total=timeout)
        # Token throughput tracking (updated in complete())
        self.prompt_tokens:     int = 0
        self.completion_tokens: int = 0

    async def health_check(self, session: aiohttp.ClientSession) -> bool:
        # All configured backends must be reachable
        all_ok = True
        for base in self._health_urls:
            ok = False
            for path in ["/health", ""]:
                try:
                    async with session.get(base + path,
                                           timeout=aiohttp.ClientTimeout(total=5)) as r:
                        if r.status == 200:
                            ok = True
                            break
                except Exception:
                    pass
            if not ok:
                log.error(f"Backend not reachable: {base}")
                all_ok = False
        return all_ok

    async def complete(self, session: aiohttp.ClientSession,
                       user_msg: str, retries: int = 3) -> list[str]:
        """
        Post to the next backend in round-robin rotation.
        Returns a list of dish name strings.
        """
        payload = {
            "model":           self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            "max_tokens":      self.max_tokens,
            "temperature":     self.temperature,
            "response_format": RESPONSE_FORMAT,
            # Disable Qwen3 thinking mode — without this the model burns all
            # tokens on internal reasoning and never outputs the JSON answer.
            "chat_template_kwargs": {"enable_thinking": False},
        }

        url = next(self._cycle)
        for attempt in range(1, retries + 1):
            try:
                async with session.post(url, json=payload,
                                        timeout=self.timeout) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        log.warning(f"HTTP {resp.status} from {url} (attempt {attempt}): {body[:200]}")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    data   = await resp.json()
                    raw    = data["choices"][0]["message"]["content"]
                    finish = data["choices"][0].get("finish_reason", "")
                    # Accumulate token counts for throughput logging
                    usage = data.get("usage") or {}
                    self.prompt_tokens     += usage.get("prompt_tokens", 0)
                    self.completion_tokens += usage.get("completion_tokens", 0)
                    try:
                        dishes = json.loads(raw)
                    except json.JSONDecodeError:
                        if finish == "length":
                            dishes = _recover_truncated_json(raw)
                            log.debug(f"Truncated response salvaged: {len(dishes)} dishes")
                        else:
                            raise
                    return [_normalise_dish(d) for d in dishes
                            if isinstance(d, str) and len(d.strip()) >= 3]
            except asyncio.TimeoutError:
                log.warning(f"Request timed out after {self.timeout.total}s (attempt {attempt})")
                await asyncio.sleep(2 ** attempt)
            except aiohttp.ClientError as e:
                log.warning(f"Request error (attempt {attempt}): {type(e).__name__}: {e}")
                await asyncio.sleep(2 ** attempt)
            except json.JSONDecodeError as e:
                log.warning(f"JSON parse error (attempt {attempt}): {e} — raw: {raw[:120]!r}")
                await asyncio.sleep(2 ** attempt)

        return []


# ── Per-restaurant processing ──────────────────────────────────────────────────
async def _call_batch(
    batch:     list[dict],
    rest_name: str,
    cuisine:   str,
    client:    "SGLangClient",
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> tuple[list[str], list[dict]]:
    """Run one LLM batch, respecting the global semaphore. Returns (dishes, batch)."""
    user_msg = _build_user_message(batch, rest_name, cuisine)
    async with semaphore:
        dishes = await client.complete(session, user_msg)
    return dishes, batch


async def _process_restaurant(
    place_id:        str,
    reviews:         list[dict],
    rest_name:       str,
    cuisine:         str,
    client:          "SGLangClient",
    session:         aiohttp.ClientSession,
    semaphore:       asyncio.Semaphore,
    reviews_per_call: int,
) -> tuple[str, list[dict], list[dict]]:
    # Fire all batches for this restaurant concurrently — the global semaphore
    # limits total in-flight requests across all restaurants.
    batches = [reviews[i : i + reviews_per_call]
               for i in range(0, len(reviews), reviews_per_call)]
    results = await asyncio.gather(*[
        _call_batch(batch, rest_name, cuisine, client, session, semaphore)
        for batch in batches
    ])

    dish_set   = {}   # dish_name → dish_id  (dedup within restaurant)
    rev_dishes = []
    for dishes, batch in results:
        for dish_name in dishes:
            if dish_name not in dish_set:
                dish_set[dish_name] = _dish_id(place_id, dish_name)
            for r in batch:
                rev_dishes.append({
                    "review_id": r["review_id"],
                    "dish_name": dish_name,
                    "place_id":  place_id,
                    "source":    "llm",
                })

    dish_rows = [
        {"dish_name": name, "place_id": place_id, "dish_id": did}
        for name, did in dish_set.items()
    ]
    return place_id, dish_rows, rev_dishes


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def _load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        return set(json.loads(CHECKPOINT_FILE.read_text()))
    return set()


def _save_checkpoint(processed: set[str]):
    CHECKPOINT_FILE.write_text(json.dumps(list(processed)))


def _save_parquets(dish_rows: list[dict], rev_dish_rows: list[dict]):
    if dish_rows:
        (pd.DataFrame(dish_rows)
           .drop_duplicates(["dish_name", "place_id"])
           .reset_index(drop=True)
           .to_parquet(OUT_DISHES, index=False))
    if rev_dish_rows:
        (pd.DataFrame(rev_dish_rows)
           .drop_duplicates(["review_id", "dish_name", "place_id"])
           .reset_index(drop=True)
           .to_parquet(OUT_REV_DISHES, index=False))


# ── Main extraction loop ───────────────────────────────────────────────────────
async def run_extraction(
    base_urls:        list[str],
    model_name:       str,
    reviews_per_call: int,
    concurrency:      int,
    resume:           bool,
    merge_meta:       bool,
):
    if not REVIEWS_FLAT.exists():
        log.error(f"Missing {REVIEWS_FLAT} — run: python build_graph_data.py --step 1b")
        return

    # ── Health check ──────────────────────────────────────────────────────────
    client = SGLangClient(base_urls, model_name=model_name)
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        if not await client.health_check(session):
            return
        log.info(f"LLM backend(s) healthy: {base_urls}")

        # ── Load data ──────────────────────────────────────────────────────────
        log.info("Loading reviews…")
        df      = pd.read_parquet(REVIEWS_FLAT)
        content = df[df["has_content"] == True].copy()
        log.info(f"Content reviews: {len(content):,} / {len(df):,} total")

        rest_map: dict[str, dict] = {}
        if RESTAURANTS_ENR.exists():
            rdf      = pd.read_parquet(RESTAURANTS_ENR)
            rest_map = rdf.set_index("place_id")[["name", "cuisine_category"]].to_dict("index")

        grouped = {
            pid: grp[["review_id", "text"]].to_dict("records")
            for pid, grp in content.groupby("place_id")
        }
        place_ids = list(grouped.keys())
        log.info(f"Restaurants to process: {len(place_ids):,}")

        # ── Resume ────────────────────────────────────────────────────────────
        processed = _load_checkpoint() if resume else set()
        if processed:
            place_ids = [p for p in place_ids if p not in processed]
            log.info(f"Skipping {len(processed):,} already done — "
                     f"{len(place_ids):,} remaining")

        # ── Load previously saved dishes when resuming ────────────────────────
        all_dish_rows: list[dict] = []
        all_rev_rows:  list[dict] = []

        if resume:
            if OUT_DISHES.exists():
                prev = pd.read_parquet(OUT_DISHES)
                all_dish_rows = prev.to_dict("records")
                log.info(f"Loaded {len(all_dish_rows):,} existing dishes from {OUT_DISHES}")
            if OUT_REV_DISHES.exists():
                prev_rev = pd.read_parquet(OUT_REV_DISHES)
                all_rev_rows = prev_rev.to_dict("records")
                log.info(f"Loaded {len(all_rev_rows):,} existing review-dish edges from {OUT_REV_DISHES}")

        # ── Meta dishes (Pass 1) — vectorised, no iterrows ───────────────────

        if merge_meta and "recommended_dishes" in df.columns:
            log.info("Loading meta dishes (vectorised)…")
            meta = df[["review_id", "place_id", "recommended_dishes"]].copy()
            meta = meta[meta["recommended_dishes"].notna() &
                        (meta["recommended_dishes"].str.strip() != "")]
            meta = meta.assign(
                dish_list=meta["recommended_dishes"].str.split(",")
            ).explode("dish_list")
            meta["dish_name"] = (meta["dish_list"]
                                 .str.strip().str.lower()
                                 .str.replace(r"\s+", " ", regex=True))
            meta = meta[meta["dish_name"].str.len() >= 3].copy()
            meta["dish_id"] = (meta["place_id"] + "::" + meta["dish_name"]).apply(
                lambda x: hashlib.md5(x.encode()).hexdigest()[:12]
            )
            all_dish_rows = meta[["dish_name", "place_id", "dish_id"]].to_dict("records")
            all_rev_rows  = (meta[["review_id", "dish_name", "place_id"]]
                             .assign(source="meta").to_dict("records"))
            log.info(f"Meta dishes loaded: {len(all_dish_rows):,} mentions")

        # ── LLM extraction ────────────────────────────────────────────────────
        semaphore     = asyncio.Semaphore(concurrency)
        start_time    = time.time()
        done_count    = 0
        review_count  = 0

        futures = [
            asyncio.ensure_future(
                _process_restaurant(
                    place_id=pid,
                    reviews=grouped[pid],
                    rest_name=rest_map.get(pid, {}).get("name", ""),
                    cuisine=rest_map.get(pid, {}).get("cuisine_category", ""),
                    client=client,
                    session=session,
                    semaphore=semaphore,
                    reviews_per_call=reviews_per_call,
                )
            )
            for pid in place_ids
        ]

        log.info(f"Starting LLM extraction: {len(futures):,} restaurants, "
                 f"concurrency={concurrency}, {reviews_per_call} reviews/call")

        pbar = atqdm(asyncio.as_completed(futures), total=len(futures),
                     desc="Restaurants", unit="rest")
        async for coro in pbar:
            pid, dish_rows, rev_dish_rows = await coro
            all_dish_rows.extend(dish_rows)
            all_rev_rows.extend(rev_dish_rows)
            processed.add(pid)
            done_count   += 1
            review_count += len(grouped.get(pid, []))

            # ── Every 50: show dishes + checkpoint ───────────────────────────
            if done_count % 50 == 0:
                elapsed   = time.time() - start_time
                rate      = done_count / elapsed * 60
                tok_s     = client.completion_tokens / elapsed
                total_tok = client.prompt_tokens + client.completion_tokens
                avg_s     = elapsed / done_count
                rest_name = rest_map.get(pid, {}).get("name", pid)
                dishes_found = [r["dish_name"] for r in dish_rows]
                _save_parquets(all_dish_rows, all_rev_rows)
                _save_checkpoint(processed)
                log.info(
                    f"[{done_count:,}/{len(place_ids):,} rest | "
                    f"{review_count:,} reviews | "
                    f"{rate:.0f} rest/min | {avg_s:.1f}s/rest | {tok_s:.0f} tok/s] "
                    f"{rest_name}: {dishes_found or '(none)'}"
                )
                log.info(
                    f"  dishes so far: {len(all_dish_rows):,} | "
                    f"total tokens: {total_tok:,}"
                )

    # ── Final save ─────────────────────────────────────────────────────────────
    _save_parquets(all_dish_rows, all_rev_rows)
    _save_checkpoint(processed)
    elapsed_total = time.time() - start_time

    dishes_df = pd.read_parquet(OUT_DISHES) if OUT_DISHES.exists() else pd.DataFrame()
    rev_df    = pd.read_parquet(OUT_REV_DISHES) if OUT_REV_DISHES.exists() else pd.DataFrame()
    avg_tok_s = client.completion_tokens / elapsed_total if elapsed_total > 0 else 0
    log.info(f"Done in {elapsed_total/60:.1f} min | avg {avg_tok_s:.0f} tok/s out | "
             f"{client.prompt_tokens + client.completion_tokens:,} total tokens")
    log.info(f"Saved {len(dishes_df):,} unique dishes → {OUT_DISHES}")
    log.info(f"Saved {len(rev_df):,} review-dish edges → {OUT_REV_DISHES}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract dish names from reviews using SGLang + Qwen3.5-35B-A3B"
    )
    parser.add_argument("--url", default="http://localhost:8082",
                        help="Primary server URL (default: http://localhost:8082)")
    parser.add_argument("--url2", default=None,
                        help="Optional second server URL for dual-GPU round-robin "
                             "(e.g. http://localhost:8083)")
    parser.add_argument("--model-name",
                        default="qwen3.5-9b",
                        help="Model name to pass in API requests")
    parser.add_argument("--reviews-per-call", type=int, default=30,
                        help="Reviews per LLM call (default: 30)")
    parser.add_argument("--concurrency", type=int, default=40,
                        help="Max concurrent requests (default: 40)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip restaurants already in checkpoint")
    parser.add_argument("--merge-meta", action="store_true", default=True,
                        help="Include meta['Recommended dishes'] in output (default: on)")
    parser.add_argument("--no-merge-meta", dest="merge_meta", action="store_false")
    parser.add_argument("--health-only", action="store_true",
                        help="Check server health and exit")
    args = parser.parse_args()

    base_urls = [args.url] + ([args.url2] if args.url2 else [])

    if args.health_only:
        async def _check():
            async with aiohttp.ClientSession() as s:
                ok = await SGLangClient(base_urls, args.model_name).health_check(s)
                print("healthy" if ok else "unreachable")
        asyncio.run(_check())
        return

    asyncio.run(run_extraction(
        base_urls=base_urls,
        model_name=args.model_name,
        reviews_per_call=args.reviews_per_call,
        concurrency=args.concurrency,
        resume=args.resume,
        merge_meta=args.merge_meta,
    ))


if __name__ == "__main__":
    main()
