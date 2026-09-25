#!/usr/bin/env python3
"""High-precision v4 refinement of v3 unknown restaurant cuisines.

Only v3 ``unknown``/``other`` rows are reconsidered. A single cuisine supported
by explicit name, Google-type, or review evidence is accepted deterministically.
Rows with multiple supported candidates are sent to a constrained 27B adjudicator.
Rows with no evidence remain unknown. The completed v3 artifact is never modified.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

from foodie.features.classify_other_cuisines_27b import CUISINE_EVIDENCE, TYPE_TO_CUISINE


DATA = Path("data")
VERSION = "unknown-cuisine-refinement-qwen27b-v4-20260909"
SOURCE = DATA / "restaurant_cuisine_27b_v3.parquet"
OUTPUT = DATA / "restaurant_cuisine_27b_v4.parquet"
CHECKPOINT = DATA / "llm_checkpoints" / f"restaurant_{VERSION}.jsonl"

# Dietary styles are modeled separately and cannot be primary cuisines here.
EXCLUDED_DIETARY = {"vegan", "vegetarian"}
ALLOWED_CUISINES = sorted(set(CUISINE_EVIDENCE) - EXCLUDED_DIETARY)


def type_tokens(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return [token.strip() for token in value.split(",") if token.strip()]


def evidence_candidates(row: dict[str, Any]) -> dict[str, list[str]]:
    """Return cuisine -> supporting source names using precision-first rules."""
    hits: dict[str, set[str]] = {}
    for token in type_tokens(row.get("types")):
        cuisine = TYPE_TO_CUISINE.get(token)
        if cuisine in ALLOWED_CUISINES:
            hits.setdefault(cuisine, set()).add("google_types")

    fields = {
        "name": str(row.get("name") or ""),
        "reviews": str(row.get("corpus") or ""),
    }
    for cuisine in ALLOWED_CUISINES:
        pattern = CUISINE_EVIDENCE[cuisine]
        for source, text in fields.items():
            if re.search(pattern, text, re.I):
                hits.setdefault(cuisine, set()).add(source)
    return {key: sorted(value) for key, value in sorted(hits.items())}


def load_llm_completed() -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not CHECKPOINT.exists():
        return completed
    for line in CHECKPOINT.read_text().splitlines():
        try:
            row = json.loads(line)
            if row.get("version") == VERSION and row.get("status") == "ok":
                completed[str(row["place_id"])] = row
        except (json.JSONDecodeError, KeyError):
            continue
    return completed


def output_frame(base: pd.DataFrame, refinements: dict[str, dict[str, Any]]) -> pd.DataFrame:
    frame = base.copy()
    frame["version"] = VERSION
    unresolved = frame["primary_cuisine"].astype(str).str.lower().isin({"unknown", "other"})
    frame.loc[unresolved, "primary_cuisine"] = "unknown"
    frame.loc[unresolved, "classification_method"] = "retained_unknown_no_evidence"
    frame.loc[unresolved, "confidence"] = 0.0
    frame.loc[unresolved, "evidence_sources_json"] = "[]"
    by_id = frame.set_index("place_id")
    for place_id, row in refinements.items():
        if place_id not in by_id.index:
            continue
        for col in ("version", "llm_model", "source_review_count", "primary_cuisine",
                    "confidence", "evidence_sources_json", "classification_method"):
            by_id.loc[place_id, col] = row[col]
    return by_id.reset_index()[base.columns]


def write_output(base: pd.DataFrame, refinements: dict[str, dict[str, Any]]) -> None:
    frame = output_frame(base, refinements)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, OUTPUT)


def deterministic_row(source: dict[str, Any], cuisine: str,
                      sources: list[str]) -> dict[str, Any]:
    return {
        "place_id": str(source["place_id"]), "version": VERSION,
        "llm_model": "none:high-precision-evidence-v4",
        "source_review_count": int(source.get("source_review_count", 0)),
        "primary_cuisine": cuisine, "confidence": 1.0,
        "evidence_sources_json": json.dumps(sources),
        "classification_method": "evidence_rule_v4", "status": "ok",
    }


def schema_for(candidates: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "primary_cuisine": {"type": "string", "enum": candidates + ["unknown"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_sources": {
                "type": "array",
                "items": {"type": "string", "enum": ["name", "google_types", "reviews"]},
                "uniqueItems": True, "maxItems": 3,
            },
        },
        "required": ["primary_cuisine", "confidence", "evidence_sources"],
        "additionalProperties": False,
    }


SYSTEM = (
    "Resolve a restaurant's primary culinary tradition from a short candidate list. "
    "The candidates were found by explicit lexical evidence but some mentions may be incidental. "
    "Prefer an explicit restaurant identity over an isolated dish, and prefer the most specific "
    "supported tradition over an umbrella category. Vegan and vegetarian are dietary attributes, "
    "not cuisines. Choose unknown only when the evidence is genuinely incidental, contradictory, "
    "or insufficient. Return schema-valid JSON only."
)


def prompt(row: dict[str, Any]) -> str:
    reviews = str(row.get("corpus") or "").strip() or "No usable review text."
    evidence = json.loads(row["candidate_evidence_json"])
    return (
        f"Restaurant: {row.get('name') or 'Unknown'}\n"
        f"Google types: {row.get('types') or 'Unknown'}\n"
        f"Candidate evidence: {json.dumps(evidence, ensure_ascii=False)}\n"
        f"Training-only reviews (up to eight):\n{reviews}\n\n"
        "Choose the single best-supported primary cuisine from the candidates, or unknown."
    )


async def main_async(args: argparse.Namespace) -> None:
    base = pd.read_parquet(SOURCE)
    restaurants = pd.read_parquet(
        DATA / "restaurants_enriched.parquet", columns=["place_id", "name", "types"]
    )
    corpora = pd.read_parquet(
        DATA / "llm_corpora/restaurant_train_corpus.parquet",
        columns=["place_id", "corpus", "source_review_count"],
    )
    unresolved_ids = base.loc[
        base["primary_cuisine"].astype(str).str.lower().isin({"unknown", "other"}), "place_id"
    ].astype(str)
    work = (pd.DataFrame({"place_id": unresolved_ids})
            .merge(restaurants, on="place_id", how="left", validate="one_to_one")
            .merge(corpora, on="place_id", how="left", validate="one_to_one"))
    work["source_review_count"] = work["source_review_count"].fillna(0).astype(int)
    work["candidate_evidence"] = [evidence_candidates(row) for row in work.to_dict("records")]
    work["candidate_evidence_json"] = work["candidate_evidence"].map(json.dumps)
    work["candidate_count"] = work["candidate_evidence"].map(len)

    refinements: dict[str, dict[str, Any]] = {}
    unique = work[work["candidate_count"] == 1]
    for row in unique.to_dict("records"):
        cuisine, sources = next(iter(row["candidate_evidence"].items()))
        refinements[str(row["place_id"])] = deterministic_row(row, cuisine, sources)

    llm_completed = load_llm_completed()
    refinements.update(llm_completed)
    ambiguous = work[work["candidate_count"] > 1]
    todo = ambiguous[~ambiguous["place_id"].astype(str).isin(llm_completed)]
    print(
        f"v4 unresolved input={len(work):,} | no evidence={(work.candidate_count == 0).sum():,} | "
        f"single-candidate auto={len(unique):,} | multi-candidate={len(ambiguous):,} | "
        f"LLM already done={len(llm_completed):,} | LLM todo={len(todo):,}", flush=True,
    )
    write_output(base, refinements)
    if args.deterministic_only or todo.empty:
        return

    urls = [url.rstrip("/") + "/api/chat" for url in args.urls.split(",") if url.strip()]
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    counter = 0
    lock = asyncio.Lock()
    started = time.monotonic()
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def classify(source: dict[str, Any]) -> dict[str, Any]:
            nonlocal counter
            candidates = sorted(source["candidate_evidence"])
            error = "unknown"
            for attempt in range(3):
                async with lock:
                    url = urls[counter % len(urls)]
                    counter += 1
                body = {
                    "model": args.model, "stream": False, "think": False,
                    "keep_alive": "30m", "format": schema_for(candidates),
                    "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 100},
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt(source)},
                    ],
                }
                try:
                    async with semaphore:
                        async with session.post(url, json=body) as response:
                            payload = await response.json()
                            if response.status != 200:
                                raise RuntimeError(f"HTTP {response.status}: {payload}")
                    raw = json.loads(payload["message"]["content"])
                    chosen = raw["primary_cuisine"]
                    if chosen not in candidates + ["unknown"]:
                        raise ValueError("output outside constrained candidates")
                    sources = raw["evidence_sources"]
                    if chosen != "unknown" and not sources:
                        sources = source["candidate_evidence"][chosen]
                    return {
                        "place_id": str(source["place_id"]), "version": VERSION,
                        "llm_model": args.model,
                        "source_review_count": int(source["source_review_count"]),
                        "primary_cuisine": chosen,
                        "confidence": float(raw["confidence"]),
                        "evidence_sources_json": json.dumps(sources),
                        "classification_method": "llm_27b_candidate_adjudication_v4",
                        "status": "ok",
                    }
                except Exception as exc:
                    error = str(exc)
                    await asyncio.sleep(1.5 * (attempt + 1))
            return {"place_id": str(source["place_id"]), "version": VERSION,
                    "llm_model": args.model, "status": "error", "error": error}

        tasks = [asyncio.create_task(classify(row)) for row in todo.to_dict("records")]
        with CHECKPOINT.open("a", buffering=1) as handle:
            failures = 0
            for done, future in enumerate(asyncio.as_completed(tasks), 1):
                result = await future
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                if result["status"] == "ok":
                    refinements[result["place_id"]] = result
                else:
                    failures += 1
                if done % args.log_every == 0 or done == len(tasks):
                    handle.flush()
                    os.fsync(handle.fileno())
                    write_output(base, refinements)
                    rate = done * 60 / max(time.monotonic() - started, 1e-6)
                    print(f"[{done:,}/{len(tasks):,}] {rate:.1f}/min failures={failures}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3.8-27b-24k:latest")
    parser.add_argument("--urls", default="http://127.0.0.1:11436,http://127.0.0.1:11437")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--deterministic-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
