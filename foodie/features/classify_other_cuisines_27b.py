#!/usr/bin/env python3
"""Repair restaurants whose canonical cuisine is ``Other``.

Explicit Google cuisine types are mapped deterministically. Only records that
remain ambiguous are sent to the 27B model, which produces one mutually
exclusive primary cuisine or ``unknown``. Existing v5 features are untouched.
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


DATA = Path("data")
VERSION = "other-cuisine-qwen27b-v3-20260909"
CHECKPOINT = DATA / "llm_checkpoints" / f"restaurant_{VERSION}.jsonl"
OUTPUT = DATA / "restaurant_cuisine_27b_v3.parquet"
CUISINES = [
    "american", "southern", "cajun_creole", "bbq", "steakhouse", "burgers",
    "pizza", "italian", "mexican", "latin_american", "chinese", "japanese",
    "korean", "thai", "vietnamese", "indian", "mediterranean",
    "middle_eastern", "african", "caribbean", "seafood",
    "vegan", "vegetarian", "cafe_bakery", "desserts", "bar_pub",
    "french", "spanish", "portuguese", "european", "eastern_european",
    "british_irish", "filipino", "taiwanese", "southeast_asian",
    "south_american", "ethiopian", "fusion", "other", "unknown",
]

# Only cuisine-bearing types belong here. Meal format, dietary certification,
# venue, and service types (breakfast, halal, cafeteria, food court, etc.) are
# deliberately excluded because they cannot establish cuisine.
TYPE_TO_CUISINE = {
    "american_restaurant": "american", "southern_us_restaurant": "southern",
    "soul_food_restaurant": "southern", "cajun_restaurant": "cajun_creole",
    "creole_restaurant": "cajun_creole", "barbecue_restaurant": "bbq",
    "bbq_restaurant": "bbq", "steak_house": "steakhouse",
    "hamburger_restaurant": "burgers", "pizza_restaurant": "pizza",
    "pizza_delivery": "pizza", "italian_restaurant": "italian",
    "mexican_restaurant": "mexican", "tex_mex_restaurant": "mexican",
    "taco_restaurant": "mexican", "burrito_restaurant": "mexican",
    "latin_american_restaurant": "latin_american",
    "salvadoran_restaurant": "latin_american", "cuban_restaurant": "caribbean",
    "caribbean_restaurant": "caribbean", "south_american_restaurant": "south_american",
    "argentinian_restaurant": "south_american", "brazilian_restaurant": "south_american",
    "peruvian_restaurant": "south_american", "colombian_restaurant": "south_american",
    "chilean_restaurant": "south_american", "chinese_restaurant": "chinese",
    "chinese_noodle_restaurant": "chinese", "dim_sum_restaurant": "chinese",
    "cantonese_restaurant": "chinese", "dumpling_restaurant": "chinese",
    "hot_pot_restaurant": "chinese", "japanese_restaurant": "japanese",
    "sushi_restaurant": "japanese", "ramen_restaurant": "japanese",
    "korean_restaurant": "korean", "thai_restaurant": "thai",
    "vietnamese_restaurant": "vietnamese", "indian_restaurant": "indian",
    "pakistani_restaurant": "indian", "bangladeshi_restaurant": "indian",
    "sri_lankan_restaurant": "indian", "mediterranean_restaurant": "mediterranean",
    "greek_restaurant": "mediterranean", "gyro_restaurant": "mediterranean",
    "middle_eastern_restaurant": "middle_eastern", "turkish_restaurant": "middle_eastern",
    "lebanese_restaurant": "middle_eastern", "israeli_restaurant": "middle_eastern",
    "persian_restaurant": "middle_eastern", "afghani_restaurant": "middle_eastern",
    "kebab_shop": "middle_eastern", "falafel_restaurant": "middle_eastern",
    "african_restaurant": "african", "moroccan_restaurant": "african",
    "ethiopian_restaurant": "ethiopian", "seafood_restaurant": "seafood",
    "fish_and_chips_restaurant": "seafood", "oyster_bar_restaurant": "seafood",
    # Vegan/vegetarian/halal are dietary styles, not geographic cuisines. They
    # are audited separately and must not hide an identifiable cuisine.
    "cafe": "cafe_bakery", "coffee_shop": "cafe_bakery", "bakery": "cafe_bakery",
    "dessert_restaurant": "desserts", "dessert_shop": "desserts",
    "ice_cream_shop": "desserts", "donut_shop": "desserts",
    "bar": "bar_pub", "pub": "bar_pub", "brewery": "bar_pub",
    "french_restaurant": "french", "spanish_restaurant": "spanish",
    "tapas_restaurant": "spanish", "basque_restaurant": "spanish",
    "portuguese_restaurant": "portuguese", "european_restaurant": "european",
    "german_restaurant": "european", "austrian_restaurant": "european",
    "belgian_restaurant": "european", "swiss_restaurant": "european",
    "eastern_european_restaurant": "eastern_european",
    "polish_restaurant": "eastern_european", "russian_restaurant": "eastern_european",
    "ukrainian_restaurant": "eastern_european", "czech_restaurant": "eastern_european",
    "romanian_restaurant": "eastern_european", "hungarian_restaurant": "eastern_european",
    "british_restaurant": "british_irish", "irish_restaurant": "british_irish",
    "scandinavian_restaurant": "european", "filipino_restaurant": "filipino",
    "philippine_restaurant": "filipino", "taiwanese_restaurant": "taiwanese",
    "burmese_restaurant": "southeast_asian", "cambodian_restaurant": "southeast_asian",
    "indonesian_restaurant": "southeast_asian", "malaysian_restaurant": "southeast_asian",
    "fusion_restaurant": "fusion",
}

NAME_RULES = [
    (re.compile(r"\b(?:sushi|sashimi|ramen)\b", re.I), "japanese"),
    (re.compile(r"\b(?:tacos?|taqueria)\b", re.I), "mexican"),
    (re.compile(r"\b(?:pizza|pizzeria)\b", re.I), "pizza"),
]

# Used only to reconcile the rare case where the model selects a cuisine but
# omits its evidence_sources metadata. A label is retained only when one of
# these high-precision anchors is actually present in the supplied source.
CUISINE_EVIDENCE = {
    "american": r"\bamerican (?:food|cuisine|restaurant)\b",
    "southern": r"\bsouthern (?:food|cuisine|cooking)\b|\bsoul food\b",
    "cajun_creole": r"\bcajun\b|\bcreole\b",
    "bbq": r"\bbbq\b|\bbarbecue\b|\bbar-b-q\b",
    "steakhouse": r"\bsteakhouse\b|\bsteak house\b",
    "burgers": r"\bburgers?\b",
    "pizza": r"\bpizza\b|\bpizzeria\b",
    "italian": r"\bitalian\b",
    "mexican": r"\bmexican\b|\btaqueria\b|\btacos?\b|\bburritos?\b|\bmole\b",
    "latin_american": r"\blatin(?:o|a)?\b|\bpupusas?\b|\bsalvadoran\b",
    "chinese": r"\bchinese\b|\bdim sum\b|\bcantonese\b|\bszechuan\b|\bsichuan\b",
    "japanese": r"\bjapanese\b|\bsushi\b|\bsashimi\b|\bramen\b",
    "korean": r"\bkorean\b|\bkimchi\b|\bbulgogi\b|\bbibimbap\b",
    "thai": r"\bthai\b|\bpad thai\b",
    "vietnamese": r"\bvietnamese\b|\bpho\b|\bbanh mi\b",
    "indian": r"\bindian\b|\bdosa\b|\bvada\b|\bpakora\b|\bbiryani\b|\bmasala\b",
    "mediterranean": r"\bmediterranean\b|\bgreek\b",
    "middle_eastern": r"\bmiddle eastern\b|\bturkish\b|\blebanese\b|\bpersian\b|\bshawarma\b|\bfalafel\b|\bkabob\b|\bkebab\b",
    "african": r"\bafrican\b|\bnigerian\b|\bghanaian\b|\bsuya\b",
    "ethiopian": r"\bethiopian\b|\binjera\b",
    "caribbean": r"\bcaribbean\b|\bJamaican\b|\bHaitian\b|\bDominican\b|\bPuerto Rican\b|\bjerk (?:chicken|pork|food|cuisine)\b|\bjamrock\b|\bkingston\b|\bsancocho\b",
    "seafood": r"\bseafood\b|\bfish and chips\b|\boyster bar\b",
    "vegan": r"\bvegan\b", "vegetarian": r"\bvegetarian\b",
    "cafe_bakery": r"\bcaf[eé]\b|\bbakery\b|\bcoffee shop\b",
    "desserts": r"\bdesserts?\b|\bice cream\b|\bdonut\b",
    "bar_pub": r"\bpub\b|\btavern\b|\bbrewery\b",
    "french": r"\bfrench\b", "spanish": r"\bspanish\b|\btapas\b|\bbasque\b",
    "portuguese": r"\bportuguese\b", "european": r"\beuropean\b|\bgerman\b|\baustrian\b|\bbelgian\b|\bswiss\b",
    "eastern_european": r"\beastern european\b|\bpolish\b|\brussian\b|\bukrainian\b|\bczech\b|\bromanian\b|\bhungarian\b",
    "british_irish": r"\bbritish\b|\birish\b",
    "filipino": r"\bfilipino\b|\bfilipina\b|\bpinoy\b",
    "taiwanese": r"\btaiwanese\b",
    "southeast_asian": r"\bburmese\b|\bcambodian\b|\bindonesian\b|\bmalaysian\b",
    "south_american": r"\bsouth american\b|\bvenezuelan\b|\becuadorian\b|\bargentinian\b|\bbrazilian\b|\bperuvian\b|\bcolombian\b|\bchilean\b|\barepas?\b|\bcachapas?\b",
    "fusion": r"\bfusion\b",
}
SOURCES = ["name", "google_types", "reviews"]
SCHEMA = {
    "type": "object",
    "properties": {
        "primary_cuisine": {"type": "string", "enum": CUISINES},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_sources": {
            "type": "array", "items": {"type": "string", "enum": SOURCES},
            "uniqueItems": True, "maxItems": 3,
        },
    },
    "required": ["primary_cuisine", "confidence", "evidence_sources"],
    "additionalProperties": False,
}
SYSTEM = (
    "Classify a restaurant's primary cuisine conservatively. Return exactly one cuisine. "
    "Use unknown when the name, Google types, and reviews do not establish it. Do not guess "
    "from generic foods or US location. Southern requires an explicit Southern-US restaurant "
    "type, an explicitly Southern identity, or multiple unmistakably Southern regional dishes; "
    "fried food, chicken, seafood, burgers, or American food alone are insufficient. Confidence "
    "1.0 is reserved for direct unambiguous evidence. Dish anchors: dosa, vada, pakora, "
    "biryani and masala imply Indian; sushi, sashimi and ramen imply Japanese; tacos, "
    "burritos and mole imply Mexican. A restaurant name alone is insufficient unless it "
    "contains an explicit cuisine or unmistakable dish. Return schema-valid JSON only."
)


def type_tokens(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return [token.strip() for token in value.split(",") if token.strip()]


def deterministic_classification(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return a high-precision label when Google provides an explicit cuisine type."""
    for token in type_tokens(row.get("types")):
        cuisine = TYPE_TO_CUISINE.get(token)
        if cuisine:
            return {
                "place_id": str(row["place_id"]), "version": VERSION,
                "llm_model": "none:google-type-map-v1",
                "source_review_count": int(row.get("source_review_count", 0)),
                "primary_cuisine": cuisine, "confidence": 1.0,
                "evidence_sources_json": json.dumps(["google_types"]),
                "classification_method": "google_type", "status": "ok",
            }
    name = str(row.get("name") or "")
    for pattern, cuisine in NAME_RULES:
        if pattern.search(name):
            return {
                "place_id": str(row["place_id"]), "version": VERSION,
                "llm_model": "none:name-rule-v1",
                "source_review_count": int(row.get("source_review_count", 0)),
                "primary_cuisine": cuisine, "confidence": 1.0,
                "evidence_sources_json": json.dumps(["name"]),
                "classification_method": "name_rule", "status": "ok",
            }
    return None


def prompt(row: dict[str, Any]) -> str:
    raw_reviews = row.get("corpus")
    reviews = raw_reviews.strip() if isinstance(raw_reviews, str) else ""
    if not reviews:
        reviews = "No usable review text."
    return (
        f"Restaurant name: {row.get('name') or 'Unknown'}\n"
        f"Google place types: {row.get('types') or 'Unknown'}\n"
        f"Up to eight training-only reviews:\n{reviews}\n\n"
        "Choose the single best-supported primary cuisine or unknown. Report which supplied "
        "sources actually support the decision."
    )


def validate(raw: dict[str, Any]) -> None:
    if set(raw) != {"primary_cuisine", "confidence", "evidence_sources"}:
        raise ValueError("unexpected output keys")
    if raw["primary_cuisine"] not in CUISINES:
        raise ValueError("unknown cuisine value")
    confidence = float(raw["confidence"])
    if not 0 <= confidence <= 1:
        raise ValueError("confidence outside [0,1]")
    sources = raw["evidence_sources"]
    if not isinstance(sources, list) or len(sources) != len(set(sources)):
        raise ValueError("invalid or duplicate evidence sources")
    if not set(sources).issubset(SOURCES):
        raise ValueError("unknown evidence source")
    if raw["primary_cuisine"] != "unknown" and not sources:
        raise ValueError("classification has no evidence source")


def reconcile_evidence(raw: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Repair omitted evidence metadata conservatively; never invent a label."""
    if raw.get("primary_cuisine") == "unknown" or raw.get("evidence_sources"):
        return raw
    pattern = CUISINE_EVIDENCE.get(str(raw.get("primary_cuisine")))
    sources = []
    if pattern and re.search(pattern, str(source.get("name") or ""), re.I):
        sources.append("name")
    corpus = source.get("corpus")
    if pattern and isinstance(corpus, str) and re.search(pattern, corpus, re.I):
        sources.append("reviews")
    if sources:
        raw["evidence_sources"] = sources
    else:
        raw["primary_cuisine"] = "unknown"
        raw["confidence"] = min(float(raw.get("confidence", 0.5)), 0.5)
    return raw


def load_completed() -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not CHECKPOINT.exists():
        return completed
    for line in CHECKPOINT.read_text().splitlines():
        try:
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("version") == VERSION:
                completed[str(row["place_id"])] = row
        except (json.JSONDecodeError, KeyError):
            continue
    return completed


def write_output(completed: dict[str, dict[str, Any]]) -> None:
    frame = pd.DataFrame([
        {key: value for key, value in row.items() if key != "status"}
        for row in completed.values()
    ])
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, OUTPUT)


async def main_async(args: argparse.Namespace) -> None:
    restaurants = pd.read_parquet(
        DATA / "restaurants_enriched.parquet",
        columns=["place_id", "name", "types", "cuisine_category"],
    )
    candidates = restaurants[restaurants["cuisine_category"].fillna("Other").eq("Other")]
    corpora = pd.read_parquet(
        DATA / "llm_corpora" / "restaurant_train_corpus.parquet",
        columns=["place_id", "corpus", "source_review_count"],
    )
    candidates = candidates.merge(corpora, on="place_id", how="left", validate="one_to_one")
    candidates["source_review_count"] = candidates["source_review_count"].fillna(0).astype(int)
    candidates = candidates.sample(frac=1, random_state=42).reset_index(drop=True)
    if args.limit:
        candidates = candidates.head(args.limit)

    completed = load_completed()
    deterministic_rows: list[dict[str, Any]] = []
    for row in candidates.to_dict("records"):
        place_id = str(row["place_id"])
        if place_id in completed:
            continue
        result = deterministic_classification(row)
        if result is not None:
            completed[place_id] = result
            deterministic_rows.append(result)
    if deterministic_rows:
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        with CHECKPOINT.open("a") as handle:
            for row in deterministic_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_output(completed)
    todo = candidates[~candidates["place_id"].astype(str).isin(completed)]
    print(
        f"Other-cuisine classification: {len(candidates):,} selected, "
        f"{len(completed):,} resolved ({len(deterministic_rows):,} new Google-type), "
        f"{len(todo):,} requiring 27B, "
        f"concurrency={args.concurrency}", flush=True,
    )
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
            error = "unknown"
            for attempt in range(3):
                async with lock:
                    url = urls[counter % len(urls)]
                    counter += 1
                body = {
                    "model": args.model, "stream": False, "think": False,
                    "keep_alive": "30m", "format": SCHEMA,
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
                    raw = reconcile_evidence(
                        json.loads(payload["message"]["content"]), source
                    )
                    validate(raw)
                    return {
                        "place_id": str(source["place_id"]), "version": VERSION,
                        "llm_model": args.model,
                        "source_review_count": int(source["source_review_count"]),
                        "primary_cuisine": raw["primary_cuisine"],
                        "confidence": float(raw["confidence"]),
                        "evidence_sources_json": json.dumps(raw["evidence_sources"]),
                        "classification_method": "llm_27b",
                        "status": "ok",
                    }
                except Exception as exc:  # preserve final failure for retry on resume
                    error = str(exc)
                    await asyncio.sleep(1.5 * (attempt + 1))
            return {
                "place_id": str(source["place_id"]), "version": VERSION,
                "llm_model": args.model, "status": "error", "error": error,
            }

        tasks = [asyncio.create_task(classify(row)) for row in todo.to_dict("records")]
        with CHECKPOINT.open("a", buffering=1) as handle:
            failures = 0
            for done, future in enumerate(asyncio.as_completed(tasks), 1):
                result = await future
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                if result["status"] == "ok":
                    completed[result["place_id"]] = result
                else:
                    failures += 1
                if done % args.log_every == 0 or done == len(tasks):
                    handle.flush()
                    os.fsync(handle.fileno())
                    write_output(completed)
                    rate = done * 60 / max(time.monotonic() - started, 1e-6)
                    print(
                        f"[{done:,}/{len(tasks):,}] {rate:.1f} restaurants/min, "
                        f"failures={failures}", flush=True,
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3.8-27b-24k:latest")
    parser.add_argument("--urls", default="http://127.0.0.1:11436,http://127.0.0.1:11437")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=300)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
