#!/usr/bin/env python3
"""Generate structured and embedding features only for an isolated new batch."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp
from foodie.features import build_llm_features_ollama as llm


DEFAULT_BATCH = Path("data/feature_batches/places_expansion_2026-09-16")
PROMPT_VERSION = "foodie-incremental-v6-8b-20260916"


async def run(args: argparse.Namespace) -> None:
    batch = args.batch_dir.resolve()
    manifest_path = batch / "batch_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prepare the isolated batch first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("outputs_are_isolated_from_article_batch"):
        raise ValueError("Refusing feature generation without an isolated-batch manifest")

    llm.PROMPT_VERSION = manifest.get("prompt_version_for_generation", PROMPT_VERSION)
    llm.CHECKPOINT_DIR = batch / "llm_checkpoints"
    llm.RESTAURANT_CORPUS = batch / "restaurant_targets.parquet"
    llm.USER_CORPUS = batch / "user_targets.parquet"
    llm.RESTAURANT_OUT = batch / "restaurant_llm_features.parquet"
    llm.USER_OUT = batch / "user_llm_features.parquet"
    llm.RESTAURANT_EMB_OUT = batch / "restaurant_llm_embeddings.parquet"
    llm.USER_EMB_OUT = batch / "user_llm_embeddings.parquet"
    urls = ["http://127.0.0.1:11436", "http://127.0.0.1:11437"]

    for entity in ("restaurant", "user"):
        corpus_path = llm.RESTAURANT_CORPUS if entity == "restaurant" else llm.USER_CORPUS
        import pandas as pd
        count = len(pd.read_parquet(corpus_path, columns=["place_id" if entity == "restaurant" else "contributor_id"]))
        if count == 0:
            print(f"{entity}: no eligible batch targets", flush=True)
            continue
        features = await llm.generate_structured(
            entity=entity, model=args.model, base_urls=urls,
            concurrency=args.concurrency, timeout=args.timeout, limit=0,
            checkpoint_every=200, log_every=50,
        )
        if len(features) != count:
            raise RuntimeError(
                f"{entity} structured extraction incomplete: {len(features):,}/{count:,}; "
                "embedding stage will not start"
            )
    # These two Ollama workers are project-local. Explicitly release the 8B
    # model before loading the embedding model (MAX_LOADED_MODELS=1).
    async with aiohttp.ClientSession() as session:
        for url in urls:
            async with session.post(
                url + "/api/generate",
                json={"model": args.model, "keep_alive": 0},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                response.raise_for_status()
    print("Released structured model; starting embeddings", flush=True)
    for entity in ("restaurant", "user"):
        corpus_path = llm.RESTAURANT_CORPUS if entity == "restaurant" else llm.USER_CORPUS
        import pandas as pd
        count = len(pd.read_parquet(corpus_path, columns=["place_id" if entity == "restaurant" else "contributor_id"]))
        if count == 0:
            continue
        await llm.generate_embeddings(
            entity=entity, model=args.embedding_model, base_urls=urls,
            batch_size=args.embedding_batch_size, projection_dim=64, limit=0,
            timeout=args.embedding_timeout, checkpoint_every=64,
        )
    print(f"BATCH FEATURE GENERATION COMPLETE: {batch}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--model", default="qwen3-8b-fast:latest")
    parser.add_argument("--embedding-model", default="qwen3-embedding:0.6b")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--embedding-timeout", type=int, default=600)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
