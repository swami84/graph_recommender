"""Structured, concurrency-safe result logging for the publication experiment."""

from __future__ import annotations

import csv
import fcntl
import os
from datetime import datetime
from pathlib import Path

RESULTS = Path(os.environ.get(
    "FOODIE_PUBLICATION_RESULTS", "results/publication_core_results.csv"
))
FIELDS = ["timestamp", "model", "condition", "seed", "val_hit_at_10",
          "val_ndcg_at_10", "test_hit_at_10", "test_ndcg_at_10", "epochs", "emb_dim"]


def condition_from_skip(skip: str) -> str:
    groups = {x.strip() for x in skip.split(",") if x.strip()}
    return {
        frozenset(): "full_llm",
        frozenset({"llm", "llm_embedding", "dish_llm"}): "non_llm",
        frozenset({"llm_embedding"}): "structured_only",
        frozenset({"llm"}): "embeddings_only",
    }.get(frozenset(groups), "custom_skip:" + ",".join(sorted(groups)))


def save_publication_result(model, condition, seed, val_metrics, test_metrics, epochs, emb_dim):
    RESULTS.parent.mkdir(exist_ok=True)
    with open(RESULTS, "a+", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0, 2)
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().astimezone().isoformat(), "model": model,
            "condition": condition, "seed": seed,
            "val_hit_at_10": f"{val_metrics['hit']:.8f}" if val_metrics else "",
            "val_ndcg_at_10": f"{val_metrics['ndcg']:.8f}" if val_metrics else "",
            "test_hit_at_10": f"{test_metrics['hit']:.8f}",
            "test_ndcg_at_10": f"{test_metrics['ndcg']:.8f}",
            "epochs": epochs, "emb_dim": emb_dim,
        })
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)
