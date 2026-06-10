#!/usr/bin/env python3
"""
run_kgat_spatial.py

Runs KGAT with the new CBG spatial graph on both GPUs simultaneously:
    GPU 0: KGAT 4L all features  +spatial_cbg
    GPU 1: KGAT 4L skip llm      +spatial_cbg

After both finish, generates predictions and runs proximity re-ranking.

Usage:
    python run_kgat_spatial.py
    python run_kgat_spatial.py --gpu0 0 --gpu1 1
    python run_kgat_spatial.py --no-proximity   # skip proximity re-ranking
"""

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_kgat_sp")

EMB_DIR  = Path("data/embeddings")
PRED_DIR = Path("data/predictions")

_BASE8 = ["--epochs", "300", "--emb-dim", "2048", "--prebuilt-features",
          "--eval-every", "50", "--batch-size", "8192",
          "--cf-layers", "4", "--kg-layers", "2"]

RUNS = [
    {
        "name":      "KGAT 4L all features +spatial_cbg",
        "script":    "recommendation_kgat.py",
        "args":      _BASE8,
        "skip":      "",
        "emb_user":  EMB_DIR / "kgat_all_user_embeddings.pt",
        "emb_item":  EMB_DIR / "kgat_all_item_embeddings.pt",
        "pred_name": "kgat_all",
    },
    {
        "name":      "KGAT 4L skip llm +spatial_cbg",
        "script":    "recommendation_kgat.py",
        "args":      _BASE8,
        "skip":      "llm",
        "emb_user":  EMB_DIR / "kgat_skip_llm_user_embeddings.pt",
        "emb_item":  EMB_DIR / "kgat_skip_llm_item_embeddings.pt",
        "pred_name": "kgat_llm",
    },
]


def _build_cmd(run: dict) -> list[str]:
    cmd = [sys.executable, run["script"]] + run["args"]
    if run["skip"]:
        cmd += ["--skip-feature-groups", run["skip"]]
    return cmd


def _gpu_env(gpu_id: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]    = str(gpu_id)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def _drain(proc: subprocess.Popen, label: str) -> None:
    for line in proc.stdout:
        sys.stdout.write(f"[{label}] {line}")
        sys.stdout.flush()
    proc.wait()


def save_predictions(run: dict) -> None:
    u_path, i_path = run["emb_user"], run["emb_item"]
    if not u_path.exists() or not i_path.exists():
        log.warning(f"  Embeddings not found for {run['name']} — skipping predictions")
        return

    log.info(f"  Generating predictions: {run['name']} …")
    u_data      = torch.load(u_path, map_location="cpu", weights_only=True)
    i_data      = torch.load(i_path, map_location="cpu", weights_only=True)
    u_emb       = u_data["embeddings"].float()
    i_emb       = i_data["embeddings"].float()
    user_id_map = u_data["id_map"]
    item_id_map = i_data["id_map"]

    from recommendation_gnn import load_interactions
    train_df, test_df, _, _ = load_interactions()
    train_sets = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    test_users = test_df["user_idx"].values
    test_items = test_df["item_idx"].values

    records = []
    for s in range(0, len(test_users), 2048):
        e      = min(s + 2048, len(test_users))
        u_idx  = test_users[s:e]
        i_idx  = test_items[s:e]
        scores = (u_emb[u_idx] @ i_emb.T).numpy()
        for k, (ui, ii) in enumerate(zip(u_idx, i_idx)):
            for excl in train_sets.get(int(ui), set()):
                scores[k, excl] = -np.inf
            top10 = np.argsort(-scores[k])[:10].tolist()
            rank  = top10.index(int(ii)) + 1 if int(ii) in top10 else None
            records.append({
                "user_idx":        int(ui),
                "true_item_idx":   int(ii),
                "top10_item_idxs": top10,
                "contributor_id":  user_id_map.get(int(ui), str(ui)),
                "true_place_id":   item_id_map.get(int(ii), str(ii)),
                "top10_place_ids": [item_id_map.get(x, str(x)) for x in top10],
                "rank":            rank,
            })

    pred_df = pd.DataFrame(records)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"{run['pred_name']}_predictions.parquet"
    pred_df.to_parquet(out, index=False)
    hit = pred_df["rank"].notna().mean()
    log.info(f"  Saved → {out}  ({len(pred_df):,} rows | Hit@10={hit:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu0",         type=int, default=0)
    parser.add_argument("--gpu1",         type=int, default=1)
    parser.add_argument("--no-proximity", action="store_true",
                        help="Skip proximity re-ranking step")
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("KGAT + CBG spatial graph")
    log.info(f"  GPU {args.gpu0}: {RUNS[0]['name']}")
    log.info(f"  GPU {args.gpu1}: {RUNS[1]['name']}")
    log.info("=" * 65)

    t0 = time.time()

    proc_a = subprocess.Popen(_build_cmd(RUNS[0]), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               env=_gpu_env(args.gpu0))
    proc_b = subprocess.Popen(_build_cmd(RUNS[1]), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               env=_gpu_env(args.gpu1))

    ta = threading.Thread(target=_drain, args=(proc_a, f"GPU{args.gpu0}"), daemon=True)
    tb = threading.Thread(target=_drain, args=(proc_b, f"GPU{args.gpu1}"), daemon=True)
    ta.start(); tb.start()
    ta.join();  tb.join()

    elapsed = timedelta(seconds=int(time.time() - t0))
    log.info(f"\nTraining done in {elapsed}")

    for run, proc in [(RUNS[0], proc_a), (RUNS[1], proc_b)]:
        if proc.returncode != 0:
            log.warning(f"  {run['name']} FAILED (rc={proc.returncode})")
        else:
            save_predictions(run)

    if not args.no_proximity:
        log.info("\nRunning proximity re-ranking on updated KGAT embeddings …")
        for run in RUNS:
            rc = subprocess.call([
                sys.executable, "rerank_proximity.py",
                "--model", run["pred_name"],
                "--alpha", "0.3", "--bandwidth", "10", "--top-k", "50",
            ])
            if rc != 0:
                log.warning(f"  Proximity re-ranking failed for {run['pred_name']}")

    log.info(f"\nTotal wall time: {timedelta(seconds=int(time.time() - t0))}")


if __name__ == "__main__":
    main()
