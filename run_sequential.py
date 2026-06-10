#!/usr/bin/env python3
"""
run_sequential.py

Step 1 — Train the three fixed models:
    GPU 0 + GPU 1 simultaneously: InfoNCE 4L  |  SimGCL-HRCL 4L
    Then GPU 0 solo:              HEK-CL cf4/kg2

Step 2 — Proximity re-ranking for every model that has saved embeddings.

Results written to:
    data/predictions/{model}_predictions.parquet
    data/predictions/{model}_prox_a30_predictions.parquet
    results/model_results.csv

Usage:
    python run_sequential.py
    python run_sequential.py --gpu0 0 --gpu1 1
    python run_sequential.py --alpha 0.4 --bw 5
    python run_sequential.py --skip-training
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
log = logging.getLogger("run_seq")

EMB_DIR  = Path("data/embeddings")
PRED_DIR = Path("data/predictions")

# ── Training runs ─────────────────────────────────────────────────────────────

_BASE8 = ["--epochs", "300", "--emb-dim", "2048", "--prebuilt-features",
          "--eval-every", "50", "--batch-size", "8192"]

TRAINING_RUNS = [
    {
        "name":      "InfoNCE 4L skip llm",
        "script":    "recommendation_infonce.py",
        "args":      _BASE8 + ["--layers", "4"],
        "skip":      "llm",
        "emb_user":  EMB_DIR / "infonce_skip_llm_user_embeddings.pt",
        "emb_item":  EMB_DIR / "infonce_skip_llm_item_embeddings.pt",
        "pred_name": "infonce_llm",
    },
    {
        "name":      "SimGCL-HRCL 4L skip llm",
        "script":    "recommendation_simgcl_hrcl.py",
        "args":      _BASE8 + ["--layers", "4"],
        "skip":      "llm",
        "emb_user":  EMB_DIR / "simgcl_hrcl_skip_llm_user_embeddings.pt",
        "emb_item":  EMB_DIR / "simgcl_hrcl_skip_llm_item_embeddings.pt",
        "pred_name": "simgcl_hrcl_llm",
    },
    {
        "name":      "HEK-CL cf4/kg2 skip llm",
        "script":    "recommendation_hek_cl.py",
        "args":      ["--epochs", "300", "--emb-dim", "512",
                      "--cf-layers", "4", "--kg-layers", "2",
                      "--prebuilt-features", "--eval-every", "50"],
        "skip":      "llm",
        "emb_user":  EMB_DIR / "hek_cl_skip_llm_user_embeddings.pt",
        "emb_item":  EMB_DIR / "hek_cl_skip_llm_item_embeddings.pt",
        "pred_name": "hek_cl_llm",
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def launch(run: dict, gpu_id: int) -> subprocess.Popen:
    cmd = _build_cmd(run)
    log.info(f"  ▶  {run['name']}  [GPU {gpu_id}]")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=_gpu_env(gpu_id))


# ── Training step ──────────────────────────────────────────────────────────────

def run_training(gpu0: int, gpu1: int) -> None:
    log.info("=" * 65)
    log.info("STEP 1 — Training")
    log.info("=" * 65)

    infonce, simgcl_hrcl, hek_cl = TRAINING_RUNS

    # ── Pair: InfoNCE (GPU 0) + SimGCL-HRCL (GPU 1) simultaneously ───────────
    log.info(f"\n  Launching in parallel:")
    log.info(f"    GPU {gpu0}: {infonce['name']}")
    log.info(f"    GPU {gpu1}: {simgcl_hrcl['name']}")

    t0     = time.time()
    proc_a = launch(infonce,      gpu0)
    proc_b = launch(simgcl_hrcl,  gpu1)

    thread_a = threading.Thread(target=_drain, args=(proc_a, f"GPU{gpu0}"), daemon=True)
    thread_b = threading.Thread(target=_drain, args=(proc_b, f"GPU{gpu1}"), daemon=True)
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    elapsed = timedelta(seconds=int(time.time() - t0))
    log.info(f"\n  Parallel pair done in {elapsed}")

    for run, proc in [(infonce, proc_a), (simgcl_hrcl, proc_b)]:
        if proc.returncode != 0:
            log.warning(f"  {run['name']} FAILED (rc={proc.returncode}) — continuing")
        else:
            save_predictions(run)

    # ── Solo: HEK-CL (GPU 0) ─────────────────────────────────────────────────
    log.info(f"\n  Running solo:")
    log.info(f"    GPU {gpu0}: {hek_cl['name']}")

    t0     = time.time()
    proc_c = launch(hek_cl, gpu0)
    _drain(proc_c, f"GPU{gpu0}")
    elapsed = timedelta(seconds=int(time.time() - t0))
    log.info(f"\n  HEK-CL done in {elapsed}")

    if proc_c.returncode != 0:
        log.warning(f"  {hek_cl['name']} FAILED (rc={proc_c.returncode}) — continuing")
    else:
        save_predictions(hek_cl)


# ── Proximity step ─────────────────────────────────────────────────────────────

def run_proximity(alpha: float, bandwidth: float, top_k: int) -> None:
    log.info("")
    log.info("=" * 65)
    log.info("STEP 2 — Proximity re-ranking (all saved models)")
    log.info("=" * 65)

    cmd = [sys.executable, "rerank_proximity.py",
           "--alpha", str(alpha), "--bandwidth", str(bandwidth), "--top-k", str(top_k)]
    log.info(f"  {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        log.error(f"rerank_proximity.py exited with rc={rc}")
    else:
        log.info("Proximity re-ranking complete.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu0",          type=int,   default=0)
    parser.add_argument("--gpu1",          type=int,   default=1)
    parser.add_argument("--alpha",         type=float, default=0.3)
    parser.add_argument("--bw",            type=float, default=10.0, dest="bandwidth")
    parser.add_argument("--top-k",         type=int,   default=50)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    total_t0 = time.time()

    if not args.skip_training:
        run_training(args.gpu0, args.gpu1)

    run_proximity(args.alpha, args.bandwidth, args.top_k)

    log.info(f"\nTotal wall time: {timedelta(seconds=int(time.time() - total_t0))}")


if __name__ == "__main__":
    main()
