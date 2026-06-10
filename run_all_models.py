#!/usr/bin/env python3
"""
run_all_models.py — Dual-GPU runner for all recommendation models.

Runs experiments in pairs: GPU 0 and GPU 1 execute simultaneously.  Each
subprocess exits when done, immediately freeing its VRAM.  After both in a
pair finish, top-10 predictions are generated on CPU from the saved embeddings.

Experiments with safe=False (InfoNCE, SimGCL-HRCL, HEK-CL) log errors and
continue; their partner on the other GPU is not affected.

Usage:
    python run_all_models.py                         # all 12 runs, GPUs 0+1
    python run_all_models.py --gpu0 0 --gpu1 1
    python run_all_models.py --runs 4 5 6 7          # subset (0-indexed), paired in order
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
log = logging.getLogger("run_all")

EMB_DIR  = Path("data/embeddings")
PRED_DIR = Path("data/predictions")
ALLOC    = "expandable_segments:True"

# ── Experiment definitions ─────────────────────────────────────────────────────

_BASE = ["--epochs", "300", "--emb-dim", "2048", "--prebuilt-features", "--eval-every", "50"]
_B8   = _BASE + ["--batch-size", "8192"]

RUNS = [
    # ── LightGCN ──────────────────────────────────────────────────────────────
    {
        "name":     "LightGCN 3L all features",
        "script":   "recommendation_gnn.py",
        "args":     _BASE + ["--layers", "3"],
        "skip":     "",
        "emb_user": EMB_DIR / "lightgcn_3l_all_user_embeddings.pt",
        "emb_item": EMB_DIR / "lightgcn_3l_all_item_embeddings.pt",
        "safe":     True,
    },
    {
        "name":     "LightGCN 3L skip llm",
        "script":   "recommendation_gnn.py",
        "args":     _BASE + ["--layers", "3"],
        "skip":     "llm",
        "emb_user": EMB_DIR / "lightgcn_3l_skip_llm_user_embeddings.pt",
        "emb_item": EMB_DIR / "lightgcn_3l_skip_llm_item_embeddings.pt",
        "safe":     True,
    },
    {
        "name":     "LightGCN 4L all features",
        "script":   "recommendation_gnn.py",
        "args":     _BASE + ["--layers", "4"],
        "skip":     "",
        "emb_user": EMB_DIR / "lightgcn_4l_all_user_embeddings.pt",
        "emb_item": EMB_DIR / "lightgcn_4l_all_item_embeddings.pt",
        "safe":     True,
    },
    {
        "name":     "LightGCN 4L skip llm",
        "script":   "recommendation_gnn.py",
        "args":     _BASE + ["--layers", "4"],
        "skip":     "llm",
        "emb_user": EMB_DIR / "lightgcn_4l_skip_llm_user_embeddings.pt",
        "emb_item": EMB_DIR / "lightgcn_4l_skip_llm_item_embeddings.pt",
        "safe":     True,
    },
    # ── KGAT ──────────────────────────────────────────────────────────────────
    {
        "name":     "KGAT 4L all features",
        "script":   "recommendation_kgat.py",
        "args":     _B8 + ["--cf-layers", "4", "--kg-layers", "2"],
        "skip":     "",
        "emb_user": EMB_DIR / "kgat_all_user_embeddings.pt",
        "emb_item": EMB_DIR / "kgat_all_item_embeddings.pt",
        "safe":     True,
    },
    {
        "name":     "KGAT 4L skip llm",
        "script":   "recommendation_kgat.py",
        "args":     _B8 + ["--cf-layers", "4", "--kg-layers", "2"],
        "skip":     "llm",
        "emb_user": EMB_DIR / "kgat_skip_llm_user_embeddings.pt",
        "emb_item": EMB_DIR / "kgat_skip_llm_item_embeddings.pt",
        "safe":     True,
    },
    # ── RaDAR ─────────────────────────────────────────────────────────────────
    {
        "name":     "RaDAR 4L all features",
        "script":   "recommendation_radar.py",
        "args":     _B8 + ["--n-layers", "4",
                           "--lambda-acl", "0.1", "--lambda-ddr", "0.01", "--lambda-mask", "0.05"],
        "skip":     "",
        "emb_user": EMB_DIR / "radar_all_user_embeddings.pt",
        "emb_item": EMB_DIR / "radar_all_item_embeddings.pt",
        "safe":     True,
    },
    {
        "name":     "RaDAR 4L skip llm",
        "script":   "recommendation_radar.py",
        "args":     _B8 + ["--n-layers", "4",
                           "--lambda-acl", "0.1", "--lambda-ddr", "0.01", "--lambda-mask", "0.05"],
        "skip":     "llm",
        "emb_user": EMB_DIR / "radar_skip_llm_user_embeddings.pt",
        "emb_item": EMB_DIR / "radar_skip_llm_item_embeddings.pt",
        "safe":     True,
    },
    # ── SimGCL ────────────────────────────────────────────────────────────────
    {
        "name":     "SimGCL 4L skip llm",
        "script":   "recommendation_simgcl.py",
        "args":     _B8 + ["--layers", "4"],
        "skip":     "llm",
        "emb_user": EMB_DIR / "simgcl_skip_llm_user_embeddings.pt",
        "emb_item": EMB_DIR / "simgcl_skip_llm_item_embeddings.pt",
        "safe":     True,
    },
    # ── InfoNCE ───────────────────────────────────────────────────────────────
    {
        "name":     "InfoNCE 4L skip llm",
        "script":   "recommendation_infonce.py",
        "args":     _B8 + ["--layers", "4"],
        "skip":     "llm",
        "emb_user": EMB_DIR / "infonce_skip_llm_user_embeddings.pt",
        "emb_item": EMB_DIR / "infonce_skip_llm_item_embeddings.pt",
        "safe":     False,
    },
    # ── SimGCL-HRCL ───────────────────────────────────────────────────────────
    {
        "name":     "SimGCL-HRCL 4L skip llm",
        "script":   "recommendation_simgcl_hrcl.py",
        "args":     _B8 + ["--layers", "4"],
        "skip":     "llm",
        "emb_user": EMB_DIR / "simgcl_hrcl_skip_llm_user_embeddings.pt",
        "emb_item": EMB_DIR / "simgcl_hrcl_skip_llm_item_embeddings.pt",
        "safe":     False,
    },
    # ── HEK-CL ────────────────────────────────────────────────────────────────
    {
        "name":     "HEK-CL cf4/kg2 skip llm",
        "script":   "recommendation_hek_cl.py",
        "args":     ["--epochs", "300", "--emb-dim", "512",
                     "--cf-layers", "4", "--kg-layers", "2",
                     "--prebuilt-features", "--eval-every", "50"],
        "skip":     "llm",
        "emb_user": EMB_DIR / "hek_cl_skip_llm_user_embeddings.pt",
        "emb_item": EMB_DIR / "hek_cl_skip_llm_item_embeddings.pt",
        "safe":     False,
    },
]

# Natural pairing: same model, different variants run on different GPUs
PAIRS: list[tuple[int, int | None]] = [
    (0,  1),   # LightGCN 3L all  + LightGCN 3L skip llm
    (2,  3),   # LightGCN 4L all  + LightGCN 4L skip llm
    (4,  5),   # KGAT 4L all      + KGAT 4L skip llm
    (6,  7),   # RaDAR 4L all     + RaDAR 4L skip llm
    (8,  9),   # SimGCL           + InfoNCE
    (10, 11),  # SimGCL-HRCL      + HEK-CL
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pred_name(run: dict) -> str:
    tag  = run["skip"].replace(",", "_") if run["skip"] else "all"
    name = run["script"].replace("recommendation_", "").replace(".py", "")
    if "gnn" in run["script"]:
        layers = run["args"][run["args"].index("--layers") + 1]
        return f"lightgcn_{layers}l_{tag}"
    return f"{name}_{tag}"


def _build_cmd(run: dict) -> list[str]:
    cmd = [sys.executable, run["script"]] + run["args"]
    if run["skip"]:
        cmd += ["--skip-feature-groups", run["skip"]]
    return cmd


def _gpu_env(gpu_id: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]        = str(gpu_id)
    env["PYTORCH_CUDA_ALLOC_CONF"]     = ALLOC
    return env


def _drain(proc: subprocess.Popen, label: str, sink: list) -> None:
    """Read subprocess stdout line-by-line, prefix with GPU label, then wait."""
    for line in proc.stdout:
        sys.stdout.write(f"[{label}] {line}")
        sys.stdout.flush()
        sink.append(line)
    proc.wait()


# ── Prediction generation ──────────────────────────────────────────────────────

def save_predictions(run: dict, pred_name: str) -> None:
    u_path, i_path = run["emb_user"], run["emb_item"]
    if not u_path.exists() or not i_path.exists():
        log.warning(f"  Embeddings not found for {run['name']} — skipping predictions")
        return

    log.info(f"  Generating predictions: {run['name']} …")

    u_data      = torch.load(u_path, map_location="cpu")
    i_data      = torch.load(i_path, map_location="cpu")
    u_emb       = u_data["embeddings"].float()
    i_emb       = i_data["embeddings"].float()
    user_id_map = u_data["id_map"]
    item_id_map = i_data["id_map"]

    from recommendation_gnn import load_interactions
    train_df, test_df, _, _ = load_interactions()
    train_sets  = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    test_users  = test_df["user_idx"].values
    test_items  = test_df["item_idx"].values

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
    out      = PRED_DIR / f"{pred_name}_predictions.parquet"
    pred_df.to_parquet(out, index=False)
    hit_rate = pred_df["rank"].notna().mean()
    log.info(f"  Saved → {out}  ({len(pred_df):,} rows | Hit@10={hit_rate:.4f})")


# ── Pair runner ────────────────────────────────────────────────────────────────

def run_pair(
    run_a:  dict,
    run_b:  dict | None,
    gpu0:   int,
    gpu1:   int,
) -> list[dict]:
    """
    Launch run_a on gpu0 and (optionally) run_b on gpu1 simultaneously.
    Stream both outputs with [GPU{n}] prefix.  Wait for both.
    Returns result dicts for each run.
    """
    hdr = "=" * 70
    print(f"\n{hdr}", flush=True)
    print(f"  GPU {gpu0}: {run_a['name']}", flush=True)
    if run_b:
        print(f"  GPU {gpu1}: {run_b['name']}", flush=True)
    print(f"{hdr}\n", flush=True)

    t0     = time.time()
    cmd_a  = _build_cmd(run_a)
    proc_a = subprocess.Popen(cmd_a, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, env=_gpu_env(gpu0))
    lines_a: list[str] = []
    thread_a = threading.Thread(target=_drain, args=(proc_a, f"GPU{gpu0}", lines_a), daemon=True)
    thread_a.start()

    proc_b, thread_b, lines_b = None, None, []
    if run_b is not None:
        cmd_b  = _build_cmd(run_b)
        proc_b = subprocess.Popen(cmd_b, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=_gpu_env(gpu1))
        thread_b = threading.Thread(target=_drain, args=(proc_b, f"GPU{gpu1}", lines_b), daemon=True)
        thread_b.start()

    thread_a.join()
    if thread_b:
        thread_b.join()

    elapsed = time.time() - t0

    results = []
    for run, proc in [(run_a, proc_a), (run_b, proc_b)]:
        if run is None or proc is None:
            continue
        rc = proc.returncode
        if rc != 0:
            level = log.error if run["safe"] else log.warning
            level(f"  {run['name']} FAILED (rc={rc})")
            results.append({"name": run["name"], "rc": rc, "elapsed": elapsed, "pred_name": None})
        else:
            pn = _pred_name(run)
            save_predictions(run, pn)
            results.append({"name": run["name"], "rc": rc, "elapsed": elapsed, "pred_name": pn})

    return results


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    print(f"\n{'='*72}")
    print("  MODEL RESULTS SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Run':<38}  {'Status':>7}  {'Time':>8}  {'Preds':>6}")
    print(f"  {'-'*36}  {'-'*7}  {'-'*8}  {'-'*6}")
    for r in results:
        status   = "OK"    if r["rc"] == 0 else f"ERR {r['rc']}"
        preds    = "saved" if r["pred_name"] else "—"
        time_str = str(timedelta(seconds=int(r["elapsed"])))
        print(f"  {r['name']:<38}  {status:>7}  {time_str:>8}  {preds:>6}")
    print(f"{'='*72}\n", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-GPU runner for all recommendation models")
    parser.add_argument("--gpu0", type=int, default=0, help="CUDA device id for slot 0 (default 0)")
    parser.add_argument("--gpu1", type=int, default=1, help="CUDA device id for slot 1 (default 1)")
    parser.add_argument("--runs", nargs="*", type=int, default=None,
                        help="0-based run indices to execute; paired in sequence (default: all)")
    args = parser.parse_args()

    if args.runs is not None:
        # User-selected subset: pair them in order [0,1], [2,3], ...
        sel  = [RUNS[i] for i in args.runs]
        pairs = [(sel[i], sel[i + 1] if i + 1 < len(sel) else None)
                 for i in range(0, len(sel), 2)]
    else:
        pairs = [(RUNS[a], RUNS[b]) for a, b in PAIRS]

    log.info(f"Starting {sum(1 + (b is not None) for a, b in pairs)} run(s) across {len(pairs)} pair(s)")
    for i, (a, b) in enumerate(pairs):
        log.info(f"  Pair {i}: GPU{args.gpu0}={a['name']}" +
                 (f"  GPU{args.gpu1}={b['name']}" if b else ""))

    all_results: list[dict] = []
    total_t0 = time.time()

    for run_a, run_b in pairs:
        pair_results = run_pair(run_a, run_b, args.gpu0, args.gpu1)
        all_results.extend(pair_results)

        # Abort if a safe-critical run failed
        for res in pair_results:
            run = next(r for r in [run_a, run_b] if r is not None and r["name"] == res["name"])
            if res["rc"] != 0 and run["safe"]:
                log.error(f"Critical run {res['name']} failed — aborting")
                print_summary(all_results)
                return

        print_summary(all_results)

    print(f"Total wall time: {timedelta(seconds=int(time.time() - total_t0))}\n")


if __name__ == "__main__":
    main()
