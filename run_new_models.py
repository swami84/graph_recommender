#!/usr/bin/env python3
"""
run_new_models.py

Trains the top-5 models sequentially on the expanded dataset using
torchrun + DDP + Adafactor (both GPUs per model, emb_dim=2048), then runs
proximity α×bandwidth grid search.

Training schedule (sequential, each model uses both GPUs via torchrun DDP):
    1. InfoNCE-KGAT-SAL  all features T=3 (batch 2048, emb 2048)
    2. KGAT              all features      (batch 8192, emb 2048)
    3. KGAT-SAL          all features T=3  (batch 2048, emb 2048)
    4. RaDAR             all features      (batch 8192, emb 2048)
    5. InfoNCE           all features      (batch 8192, emb 2048)

After training:
    Proximity α×bandwidth grid search for all 5 models.

Usage:
    python run_new_models.py
    python run_new_models.py --no-proximity
    python run_new_models.py --no-training   # proximity grid search only
    python run_new_models.py --full-test     # evaluate on full expanded item set
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
import polars as pl
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_new")

EMB_DIR  = Path("data/embeddings")
PRED_DIR = Path("data/predictions")

# ── Run definitions ────────────────────────────────────────────────────────────

_BASE8 = ["--epochs", "300", "--emb-dim", "2048", "--prebuilt-features",
          "--eval-every", "50", "--batch-size", "8192"]

_SAL_ARGS = ["--epochs", "300", "--emb-dim", "2048", "--prebuilt-features",
             "--eval-every", "50", "--batch-size", "2048",
             "--cf-layers", "4", "--kg-layers", "2",
             "--n-periods", "3", "--lambda-sal", "1e-6",
             "--no-spatial-cbg"]

RUNS = [
    # Round 1 ──────────────────────────────────────────────────────────────────
    {
        "name":      "InfoNCE-KGAT-SAL all features T=3",
        "script":    "recommendation_infonce_kgat_sal.py",
        "args":      _SAL_ARGS,
        "skip":      "",
        "emb_user":  EMB_DIR / "infonce_kgat_sal_all_T3_nospatial_user_embeddings.pt",
        "emb_item":  EMB_DIR / "infonce_kgat_sal_all_T3_nospatial_item_embeddings.pt",
        "pred_name": "infonce_kgat_sal_all_T3_nospatial",
        "safe":      True,
    },
    {
        "name":      "KGAT all features",
        "script":    "recommendation_kgat.py",
        "args":      _BASE8 + ["--cf-layers", "4", "--kg-layers", "2"],
        "skip":      "",
        "emb_user":  EMB_DIR / "kgat_all_user_embeddings.pt",
        "emb_item":  EMB_DIR / "kgat_all_item_embeddings.pt",
        "pred_name": "kgat_all",
        "safe":      True,
    },
    # Round 2 ──────────────────────────────────────────────────────────────────
    {
        "name":      "KGAT-SAL all features T=3",
        "script":    "recommendation_kgat_sal.py",
        "args":      _SAL_ARGS,
        "skip":      "",
        "emb_user":  EMB_DIR / "kgat_sal_all_T3_nospatial_user_embeddings.pt",
        "emb_item":  EMB_DIR / "kgat_sal_all_T3_nospatial_item_embeddings.pt",
        "pred_name": "kgat_sal_all_T3_nospatial",
        "safe":      True,
    },
    {
        "name":      "RaDAR all features",
        "script":    "recommendation_radar.py",
        "args":      _BASE8 + ["--n-layers", "4"],
        "skip":      "",
        "emb_user":  EMB_DIR / "radar_all_user_embeddings.pt",
        "emb_item":  EMB_DIR / "radar_all_item_embeddings.pt",
        "pred_name": "radar_all",
        "safe":      True,
    },
    # Round 3 ──────────────────────────────────────────────────────────────────
    {
        "name":      "InfoNCE all features",
        "script":    "recommendation_infonce.py",
        "args":      _BASE8 + ["--layers", "4"],
        "skip":      "",
        "emb_user":  EMB_DIR / "infonce_all_user_embeddings.pt",
        "emb_item":  EMB_DIR / "infonce_all_item_embeddings.pt",
        "pred_name": "infonce_all",
        "safe":      True,
    },
]

# Proximity grid-search models
PROXIMITY_MODELS = [
    {
        "pred_name":   "infonce_kgat_sal_all_T3_nospatial",
        "emb_user":    "infonce_kgat_sal_all_T3_nospatial_user_embeddings.pt",
        "emb_item":    "infonce_kgat_sal_all_T3_nospatial_item_embeddings.pt",
        "csv_model":   "InfoNCE-KGAT-SAL",
        "csv_variant": "all features T=3",
        "n_layers":    4,
        "skip_groups": "",
    },
    {
        "pred_name":   "kgat_all",
        "emb_user":    "kgat_all_user_embeddings.pt",
        "emb_item":    "kgat_all_item_embeddings.pt",
        "csv_model":   "KGAT",
        "csv_variant": "all features",
        "n_layers":    4,
        "skip_groups": "",
    },
    {
        "pred_name":   "kgat_sal_all_T3_nospatial",
        "emb_user":    "kgat_sal_all_T3_nospatial_user_embeddings.pt",
        "emb_item":    "kgat_sal_all_T3_nospatial_item_embeddings.pt",
        "csv_model":   "KGAT-SAL",
        "csv_variant": "all features T=3",
        "n_layers":    4,
        "skip_groups": "",
    },
    {
        "pred_name":   "radar_all",
        "emb_user":    "radar_all_user_embeddings.pt",
        "emb_item":    "radar_all_item_embeddings.pt",
        "csv_model":   "RaDAR",
        "csv_variant": "all features",
        "n_layers":    4,
        "skip_groups": "",
    },
    {
        "pred_name":   "infonce_all",
        "emb_user":    "infonce_all_user_embeddings.pt",
        "emb_item":    "infonce_all_item_embeddings.pt",
        "csv_model":   "InfoNCE",
        "csv_variant": "all features",
        "n_layers":    4,
        "skip_groups": "",
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_cmd(run: dict) -> list[str]:
    cmd = ["torchrun", "--nproc_per_node", "2", run["script"]] + run["args"]
    if run["skip"]:
        cmd += ["--skip-feature-groups", run["skip"]]
    return cmd


def _gpu_env(original_test: bool = True) -> dict:
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if original_test:
        env["FOODIE_ORIGINAL_TEST"] = "1"
    return env


def _drain(proc: subprocess.Popen, label: str) -> None:
    for line in proc.stdout:
        sys.stdout.write(f"[{label}] {line}")
        sys.stdout.flush()
    proc.wait()


def save_predictions(run: dict) -> None:
    u_path, i_path = run["emb_user"], run["emb_item"]
    if not u_path.exists() or not i_path.exists():
        log.warning(f"  Embeddings missing for {run['name']} — skipping predictions")
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


def run_single(run: dict, original_test: bool = True) -> None:
    log.info(f"\n{'='*65}")
    log.info(f"  torchrun (2 GPUs): {run['name']}")
    log.info(f"{'='*65}")

    t0   = time.time()
    proc = subprocess.Popen(_build_cmd(run), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                             env=_gpu_env(original_test))
    t = threading.Thread(target=_drain, args=(proc, run["name"][:20]), daemon=True)
    t.start()
    t.join()
    log.info(f"\n  Done in {timedelta(seconds=int(time.time() - t0))}")

    if proc.returncode != 0:
        if run["safe"]:
            log.error(f"  {run['name']} FAILED (rc={proc.returncode})")
        else:
            log.warning(f"  {run['name']} FAILED (rc={proc.returncode}) — continuing")
    else:
        save_predictions(run)


# ── Proximity grid search ──────────────────────────────────────────────────────

def run_proximity_grid(top_k: int) -> None:
    import rerank_proximity as rp

    log.info(f"\n{'='*65}")
    log.info("  Proximity grid search")
    log.info(f"  α:         {rp.ALPHA_GRID}")
    log.info(f"  bandwidth: {rp.BANDWIDTH_GRID} km")
    log.info(f"{'='*65}")

    log.info("Loading restaurant coordinates …")
    coords_pl = rp.load_restaurant_coords()

    log.info("Loading interactions …")
    from recommendation_gnn import load_interactions
    train_df, test_df, user_enc, item_enc = load_interactions()
    user_dec = {v: k for k, v in user_enc.items()}
    item_dec = {v: k for k, v in item_enc.items()}

    log.info("Computing user home centroids …")
    user_centroids = rp.build_user_centroids(train_df, user_dec, item_dec, coords_pl)
    n_home = int(np.sum(~np.isnan(user_centroids[:, 0])))
    log.info(f"  {n_home:,} / {len(user_centroids):,} users have a home centroid")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    base_metrics = rp.load_base_metrics([m["pred_name"] for m in PROXIMITY_MODELS])

    models_caches = []
    for model in PROXIMITY_MODELS:
        log.info(f"\n── Building cache: {model['pred_name']} ──")
        cache = rp.compute_candidates(
            model, train_df, test_df, user_dec, item_dec,
            user_centroids, coords_pl, top_k,
        )
        if cache is not None:
            models_caches.append((model, cache))

    if not models_caches:
        log.error("No caches built — embeddings missing. Skipping grid search.")
        return

    grid_df = rp.run_grid_search(
        models_caches, rp.ALPHA_GRID, rp.BANDWIDTH_GRID, base_metrics,
    )

    grid_csv = Path("results/proximity_grid_search.csv")
    grid_csv.parent.mkdir(parents=True, exist_ok=True)
    if grid_csv.exists():
        existing = pl.read_csv(grid_csv)
        grid_df  = pl.concat([existing, grid_df], how="diagonal")
    grid_df.write_csv(grid_csv)
    log.info(f"\nGrid results appended → {grid_csv}  ({len(grid_df):,} total rows)")

    best_df = (
        grid_df.sort("ndcg_at_10", descending=True)
        .group_by("pred_name")
        .first()
        .sort("ndcg_at_10", descending=True)
    )
    print(f"\n{'='*82}")
    print(f"  {'Model':<36}  {'α':>5}  {'BW(km)':>7}  {'R@10':>6}  {'nDCG':>7}  {'Δ nDCG':>8}")
    print(f"  {'-'*34}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*8}")
    for r in best_df.iter_rows(named=True):
        print(f"  {r['pred_name']:<36}  {r['alpha']:>5.2f}  {r['bandwidth_km']:>7.1f}  "
              f"{r['recall_at_10']:>6.4f}  {r['ndcg_at_10']:>7.4f}  {r['delta_ndcg']:>+8.4f}")
    print(f"{'='*82}\n")

    for model, cache in models_caches:
        best = (
            grid_df.filter(pl.col("pred_name") == model["pred_name"])
            .sort("ndcg_at_10", descending=True)
            .row(0, named=True)
        )
        rp.rerank_and_save(
            model, cache,
            alpha        = best["alpha"],
            bandwidth_km = best["bandwidth_km"],
            top_k        = top_k,
            orig_metrics = base_metrics.get(model["pred_name"]),
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k",        type=int, default=50)
    parser.add_argument("--no-proximity", action="store_true")
    parser.add_argument("--no-training",  action="store_true",
                        help="Skip training; run proximity grid search on existing embeddings")
    parser.add_argument("--full-test",    action="store_true",
                        help="Evaluate on full expanded item set (default: original CBGs only)")
    args = parser.parse_args()

    original_test = not args.full_test
    if original_test:
        log.info("Test split: original top-2000 CBG restaurants only (FOODIE_ORIGINAL_TEST=1)")
    else:
        log.info("Test split: full expanded item set")

    total_t0 = time.time()

    if not args.no_training:
        for run in RUNS:
            run_single(run, original_test)

    if not args.no_proximity:
        run_proximity_grid(args.top_k)

    log.info(f"\nTotal wall time: {timedelta(seconds=int(time.time() - total_t0))}")


if __name__ == "__main__":
    main()
