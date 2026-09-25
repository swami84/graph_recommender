#!/usr/bin/env python3
"""rerank_proximity.py — Post-inference proximity re-ranking with alpha/bandwidth grid search.

Architecture:
  1. compute_candidates()  — full matmul (n_test × n_items), vectorized top-K + haversine.
                             Result cached in CandidateCache; score matrix freed immediately.
  2. evaluate_blend()      — pure numpy blend of cached scores at any (alpha, bandwidth).
                             Called hundreds of times during grid search at negligible cost.

Single-model mode (backward compatible):
    python -m foodie.modeling.rerank_proximity --model kgat_all --alpha 0.3 --bandwidth 10

Grid-search mode (sweeps alpha × bandwidth on GRID_SEARCH_MODELS):
    python -m foodie.modeling.rerank_proximity --grid-search
    python -m foodie.modeling.rerank_proximity --grid-search --top-k 100
"""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch

from foodie.modeling.model_results import save_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rerank_prox")

EMB_DIR       = Path("data/embeddings")
PRED_DIR      = Path("data/predictions")
REST_ENRICHED = Path("data/restaurants_enriched.parquet")
GRID_CSV      = Path("results/proximity_grid_search.csv")
FOCUSED_GRID_CSV = Path("results/llm_feature_proximity_grid.csv")

# ── Model registry ─────────────────────────────────────────────────────────────

MODELS = [
    {"pred_name": "lightgcn_3l_all_pubsplit", "emb_user": "lightgcn_3l_all_pubsplit_user_embeddings.pt", "emb_item": "lightgcn_3l_all_pubsplit_item_embeddings.pt", "csv_model": "LightGCN", "csv_variant": "LLM v2 publication split", "n_layers": 3, "skip_groups": ""},
    {"pred_name": "kgat_sal_all_T3_nospatial_pubsplit", "emb_user": "kgat_sal_all_T3_nospatial_pubsplit_user_embeddings.pt", "emb_item": "kgat_sal_all_T3_nospatial_pubsplit_item_embeddings.pt", "csv_model": "KGAT-SAL", "csv_variant": "LLM v2 publication split", "n_layers": 4, "skip_groups": ""},
    {"pred_name": "infonce_kgat_sal_all_T3_nospatial_pubsplit", "emb_user": "infonce_kgat_sal_all_T3_nospatial_pubsplit_user_embeddings.pt", "emb_item": "infonce_kgat_sal_all_T3_nospatial_pubsplit_item_embeddings.pt", "csv_model": "InfoNCE-KGAT-SAL", "csv_variant": "LLM v2 publication split", "n_layers": 4, "skip_groups": ""},
    {"pred_name": "lightgcn_3l_all",           "emb_user": "lightgcn_3l_all_user_embeddings.pt",              "emb_item": "lightgcn_3l_all_item_embeddings.pt",              "csv_model": "LightGCN",   "csv_variant": "all features", "n_layers": 3, "skip_groups": ""},
    {"pred_name": "lightgcn_3l_llm",           "emb_user": "lightgcn_3l_skip_llm_user_embeddings.pt",         "emb_item": "lightgcn_3l_skip_llm_item_embeddings.pt",         "csv_model": "LightGCN",   "csv_variant": "skip llm",     "n_layers": 3, "skip_groups": "llm"},
    {"pred_name": "lightgcn_4l_all",           "emb_user": "lightgcn_4l_all_user_embeddings.pt",              "emb_item": "lightgcn_4l_all_item_embeddings.pt",              "csv_model": "LightGCN",   "csv_variant": "all features", "n_layers": 4, "skip_groups": ""},
    {"pred_name": "lightgcn_4l_llm",           "emb_user": "lightgcn_4l_skip_llm_user_embeddings.pt",         "emb_item": "lightgcn_4l_skip_llm_item_embeddings.pt",         "csv_model": "LightGCN",   "csv_variant": "skip llm",     "n_layers": 4, "skip_groups": "llm"},
    {"pred_name": "kgat_all",                  "emb_user": "kgat_all_user_embeddings.pt",                     "emb_item": "kgat_all_item_embeddings.pt",                     "csv_model": "KGAT",       "csv_variant": "all features", "n_layers": 4, "skip_groups": ""},
    {"pred_name": "kgat_llm",                  "emb_user": "kgat_skip_llm_user_embeddings.pt",                "emb_item": "kgat_skip_llm_item_embeddings.pt",                "csv_model": "KGAT",       "csv_variant": "skip llm",     "n_layers": 4, "skip_groups": "llm"},
    {"pred_name": "radar_all",                 "emb_user": "radar_all_user_embeddings.pt",                    "emb_item": "radar_all_item_embeddings.pt",                    "csv_model": "RaDAR",      "csv_variant": "all features", "n_layers": 4, "skip_groups": ""},
    {"pred_name": "radar_llm",                 "emb_user": "radar_skip_llm_user_embeddings.pt",               "emb_item": "radar_skip_llm_item_embeddings.pt",               "csv_model": "RaDAR",      "csv_variant": "skip llm",     "n_layers": 4, "skip_groups": "llm"},
    {"pred_name": "simgcl_llm",                "emb_user": "simgcl_skip_llm_user_embeddings.pt",              "emb_item": "simgcl_skip_llm_item_embeddings.pt",              "csv_model": "SimGCL",     "csv_variant": "skip llm",     "n_layers": 4, "skip_groups": "llm"},
    {"pred_name": "infonce_llm",               "emb_user": "infonce_skip_llm_user_embeddings.pt",             "emb_item": "infonce_skip_llm_item_embeddings.pt",             "csv_model": "InfoNCE",    "csv_variant": "skip llm",     "n_layers": 4, "skip_groups": "llm"},
    {"pred_name": "simgcl_hrcl_llm",           "emb_user": "simgcl_hrcl_skip_llm_user_embeddings.pt",        "emb_item": "simgcl_hrcl_skip_llm_item_embeddings.pt",        "csv_model": "SimGCL-HRCL","csv_variant": "skip llm",     "n_layers": 4, "skip_groups": "llm"},
    {"pred_name": "hek_cl_all",                "emb_user": "hek_cl_all_user_embeddings.pt",                   "emb_item": "hek_cl_all_item_embeddings.pt",                   "csv_model": "HEK-CL",     "csv_variant": "all features", "n_layers": 6, "skip_groups": ""},
    {"pred_name": "hek_cl_llm",                "emb_user": "hek_cl_skip_llm_user_embeddings.pt",              "emb_item": "hek_cl_skip_llm_item_embeddings.pt",              "csv_model": "HEK-CL",     "csv_variant": "skip llm",     "n_layers": 4, "skip_groups": "llm"},
    {"pred_name": "infonce_all",               "emb_user": "infonce_all_user_embeddings.pt",                   "emb_item": "infonce_all_item_embeddings.pt",                   "csv_model": "InfoNCE",    "csv_variant": "all features", "n_layers": 4, "skip_groups": ""},
    {"pred_name": "infonce_kgat_sal_all_T3_nospatial", "emb_user": "infonce_kgat_sal_all_T3_nospatial_user_embeddings.pt", "emb_item": "infonce_kgat_sal_all_T3_nospatial_item_embeddings.pt", "csv_model": "InfoNCE-KGAT-SAL", "csv_variant": "all features T=3", "n_layers": 4, "skip_groups": ""},
    {"pred_name": "ifl_gcl_kg_all",            "emb_user": "ifl_gcl_kg_all_user_embeddings.pt",               "emb_item": "ifl_gcl_kg_all_item_embeddings.pt",               "csv_model": "IFL-GCL-KG", "csv_variant": "all features", "n_layers": 4, "skip_groups": ""},
    {"pred_name": "ifl_gcl_kg_llm",            "emb_user": "ifl_gcl_kg_skip_llm_user_embeddings.pt",          "emb_item": "ifl_gcl_kg_skip_llm_item_embeddings.pt",          "csv_model": "IFL-GCL-KG", "csv_variant": "skip llm",     "n_layers": 4, "skip_groups": "llm"},
    {"pred_name": "kgat_sal_all_T3_nospatial", "emb_user": "kgat_sal_all_T3_nospatial_user_embeddings.pt",    "emb_item": "kgat_sal_all_T3_nospatial_item_embeddings.pt",    "csv_model": "KGAT-SAL",   "csv_variant": "all features", "n_layers": 4, "skip_groups": ""},
]

MODELS_BY_NAME = {m["pred_name"]: m for m in MODELS}

# Grid search targets (best-performing variant per model family)
GRID_SEARCH_MODELS = [
    "infonce_kgat_sal_all_T3_nospatial_pubsplit",
    "kgat_sal_all_T3_nospatial_pubsplit",
    "lightgcn_3l_all_pubsplit",
]

ALPHA_GRID     = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
BANDWIDTH_GRID = [5.0, 10.0, 20.0, 50.0]


# ── Math helpers ────────────────────────────────────────────────────────────────

def haversine_matrix(
    lat1: np.ndarray, lng1: np.ndarray,
    lat2: np.ndarray, lng2: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine — all inputs same broadcast-compatible shape, returns km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def normalize_rows(x: np.ndarray) -> np.ndarray:
    mins = x.min(axis=1, keepdims=True)
    maxs = x.max(axis=1, keepdims=True)
    rng  = maxs - mins
    return np.where(rng > 0, (x - mins) / rng, np.ones_like(x))


# ── Data loading ────────────────────────────────────────────────────────────────

def load_restaurant_coords() -> pl.DataFrame:
    return pl.read_parquet(REST_ENRICHED, columns=["place_id", "lat", "lng"])


def build_user_centroids(
    train_df_pd,
    user_dec: dict,
    item_dec: dict,
    coords_pl: pl.DataFrame,
) -> np.ndarray:
    """Returns (n_users, 2) float64 array of median [lat, lng] per user."""
    n_users   = len(user_dec)
    place_ids = [item_dec.get(int(ii)) for ii in train_df_pd["item_idx"].values]

    train_pl = (
        pl.DataFrame({
            "user_idx": pl.Series(train_df_pd["user_idx"].values.astype(np.int64)),
            "place_id": pl.Series(place_ids),
        })
        .filter(pl.col("place_id").is_not_null())
    )

    centroids = (
        train_pl.join(coords_pl, on="place_id", how="inner")
        .group_by("user_idx")
        .agg(pl.col("lat").median(), pl.col("lng").median())
    )

    result  = np.full((n_users, 2), np.nan, dtype=np.float64)
    ui_arr  = centroids["user_idx"].to_numpy()
    lat_arr = centroids["lat"].to_numpy()
    lng_arr = centroids["lng"].to_numpy()
    valid   = (ui_arr >= 0) & (ui_arr < n_users)
    result[ui_arr[valid], 0] = lat_arr[valid]
    result[ui_arr[valid], 1] = lng_arr[valid]
    return result


# ── Candidate cache ─────────────────────────────────────────────────────────────

@dataclass
class CandidateCache:
    topk_emb_idxs:     np.ndarray          # (n_valid, top_k) int32
    topk_model_scores: np.ndarray          # (n_valid, top_k) float32
    topk_dists_km:     np.ndarray          # (n_valid, top_k) float32 — inf where no coord
    true_emb_idxs:     np.ndarray          # (n_valid,) int64 — -1 if true item not in emb
    test_global_u:     np.ndarray          # (n_valid,) global user_idx
    test_global_i:     np.ndarray          # (n_valid,) global item_idx
    emb_item_id_map:   dict                # emb idx → place_id
    emb_dim:           int = 0


def compute_candidates(
    model: dict,
    train_df_pd,
    test_df_pd,
    user_dec: dict,
    item_dec: dict,
    user_centroids: np.ndarray,
    coords_pl: pl.DataFrame,
    top_k: int,
) -> Optional[CandidateCache]:
    u_path = EMB_DIR / model["emb_user"]
    i_path = EMB_DIR / model["emb_item"]
    if not u_path.exists() or not i_path.exists():
        log.warning(f"  Embeddings missing for {model['pred_name']} — skipping")
        return None

    log.info("  Loading embeddings …")
    u_data = torch.load(u_path, map_location="cpu", weights_only=True)
    i_data = torch.load(i_path, map_location="cpu", weights_only=True)
    u_emb: np.ndarray      = u_data["embeddings"].float().numpy()
    i_emb: np.ndarray      = i_data["embeddings"].float().numpy()
    emb_user_id_map: dict  = u_data["id_map"]
    emb_item_id_map: dict  = i_data["id_map"]
    emb_dim                = u_emb.shape[1]

    contrib_to_emb_u = {v: k for k, v in emb_user_id_map.items()}
    place_to_emb_i   = {v: k for k, v in emb_item_id_map.items()}

    # Item lat/lng aligned to embedding indices
    n_emb_items = i_emb.shape[0]
    coord_map   = {r[0]: (r[1], r[2]) for r in coords_pl.iter_rows()}
    emb_lat = np.array(
        [coord_map.get(emb_item_id_map.get(i, ""), (np.nan, np.nan))[0] for i in range(n_emb_items)],
        dtype=np.float32,
    )
    emb_lng = np.array(
        [coord_map.get(emb_item_id_map.get(i, ""), (np.nan, np.nan))[1] for i in range(n_emb_items)],
        dtype=np.float32,
    )

    # Vectorized lookup arrays: global idx → embedding idx
    n_global_u = max(user_dec.keys()) + 1
    n_global_i = max(item_dec.keys()) + 1
    global_u_to_emb = np.full(n_global_u, -1, dtype=np.int64)
    for gu, cid in user_dec.items():
        global_u_to_emb[gu] = contrib_to_emb_u.get(cid, -1)
    global_i_to_emb = np.full(n_global_i, -1, dtype=np.int64)
    for gi, pid in item_dec.items():
        global_i_to_emb[gi] = place_to_emb_i.get(pid, -1)

    # Train exclusion sets in embedding space — polars groupby
    train_u = train_df_pd["user_idx"].values.astype(np.int64)
    train_i = train_df_pd["item_idx"].values.astype(np.int64)
    eu_train = global_u_to_emb[train_u]
    ei_train = global_i_to_emb[train_i]
    valid_train = (eu_train >= 0) & (ei_train >= 0)

    excl_df = (
        pl.DataFrame({"eu": eu_train[valid_train], "ei": ei_train[valid_train]})
        .group_by("eu")
        .agg(pl.col("ei").unique())
    )
    train_excl: dict[int, list] = {r[0]: r[1] for r in excl_df.iter_rows()}

    # Test set
    test_users = test_df_pd["user_idx"].values.astype(np.int64)
    test_items = test_df_pd["item_idx"].values.astype(np.int64)
    eu_test    = global_u_to_emb[test_users]
    valid_mask = eu_test >= 0
    valid_emb_u = eu_test[valid_mask]

    log.info(f"  Full matmul: {valid_mask.sum():,} users × {n_emb_items:,} items "
             f"({valid_mask.sum() * n_emb_items * 4 / 1e9:.1f} GB) …")
    all_scores: np.ndarray = u_emb[valid_emb_u] @ i_emb.T   # (n_valid, n_items)

    log.info("  Masking train items …")
    for i, eu in enumerate(valid_emb_u):
        excl = train_excl.get(int(eu))
        if excl:
            all_scores[i, excl] = -np.inf

    log.info(f"  Top-{top_k} argpartition …")
    tk   = min(top_k, n_emb_items - 1)
    part = np.argpartition(-all_scores, tk, axis=1)[:, :top_k]
    ri   = np.arange(len(valid_emb_u))[:, None]
    srt  = np.argsort(-all_scores[ri, part], axis=1)
    topk_emb_idxs     = part[ri, srt].astype(np.int32)
    topk_model_scores = all_scores[ri, topk_emb_idxs].astype(np.float32)
    del all_scores

    log.info("  Vectorized haversine …")
    topk_lats  = emb_lat[topk_emb_idxs]                                         # (n, top_k)
    topk_lngs  = emb_lng[topk_emb_idxs]
    home_lats  = user_centroids[test_users[valid_mask], 0][:, None].astype(np.float32)
    home_lngs  = user_centroids[test_users[valid_mask], 1][:, None].astype(np.float32)
    has_home   = ~np.isnan(home_lats[:, 0])

    topk_dists = np.full_like(topk_lats, np.inf)
    if has_home.any():
        topk_dists[has_home] = haversine_matrix(
            home_lats[has_home], home_lngs[has_home],
            topk_lats[has_home], topk_lngs[has_home],
        ).astype(np.float32)
    topk_dists = np.where(np.isnan(topk_lats), np.inf, topk_dists)

    return CandidateCache(
        topk_emb_idxs     = topk_emb_idxs,
        topk_model_scores = topk_model_scores,
        topk_dists_km     = topk_dists,
        true_emb_idxs     = global_i_to_emb[test_items[valid_mask]],
        test_global_u     = test_users[valid_mask],
        test_global_i     = test_items[valid_mask],
        emb_item_id_map   = emb_item_id_map,
        emb_dim           = emb_dim,
    )


# ── Blend & evaluate (fast — no matmul) ────────────────────────────────────────

def evaluate_blend(cache: CandidateCache, alpha: float, bandwidth_km: float) -> dict:
    prox = np.exp(-cache.topk_dists_km.astype(np.float64) / bandwidth_km)
    prox[np.isinf(cache.topk_dists_km)] = 0.0

    blended = (
        (1.0 - alpha) * normalize_rows(cache.topk_model_scores.astype(np.float64))
        + alpha        * normalize_rows(prox)
    )

    top10_order = np.argsort(-blended, axis=1)[:, :10]
    top10_emb   = cache.topk_emb_idxs[np.arange(len(blended))[:, None], top10_order]

    matches = top10_emb == cache.true_emb_idxs[:, None]   # (n, 10) — false when true=-1
    hit     = matches.any(axis=1)
    ranks   = np.where(hit, np.argmax(matches, axis=1) + 1.0, np.nan)

    recall = float(np.mean(hit))
    precision = recall / 10.0  # one held-out relevant item per user
    ndcg = float(np.mean(np.where(hit, 1.0 / np.log2(ranks + 1), 0.0)))
    return {
        "precision_at_10": round(precision, 4),
        "recall_at_10": round(recall, 4),
        "ndcg_at_10": round(ndcg, 4),
    }


# ── Grid search ────────────────────────────────────────────────────────────────

def load_base_metrics(pred_names: list) -> dict:
    base = {}
    for name in pred_names:
        p = PRED_DIR / f"{name}_predictions.parquet"
        if not p.exists():
            continue
        df        = pl.read_parquet(p, columns=["rank"])
        hit       = df["rank"].is_not_null().mean()
        ranks_arr = df["rank"].cast(pl.Float64).fill_null(0.0).to_numpy()
        ndcg      = float(np.mean(np.where(ranks_arr > 0, 1.0 / np.log2(ranks_arr + 1), 0.0)))
        base[name] = {"recall_at_10": round(float(hit), 4), "ndcg_at_10": round(ndcg, 4)}
    return base


def run_grid_search(
    models_caches: list,
    alphas: list,
    bandwidths: list,
    base_metrics: dict,
) -> pl.DataFrame:
    rows = []
    for model, cache in models_caches:
        name      = model["pred_name"]
        base_ndcg = base_metrics.get(name, {}).get("ndcg_at_10", 0.0)
        log.info(f"\n  Grid: {name}  ({len(alphas) * len(bandwidths)} combos, base nDCG={base_ndcg:.4f})")
        best_ndcg, best_row = -1.0, None

        for bw in bandwidths:
            for alpha in alphas:
                m = evaluate_blend(cache, alpha, bw)
                row = {
                    "pred_name":      name,
                    "csv_model":      model["csv_model"],
                    "alpha":          alpha,
                    "bandwidth_km":   bw,
                    **m,
                    "delta_ndcg":     round(m["ndcg_at_10"] - base_ndcg, 4),
                    "base_ndcg":      base_ndcg,
                }
                rows.append(row)
                if m["ndcg_at_10"] > best_ndcg:
                    best_ndcg = m["ndcg_at_10"]
                    best_row  = row

        if best_row:
            log.info(f"    Best → α={best_row['alpha']}  bw={best_row['bandwidth_km']} km  "
                     f"nDCG={best_row['ndcg_at_10']:.4f}  Δ={best_row['delta_ndcg']:+.4f}")

    return pl.DataFrame(rows)


# ── Save re-ranked predictions ─────────────────────────────────────────────────

def rerank_and_save(
    model: dict,
    cache: CandidateCache,
    alpha: float,
    bandwidth_km: float,
    top_k: int,
    orig_metrics: Optional[dict] = None,
) -> None:
    prox = np.exp(-cache.topk_dists_km.astype(np.float64) / bandwidth_km)
    prox[np.isinf(cache.topk_dists_km)] = 0.0
    blended = (
        (1.0 - alpha) * normalize_rows(cache.topk_model_scores.astype(np.float64))
        + alpha        * normalize_rows(prox)
    )

    top10_order = np.argsort(-blended, axis=1)[:, :10]
    top10_emb   = cache.topk_emb_idxs[np.arange(len(blended))[:, None], top10_order]
    matches     = top10_emb == cache.true_emb_idxs[:, None]
    hit         = matches.any(axis=1)
    ranks_raw   = np.where(hit, np.argmax(matches, axis=1) + 1, 0).tolist()

    df = pl.DataFrame({
        "user_idx":        cache.test_global_u.tolist(),
        "true_item_idx":   cache.test_global_i.tolist(),
        "top10_item_idxs": top10_emb.tolist(),
        "rank":            [r if r > 0 else None for r in ranks_raw],
        "alpha":           [alpha] * len(cache.test_global_u),
    })

    alpha_tag = f"a{int(alpha * 100):02d}"
    out_path  = PRED_DIR / f"{model['pred_name']}_prox_{alpha_tag}_predictions.parquet"
    df.write_parquet(out_path)

    metrics = evaluate_blend(cache, alpha, bandwidth_km)
    log.info(f"  Saved → {out_path}  ({len(df):,} rows)")
    log.info(f"  P@10={metrics['precision_at_10']}  R@10={metrics['recall_at_10']}  nDCG@10={metrics['ndcg_at_10']}")
    if orig_metrics:
        delta = metrics["ndcg_at_10"] - orig_metrics.get("ndcg_at_10", 0.0)
        log.info(f"  Δ nDCG vs base: {delta:+.4f}  (base={orig_metrics.get('ndcg_at_10', 0.0):.4f})")

    save_result(
        model       = model["csv_model"],
        variant     = f"{model['csv_variant']} +proximity",
        precision   = metrics["precision_at_10"],
        recall      = metrics["recall_at_10"],
        ndcg        = metrics["ndcg_at_10"],
        epochs      = 0,
        emb_dim     = cache.emb_dim,
        n_layers    = model["n_layers"],
        skip_groups = model["skip_groups"],
        notes       = f"proximity_reranked alpha={alpha} bw={bandwidth_km}km top_k={top_k}",
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",       type=float, default=0.3)
    parser.add_argument("--bandwidth",   type=float, default=10.0)
    parser.add_argument("--top-k",       type=int,   default=50)
    parser.add_argument("--model",       type=str,   default=None,
                        help="Single pred_name for single-model mode")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated pred_names; limits grid or normal mode")
    parser.add_argument("--grid-search", action="store_true",
                        help="Sweep alpha × bandwidth grid on GRID_SEARCH_MODELS")
    parser.add_argument("--publication-split", action="store_true",
                        help="Use the chronological train/validation/test publication split")
    parser.add_argument("--grid-output", default=str(FOCUSED_GRID_CSV),
                        help="Grid CSV path (focused output by default; never overwrites legacy grid)")
    args = parser.parse_args()

    log.info("Loading restaurant coordinates …")
    coords_pl = load_restaurant_coords()

    log.info("Loading interactions …")
    from recommendation_gnn import load_interactions, load_interaction_splits
    if args.publication_split:
        train_df, _, test_df, user_enc, item_enc = load_interaction_splits()
    else:
        train_df, test_df, user_enc, item_enc = load_interactions()
    user_dec = {v: k for k, v in user_enc.items()}
    item_dec = {v: k for k, v in item_enc.items()}

    log.info("Computing user home centroids …")
    user_centroids = build_user_centroids(train_df, user_dec, item_dec, coords_pl)
    n_home = int(np.sum(~np.isnan(user_centroids[:, 0])))
    log.info(f"  {n_home:,} / {len(user_centroids):,} users have a home centroid")

    PRED_DIR.mkdir(parents=True, exist_ok=True)

    if args.grid_search:
        requested = [v.strip() for v in args.models.split(",") if v.strip()]
        names = requested or GRID_SEARCH_MODELS
        models = [MODELS_BY_NAME[n] for n in names if n in MODELS_BY_NAME]
        n_combos = len(ALPHA_GRID) * len(BANDWIDTH_GRID)
        log.info(f"\nGrid search: {len(models)} models × {n_combos} (α, bw) combos = "
                 f"{len(models) * n_combos} total evaluations")
        log.info(f"  α:         {ALPHA_GRID}")
        log.info(f"  bandwidth: {BANDWIDTH_GRID} km")

        base_metrics = load_base_metrics(names)

        models_caches = []
        for model in models:
            log.info(f"\n── Building cache: {model['pred_name']} ──")
            cache = compute_candidates(
                model, train_df, test_df, user_dec, item_dec,
                user_centroids, coords_pl, args.top_k,
            )
            if cache is not None:
                models_caches.append((model, cache))

        log.info(f"\n{'='*65}")
        log.info("  Running grid search …")
        log.info(f"{'='*65}")
        grid_df = run_grid_search(models_caches, ALPHA_GRID, BANDWIDTH_GRID, base_metrics)

        grid_output = Path(args.grid_output)
        grid_output.parent.mkdir(parents=True, exist_ok=True)
        grid_df.write_csv(grid_output)
        log.info(f"\nFull grid saved → {grid_output}")

        # Best per model summary
        best_df = (
            grid_df.sort("ndcg_at_10", descending=True)
            .group_by("pred_name")
            .first()
            .sort("ndcg_at_10", descending=True)
        )
        print(f"\n{'='*82}")
        print(f"  {'Model':<32}  {'α':>5}  {'BW(km)':>7}  {'R@10':>6}  {'nDCG':>7}  {'Δ nDCG':>8}")
        print(f"  {'-'*30}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*8}")
        for r in best_df.iter_rows(named=True):
            print(f"  {r['pred_name']:<32}  {r['alpha']:>5.2f}  {r['bandwidth_km']:>7.1f}  "
                  f"{r['recall_at_10']:>6.4f}  {r['ndcg_at_10']:>7.4f}  {r['delta_ndcg']:>+8.4f}")
        print(f"{'='*82}\n")

        # Save best parquet + model_results.csv row per model
        for model, cache in models_caches:
            best = (
                grid_df.filter(pl.col("pred_name") == model["pred_name"])
                .sort("ndcg_at_10", descending=True)
                .row(0, named=True)
            )
            rerank_and_save(
                model, cache,
                alpha        = best["alpha"],
                bandwidth_km = best["bandwidth_km"],
                top_k        = args.top_k,
                orig_metrics = base_metrics.get(model["pred_name"]),
            )

    else:
        # Single-model / all-models mode (backward compatible)
        requested = [v.strip() for v in args.models.split(",") if v.strip()]
        if args.model:
            requested = [args.model]
        models = MODELS if not requested else [m for m in MODELS if m["pred_name"] in requested]
        if not models:
            log.error(f"No model matching --model '{args.model}'")
            return

        base_metrics = load_base_metrics([m["pred_name"] for m in models])

        for model in models:
            log.info(f"\n── {model['pred_name']} ──")
            cache = compute_candidates(
                model, train_df, test_df, user_dec, item_dec,
                user_centroids, coords_pl, args.top_k,
            )
            if cache is None:
                continue
            rerank_and_save(
                model, cache,
                alpha        = args.alpha,
                bandwidth_km = args.bandwidth,
                top_k        = args.top_k,
                orig_metrics = base_metrics.get(model["pred_name"]),
            )


if __name__ == "__main__":
    main()
