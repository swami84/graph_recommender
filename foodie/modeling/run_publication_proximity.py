#!/usr/bin/env python3
"""Leakage-safe proximity tuning for the selected publication model.

The proximity hyperparameters are selected from validation metrics averaged
over seeds 42/43/44.  The selected configuration is then evaluated exactly
once on each seed's test predictions.  User location is estimated exclusively
from training interactions; test ranking excludes both train and validation
restaurants.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from foodie.modeling.recommendation_gnn import load_interaction_splits


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("publication_proximity")

EMB_DIR = Path(os.environ.get("FOODIE_EMBEDDING_DIR", "data/embeddings"))
PRED_DIR = Path(os.environ.get("FOODIE_PREDICTION_DIR", "data/predictions"))
RESULT_DIR = Path(os.environ.get("FOODIE_PROXIMITY_RESULT_DIR", "results"))
RESTAURANTS = Path(os.environ.get(
    "FOODIE_RESTAURANTS_ENR", "data/restaurants_enriched.parquet"
))
CORE_RESULTS = Path(os.environ.get(
    "FOODIE_PUBLICATION_RESULTS", "results/publication_core_results.csv"
))
PREFIX = os.environ.get("FOODIE_PROXIMITY_PREFIX", "publication_proximity")
SEEDS = (42, 43, 44)

ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
BANDWIDTHS_KM = (5.0, 10.0, 20.0, 50.0)
CANDIDATE_DEPTHS = (25, 50, 100)
MAX_DEPTH = max(CANDIDATE_DEPTHS)

VAL_GRID = RESULT_DIR / f"{PREFIX}_validation_grid.csv"
SELECTION = RESULT_DIR / f"{PREFIX}_selection.json"
TEST_RESULTS = RESULT_DIR / f"{PREFIX}_test_by_seed.csv"
SUMMARY = RESULT_DIR / f"{PREFIX}_summary.md"


def haversine(lat1, lon1, lat2, lon2):
    radius = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1))
        * np.cos(np.radians(lat2))
        * np.sin(dlon / 2.0) ** 2
    )
    return radius * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def normalize_rows(values: np.ndarray) -> np.ndarray:
    lo = values.min(axis=1, keepdims=True)
    hi = values.max(axis=1, keepdims=True)
    span = hi - lo
    return np.divide(values - lo, span, out=np.zeros_like(values), where=span > 0)


def metrics(true_items: np.ndarray, ranked: np.ndarray) -> dict[str, float]:
    matches = ranked == true_items[:, None]
    hit = matches.any(axis=1)
    rank = np.where(hit, matches.argmax(axis=1) + 1, 0)
    ndcg = np.zeros(len(rank), dtype=np.float64)
    ndcg[hit] = 1.0 / np.log2(rank[hit] + 1)
    return {
        "hit_at_10": float(hit.mean()),
        "ndcg_at_10": float(ndcg.mean()),
    }


def embedding_paths(seed: int) -> tuple[Path, Path]:
    stem = f"kgat_sal_all_T3_pubsplit_seed{seed}"
    return (
        EMB_DIR / f"{stem}_user_embeddings.pt",
        EMB_DIR / f"{stem}_item_embeddings.pt",
    )


def load_embeddings(seed: int):
    user_path, item_path = embedding_paths(seed)
    user = torch.load(user_path, map_location="cpu", weights_only=True)
    item = torch.load(item_path, map_location="cpu", weights_only=True)
    return user, item


def build_id_maps(user_data, item_data, user_dec, item_dec):
    contributor_to_emb = {value: key for key, value in user_data["id_map"].items()}
    place_to_emb = {value: key for key, value in item_data["id_map"].items()}
    global_user_to_emb = np.array(
        [contributor_to_emb[user_dec[i]] for i in range(len(user_dec))], dtype=np.int64
    )
    global_item_to_emb = np.array(
        [place_to_emb[item_dec[i]] for i in range(len(item_dec))], dtype=np.int64
    )
    return global_user_to_emb, global_item_to_emb


def group_exclusions(frames, global_item_to_emb):
    frame = pd.concat(frames, ignore_index=True)
    result = {}
    for user_idx, group in frame.groupby("user_idx", sort=False):
        result[int(user_idx)] = global_item_to_emb[
            group["item_idx"].to_numpy(dtype=np.int64)
        ]
    return result


def training_centroids(train_df, item_dec, restaurant_coords):
    frame = train_df[["user_idx", "item_idx"]].copy()
    frame["place_id"] = frame["item_idx"].map(item_dec)
    frame = frame.merge(restaurant_coords, on="place_id", how="left")
    centroids = frame.groupby("user_idx")[["lat", "lng"]].median()
    result = np.full((int(train_df.user_idx.max()) + 1, 2), np.nan, dtype=np.float32)
    idx = centroids.index.to_numpy(dtype=np.int64)
    result[idx] = centroids[["lat", "lng"]].to_numpy(dtype=np.float32)
    return result


def aligned_item_coordinates(item_id_map, restaurant_coords):
    lookup = restaurant_coords.set_index("place_id")[["lat", "lng"]]
    place_ids = [item_id_map[i] for i in range(len(item_id_map))]
    return lookup.reindex(place_ids).to_numpy(dtype=np.float32)


def compute_candidates(
    seed: int,
    target_df: pd.DataFrame,
    exclusion_frames: list[pd.DataFrame],
    user_dec: dict,
    item_dec: dict,
    centroids: np.ndarray,
    restaurant_coords: pd.DataFrame,
    device: torch.device,
    chunk_size: int,
):
    user_data, item_data = load_embeddings(seed)
    global_user_to_emb, global_item_to_emb = build_id_maps(
        user_data, item_data, user_dec, item_dec
    )
    exclusions = group_exclusions(exclusion_frames, global_item_to_emb)

    user_embeddings = user_data["embeddings"].float().to(device)
    item_embeddings = item_data["embeddings"].float().to(device)
    item_coords = aligned_item_coordinates(item_data["id_map"], restaurant_coords)

    global_users = target_df["user_idx"].to_numpy(dtype=np.int64)
    global_true = target_df["item_idx"].to_numpy(dtype=np.int64)
    embedded_users = global_user_to_emb[global_users]
    embedded_true = global_item_to_emb[global_true]
    n = len(target_df)

    top_items = np.empty((n, MAX_DEPTH), dtype=np.int32)
    top_scores = np.empty((n, MAX_DEPTH), dtype=np.float32)

    log.info(
        "Seed %d: ranking %s users against %s items on %s",
        seed, f"{n:,}", f"{len(item_embeddings):,}", device,
    )
    with torch.inference_mode():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            eu = torch.as_tensor(embedded_users[start:end], device=device)
            scores = user_embeddings[eu] @ item_embeddings.T

            row_parts, col_parts = [], []
            for local_row, global_user in enumerate(global_users[start:end]):
                cols = exclusions.get(int(global_user))
                if cols is not None and len(cols):
                    row_parts.append(np.full(len(cols), local_row, dtype=np.int64))
                    col_parts.append(cols)
            if row_parts:
                rows = torch.as_tensor(np.concatenate(row_parts), device=device)
                cols = torch.as_tensor(np.concatenate(col_parts), device=device)
                scores[rows, cols] = -torch.inf

            values, indices = torch.topk(scores, k=MAX_DEPTH, dim=1)
            top_items[start:end] = indices.cpu().numpy().astype(np.int32)
            top_scores[start:end] = values.cpu().numpy().astype(np.float32)
            if end == n or end % (chunk_size * 20) == 0:
                log.info("  seed %d: %s/%s users", seed, f"{end:,}", f"{n:,}")

    homes = centroids[global_users]
    candidate_coords = item_coords[top_items]
    distances = haversine(
        homes[:, None, 0], homes[:, None, 1],
        candidate_coords[:, :, 0], candidate_coords[:, :, 1],
    ).astype(np.float32)
    invalid = np.isnan(homes[:, 0])[:, None] | np.isnan(candidate_coords[:, :, 0])
    distances[invalid] = np.inf

    del user_embeddings, item_embeddings
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "global_users": global_users,
        "global_true": global_true,
        "embedded_true": embedded_true,
        "top_items": top_items,
        "top_scores": top_scores,
        "distances": distances,
        "item_id_map": item_data["id_map"],
    }


def rerank(cache, alpha: float, bandwidth: float, depth: int):
    scores = cache["top_scores"][:, :depth].astype(np.float64)
    distances = cache["distances"][:, :depth].astype(np.float64)
    proximity = np.exp(-distances / bandwidth)
    proximity[np.isinf(distances)] = 0.0
    blended = (1.0 - alpha) * normalize_rows(scores) + alpha * normalize_rows(proximity)
    order = np.argpartition(-blended, kth=9, axis=1)[:, :10]
    rows = np.arange(len(order))[:, None]
    order = order[rows, np.argsort(-blended[rows, order], axis=1)]
    ranked = cache["top_items"][:, :depth][rows, order]
    ranked_scores = blended[rows, order]
    return ranked, ranked_scores


def validation_grid(cache, seed: int, completed: set[tuple] | None = None):
    completed = completed or set()
    rows = []
    for depth in CANDIDATE_DEPTHS:
        for bandwidth in BANDWIDTHS_KM:
            for alpha in ALPHAS:
                key = (seed, float(alpha), float(bandwidth), int(depth))
                if key in completed:
                    continue
                ranked, _ = rerank(cache, alpha, bandwidth, depth)
                result = metrics(cache["embedded_true"], ranked)
                rows.append({
                    "seed": seed,
                    "alpha": alpha,
                    "bandwidth_km": bandwidth,
                    "candidate_depth": depth,
                    **result,
                })
    return rows


def save_test_predictions(cache, ranked, ranked_scores, seed, config):
    item_map = cache["item_id_map"]
    place_ids = [[item_map[int(i)] for i in row] for row in ranked]
    output = pd.DataFrame({
        "user_idx": cache["global_users"],
        "true_item_idx": cache["global_true"],
        "top10_item_embedding_idxs": list(ranked),
        "top10_place_ids": place_ids,
        "top10_blended_scores": list(ranked_scores.astype(np.float32)),
        "alpha": config["alpha"],
        "bandwidth_km": config["bandwidth_km"],
        "candidate_depth": config["candidate_depth"],
    })
    path = PRED_DIR / f"kgat_sal_full_llm_pubsplit_seed{seed}_proximity_predictions.parquet"
    output.to_parquet(path, index=False)
    log.info("Saved %s", path)


def main():
    global ALPHAS, BANDWIDTHS_KM, CANDIDATE_DEPTHS, MAX_DEPTH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument(
        "--expanded-grid", action="store_true",
        help="Extend the validation-only search beyond the prior boundary values",
    )
    parser.add_argument(
        "--resume-grid", action="store_true",
        help="Reuse validation configurations already present in the output grid",
    )
    parser.add_argument(
        "--deep-refinement", action="store_true",
        help="Refine a boundary optimum at depths 1600 and 3200 near alpha/bandwidth winner",
    )
    args = parser.parse_args()

    if args.expanded_grid:
        ALPHAS = tuple(np.round(np.arange(0.1, 1.0, 0.1), 1))
        BANDWIDTHS_KM = (1.0, 2.5, 5.0, 10.0, 20.0, 50.0)
        CANDIDATE_DEPTHS = (25, 50, 100, 200, 400, 800)
        MAX_DEPTH = max(CANDIDATE_DEPTHS)
    if args.deep_refinement:
        ALPHAS = (0.4, 0.5, 0.6, 0.7, 0.8)
        BANDWIDTHS_KM = (1.0, 2.5, 5.0)
        CANDIDATE_DEPTHS = (1600, 3200)
        MAX_DEPTH = max(CANDIDATE_DEPTHS)

    RESULT_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df, user_enc, item_enc = load_interaction_splits()
    user_dec = {value: key for key, value in user_enc.items()}
    item_dec = {value: key for key, value in item_enc.items()}
    coords = pd.read_parquet(RESTAURANTS, columns=["place_id", "lat", "lng"])
    centroids = training_centroids(train_df, item_dec, coords)

    grid_rows = []
    completed = set()
    if args.resume_grid and VAL_GRID.exists():
        prior_grid = pd.read_csv(VAL_GRID)
        grid_rows = prior_grid.to_dict("records")
        completed = {
            (int(row.seed), float(row.alpha), float(row.bandwidth_km),
             int(row.candidate_depth))
            for row in prior_grid.itertuples(index=False)
        }
        log.info("Resuming with %d completed validation configurations", len(completed))
    for seed in SEEDS:
        cache = compute_candidates(
            seed, val_df, [train_df], user_dec, item_dec, centroids, coords,
            device, args.chunk_size,
        )
        grid_rows.extend(validation_grid(cache, seed, completed))
        pd.DataFrame(grid_rows).to_csv(VAL_GRID, index=False)

    grid = pd.DataFrame(grid_rows)
    aggregate = (
        grid.groupby(["alpha", "bandwidth_km", "candidate_depth"])
        .agg(
            val_hit_at_10_mean=("hit_at_10", "mean"),
            val_hit_at_10_std=("hit_at_10", "std"),
            val_ndcg_at_10_mean=("ndcg_at_10", "mean"),
            val_ndcg_at_10_std=("ndcg_at_10", "std"),
        )
        .reset_index()
        .sort_values(
            ["val_ndcg_at_10_mean", "val_hit_at_10_mean", "candidate_depth"],
            ascending=[False, False, True],
        )
    )
    best = aggregate.iloc[0].to_dict()
    config = {
        "model": "KGAT-SAL",
        "condition": "full_llm",
        "selection_metric": "mean validation NDCG@10 across seeds 42,43,44",
        "alpha": float(best["alpha"]),
        "bandwidth_km": float(best["bandwidth_km"]),
        "candidate_depth": int(best["candidate_depth"]),
        "val_hit_at_10_mean": float(best["val_hit_at_10_mean"]),
        "val_hit_at_10_std": float(best["val_hit_at_10_std"]),
        "val_ndcg_at_10_mean": float(best["val_ndcg_at_10_mean"]),
        "val_ndcg_at_10_std": float(best["val_ndcg_at_10_std"]),
        "search_space": {
            "alphas": list(ALPHAS),
            "bandwidths_km": list(BANDWIDTHS_KM),
            "candidate_depths": list(CANDIDATE_DEPTHS),
        },
    }
    SELECTION.write_text(json.dumps(config, indent=2) + "\n")
    log.info("Frozen proximity configuration: %s", config)

    test_rows = []
    for seed in SEEDS:
        cache = compute_candidates(
            seed, test_df, [train_df, val_df], user_dec, item_dec, centroids, coords,
            device, args.chunk_size,
        )
        ranked, ranked_scores = rerank(
            cache, config["alpha"], config["bandwidth_km"], config["candidate_depth"]
        )
        result = metrics(cache["embedded_true"], ranked)
        test_rows.append({"seed": seed, **result})
        save_test_predictions(cache, ranked, ranked_scores, seed, config)

    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(TEST_RESULTS, index=False)
    raw = pd.read_csv(CORE_RESULTS)
    raw = raw[(raw.model == "KGAT-SAL") & (raw.condition == "full_llm")]
    raw = raw.sort_values("timestamp").drop_duplicates("seed", keep="last")
    merged = test_results.merge(
        raw[["seed", "test_hit_at_10", "test_ndcg_at_10"]], on="seed"
    )
    merged["delta_hit_at_10"] = merged.hit_at_10 - merged.test_hit_at_10
    merged["delta_ndcg_at_10"] = merged.ndcg_at_10 - merged.test_ndcg_at_10

    lines = [
        "# Publication proximity reranking",
        "",
        "Hyperparameters were selected exclusively by mean validation NDCG@10 across "
        "seeds 42, 43, and 44. User centroids use training interactions only. Test "
        "ranking excludes both training and validation restaurants.",
        "",
        "## Frozen configuration",
        "",
        f"- Blend weight (alpha): {config['alpha']}",
        f"- Distance bandwidth: {config['bandwidth_km']} km",
        f"- Candidate depth: {config['candidate_depth']}",
        f"- Validation Hit@10: {config['val_hit_at_10_mean']:.6f} ± {config['val_hit_at_10_std']:.6f}",
        f"- Validation NDCG@10: {config['val_ndcg_at_10_mean']:.6f} ± {config['val_ndcg_at_10_std']:.6f}",
        "",
        "## One-time test evaluation",
        "",
        merged.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Proximity Hit@10: **{merged.hit_at_10.mean():.6f} ± {merged.hit_at_10.std():.6f}**",
        f"Proximity NDCG@10: **{merged.ndcg_at_10.mean():.6f} ± {merged.ndcg_at_10.std():.6f}**",
        f"Mean Hit@10 change: **{merged.delta_hit_at_10.mean():+.6f}**",
        f"Mean NDCG@10 change: **{merged.delta_ndcg_at_10.mean():+.6f}**",
        "",
    ]
    SUMMARY.write_text("\n".join(lines))
    log.info("Wrote %s and %s", TEST_RESULTS, SUMMARY)


if __name__ == "__main__":
    main()
