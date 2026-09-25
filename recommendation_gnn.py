#!/usr/bin/env python3
"""
recommendation_gnn.py — Phase 4: LightGCN-style GNN recommendation.

Uses PyTorch Geometric HeteroData to model the reviewer–restaurant bipartite
graph, with optional side nodes (Cuisine, CBG) for richer message passing.

Loss:   BPR (Bayesian Personalised Ranking)
Eval:   Precision@10, Recall@10, NDCG@10 (same leave-one-out split as baseline)
Output: data/embeddings/{reviewer,restaurant}_embeddings.pt
        models/lightgcn_gnn.pt

Usage:
    python recommendation_gnn.py                 # train + evaluate
    python recommendation_gnn.py --epochs 100
    python recommendation_gnn.py --recommend CONTRIBUTOR_ID
"""

import argparse
import copy
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.checkpoint import checkpoint as grad_ckpt

from model_results import save_result
from publication_results import condition_from_skip, save_publication_result


class _SparseMmChunkedBF16(torch.autograd.Function):
    """Chunked sparse-dense mm for BF16 embeddings via float32.
    Avoids materializing the full 2 GiB float32 copy at D=2048 by processing
    256-dim strips; peak float32 per chunk ≈ 0.26 GiB instead of 2.03 GiB."""
    _CHUNK = 256

    @staticmethod
    def forward(ctx, emb, adj_sp):
        out  = torch.empty_like(emb)
        C    = _SparseMmChunkedBF16._CHUNK
        adj_f = adj_sp.float()
        for d0 in range(0, emb.shape[1], C):
            d1 = min(d0 + C, emb.shape[1])
            with torch.amp.autocast(device_type="cuda", enabled=False):
                out[:, d0:d1] = torch.sparse.mm(adj_f, emb[:, d0:d1].float()).to(emb.dtype)
        ctx.adj_sp = adj_sp
        return out

    @staticmethod
    def backward(ctx, grad_out):
        adj_t    = ctx.adj_sp.t().coalesce()
        grad_emb = torch.empty_like(grad_out)
        C        = _SparseMmChunkedBF16._CHUNK
        for d0 in range(0, grad_out.shape[1], C):
            d1 = min(d0 + C, grad_out.shape[1])
            with torch.amp.autocast(device_type="cuda", enabled=False):
                grad_emb[:, d0:d1] = torch.sparse.mm(
                    adj_t, grad_out[:, d0:d1].float()
                ).to(grad_out.dtype)
        return grad_emb, None


def _sparse_mm_f32(adj: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    """Sparse-dense matmul via float32 (CUDA sparse lacks BF16/FP16 support).
    BF16 inputs use chunked upcast to avoid the full 2 GiB float32 temporary."""
    if emb.dtype == torch.bfloat16:
        return _SparseMmChunkedBF16.apply(emb, adj)
    out_dtype = emb.dtype
    with torch.amp.autocast(device_type="cuda", enabled=False):
        out = torch.sparse.mm(adj.float(), emb.float())
    return out.to(out_dtype)


def _is_main_rank() -> bool:
    """True when not in distributed mode or on rank 0."""
    return not dist.is_initialized() or dist.get_rank() == 0


def _sync_grads(model: torch.nn.Module) -> None:
    """AllReduce gradients across distributed ranks (used with ZeRO-2)."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    ws = dist.get_world_size()
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
            p.grad.data /= ws


def init_distributed() -> int:
    """Initialize distributed training with NCCL backend; returns local_rank.
    Safe to call in single-GPU mode (no WORLD_SIZE env var) — returns 0."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def dropout_adj(adj: torch.Tensor, dropout: float, training: bool) -> torch.Tensor:
    """Randomly drop edges from a sparse adj matrix during training (stochastic regularisation)."""
    if not training or dropout == 0.0:
        return adj
    adj = adj.coalesce()
    mask = torch.rand(adj.values().shape[0], device=adj.device) > dropout
    new_vals = adj.values()[mask] / (1.0 - dropout)   # rescale to keep expected degree
    return torch.sparse_coo_tensor(
        adj.indices()[:, mask], new_vals, adj.shape
    ).coalesce()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gnn")
log.setLevel(logging.INFO)
if not log.handlers:
    _lh = logging.StreamHandler()
    _lh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_lh)
    log.propagate = False

REVIEWS_FLAT      = Path(os.environ.get("FOODIE_REVIEWS_FLAT", "data/reviews_flat.parquet"))
RESTAURANTS_ENR   = Path(os.environ.get("FOODIE_RESTAURANTS_ENR", "data/restaurants_enriched.parquet"))
DISH_PROFILES     = Path("data/dish_profiles.parquet")
DISH_SIMILARITIES = Path("data/dish_similarities.parquet")
USER_EXT_FEATURES  = Path("data/user_extended_features.parquet")
USER_LLM_FEATURES  = Path("data/user_llm_features.parquet")
USER_LLM_EMBEDDING = Path("data/user_llm_embeddings.parquet")
ITEM_EXT_FEATURES   = Path("data/item_extended_features.parquet")
RESTAURANT_LLM_FEAT = Path("data/restaurant_llm_features.parquet")
ITEM_LLM_EMBEDDING  = Path("data/restaurant_llm_embeddings.parquet")
MODEL_DIR         = Path(os.environ.get("FOODIE_MODEL_DIR", "models"))
MODEL_FILE        = MODEL_DIR / "lightgcn_gnn.pt"
ENCODERS_FILE     = MODEL_DIR / "gnn_encoders.pkl"
EMB_DIR           = Path(os.environ.get("FOODIE_EMBEDDING_DIR", "data/embeddings"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_DIR = Path(os.environ.get("FOODIE_FEATURE_DIR", "data"))
USER_PREBUILT  = FEATURE_DIR / "user_features_prebuilt.parquet"
ITEM_PREBUILT  = FEATURE_DIR / "item_features_prebuilt.parquet"
FEATURE_GROUPS = FEATURE_DIR / "feature_groups.json"


def _align_prebuilt(feat_df: pd.DataFrame, id_col: str, enc: dict,
                    skip_groups: list[str] | None, groups_key: str) -> np.ndarray:
    """
    Vectorised fast-path: reindex a prebuilt feature DataFrame to the encoder's
    integer order, optionally dropping feature groups.

    feat_df  — DataFrame with id_col as identifier and all feature columns
    id_col   — 'contributor_id' or 'place_id'
    enc      — {id → integer_index}
    skip_groups — list of group names to drop (e.g. ['nlp', 'llm'])
    groups_key  — 'user' or 'item' key in feature_groups.json
    """
    skip_groups = skip_groups or []
    feat_cols = [c for c in feat_df.columns if c != id_col]

    if skip_groups and FEATURE_GROUPS.exists():
        import json
        with open(FEATURE_GROUPS) as f:
            all_groups = json.load(f).get(groups_key, {})
        drop_cols: set[str] = set()
        for g in skip_groups:
            drop_cols.update(all_groups.get(g, []))
        feat_cols = [c for c in feat_cols if c not in drop_cols]
        if drop_cols:
            dropped = sum(len(all_groups.get(g, [])) for g in skip_groups)
            log.info(f"  Skipping groups {skip_groups}: -{dropped} cols → {len(feat_cols)} remaining")

    feat_df = feat_df[[id_col] + feat_cols]
    n = len(enc)
    feat = np.zeros((n, len(feat_cols)), dtype=np.float32)

    # Vectorised reindex via merge on encoder order
    order_df = pd.DataFrame({id_col: list(enc.keys()),
                              "_idx": list(enc.values())})
    merged = order_df.merge(feat_df, on=id_col, how="left")
    merged = merged.sort_values("_idx")
    vals = merged[feat_cols].values.astype(np.float32)
    feat = np.nan_to_num(vals, nan=0.0)
    return feat


# ── Side-feature builders ──────────────────────────────────────────────────────
def build_user_features(df: pd.DataFrame, user_enc: dict,
                        prebuilt: bool = False,
                        skip_groups: list[str] | None = None) -> np.ndarray:
    """
    (n_users, feat_dim) float32 matrix from reviewer metadata.
    Pass prebuilt=True to load from data/user_features_prebuilt.parquet (fast path).
    Pass skip_groups=['extended','pref'] to drop feature groups at load time.
    """
    if prebuilt and USER_PREBUILT.exists():
        log.info(f"Loading prebuilt user features from {USER_PREBUILT} …")
        feat_df = pd.read_parquet(USER_PREBUILT)
        feat = _align_prebuilt(feat_df, "contributor_id", user_enc,
                               skip_groups, "user")
        log.info(f"User features (prebuilt): {feat.shape[1]} dims for {feat.shape[0]:,} users")
        return feat
    if prebuilt:
        log.warning(f"{USER_PREBUILT} not found — falling back to full build; "
                    "run build_training_features.py to generate it")

    cols = ["contributor_id", "predicted_race", "predicted_gender",
            "is_local_guide", "reviewer_reviews"]
    user_meta = (
        df.sort_values("timestamp_days_ago", ascending=True, na_position="last")
        .drop_duplicates("contributor_id")[cols].copy()
    )
    user_meta["predicted_race"]   = user_meta["predicted_race"].fillna("unknown").astype(str)
    user_meta["predicted_gender"] = user_meta["predicted_gender"].fillna("unknown").astype(str)
    user_meta["is_local_guide"]   = user_meta["is_local_guide"].fillna(False).astype(float)
    user_meta["log_reviews"] = np.log1p(
        user_meta["reviewer_reviews"].fillna(0).clip(lower=0).astype(float)
    )
    mx = user_meta["log_reviews"].max()
    if mx > 0:
        user_meta["log_reviews"] /= mx

    # User location: median lat/lng across all restaurants the user has reviewed
    loc_cols: list[str] = []
    if RESTAURANTS_ENR.exists():
        rest_loc = pd.read_parquet(RESTAURANTS_ENR, columns=["place_id", "lat", "lng"])
        rest_loc["lat"] = pd.to_numeric(rest_loc["lat"], errors="coerce")
        rest_loc["lng"] = pd.to_numeric(rest_loc["lng"], errors="coerce")
        user_loc = (
            df[["contributor_id", "place_id"]]
            .merge(rest_loc, on="place_id", how="left")
            .groupby("contributor_id")[["lat", "lng"]].median()
        )
        lat_min, lat_max = user_loc["lat"].min(), user_loc["lat"].max()
        lng_min, lng_max = user_loc["lng"].min(), user_loc["lng"].max()
        user_loc["lat_norm"] = (user_loc["lat"] - lat_min) / (lat_max - lat_min + 1e-8)
        user_loc["lng_norm"] = (user_loc["lng"] - lng_min) / (lng_max - lng_min + 1e-8)
        user_meta = user_meta.merge(
            user_loc[["lat_norm", "lng_norm"]], left_on="contributor_id",
            right_index=True, how="left"
        )
        user_meta[["lat_norm", "lng_norm"]] = user_meta[["lat_norm", "lng_norm"]].fillna(0.5)
        loc_cols = ["lat_norm", "lng_norm"]

    race_d   = pd.get_dummies(user_meta["predicted_race"],   prefix="race",   dtype=float)
    gender_d = pd.get_dummies(user_meta["predicted_gender"], prefix="gender", dtype=float)
    feat_df  = pd.concat(
        [user_meta[["contributor_id", "is_local_guide", "log_reviews"] + loc_cols],
         race_d, gender_d],
        axis=1,
    )
    feat_cols = [c for c in feat_df.columns if c != "contributor_id"]

    # ── Extended user features ─────────────────────────────────────────────
    if USER_EXT_FEATURES.exists():
        ext = pd.read_parquet(USER_EXT_FEATURES)
        ext_cols = [c for c in ext.columns if c != "contributor_id"]
        feat_df = feat_df.merge(ext, on="contributor_id", how="left")
        feat_df[ext_cols] = feat_df[ext_cols].fillna(0.0)
        feat_cols = feat_cols + ext_cols
        log.info(f"  Extended user features joined: +{len(ext_cols)} dims "
                 f"({len(feat_cols)} total)")

    # ── Leakage-safe consolidated LLM features ────────────────────────────
    if USER_LLM_FEATURES.exists():
        pref = pd.read_parquet(USER_LLM_FEATURES)
        pref_cols = [c for c in pref.columns if c != "contributor_id"
                     and pd.api.types.is_numeric_dtype(pref[c])
                     and c != "source_review_count"]
        feat_df = feat_df.merge(pref[["contributor_id"] + pref_cols], on="contributor_id", how="left")
        for col in pref_cols:
            default = 0.0 if any(x in col for x in ("_known", "_confidence", "cuisine_", "meal_")) else 0.5
            feat_df[col] = feat_df[col].fillna(default)
        feat_cols += pref_cols
        coverage = feat_df["contributor_id"].isin(pref["contributor_id"]).mean()
        log.info(f"  User LLM features joined: +{len(pref_cols)} dims "
                 f"({len(feat_cols)} total, coverage={coverage:.1%})")

    if USER_LLM_EMBEDDING.exists():
        emb = pd.read_parquet(USER_LLM_EMBEDDING)
        emb_cols = [c for c in emb.columns if c.startswith("llm_emb_")]
        feat_df = feat_df.merge(emb[["contributor_id"] + emb_cols], on="contributor_id", how="left")
        feat_df[emb_cols] = feat_df[emb_cols].fillna(0.0)
        feat_cols += emb_cols

    feat      = np.zeros((len(user_enc), len(feat_cols)), dtype=np.float32)
    for _, row in feat_df.iterrows():
        uid = user_enc.get(row["contributor_id"])
        if uid is not None:
            feat[uid] = row[feat_cols].values.astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0)
    log.info(f"User features: {len(feat_cols)} dims — "
             f"{len([c for c in feat_cols if c.startswith('race')])} race, "
             f"{len([c for c in feat_cols if c.startswith('gender')])} gender, "
             f"{len(loc_cols)} location, "
             f"{2} other (is_local_guide, log_reviews)")
    return feat


def build_item_features(item_enc: dict, llm_feat_mode: str = "fast", use_nlp_feat: bool = True,
                        prebuilt: bool = False,
                        skip_groups: list[str] | None = None) -> np.ndarray | None:
    """
    (n_items, feat_dim) float32 matrix from restaurant metadata.
    Pass prebuilt=True to load from data/item_features_prebuilt.parquet (fast path).
    Pass skip_groups=['nlp','llm'] to drop feature groups at load time.
    """
    if prebuilt and ITEM_PREBUILT.exists():
        log.info(f"Loading prebuilt item features from {ITEM_PREBUILT} …")
        feat_df = pd.read_parquet(ITEM_PREBUILT)
        feat = _align_prebuilt(feat_df, "place_id", item_enc, skip_groups, "item")
        log.info(f"Item features (prebuilt): {feat.shape[1]} dims for {feat.shape[0]:,} items")
        return feat
    if prebuilt:
        log.warning(f"{ITEM_PREBUILT} not found — falling back to full build; "
                    "run build_training_features.py to generate it")

    if not RESTAURANTS_ENR.exists():
        log.warning("restaurants_enriched.parquet not found — item features disabled")
        return None

    _price_map = {
        "PRICE_LEVEL_INEXPENSIVE":    1.0,
        "PRICE_LEVEL_MODERATE":       2.0,
        "PRICE_LEVEL_EXPENSIVE":      3.0,
        "PRICE_LEVEL_VERY_EXPENSIVE": 4.0,
    }

    rest = pd.read_parquet(RESTAURANTS_ENR)
    rest["cuisine_category"] = rest["cuisine_category"].fillna("Other").astype(str)
    rest["price_num"] = rest["price_level"].map(_price_map).fillna(2.0)
    rest["rating"]    = pd.to_numeric(rest["rating"], errors="coerce")
    rest["rating"]    = rest["rating"].fillna(rest["rating"].median())
    rest["rating_count"] = pd.to_numeric(rest["user_rating_count"], errors="coerce").fillna(0)

    rest["lat"] = pd.to_numeric(rest["lat"], errors="coerce")
    rest["lng"] = pd.to_numeric(rest["lng"], errors="coerce")
    rest["lat_norm"] = (rest["lat"] - rest["lat"].min()) / (rest["lat"].max() - rest["lat"].min() + 1e-8)
    rest["lng_norm"] = (rest["lng"] - rest["lng"].min()) / (rest["lng"].max() - rest["lng"].min() + 1e-8)
    rest[["lat_norm", "lng_norm"]] = rest[["lat_norm", "lng_norm"]].fillna(0.5)

    rest["price_norm"]        = (rest["price_num"].clip(1, 4) - 1) / 3.0
    rest["rating_norm"]       = (rest["rating"].clip(1, 5) - 1) / 4.0
    rest["log_rating_count"]  = np.log1p(rest["rating_count"])
    mx = rest["log_rating_count"].max()
    if mx > 0:
        rest["log_rating_count"] /= mx

    # ── Aggregate per-review scores from reviews_flat ──────────────────────────
    if REVIEWS_FLAT.exists():
        rev = pd.read_parquet(REVIEWS_FLAT, columns=["place_id", "food_score",
                                                      "service_score", "atmosphere_score",
                                                      "meal_type"])
        # Scores are stored as strings like "4", "5…" — strip trailing ellipsis
        for col in ["food_score", "service_score", "atmosphere_score"]:
            rev[col] = pd.to_numeric(
                rev[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
            )

        score_agg = (
            rev.groupby("place_id")[["food_score", "service_score", "atmosphere_score"]]
            .mean()
            .rename(columns={"food_score": "food_score_mean",
                             "service_score": "service_score_mean",
                             "atmosphere_score": "atmosphere_score_mean"})
        )
        for col in ["food_score_mean", "service_score_mean", "atmosphere_score_mean"]:
            score_agg[col] = (score_agg[col].clip(1, 5) - 1) / 4.0

        # Meal-type distribution per restaurant
        _meal_clean = rev["meal_type"].astype(str).str.extract(
            r"(Dinner|Lunch|Brunch|Breakfast)", expand=False
        )
        meal_counts = (
            rev.assign(meal_clean=_meal_clean)
            .groupby("place_id")["meal_clean"]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
            .rename(columns=lambda c: f"meal_pct_{c.lower()}")
        )
        for col in ["meal_pct_dinner", "meal_pct_lunch", "meal_pct_brunch", "meal_pct_breakfast"]:
            if col not in meal_counts.columns:
                meal_counts[col] = 0.0

        rest = (rest.set_index("place_id")
                    .join(score_agg, how="left")
                    .join(meal_counts[["meal_pct_dinner", "meal_pct_lunch",
                                       "meal_pct_brunch", "meal_pct_breakfast"]], how="left")
                    .reset_index())
        for col in ["food_score_mean", "service_score_mean", "atmosphere_score_mean",
                    "meal_pct_dinner", "meal_pct_lunch", "meal_pct_brunch", "meal_pct_breakfast"]:
            rest[col] = rest[col].fillna(0.0)
        review_cols = ["food_score_mean", "service_score_mean", "atmosphere_score_mean",
                       "meal_pct_dinner", "meal_pct_lunch", "meal_pct_brunch", "meal_pct_breakfast"]
    else:
        review_cols = []

    cuisine_d = pd.get_dummies(rest["cuisine_category"], prefix="cuisine", dtype=float)
    numeric_cols = ["price_norm", "rating_norm", "log_rating_count",
                    "lat_norm", "lng_norm"] + review_cols

    # ── Extended item features ─────────────────────────────────────────────
    if ITEM_EXT_FEATURES.exists():
        ext = pd.read_parquet(ITEM_EXT_FEATURES)
        ext_cols = [c for c in ext.columns if c != "place_id"]
        rest = rest.merge(ext, on="place_id", how="left")
        for col in ext_cols:
            rest[col] = rest[col].fillna(0.0)
        numeric_cols = numeric_cols + ext_cols
        log.info(f"  Extended item features joined: +{len(ext_cols)} dims")

    # ── Restaurant LLM/fast features ──────────────────────────────────────
    if RESTAURANT_LLM_FEAT.exists() and llm_feat_mode != "none":
        llm_ext   = pd.read_parquet(RESTAURANT_LLM_FEAT)
        load_cols = [c for c in llm_ext.columns if c != "place_id"
                     and pd.api.types.is_numeric_dtype(llm_ext[c])
                     and c != "source_review_count"]
        rest = rest.merge(llm_ext[["place_id"] + load_cols], on="place_id", how="left")
        for col in load_cols:
            default = 0.0 if any(x in col for x in ("_known", "_confidence", "cuisine_", "meal_")) else 0.5
            rest[col] = rest[col].fillna(default)
        numeric_cols = numeric_cols + load_cols
        log.info(f"  Restaurant features joined ({llm_feat_mode}): +{len(load_cols)} dims")

    if llm_feat_mode == "full" and ITEM_LLM_EMBEDDING.exists():
        emb = pd.read_parquet(ITEM_LLM_EMBEDDING)
        emb_cols = [c for c in emb.columns if c.startswith("llm_emb_")]
        rest = rest.merge(emb[["place_id"] + emb_cols], on="place_id", how="left")
        rest[emb_cols] = rest[emb_cols].fillna(0.0)
        numeric_cols += emb_cols

    feat_df = pd.concat([rest[["place_id"] + numeric_cols], cuisine_d], axis=1)

    feat_cols = [c for c in feat_df.columns if c != "place_id"]
    feat      = np.zeros((len(item_enc), len(feat_cols)), dtype=np.float32)
    feat_map  = feat_df.set_index("place_id")[feat_cols]
    item_dec  = {v: k for k, v in item_enc.items()}
    for item_idx, place_id in item_dec.items():
        if place_id in feat_map.index:
            feat[item_idx] = feat_map.loc[place_id].values.astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0)
    log.info(f"Item features: {len(feat_cols)} dims — "
             f"{len([c for c in feat_cols if c.startswith('cuisine')])} cuisine + "
             f"{len(numeric_cols)} numeric "
             f"(price, rating, rating_count, lat/lng, food/service/atmosphere scores, meal-type %)")
    return feat


# ── Dish graph data builder ────────────────────────────────────────────────────
def build_dish_data(
    train_df: pd.DataFrame,
    user_enc: dict,
    item_enc: dict,
) -> tuple[dict, torch.Tensor, torch.Tensor, np.ndarray | None]:
    """
    Build dish encoders and two sparse adj matrices for the hetero GNN.

    Returns:
        dish_enc  — dict {dish_name_norm: dish_idx}
        adj_dr    — (n_dishes, n_items) dish→restaurant count-weighted, D-normalised
        adj_dd    — (n_dishes, n_dishes) dish-dish similarity, D-normalised
        dish_feat — (n_dishes, feat_dim) float32 or None
    """
    from pathlib import Path as _P

    n_items = len(item_enc)

    # ── Build dish vocabulary from dishes.parquet + profiles ──────────────────
    dishes_path = _P("data/dishes.parquet")
    dish_names_set: set[str] = set()
    if dishes_path.exists():
        dsh = pd.read_parquet(dishes_path, columns=["dish_name"])
        dish_names_set.update(dsh["dish_name"].str.lower().str.strip().unique())
    if DISH_PROFILES.exists():
        dp = pd.read_parquet(DISH_PROFILES, columns=["dish_name"])
        dish_names_set.update(dp["dish_name"].str.lower().str.strip().unique())
    if not dish_names_set:
        log.warning("No dish data found — dish graph disabled.")
        return {}, None, None, None

    dish_names = sorted(dish_names_set)
    dish_enc   = {d: i for i, d in enumerate(dish_names)}
    n_dishes   = len(dish_enc)
    log.info(f"Dish vocabulary: {n_dishes:,} unique dishes")

    # ── adj_dr : (n_dishes, n_items) ──────────────────────────────────────────
    adj_dr = None
    dishes_path = _P("data/dishes.parquet")
    if dishes_path.exists():
        dsh = pd.read_parquet(dishes_path, columns=["dish_name", "place_id"])
        dsh["dish_norm"] = dsh["dish_name"].str.lower().str.strip()
        dsh["dish_idx"]  = dsh["dish_norm"].map(dish_enc)
        dsh["item_idx"]  = dsh["place_id"].map(item_enc)
        dsh = dsh.dropna(subset=["dish_idx", "item_idx"]).copy()
        dsh["dish_idx"] = dsh["dish_idx"].astype(int)
        dsh["item_idx"] = dsh["item_idx"].astype(int)

        agg = dsh.groupby(["dish_idx", "item_idx"]).size().reset_index(name="cnt")
        rows  = agg["dish_idx"].values
        cols  = agg["item_idx"].values
        vals  = np.log1p(agg["cnt"].values).astype(np.float32)
        N_r, N_c = n_dishes, n_items
        deg_r = np.maximum(np.bincount(rows, minlength=N_r).astype(np.float32), 1)
        deg_c = np.maximum(np.bincount(cols, minlength=N_c).astype(np.float32), 1)
        vals  = vals / np.sqrt(deg_r[rows]) / np.sqrt(deg_c[cols])
        indices = torch.LongTensor(np.stack([rows, cols]))
        adj_dr  = torch.sparse_coo_tensor(
            indices, torch.FloatTensor(vals), (N_r, N_c)
        ).coalesce().to(DEVICE)
        log.info(f"adj_dr: {N_r:,} dishes × {N_c:,} items  ({len(vals):,} edges)")

    # ── adj_dd : (n_dishes, n_dishes) ─────────────────────────────────────────
    adj_dd = None
    if DISH_SIMILARITIES.exists():
        sim_df = pd.read_parquet(DISH_SIMILARITIES)
        sim_df["a_idx"] = sim_df["dish_a"].map(dish_enc)
        sim_df["b_idx"] = sim_df["dish_b"].map(dish_enc)
        sim_df = sim_df.dropna(subset=["a_idx", "b_idx"]).copy()
        sim_df["a_idx"] = sim_df["a_idx"].astype(int)
        sim_df["b_idx"] = sim_df["b_idx"].astype(int)
        # Symmetric
        a_idx = np.concatenate([sim_df["a_idx"].values, sim_df["b_idx"].values])
        b_idx = np.concatenate([sim_df["b_idx"].values, sim_df["a_idx"].values])
        sims  = np.concatenate([sim_df["similarity"].values] * 2).astype(np.float32)
        deg   = np.maximum(np.bincount(a_idx, minlength=n_dishes).astype(np.float32), 1)
        sims  = sims / np.sqrt(deg[a_idx]) / np.sqrt(deg[b_idx])
        indices = torch.LongTensor(np.stack([a_idx, b_idx]))
        adj_dd  = torch.sparse_coo_tensor(
            indices, torch.FloatTensor(sims), (n_dishes, n_dishes)
        ).coalesce().to(DEVICE)
        log.info(f"adj_dd: {n_dishes:,}×{n_dishes:,}  ({len(sims):,} edges)")

    # ── dish_feat : (n_dishes, feat_dim) ──────────────────────────────────────
    dish_feat = None
    if DISH_PROFILES.exists():
        dp = pd.read_parquet(DISH_PROFILES)
        dp["dish_norm"] = dp["dish_name"].str.lower().str.strip()
        feat_cols = [c for c in dp.columns if c.startswith("f")]
        if feat_cols:
            mat = np.zeros((n_dishes, len(feat_cols)), dtype=np.float32)
            for _, row in dp.iterrows():
                idx = dish_enc.get(row["dish_norm"])
                if idx is not None:
                    mat[idx] = row[feat_cols].values.astype(np.float32)
            dish_feat = np.nan_to_num(mat, nan=0.0)
            log.info(f"Dish features: {n_dishes:,} dishes × {len(feat_cols)} dims")

    return dish_enc, adj_dr, adj_dd, dish_feat


# ── LightGCN model ─────────────────────────────────────────────────────────────
class LightGCN(nn.Module):
    """
    Simplified LightGCN on a bipartite reviewer–restaurant graph.
    Reference: He et al. (2020) LightGCN: Simplifying and Powering Graph
    Convolution Network for Recommendation.
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64, n_layers: int = 3,
                 user_feat: np.ndarray | None = None,
                 item_feat: np.ndarray | None = None,
                 grad_checkpoint: bool = False,
                 edge_dropout: float = 0.0,
                 use_film: bool = False):
        super().__init__()
        self.n_users         = n_users
        self.n_items         = n_items
        self.n_layers        = n_layers
        self.grad_checkpoint = grad_checkpoint
        self.edge_dropout    = edge_dropout

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        for name, feat in [("user", user_feat), ("item", item_feat)]:
            if feat is not None:
                self.register_buffer(f"{name}_feat", torch.tensor(feat, dtype=torch.float32))
                if use_film:
                    setattr(self, f"{name}_film", FiLMFusion(feat.shape[1], emb_dim))
                    setattr(self, f"{name}_proj", None)
                else:
                    proj = nn.Linear(feat.shape[1], emb_dim, bias=False)
                    nn.init.xavier_uniform_(proj.weight)
                    setattr(self, f"{name}_proj", proj)
                    setattr(self, f"{name}_film", None)
            else:
                setattr(self, f"{name}_feat", None)
                setattr(self, f"{name}_proj", None)
                setattr(self, f"{name}_film", None)

    def _propagate(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run LightGCN propagation; return final user/item embeddings."""
        u_init = self.user_emb.weight
        i_init = self.item_emb.weight
        if self.user_film is not None:
            u_init = self.user_film(u_init, self.user_feat)
        elif self.user_proj is not None:
            u_init = u_init + self.user_proj(self.user_feat)
        if self.item_film is not None:
            i_init = self.item_film(i_init, self.item_feat)
        elif self.item_proj is not None:
            i_init = i_init + self.item_proj(self.item_feat)

        all_emb  = torch.cat([u_init, i_init], dim=0)
        adj_used = dropout_adj(adj, self.edge_dropout, self.training)

        def _gcn_loop(e):
            s = e
            for _ in range(self.n_layers):
                e = _sparse_mm_f32(adj_used, e)
                s = s + e
            return s / (self.n_layers + 1)

        if self.grad_checkpoint and self.training:
            final = grad_ckpt(_gcn_loop, all_emb, use_reentrant=False)
        else:
            final = _gcn_loop(all_emb)
        return final[:self.n_users], final[self.n_users:]

    def forward(self, adj, users, pos_items, neg_items):
        u_emb, i_emb = self._propagate(adj)
        u   = u_emb[users]
        pos = i_emb[pos_items]
        neg = i_emb[neg_items]
        return u, pos, neg, self.user_emb.weight, self.item_emb.weight

    def get_embeddings(self, adj):
        with torch.no_grad():
            return self._propagate(adj)

    def recommend(self, adj, user_idx: int, exclude_items: set, top_n: int = 10):
        u_emb, i_emb = self.get_embeddings(adj)
        scores = (u_emb[user_idx] @ i_emb.T).cpu().numpy()
        for it in exclude_items:
            scores[it] = -np.inf
        return np.argsort(-scores)[:top_n]


def bpr_loss(u, pos, neg, reg: float = 1e-4):
    pos_score = (u * pos).sum(dim=1)
    neg_score = (u * neg).sum(dim=1)
    bpr = -F.logsigmoid(pos_score - neg_score).mean()
    # Batch-level L2 reg: only penalise embeddings of users/items in this batch.
    # Normalised per sample so scale is independent of n_users and batch size.
    reg_loss = reg * (u.norm(2, dim=1).pow(2) +
                      pos.norm(2, dim=1).pow(2) +
                      neg.norm(2, dim=1).pow(2)).mean() / 2
    return bpr + reg_loss


class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation: output = γ(feat) · emb + β(feat).
    Initialised so γ≈1 and β≈0 — identity at epoch 0, same as no side features.
    """
    def __init__(self, feat_dim: int, emb_dim: int):
        super().__init__()
        self.gamma = nn.Linear(feat_dim, emb_dim)
        self.beta  = nn.Linear(feat_dim, emb_dim)
        nn.init.zeros_(self.gamma.weight);  nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);   nn.init.zeros_(self.beta.bias)

    def forward(self, emb: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        return self.gamma(feat) * emb + self.beta(feat)


# ── Dish Heterogeneous GNN ─────────────────────────────────────────────────────
class DishHeteroGNN(nn.Module):
    """
    LightGCN + dish content path.

    On top of the standard bipartite user↔restaurant LightGCN, dish nodes
    provide a content-based signal:
      1. Each dish aggregates embeddings from restaurants that serve it (adj_dr.T)
      2. Dish similarity propagates that signal to semantically related dishes (adj_dd)
      3. Restaurants receive the enriched dish embeddings back (adj_dr.T)

    This lets restaurants sharing similar dish profiles cluster together even
    without review overlap — especially useful for cold-start restaurants.
    A learned scalar α ∈ (0,1) controls how much the dish path contributes.
    """

    def __init__(
        self,
        n_users:         int,
        n_items:         int,
        n_dishes:        int,
        emb_dim:         int = 64,
        n_layers:        int = 3,
        user_feat:       np.ndarray | None = None,
        item_feat:       np.ndarray | None = None,
        dish_feat:       np.ndarray | None = None,
        grad_checkpoint: bool = False,
        edge_dropout:    float = 0.0,
        use_film:        bool = False,
    ):
        super().__init__()
        self.n_users         = n_users
        self.n_items         = n_items
        self.n_dishes        = n_dishes
        self.n_layers        = n_layers
        self.grad_checkpoint = grad_checkpoint
        self.edge_dropout    = edge_dropout

        self.user_emb  = nn.Embedding(n_users,  emb_dim)
        self.item_emb  = nn.Embedding(n_items,  emb_dim)
        self.dish_emb  = nn.Embedding(n_dishes, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.dish_emb.weight)

        for name, feat in [("user", user_feat), ("item", item_feat), ("dish", dish_feat)]:
            if feat is not None:
                self.register_buffer(f"{name}_feat", torch.tensor(feat, dtype=torch.float32))
                if use_film:
                    setattr(self, f"{name}_film", FiLMFusion(feat.shape[1], emb_dim))
                    setattr(self, f"{name}_proj", None)
                else:
                    proj = nn.Linear(feat.shape[1], emb_dim, bias=False)
                    nn.init.xavier_uniform_(proj.weight)
                    setattr(self, f"{name}_proj", proj)
                    setattr(self, f"{name}_film", None)
            else:
                setattr(self, f"{name}_feat", None)
                setattr(self, f"{name}_proj", None)
                setattr(self, f"{name}_film", None)

        self.log_dish_alpha = nn.Parameter(torch.tensor(-2.3))

    def _proj(self, name: str) -> torch.Tensor:
        """Initial embedding for a node type, adding side-feature projection if available."""
        emb  = getattr(self, f"{name}_emb").weight
        film = getattr(self, f"{name}_film", None)
        proj = getattr(self, f"{name}_proj")
        feat = getattr(self, f"{name}_feat")
        if film is not None:
            return film(emb, feat)
        if proj is not None:
            return emb + proj(feat)
        return emb

    def _propagate(
        self,
        adj:    torch.Tensor,
        adj_dr: torch.Tensor | None,
        adj_dd: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u_init = self._proj("user")
        i_init = self._proj("item")
        d_init = self._proj("dish")

        all_emb  = torch.cat([u_init, i_init], dim=0)
        adj_used = dropout_adj(adj, self.edge_dropout, self.training)

        def _gcn_loop(e):
            s = e
            for _ in range(self.n_layers):
                e = _sparse_mm_f32(adj_used, e)
                s = s + e
            return s / (self.n_layers + 1)

        if self.grad_checkpoint and self.training:
            final = grad_ckpt(_gcn_loop, all_emb, use_reentrant=False)
        else:
            final = _gcn_loop(all_emb)
        u_lgcn = final[:self.n_users]
        i_lgcn = final[self.n_users:]

        if adj_dr is None:
            return u_lgcn, i_lgcn

        alpha = torch.sigmoid(self.log_dish_alpha)

        d_emb = d_init + torch.sparse.mm(adj_dr, i_init)
        if adj_dd is not None:
            d_emb = d_emb + torch.sparse.mm(adj_dd, d_init)
        d_emb = d_emb / (2.0 + (1.0 if adj_dd is not None else 0.0))

        i_dish  = torch.sparse.mm(adj_dr.t(), d_emb)
        i_final = i_lgcn + alpha * i_dish
        return u_lgcn, i_final

    def forward(self, adj, adj_dr, adj_dd, users, pos_items, neg_items):
        u_emb, i_emb = self._propagate(adj, adj_dr, adj_dd)
        u   = u_emb[users]
        pos = i_emb[pos_items]
        neg = i_emb[neg_items]
        return u, pos, neg, self.user_emb.weight, self.item_emb.weight

    def get_embeddings(self, adj, adj_dr=None, adj_dd=None):
        with torch.no_grad():
            return self._propagate(adj, adj_dr, adj_dd)

    def recommend(self, adj, adj_dr, adj_dd,
                  user_idx: int, exclude_items: set, top_n: int = 10):
        u_emb, i_emb = self.get_embeddings(adj, adj_dr, adj_dd)
        scores = (u_emb[user_idx] @ i_emb.T).cpu().numpy()
        for it in exclude_items:
            scores[it] = -np.inf
        return np.argsort(-scores)[:top_n]


# ── Adjacency matrix ───────────────────────────────────────────────────────────
def build_adj(train_df: pd.DataFrame, n_users: int, n_items: int) -> torch.Tensor:
    """
    Build a normalised symmetric adjacency matrix for the bipartite graph.
    Rows/cols: [users | items], shape (n_users + n_items, n_users + n_items).
    """
    rows_ui = train_df["user_idx"].values
    cols_ui = train_df["item_idx"].values + n_users
    # Symmetric: user→item and item→user
    rows = np.concatenate([rows_ui, cols_ui])
    cols = np.concatenate([cols_ui, rows_ui])
    vals = np.ones(len(rows), dtype=np.float32)

    N = n_users + n_items
    # Degree normalisation: D^{-1/2} A D^{-1/2}
    degrees = np.bincount(rows, minlength=N).astype(np.float32)
    degrees = np.maximum(degrees, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(degrees)
    vals = vals * d_inv_sqrt[rows] * d_inv_sqrt[cols]

    indices = torch.LongTensor(np.stack([rows, cols]))
    values  = torch.FloatTensor(vals)
    return torch.sparse_coo_tensor(indices, values, (N, N)).coalesce().to(DEVICE)


# ── Data loading (mirrors baseline.py) ────────────────────────────────────────
def _original_item_ids() -> set:
    """Return place_ids belonging to the original top-2000 CBGs."""
    orig_cbgs = set(
        pd.concat([
            pd.read_csv(Path("data/top1000_cbgs.csv"),      dtype=str),
            pd.read_csv(Path("data/top1001_2000_cbgs.csv"), dtype=str),
        ])["cbg_str"]
    )
    rest = pd.read_parquet(RESTAURANTS_ENR, columns=["place_id", "cbg"])
    rest["cbg"] = rest["cbg"].astype(str)
    return set(rest.loc[rest["cbg"].isin(orig_cbgs), "place_id"])


def _load_interaction_frame(min_reviews: int) -> tuple[pd.DataFrame, dict, dict]:
    """Load, deduplicate, filter, and encode the implicit-feedback table."""
    from build_llm_features_ollama import assign_interaction_splits
    df = assign_interaction_splits(pd.read_parquet(REVIEWS_FLAT), min_reviews)

    user_ids = sorted(df["contributor_id"].unique())
    item_ids = sorted(df["place_id"].unique())
    user_enc = {u: i for i, u in enumerate(user_ids)}
    item_enc = {it: i for i, it in enumerate(item_ids)}
    df["user_idx"] = df["contributor_id"].map(user_enc)
    df["item_idx"] = df["place_id"].map(item_enc)
    return df, user_enc, item_enc


def load_interaction_splits(min_reviews: int = 4):
    """Return chronological train/validation/test splits.

    Each eligible user contributes their newest unique restaurant to test and
    their second-newest to validation.  All older interactions form training.
    Four unique restaurants are required so every user retains at least two
    training interactions.
    """
    df, user_enc, item_enc = _load_interaction_frame(min_reviews)
    train_df = df[df["split"].eq("train")].copy()
    val_df = df[df["split"].eq("validation")].copy()
    test_df = df[df["split"].eq("test")].copy()
    log.info(
        f"Train {len(train_df):,} | Validation {len(val_df):,} | "
        f"Test {len(test_df):,} | Users {len(user_enc):,} | Items {len(item_enc):,}"
    )
    return train_df, val_df, test_df, user_enc, item_enc


def load_interactions(min_reviews: int = 3):
    # When FOODIE_ORIGINAL_TEST=1 the test split is restricted to restaurants
    # from the original top-2000 CBGs, so models trained on the expanded dataset
    # are evaluated against the same population as pre-expansion runs.
    original_test = os.environ.get("FOODIE_ORIGINAL_TEST", "0") == "1"

    df, user_enc, item_enc = _load_interaction_frame(min_reviews)

    # Build test split: most recent interaction per user.
    # With original_test=True, only interactions with original-CBG restaurants
    # are eligible as test items; training still uses the full expanded set.
    df_sorted = df.sort_values("timestamp_days_ago", ascending=True, na_position="last")
    if original_test:
        orig_ids    = _original_item_ids()
        df_eligible = df_sorted[df_sorted["place_id"].isin(orig_ids)]
        test_idx    = df_eligible.groupby("contributor_id").head(1).index
        log.info(f"FOODIE_ORIGINAL_TEST=1: test restricted to "
                 f"{len(orig_ids):,} original-CBG restaurants")
    else:
        test_idx = df_sorted.groupby("contributor_id").head(1).index

    train_df = df.drop(index=test_idx)
    test_df  = df.loc[test_idx]

    log.info(f"Train {len(train_df):,} | Test {len(test_df):,} | "
             f"Users {len(user_enc):,} | Items {len(item_enc):,}")
    return train_df, test_df, user_enc, item_enc


# ── Negative sampling ──────────────────────────────────────────────────────────
def sample_negatives(train_df: pd.DataFrame, n_items: int,
                     hard_ratio: float = 0.5,
                     emb_pool: "np.ndarray | None" = None,
                     rng: "np.random.Generator | None" = None) -> pd.DataFrame:
    """
    Sample one negative per training pair.
    hard_ratio fraction are "hard":
      - if emb_pool provided: item most similar in embedding space to the positive
      - otherwise: top-quartile popular item (original behaviour)
    """
    rng      = rng if rng is not None else np.random.default_rng()
    n        = len(train_df)
    uids     = train_df["user_idx"].to_numpy(dtype=np.int64, copy=False)
    pos_arr  = train_df["item_idx"].to_numpy(dtype=np.int64, copy=False)

    if emb_pool is not None:
        k          = emb_pool.shape[1]
        choice     = rng.integers(0, k, size=n)
        hard_negs  = emb_pool[pos_arr, choice]   # one hard neg per positive item
    else:
        item_counts = train_df["item_idx"].value_counts()
        top_k       = max(100, len(item_counts) // 4)
        pop_pool    = item_counts.nlargest(top_k).index.values
        hard_negs   = pop_pool[rng.integers(0, len(pop_pool), size=n)]

    rand_negs = rng.integers(0, n_items, size=n)
    negs      = np.where(rng.random(n) < hard_ratio, hard_negs, rand_negs)

    # Reject observed user-item pairs in vectorized integer-key space. The old
    # per-row Python set loop performed the identical rejection but dominated
    # epochs at ~772k interactions.
    positive_keys = uids * np.int64(n_items) + pos_arr
    invalid = np.isin(uids * np.int64(n_items) + negs, positive_keys)
    while invalid.any():
        negs[invalid] = rng.integers(0, n_items, size=int(invalid.sum()))
        invalid_idx = np.flatnonzero(invalid)
        invalid[invalid_idx] = np.isin(
            uids[invalid_idx] * np.int64(n_items) + negs[invalid_idx], positive_keys
        )

    return train_df.assign(neg_idx=negs)


def build_hard_neg_pool(i_emb: torch.Tensor, k: int = 50) -> np.ndarray:
    """
    For each item find its k most similar neighbours in embedding space.
    Returns (n_items, k) int32 array of item indices.
    GPU-accelerated with float16 similarity computation.
    """
    n      = i_emb.shape[0]
    emb_n  = F.normalize(i_emb.detach().float(), dim=1)
    pool   = np.empty((n, k), dtype=np.int32)
    CHUNK  = 512

    for start in range(0, n, CHUNK):
        end  = min(start + CHUNK, n)
        sims = (emb_n[start:end].half() @ emb_n.half().T).float()  # (chunk, n)
        for i in range(end - start):
            sims[i, start + i] = -2.0              # mask self-similarity
        _, top_idx = sims.topk(k, dim=1)
        pool[start:end] = top_idx.cpu().numpy()

    return pool


# ── Metrics ────────────────────────────────────────────────────────────────────
def ndcg_at_k(relevant: set, ranked: list, k: int) -> float:
    dcg   = sum(1.0 / np.log2(i + 2) for i, x in enumerate(ranked[:k]) if x in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


# ── Validation helper ──────────────────────────────────────────────────────────
def _eval_recall_at_k(model: "LightGCN", adj: torch.Tensor,
                      val_df: pd.DataFrame, excl_df: pd.DataFrame,
                      n_items: int, k: int = 10,
                      adj_dr=None, adj_dd=None) -> float:
    """Recall@k on val split; used for early stopping."""
    model.eval()
    with torch.no_grad():
        if adj_dr is not None and adj_dd is not None:
            u_emb, i_emb = model.get_embeddings(adj, adj_dr, adj_dd)
        else:
            u_emb, i_emb = model.get_embeddings(adj)

    excl_map = excl_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    val_map  = val_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    val_users = np.array(list(val_map.keys()))

    rec_scores: list[float] = []
    CHUNK = 4096
    for start in range(0, len(val_users), CHUNK):
        chunk   = val_users[start : start + CHUNK]
        chunk_t = torch.tensor(chunk, dtype=torch.long, device=u_emb.device)
        scores  = (u_emb[chunk_t] @ i_emb.T).float().cpu().numpy()
        for i, uid in enumerate(chunk):
            relevant = val_map.get(int(uid), set())
            if not relevant:
                continue
            exclude = excl_map.get(int(uid), set())
            row = scores[i].copy()
            for it in exclude:
                row[it] = -np.inf
            top_idx = np.argsort(-row)[: k + len(exclude)]
            ranked  = [int(x) for x in top_idx if x not in exclude][:k]
            rec_scores.append(sum(1 for x in ranked if x in relevant) / len(relevant))

    return float(np.mean(rec_scores)) if rec_scores else 0.0


def _eval_ranking_at_k(model: "LightGCN", adj: torch.Tensor,
                       eval_df: pd.DataFrame, excl_df: pd.DataFrame,
                       n_items: int, k: int = 10,
                       adj_dr=None, adj_dd=None) -> dict[str, float]:
    """Full-catalog Hit/Recall and NDCG; NDCG is the checkpoint objective."""
    model.eval()
    with torch.no_grad():
        if adj_dr is not None and adj_dd is not None:
            u_emb, i_emb = model.get_embeddings(adj, adj_dr, adj_dd)
        else:
            u_emb, i_emb = model.get_embeddings(adj)
    excl_map = excl_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    rel_map = eval_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    users = np.asarray(list(rel_map), dtype=np.int64)
    recalls, ndcgs = [], []
    for start in range(0, len(users), 4096):
        chunk = users[start:start + 4096]
        scores = (u_emb[torch.as_tensor(chunk, device=u_emb.device)] @ i_emb.T).float().cpu().numpy()
        for row_idx, uid in enumerate(chunk):
            relevant = rel_map[int(uid)]
            for item in excl_map.get(int(uid), set()):
                scores[row_idx, item] = -np.inf
            # Exact top-k without sorting the other ~61k candidates. This is
            # rank-equivalent to a full argsort and drastically reduces CPU time.
            top = np.argpartition(-scores[row_idx], k - 1)[:k]
            ranked = top[np.argsort(-scores[row_idx, top])].tolist()
            recalls.append(len(relevant.intersection(ranked)) / len(relevant))
            ndcgs.append(ndcg_at_k(relevant, ranked, k))
    return {
        "hit": float(np.mean([x > 0 for x in recalls])) if recalls else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


# ── Train ──────────────────────────────────────────────────────────────────────
def train(epochs: int = 50, emb_dim: int = 64, n_layers: int = 3,
          lr: float = 1e-3, batch_size: int = 2048, save: bool = True,
          use_dish_graph: bool = False,
          use_fp16: bool = False,
          grad_checkpoint: bool = False,
          edge_dropout: float = 0.0,
          use_film: bool = False,
          warm_restart: bool = False,
          t0: int = 200,
          hard_neg_refresh: int = 50,
          llm_feat_mode: str = "fast",
          use_nlp_feat: bool = True,
          eval_every: int = 50,
          prebuilt_features: bool = False,
          skip_feature_groups: list[str] | None = None,
          publication_split: bool = False,
          seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    if publication_split:
        train_df, val_df, test_df, user_enc, item_enc = load_interaction_splits()
    else:
        train_df, test_df, user_enc, item_enc = load_interactions()
        val_df = test_df
    n_users, n_items = len(user_enc), len(item_enc)

    # ── Side features ───────────────────────────────────────────────────────────
    user_feat = build_user_features(train_df, user_enc,
                                    prebuilt=prebuilt_features,
                                    skip_groups=skip_feature_groups)
    item_feat = build_item_features(item_enc, llm_feat_mode=llm_feat_mode, use_nlp_feat=use_nlp_feat,
                                    prebuilt=prebuilt_features,
                                    skip_groups=skip_feature_groups)

    adj = build_adj(train_df, n_users, n_items)

    # ── Dish hetero graph ───────────────────────────────────────────────────────
    dish_enc = {}
    adj_dr = adj_dd = dish_feat = None
    if use_dish_graph:
        dish_enc, adj_dr, adj_dd, dish_feat = build_dish_data(
            train_df, user_enc, item_enc
        )
        if not dish_enc:
            log.warning("No dish data available — falling back to standard LightGCN")
            use_dish_graph = False

    if use_dish_graph:
        n_dishes = len(dish_enc)
        model = DishHeteroGNN(
            n_users, n_items, n_dishes, emb_dim=emb_dim, n_layers=n_layers,
            user_feat=user_feat, item_feat=item_feat, dish_feat=dish_feat,
            grad_checkpoint=grad_checkpoint, edge_dropout=edge_dropout, use_film=use_film,
        ).to(DEVICE)
        model_type = "DishHeteroGNN"
    else:
        model = LightGCN(n_users, n_items, emb_dim=emb_dim, n_layers=n_layers,
                         user_feat=user_feat, item_feat=item_feat,
                         grad_checkpoint=grad_checkpoint, edge_dropout=edge_dropout,
                         use_film=use_film).to(DEVICE)
        model_type = "LightGCN"

    proj_params = [p for n, p in model.named_parameters() if "proj" in n or "alpha" in n]
    cf_params   = [p for n, p in model.named_parameters() if "proj" not in n and "alpha" not in n]
    optimizer = Adam([
        {"params": cf_params,   "weight_decay": 0},
        {"params": proj_params, "weight_decay": 1e-2},
    ], lr=lr)
    if warm_restart:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=2, eta_min=2e-5)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)

    use_amp = use_fp16 and torch.cuda.is_available()
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    log.info(f"Training {model_type} | device={DEVICE} | epochs={epochs} | "
             f"emb_dim={emb_dim} | layers={n_layers} | lr={lr} | "
             f"fp16={use_amp} | grad_checkpoint={grad_checkpoint} | "
             f"edge_dropout={edge_dropout} | film={use_film} | "
             f"warm_restart={warm_restart} | hard_neg_refresh={hard_neg_refresh}")
    if use_dish_graph:
        alpha_init = torch.sigmoid(model.log_dish_alpha).item()
        log.info(f"  Dish path: adj_dr={'yes' if adj_dr is not None else 'no'} "
                 f"adj_dd={'yes' if adj_dd is not None else 'no'} "
                 f"alpha_init={alpha_init:.3f}")

    def _make_tensors(df):
        return (
            torch.tensor(df["user_idx"].values, dtype=torch.long).to(DEVICE),
            torch.tensor(df["item_idx"].values, dtype=torch.long).to(DEVICE),
            torch.tensor(df["neg_idx"].values,  dtype=torch.long).to(DEVICE),
        )

    emb_pool       = None   # embedding-space hard neg pool; built after first hard_neg_refresh epochs
    train_with_neg = sample_negatives(train_df, n_items, emb_pool=emb_pool, rng=rng)
    users_t, pos_items, neg_items = _make_tensors(train_with_neg)

    best_ndcg       = -1.0
    best_val_metrics = None
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        model.train()

        do_resample = epoch > 1 and epoch % 10 == 1

        # Refresh embedding-based hard neg pool every hard_neg_refresh epochs
        if hard_neg_refresh > 0 and epoch > 1 and (epoch - 1) % hard_neg_refresh == 0:
            model.eval()
            with torch.no_grad():
                if use_dish_graph:
                    _, i_emb_c = model.get_embeddings(adj, adj_dr, adj_dd)
                else:
                    _, i_emb_c = model.get_embeddings(adj)
            emb_pool = build_hard_neg_pool(i_emb_c, k=50)
            del i_emb_c
            model.train()
            log.info(f"  [Epoch {epoch}] Hard neg pool refreshed")
            do_resample = True

        if do_resample:
            train_with_neg = sample_negatives(train_df, n_items, emb_pool=emb_pool, rng=rng)
            users_t, pos_items, neg_items = _make_tensors(train_with_neg)

        optimizer.zero_grad()

        # Step 1: propagate once to get the full embedding matrices
        if use_dish_graph:
            u_emb_e, i_emb_e = model._propagate(adj, adj_dr, adj_dd)
        else:
            u_emb_e, i_emb_e = model._propagate(adj)

        # Step 2: accumulate BPR gradients in mini-batches using DETACHED slices.
        # Materialising u[users_t] (359K × 2048) all at once consumes 8+ GB;
        # mini-batching keeps peak below ~50 MB per batch.
        emb_dim   = u_emb_e.shape[1]
        n         = len(users_t)
        n_batches = max(1, (n + batch_size - 1) // batch_size)
        grad_u    = torch.zeros_like(u_emb_e)
        grad_i    = torch.zeros_like(i_emb_e)
        epoch_loss = 0.0

        for b in range(n_batches):
            s  = b * batch_size
            e_ = min(s + batch_size, n)
            u_b = u_emb_e[users_t[s:e_]].detach().requires_grad_(True)
            p_b = i_emb_e[pos_items[s:e_]].detach().requires_grad_(True)
            n_b = i_emb_e[neg_items[s:e_]].detach().requires_grad_(True)

            loss_b = bpr_loss(u_b, p_b, n_b) / n_batches
            loss_b.backward()
            epoch_loss += loss_b.item()

            idx_u = users_t[s:e_].unsqueeze(1).expand(-1, emb_dim)
            idx_p = pos_items[s:e_].unsqueeze(1).expand(-1, emb_dim)
            idx_n = neg_items[s:e_].unsqueeze(1).expand(-1, emb_dim)
            grad_u.scatter_add_(0, idx_u, u_b.grad)
            grad_i.scatter_add_(0, idx_p, p_b.grad)
            grad_i.scatter_add_(0, idx_n, n_b.grad)

        # Step 3: single backward through GCN with the accumulated embedding grads
        torch.autograd.backward([u_emb_e, i_emb_e], [grad_u, grad_i])

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            extra = ""
            if use_dish_graph:
                alpha = torch.sigmoid(model.log_dish_alpha).item()
                extra = f"  dish_alpha={alpha:.3f}"
            log.info(f"  Epoch {epoch}/{epochs}  loss={epoch_loss:.4f}  "
                     f"lr={scheduler.get_last_lr()[0]:.2e}{extra}")

        # ── Periodic validation + best-checkpoint tracking ───────────────────
        if eval_every > 0 and epoch % eval_every == 0:
            val_metrics = _eval_ranking_at_k(
                model, adj, val_df, train_df, n_items, k=10,
                adj_dr=adj_dr if use_dish_graph else None,
                adj_dd=adj_dd if use_dish_graph else None,
            )
            model.train()
            if val_metrics["ndcg"] > best_ndcg:
                best_ndcg       = val_metrics["ndcg"]
                best_val_metrics = val_metrics
                best_state_dict = copy.deepcopy(model.state_dict())
                log.info(f"  [Epoch {epoch}] val Hit@10={val_metrics['hit']:.4f} "
                         f"NDCG@10={val_metrics['ndcg']:.4f}  *** new best ***")
            else:
                log.info(f"  [Epoch {epoch}] val Hit@10={val_metrics['hit']:.4f} "
                         f"NDCG@10={val_metrics['ndcg']:.4f} (best={best_ndcg:.4f})")

    # ── Load best checkpoint before final evaluation ─────────────────────────
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        log.info(f"Loaded best checkpoint (val NDCG@10={best_ndcg:.4f})")

    # ── Evaluate ────────────────────────────────────────────────────────────────
    model.eval()
    K = 10
    seen_df = pd.concat([train_df, val_df], ignore_index=True) if publication_split else train_df
    user_pos_excl = seen_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    log.info("Computing embeddings for evaluation…")
    if use_dish_graph:
        u_emb_all, i_emb_all = model.get_embeddings(adj, adj_dr, adj_dd)
    else:
        u_emb_all, i_emb_all = model.get_embeddings(adj)
    log.info(f"  Embedding norms — users mean={u_emb_all.norm(dim=1).mean():.4f}, "
             f"items mean={i_emb_all.norm(dim=1).mean():.4f}")

    test_users    = test_df["user_idx"].unique()
    CHUNK         = 4096
    test_relevant = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    n_test_in_train = sum(
        1 for uid, rel in test_relevant.items()
        if rel & user_pos_excl.get(uid, set())
    )
    log.info(f"  Test items that also appear in train (should be ~0): "
             f"{n_test_in_train:,} / {len(test_relevant):,}")

    ndcg_scores, prec_scores, rec_scores = [], [], []
    prediction_records = []
    total_hits = 0

    log.info(f"Evaluating {len(test_users):,} test users in chunks of {CHUNK}…")
    for chunk_start in range(0, len(test_users), CHUNK):
        chunk_users = test_users[chunk_start : chunk_start + CHUNK]
        chunk_t = torch.tensor(chunk_users, dtype=torch.long, device=u_emb_all.device)
        scores  = (u_emb_all[chunk_t] @ i_emb_all.T).cpu().numpy()

        for i, user_idx_val in enumerate(chunk_users):
            relevant = test_relevant.get(int(user_idx_val), set())
            if not relevant:
                continue
            exclude = user_pos_excl.get(int(user_idx_val), set())
            row = scores[i].copy()
            for it in exclude:
                row[it] = -np.inf
            top_idx = np.argpartition(row, -min(100, n_items))[-min(100, n_items):]
            top_idx = top_idx[np.argsort(-row[top_idx])]
            ranked  = [int(x) for x in top_idx if x not in exclude][:K]

            if chunk_start == 0 and i < 5:
                log.info(f"  [diag] user={int(user_idx_val):8d} | "
                         f"relevant={sorted(int(v) for v in relevant)} | "
                         f"excl_n={len(exclude)} | "
                         f"top5={ranked[:5]} | "
                         f"hit={'YES' if relevant & set(ranked) else 'no'}")

            ndcg_scores.append(ndcg_at_k(relevant, ranked, K))
            hits = sum(1 for x in ranked if x in relevant)
            total_hits += hits
            prec_scores.append(hits / K)
            rec_scores.append(hits / len(relevant))
            true_item = next(iter(relevant)) if len(relevant) == 1 else None
            prediction_records.append((int(user_idx_val), true_item, ranked))

    log.info(f"  Users evaluated: {len(prec_scores):,} | Total hits@10: {total_hits:,}")

    precision_val = np.mean(prec_scores)
    recall_val    = np.mean(rec_scores)
    ndcg_val      = np.mean(ndcg_scores)

    print("\n=== LightGCN GNN Results ===")
    print(f"  Precision@{K}: {precision_val:.4f}")
    print(f"  Recall@{K}:    {recall_val:.4f}")
    print(f"  Hit@{K}:       {recall_val:.4f}  (one held-out item/user)")
    print(f"  NDCG@{K}:      {ndcg_val:.4f}")
    print()

    skip_str = ",".join(skip_feature_groups) if skip_feature_groups else ""
    variant  = f"skip {skip_str}" if skip_str else "all features"
    save_result(
        model="LightGCN", variant=variant,
        precision=precision_val, recall=recall_val, ndcg=ndcg_val,
        epochs=epochs, emb_dim=emb_dim, n_layers=n_layers,
        skip_groups=skip_str,
        notes=(f"seed={seed}; checkpoint=val_ndcg@10; "
               f"val_hit@10={best_val_metrics['hit']:.6f}; "
               f"val_ndcg@10={best_val_metrics['ndcg']:.6f}"
               if best_val_metrics else f"seed={seed}; checkpoint=final"),
    )
    if publication_split:
        save_publication_result(
            "LightGCN", condition_from_skip(skip_str), seed,
            best_val_metrics,
            {"hit": float(recall_val), "ndcg": float(ndcg_val)}, epochs, emb_dim,
        )

    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        EMB_DIR.mkdir(exist_ok=True)

        # Seed-specific publication artifacts avoid overwriting replicate runs.
        legacy_outputs = not publication_split
        if legacy_outputs:
            torch.save(model.state_dict(), MODEL_FILE)
        condition_tag = (f"skip_{skip_str.replace(',', '_')}" if skip_str else "all")
        encoder_path = (ENCODERS_FILE if legacy_outputs else
                        MODEL_DIR / f"gnn_encoders_{condition_tag}_seed{seed}.pkl")
        with open(encoder_path, "wb") as f:
            pickle.dump({
                "user_enc":       user_enc,
                "item_enc":       item_enc,
                "user_dec":       {v: k for k, v in user_enc.items()},
                "item_dec":       {v: k for k, v in item_enc.items()},
                "dish_enc":       dish_enc,
                "n_users":        n_users,
                "n_items":        n_items,
                "n_dishes":       len(dish_enc),
                "emb_dim":        emb_dim,
                "n_layers":       n_layers,
                "user_feat":      user_feat,
                "item_feat":      item_feat,
                "dish_feat":      dish_feat,
                "use_dish_graph": use_dish_graph,
                "use_film":       use_film,
                "edge_dropout":   edge_dropout,
            }, f)

        if legacy_outputs:
            torch.save(u_emb_all.cpu(), EMB_DIR / "reviewer_embeddings.pt")
            torch.save(i_emb_all.cpu(), EMB_DIR / "restaurant_embeddings.pt")

        file_tag   = f"{n_layers}l_skip_{skip_str.replace(',', '_')}" if skip_str else f"{n_layers}l_all"
        if publication_split:
            file_tag += "_pubsplit"
        file_tag += f"_seed{seed}"
        model_path = MODEL_DIR / f"lightgcn_{file_tag}.pt"
        torch.save(model.state_dict(), model_path)
        user_dec_map = {v: k for k, v in user_enc.items()}
        item_dec_map = {v: k for k, v in item_enc.items()}
        torch.save({"embeddings": u_emb_all.cpu(), "id_map": user_dec_map},
                   EMB_DIR / f"lightgcn_{file_tag}_user_embeddings.pt")
        torch.save({"embeddings": i_emb_all.cpu(), "id_map": item_dec_map},
                   EMB_DIR / f"lightgcn_{file_tag}_item_embeddings.pt")
        pred_rows = []
        for uid, true_item, ranked in prediction_records:
            rank = ranked.index(true_item) + 1 if true_item in ranked else None
            pred_rows.append({
                "user_idx": uid, "true_item_idx": true_item,
                "top10_item_idxs": ranked,
                "contributor_id": user_dec_map[uid],
                "true_place_id": item_dec_map.get(true_item),
                "top10_place_ids": [item_dec_map[x] for x in ranked],
                "rank": rank,
            })
        pred_dir = Path(os.environ.get("FOODIE_PREDICTION_DIR", "data/predictions"))
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pred_rows).to_parquet(
            pred_dir / f"lightgcn_{file_tag}_predictions.parquet", index=False
        )
        log.info(f"Model → {model_path} | Embeddings → {EMB_DIR}/")

    return model, adj, user_enc, item_enc


# ── Inference ──────────────────────────────────────────────────────────────────
def recommend(contributor_id: str, top_n: int = 10):
    if not MODEL_FILE.exists() or not ENCODERS_FILE.exists():
        log.error("No model found. Run training first.")
        return []

    with open(ENCODERS_FILE, "rb") as f:
        enc = pickle.load(f)

    user_enc  = enc["user_enc"]
    item_dec  = enc["item_dec"]
    n_users   = enc["n_users"]
    n_items   = enc["n_items"]
    emb_dim   = enc["emb_dim"]
    n_layers       = enc.get("n_layers", 3)
    user_feat      = enc.get("user_feat")
    item_feat      = enc.get("item_feat")
    dish_enc       = enc.get("dish_enc", {})
    dish_feat      = enc.get("dish_feat")
    n_dishes       = enc.get("n_dishes", 0)
    use_dish_graph = enc.get("use_dish_graph", False)

    if contributor_id not in user_enc:
        log.warning(f"Unknown user: {contributor_id}")
        return []

    user_idx = user_enc[contributor_id]
    train_df, _, _, _ = load_interactions()
    adj = build_adj(train_df, n_users, n_items)

    if use_dish_graph and n_dishes > 0:
        model = DishHeteroGNN(
            n_users, n_items, n_dishes, emb_dim=emb_dim, n_layers=n_layers,
            user_feat=user_feat, item_feat=item_feat, dish_feat=dish_feat,
            edge_dropout=0.0, use_film=enc.get("use_film", False),
        ).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
        model.eval()
        _, adj_dr, adj_dd, _ = build_dish_data(train_df, user_enc, item_enc)
        exclude = set(train_df[train_df["user_idx"] == user_idx]["item_idx"].values)
        top_idx = model.recommend(adj, adj_dr, adj_dd, user_idx, exclude, top_n=top_n)
    else:
        model = LightGCN(n_users, n_items, emb_dim=emb_dim, n_layers=n_layers,
                         user_feat=user_feat, item_feat=item_feat,
                         edge_dropout=0.0, use_film=enc.get("use_film", False)).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
        model.eval()
        exclude = set(train_df[train_df["user_idx"] == user_idx]["item_idx"].values)
        top_idx = model.recommend(adj, user_idx, exclude, top_n=top_n)

    restaurants = pd.read_parquet(RESTAURANTS_ENR) if RESTAURANTS_ENR.exists() else None
    rest_map = restaurants.set_index("place_id").to_dict("index") if restaurants is not None else {}

    return [
        {
            "place_id": item_dec[idx],
            "name":     rest_map.get(item_dec[idx], {}).get("name", ""),
            "cuisine":  rest_map.get(item_dec[idx], {}).get("cuisine_category", ""),
        }
        for idx in top_idx
    ]


def main():
    parser = argparse.ArgumentParser(description="LightGCN GNN recommendation")
    parser.add_argument("--epochs",         type=int, default=50)
    parser.add_argument("--emb-dim",        type=int, default=64)
    parser.add_argument("--batch-size",     type=int, default=2048,
                        help="BPR mini-batch size (default 2048)")
    parser.add_argument("--layers",         type=int, default=3)
    parser.add_argument("--recommend",      metavar="CONTRIBUTOR_ID")
    parser.add_argument("--no-save",        action="store_true")
    parser.add_argument("--use-dish-graph", action="store_true",
                        help="Enable dish heterogeneous graph path (requires "
                             "extract_dish_sentiment.py + build_dish_embeddings.py output)")
    parser.add_argument("--fp16", action="store_true",
                        help="Mixed-precision training (float16 activations, ~2× memory saving)")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Gradient checkpointing: recompute layer activations during "
                             "backward instead of storing them (~3-4 GB saving at emb_dim=2048)")
    parser.add_argument("--edge-dropout",     type=float, default=0.0,
                        help="Drop fraction of adj edges per epoch — stochastic regularisation "
                             "(0=off, try 0.1)")
    parser.add_argument("--film",             action="store_true",
                        help="FiLM feature gating: γ(feat)·emb + β(feat) instead of emb+proj(feat)")
    parser.add_argument("--warm-restart",     action="store_true",
                        help="CosineAnnealingWarmRestarts instead of CosineAnnealingLR")
    parser.add_argument("--t0",               type=int, default=200,
                        help="First restart period for warm restarts (default 200)")
    parser.add_argument("--hard-neg-refresh", type=int, default=50,
                        help="Rebuild embedding-space hard neg pool every N epochs (0=disable, "
                             "default 50)")
    parser.add_argument("--llm-feat-mode",   default="fast",
                        choices=["none", "fast", "full"],
                        help="Restaurant LLM features: none=skip, fast=analytics only, "
                             "full=analytics+LLM-extracted (default: fast)")
    parser.add_argument("--no-nlp-feat",    action="store_true",
                        help="Disable NLP restaurant features (restaurant_nlp_features.parquet)")
    parser.add_argument("--eval-every",    type=int, default=50,
                        help="Evaluate Recall@10 every N epochs and save best checkpoint "
                             "(0=disable, only eval at end; default 50)")
    parser.add_argument("--prebuilt-features", action="store_true",
                        help="Load pre-joined feature matrices from "
                             "data/*_features_prebuilt.parquet instead of re-joining. "
                             "Run build_training_features.py first.")
    parser.add_argument("--skip-feature-groups", default="",
                        help="Comma-separated feature groups to drop at load time "
                             "(user groups: base,extended,pref; "
                             "item groups: base,nlp,extended,llm). "
                             "E.g. --skip-feature-groups nlp,llm")
    parser.add_argument("--publication-split", action="store_true",
                        help="Use chronological train/validation/test splits (requires >=4 "
                             "unique restaurants per user) and select checkpoints on validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization and negative sampling")
    args = parser.parse_args()

    skip_groups = [g.strip() for g in args.skip_feature_groups.split(",") if g.strip()] \
        if args.skip_feature_groups else None

    if args.recommend:
        recs = recommend(args.recommend)
        print(f"\nTop recommendations for {args.recommend}:")
        for i, r in enumerate(recs, 1):
            print(f"  {i:2}. {r['name']} ({r['cuisine']})")
    else:
        train(epochs=args.epochs, emb_dim=args.emb_dim, n_layers=args.layers,
              batch_size=args.batch_size,
              save=not args.no_save, use_dish_graph=args.use_dish_graph,
              use_fp16=args.fp16, grad_checkpoint=args.grad_checkpoint,
              edge_dropout=args.edge_dropout, use_film=args.film,
              warm_restart=args.warm_restart, t0=args.t0,
              hard_neg_refresh=args.hard_neg_refresh,
              llm_feat_mode=args.llm_feat_mode,
              use_nlp_feat=not args.no_nlp_feat,
              eval_every=args.eval_every,
              prebuilt_features=args.prebuilt_features,
              skip_feature_groups=skip_groups,
              publication_split=args.publication_split,
              seed=args.seed)


if __name__ == "__main__":
    main()
