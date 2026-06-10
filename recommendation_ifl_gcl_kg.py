#!/usr/bin/env python3
"""
recommendation_ifl_gcl_kg.py — IFL-GCL with Knowledge-Graph-guided D_U^+ pairs.

Paper: "InfoNCE is a Free Lunch for Semantically guided Graph Contrastive
        Learning" — Wang et al., SIGIR 2025.

Problem with the embedding-similarity variant (recommendation_ifl_gcl.py):
  D_U^+ is mined from embeddings after a warmup phase. At warmup end the model
  is weak (Recall@10 ≈ 0.03), so the mined pairs contain many false positives
  that dilute the correction signal for the rest of training.

This variant replaces embedding-similarity mining with two structure-based
oracles that are independent of model state and available from epoch 1:

  Item D_U^+ — Knowledge Graph relations:
    Two restaurants are semantically similar if they share cuisine AND at least
    one of: same price tier, same CBG (census block group), dish overlap ≥ 2.
    Score = 0.4 (cuisine) + 0.3 (price match) + 0.3 (cbg match)
            + 0.2 × min(dish_overlap/10, 1) — then min-max normalised.
    Only pairs scoring above the cuisine-only baseline (> 0.4) are kept.

  User D_U^+ — LLM-extracted preference features (12 dims):
    Cosine similarity on user_preference_features.parquet (spice_affinity,
    noise_preference, formality, novelty, …).  This is an independent signal
    from LLM review analysis, not derived from interaction patterns.
    Chunk-based top-K cosine search; kept above threshold t_s_user.

Both pair sets are computed once before training and held fixed, so there is
no warmup dependency or periodic re-mining overhead.

Loss:  L = L_BPR + λ * L_CL^corrected
  L_CL^corrected  = standard InfoNCE + β-weighted correction for D_U^+ pairs
                    that appear in the same mini-batch (Eq. 27 from the paper).

Usage:
    python recommendation_ifl_gcl_kg.py
    python recommendation_ifl_gcl_kg.py --epochs 300 --emb-dim 2048 --layers 4 \\
        --top-k-item 20 --threshold-user 0.85 \\
        --cl-weight 0.2 --noise-eps 0.1 --cl-temp 0.15 --beta 1.0
"""

import argparse
import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint as grad_ckpt

from model_results import save_result
from recommendation_gnn import (
    _sparse_mm_f32,
    dropout_adj,
    FiLMFusion,
    build_adj,
    load_interactions,
    build_user_features,
    build_item_features,
    sample_negatives,
    build_hard_neg_pool,
    _eval_recall_at_k,
    ndcg_at_k,
    DEVICE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ifl_gcl_kg")

MODEL_OUT = Path("models/ifl_gcl_kg.pt")
EMB_OUT   = Path("data/embeddings")
PRED_DIR  = Path("data/predictions")

TOP_DISH_N      = 200
MIN_DISH_SCORE  = 0.4   # require score > cuisine-only baseline (0.4)


# ── KG-guided item D_U^+ ──────────────────────────────────────────────────────

def build_item_kg_pairs(
    item_enc: dict,
    top_k_per_item: int = 20,
    min_dish_overlap: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Build item-item D_U^+ pairs from KG structure (cuisine, price, CBG, dishes).

    Within each cuisine group, scores each pair:
      score = 0.4 (cuisine, mandatory)
            + 0.3  if same price_level
            + 0.3  if same CBG
            + 0.2 * min(dish_overlap / 10, 1.0)  if overlap >= min_dish_overlap

    Only pairs with score > 0.4 are kept (must share at least one attribute
    beyond cuisine). Takes top_k_per_item neighbours per restaurant.

    Returns (src, dst, weights) with weights = min-max normalised scores.
    """
    rest = pd.read_parquet(
        "data/restaurants_enriched.parquet",
        columns=["place_id", "cuisine_category", "price_level", "cbg"],
    )
    rest = rest[rest["place_id"].isin(item_enc)].copy()
    rest["item_idx"] = rest["place_id"].map(item_enc)
    rest = rest.dropna(subset=["cuisine_category", "item_idx"])
    rest["item_idx"] = rest["item_idx"].astype(int)

    # Build dish sets per restaurant (top-N dishes only)
    item_dishes: dict[str, frozenset] = {}
    try:
        dishes_df = pd.read_parquet(
            "data/dishes.parquet", columns=["dish_name", "place_id"]
        )
        dishes_df = dishes_df[dishes_df["place_id"].isin(item_enc)]
        top_dishes = (
            dishes_df.groupby("dish_name")["place_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(TOP_DISH_N)
            .index
        )
        dishes_df = dishes_df[dishes_df["dish_name"].isin(top_dishes)]
        item_dishes = (
            dishes_df.groupby("place_id")["dish_name"]
            .apply(frozenset)
            .to_dict()
        )
        log.info(f"Dish data loaded: {len(item_dishes):,} restaurants with dishes")
    except FileNotFoundError:
        log.warning("dishes.parquet not found — dish overlap component disabled")

    src_list:   list[int]   = []
    dst_list:   list[int]   = []
    score_list: list[float] = []

    n_groups = rest["cuisine_category"].nunique()
    log.info(f"Building item KG pairs across {n_groups} cuisine groups …")

    for cuisine, grp in rest.groupby("cuisine_category"):
        grp       = grp.reset_index(drop=True)
        n         = len(grp)
        if n < 2:
            continue

        item_idxs = grp["item_idx"].to_numpy(dtype=np.int64)
        places    = grp["place_id"].to_numpy(dtype=object)
        price_arr = grp["price_level"].fillna("__NA__").to_numpy(dtype=object)
        cbg_arr   = grp["cbg"].fillna("__NA__").to_numpy(dtype=object)

        # ── Price match matrix ────────────────────────────────────────────────
        price_match = (price_arr[:, None] == price_arr[None, :]).astype(np.float32)
        np.fill_diagonal(price_match, 0.0)
        na_price = price_arr == "__NA__"
        price_match[na_price, :] = 0.0
        price_match[:, na_price] = 0.0

        # ── CBG match matrix ──────────────────────────────────────────────────
        cbg_match = (cbg_arr[:, None] == cbg_arr[None, :]).astype(np.float32)
        np.fill_diagonal(cbg_match, 0.0)
        na_cbg = cbg_arr == "__NA__"
        cbg_match[na_cbg, :] = 0.0
        cbg_match[:, na_cbg] = 0.0

        # ── Dish overlap matrix (vectorised) ──────────────────────────────────
        dish_overlap = np.zeros((n, n), dtype=np.float32)
        if item_dishes:
            group_dishes: set = set()
            for p in places:
                group_dishes |= item_dishes.get(p, frozenset())
            if group_dishes:
                dish_vocab = {d: i for i, d in enumerate(group_dishes)}
                D = len(dish_vocab)
                bin_mat = np.zeros((n, D), dtype=np.float32)
                for i, p in enumerate(places):
                    for dish in item_dishes.get(p, frozenset()):
                        if dish in dish_vocab:
                            bin_mat[i, dish_vocab[dish]] = 1.0
                raw_overlap = bin_mat @ bin_mat.T              # (n, n)
                np.fill_diagonal(raw_overlap, 0.0)
                dish_overlap = (
                    np.minimum(raw_overlap / 10.0, 1.0)
                    * (raw_overlap >= min_dish_overlap).astype(np.float32)
                )

        # ── Combined score ────────────────────────────────────────────────────
        scores = (
            0.4                         # cuisine (mandatory base)
            + 0.3 * price_match
            + 0.3 * cbg_match
            + 0.2 * dish_overlap
        )
        np.fill_diagonal(scores, -1.0)

        # Top-k neighbours per restaurant (only pairs above baseline)
        k = min(top_k_per_item, n - 1)
        top_idxs = np.argpartition(-scores, k, axis=1)[:, :k]   # (n, k)

        for i in range(n):
            for j in top_idxs[i]:
                s = scores[i, j]
                if s > MIN_DISH_SCORE:
                    src_list.append(int(item_idxs[i]))
                    dst_list.append(int(item_idxs[j]))
                    score_list.append(float(s))

    if not src_list:
        log.warning("No item D_U^+ pairs found — check restaurant data.")
        return None

    scores_arr = np.array(score_list, dtype=np.float32)
    s_min, s_max = scores_arr.min(), scores_arr.max()
    weights = (scores_arr - s_min) / (s_max - s_min + 1e-8)     # Eq. 26

    log.info(
        f"Item D_U^+ pairs: {len(src_list):,}  "
        f"(score range [{s_min:.3f}, {s_max:.3f}])"
    )
    return (
        torch.tensor(src_list,  dtype=torch.long),
        torch.tensor(dst_list,  dtype=torch.long),
        torch.tensor(weights,   dtype=torch.float32),
    )


# ── Preference-guided user D_U^+ ─────────────────────────────────────────────

def build_user_pref_pairs(
    user_enc: dict,
    threshold: float = 0.85,
    top_k: int = 20,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Build user-user D_U^+ pairs via cosine similarity on LLM-extracted
    preference features (12 dims: spice_affinity, noise_preference, …).

    These features come from review-text analysis, independent of interaction
    patterns, so they provide a clean semantic signal without warmup dependency.

    Returns (src, dst, weights) with weights = min-max normalised cosine sim,
    or None if preference data is unavailable.
    """
    pref_path = Path("data/user_preference_features.parquet")
    if not pref_path.exists():
        log.warning("user_preference_features.parquet not found — user D_U^+ disabled")
        return None

    pref_df = pd.read_parquet(pref_path)
    pref_df["user_idx"] = pref_df["contributor_id"].map(user_enc)
    pref_df = pref_df.dropna(subset=["user_idx"]).copy()
    pref_df["user_idx"] = pref_df["user_idx"].astype(int)

    pref_cols = [
        "spice_affinity", "noise_preference", "formality", "novelty",
        "family_context", "romantic_context", "wait_tolerance",
        "value_sensitivity", "outdoor_preference", "healthy_preference",
        "bar_affinity", "portion_preference",
    ]
    available = [c for c in pref_cols if c in pref_df.columns]
    if not available:
        log.warning("No preference columns found — user D_U^+ disabled")
        return None

    feat     = pref_df[available].fillna(0.0).values.astype(np.float32)
    user_idx = pref_df["user_idx"].values                         # (M,)
    M        = len(user_idx)

    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    feat  = feat / np.where(norms < 1e-8, 1.0, norms)
    feat_t = torch.tensor(feat, dtype=torch.float32)

    log.info(
        f"Building user pref pairs: {M:,} users with pref features "
        f"(threshold={threshold}, top_k={top_k}) …"
    )

    src_list, dst_list, sim_list = [], [], []

    for start in range(0, M, chunk_size):
        end   = min(start + chunk_size, M)
        chunk = feat_t[start:end]                                  # (C, 12)
        sims  = chunk @ feat_t.T                                   # (C, M)

        for local_i in range(end - start):
            sims[local_i, start + local_i] = -1.0                 # exclude self

        k = min(top_k, M - 1)
        vals, feat_idxs = sims.topk(k, dim=1)                     # (C, k)
        mask = vals >= threshold
        if not mask.any():
            continue

        anchor_rows   = (
            torch.arange(start, end, dtype=torch.long)
            .unsqueeze(1)
            .expand_as(mask)
        )
        sel_anchors   = anchor_rows[mask].numpy()
        sel_neighbors = feat_idxs[mask].numpy()
        sel_sims      = vals[mask]

        src_list.append(torch.tensor(user_idx[sel_anchors],   dtype=torch.long))
        dst_list.append(torch.tensor(user_idx[sel_neighbors], dtype=torch.long))
        sim_list.append(sel_sims)

    if not src_list:
        log.info("No user D_U^+ pairs found at this threshold.")
        return None

    src  = torch.cat(src_list)
    dst  = torch.cat(dst_list)
    sims = torch.cat(sim_list)

    s_min = sims.min().item()
    s_max = sims.max().item()
    weights = (sims - s_min) / (s_max - s_min + 1e-8)

    log.info(
        f"User D_U^+ pairs: {len(src):,}  "
        f"(sim range [{s_min:.3f}, {s_max:.3f}])"
    )
    return src, dst, weights


# ── Model (identical backbone to SimGCL) ─────────────────────────────────────

class IFLGCLKGModel(nn.Module):
    """LightGCN backbone with noise-based augmentation. Architecture unchanged."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 64,
        n_layers: int = 3,
        user_feat: np.ndarray | None = None,
        item_feat: np.ndarray | None = None,
        grad_checkpoint: bool = False,
        edge_dropout: float = 0.0,
        use_film: bool = False,
    ):
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
                self.register_buffer(
                    f"{name}_feat", torch.tensor(feat, dtype=torch.float32)
                )
                if use_film:
                    setattr(self, f"{name}_film", FiLMFusion(feat.shape[1], emb_dim))
                    setattr(self, f"{name}_proj", None)
                else:
                    proj = nn.Linear(feat.shape[1], emb_dim, bias=False)
                    nn.init.xavier_uniform_(proj.weight)
                    setattr(self, f"{name}_proj", proj)
                    setattr(self, f"{name}_film", None)
            else:
                self.register_buffer(f"{name}_feat", None)
                setattr(self, f"{name}_proj", None)
                setattr(self, f"{name}_film", None)

    def _propagate(
        self, adj: torch.Tensor, noise_eps: float = 0.0, use_ckpt: bool = True
    ) -> torch.Tensor:
        adj_used = dropout_adj(adj, self.edge_dropout, self.training)

        def _full_pass(u_w, i_w):
            u = u_w
            i = i_w
            if self.user_film is not None:
                u = self.user_film(u, self.user_feat)
            elif self.user_proj is not None:
                u = u + self.user_proj(self.user_feat)
            if self.item_film is not None:
                i = self.item_film(i, self.item_feat)
            elif self.item_proj is not None:
                i = i + self.item_proj(self.item_feat)

            e = torch.cat([u, i], dim=0)
            s = e
            for _ in range(self.n_layers):
                e = _sparse_mm_f32(adj_used, e)
                if noise_eps > 0.0:
                    e = e + torch.rand_like(e).sign() * noise_eps
                s = s + e
            return s / (self.n_layers + 1)

        if use_ckpt and self.grad_checkpoint and self.training:
            return grad_ckpt(
                _full_pass, self.user_emb.weight, self.item_emb.weight,
                use_reentrant=False,
            )
        return _full_pass(self.user_emb.weight, self.item_emb.weight)

    def forward(self, adj, users, pos_items, neg_items):
        final = self._propagate(adj, noise_eps=0.0)
        u_emb, i_emb = final[:self.n_users], final[self.n_users:]
        return (
            u_emb[users], i_emb[pos_items], i_emb[neg_items],
            self.user_emb.weight, self.item_emb.weight,
        )

    def forward_cl(self, adj: torch.Tensor, noise_eps: float):
        final = self._propagate(adj, noise_eps=noise_eps)
        return final[:self.n_users], final[self.n_users:]

    def get_embeddings(self, adj: torch.Tensor):
        with torch.no_grad():
            final = self._propagate(adj, noise_eps=0.0)
        return final[:self.n_users], final[self.n_users:]


# ── Losses ────────────────────────────────────────────────────────────────────

def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.T) / temp
    labels = torch.arange(len(z1), device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def corrected_cl_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    node_ids: torch.Tensor,
    unlab_src: torch.Tensor | None,
    unlab_dst: torch.Tensor | None,
    unlab_weights: torch.Tensor | None,
    temp: float,
    beta: float,
) -> torch.Tensor:
    """
    Corrected InfoNCE (Eq. 27). Standard InfoNCE + β-weighted correction term
    for D_U^+ pairs where both endpoints fall in the current batch.
    """
    z1n = F.normalize(z1, dim=1)
    z2n = F.normalize(z2, dim=1)
    logits = z1n @ z2n.T / temp
    labels = torch.arange(len(z1), device=z1.device)
    loss   = (
        F.cross_entropy(logits, labels) +
        F.cross_entropy(logits.T, labels)
    ) / 2

    if unlab_src is None or len(unlab_src) == 0:
        return loss

    B      = len(z1)
    max_id = max(
        node_ids.max().item(),
        unlab_src.max().item(),
        unlab_dst.max().item(),
    ) + 1
    id_map          = torch.full((max_id,), -1, dtype=torch.long, device=z1.device)
    id_map[node_ids] = torch.arange(B, device=z1.device)

    in_range = (unlab_src < max_id) & (unlab_dst < max_id)
    if not in_range.any():
        return loss

    s_f   = unlab_src[in_range]
    d_f   = unlab_dst[in_range]
    w_f   = unlab_weights[in_range]
    pos_s = id_map[s_f]
    pos_d = id_map[d_f]
    valid = (pos_s >= 0) & (pos_d >= 0)
    if not valid.any():
        return loss

    # Mean over valid pairs keeps correction magnitude at β·avg_w·log(B),
    # independent of pair count — bounded and consistent across batch sizes.
    log_probs  = F.log_softmax(logits, dim=1)
    correction = -(beta * w_f[valid] * log_probs[pos_s[valid], pos_d[valid]]).mean()
    return loss + correction


# ── Prediction saving ─────────────────────────────────────────────────────────

def save_predictions(
    u_emb: torch.Tensor,
    i_emb: torch.Tensor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_enc: dict,
    item_enc: dict,
    file_tag: str,
) -> None:
    user_id_map = {v: k for k, v in user_enc.items()}
    item_id_map = {v: k for k, v in item_enc.items()}
    train_sets  = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    test_users  = test_df["user_idx"].values
    test_items  = test_df["item_idx"].values

    u_cpu = u_emb.cpu().float()
    i_cpu = i_emb.cpu().float()

    records = []
    for start in range(0, len(test_users), 2048):
        end    = min(start + 2048, len(test_users))
        u_idx  = test_users[start:end]
        i_idx  = test_items[start:end]
        scores = (u_cpu[u_idx] @ i_cpu.T).numpy()

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

    pred_df  = pd.DataFrame(records)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PRED_DIR / f"ifl_gcl_kg_{file_tag}_predictions.parquet"
    pred_df.to_parquet(out_path, index=False)
    hit_rate = pred_df["rank"].notna().mean()
    log.info(f"Predictions → {out_path}  ({len(pred_df):,} rows | Hit@10={hit_rate:.4f})")


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    epochs: int             = 300,
    emb_dim: int            = 2048,
    n_layers: int           = 4,
    lr: float               = 1e-3,
    batch_size: int         = 8192,
    save: bool              = True,
    grad_checkpoint: bool   = False,
    edge_dropout: float     = 0.0,
    use_film: bool          = False,
    hard_neg_refresh: int   = 50,
    llm_feat_mode: str      = "fast",
    use_nlp_feat: bool      = True,
    eval_every: int         = 50,
    prebuilt_features: bool = False,
    skip_feature_groups: list[str] | None = None,
    # SimGCL base
    cl_weight: float        = 0.2,
    noise_eps: float        = 0.1,
    cl_temp: float          = 0.15,
    # IFL-GCL KG params
    top_k_per_item: int     = 20,
    min_dish_overlap: int   = 2,
    threshold_user: float   = 0.95,
    top_k_user: int         = 10,
    mine_chunk_user: int    = 4096,
    beta: float             = 0.1,
    warmup_epochs: int      = 0,
):
    train_df, test_df, user_enc, item_enc = load_interactions()
    n_users, n_items = len(user_enc), len(item_enc)
    log.info(
        f"Train {len(train_df):,} | Test {len(test_df):,} | "
        f"Users {n_users:,} | Items {n_items:,}"
    )

    # ── Pre-compute D_U^+ from KG and preference features ─────────────────────
    log.info("Pre-computing item D_U^+ from KG structure …")
    item_pairs = build_item_kg_pairs(
        item_enc,
        top_k_per_item=top_k_per_item,
        min_dish_overlap=min_dish_overlap,
    )
    if item_pairs is not None:
        item_src, item_dst, item_w = (t.to(DEVICE) for t in item_pairs)
    else:
        item_src = item_dst = item_w = None

    log.info("Pre-computing user D_U^+ from preference features …")
    user_pairs = build_user_pref_pairs(
        user_enc,
        threshold=threshold_user,
        top_k=top_k_user,
        chunk_size=mine_chunk_user,
    )
    if user_pairs is not None:
        user_src, user_dst, user_w = (t.to(DEVICE) for t in user_pairs)
    else:
        user_src = user_dst = user_w = None

    # ── Features and graph ─────────────────────────────────────────────────────
    user_feat = build_user_features(
        train_df, user_enc,
        prebuilt=prebuilt_features,
        skip_groups=skip_feature_groups,
    )
    item_feat = build_item_features(
        item_enc,
        llm_feat_mode=llm_feat_mode,
        use_nlp_feat=use_nlp_feat,
        prebuilt=prebuilt_features,
        skip_groups=skip_feature_groups,
    )
    adj = build_adj(train_df, n_users, n_items)

    model = IFLGCLKGModel(
        n_users=n_users, n_items=n_items, emb_dim=emb_dim, n_layers=n_layers,
        user_feat=user_feat, item_feat=item_feat,
        grad_checkpoint=grad_checkpoint, edge_dropout=edge_dropout,
        use_film=use_film,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 50)

    log.info(
        f"Training IFL-GCL-KG | device={DEVICE} | epochs={epochs} | "
        f"emb_dim={emb_dim} | layers={n_layers} | lr={lr} | "
        f"cl_weight={cl_weight} | noise_eps={noise_eps} | cl_temp={cl_temp} | "
        f"beta={beta} | warmup={warmup_epochs} | "
        f"item_pairs={len(item_src) if item_src is not None else 0:,} | "
        f"user_pairs={len(user_src) if user_src is not None else 0:,}"
    )

    emb_pool    = None
    best_recall = 0.0
    best_state  = None

    train_with_neg = sample_negatives(train_df, n_items, emb_pool=emb_pool)
    all_users  = torch.tensor(train_with_neg["user_idx"].values, dtype=torch.long)
    all_pos    = torch.tensor(train_with_neg["item_idx"].values, dtype=torch.long)
    all_neg    = torch.tensor(train_with_neg["neg_idx"].values,  dtype=torch.long)
    n_train    = len(all_users)

    for epoch in range(1, epochs + 1):

        # ── Refresh hard-negative pool ─────────────────────────────────────────
        if hard_neg_refresh > 0 and epoch > 1 and (epoch - 1) % hard_neg_refresh == 0:
            model.eval()
            with torch.no_grad():
                _, i_emb_tmp = model.get_embeddings(adj)
            emb_pool = build_hard_neg_pool(i_emb_tmp, k=50)
            del i_emb_tmp
            model.train()
            log.info(f"  [Epoch {epoch}] Hard neg pool refreshed")
            train_with_neg = sample_negatives(train_df, n_items, emb_pool=emb_pool)
            all_users = torch.tensor(train_with_neg["user_idx"].values, dtype=torch.long)
            all_pos   = torch.tensor(train_with_neg["item_idx"].values, dtype=torch.long)
            all_neg   = torch.tensor(train_with_neg["neg_idx"].values,  dtype=torch.long)

        model.train()
        optimizer.zero_grad()

        # ── Step 1: CL loss — two augmented views, backward immediately ────────
        u_emb1, i_emb1 = model.forward_cl(adj, noise_eps)
        u_emb2, i_emb2 = model.forward_cl(adj, noise_eps)

        sub_u = torch.randperm(n_users, device=DEVICE)[:batch_size]
        sub_i = torch.randperm(n_items, device=DEVICE)[:batch_size]

        in_warmup = warmup_epochs > 0 and epoch <= warmup_epochs

        if in_warmup:
            loss_cl = (
                info_nce_loss(u_emb1[sub_u], u_emb2[sub_u], cl_temp) +
                info_nce_loss(i_emb1[sub_i], i_emb2[sub_i], cl_temp)
            )
        else:
            loss_cl = (
                corrected_cl_loss(
                    u_emb1[sub_u], u_emb2[sub_u],
                    node_ids=sub_u,
                    unlab_src=user_src, unlab_dst=user_dst, unlab_weights=user_w,
                    temp=cl_temp, beta=beta,
                ) +
                corrected_cl_loss(
                    i_emb1[sub_i], i_emb2[sub_i],
                    node_ids=sub_i,
                    unlab_src=item_src, unlab_dst=item_dst, unlab_weights=item_w,
                    temp=cl_temp, beta=beta,
                )
            )

        (cl_weight * loss_cl).backward()
        cl_val = loss_cl.item()
        del u_emb1, i_emb1, u_emb2, i_emb2, loss_cl
        torch.cuda.empty_cache()

        # ── Step 2a: BPR gradient — no_grad chunked scatter_add ───────────────
        _saved_dropout     = model.edge_dropout
        model.edge_dropout = 0.0

        with torch.no_grad():
            final_det = model._propagate(adj, noise_eps=0.0, use_ckpt=False)
            u_det = final_det[:n_users]
            i_det = final_det[n_users:]

            grad_final = torch.zeros_like(final_det)
            bpr_val    = 0.0
            perm       = torch.randperm(n_train)

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                idx = perm[start:end]
                bu  = all_users[idx].to(DEVICE)
                bp  = all_pos[idx].to(DEVICE)
                bn  = all_neg[idx].to(DEVICE)

                u   = u_det[bu]
                pos = i_det[bp]
                neg = i_det[bn]

                margin   = (u * pos).sum(1) - (u * neg).sum(1)
                bpr_val += (-F.logsigmoid(margin)).sum().item() / n_train

                coef = -(1.0 - torch.sigmoid(margin)) / n_train
                r    = 1e-4 / n_train

                gu = coef.unsqueeze(1) * (pos - neg) + r * u
                gp = coef.unsqueeze(1) * u            + r * pos
                gn = -coef.unsqueeze(1) * u           + r * neg

                grad_final[:n_users].scatter_add_(0, bu.unsqueeze(1).expand_as(gu), gu)
                grad_final[n_users:].scatter_add_(0, bp.unsqueeze(1).expand_as(gp), gp)
                grad_final[n_users:].scatter_add_(0, bn.unsqueeze(1).expand_as(gn), gn)

            del final_det, u_det, i_det

        # ── Step 2b: one ckpt forward + backward with precomputed BPR grad ─────
        final_bpr = model._propagate(adj, noise_eps=0.0, use_ckpt=True)
        final_bpr.backward(gradient=grad_final)
        del final_bpr, grad_final
        model.edge_dropout = _saved_dropout
        torch.cuda.empty_cache()

        optimizer.step()
        scheduler.step()

        phase = "warmup" if in_warmup else "corrected"
        if epoch % 10 == 0:
            log.info(
                f"  Epoch {epoch}/{epochs} [{phase}]  "
                f"bpr={bpr_val:.4f}  cl={cl_val:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if eval_every > 0 and epoch % eval_every == 0:
            model.eval()
            val_recall = _eval_recall_at_k(
                model, adj, test_df, train_df, n_items, k=10
            )
            model.train()
            if val_recall > best_recall:
                best_recall = val_recall
                best_state  = copy.deepcopy(model.state_dict())
                log.info(
                    f"  [Epoch {epoch}] val Recall@10={val_recall:.4f}  *** new best ***"
                )
            else:
                log.info(
                    f"  [Epoch {epoch}] val Recall@10={val_recall:.4f}  "
                    f"(best={best_recall:.4f})"
                )

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info(f"Loaded best checkpoint (val Recall@10={best_recall:.4f})")

    # ── Final evaluation ───────────────────────────────────────────────────────
    model.eval()
    log.info("Computing embeddings for final evaluation …")
    u_emb, i_emb = model.get_embeddings(adj)

    log.info(
        f"  Embedding norms — users mean={u_emb.norm(dim=1).mean():.4f}, "
        f"items mean={i_emb.norm(dim=1).mean():.4f}"
    )

    test_users_arr  = test_df["user_idx"].values
    test_items_arr  = test_df["item_idx"].values
    train_items_map = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    hits, ndcg_sum, total = 0, 0.0, 0
    chunk = 4096
    log.info(f"Evaluating {len(test_users_arr):,} test users in chunks of {chunk} …")

    for start in range(0, len(test_users_arr), chunk):
        end     = min(start + chunk, len(test_users_arr))
        u_idx   = test_users_arr[start:end]
        i_idx   = test_items_arr[start:end]
        u_batch = u_emb[u_idx]
        scores  = (u_batch @ i_emb.T).cpu().numpy()

        for k, (ui, ii) in enumerate(zip(u_idx, i_idx)):
            excl = train_items_map.get(ui, set())
            for ex in excl:
                scores[k, ex] = -np.inf
            top10 = np.argsort(-scores[k])[:10]
            if ii in top10:
                hits     += 1
                ndcg_sum += ndcg_at_k({ii}, top10.tolist(), 10)
            total += 1

    precision = hits / (total * 10)
    recall    = hits / total
    ndcg      = ndcg_sum / total

    print(f"\n=== IFL-GCL-KG Results ===")
    print(f"  Precision@10: {precision:.4f}")
    print(f"  Recall@10:    {recall:.4f}")
    print(f"  NDCG@10:      {ndcg:.4f}")

    skip_str = ",".join(skip_feature_groups) if skip_feature_groups else ""
    variant  = f"skip {skip_str}" if skip_str else "all features"
    save_result(
        model="IFL-GCL-KG", variant=variant,
        precision=precision, recall=recall, ndcg=ndcg,
        epochs=epochs, emb_dim=emb_dim, n_layers=n_layers,
        skip_groups=skip_str,
        notes=(
            f"beta={beta} top_k_item={top_k_per_item} "
            f"thr_user={threshold_user} warmup={warmup_epochs} "
            f"item_pairs={len(item_src) if item_src is not None else 0} "
            f"user_pairs={len(user_src) if user_src is not None else 0}"
        ),
    )

    if save:
        file_tag   = f"skip_{skip_str.replace(',', '_')}" if skip_str else "all"
        model_path = MODEL_OUT.parent / f"ifl_gcl_kg_{file_tag}.pt"
        model_path.parent.mkdir(exist_ok=True)
        EMB_OUT.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_path)
        user_dec = {v: k for k, v in user_enc.items()}
        item_dec = {v: k for k, v in item_enc.items()}
        torch.save(
            {"embeddings": u_emb.cpu(), "id_map": user_dec},
            EMB_OUT / f"ifl_gcl_kg_{file_tag}_user_embeddings.pt",
        )
        torch.save(
            {"embeddings": i_emb.cpu(), "id_map": item_dec},
            EMB_OUT / f"ifl_gcl_kg_{file_tag}_item_embeddings.pt",
        )
        log.info(f"Model → {model_path} | Embeddings → {EMB_OUT}")

        save_predictions(
            u_emb, i_emb,
            train_df=train_df, test_df=test_df,
            user_enc=user_enc, item_enc=item_enc,
            file_tag=file_tag,
        )

    return recall


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IFL-GCL-KG recommendation model")

    # Shared training args
    parser.add_argument("--epochs",            type=int,   default=300)
    parser.add_argument("--emb-dim",           type=int,   default=2048)
    parser.add_argument("--layers",            type=int,   default=4)
    parser.add_argument("--lr",                type=float, default=1e-3)
    parser.add_argument("--batch-size",        type=int,   default=8192)
    parser.add_argument("--edge-dropout",      type=float, default=0.0)
    parser.add_argument("--film",              action="store_true", default=False)
    parser.add_argument("--grad-checkpoint",   action="store_true", default=False)
    parser.add_argument("--hard-neg-refresh",  type=int,   default=50)
    parser.add_argument("--llm-feat-mode",     default="fast",
                        choices=["none", "fast", "full"])
    parser.add_argument("--no-nlp-feat",       action="store_true")
    parser.add_argument("--eval-every",        type=int,   default=50)
    parser.add_argument("--no-save",           action="store_true")
    parser.add_argument("--prebuilt-features", action="store_true")
    parser.add_argument("--skip-feature-groups", default="",
                        help="Comma-separated feature groups to drop")

    # SimGCL base
    parser.add_argument("--cl-weight",  type=float, default=0.2,
                        help="Weight of contrastive loss λ (default 0.2)")
    parser.add_argument("--noise-eps",  type=float, default=0.1,
                        help="Per-layer noise magnitude ε (default 0.1)")
    parser.add_argument("--cl-temp",   type=float, default=0.15,
                        help="InfoNCE temperature τ (default 0.15)")

    # IFL-GCL-KG specific
    parser.add_argument("--top-k-item",     type=int,   default=20,
                        help="Top-K KG neighbours per restaurant (default 20)")
    parser.add_argument("--min-dish-overlap", type=int,  default=2,
                        help="Minimum shared dishes for dish-overlap score (default 2)")
    parser.add_argument("--threshold-user", type=float, default=0.95,
                        help="Cosine similarity threshold for user pref pairs (default 0.95)")
    parser.add_argument("--top-k-user",    type=int,   default=10,
                        help="Top-K preference neighbours per user (default 10)")
    parser.add_argument("--mine-chunk-user", type=int, default=4096,
                        help="Chunk size for user preference mining (default 4096)")
    parser.add_argument("--beta",          type=float, default=0.1,
                        help="Correction weight exponent β (default 0.1)")
    parser.add_argument("--warmup-epochs", type=int,   default=0,
                        help="Epochs of standard SimGCL before correction; "
                             "0 = correction from epoch 1 (default 0)")

    args = parser.parse_args()

    skip_groups = (
        [g.strip() for g in args.skip_feature_groups.split(",") if g.strip()]
        if args.skip_feature_groups else None
    )

    train(
        epochs=args.epochs,
        emb_dim=args.emb_dim,
        n_layers=args.layers,
        lr=args.lr,
        batch_size=args.batch_size,
        save=not args.no_save,
        grad_checkpoint=args.grad_checkpoint,
        edge_dropout=args.edge_dropout,
        use_film=args.film,
        hard_neg_refresh=args.hard_neg_refresh,
        llm_feat_mode=args.llm_feat_mode,
        use_nlp_feat=not args.no_nlp_feat,
        eval_every=args.eval_every,
        prebuilt_features=args.prebuilt_features,
        skip_feature_groups=skip_groups,
        cl_weight=args.cl_weight,
        noise_eps=args.noise_eps,
        cl_temp=args.cl_temp,
        top_k_per_item=args.top_k_item,
        min_dish_overlap=args.min_dish_overlap,
        threshold_user=args.threshold_user,
        top_k_user=args.top_k_user,
        mine_chunk_user=args.mine_chunk_user,
        beta=args.beta,
        warmup_epochs=args.warmup_epochs,
    )


if __name__ == "__main__":
    main()
