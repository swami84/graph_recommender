#!/usr/bin/env python3
"""
recommendation_kgat.py — KGAT: Knowledge Graph Attention Network for Recommendation.

Paper: "KGAT: Knowledge Graph Attention Network for Recommendation"
       Wang et al., KDD 2019.

Key idea over LightGCN / SimGCL:
  Explicitly encodes side information (cuisine type, price tier, CBG location,
  popular dishes) as a Knowledge Graph and uses attentive propagation over the
  joint CF + KG graph.  Item embeddings are enriched by KG context before the
  user-item CF propagation, letting the model reason about *why* a user might
  like a restaurant (right cuisine? right price? same neighbourhood?).

Knowledge Graph triples built from:
  (restaurant, HAS_CUISINE,  cuisine_entity)    — 17 categories
  (restaurant, HAS_PRICE,    price_entity)       — 4 tiers
  (restaurant, IN_CBG,       cbg_entity)         — up to 1 015 CBGs
  (restaurant, SERVES_DISH,  dish_entity)        — top-200 common dishes

Attention formula (inner-product, no W_r projection):
  att(h, r, t) = (e_h + e_r)^T e_t  /  sqrt(d)
  π(h, r, t)   = softmax over N(h)

Propagation:
  1. L_KG layers:  attentive aggregation over KG → enriched item embeddings
  2. L_CF layers:  LightGCN over user-item bipartite graph

Loss: BPR (same as LightGCN / SimGCL; no separate KG auxiliary task)

Usage:
    python -m foodie.modeling.recommendation_kgat
    python -m foodie.modeling.recommendation_kgat --epochs 300 --emb-dim 2048 \\
        --kg-layers 2 --cf-layers 4 --eval-every 50
"""

import argparse
import copy
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint as grad_ckpt
from transformers import Adafactor

from foodie.modeling.model_results import save_result
from foodie.modeling.recommendation_gnn import (
    _sparse_mm_f32, dropout_adj,
    bpr_loss,
    build_adj,
    load_interactions,
    build_user_features,
    build_item_features,
    sample_negatives,
    build_hard_neg_pool,
    _eval_recall_at_k,
    ndcg_at_k,
    _is_main_rank, _sync_grads, init_distributed,
    DEVICE, RESTAURANTS_ENR,
)

_KG_AGG_CHUNK = 512   # embedding-dim strip for KG aggregation — keeps (E,chunk) not (E,d)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kgat")
log.setLevel(logging.INFO)
if not log.handlers:
    _lh = logging.StreamHandler()
    _lh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_lh)
    log.propagate = False

MODEL_OUT = Path("models/kgat.pt")
EMB_OUT   = Path("data/embeddings")

# KG relation IDs
REL_CUISINE  = 0
REL_PRICE    = 1
REL_CBG      = 2
REL_DISH     = 3
REL_CBG_NEAR = 4   # spatial: CBG → neighboring CBG (k-NN by centroid distance)
N_RELATIONS  = 5

CBG_SPATIAL_K = 5  # k nearest CBG neighbors per CBG

TOP_DISH_N = 200   # keep top-200 most common dishes as KG entities
DISHES_FILE = Path(os.environ.get("FOODIE_DISHES_FILE", "data/dishes.parquet"))


# ── Knowledge Graph construction ───────────────────────────────────────────────

def build_kg(item_enc: dict, use_spatial_cbg: bool = True,
             use_dish_kg: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Build KG triples from restaurant side-information.

    Returns:
        head_ids   LongTensor (n_triples,)  — item indices (into item_enc)
        rel_ids    LongTensor (n_triples,)  — relation id
        tail_ids   LongTensor (n_triples,)  — entity indices (offset by n_items)
        n_kg_ents  int                       — number of KG entities (non-item)
    """
    n_items = len(item_enc)
    rest = pd.read_parquet(
        str(RESTAURANTS_ENR),
        columns=["place_id", "cuisine_category", "price_level", "cbg", "lat", "lng"],
    )
    # Only keep items that appear in item_enc
    rest = rest[rest["place_id"].isin(item_enc)].copy()
    rest["item_idx"] = rest["place_id"].map(item_enc)

    # Build entity index registries
    cuisine_enc: dict[str, int] = {}
    price_enc:   dict[str, int] = {}
    cbg_enc:     dict[str, int] = {}
    dish_enc:    dict[str, int] = {}

    ent_offset = 0  # running offset into KG entity space

    def _reg(d: dict, key: str) -> int:
        if key not in d:
            d[key] = len(d)
        return d[key]

    heads, rels, tails = [], [], []

    # ── Cuisine triples ────────────────────────────────────────────────────────
    for _, row in rest.dropna(subset=["cuisine_category"]).iterrows():
        eid = _reg(cuisine_enc, row["cuisine_category"])
        heads.append(int(row["item_idx"]))
        rels.append(REL_CUISINE)
        tails.append(eid)          # offset applied later

    ent_offset += len(cuisine_enc)

    # ── Price triples ──────────────────────────────────────────────────────────
    for _, row in rest.dropna(subset=["price_level"]).iterrows():
        eid = _reg(price_enc, row["price_level"])
        heads.append(int(row["item_idx"]))
        rels.append(REL_PRICE)
        tails.append(ent_offset + eid)

    ent_offset += len(price_enc)

    # ── CBG triples ────────────────────────────────────────────────────────────
    cbg_offset = ent_offset   # save start of CBG block for spatial triples below
    for _, row in rest.dropna(subset=["cbg"]).iterrows():
        eid = _reg(cbg_enc, row["cbg"])
        heads.append(int(row["item_idx"]))
        rels.append(REL_CBG)
        tails.append(ent_offset + eid)

    ent_offset += len(cbg_enc)

    # ── CBG spatial triples (CBG → k-NN neighboring CBG) ──────────────────────
    if use_spatial_cbg:
        # CBG centroids = median lat/lng of member restaurants
        cbg_coords = (
            rest.dropna(subset=["cbg", "lat", "lng"])
            .groupby("cbg")[["lat", "lng"]]
            .median()
        )
        cbg_coords = cbg_coords[cbg_coords.index.isin(cbg_enc)]

        if len(cbg_coords) > 1:
            lat  = np.radians(cbg_coords["lat"].values)
            lng  = np.radians(cbg_coords["lng"].values)
            dlat = lat[:, None] - lat[None, :]
            dlng = lng[:, None] - lng[None, :]
            a    = np.sin(dlat / 2) ** 2 + (
                   np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlng / 2) ** 2)
            dist_km = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            np.fill_diagonal(dist_km, np.inf)

            k = min(CBG_SPATIAL_K, len(cbg_coords) - 1)
            cbg_list = cbg_coords.index.tolist()
            n_spatial = 0
            for i, cbg_h in enumerate(cbg_list):
                nn_indices = np.argpartition(dist_km[i], k)[:k]
                h_eid = cbg_offset + cbg_enc[cbg_h]
                for j in nn_indices:
                    cbg_t = cbg_list[j]
                    t_eid = cbg_offset + cbg_enc[cbg_t]
                    heads.append(h_eid)
                    rels.append(REL_CBG_NEAR)
                    tails.append(t_eid)
                    n_spatial += 1
            log.info(f"CBG spatial triples: {n_spatial} ({len(cbg_list)} CBGs × k={k})")
    else:
        log.info("CBG spatial triples: disabled (--no-spatial-cbg)")

    # Dish links are review-derived LLM information. Publication runs use a
    # training-only link file, while the non-LLM condition omits this relation.
    if use_dish_kg and DISHES_FILE.exists():
        dishes = pd.read_parquet(DISHES_FILE, columns=["dish_name", "place_id"])
        dishes = dishes[dishes["place_id"].isin(item_enc)]
        top_dishes = (
            dishes.groupby("dish_name")["place_id"].nunique()
            .sort_values(ascending=False)
            .head(TOP_DISH_N)
            .index
        )
        dishes = dishes[dishes["dish_name"].isin(top_dishes)]
        for row in dishes.itertuples(index=False):
            item_idx = item_enc.get(row.place_id)
            if item_idx is None:
                continue
            eid = _reg(dish_enc, row.dish_name)
            heads.append(item_idx)
            rels.append(REL_DISH)
            tails.append(ent_offset + eid)
    else:
        log.info("Dish KG triples: disabled or no leakage-safe dish file")

    ent_offset += len(dish_enc)

    log.info(
        f"KG entities: cuisine={len(cuisine_enc)} price={len(price_enc)} "
        f"cbg={len(cbg_enc)} dish={len(dish_enc)} total={ent_offset}"
    )
    log.info(f"KG triples: {len(heads):,} (incl. CBG spatial)")

    head_t = torch.LongTensor(heads).to(DEVICE)
    rel_t  = torch.LongTensor(rels).to(DEVICE)
    tail_t = torch.LongTensor(tails).to(DEVICE)

    return head_t, rel_t, tail_t, ent_offset


# ── KGAT Model ─────────────────────────────────────────────────────────────────

class KGAT(nn.Module):
    """
    KGAT: attentive propagation over the joint CF + KG graph.

    Entity space layout (for entity_emb):
      indices 0 .. n_items-1          → restaurant embeddings
      indices n_items .. n_items+n_kg_ents-1 → KG entity embeddings

    Propagation:
      1. KG layers: for each item, attend to its KG entity neighbors and
         aggregate their embeddings → enriched item representations.
      2. CF layers: LightGCN on user-item bipartite graph using enriched items.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_kg_ents: int,
        emb_dim: int,
        n_kg_layers: int,
        n_cf_layers: int,
        kg_heads: torch.Tensor,
        kg_rels:  torch.Tensor,
        kg_tails: torch.Tensor,
        user_feat: np.ndarray | None = None,
        item_feat: np.ndarray | None = None,
        edge_dropout: float = 0.0,
        kg_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_users     = n_users
        self.n_items     = n_items
        self.n_kg_ents   = n_kg_ents
        self.n_kg_layers = n_kg_layers
        self.n_cf_layers = n_cf_layers
        self.edge_dropout = edge_dropout
        self.kg_dropout   = kg_dropout

        n_total_ents = n_items + n_kg_ents

        self.user_emb   = nn.Embedding(n_users, emb_dim)
        self.entity_emb = nn.Embedding(n_total_ents, emb_dim)
        self.rel_emb    = nn.Embedding(N_RELATIONS, emb_dim)

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # Register KG triples (fixed; not learned)
        self.register_buffer("kg_heads", kg_heads)
        self.register_buffer("kg_rels",  kg_rels)
        self.register_buffer("kg_tails", kg_tails)

        # Optional feature projection (additive to item embedding at layer 0)
        if item_feat is not None:
            self.register_buffer("item_feat", torch.tensor(item_feat, dtype=torch.float32))
            self.item_proj = nn.Linear(item_feat.shape[1], emb_dim, bias=False)
            nn.init.xavier_uniform_(self.item_proj.weight)
        else:
            self.register_buffer("item_feat", None)
            self.item_proj = None

        if user_feat is not None:
            self.register_buffer("user_feat", torch.tensor(user_feat, dtype=torch.float32))
            self.user_proj = nn.Linear(user_feat.shape[1], emb_dim, bias=False)
            nn.init.xavier_uniform_(self.user_proj.weight)
        else:
            self.register_buffer("user_feat", None)
            self.user_proj = None

        self._scale = emb_dim ** -0.5

    # ── KG attention propagation ───────────────────────────────────────────────

    def _kg_propagate(self, entity_emb: torch.Tensor) -> torch.Tensor:
        """
        One layer of attentive KG aggregation.

        att(h, r, t) = (e_h + e_r)^T e_t * scale
        π(h, r, t)   = softmax over N(h)
        e_h^new      = LeakyReLU(e_h + Σ_t π * e_t)
        """
        heads = self.kg_heads  # (E,)
        rels  = self.kg_rels   # (E,)
        tails = self.kg_tails  # (E,)

        # Optional KG edge dropout
        if self.training and self.kg_dropout > 0.0:
            mask  = torch.rand(len(heads), device=DEVICE) > self.kg_dropout
            heads = heads[mask]
            rels  = rels[mask]
            tails = tails[mask]

        # Fuse e_h + e_r into one tensor — avoids materialising 3 separate (E,d) tensors
        e_hr = entity_emb[heads] + self.rel_emb(rels)   # (E, d); saves 1×(E,d) vs original
        att_logits = (e_hr * entity_emb[tails]).sum(dim=-1) * self._scale  # (E,)
        del e_hr

        # Scatter-softmax (stable) per head entity
        n_total   = entity_emb.size(0)
        _dt       = entity_emb.dtype
        max_logit = torch.full((n_total,), float("-inf"), device=DEVICE, dtype=_dt)
        max_logit.scatter_reduce_(0, heads, att_logits, reduce="amax", include_self=True)
        exp_logits = torch.exp(att_logits - max_logit[heads])          # (E,)

        sum_exp = torch.zeros(n_total, device=DEVICE, dtype=_dt)
        sum_exp.scatter_add_(0, heads, exp_logits)
        att_w = exp_logits / (sum_exp[heads] + 1e-9)                   # (E,)

        # Chunked aggregation: process embedding dims in _KG_AGG_CHUNK strips
        # keeps peak at 2×(E,chunk) instead of 2×(E,d)
        d   = entity_emb.size(1)
        agg = torch.zeros_like(entity_emb)
        for d0 in range(0, d, _KG_AGG_CHUNK):
            d1    = min(d0 + _KG_AGG_CHUNK, d)
            e_t_c = entity_emb[tails, d0:d1]                           # (E, chunk)
            w_c   = att_w.unsqueeze(1) * e_t_c                         # (E, chunk)
            agg[:, d0:d1].scatter_add_(0, heads.unsqueeze(1).expand_as(w_c), w_c)
            del e_t_c, w_c

        return F.leaky_relu(entity_emb + agg, negative_slope=0.2)

    # ── CF propagation ─────────────────────────────────────────────────────────

    def _cf_propagate(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LightGCN layers on user-item bipartite graph."""
        all_emb = torch.cat([user_emb, item_emb], dim=0)  # (n_users + n_items, d)
        adj_used = dropout_adj(adj, self.edge_dropout, self.training)

        summed = all_emb
        for _ in range(self.n_cf_layers):
            all_emb = _sparse_mm_f32(adj_used, all_emb)
            summed  = summed + all_emb

        final = summed / (self.n_cf_layers + 1)
        return final[:self.n_users], final[self.n_users:]

    # ── Full forward pass ──────────────────────────────────────────────────────

    def _get_enriched_items(self) -> torch.Tensor:
        """Run KG propagation to enrich item embeddings."""
        ent_emb = self.entity_emb.weight  # (n_items + n_kg_ents, d)

        # Optionally add item feature projection at layer 0
        if self.item_proj is not None:
            proj = self.item_proj(self.item_feat)         # (n_items, d)
            ent_emb = ent_emb.clone()
            ent_emb[:self.n_items] = ent_emb[:self.n_items] + proj

        summed = ent_emb
        for _ in range(self.n_kg_layers):
            ent_emb = self._kg_propagate(ent_emb)
            summed  = summed + ent_emb

        enriched = summed / (self.n_kg_layers + 1)
        return enriched[:self.n_items]     # only item slice

    def forward(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ):
        u_emb0 = self.user_emb.weight
        if self.user_proj is not None:
            u_emb0 = u_emb0 + self.user_proj(self.user_feat)

        item_emb = self._get_enriched_items()
        u_final, i_final = self._cf_propagate(u_emb0, item_emb, adj)

        return (
            u_final[users],
            i_final[pos_items],
            i_final[neg_items],
            self.user_emb.weight,     # for L2 reg
            self.entity_emb.weight[:self.n_items],
        )

    def _propagate_all(self, adj: torch.Tensor, use_ckpt: bool = False) -> torch.Tensor:
        """
        Full forward returning cat([u_final, i_final]) — used for manual-gradient BPR.
        use_ckpt=True wraps the KG+CF pass in gradient checkpointing.
        """
        def _full_pass(u_w, ent_w):
            u_emb0 = u_w
            if self.user_proj is not None:
                u_emb0 = u_emb0 + self.user_proj(self.user_feat)
            ent_emb = ent_w
            if self.item_proj is not None:
                proj = self.item_proj(self.item_feat)
                ent_emb = ent_emb.clone()
                ent_emb[:self.n_items] = ent_emb[:self.n_items] + proj
            summed = ent_emb
            for _ in range(self.n_kg_layers):
                ent_emb = self._kg_propagate(ent_emb)
                summed  = summed + ent_emb
            item_emb = summed[:self.n_items] / (self.n_kg_layers + 1)
            u_final, i_final = self._cf_propagate(u_emb0, item_emb, adj)
            return torch.cat([u_final, i_final], dim=0)

        if use_ckpt and self.training:
            return grad_ckpt(_full_pass, self.user_emb.weight, self.entity_emb.weight,
                             use_reentrant=False)
        return _full_pass(self.user_emb.weight, self.entity_emb.weight)

    def get_embeddings(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Clean embeddings for evaluation — no dropout, no grad."""
        with torch.no_grad():
            u_emb0 = self.user_emb.weight
            if self.user_proj is not None:
                u_emb0 = u_emb0 + self.user_proj(self.user_feat)
            item_emb = self._get_enriched_items()
            u_final, i_final = self._cf_propagate(u_emb0, item_emb, adj)
        return u_final, i_final


# ── Training ───────────────────────────────────────────────────────────────────

CHECKPOINT_CSV = Path("results/training_checkpoints.csv")


def _log_checkpoint(run_id: str, variant: str, epoch: int, recall: float) -> None:
    import csv
    from datetime import datetime
    CHECKPOINT_CSV.parent.mkdir(exist_ok=True)
    write_header = not CHECKPOINT_CSV.exists()
    with open(CHECKPOINT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "run_id", "model", "variant", "epoch", "recall_at_10"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    run_id, "KGAT", variant, epoch, f"{recall:.4f}"])


def train(
    epochs: int           = 300,
    emb_dim: int          = 2048,
    n_kg_layers: int      = 2,
    n_cf_layers: int      = 4,
    lr: float             = 1e-3,
    batch_size: int       = 2048,
    save: bool            = True,
    edge_dropout: float   = 0.1,
    kg_dropout: float     = 0.1,
    hard_neg_refresh: int = 50,
    llm_feat_mode: str    = "fast",
    use_nlp_feat: bool    = True,
    eval_every: int       = 50,
    prebuilt_features: bool = False,
    skip_feature_groups: list[str] | None = None,
    use_spatial_cbg: bool = True,
    run_tag: str = "",
):
    from datetime import datetime
    train_df, test_df, user_enc, item_enc = load_interactions()
    n_users, n_items = len(user_enc), len(item_enc)

    skip_str    = ",".join(skip_feature_groups) if skip_feature_groups else ""
    spatial_tag = "+spatial_cbg" if use_spatial_cbg else "no_spatial"
    ckpt_variant = f"{'skip ' + skip_str if skip_str else 'all features'} {spatial_tag}"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    user_feat = build_user_features(train_df, user_enc,
                                    prebuilt=prebuilt_features,
                                    skip_groups=skip_feature_groups)
    item_feat = build_item_features(item_enc, llm_feat_mode=llm_feat_mode,
                                    use_nlp_feat=use_nlp_feat,
                                    prebuilt=prebuilt_features,
                                    skip_groups=skip_feature_groups)
    adj = build_adj(train_df, n_users, n_items)

    use_dish_kg = "dish_llm" not in (skip_feature_groups or [])
    kg_heads, kg_rels, kg_tails, n_kg_ents = build_kg(
        item_enc, use_spatial_cbg=use_spatial_cbg, use_dish_kg=use_dish_kg
    )

    model = KGAT(
        n_users=n_users,
        n_items=n_items,
        n_kg_ents=n_kg_ents,
        emb_dim=emb_dim,
        n_kg_layers=n_kg_layers,
        n_cf_layers=n_cf_layers,
        kg_heads=kg_heads,
        kg_rels=kg_rels,
        kg_tails=kg_tails,
        user_feat=user_feat,
        item_feat=item_feat,
        edge_dropout=edge_dropout,
        kg_dropout=kg_dropout,
    ).to(DEVICE)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = model.to(torch.bfloat16)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if dist.is_initialized() else model

    optimizer = Adafactor(model.parameters(), lr=lr,
                          relative_step=False, scale_parameter=False, warmup_init=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 50)

    log.info(
        f"Training KGAT | device={DEVICE} | epochs={epochs} | emb_dim={emb_dim} | "
        f"kg_layers={n_kg_layers} | cf_layers={n_cf_layers} | "
        f"edge_dropout={edge_dropout} | kg_dropout={kg_dropout}"
    )

    neg_pool    = None
    best_recall = 0.0
    best_state  = None

    all_users_t = torch.LongTensor(train_df["user_idx"].values)
    all_pos_t   = torch.LongTensor(train_df["item_idx"].values)
    n_train     = len(train_df)

    for epoch in range(1, epochs + 1):

        if epoch % hard_neg_refresh == 1:
            with torch.no_grad():
                _, i_emb = raw_model.get_embeddings(adj)
            neg_pool = build_hard_neg_pool(i_emb, k=50)
            if epoch > 1:
                log.info(f"  [Epoch {epoch}] Hard neg pool refreshed")

        raw_model.train()

        # Sample one negative per training pair for this epoch
        neg_df   = sample_negatives(train_df, n_items, emb_pool=neg_pool)
        all_neg_t = torch.LongTensor(neg_df["neg_idx"].values)

        optimizer.zero_grad()

        # ── Manual-gradient BPR (same pattern as SimGCL) ──────────────────
        # Phase a: no_grad chunked loop → compute d(BPR)/d(final_emb)
        # Phase b: one ckpt forward + one backward with precomputed gradient
        _saved_dropout    = raw_model.edge_dropout
        raw_model.edge_dropout = 0.0

        with torch.no_grad():
            final_det = raw_model._propagate_all(adj, use_ckpt=False)
            u_det = final_det[:n_users]
            i_det = final_det[n_users:]

            grad_final = torch.zeros_like(final_det)
            bpr_val    = 0.0
            perm       = torch.randperm(n_train)

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                idx = perm[start:end]
                bu  = all_users_t[idx].to(DEVICE)
                bp  = all_pos_t[idx].to(DEVICE)
                bn  = all_neg_t[idx].to(DEVICE)

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

        final_bpr = raw_model._propagate_all(adj, use_ckpt=True)
        scalar = (final_bpr * grad_final.detach()).sum()
        del final_bpr   # not needed during checkpoint recomputation — free 2 GiB before backward
        scalar.backward()
        del grad_final, scalar
        raw_model.edge_dropout = _saved_dropout
        torch.cuda.empty_cache()

        optimizer.step()
        scheduler.step()

        if _is_main_rank() and (epoch == 1 or epoch % 10 == 0):
            log.info(
                f"  Epoch {epoch}/{epochs}  "
                f"bpr={bpr_val:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if eval_every > 0 and epoch % eval_every == 0:
            raw_model.eval()
            val_recall = _eval_recall_at_k(
                raw_model, adj, test_df, train_df, n_items, k=10
            )
            raw_model.train()
            _log_checkpoint(run_id, ckpt_variant, epoch, val_recall)
            if val_recall > best_recall:
                best_recall = val_recall
                best_state  = copy.deepcopy(raw_model.state_dict())
                log.info(f"  [Epoch {epoch}] val Recall@10={val_recall:.4f}  *** new best ***")
            else:
                log.info(f"  [Epoch {epoch}] val Recall@10={val_recall:.4f}  (best={best_recall:.4f})")

    if best_state is not None:
        raw_model.load_state_dict(best_state)
        log.info(f"Loaded best checkpoint (val Recall@10={best_recall:.4f})")

    # ── Final evaluation ───────────────────────────────────────────────────────
    raw_model.eval()
    log.info("Computing embeddings for final evaluation…")
    u_emb, i_emb = raw_model.get_embeddings(adj)

    log.info(f"  Embedding norms — users mean={u_emb.norm(dim=1).mean():.4f}, "
             f"items mean={i_emb.norm(dim=1).mean():.4f}")

    test_users  = test_df["user_idx"].values
    test_items  = test_df["item_idx"].values
    train_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    hits, ndcg_sum, total = 0, 0.0, 0
    chunk = 4096

    for start in range(0, len(test_users), chunk):
        end     = min(start + chunk, len(test_users))
        u_idx   = test_users[start:end]
        i_idx   = test_items[start:end]
        u_batch = u_emb[u_idx]
        scores  = (u_batch @ i_emb.T).float().cpu().numpy()

        for k, (ui, ii) in enumerate(zip(u_idx, i_idx)):
            excl = train_items.get(ui, set())
            for ex in excl:
                scores[k, ex] = -np.inf
            top10 = np.argsort(-scores[k])[:10]
            if ii in top10:
                hits += 1
                ndcg_sum += ndcg_at_k({ii}, top10.tolist(), 10)
            total += 1

    precision = hits / (total * 10)
    recall    = hits / total
    ndcg      = ndcg_sum / total

    print(f"\n=== KGAT Results ===")
    print(f"  Precision@10: {precision:.4f}")
    print(f"  Recall@10:    {recall:.4f}")
    print(f"  NDCG@10:      {ndcg:.4f}")

    skip_str = ",".join(skip_feature_groups) if skip_feature_groups else ""
    base_variant = f"skip {skip_str}" if skip_str else "all features"
    variant  = f"{base_variant} +spatial_cbg" if use_spatial_cbg else base_variant
    if run_tag:
        variant = f"{variant} [{run_tag}]"
    save_result(
        model="KGAT", variant=variant,
        precision=precision, recall=recall, ndcg=ndcg,
        epochs=epochs, emb_dim=emb_dim, n_layers=n_cf_layers,
        skip_groups=skip_str,
        notes=(f"CBG spatial k={CBG_SPATIAL_K}" if use_spatial_cbg else "no_spatial_cbg")
              + (f" {run_tag}" if run_tag else ""),
    )

    if save:
        if _is_main_rank():
            spatial_suffix = "" if use_spatial_cbg else "_nospatial"
            file_tag   = f"skip_{skip_str.replace(',', '_')}" if skip_str else "all"
            file_tag   = f"{file_tag}{spatial_suffix}" + (f"_{run_tag}" if run_tag else "")
            model_path = MODEL_OUT.parent / f"kgat_{file_tag}.pt"
            model_path.parent.mkdir(exist_ok=True)
            EMB_OUT.mkdir(exist_ok=True)
            torch.save(raw_model.state_dict(), model_path)
            user_dec = {v: k for k, v in user_enc.items()}
            item_dec = {v: k for k, v in item_enc.items()}
            torch.save({"embeddings": u_emb.cpu(), "id_map": user_dec},
                       EMB_OUT / f"kgat_{file_tag}_user_embeddings.pt")
            torch.save({"embeddings": i_emb.cpu(), "id_map": item_dec},
                       EMB_OUT / f"kgat_{file_tag}_item_embeddings.pt")
            log.info(f"Model → {model_path} | Embeddings → {EMB_OUT}")

    return recall


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KGAT recommendation model")
    parser.add_argument("--epochs",      type=int,   default=300)
    parser.add_argument("--emb-dim",     type=int,   default=2048)
    parser.add_argument("--kg-layers",   type=int,   default=2,
                        help="KG attentive propagation layers (default 2)")
    parser.add_argument("--cf-layers",   type=int,   default=4,
                        help="CF LightGCN propagation layers (default 4)")
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch-size",  type=int,   default=2048)
    parser.add_argument("--edge-dropout",type=float, default=0.1)
    parser.add_argument("--kg-dropout",  type=float, default=0.1,
                        help="KG edge dropout during training (default 0.1)")
    parser.add_argument("--hard-neg-refresh", type=int, default=50)
    parser.add_argument("--llm-feat-mode", default="fast",
                        choices=["none", "fast", "full"])
    parser.add_argument("--no-nlp-feat", action="store_true")
    parser.add_argument("--eval-every",  type=int,   default=50)
    parser.add_argument("--no-save",     action="store_true")
    parser.add_argument("--prebuilt-features", action="store_true",
                        help="Load pre-joined feature matrices from "
                             "data/*_features_prebuilt.parquet. "
                             "Run build_training_features.py first.")
    parser.add_argument("--skip-feature-groups", default="",
                        help="Comma-separated feature groups to drop "
                             "(user: base,extended,pref; item: base,nlp,extended,llm)")
    parser.add_argument("--no-spatial-cbg", action="store_true",
                        help="Disable CBG-CBG spatial KG triples (baseline comparison)")
    parser.add_argument("--run-tag", default="",
                        help="tag appended to output filenames + result variant (for A/B runs)")
    args, _ = parser.parse_known_args()
    init_distributed()

    skip_groups = [g.strip() for g in args.skip_feature_groups.split(",") if g.strip()] \
        if args.skip_feature_groups else None

    train(
        epochs=args.epochs,
        emb_dim=args.emb_dim,
        n_kg_layers=args.kg_layers,
        n_cf_layers=args.cf_layers,
        lr=args.lr,
        batch_size=args.batch_size,
        save=not args.no_save,
        edge_dropout=args.edge_dropout,
        kg_dropout=args.kg_dropout,
        hard_neg_refresh=args.hard_neg_refresh,
        llm_feat_mode=args.llm_feat_mode,
        use_nlp_feat=not args.no_nlp_feat,
        eval_every=args.eval_every,
        prebuilt_features=args.prebuilt_features,
        skip_feature_groups=skip_groups,
        use_spatial_cbg=not args.no_spatial_cbg,
        run_tag=args.run_tag,
    )


if __name__ == "__main__":
    main()
