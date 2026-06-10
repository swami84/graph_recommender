#!/usr/bin/env python3
"""
recommendation_kgat_sal.py — KGAT + Personalized Self-Augmented Learning (SAL).

Extends KGAT (Wang et al., KDD 2019) with the Personalized Self-Augmented Learning
paradigm from SelfGNN (Liu et al., SIGIR 2024), adapted for restaurant recommendation.

New components over vanilla KGAT:
  1. Temporal period splitting — interactions divided into T short-term periods;
     per-period LightGCN on KG-enriched item embeddings captures how collaborative
     signals shift across time (seasonal dining, neighborhood changes, life events).
  2. Stability-gated temporal aggregation — per-user stability weights (learned MLP)
     gate each period's contribution, down-weighting noisy one-off visits (tourist
     reviews, temporary cravings). Aggregated via learnable weighted mean (replaces
     GRU to avoid 2.9 GiB cuDNN workspace at 131k batch size).
  3. Personalized SAL denoising — SAL margin loss trains the stability weighter to
     align short-term scores with long-term preference directions.

Architecture:
  KGAT KG propagation (shared)  → enriched item embeddings
  Full CF on global adj          → u_kgat, i_kgat    (long-term, BPR signal)
  Per-period CF on adj_t (×T)   → e_u_t             (no_grad, outside ckpt)
  period_proj(e_u_t, u_kgat)    → d_temp projections (inside ckpt, cheap recompute)
  StabilityWeighter              → w_t               (scalar gate per user per period)
  TemporalModule: weighted mean + out_proj → u_temporal
  u_final = u_kgat + u_temporal

Loss:
  L = L_BPR(u_final, i_kgat) + λ_sal * L_SAL + λ2 * ||Θ||²

L_SAL trains StabilityWeighter to align short-term scores with long-term scores:
  Sample (u_i,v_j) and (u_p,v_p') from period-t sub-graph.
  d1 = w_{t,i} * s̄(u_i,v_j) - w_{t,p} * s̄(u_p,v_p')    (long-term margin, supervises)
  d2 = s_t(u_i,v_j) - s_t(u_p,v_p')                        (short-term margin, direction)
  L_SAL = Σ_t max(0, 1 - d1 · d2)

Usage:
    python recommendation_kgat_sal.py --prebuilt-features
    python recommendation_kgat_sal.py --epochs 300 --emb-dim 2048 --n-periods 3 \\
        --kg-layers 2 --cf-layers 4 --d-temporal 256 --d-sal 32 \\
        --lambda-sal 1e-6 --eval-every 50 --prebuilt-features
"""

import argparse
import copy
import csv
import logging
import os
from datetime import datetime
from pathlib import Path

import contextlib
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

from model_results import save_result
from recommendation_gnn import (
    _sparse_mm_f32, dropout_adj,
    build_adj, load_interactions,
    build_user_features, build_item_features,
    sample_negatives, build_hard_neg_pool,
    _eval_recall_at_k, ndcg_at_k, _is_main_rank, _sync_grads, init_distributed, DEVICE,
)
from recommendation_kgat import build_kg, N_RELATIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kgat_sal")
log.setLevel(logging.INFO)
if not log.handlers:
    _lh = logging.StreamHandler()
    _lh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_lh)
    log.propagate = False

MODEL_OUT      = Path("models/kgat_sal.pt")
EMB_OUT        = Path("data/embeddings")
PRED_DIR       = Path("data/predictions")
CHECKPOINT_CSV = Path("results/training_checkpoints.csv")

CBG_SPATIAL_K  = 5   # inherited from KGAT config
TOP_DISH_N     = 200
_KG_AGG_CHUNK  = 512  # embedding-dim strip for KG aggregation — keeps (E,chunk) not (E,d)


# ── Period adjacency builder ───────────────────────────────────────────────────

def build_period_adjs(
    train_df: pd.DataFrame,
    n_users:  int,
    n_items:  int,
    n_periods: int,
) -> tuple[list[torch.Tensor], list[pd.DataFrame]]:
    """
    Split train_df into n_periods temporal sub-graphs by timestamp_days_ago quantile.
    Period 0 = oldest interactions, period T-1 = most recent (feeds last into GRU).
    Returns (adj_list, df_list) in temporal order oldest → newest.
    """
    if "timestamp_days_ago" not in train_df.columns or train_df["timestamp_days_ago"].isna().all():
        log.warning("No timestamp_days_ago — splitting interactions randomly into %d periods", n_periods)
        perm  = np.random.permutation(len(train_df))
        bands = [train_df.iloc[chunk].copy() for chunk in np.array_split(perm, n_periods)]
    else:
        qs         = np.linspace(0, 1, n_periods + 1)
        thresholds = train_df["timestamp_days_ago"].quantile(qs).values

        # Smaller days_ago = more recent; Q0=min (most recent), Q1=max (oldest).
        # Build bands in ascending order of days_ago (index 0 = most recent).
        bands = []
        for i in range(n_periods):
            lo, hi = thresholds[i], thresholds[i + 1]
            if i < n_periods - 1:
                mask = (train_df["timestamp_days_ago"] >= lo) & (train_df["timestamp_days_ago"] < hi)
            else:
                mask = train_df["timestamp_days_ago"] >= lo
            bands.append(train_df[mask].copy())

        # Reverse so period 0 = oldest (largest days_ago) → most recent period last.
        bands = bands[::-1]

    period_adjs = []
    for t, pdf in enumerate(bands):
        n_e = len(pdf)
        log.info("  Period %d: %d edges (%.1f%%)", t, n_e, 100 * n_e / max(len(train_df), 1))
        period_adjs.append(build_adj(pdf, n_users, n_items) if n_e > 0
                           else build_adj(train_df.iloc[:0], n_users, n_items))

    return period_adjs, bands


# ── Auxiliary modules ──────────────────────────────────────────────────────────

class StabilityWeighter(nn.Module):
    """
    Per-user stability weight w_{t,i} = σ(Γ W2 + b2),
    where Γ = σ((ê_proj + e_t_proj + ê_proj ⊙ e_t_proj) W1 + b1).

    Operates on d_temporal-dim projections (not full emb_dim) to keep the gate
    tensor small — inputs are pre-projected period embeddings.
    Gradient flows only through W1/W2 (SAL + BPR via temporal).
    """

    def __init__(self, d_temporal: int, d_sal: int = 32):
        super().__init__()
        self.W1 = nn.Linear(d_temporal, d_sal)
        self.W2 = nn.Linear(d_sal, 1)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.zeros_(self.W1.bias)
        nn.init.zeros_(self.W2.bias)

    def forward(self, long_proj: torch.Tensor, short_proj: torch.Tensor) -> torch.Tensor:
        # long_proj, short_proj: (N, d_temporal) — both detached
        gate  = long_proj + short_proj + long_proj * short_proj  # element-wise interaction
        gamma = torch.sigmoid(self.W1(gate))                     # (N, d_sal)
        return torch.sigmoid(self.W2(gamma)).squeeze(-1)         # (N,)


class TemporalModule(nn.Module):
    """
    Stability-gated weighted mean aggregator over T period embeddings.

    Replaces the GRU from the original design to avoid cuDNN workspace allocation
    (~2.9 GiB for batch_size=131k, T=3), which caused OOM during ckpt recompute.

    period_proj: D → d_temp (shared with stability input projection)
    period_bias: learnable recency weights (softmax → scalar per period)
    out_proj:    d_temp → D
    """

    def __init__(self, emb_dim: int, d_temporal: int = 256, n_periods: int = 3):
        super().__init__()
        self.period_proj = nn.Linear(emb_dim, d_temporal, bias=False)
        self.period_bias = nn.Parameter(torch.zeros(n_periods))
        self.out_proj    = nn.Linear(d_temporal, emb_dim, bias=False)
        nn.init.xavier_uniform_(self.period_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        period_projs:   list[torch.Tensor],   # T × (N, d_temp) — pre-projected
        period_weights: list[torch.Tensor],   # T × (N,) — stability gates (with grad)
    ) -> torch.Tensor:
        recency = F.softmax(self.period_bias, dim=0)   # (T,) learnable recency
        agg = torch.zeros_like(period_projs[0])
        for t, (p, w) in enumerate(zip(period_projs, period_weights)):
            agg = agg + (w * recency[t]).unsqueeze(1) * p
        return self.out_proj(agg)                      # (N, D)


# ── KGAT-SAL Model ─────────────────────────────────────────────────────────────

class KGAT_SAL(nn.Module):
    """
    KGAT with Personalized Self-Augmented Learning.

    KG + full-graph CF  → u_kgat, i_kgat   (long-term KGAT embeddings)
    Per-period CF (×T)  → e_u_t             (temporal features, no_grad)
    StabilityWeighter   → w_t               (per-user gate, trained by SAL + BPR)
    TemporalModule      → u_temporal        (dynamic user preference signal)
    u_final = u_kgat + u_temporal

    period_adjs must be set after construction: model.period_adjs = period_adjs
    """

    def __init__(
        self,
        n_users:    int,
        n_items:    int,
        n_kg_ents:  int,
        emb_dim:    int,
        n_kg_layers: int,
        n_cf_layers: int,
        kg_heads:   torch.Tensor,
        kg_rels:    torch.Tensor,
        kg_tails:   torch.Tensor,
        user_feat:  np.ndarray | None = None,
        item_feat:  np.ndarray | None = None,
        edge_dropout: float = 0.0,
        kg_dropout:   float = 0.1,
        d_temporal:   int   = 256,
        d_sal:        int   = 32,
        n_periods:    int   = 3,
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

        self.register_buffer("kg_heads", kg_heads)
        self.register_buffer("kg_rels",  kg_rels)
        self.register_buffer("kg_tails", kg_tails)

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

        self.d_temporal = d_temporal
        self.stability  = StabilityWeighter(d_temporal, d_sal)
        self.temporal   = TemporalModule(emb_dim, d_temporal, n_periods=n_periods)
        self._scale     = emb_dim ** -0.5

        # Set by caller after construction (sparse tensors can't be buffers portably)
        self.period_adjs: list[torch.Tensor] = []

    # ── KG propagation (identical to KGAT) ────────────────────────────────────

    def _kg_propagate(self, entity_emb: torch.Tensor) -> torch.Tensor:
        heads, rels, tails = self.kg_heads, self.kg_rels, self.kg_tails
        if self.training and self.kg_dropout > 0.0:
            mask  = torch.rand(len(heads), device=DEVICE) > self.kg_dropout
            heads, rels, tails = heads[mask], rels[mask], tails[mask]

        # Fuse e_h + e_r — avoids 3 separate (E,d) tensors in memory simultaneously
        e_hr = entity_emb[heads] + self.rel_emb(rels)   # (E, d)
        att_logits = (e_hr * entity_emb[tails]).sum(dim=-1) * self._scale  # (E,)
        del e_hr

        n_total   = entity_emb.size(0)
        _dt       = entity_emb.dtype
        max_logit = torch.full((n_total,), float("-inf"), device=DEVICE, dtype=_dt)
        max_logit.scatter_reduce_(0, heads, att_logits, reduce="amax", include_self=True)
        exp_logits = torch.exp(att_logits - max_logit[heads])
        sum_exp    = torch.zeros(n_total, device=DEVICE, dtype=_dt)
        sum_exp.scatter_add_(0, heads, exp_logits)
        att_w = exp_logits / (sum_exp[heads] + 1e-9)   # (E,)

        # Chunked scatter — peak kept at 2×(E,chunk) instead of 2×(E,d)
        d   = entity_emb.size(1)
        agg = torch.zeros_like(entity_emb)
        for d0 in range(0, d, _KG_AGG_CHUNK):
            d1    = min(d0 + _KG_AGG_CHUNK, d)
            e_t_c = entity_emb[tails, d0:d1]               # (E, chunk)
            w_c   = att_w.unsqueeze(1) * e_t_c             # (E, chunk)
            agg[:, d0:d1].scatter_add_(0, heads.unsqueeze(1).expand_as(w_c), w_c)
            del e_t_c, w_c

        return F.leaky_relu(entity_emb + agg, negative_slope=0.2)

    def _get_enriched_items(self) -> torch.Tensor:
        ent_emb = self.entity_emb.weight
        if self.item_proj is not None:
            proj = self.item_proj(self.item_feat)
            ent_emb = ent_emb.clone()
            ent_emb[:self.n_items] = ent_emb[:self.n_items] + proj
        summed = ent_emb
        for _ in range(self.n_kg_layers):
            ent_emb = self._kg_propagate(ent_emb)
            summed  = summed + ent_emb
        return (summed / (self.n_kg_layers + 1))[:self.n_items]

    # ── CF propagation ─────────────────────────────────────────────────────────

    def _cf_propagate(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        adj:      torch.Tensor,
        use_dropout: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_emb  = torch.cat([user_emb, item_emb], dim=0)
        adj_used = dropout_adj(adj, self.edge_dropout, self.training and use_dropout)
        summed   = all_emb
        for _ in range(self.n_cf_layers):
            all_emb = _sparse_mm_f32(adj_used, all_emb)
            summed  = summed + all_emb
        final = summed / (self.n_cf_layers + 1)
        return final[:self.n_users], final[self.n_users:]

    # ── Full forward (BPR training) ────────────────────────────────────────────

    def _propagate_all(self, adj: torch.Tensor, use_ckpt: bool = False) -> torch.Tensor:
        """
        Returns cat([u_final, i_kgat]) where u_final = u_kgat + u_temporal.

        Period CF passes are pre-computed OUTSIDE the grad_ckpt closure so they are
        not recomputed during backward (each pass ≈1.2 GiB; T=3 → 3.6 GiB peak saved).
        The period embeddings are passed as extra tensor arguments into _full so ckpt
        can still checkpoint all other activations (KG + full CF + temporal projection).
        """
        # Pre-compute period CF embeddings (no_grad — not part of BPR computation graph).
        # Project to d_temporal immediately: stores 3×128 MB instead of 3×1.02 GiB,
        # freeing ~2.7 GiB that would otherwise be live during the ckpt backward recompute.
        with torch.no_grad():
            u0_pre = self.user_emb.weight.detach()
            if self.user_proj is not None:
                u0_pre = u0_pre + self.user_proj(self.user_feat)
            enriched_pre = self._get_enriched_items().detach()
            period_u_raw: list[torch.Tensor] = []
            for adj_t in self.period_adjs:
                u_t, _ = self._cf_propagate(u0_pre, enriched_pre, adj_t, use_dropout=False)
                period_u_raw.append(self.temporal.period_proj(u_t).detach())  # (n_users, d_temp)

        def _full(u_w: torch.Tensor, ent_w: torch.Tensor, *period_tensors) -> torch.Tensor:
            u0 = u_w + self.user_proj(self.user_feat) if self.user_proj is not None else u_w

            # KG propagation
            ent_emb = ent_w
            if self.item_proj is not None:
                proj = self.item_proj(self.item_feat)
                ent_emb = ent_emb.clone()
                ent_emb[:self.n_items] = ent_emb[:self.n_items] + proj
            summed_ent = ent_emb
            for _ in range(self.n_kg_layers):
                ent_emb    = self._kg_propagate(ent_emb)
                summed_ent = summed_ent + ent_emb
            enriched = (summed_ent / (self.n_kg_layers + 1))[:self.n_items]

            # Full CF on global adj → long-term embeddings
            adj_used = dropout_adj(adj, self.edge_dropout, self.training)
            all_emb  = torch.cat([u0, enriched], dim=0)
            summed   = all_emb
            for _ in range(self.n_cf_layers):
                all_emb = _sparse_mm_f32(adj_used, all_emb)
                summed  = summed + all_emb
            final  = summed / (self.n_cf_layers + 1)
            u_kgat = final[:self.n_users]
            i_kgat = final[self.n_users:]

            # Temporal: period_tensors are pre-projected to d_temp outside ckpt.
            # Only long_proj (from u_kgat) is computed here; its recompute during
            # ckpt backward is cheap (N × d_temp matmul, ~128 MB).
            if period_tensors:
                long_proj    = self.temporal.period_proj(u_kgat.detach())  # (N, d_temp)
                period_projs = list(period_tensors)                        # already (N, d_temp)
                weights      = [self.stability(long_proj, pp) for pp in period_projs]
                u_temporal   = self.temporal(period_projs, weights)
            else:
                u_temporal = torch.zeros_like(u_kgat)

            return torch.cat([u_kgat + u_temporal, i_kgat], dim=0)

        if use_ckpt and self.training:
            return grad_ckpt(_full, self.user_emb.weight, self.entity_emb.weight,
                             *period_u_raw, use_reentrant=False)
        return _full(self.user_emb.weight, self.entity_emb.weight, *period_u_raw)

    # ── SAL loss ───────────────────────────────────────────────────────────────

    def sal_loss(
        self,
        period_dfs: list[pd.DataFrame],
        n_samples:  int = 4096,
    ) -> torch.Tensor:
        """
        Personalized Self-Augmented Learning margin loss.

        For each period t, samples pairs of edges (u_i,v_j) and (u_p,v_p'),
        computes per-user stability weights, and applies a margin loss that
        pushes short-term scores to agree with long-term directional signals.

        Long-term embedding: user_emb + proj (before CF) and KG-enriched items.
        This avoids a separate full KGAT forward while still providing a
        semantically meaningful reference that captures item attributes and
        user static preferences.

        Gradient flows only through stability weights (W1, b1, W2, b2).
        """
        with torch.no_grad():
            u0 = self.user_emb.weight
            if self.user_proj is not None:
                u0 = u0 + self.user_proj(self.user_feat)
            enriched = self._get_enriched_items()   # KG-enriched, no CF (cheap)

        loss_terms: list[torch.Tensor] = []

        for period_df, adj_t in zip(period_dfs, self.period_adjs):
            if len(period_df) < 4:
                continue

            with torch.no_grad():
                u_t, i_t = self._cf_propagate(u0, enriched, adj_t, use_dropout=False)

            n_samp   = min(n_samples, len(period_df))
            u_arr    = period_df["user_idx"].values
            v_arr    = period_df["item_idx"].values
            perm1    = np.random.permutation(len(period_df))[:n_samp]
            perm2    = np.random.permutation(len(period_df))[:n_samp]

            u_i = torch.LongTensor(u_arr[perm1]).to(DEVICE)
            v_j = torch.LongTensor(v_arr[perm1]).to(DEVICE)
            u_p = torch.LongTensor(u_arr[perm2]).to(DEVICE)
            v_p = torch.LongTensor(v_arr[perm2]).to(DEVICE)

            # Short-term scores from period embeddings (detached, direction signal)
            with torch.no_grad():
                s_ij = (u_t[u_i] * i_t[v_j]).sum(-1)
                s_pj = (u_t[u_p] * i_t[v_p]).sum(-1)
                d2   = (s_ij - s_pj).detach()

                # Long-term scores from pre-CF embeddings (supervision signal)
                sbar_ij = (u0[u_i] * enriched[v_j]).sum(-1)
                sbar_pj = (u0[u_p] * enriched[v_p]).sum(-1)

            # Stability weights — project D→d_temp first; grad via period_proj + W1/W2
            long_i  = self.temporal.period_proj(u0[u_i].detach())       # (n_samp, d_temp)
            short_i = self.temporal.period_proj(u_t[u_i].detach())
            long_p  = self.temporal.period_proj(u0[u_p].detach())
            short_p = self.temporal.period_proj(u_t[u_p].detach())
            w_i = self.stability(long_i, short_i)                        # (n_samp,)
            w_p = self.stability(long_p, short_p)                        # (n_samp,)

            # d1: long-term directional difference weighted by stability
            d1 = w_i * sbar_ij.detach() - w_p * sbar_pj.detach()   # grad via w_i, w_p

            sal_t = F.relu(1.0 - d1 * d2).mean()
            loss_terms.append(sal_t)

        if not loss_terms:
            return torch.zeros(1, device=DEVICE, requires_grad=True).squeeze()
        return torch.stack(loss_terms).mean()

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def get_embeddings(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Clean embeddings for evaluation; accesses stored period_adjs."""
        with torch.no_grad():
            final = self._propagate_all(adj, use_ckpt=False)
        return final[:self.n_users], final[self.n_users:]


# ── Checkpoint logging (mirrors KGAT) ──────────────────────────────────────────

def _log_checkpoint(run_id: str, variant: str, epoch: int, recall: float) -> None:
    CHECKPOINT_CSV.parent.mkdir(exist_ok=True)
    write_header = not CHECKPOINT_CSV.exists()
    with open(CHECKPOINT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "run_id", "model", "variant", "epoch", "recall_at_10"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    run_id, "KGAT-SAL", variant, epoch, f"{recall:.4f}"])


# ── Prediction saver ────────────────────────────────────────────────────────────

def save_predictions(
    u_emb:    torch.Tensor,
    i_emb:    torch.Tensor,
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    user_enc: dict,
    item_enc: dict,
    file_tag: str,
) -> None:
    """Generate top-10 predictions for all test users and write to parquet."""
    user_id_map = {v: k for k, v in user_enc.items()}
    item_id_map = {v: k for k, v in item_enc.items()}
    train_sets  = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    test_users  = test_df["user_idx"].values
    test_items  = test_df["item_idx"].values
    u_cpu       = u_emb.cpu().float()
    i_cpu       = i_emb.cpu().float()

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
    out_path = PRED_DIR / f"kgat_sal_{file_tag}_predictions.parquet"
    pred_df.to_parquet(out_path, index=False)
    hit_rate = pred_df["rank"].notna().mean()
    log.info("Predictions → %s  (%d rows | Hit@10=%.4f)", out_path, len(pred_df), hit_rate)


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    epochs: int           = 300,
    emb_dim: int          = 2048,
    n_kg_layers: int      = 2,
    n_cf_layers: int      = 4,
    n_periods: int        = 3,
    d_temporal: int       = 256,
    d_sal: int            = 32,
    lambda_sal: float     = 1e-6,
    lr: float             = 1e-3,
    batch_size: int       = 2048,
    sal_samples: int      = 4096,
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
) -> float:
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    skip_str = ",".join(skip_feature_groups) if skip_feature_groups else ""
    variant  = f"{'skip ' + skip_str if skip_str else 'all features'} T={n_periods}"

    train_df, test_df, user_enc, item_enc = load_interactions()
    n_users, n_items = len(user_enc), len(item_enc)

    user_feat = build_user_features(train_df, user_enc,
                                    prebuilt=prebuilt_features,
                                    skip_groups=skip_feature_groups)
    item_feat = build_item_features(item_enc, llm_feat_mode=llm_feat_mode,
                                    use_nlp_feat=use_nlp_feat,
                                    prebuilt=prebuilt_features,
                                    skip_groups=skip_feature_groups)
    adj = build_adj(train_df, n_users, n_items)

    log.info("Building temporal period adjacency matrices (T=%d)…", n_periods)
    period_adjs, period_dfs = build_period_adjs(train_df, n_users, n_items, n_periods)

    log.info("Building knowledge graph…")
    kg_heads, kg_rels, kg_tails, n_kg_ents = build_kg(item_enc, use_spatial_cbg=use_spatial_cbg)

    model = KGAT_SAL(
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
        d_temporal=d_temporal,
        d_sal=d_sal,
        n_periods=n_periods,
    ).to(DEVICE)
    model.period_adjs = period_adjs
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = model.to(torch.bfloat16)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if dist.is_initialized() else model

    optimizer = Adafactor(model.parameters(), lr=lr,
                          relative_step=False, scale_parameter=False, warmup_init=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 50)

    log.info(
        "Training KGAT-SAL | device=%s | epochs=%d | emb_dim=%d | "
        "kg_layers=%d | cf_layers=%d | T=%d | d_temp=%d | d_sal=%d | "
        "λ_sal=%.1e | edge_dr=%.2f | kg_dr=%.2f",
        DEVICE, epochs, emb_dim, n_kg_layers, n_cf_layers,
        n_periods, d_temporal, d_sal, lambda_sal, edge_dropout, kg_dropout,
    )

    all_users_t = torch.LongTensor(train_df["user_idx"].values)
    all_pos_t   = torch.LongTensor(train_df["item_idx"].values)
    n_train     = len(train_df)
    neg_pool    = None
    best_recall = 0.0
    best_state  = None

    for epoch in range(1, epochs + 1):

        # ── Hard negative pool refresh ─────────────────────────────────────────
        if epoch % hard_neg_refresh == 1:
            with torch.no_grad():
                _, i_emb_tmp = raw_model.get_embeddings(adj)
            neg_pool = build_hard_neg_pool(i_emb_tmp, k=50)
            del i_emb_tmp
            if epoch > 1:
                log.info("  [Epoch %d] Hard neg pool refreshed", epoch)

        raw_model.train()
        neg_df    = sample_negatives(train_df, n_items, emb_pool=neg_pool)
        all_neg_t = torch.LongTensor(neg_df["neg_idx"].values)

        optimizer.zero_grad()

        # ── Step 1: SAL backward — trains stability_weighter ──────────────────
        L_sal = raw_model.sal_loss(period_dfs, n_samples=sal_samples)
        _nosync = model.no_sync() if dist.is_initialized() else contextlib.nullcontext()
        with _nosync:
            (lambda_sal * L_sal).backward()
        sal_val = L_sal.item()
        del L_sal
        torch.cuda.empty_cache()

        # ── Step 2a: BPR gradient — manual scatter_add (no_grad) ─────────────
        _saved_dropout    = raw_model.edge_dropout
        raw_model.edge_dropout = 0.0

        with torch.no_grad():
            final_det = raw_model._propagate_all(adj, use_ckpt=False)
            u_det     = final_det[:n_users]
            i_det     = final_det[n_users:]

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

        # ── Step 2b: ckpt forward + backward adds to SAL-accumulated grads ───
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
                "  Epoch %d/%d  bpr=%.4f  sal=%.6f  lr=%.2e",
                epoch, epochs, bpr_val, sal_val, scheduler.get_last_lr()[0],
            )

        if eval_every > 0 and epoch % eval_every == 0:
            raw_model.eval()
            val_recall = _eval_recall_at_k(raw_model, adj, test_df, train_df, n_items, k=10)
            raw_model.train()
            _log_checkpoint(run_id, variant, epoch, val_recall)
            if val_recall > best_recall:
                best_recall = val_recall
                best_state  = copy.deepcopy(raw_model.state_dict())
                log.info("  [Epoch %d] val Recall@10=%.4f  *** new best ***", epoch, val_recall)
            else:
                log.info("  [Epoch %d] val Recall@10=%.4f  (best=%.4f)",
                         epoch, val_recall, best_recall)

    # ── Restore best checkpoint ────────────────────────────────────────────────
    if best_state is not None:
        raw_model.load_state_dict(best_state)
        log.info("Loaded best checkpoint (val Recall@10=%.4f)", best_recall)

    # ── Final evaluation ───────────────────────────────────────────────────────
    raw_model.eval()
    log.info("Computing final embeddings…")
    u_emb, i_emb = raw_model.get_embeddings(adj)

    log.info("  Embedding norms — users mean=%.4f  items mean=%.4f",
             u_emb.norm(dim=1).mean().item(), i_emb.norm(dim=1).mean().item())

    test_users  = test_df["user_idx"].values
    test_items  = test_df["item_idx"].values
    train_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    hits, ndcg_sum, total = 0, 0.0, 0
    chunk = 4096

    for start in range(0, len(test_users), chunk):
        end     = min(start + chunk, len(test_users))
        u_idx   = test_users[start:end]
        i_idx   = test_items[start:end]
        scores  = (u_emb[u_idx] @ i_emb.T).cpu().numpy()

        for k, (ui, ii) in enumerate(zip(u_idx, i_idx)):
            for ex in train_items.get(ui, set()):
                scores[k, ex] = -np.inf
            top10 = np.argsort(-scores[k])[:10]
            if ii in top10:
                hits += 1
                ndcg_sum += ndcg_at_k({ii}, top10.tolist(), 10)
            total += 1

    precision = hits / (total * 10)
    recall    = hits / total
    ndcg      = ndcg_sum / total

    print(f"\n=== KGAT-SAL Results ===")
    print(f"  Precision@10: {precision:.4f}")
    print(f"  Recall@10:    {recall:.4f}")
    print(f"  NDCG@10:      {ndcg:.4f}")

    spatial_tag = "+spatial_cbg" if use_spatial_cbg else "no_spatial"
    full_variant = f"{'skip ' + skip_str if skip_str else 'all features'} T={n_periods} {spatial_tag}"
    save_result(
        model="KGAT-SAL", variant=full_variant,
        precision=precision, recall=recall, ndcg=ndcg,
        epochs=epochs, emb_dim=emb_dim, n_layers=n_cf_layers,
        skip_groups=skip_str,
        notes=(
            f"T={n_periods} d_temp={d_temporal} d_sal={d_sal} "
            f"λ_sal={lambda_sal:.0e} kg_layers={n_kg_layers}"
        ),
    )

    if save:
        if _is_main_rank():
            spatial_suffix = "" if use_spatial_cbg else "_nospatial"
            file_tag   = f"skip_{skip_str.replace(',', '_')}" if skip_str else "all"
            file_tag   = f"{file_tag}_T{n_periods}{spatial_suffix}"
            model_path = MODEL_OUT.parent / f"kgat_sal_{file_tag}.pt"
            model_path.parent.mkdir(exist_ok=True)
            EMB_OUT.mkdir(exist_ok=True)

            # Save model (excluding period_adjs which are runtime tensors)
            torch.save(raw_model.state_dict(), model_path)

            user_dec = {v: k for k, v in user_enc.items()}
            item_dec = {v: k for k, v in item_enc.items()}
            torch.save({"embeddings": u_emb.cpu(), "id_map": user_dec},
                       EMB_OUT / f"kgat_sal_{file_tag}_user_embeddings.pt")
            torch.save({"embeddings": i_emb.cpu(), "id_map": item_dec},
                       EMB_OUT / f"kgat_sal_{file_tag}_item_embeddings.pt")
            log.info("Model → %s | Embeddings → %s", model_path, EMB_OUT)

            save_predictions(u_emb, i_emb, train_df, test_df, user_enc, item_enc, file_tag)

    return recall


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KGAT + Personalized SAL recommendation model")
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--emb-dim",      type=int,   default=2048)
    parser.add_argument("--kg-layers",    type=int,   default=2,
                        help="KG attentive propagation layers (default 2)")
    parser.add_argument("--cf-layers",    type=int,   default=4,
                        help="CF LightGCN propagation layers (default 4)")
    parser.add_argument("--n-periods",    type=int,   default=3,
                        help="Number of temporal periods T (default 3)")
    parser.add_argument("--d-temporal",   type=int,   default=256,
                        help="Temporal projection dim for period aggregation (default 256)")
    parser.add_argument("--d-sal",        type=int,   default=32,
                        help="Hidden dim for stability weighter MLP (default 32)")
    parser.add_argument("--lambda-sal",   type=float, default=1e-6,
                        help="SAL loss weight (default 1e-6)")
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--batch-size",   type=int,   default=2048)
    parser.add_argument("--sal-samples",  type=int,   default=4096,
                        help="Edges sampled per period per epoch for SAL (default 4096)")
    parser.add_argument("--edge-dropout", type=float, default=0.1)
    parser.add_argument("--kg-dropout",   type=float, default=0.1)
    parser.add_argument("--hard-neg-refresh", type=int, default=50)
    parser.add_argument("--llm-feat-mode", default="fast", choices=["none", "fast", "full"])
    parser.add_argument("--no-nlp-feat",  action="store_true")
    parser.add_argument("--eval-every",   type=int,   default=50)
    parser.add_argument("--no-save",      action="store_true")
    parser.add_argument("--prebuilt-features", action="store_true",
                        help="Load pre-joined feature matrices (run build_training_features.py first)")
    parser.add_argument("--skip-feature-groups", default="",
                        help="Comma-separated feature groups to drop "
                             "(user: base,extended,pref; item: base,nlp,extended,llm)")
    parser.add_argument("--no-spatial-cbg", action="store_true",
                        help="Disable CBG-CBG spatial KG triples")
    args, _ = parser.parse_known_args()
    init_distributed()

    skip_groups = [g.strip() for g in args.skip_feature_groups.split(",") if g.strip()] \
        if args.skip_feature_groups else None

    train(
        epochs               = args.epochs,
        emb_dim              = args.emb_dim,
        n_kg_layers          = args.kg_layers,
        n_cf_layers          = args.cf_layers,
        n_periods            = args.n_periods,
        d_temporal           = args.d_temporal,
        d_sal                = args.d_sal,
        lambda_sal           = args.lambda_sal,
        lr                   = args.lr,
        batch_size           = args.batch_size,
        sal_samples          = args.sal_samples,
        save                 = not args.no_save,
        edge_dropout         = args.edge_dropout,
        kg_dropout           = args.kg_dropout,
        hard_neg_refresh     = args.hard_neg_refresh,
        llm_feat_mode        = args.llm_feat_mode,
        use_nlp_feat         = not args.no_nlp_feat,
        eval_every           = args.eval_every,
        prebuilt_features    = args.prebuilt_features,
        skip_feature_groups  = skip_groups,
        use_spatial_cbg      = not args.no_spatial_cbg,
    )


if __name__ == "__main__":
    main()
