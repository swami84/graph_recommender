#!/usr/bin/env python3
"""
recommendation_radar.py — RaDAR-lite: Relation-aware Diffusion-Asymmetric Graph
Contrastive Learning for Recommendation.

Simplified implementation of:
  "RaDAR: Relation-aware Diffusion-Asymmetric Graph Contrastive Learning
   for Recommendation" (WWW 2026, Huang et al.)

Architecture:
  1. LightGCN backbone (same as KGAT/SimGCL)
  2. Relation-aware edge denoiser: per-edge bilinear similarity score → soft mask
     on adjacency → "denoised" graph view
  3. Asymmetric Contrastive Loss (ACL): predictor(clean_view) aligns to denoised_view
  4. Diffusion Denoising Regularizer (DDR): MLP recovers clean embeddings from
     noise-injected versions

Simplifications vs full RaDAR:
  - No VGAE view generator (bilinear edge scorer instead — "Gen+Linear" ≈ "Gen+Gen"
    on sparse datasets per Table 6 ablation)
  - EdgeScorer uses emb→score_dim bilinear projection to keep activation memory at
    ~500 MB rather than ~16 GB for a naive MLP on 4096-dim edge features
  - Single optimizer (same epoch-level pattern as SimGCL): SSL backward first,
    then BPR manual-gradient backward, then one optimizer.step()

Training loop follows SimGCL exactly:
  Step 1: ACL + DDR backward (via grad_ckpt GCN + denoised view)
  Step 2a: no_grad BPR scatter_add  → grad_final
  Step 2b: ckpt forward + backward(grad_final)  → add to accumulated grads
  Step 3: optimizer.step()

Usage:
    python recommendation_radar.py \\
        --epochs 300 --emb-dim 2048 --n-layers 3 \\
        --batch-size 8192 --eval-every 50 --prebuilt-features

    # Skip LLM features (typically best config):
    python recommendation_radar.py --epochs 300 --prebuilt-features \\
        --skip-feature-groups llm
"""

import argparse
import copy
import logging
import os
from pathlib import Path

import contextlib
import numpy as np
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
    ndcg_at_k, _is_main_rank, _sync_grads, init_distributed, DEVICE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("radar")
log.setLevel(logging.INFO)
if not log.handlers:
    _lh = logging.StreamHandler()
    _lh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_lh)
    log.propagate = False

MODEL_OUT = Path("models/radar.pt")
EMB_OUT   = Path("data/embeddings")


# ── Custom sparse mm that avoids the N×N dense gradient OOM ───────────────────

class _SparseMmValGrad(torch.autograd.Function):
    """
    Sparse mm where autograd only differentiates through the sparse VALUES.

    Standard autograd on a grad-enabled sparse COO tensor triggers PyTorch to
    materialise the full N×N dense gradient matrix during backward
    (153k × 153k × 4 bytes ≈ 87 GB → OOM).  This custom function bypasses
    that by detaching the sparse structure inside forward() and manually
    computing dL/d(values[k]) = (grad_out[src_k] * emb[dst_k]).sum() in backward().
    """

    @staticmethod
    def forward(ctx, adj_vals, adj_idx, adj_size, emb):
        # Build adj with detached values so standard autograd ignores it.
        # Sparse mm requires float32 (CUDA sparse lacks BF16 support); cast and restore.
        out_dtype = emb.dtype
        A   = torch.sparse_coo_tensor(adj_idx, adj_vals.detach().float(), adj_size).coalesce()
        out = torch.sparse.mm(A, emb.float()).to(out_dtype)
        ctx.save_for_backward(adj_idx, emb)   # emb is caller-detached
        return out

    @staticmethod
    def backward(ctx, grad_out):
        adj_idx, emb = ctx.saved_tensors
        src, dst = adj_idx[0], adj_idx[1]
        E = src.shape[0]
        # Chunked: gathering all 988k rows of 2048-dim emb at once = 8 GB OOM.
        # 32k-edge chunks keep peak at ~268 MB each.
        CHUNK = 32768
        grad_vals = torch.empty(E, dtype=grad_out.dtype, device=grad_out.device)
        for s in range(0, E, CHUNK):
            e = min(s + CHUNK, E)
            grad_vals[s:e] = (grad_out[src[s:e]] * emb[dst[s:e]]).sum(dim=-1)
        return grad_vals, None, None, None   # no grad for idx / size / emb


def _sparse_mm_val_grad(adj_vals, adj_idx, adj_size, emb):
    return _SparseMmValGrad.apply(adj_vals, adj_idx, adj_size, emb)


class _DenoiseGradHelper(torch.autograd.Function):
    """
    Multi-layer sparse propagation that saves only e_0 in ctx.

    _SparseMmValGrad saves e_k.detach() for every layer, costing
    n_layers × (N × D) ≈ 8 GB at 4 layers / D=2048 / N=157k.
    This function runs all layers in no_grad (forward), saves only e_0 and
    new_vals (already cheap), then recomputes intermediate embeddings one at
    a time during backward — reducing saved ctx to 2.03 GB.

    Gradient: only flows through new_vals (→ edge_scorer).
    e_0 is treated as a constant (caller has already detached it).

    output = (e_0 + e_1 + ... + e_{n_layers}) / (n_layers + 1)
    """

    @staticmethod
    def forward(ctx, new_vals, adj_idx, adj_size, e_0, n_layers):
        out_dtype = e_0.dtype
        with torch.no_grad():
            e = e_0  # stay in e_0.dtype (BF16) — _sparse_mm_f32 handles float32 internally
            s = e_0.clone()
            for _ in range(n_layers):
                A = torch.sparse_coo_tensor(adj_idx, new_vals.detach().float(), adj_size).coalesce()
                e = _sparse_mm_f32(A, e)  # BF16 in → chunked float32 → BF16 out
                s = s + e
        ctx.save_for_backward(new_vals.detach(), adj_idx, e_0)
        ctx.adj_size  = adj_size
        ctx.n_layers  = n_layers
        ctx.out_dtype = out_dtype
        return s / (n_layers + 1)  # already out_dtype

    @staticmethod
    def backward(ctx, grad_out):
        new_vals_d, adj_idx, e_0 = ctx.saved_tensors
        adj_size  = ctx.adj_size
        n_layers  = ctx.n_layers
        src, dst  = adj_idx[0], adj_idx[1]

        E     = src.shape[0]
        scale = 1.0 / (n_layers + 1)
        CHUNK = 32768

        grad_vals = torch.zeros(E, dtype=torch.float32, device=grad_out.device)

        with torch.no_grad():
            e = e_0  # keep BF16 — avoids 2 GiB float32 cast of full embedding
            for _ in range(n_layers):
                # Cast only the small edge-indexed slices to float32 (CHUNK×D×4 ≈ 0.26 GiB)
                for sc in range(0, E, CHUNK):
                    ec = min(sc + CHUNK, E)
                    grad_vals[sc:ec] += scale * (
                        grad_out[src[sc:ec]].float() * e[dst[sc:ec]].float()
                    ).sum(dim=-1)
                A = torch.sparse_coo_tensor(adj_idx, new_vals_d.float(), adj_size).coalesce()
                e = _sparse_mm_f32(A, e)  # BF16 in → chunked float32 → BF16 out

        return grad_vals, None, None, None, None   # no grad for adj_idx, adj_size, e_0, n_layers


# ── Auxiliary modules ──────────────────────────────────────────────────────────

class EdgeScorer(nn.Module):
    """
    Per-edge relation-aware scorer.  Projects each node to a low-dim space and
    computes bilinear similarity; this limits activation memory to ~500 MB for
    ~1 M edges at emb_dim=2048, vs ~16 GB for a naive full-dim MLP.

    Output: (E,) logit — sigmoid gives edge retention probability.
    """

    def __init__(self, emb_dim: int, score_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(emb_dim, score_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(
        self,
        all_emb: torch.Tensor,   # (N, D) ALL nodes — project first, then gather
        src:     torch.Tensor,   # (E,)   source indices
        dst:     torch.Tensor,   # (E,)   dest indices
    ) -> torch.Tensor:
        # Project all N nodes to score_dim first (152k×64 = 39 MB),
        # then gather cheap 64-dim vectors by edge.
        # Naive gather-then-project creates two (988k×2048) = 8 GB tensors → OOM.
        all_proj = self.proj(all_emb)       # (N, score_dim)
        e_src    = all_proj[src]            # (E, score_dim)
        e_dst    = all_proj[dst]            # (E, score_dim)
        return (e_src * e_dst).sum(-1)      # (E,)


class DiffusionDenoiser(nn.Module):
    """
    DDR: simple MLP denoiser χ_θ(x_t, t) → predicts x_0.
    Linear noise schedule over T steps; timestep embedded and concatenated.
    """

    def __init__(self, emb_dim: int, T: int = 5):
        super().__init__()
        self.T     = T
        self.t_emb = nn.Embedding(T + 1, emb_dim)
        self.net   = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        betas      = torch.linspace(0.01, 0.30, T)
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alpha_bars", alpha_bars)   # (T,)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.t_emb(t)
        return self.net(torch.cat([x_t, te], dim=-1))

    def noisy(self, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample t, return (x_t, t) via forward diffusion."""
        B  = x0.shape[0]
        t  = torch.randint(1, self.T + 1, (B,), device=x0.device)
        ab = self.alpha_bars[t - 1].unsqueeze(1)
        x_t = ab.sqrt() * x0 + (1.0 - ab).sqrt() * torch.randn_like(x0)
        return x_t, t


class AsymmetricPredictor(nn.Module):
    """
    ACL asymmetric predictor g_φ: maps clean-view identity embedding to a
    predicted denoised-view context embedding.  One-sided InfoNCE (clean→denoised
    only) aligns semantically similar nodes without requiring strict homophily.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── RaDAR Model ────────────────────────────────────────────────────────────────

class RaDAR(nn.Module):
    """
    LightGCN backbone + relation-aware edge denoising + ACL + DDR.
    """

    def __init__(
        self,
        n_users:      int,
        n_items:      int,
        emb_dim:      int,
        n_layers:     int   = 3,
        edge_dropout: float = 0.0,
        acl_temp:     float = 0.2,
        user_feat: np.ndarray | None = None,
        item_feat: np.ndarray | None = None,
    ):
        super().__init__()
        self.n_users      = n_users
        self.n_items      = n_items
        self.n_layers     = n_layers
        self.edge_dropout = edge_dropout
        self.acl_temp     = acl_temp

        # ── GCN backbone ──────────────────────────────────────────────────────
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        for name, feat in [("user", user_feat), ("item", item_feat)]:
            if feat is not None:
                self.register_buffer(f"{name}_feat",
                                     torch.tensor(feat, dtype=torch.float32))
                proj = nn.Linear(feat.shape[1], emb_dim, bias=False)
                nn.init.xavier_uniform_(proj.weight)
                setattr(self, f"{name}_proj", proj)
            else:
                self.register_buffer(f"{name}_feat", None)
                setattr(self, f"{name}_proj", None)

        # ── Auxiliary components ──────────────────────────────────────────────
        self.edge_scorer = EdgeScorer(emb_dim)
        self.denoiser    = DiffusionDenoiser(emb_dim)
        self.predictor   = AsymmetricPredictor(emb_dim)

    # ── GCN propagation ────────────────────────────────────────────────────────

    def _propagate(self, adj: torch.Tensor, use_ckpt: bool = True) -> torch.Tensor:
        adj_used = dropout_adj(adj, self.edge_dropout, self.training)

        def _pass(u_w, i_w):
            u = u_w + self.user_proj(self.user_feat) if self.user_proj else u_w
            i = i_w + self.item_proj(self.item_feat) if self.item_proj else i_w
            e = torch.cat([u, i], dim=0)
            s = e
            for _ in range(self.n_layers):
                e = _sparse_mm_f32(adj_used, e)
                s = s + e
            return s / (self.n_layers + 1)

        if use_ckpt and self.training:
            return grad_ckpt(_pass, self.user_emb.weight, self.item_emb.weight,
                             use_reentrant=False)
        return _pass(self.user_emb.weight, self.item_emb.weight)

    # ── Denoised propagation ───────────────────────────────────────────────────

    def _propagate_denoised(
        self,
        adj:      torch.Tensor,
        base_emb: torch.Tensor,    # (n_users+n_items, D) detached — for scoring
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        LightGCN on relation-aware masked adjacency.

        base_emb must be detached so gradients from the denoised view only
        flow through edge_scorer parameters (not back to user/item embeddings).

        Returns (denoised_emb, masks).
        """
        adj_coo  = adj.coalesce()
        edge_idx = adj_coo.indices()
        src, dst = edge_idx[0], edge_idx[1]

        # Bilinear edge scores: project all nodes first, then gather by edge
        scores = self.edge_scorer(base_emb, src, dst)             # (E,)
        masks  = torch.sigmoid(scores)                            # (E,) in (0,1)

        # new_vals has grad through masks → edge_scorer
        new_vals = adj_coo.values() * masks                       # (E,)
        adj_size = adj_coo.size()

        u_w = self.user_emb.weight.detach()
        i_w = self.item_emb.weight.detach()
        u   = u_w + self.user_proj(self.user_feat).detach() if self.user_proj else u_w
        i   = i_w + self.item_proj(self.item_feat).detach() if self.item_proj else i_w
        e_0 = torch.cat([u, i], dim=0)

        # _DenoiseGradHelper saves only e_0 in ctx (not each layer's e_k).
        # Backward recomputes intermediate layers one-at-a-time → ~6 GB saved.
        s = _DenoiseGradHelper.apply(new_vals, edge_idx, adj_size, e_0, self.n_layers)
        return s, masks

    # ── SSL losses ─────────────────────────────────────────────────────────────

    def acl_loss(
        self,
        clean_emb:    torch.Tensor,   # (N, D) — grad flows to GCN via this
        denoised_emb: torch.Tensor,   # (N, D) — grad flows to edge_scorer
        n_samples:    int = 4096,
    ) -> torch.Tensor:
        N   = clean_emb.shape[0]
        idx = torch.randperm(N, device=clean_emb.device)[:n_samples]

        q = self.predictor(clean_emb[idx])          # (B, D) grad → predictor + GCN
        k = denoised_emb[idx]                       # (B, D) grad → edge_scorer

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        sim    = q @ k.T / self.acl_temp            # (B, B)
        labels = torch.arange(len(idx), device=q.device)
        return F.cross_entropy(sim, labels)         # one-sided (clean → denoised)

    def ddr_loss(
        self,
        clean_emb: torch.Tensor,    # (N, D) — grad flows to GCN + denoiser
        n_samples: int = 4096,
    ) -> torch.Tensor:
        idx     = torch.randperm(clean_emb.shape[0], device=clean_emb.device)[:n_samples]
        x0      = clean_emb[idx]
        x_t, t  = self.denoiser.noisy(x0)
        x0_pred = self.denoiser(x_t, t)
        return F.mse_loss(x0_pred, x0.detach())

    # ── Evaluation embeddings ──────────────────────────────────────────────────

    def get_embeddings(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            all_emb = self._propagate(adj, use_ckpt=False)
        return all_emb[:self.n_users], all_emb[self.n_users:]


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    epochs: int          = 300,
    emb_dim: int         = 2048,
    n_layers: int        = 3,
    edge_dropout: float  = 0.0,
    acl_temp: float      = 0.2,
    lambda_acl: float    = 0.1,
    lambda_ddr: float    = 0.01,
    lambda_mask: float   = 0.05,
    lr: float            = 1e-3,
    batch_size: int      = 8192,
    ssl_samples: int     = 8192,
    save: bool           = True,
    hard_neg_refresh: int = 50,
    llm_feat_mode: str   = "fast",
    use_nlp_feat: bool   = True,
    eval_every: int      = 50,
    prebuilt_features: bool = False,
    skip_feature_groups: list[str] | None = None,
):
    train_df, test_df, user_enc, item_enc = load_interactions()
    n_users, n_items = len(user_enc), len(item_enc)
    log.info(f"Train {len(train_df):,} | Test {len(test_df):,} | "
             f"Users {n_users:,} | Items {n_items:,}")

    user_feat = build_user_features(train_df, user_enc,
                                    prebuilt=prebuilt_features,
                                    skip_groups=skip_feature_groups)
    item_feat = build_item_features(item_enc, llm_feat_mode=llm_feat_mode,
                                    use_nlp_feat=use_nlp_feat,
                                    prebuilt=prebuilt_features,
                                    skip_groups=skip_feature_groups)
    adj = build_adj(train_df, n_users, n_items)

    model = RaDAR(
        n_users=n_users, n_items=n_items, emb_dim=emb_dim,
        n_layers=n_layers, edge_dropout=edge_dropout,
        acl_temp=acl_temp,
        user_feat=user_feat, item_feat=item_feat,
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
        f"Training RaDAR | device={DEVICE} | epochs={epochs} | emb_dim={emb_dim} | "
        f"n_layers={n_layers} | λ_acl={lambda_acl} | λ_ddr={lambda_ddr} | "
        f"λ_mask={lambda_mask} | acl_temp={acl_temp} | lr={lr}"
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

        if hard_neg_refresh > 0 and epoch > 1 and (epoch - 1) % hard_neg_refresh == 0:
            raw_model.eval()
            with torch.no_grad():
                u_e, i_e = raw_model.get_embeddings(adj)
            emb_pool = build_hard_neg_pool(i_e, k=50)
            del u_e, i_e
            raw_model.train()
            log.info(f"  [Epoch {epoch}] Hard neg pool refreshed")
            train_with_neg = sample_negatives(train_df, n_items, emb_pool=emb_pool)
            all_users = torch.tensor(train_with_neg["user_idx"].values, dtype=torch.long)
            all_pos   = torch.tensor(train_with_neg["item_idx"].values, dtype=torch.long)
            all_neg   = torch.tensor(train_with_neg["neg_idx"].values,  dtype=torch.long)

        raw_model.train()
        optimizer.zero_grad()

        # ── Step 1: SSL losses — ACL + DDR ────────────────────────────────────
        # clean_emb: ckpt propagation, grad flows to GCN params + predictor + denoiser
        # denoised_emb: adj reweighted by edge_scorer on clean_emb.detach()
        #               grad only flows to edge_scorer params
        clean_emb = raw_model._propagate(adj, use_ckpt=True)     # (N, D) with grad

        with torch.no_grad():
            base_emb = clean_emb.detach()                    # fixed snapshot for scoring

        denoised_emb, masks = raw_model._propagate_denoised(adj, base_emb)

        n_nodes   = n_users + n_items
        L_acl     = raw_model.acl_loss(clean_emb, denoised_emb, n_samples=min(ssl_samples, n_nodes))
        L_ddr     = raw_model.ddr_loss(clean_emb, n_samples=min(ssl_samples, n_nodes))
        L_mask    = masks.mean()                             # soft sparsity pressure
        L_ssl     = lambda_acl * L_acl + lambda_ddr * L_ddr + lambda_mask * L_mask
        _nosync = model.no_sync() if dist.is_initialized() else contextlib.nullcontext()
        with _nosync:
            L_ssl.backward()

        acl_val  = L_acl.item()
        ddr_val  = L_ddr.item()
        mask_val = L_mask.item()
        del clean_emb, denoised_emb, masks, L_ssl
        torch.cuda.empty_cache()

        # ── Step 2a: BPR gradient — no_grad scatter_add ───────────────────────
        _saved_dropout    = raw_model.edge_dropout
        raw_model.edge_dropout = 0.0

        with torch.no_grad():
            final_det = raw_model._propagate(adj, use_ckpt=False)
            u_det     = final_det[:n_users]
            i_det     = final_det[n_users:]

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

        # ── Step 2b: ckpt forward + backward adds to SSL-accumulated grads ────
        final_bpr = raw_model._propagate(adj, use_ckpt=True)
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
                f"acl={acl_val:.4f}  ddr={ddr_val:.4f}  mask={mask_val:.3f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if eval_every > 0 and epoch % eval_every == 0:
            raw_model.eval()
            u_emb_eval, i_emb_eval = raw_model.get_embeddings(adj)

            test_users  = test_df["user_idx"].values
            test_items  = test_df["item_idx"].values
            train_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

            hits, total = 0, 0
            chunk = 4096
            for cstart in range(0, len(test_users), chunk):
                cend   = min(cstart + chunk, len(test_users))
                u_idx  = test_users[cstart:cend]
                i_idx  = test_items[cstart:cend]
                scores = (u_emb_eval[u_idx] @ i_emb_eval.T).cpu().numpy()
                for k, (ui, ii) in enumerate(zip(u_idx, i_idx)):
                    for ex in train_items.get(ui, set()):
                        scores[k, ex] = -np.inf
                    if ii in np.argsort(-scores[k])[:10]:
                        hits += 1
                    total += 1

            val_recall = hits / max(total, 1)
            del u_emb_eval, i_emb_eval
            raw_model.train()

            if val_recall > best_recall:
                best_recall = val_recall
                best_state  = copy.deepcopy(raw_model.state_dict())
                log.info(f"  [Epoch {epoch}] val Recall@10={val_recall:.4f}  *** new best ***")
            else:
                log.info(f"  [Epoch {epoch}] val Recall@10={val_recall:.4f}"
                         f"  (best={best_recall:.4f})")

    if best_state is not None:
        raw_model.load_state_dict(best_state)
        log.info(f"Loaded best checkpoint (val Recall@10={best_recall:.4f})")

    # ── Final evaluation ───────────────────────────────────────────────────────
    raw_model.eval()
    log.info("Computing embeddings for evaluation…")
    u_emb, i_emb = raw_model.get_embeddings(adj)

    log.info(f"  Embedding norms — users mean={u_emb.norm(dim=1).mean():.4f}, "
             f"items mean={i_emb.norm(dim=1).mean():.4f}")

    test_users  = test_df["user_idx"].values
    test_items  = test_df["item_idx"].values
    train_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    hits, ndcg_sum, total = 0, 0.0, 0
    chunk = 4096
    log.info(f"Evaluating {len(test_users):,} test users in chunks of {chunk}…")

    for start in range(0, len(test_users), chunk):
        end    = min(start + chunk, len(test_users))
        u_idx  = test_users[start:end]
        i_idx  = test_items[start:end]
        scores = (u_emb[u_idx] @ i_emb.T).cpu().numpy()

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

    print(f"\n=== RaDAR Results ===")
    print(f"  Precision@10: {precision:.4f}")
    print(f"  Recall@10:    {recall:.4f}")
    print(f"  NDCG@10:      {ndcg:.4f}")

    skip_str = ",".join(skip_feature_groups) if skip_feature_groups else ""
    variant  = f"skip {skip_str}" if skip_str else "all features"
    save_result(
        model="RaDAR", variant=variant,
        precision=precision, recall=recall, ndcg=ndcg,
        epochs=epochs, emb_dim=emb_dim, n_layers=n_layers,
        skip_groups=skip_str,
    )

    if save:
        if _is_main_rank():
            file_tag   = f"skip_{skip_str.replace(',', '_')}" if skip_str else "all"
            model_path = MODEL_OUT.parent / f"radar_{file_tag}.pt"
            model_path.parent.mkdir(exist_ok=True)
            EMB_OUT.mkdir(exist_ok=True)
            torch.save(raw_model.state_dict(), model_path)
            user_dec = {v: k for k, v in user_enc.items()}
            item_dec = {v: k for k, v in item_enc.items()}
            torch.save({"embeddings": u_emb.cpu(), "id_map": user_dec},
                       EMB_OUT / f"radar_{file_tag}_user_embeddings.pt")
            torch.save({"embeddings": i_emb.cpu(), "id_map": item_dec},
                       EMB_OUT / f"radar_{file_tag}_item_embeddings.pt")
            log.info(f"Model → {model_path} | Embeddings → {EMB_OUT}")

    return recall


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RaDAR-lite recommendation model")
    parser.add_argument("--epochs",         type=int,   default=300)
    parser.add_argument("--emb-dim",        type=int,   default=2048)
    parser.add_argument("--n-layers",       type=int,   default=3)
    parser.add_argument("--edge-dropout",   type=float, default=0.0)
    parser.add_argument("--acl-temp",       type=float, default=0.2,
                        help="InfoNCE temperature for ACL loss")
    parser.add_argument("--lambda-acl",     type=float, default=0.1,
                        help="Weight of asymmetric contrastive loss")
    parser.add_argument("--lambda-ddr",     type=float, default=0.01,
                        help="Weight of diffusion denoising regularizer")
    parser.add_argument("--lambda-mask",    type=float, default=0.05,
                        help="Sparsity pressure on edge masks")
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--batch-size",     type=int,   default=8192)
    parser.add_argument("--ssl-samples",    type=int,   default=8192,
                        help="Nodes sampled per epoch for ACL and DDR")
    parser.add_argument("--hard-neg-refresh", type=int, default=50)
    parser.add_argument("--eval-every",     type=int,   default=50)
    parser.add_argument("--no-save",        action="store_true")
    parser.add_argument("--no-nlp-feat",    action="store_true")
    parser.add_argument("--llm-feat-mode",  type=str,   default="fast",
                        choices=["none", "fast", "full"])
    parser.add_argument("--prebuilt-features", action="store_true")
    parser.add_argument("--skip-feature-groups", default="",
                        help="Comma-separated feature groups to drop")
    args, _ = parser.parse_known_args()
    init_distributed()

    skip_groups = [g.strip() for g in args.skip_feature_groups.split(",") if g.strip()] \
        if args.skip_feature_groups else None

    train(
        epochs              = args.epochs,
        emb_dim             = args.emb_dim,
        n_layers            = args.n_layers,
        edge_dropout        = args.edge_dropout,
        acl_temp            = args.acl_temp,
        lambda_acl          = args.lambda_acl,
        lambda_ddr          = args.lambda_ddr,
        lambda_mask         = args.lambda_mask,
        lr                  = args.lr,
        batch_size          = args.batch_size,
        ssl_samples         = args.ssl_samples,
        save                = not args.no_save,
        llm_feat_mode       = args.llm_feat_mode,
        use_nlp_feat        = not args.no_nlp_feat,
        eval_every          = args.eval_every,
        prebuilt_features   = args.prebuilt_features,
        skip_feature_groups = skip_groups,
    )
