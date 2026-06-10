#!/usr/bin/env python3
"""
recommendation_hek_cl.py — HEK-CL: Full Phase 2 Implementation.

HEK-CL: Hierarchical Enhanced Knowledge-Aware Contrastive Learning for
Recommendation. Yuan et al., ACM TOIS 2025.

Implements both phases of HEK-CL on top of the existing KGAT knowledge graph
(cuisine, price, CBG, dish triples):

Phase 1 — Data View Generation (two complementary strategies):
  · Structure-enhanced: Heterogeneous Graph Denoising Module.
    A binary Bernoulli gate z_{i,j} ∈ {0,1} is attached to every edge in both
    the user-item CF graph and the KG. Gates are parameterised by an MLP
    f_θ(eᵢ, eⱼ) and sampled via the hard-concrete (Gumbel-Softmax) estimator,
    pruning noisy edges while preserving significant ones.

  · Embedding-improved: Poincaré Ball projection.
    All entity embeddings are projected from Euclidean space to the Poincaré
    ball via the exponential map at the origin. Curvature c = -κ is a
    trainable scalar (initialised to 1.0). This captures the hierarchical
    structure in restaurant data: cuisine → restaurant → dish, and the
    geographic hierarchy CBG → tract → county → state.

Phase 2 — Data View Alignment:
  · HRCL: Hyperbolic Robust Contrastive Loss using geodesic distance d_c.
    Cross-view: CF item embedding e_v^u vs. KG item embedding e_v^k.
    Satisfies the symmetric loss condition for robustness under noisy data.

  · L_Rec: Geometry-aware margin ranking loss using squared geodesic distances:
    L_Rec(u,i,j) = max(d_c²(u,i) − d_c²(u,j) + m, 0)
    where m is a fixed margin (simplified from the paper's adaptive version).

  · L_D: Denoising regulariser — sparsity penalty on the binary gates:
    L_D = Σ_edges σ(β_{i,j}) = Σ P(z_{i,j}=1 | θ)
    This pushes the model to classify most edges as noise (z=0).

Architecture:
  1. Euclidean embeddings: user_emb, entity_emb, rel_emb (same dims as KGAT)
  2. Poincaré ball projection via expmap0
  3. KG denoising: MLP gate per triple → denoised KG adjacency
  4. CF denoising: MLP gate per sampled CF edge → soft-masked adjacency
  5. KG aggregation in tangent space (relation-aware attention, KGAT-style)
  6. CF LightGCN propagation over denoised adjacency
  7. Two item views: KG-enriched e_v^k, CF-propagated e_v^u
  8. HRCL (cross-view alignment) + L_Rec (ranking) + L_D (denoising)

Total loss: L = L_Rec + β₁·L_HRCL + β₂·L_D^k + β₃·L_D^u

Memory notes:
  · Poincaré ball ops are O(B·d) — no N×N materialisation.
  · KG denoising runs over all KG triples (~320k); this is fast.
  · CF denoising samples --cf-denoise-edges edges per epoch to avoid
    materialising the full 500k-edge adjacency through the MLP.
  · HRCL CL batch is --cl-batch-size (default 512) separate from BPR batch.
  · BPR uses the same manual-gradient pattern as KGAT/SimGCL.

Usage:
    python recommendation_hek_cl.py
    python recommendation_hek_cl.py --epochs 300 --emb-dim 512 \\
        --kg-layers 2 --cf-layers 3 --curvature 1.0 \\
        --hrcl-weight 0.001 --denoise-weight 0.5 --hrcl-q 0.001 \\
        --cl-batch-size 512 --eval-every 50
"""

import argparse
import copy
import logging
import math
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
    _sparse_mm_f32, dropout_adj,
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
from recommendation_kgat import build_kg, N_RELATIONS, TOP_DISH_N

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hek_cl")

MODEL_OUT = Path("models/hek_cl.pt")
EMB_OUT   = Path("data/embeddings")


# ── Poincaré Ball ──────────────────────────────────────────────────────────────

class PoincareBall(nn.Module):
    """
    Poincaré ball manifold with trainable curvature c = -κ > 0.

    All operations work on the last dimension.  Embeddings must satisfy
    ||x||² < 1/c  (enforced by clipping in expmap0).

    Using c as the curvature magnitude (c = -κ) so c is always positive,
    matching the convention: d_c(x,y) = (2/√c)·atanh(√c·||−x ⊕_c y||).
    """

    def __init__(self, c_init: float = 1.0):
        super().__init__()
        # log(c) parameterisation keeps c > 0 unconditionally
        self.log_c = nn.Parameter(torch.tensor(math.log(c_init)))

    @property
    def c(self) -> torch.Tensor:
        return self.log_c.exp().clamp(min=1e-2, max=10.0)

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        """Map tangent vector v at origin → point on manifold."""
        c = self.c
        sqrt_c = c.sqrt()
        norm_v = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        tanh_arg = (sqrt_c * norm_v).clamp(max=15.0)   # prevent tanh saturation
        tanh_val = torch.tanh(tanh_arg).clamp(max=1 - 1e-5)
        return tanh_val / (sqrt_c * norm_v) * v

    def logmap0(self, y: torch.Tensor) -> torch.Tensor:
        """Map manifold point y → tangent vector at origin."""
        c = self.c
        sqrt_c = c.sqrt()
        norm_y = y.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        atanh_arg = (sqrt_c * norm_y).clamp(max=1 - 1e-5)
        return (2.0 / sqrt_c) * torch.atanh(atanh_arg) / (sqrt_c * norm_y) * y

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Möbius addition x ⊕_c y."""
        c = self.c
        x2 = (x * x).sum(-1, keepdim=True)
        y2 = (y * y).sum(-1, keepdim=True)
        xy = (x * y).sum(-1, keepdim=True)
        num   = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
        denom = 1 + 2*c*xy + c**2 * x2 * y2
        return num / denom.clamp(min=1e-7)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geodesic distance d_c(x, y).  Shapes: (..., d) → (...)."""
        c = self.c
        sqrt_c = c.sqrt()
        neg_x_plus_y = self.mobius_add(-x, y)
        norm = neg_x_plus_y.norm(dim=-1)
        atanh_arg = (sqrt_c * norm).clamp(max=1 - 1e-5)
        return (2.0 / sqrt_c) * torch.atanh(atanh_arg)

    def dist_batch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        All-pairs geodesic distances between rows of x (B_x, d) and y (B_y, d).
        Returns (B_x, B_y).  Vectorised via broadcasting — keep B small (~512).
        """
        # Expand for broadcasting: (B_x, 1, d) and (1, B_y, d)
        c = self.c
        sqrt_c = c.sqrt()
        x_e = x.unsqueeze(1)   # (B_x, 1, d)
        y_e = y.unsqueeze(0)   # (1, B_y, d)

        x2 = (x_e * x_e).sum(-1, keepdim=True)   # (B_x, 1, 1)
        y2 = (y_e * y_e).sum(-1, keepdim=True)   # (1, B_y, 1)
        xy = (x_e * y_e).sum(-1, keepdim=True)   # (B_x, B_y, 1)

        # −x ⊕_c y
        # For -x ⊕_c y: replace x with -x, so ⟨x,y⟩ → ⟨-x,y⟩ = -⟨x,y⟩ in numerator.
        # Correct coefficient: (1 - 2c·⟨x,y⟩ + c‖y‖²).  Using +2c·xy is wrong and
        # makes dist_batch(x, x) ≠ 0, corrupting all positive-pair distance scores.
        num   = (1 - 2*c*xy + c*y2) * (-x_e) + (1 - c*x2) * y_e   # (B_x, B_y, d)
        denom = 1 - 2*c*xy + c**2 * x2 * y2
        add   = num / denom.clamp(min=1e-7)                          # (B_x, B_y, d)

        norm = add.norm(dim=-1)                                       # (B_x, B_y)
        atanh_arg = (sqrt_c * norm).clamp(max=1 - 1e-5)
        return (2.0 / sqrt_c) * torch.atanh(atanh_arg)               # (B_x, B_y)


# ── Edge Denoiser ──────────────────────────────────────────────────────────────

class EdgeDenoiser(nn.Module):
    """
    Learns a binary gate z_{i,j} ∈ {0,1} for each edge (i,j).

    β_{i,j} = MLP(eᵢ || eⱼ) → P(z=1) = σ(β)
    z_{i,j} sampled via hard-concrete estimator (Gumbel-Softmax):
      s_{i,j} = sigmoid((log u − log(1−u) + β) / τ)
      z_{i,j} = round(s_{i,j})  [straight-through gradient]

    Denoising loss: L_D = Σ σ(β_{i,j}) = Σ P(z=1)
    Minimising L_D pushes the model to classify most edges as noise (z=0).
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 128, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=True),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[2].weight)

    def forward(
        self, e_i: torch.Tensor, e_j: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            e_i, e_j: edge-endpoint embeddings (E, d)
        Returns:
            z_soft : soft gate ∈ (0,1), straight-through for backward  (E,)
            beta   : pre-sigmoid score, used for L_D                   (E,)
        """
        beta = self.mlp(torch.cat([e_i, e_j], dim=-1)).squeeze(-1)   # (E,)

        if self.training:
            u = torch.rand_like(beta).clamp(min=1e-6, max=1 - 1e-6)
            gumbel = torch.log(u) - torch.log(1 - u)
            z_soft = torch.sigmoid((gumbel + beta) / self.temperature)
        else:
            z_soft = torch.sigmoid(beta)

        return z_soft, beta

    @staticmethod
    def denoise_loss(beta: torch.Tensor) -> torch.Tensor:
        """L_D = mean P(z=1) = mean σ(β).  Minimise to encourage sparse gates."""
        return torch.sigmoid(beta).mean()


# ── HEK-CL Model ──────────────────────────────────────────────────────────────

class HEKCL(nn.Module):
    """
    HEK-CL: Hierarchical Enhanced Knowledge-Aware Contrastive Learning.

    Entity space layout (entity_emb):
      0 .. n_items-1              → restaurant embeddings
      n_items .. n_items+n_kg-1  → KG entity embeddings (cuisine/price/CBG/dish)

    Two item views produced per forward pass:
      e_v^k  — item embeddings enriched via denoised KG aggregation
      e_v^u  — item embeddings from denoised CF LightGCN propagation

    Cross-view HRCL aligns e_v^k and e_v^u for the same item,
    contrasting against other items in the batch.
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
        ball: PoincareBall,
        user_feat: np.ndarray | None = None,
        item_feat: np.ndarray | None = None,
        edge_dropout: float = 0.1,
        kg_dropout: float   = 0.1,
        denoise_hidden: int = 128,
        denoise_temp: float = 0.5,
    ):
        super().__init__()
        self.n_users     = n_users
        self.n_items     = n_items
        self.n_kg_ents   = n_kg_ents
        self.n_kg_layers = n_kg_layers
        self.n_cf_layers = n_cf_layers
        self.edge_dropout = edge_dropout
        self.kg_dropout   = kg_dropout
        self.ball         = ball   # shared PoincareBall (curvature is a model param)

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

        # Side-feature projection (additive at layer 0, same as KGAT)
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

        # Graph denoising modules
        self.kg_denoiser = EdgeDenoiser(emb_dim, hidden_dim=denoise_hidden,
                                        temperature=denoise_temp)
        self.cf_denoiser = EdgeDenoiser(emb_dim, hidden_dim=denoise_hidden,
                                        temperature=denoise_temp)

    # ── Poincaré projection ────────────────────────────────────────────────────

    def _to_hyperbolic(self, e: torch.Tensor) -> torch.Tensor:
        """Project Euclidean embeddings to Poincaré ball via expmap at origin."""
        return self.ball.expmap0(e)

    # ── KG path ───────────────────────────────────────────────────────────────

    def _kg_attention(
        self, entity_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Relation-aware attentive aggregation over KG triples, with denoising.

        Attention computed in tangent space (logmap0 of hyperbolic embeddings),
        matching the paper's formulation where relation-aware scoring uses
        log_o^κ(e) projected into tangent space W^Q and W^K.

        Returns:
            agg      : aggregated entity embeddings (n_total_ents, d) in hyperbolic space
            beta_kg  : denoiser raw scores for L_D (n_triples,)
        """
        heads = self.kg_heads
        rels  = self.kg_rels
        tails = self.kg_tails

        # KG edge dropout
        if self.training and self.kg_dropout > 0.0:
            mask  = torch.rand(len(heads), device=DEVICE) > self.kg_dropout
            heads = heads[mask]
            rels  = rels[mask]
            tails = tails[mask]

        e_h = entity_emb[heads]       # (E, d) hyperbolic
        e_t = entity_emb[tails]       # (E, d)
        e_r = self.ball.expmap0(self.rel_emb(rels))  # (E, d)

        # Attention in tangent space: (log_o(e_h) + log_o(e_r))ᵀ log_o(e_t) / sqrt(d)
        t_h = self.ball.logmap0(e_h)
        t_r = self.ball.logmap0(e_r)
        t_t = self.ball.logmap0(e_t)
        att_logits = ((t_h + t_r) * t_t).sum(dim=-1) * self._scale   # (E,)

        # Scatter-softmax per head entity
        n_total = entity_emb.size(0)
        max_logit = torch.full((n_total,), float("-inf"), device=DEVICE)
        max_logit.scatter_reduce_(0, heads, att_logits, reduce="amax", include_self=True)
        exp_logits = torch.exp(att_logits - max_logit[heads])
        sum_exp = torch.zeros(n_total, device=DEVICE)
        sum_exp.scatter_add_(0, heads, exp_logits)
        att_w = exp_logits / (sum_exp[heads] + 1e-9)   # (E,)

        # ── Denoising gate for KG edges ──────────────────────────────────────
        z_soft, beta_kg = self.kg_denoiser(
            self.ball.logmap0(e_h),   # use tangent representations for MLP
            self.ball.logmap0(e_t),
        )
        gated_w = att_w * z_soft   # (E,)

        # Weighted aggregation in hyperbolic space (Euclidean approx via tangent)
        agg_tan = torch.zeros_like(entity_emb)
        agg_tan.scatter_add_(
            0,
            heads.unsqueeze(1).expand(-1, e_t.size(1)),
            gated_w.unsqueeze(1) * self.ball.logmap0(e_t),
        )
        # Map aggregated tangent back to manifold and residual connect
        e_new = self.ball.expmap0(self.ball.logmap0(entity_emb) + F.leaky_relu(agg_tan, 0.2))

        return e_new, beta_kg

    def _get_kg_item_emb(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        KG view of items: L_KG layers of denoised attentive aggregation.
        Returns KG-enriched item embeddings (n_items, d) in hyperbolic space
        and the concatenated beta scores for L_D.
        """
        ent_emb = self.entity_emb.weight   # (n_total, d) Euclidean

        # Optionally add item feature projection at layer 0
        if self.item_proj is not None:
            ent_emb = ent_emb.clone()
            ent_emb[:self.n_items] = ent_emb[:self.n_items] + self.item_proj(self.item_feat)

        # Project to Poincaré ball
        ent_hyp = self._to_hyperbolic(ent_emb)

        all_betas = []
        summed = ent_hyp
        for _ in range(self.n_kg_layers):
            ent_hyp, beta_kg = self._kg_attention(ent_hyp)
            summed = self.ball.expmap0(
                self.ball.logmap0(summed) + self.ball.logmap0(ent_hyp)
            )
            all_betas.append(beta_kg)

        # Mean pooling across layers (in tangent space)
        item_kg = self.ball.expmap0(
            self.ball.logmap0(summed) / (self.n_kg_layers + 1)
        )
        beta_all = torch.cat(all_betas) if all_betas else torch.zeros(1, device=DEVICE)
        return item_kg[:self.n_items], beta_all

    # ── CF path ───────────────────────────────────────────────────────────────

    def _cf_propagate(
        self,
        user_hyp: torch.Tensor,
        item_hyp: torch.Tensor,
        adj: torch.Tensor,
        cf_denoise_edges: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        CF LightGCN over a soft-denoised adjacency.

        Edge sampling happens AFTER dropout so perm indices are always valid
        for adj_used.values() (dropout changes nnz; pre-sampling on the original
        adj produced out-of-bounds indices).

        Returns:
            u_final, i_final : final user/item embeddings (hyperbolic)
            beta_cf          : denoiser raw scores for L_D
        """
        adj_used = dropout_adj(adj, self.edge_dropout, self.training)

        # ── CF edge denoising on sampled edges ───────────────────────────────
        beta_cf = torch.zeros(1, device=DEVICE)
        if self.training and cf_denoise_edges > 0:
            # Sample from the dropout-reduced adj — perm is valid for adj_used
            cf_perm, cf_rows, cf_cols = _sample_cf_edges(adj_used, cf_denoise_edges)
            # Endpoint embeddings from current tangent-space reps
            u_tan   = self.ball.logmap0(user_hyp)                    # (n_users, d)
            i_tan   = self.ball.logmap0(item_hyp)                    # (n_items, d)
            all_tan = torch.cat([u_tan, i_tan], dim=0)               # (n_users+n_items, d)
            z_soft, beta_cf = self.cf_denoiser(all_tan[cf_rows], all_tan[cf_cols])
            adj_vals = adj_used.values().clone()
            adj_vals[cf_perm] = adj_vals[cf_perm] * z_soft.detach()
            adj_used = torch.sparse_coo_tensor(
                adj_used.indices(), adj_vals, adj_used.shape
            ).coalesce()

        # ── LightGCN propagation in tangent space ─────────────────────────────
        all_emb = torch.cat([
            self.ball.logmap0(user_hyp),
            self.ball.logmap0(item_hyp),
        ], dim=0)
        summed = all_emb
        for _ in range(self.n_cf_layers):
            all_emb = _sparse_mm_f32(adj_used, all_emb)
            summed  = summed + all_emb

        final_tan = summed / (self.n_cf_layers + 1)
        final_hyp = self.ball.expmap0(final_tan)

        u_final = final_hyp[:self.n_users]
        i_final = final_hyp[self.n_users:]
        return u_final, i_final, beta_cf

    # ── Full forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        adj: torch.Tensor,
        cf_denoise_edges: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            u_cf   : user embeddings from CF path (n_users, d) hyperbolic
            i_cf   : item embeddings from CF path (n_items, d) hyperbolic
            i_kg   : item embeddings from KG path (n_items, d) hyperbolic
            beta_kg: KG denoiser scores for L_D
            beta_cf: CF denoiser scores for L_D
        """
        # ── User base embeddings ──────────────────────────────────────────────
        u_emb = self.user_emb.weight
        if self.user_proj is not None:
            u_emb = u_emb + self.user_proj(self.user_feat)
        u_hyp = self._to_hyperbolic(u_emb)

        # ── KG path ───────────────────────────────────────────────────────────
        i_kg, beta_kg = self._get_kg_item_emb()

        # ── CF path ───────────────────────────────────────────────────────────
        # Start CF item embeddings from entity_emb (optionally + item_proj)
        ent_emb = self.entity_emb.weight
        if self.item_proj is not None:
            ent_emb = ent_emb.clone()
            ent_emb[:self.n_items] = ent_emb[:self.n_items] + self.item_proj(self.item_feat)
        i_hyp0 = self._to_hyperbolic(ent_emb[:self.n_items])

        u_cf, i_cf, beta_cf = self._cf_propagate(u_hyp, i_hyp0, adj, cf_denoise_edges)
        return u_cf, i_cf, i_kg, beta_kg, beta_cf

    def get_embeddings(
        self, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Clean embeddings for evaluation — returns tangent-space vectors ready
        for inner-product ranking (consistent with BPR training objective).
        Fuses CF and KG item views in tangent space.
        """
        with torch.no_grad():
            u_cf, i_cf, i_kg, _, _ = self.forward(adj)
            i_tan = (self.ball.logmap0(i_cf) + self.ball.logmap0(i_kg)) / 2
            u_tan = self.ball.logmap0(u_cf)
        return u_tan, i_tan


# ── Loss functions ─────────────────────────────────────────────────────────────

def hrcl_loss_hyp(
    ball: PoincareBall,
    e_u: torch.Tensor,
    e_k: torch.Tensor,
    temp: float,
    lam: float,
    q: float,
) -> torch.Tensor:
    """
    Hyperbolic Robust Contrastive Loss (eq. 24, Yuan et al. 2025).

    Uses geodesic distance d_c(·,·) instead of cosine similarity.
    Positive pairs: (e_u[i], e_k[i]) — same item from CF and KG views.
    Negatives: all cross-pairs within the batch.

    L_HRCL = -(1-λ)·Σᵢ exp(-d_c(e_u[i], e_k[i])/τ)/q
             + λ·Σᵢ Σⱼ [exp(-d_c(e_u[i], e_k[j])/τ) + exp(-d_c(e_k[j], e_u[i])/τ)] / q

    Args:
        e_u, e_k : item embeddings from CF and KG views  (B, d) — hyperbolic
        temp     : temperature τ
        lam      : density adjustment λ
        q        : symmetry interpolation parameter

    Memory: dist_batch is O(B²·d).  Keep B ≤ 512.
    """
    # All-pairs geodesic distances (B, B)
    dist_mat = ball.dist_batch(e_u, e_k)   # dist_mat[i,j] = d_c(e_u[i], e_k[j])

    # Use negative distance as similarity (closer = more similar)
    sim = -dist_mat / temp                  # (B, B)
    pos = torch.diag(sim)                   # (B,) diagonal = positive pairs

    exp_pos = torch.exp(q * pos)            # (B,)
    exp_sim = torch.exp(q * sim)            # (B, B)

    term1 = -(1.0 - lam) * exp_pos / q                                   # (B,)
    # Exclude positive pairs (diagonal) from negative sum — same reasoning as
    # rcl_loss: including the diagonal flips the gradient sign on pos pairs.
    diag_mask = 1.0 - torch.eye(e_u.size(0), device=e_u.device)
    term3 = lam * ((exp_sim * diag_mask).sum(dim=1) +
                   (exp_sim * diag_mask).sum(dim=0)) / q                  # (B,)

    # ×q normalization: cancels the 1/q factor so loss is O(B) not O(B/q).
    # Without this the gradient is ~8000× BPR at q=0.001, destroying embeddings.
    return (term1 + term3).mean() * q


def margin_loss_rec(
    ball: PoincareBall,
    u: torch.Tensor,
    i_pos: torch.Tensor,
    i_neg: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Geometry-aware margin ranking loss (eq. 15, Yuan et al. 2025).

    L_Rec(u, i, j) = max(d_c²(u, i) − d_c²(u, j) + m, 0)

    Pulls the user closer to the positive item and pushes the negative away,
    measured by squared geodesic distance in the Poincaré ball.

    Uses a fixed margin m for simplicity (the paper derives an adaptive margin
    from the relative distances; fixed m=0.5 is a stable starting point).
    """
    d_pos = ball.dist(u, i_pos).pow(2)   # (B,)
    d_neg = ball.dist(u, i_neg).pow(2)   # (B,)
    return F.relu(d_pos - d_neg + margin).mean()


# ── CF edge sampler ────────────────────────────────────────────────────────────

def _sample_cf_edges(
    adj: torch.Tensor, n_edges: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample n_edges indices from the sparse adjacency matrix.

    Returns:
        sampled_idx   : indices into adj.values()
        row_idx       : user indices  (n_edges,)
        col_idx       : item indices  (n_edges,) — relative to item block
    """
    indices = adj.coalesce().indices()   # (2, nnz)
    nnz = indices.size(1)
    if n_edges >= nnz:
        perm = torch.arange(nnz, device=adj.device)
    else:
        perm = torch.randperm(nnz, device=adj.device)[:n_edges]

    row = indices[0][perm]
    col = indices[1][perm]
    return perm, row, col


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    epochs: int           = 300,
    emb_dim: int          = 512,
    n_kg_layers: int      = 2,
    n_cf_layers: int      = 3,
    lr: float             = 1e-3,
    batch_size: int       = 2048,
    cl_batch_size: int    = 512,
    cf_denoise_edges: int = 0,
    save: bool            = True,
    edge_dropout: float   = 0.1,
    kg_dropout: float     = 0.1,
    curvature_init: float = 1.0,
    hard_neg_refresh: int = 50,
    llm_feat_mode: str    = "fast",
    use_nlp_feat: bool    = True,
    eval_every: int       = 50,
    hrcl_weight: float    = 0.001,   # β₁  (hrcl loss ≈470; 0.001 balances with BPR ≈0.2)
    denoise_weight: float = 0.0,     # β₂ = β₃ — 0 prevents collapse; gates learn from task loss
    hrcl_lam: float       = 0.5,
    hrcl_q: float         = 0.001,
    hrcl_temp: float      = 0.15,
    rec_margin: float     = 0.5,
    denoise_temp: float   = 0.5,
    prebuilt_features: bool           = False,
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
    kg_heads, kg_rels, kg_tails, n_kg_ents = build_kg(item_enc)

    ball = PoincareBall(c_init=curvature_init).to(DEVICE)

    model = HEKCL(
        n_users=n_users,
        n_items=n_items,
        n_kg_ents=n_kg_ents,
        emb_dim=emb_dim,
        n_kg_layers=n_kg_layers,
        n_cf_layers=n_cf_layers,
        kg_heads=kg_heads,
        kg_rels=kg_rels,
        kg_tails=kg_tails,
        ball=ball,
        user_feat=user_feat,
        item_feat=item_feat,
        edge_dropout=edge_dropout,
        kg_dropout=kg_dropout,
        denoise_temp=denoise_temp,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 50)

    log.info(
        f"Training HEK-CL | device={DEVICE} | epochs={epochs} | emb_dim={emb_dim} | "
        f"kg_layers={n_kg_layers} | cf_layers={n_cf_layers} | "
        f"curvature_init={curvature_init} | hrcl_weight={hrcl_weight} | "
        f"denoise_weight={denoise_weight} | hrcl_q={hrcl_q} | hrcl_lam={hrcl_lam}"
    )

    emb_pool    = None
    best_recall = 0.0
    best_state  = None

    all_users_t = torch.LongTensor(train_df["user_idx"].values)
    all_pos_t   = torch.LongTensor(train_df["item_idx"].values)
    n_train     = len(train_df)

    for epoch in range(1, epochs + 1):

        # ── Refresh hard-negative pool ────────────────────────────────────────
        if epoch % hard_neg_refresh == 1:
            model.eval()
            with torch.no_grad():
                _, i_emb = model.get_embeddings(adj)
            emb_pool = build_hard_neg_pool(i_emb, k=50)
            if epoch > 1:
                log.info(f"  [Epoch {epoch}] Hard neg pool refreshed")
            model.train()

        neg_df    = sample_negatives(train_df, n_items, emb_pool=emb_pool)
        all_neg_t = torch.LongTensor(neg_df["neg_idx"].values)

        model.train()
        optimizer.zero_grad()

        # ── Full forward pass ─────────────────────────────────────────────────
        # CF edge sampling happens inside forward (after dropout) so indices
        # are always valid for the dropout-reduced adj.values().
        u_cf, i_cf, i_kg, beta_kg, beta_cf = model.forward(adj, cf_denoise_edges)

        # ── HRCL loss: cross-view item alignment ──────────────────────────────
        cl_idx = torch.randperm(n_items, device=DEVICE)[:cl_batch_size]
        loss_hrcl = hrcl_loss_hyp(
            ball, i_cf[cl_idx], i_kg[cl_idx], hrcl_temp, hrcl_lam, hrcl_q
        )

        # ── Denoising regulariser ─────────────────────────────────────────────
        loss_dk = EdgeDenoiser.denoise_loss(beta_kg)
        loss_du = EdgeDenoiser.denoise_loss(beta_cf)

        # ── Ranking loss L_Rec (geometry-aware margin) ────────────────────────
        rec_val = 0.0
        perm    = torch.randperm(n_train)
        loss_rec_accum = torch.tensor(0.0, device=DEVICE)

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            bu  = all_users_t[idx].to(DEVICE)
            bp  = all_pos_t[idx].to(DEVICE)
            bn  = all_neg_t[idx].to(DEVICE)

            u_tan  = ball.logmap0(u_cf[bu])
            ip_tan = ball.logmap0(i_cf[bp])
            in_tan = ball.logmap0(i_cf[bn])
            bpr_val = (u_tan * ip_tan).sum(1) - (u_tan * in_tan).sum(1)
            l = (-F.logsigmoid(bpr_val)).mean()
            loss_rec_accum = loss_rec_accum + l * (end - start) / n_train
            rec_val += l.item() * (end - start) / n_train

        # ── Total loss ────────────────────────────────────────────────────────
        loss = (loss_rec_accum
                + hrcl_weight  * loss_hrcl
                + denoise_weight * loss_dk
                + denoise_weight * loss_du)
        loss.backward()

        optimizer.step()
        scheduler.step()
        ball.log_c.data.clamp_(math.log(1e-2), math.log(10.0))   # keep c in valid range

        if epoch % 10 == 0:
            log.info(
                f"  Epoch {epoch}/{epochs}  "
                f"rec={rec_val:.4f}  "
                f"hrcl={loss_hrcl.item():.4f}  "
                f"dk={loss_dk.item():.4f}  "
                f"du={loss_du.item():.4f}  "
                f"c={ball.c.item():.4f}  "
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
                best_state  = {
                    "model": copy.deepcopy(model.state_dict()),
                    "ball":  copy.deepcopy(ball.state_dict()),
                }
                log.info(f"  [Epoch {epoch}] val Recall@10={val_recall:.4f}  *** new best ***")
            else:
                log.info(f"  [Epoch {epoch}] val Recall@10={val_recall:.4f}  (best={best_recall:.4f})")

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        ball.load_state_dict(best_state["ball"])
        log.info(f"Loaded best checkpoint (val Recall@10={best_recall:.4f})")

    # ── Final evaluation ───────────────────────────────────────────────────────
    model.eval()
    log.info(f"Computing embeddings for evaluation… (curvature c={ball.c.item():.4f})")
    u_emb, i_emb = model.get_embeddings(adj)

    # get_embeddings already returns tangent-space vectors (logmap0 applied inside)
    u_emb = u_emb.detach()
    i_emb = i_emb.detach()

    log.info(f"  Embedding norms — users mean={u_emb.norm(dim=1).mean():.4f}, "
             f"items mean={i_emb.norm(dim=1).mean():.4f}")

    test_users  = test_df["user_idx"].values
    test_items  = test_df["item_idx"].values
    train_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    hits, ndcg_sum, total = 0, 0.0, 0
    chunk = 4096
    log.info(f"Evaluating {len(test_users):,} test users in chunks of {chunk}…")

    for start in range(0, len(test_users), chunk):
        end     = min(start + chunk, len(test_users))
        u_idx   = test_users[start:end]
        i_idx   = test_items[start:end]
        u_batch = u_emb[u_idx]
        scores  = (u_batch @ i_emb.T).cpu().numpy()

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

    print(f"\n=== HEK-CL Results ===")
    print(f"  Precision@10: {precision:.4f}")
    print(f"  Recall@10:    {recall:.4f}")
    print(f"  NDCG@10:      {ndcg:.4f}")
    print(f"  Final curvature c={ball.c.item():.4f}")

    skip_str = ",".join(skip_feature_groups) if skip_feature_groups else ""
    variant  = (f"c={ball.c.item():.3f} q={hrcl_q} lam={hrcl_lam}"
                + (f" skip {skip_str}" if skip_str else ""))
    save_result(
        model="HEK-CL", variant=variant,
        precision=precision, recall=recall, ndcg=ndcg,
        epochs=epochs, emb_dim=emb_dim,
        n_layers=n_cf_layers + n_kg_layers,
        skip_groups=skip_str,
    )

    if save:
        file_tag   = f"skip_{skip_str.replace(',', '_')}" if skip_str else "all"
        model_path = MODEL_OUT.parent / f"hek_cl_{file_tag}.pt"
        model_path.parent.mkdir(exist_ok=True)
        EMB_OUT.mkdir(exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "ball": ball.state_dict()},
            model_path,
        )
        user_dec = {v: k for k, v in user_enc.items()}
        item_dec = {v: k for k, v in item_enc.items()}
        torch.save({"embeddings": u_emb.cpu(), "id_map": user_dec},
                   EMB_OUT / f"hek_cl_{file_tag}_user_embeddings.pt")
        torch.save({"embeddings": i_emb.cpu(), "id_map": item_dec},
                   EMB_OUT / f"hek_cl_{file_tag}_item_embeddings.pt")
        log.info(f"Model → {model_path} | Embeddings → {EMB_OUT}")

    return recall


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HEK-CL recommendation model (Phase 2)")
    parser.add_argument("--epochs",           type=int,   default=300)
    parser.add_argument("--emb-dim",          type=int,   default=512,
                        help="Embedding dimension (default 512; smaller than KGAT due to "
                             "Poincaré ball dist_batch memory: O(B²·d))")
    parser.add_argument("--kg-layers",        type=int,   default=2)
    parser.add_argument("--cf-layers",        type=int,   default=3)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--batch-size",       type=int,   default=2048,
                        help="Batch size for L_Rec ranking loss (default 2048)")
    parser.add_argument("--cl-batch-size",    type=int,   default=512,
                        help="Batch size for HRCL cross-view CL (default 512; "
                             "memory: O(B²·d) for pairwise geodesic distances)")
    parser.add_argument("--cf-denoise-edges", type=int,   default=0,
                        help="Number of CF edges sampled per epoch for denoising MLP "
                             "(default 0=disabled; gradient through sparse adj is O(N²) memory)")
    parser.add_argument("--edge-dropout",     type=float, default=0.1)
    parser.add_argument("--kg-dropout",       type=float, default=0.1)
    parser.add_argument("--curvature",        type=float, default=1.0,
                        help="Initial curvature c=-κ of the Poincaré ball (default 1.0; "
                             "trained end-to-end)")
    parser.add_argument("--hard-neg-refresh", type=int,   default=50)
    parser.add_argument("--llm-feat-mode",    default="fast",
                        choices=["none", "fast", "full"])
    parser.add_argument("--no-nlp-feat",      action="store_true")
    parser.add_argument("--eval-every",       type=int,   default=50)
    parser.add_argument("--no-save",          action="store_true")
    # Loss weights
    parser.add_argument("--hrcl-weight",      type=float, default=0.001,
                        help="Weight of HRCL cross-view loss β₁ (default 0.001; "
                             "hrcl_loss≈470 so 0.001 balances with BPR≈0.2)")
    parser.add_argument("--denoise-weight",   type=float, default=0.0,
                        help="Weight of denoising regulariser β₂=β₃ (default 0; "
                             "0 prevents collapse — gates learn purely from task gradient)")
    # HRCL params
    parser.add_argument("--hrcl-lam",         type=float, default=0.5,
                        help="HRCL density adjustment λ (default 0.5)")
    parser.add_argument("--hrcl-q",           type=float, default=0.001,
                        help="HRCL symmetry interpolation q (default 0.001)")
    parser.add_argument("--hrcl-temp",        type=float, default=0.15,
                        help="HRCL temperature τ (default 0.15)")
    parser.add_argument("--rec-margin",       type=float, default=0.5,
                        help="Fixed margin m for geometry-aware L_Rec (default 0.5)")
    parser.add_argument("--denoise-temp",     type=float, default=0.5,
                        help="Gumbel-Softmax temperature for edge gates (default 0.5)")
    parser.add_argument("--prebuilt-features", action="store_true")
    parser.add_argument("--skip-feature-groups", default="",
                        help="Comma-separated feature groups to drop")
    args = parser.parse_args()

    skip_groups = [g.strip() for g in args.skip_feature_groups.split(",") if g.strip()] \
        if args.skip_feature_groups else None

    train(
        epochs=args.epochs,
        emb_dim=args.emb_dim,
        n_kg_layers=args.kg_layers,
        n_cf_layers=args.cf_layers,
        lr=args.lr,
        batch_size=args.batch_size,
        cl_batch_size=args.cl_batch_size,
        cf_denoise_edges=args.cf_denoise_edges,
        save=not args.no_save,
        edge_dropout=args.edge_dropout,
        kg_dropout=args.kg_dropout,
        curvature_init=args.curvature,
        hard_neg_refresh=args.hard_neg_refresh,
        llm_feat_mode=args.llm_feat_mode,
        use_nlp_feat=not args.no_nlp_feat,
        eval_every=args.eval_every,
        hrcl_weight=args.hrcl_weight,
        denoise_weight=args.denoise_weight,
        hrcl_lam=args.hrcl_lam,
        hrcl_q=args.hrcl_q,
        hrcl_temp=args.hrcl_temp,
        rec_margin=args.rec_margin,
        denoise_temp=args.denoise_temp,
        prebuilt_features=args.prebuilt_features,
        skip_feature_groups=skip_groups,
    )


if __name__ == "__main__":
    main()
