#!/usr/bin/env python3
"""
recommendation_simgcl_hrcl.py — SimGCL with Hyperbolic Robust Contrastive Loss.

Phase 1 implementation of HEK-CL (Yuan et al., ACM TOIS 2025, §3.2):
replaces InfoNCE with the Robust Contrastive Loss (RCL) while keeping the
SimGCL architecture (noise-augmented views, BPR ranking loss) unchanged.

Motivation for the swap:
  InfoNCE is an asymmetric loss in gradient space and does not satisfy the
  symmetric property, making it non-robust when augmented views are noisy.
  For scraped Google Maps review data (fake reviews, spam, tourist vs. local
  bias), training views are inherently noisy — HRCL provides theoretical
  bounds on performance under noise level η < 0.5.

  R(f*) ≤ ε + 2η_max · e^{s_max} / (1 - 2η_max)     (Corollary 3.2)

Key change over SimGCL:
  info_nce_loss(z1, z2, τ)  →  rcl_loss(z1, z2, τ, λ, q)

  The RCL loss (eq. 18, Yuan et al. 2025) satisfies the symmetry condition
  l(s, 1) + l(s, −1) = const for all s ∈ ℝ, providing noise robustness.
  When q → 0, RCL converges to InfoNCE (Corollary 3.1).

Loss:  L = L_BPR + β₁ · L_RCL
  L_BPR  = standard BPR (manual gradient, identical to SimGCL)
  L_RCL  = Robust Contrastive Loss over two noise-augmented views:

           L_RCL = Σᵢ { -exp(q·s⁺ᵢ/τ)/q
                       + λ · exp(q·s⁺ᵢ/τ)/q
                       + λ · Σⱼ [exp(q·sᵢⱼ/τ) + exp(q·sⱼᵢ/τ)] / q }

  where s⁺ᵢ = cos_sim(z1[i], z2[i]),  sᵢⱼ = cos_sim(z1[i], z2[j])

New hyperparameters vs SimGCL:
  --hrcl-lam   λ  density / balance factor (default 0.5)
  --hrcl-q     q  symmetry interpolation   (default 0.001, paper optimal)
                   q→0: converges to InfoNCE; q→1: full symmetric loss

Usage:
    python recommendation_simgcl_hrcl.py
    python recommendation_simgcl_hrcl.py --epochs 300 --emb-dim 2048 --layers 4 \\
        --cl-weight 0.001 --hrcl-lam 0.5 --hrcl-q 0.001 --eval-every 50
"""

import argparse
import copy
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint as grad_ckpt

from model_results import save_result
from recommendation_gnn import (
    _sparse_mm_f32, dropout_adj,
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
log = logging.getLogger("simgcl_hrcl")

MODEL_OUT = Path("models/simgcl_hrcl.pt")
EMB_OUT   = Path("data/embeddings")


# ── Model (identical to SimGCL backbone) ──────────────────────────────────────

class SimGCLBackbone(nn.Module):
    """
    LightGCN backbone with noise-based contrastive augmentation.
    Architecture is identical to SimGCL; only the CL loss changes.
    """

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
            return grad_ckpt(_full_pass, self.user_emb.weight, self.item_emb.weight,
                             use_reentrant=False)
        return _full_pass(self.user_emb.weight, self.item_emb.weight)

    def forward_cl(self, adj: torch.Tensor, noise_eps: float):
        final = self._propagate(adj, noise_eps=noise_eps)
        return final[:self.n_users], final[self.n_users:]

    def get_embeddings(self, adj: torch.Tensor):
        with torch.no_grad():
            final = self._propagate(adj, noise_eps=0.0)
        return final[:self.n_users], final[self.n_users:]


# ── Robust Contrastive Loss ────────────────────────────────────────────────────

def rcl_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temp: float,
    lam: float,
    q: float,
) -> torch.Tensor:
    """
    Robust Contrastive Loss (Euclidean / cosine version of HRCL).

    Implements eq. (18) from Yuan et al. (2025):

        L_RCL = Σᵢ { term1ᵢ + term2ᵢ + term3ᵢ } / B

    where:
        term1ᵢ = -exp(q · s⁺ᵢ / τ) / q          ← positive pair (maximize sim)
        term2ᵢ = λ · exp(q · s⁺ᵢ / τ) / q        ← balance factor
        term3ᵢ = λ · Σⱼ [exp(q·sᵢⱼ/τ) + exp(q·sⱼᵢ/τ)] / q   ← push negatives

    The symmetry condition l(s,1) + l(s,−1) = const is satisfied at q → 1.
    As q → 0, L_RCL → L_InfoNCE + log(λ)  (Corollary 3.1).

    Args:
        z1, z2 : augmented view embeddings  (B, d)
        temp   : temperature τ
        lam    : density adjustment λ  (default 0.5)
        q      : symmetry interpolation (default 0.001, paper optimal)

    Numerical note: with q=0.001 and cosine sim ∈ [−1,1], exp(q·s/τ) ≈ 1±ε,
    so this is numerically stable without log-sum-exp tricks.
    """
    z1 = F.normalize(z1, dim=1)   # (B, d)
    z2 = F.normalize(z2, dim=1)

    # Full similarity matrix (B, B), scaled by temperature
    sim = torch.mm(z1, z2.T) / temp           # sᵢⱼ = cos(z1[i], z2[j]) / τ

    pos = torch.diag(sim)                      # (B,) positive pair scores s⁺ᵢ

    exp_pos  = torch.exp(q * pos)              # (B,)
    exp_sim  = torch.exp(q * sim)              # (B, B)

    term1 = -exp_pos / q                                                   # (B,)
    term2 = lam * exp_pos / q                                              # (B,)
    # Exclude positive pairs (diagonal) from the negative sum.
    # Including them creates net gradient (−1+3λ)·exp/q on s⁺; at λ=0.5 that
    # is +0.5 — positive — so minimisation would push positive pairs APART.
    # Masking the diagonal gives net gradient (−1+λ)·exp/q < 0 for any λ<1. ✓
    diag_mask = 1.0 - torch.eye(z1.size(0), device=z1.device)
    term3 = lam * ((exp_sim * diag_mask).sum(dim=1) +
                   (exp_sim * diag_mask).sum(dim=0)) / q                  # (B,)

    # Multiply by q to cancel the 1/q factor: without this the loss is O(B/q).
    # At q=0.001 and B=8192 the raw loss ≈ 16M and the gradient is ~8000×BPR,
    # blowing up the BPR embeddings.  After ×q the gradient scale is O(1) —
    # comparable to InfoNCE — so cl_weight=0.2 is appropriate (same as SimGCL).
    return (term1 + term2 + term3).mean() * q


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    epochs: int        = 300,
    emb_dim: int       = 2048,
    n_layers: int      = 4,
    lr: float          = 1e-3,
    batch_size: int    = 8192,
    save: bool         = True,
    grad_checkpoint: bool = False,
    edge_dropout: float   = 0.0,
    use_film: bool        = False,
    hard_neg_refresh: int = 50,
    llm_feat_mode: str    = "fast",
    use_nlp_feat: bool    = True,
    eval_every: int       = 20,
    # Shared CL params
    cl_weight: float   = 0.001,   # β₁ — rcl_loss after ×q is still O(B)≈16k, not O(log B)≈9;
    noise_eps: float   = 0.1,     #      needs to be ~log(B)/B times SimGCL's 0.2
    # HRCL-specific
    hrcl_lam: float    = 0.5,     # λ: density adjustment / balance factor
    hrcl_q: float      = 0.001,   # q: symmetry interpolation (paper optimal)
    hrcl_temp: float   = 0.15,    # τ: temperature (same default as SimGCL's cl_temp)
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

    model = SimGCLBackbone(
        n_users=n_users, n_items=n_items, emb_dim=emb_dim, n_layers=n_layers,
        user_feat=user_feat, item_feat=item_feat,
        grad_checkpoint=grad_checkpoint, edge_dropout=edge_dropout,
        use_film=use_film,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 50)

    log.info(
        f"Training SimGCL-HRCL | device={DEVICE} | epochs={epochs} | emb_dim={emb_dim} | "
        f"layers={n_layers} | cl_weight={cl_weight} | noise_eps={noise_eps} | "
        f"hrcl_lam={hrcl_lam} | hrcl_q={hrcl_q} | hrcl_temp={hrcl_temp}"
    )

    emb_pool    = None
    best_recall = 0.0
    best_state  = None
    rng         = np.random.default_rng()

    train_with_neg = sample_negatives(train_df, n_items, emb_pool=emb_pool)
    all_users  = torch.tensor(train_with_neg["user_idx"].values, dtype=torch.long)
    all_pos    = torch.tensor(train_with_neg["item_idx"].values, dtype=torch.long)
    all_neg    = torch.tensor(train_with_neg["neg_idx"].values,  dtype=torch.long)
    n_train    = len(all_users)

    for epoch in range(1, epochs + 1):

        # ── Refresh hard-negative pool ────────────────────────────────────────
        if hard_neg_refresh > 0 and epoch > 1 and (epoch - 1) % hard_neg_refresh == 0:
            model.eval()
            with torch.no_grad():
                _, i_emb = model.get_embeddings(adj)
            emb_pool = build_hard_neg_pool(i_emb, k=50)
            del i_emb
            model.train()
            log.info(f"  [Epoch {epoch}] Hard neg pool refreshed")
            train_with_neg = sample_negatives(train_df, n_items, emb_pool=emb_pool)
            all_users = torch.tensor(train_with_neg["user_idx"].values, dtype=torch.long)
            all_pos   = torch.tensor(train_with_neg["item_idx"].values, dtype=torch.long)
            all_neg   = torch.tensor(train_with_neg["neg_idx"].values,  dtype=torch.long)

        model.train()
        optimizer.zero_grad()

        # ── Step 1: HRCL loss — two augmented views, backprop immediately ────
        u_emb1, i_emb1 = model.forward_cl(adj, noise_eps)
        u_emb2, i_emb2 = model.forward_cl(adj, noise_eps)

        sub_u = torch.randperm(n_users, device=DEVICE)[:batch_size]
        sub_i = torch.randperm(n_items, device=DEVICE)[:batch_size]

        loss_cl = (
            rcl_loss(u_emb1[sub_u], u_emb2[sub_u], hrcl_temp, hrcl_lam, hrcl_q) +
            rcl_loss(i_emb1[sub_i], i_emb2[sub_i], hrcl_temp, hrcl_lam, hrcl_q)
        )
        (cl_weight * loss_cl).backward()
        cl_val = loss_cl.item()
        del u_emb1, i_emb1, u_emb2, i_emb2, loss_cl
        torch.cuda.empty_cache()

        # ── Step 2a: BPR gradient — no_grad chunked scatter_add ─────────────
        _saved_dropout    = model.edge_dropout
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

        # ── Step 2b: one ckpt forward + backward with precomputed BPR grad ──
        final_bpr = model._propagate(adj, noise_eps=0.0, use_ckpt=True)
        final_bpr.backward(gradient=grad_final)
        del final_bpr, grad_final
        model.edge_dropout = _saved_dropout
        torch.cuda.empty_cache()

        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            log.info(
                f"  Epoch {epoch}/{epochs}  "
                f"bpr={bpr_val:.4f}  "
                f"cl(hrcl)={cl_val:.4f}  "
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
                log.info(f"  [Epoch {epoch}] val Recall@10={val_recall:.4f}  *** new best ***")
            else:
                log.info(f"  [Epoch {epoch}] val Recall@10={val_recall:.4f}  (best={best_recall:.4f})")

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info(f"Loaded best checkpoint (val Recall@10={best_recall:.4f})")

    # ── Final evaluation ───────────────────────────────────────────────────────
    model.eval()
    log.info("Computing embeddings for evaluation…")
    u_emb, i_emb = model.get_embeddings(adj)

    log.info(f"  Embedding norms — users mean={u_emb.norm(dim=1).mean():.4f}, "
             f"items mean={i_emb.norm(dim=1).mean():.4f}")

    import pandas as pd
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

    print(f"\n=== SimGCL-HRCL Results ===")
    print(f"  Precision@10: {precision:.4f}")
    print(f"  Recall@10:    {recall:.4f}")
    print(f"  NDCG@10:      {ndcg:.4f}")

    skip_str = ",".join(skip_feature_groups) if skip_feature_groups else ""
    variant  = f"q={hrcl_q} lam={hrcl_lam}" + (f" skip {skip_str}" if skip_str else "")
    save_result(
        model="SimGCL-HRCL", variant=variant,
        precision=precision, recall=recall, ndcg=ndcg,
        epochs=epochs, emb_dim=emb_dim, n_layers=n_layers,
        skip_groups=skip_str,
    )

    if save:
        file_tag   = f"skip_{skip_str.replace(',', '_')}" if skip_str else "all"
        model_path = MODEL_OUT.parent / f"simgcl_hrcl_{file_tag}.pt"
        model_path.parent.mkdir(exist_ok=True)
        EMB_OUT.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_path)
        user_dec = {v: k for k, v in user_enc.items()}
        item_dec = {v: k for k, v in item_enc.items()}
        torch.save({"embeddings": u_emb.cpu(), "id_map": user_dec},
                   EMB_OUT / f"simgcl_hrcl_{file_tag}_user_embeddings.pt")
        torch.save({"embeddings": i_emb.cpu(), "id_map": item_dec},
                   EMB_OUT / f"simgcl_hrcl_{file_tag}_item_embeddings.pt")
        log.info(f"Model → {model_path} | Embeddings → {EMB_OUT}")

    return recall


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SimGCL-HRCL recommendation model (Phase 1)")
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--emb-dim",      type=int,   default=2048)
    parser.add_argument("--layers",       type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--batch-size",   type=int,   default=8192)
    parser.add_argument("--edge-dropout", type=float, default=0.1)
    parser.add_argument("--film",         action="store_true", default=False)
    parser.add_argument("--grad-checkpoint", action="store_true", default=False)
    parser.add_argument("--hard-neg-refresh", type=int, default=50)
    parser.add_argument("--llm-feat-mode", default="fast",
                        choices=["none", "fast", "full"])
    parser.add_argument("--no-nlp-feat", action="store_true")
    parser.add_argument("--eval-every",  type=int,   default=20)
    parser.add_argument("--no-save",     action="store_true")
    # Shared CL params
    parser.add_argument("--cl-weight",   type=float, default=0.001,
                        help="Weight of HRCL loss β₁ (default 0.001); rcl_loss after ×q "
                             "is still O(B)≈16k not O(log B)≈9, so much smaller than SimGCL's 0.2")
    parser.add_argument("--noise-eps",   type=float, default=0.1,
                        help="Per-layer noise magnitude ε (default 0.1)")
    # HRCL-specific
    parser.add_argument("--hrcl-lam",    type=float, default=0.5,
                        help="HRCL density adjustment λ (default 0.5)")
    parser.add_argument("--hrcl-q",      type=float, default=0.001,
                        help="HRCL symmetry interpolation q (default 0.001, paper optimal; "
                             "q→0 ≈ InfoNCE, q→1 = full symmetric loss)")
    parser.add_argument("--hrcl-temp",   type=float, default=0.15,
                        help="HRCL temperature τ (default 0.15)")
    parser.add_argument("--prebuilt-features", action="store_true")
    parser.add_argument("--skip-feature-groups", default="",
                        help="Comma-separated feature groups to drop")
    args = parser.parse_args()

    skip_groups = [g.strip() for g in args.skip_feature_groups.split(",") if g.strip()] \
        if args.skip_feature_groups else None

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
        cl_weight=args.cl_weight,
        noise_eps=args.noise_eps,
        hrcl_lam=args.hrcl_lam,
        hrcl_q=args.hrcl_q,
        hrcl_temp=args.hrcl_temp,
        prebuilt_features=args.prebuilt_features,
        skip_feature_groups=skip_groups,
    )


if __name__ == "__main__":
    main()
