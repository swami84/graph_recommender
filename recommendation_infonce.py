#!/usr/bin/env python3
"""
recommendation_infonce.py — Pure InfoNCE Contrastive Learning for Recommendation.

A self-supervised baseline that trains using ONLY symmetric InfoNCE contrastive
loss — no BPR ranking supervision. Two augmented views are generated via
per-layer uniform noise injection (identical to SimGCL's augmentation strategy).

Purpose: ablation to isolate the pure CL signal before layering in HRCL (Phase 1)
and the full HEK-CL stack (Phase 2). Answers: how much does ranking supervision
(BPR) contribute vs. contrastive alignment alone?

Loss:  L = L_CL  (symmetric InfoNCE, no BPR)
Views: two LightGCN propagations with per-layer uniform noise ε

Architecture is identical to SimGCL — LightGCN backbone with optional
user/item side-feature projection. Only the training objective changes.

Usage:
    python recommendation_infonce.py
    python recommendation_infonce.py --epochs 300 --emb-dim 2048 --layers 4 \\
        --noise-eps 0.1 --cl-temp 0.15 --eval-every 50
"""

import argparse
import contextlib
import copy
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from transformers import Adafactor
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
    _is_main_rank, _sync_grads, init_distributed,
    DEVICE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("infonce")
log.setLevel(logging.INFO)
if not log.handlers:
    _lh = logging.StreamHandler()
    _lh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_lh)
    log.propagate = False

MODEL_OUT = Path("models/infonce.pt")
EMB_OUT   = Path("data/embeddings")


# ── Model (identical backbone to SimGCL) ──────────────────────────────────────

class InfoNCEModel(nn.Module):
    """
    LightGCN backbone with noise-based contrastive augmentation.

    Standard forward pass (for evaluation) is identical to LightGCN.
    forward_cl() runs the same propagation with per-layer uniform noise,
    producing augmented views for the InfoNCE contrastive objective.

    No BPR head — training signal comes entirely from CL alignment.
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

    def _propagate(self, adj: torch.Tensor, noise_eps: float = 0.0) -> torch.Tensor:
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

        if self.grad_checkpoint and self.training:
            return grad_ckpt(_full_pass, self.user_emb.weight, self.item_emb.weight,
                             use_reentrant=False)
        return _full_pass(self.user_emb.weight, self.item_emb.weight)

    def forward_cl(self, adj: torch.Tensor, noise_eps: float):
        """Augmented forward — two calls produce two independent views."""
        final = self._propagate(adj, noise_eps=noise_eps)
        return final[:self.n_users], final[self.n_users:]

    def get_embeddings(self, adj: torch.Tensor):
        """Clean embeddings for evaluation — no noise, no grad."""
        with torch.no_grad():
            final = self._propagate(adj, noise_eps=0.0)
        return final[:self.n_users], final[self.n_users:]


# ── InfoNCE loss ───────────────────────────────────────────────────────────────

def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float) -> torch.Tensor:
    """
    Symmetric InfoNCE between two augmented views.
    Positive pairs: (z1[i], z2[i]).  Negatives: all other rows in batch.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.T) / temp
    labels = torch.arange(len(z1), device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    epochs: int           = 300,
    emb_dim: int          = 2048,
    n_layers: int         = 4,
    lr: float             = 1e-3,
    weight_decay: float   = 1e-4,
    batch_size: int       = 8192,
    save: bool            = True,
    grad_checkpoint: bool = False,
    edge_dropout: float   = 0.0,
    use_film: bool        = False,
    llm_feat_mode: str    = "fast",
    use_nlp_feat: bool    = True,
    eval_every: int       = 50,
    noise_eps: float      = 0.1,
    cl_temp: float        = 0.15,
    cl_temp_init: float   = 0.5,
    lambda_bpr: float     = 1.0,
    hard_neg_refresh: int = 50,
    prebuilt_features: bool         = False,
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

    model = InfoNCEModel(
        n_users=n_users, n_items=n_items, emb_dim=emb_dim, n_layers=n_layers,
        user_feat=user_feat, item_feat=item_feat,
        grad_checkpoint=grad_checkpoint, edge_dropout=edge_dropout,
        use_film=use_film,
    ).to(DEVICE)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = model.to(torch.bfloat16)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if dist.is_initialized() else model

    # L2 regularisation is applied manually in the BPR gradient loop (r term)
    optimizer = Adafactor(model.parameters(), lr=lr,
                          relative_step=False, scale_parameter=False, warmup_init=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 50)

    log.info(
        f"Training InfoNCE | device={DEVICE} | epochs={epochs} | emb_dim={emb_dim} | "
        f"layers={n_layers} | lr={lr} | wd={weight_decay} | "
        f"noise_eps={noise_eps} | cl_temp={cl_temp}"
    )

    # Pre-build interaction pairs for user-item InfoNCE.
    # The positive pair is (user, interacted_item); negatives are all other
    # cross-pairs in the mini-batch.  This directly optimises user-item affinity.
    train_pairs  = sample_negatives(train_df, n_items)
    all_users_tr = torch.tensor(train_pairs["user_idx"].values, dtype=torch.long)
    all_pos_tr   = torch.tensor(train_pairs["item_idx"].values, dtype=torch.long)
    all_neg_tr   = torch.tensor(train_pairs["neg_idx"].values,  dtype=torch.long)
    n_train      = len(all_users_tr)
    emb_dim      = raw_model.user_emb.weight.size(1)

    best_recall  = 0.0
    best_state   = None
    hard_neg_pool = None

    for epoch in range(1, epochs + 1):

        # ── Hard negative pool refresh ─────────────────────────────────────────
        if hard_neg_refresh > 0 and epoch % hard_neg_refresh == 1:
            with torch.no_grad():
                _, i_emb_pool = raw_model.get_embeddings(adj)
            hard_neg_pool = build_hard_neg_pool(i_emb_pool, k=50)
            del i_emb_pool
            neg_df     = sample_negatives(train_df, n_items, hard_ratio=0.5, emb_pool=hard_neg_pool)
            all_neg_tr = torch.tensor(neg_df["neg_idx"].values, dtype=torch.long)
            if epoch > 1:
                log.info("  [Epoch %d] Hard neg pool refreshed", epoch)

        cl_temp_curr = cl_temp_init * (cl_temp / cl_temp_init) ** (epoch / epochs)

        raw_model.train()
        optimizer.zero_grad()

        # ── Manual-gradient InfoNCE over ALL training pairs ───────────────────
        # Replicates SimGCL's BPR manual-gradient pattern for InfoNCE.
        # A single-batch sample covers only ~1.65% of 494k pairs; most users
        # receive no gradient update each epoch.  Manual accumulation covers
        # 100% of pairs with one GCN forward + one GCN backward per epoch.
        #
        #   Step a: no_grad GCN forward → u_det, i_det
        #   Step b: for each mini-batch, create leaf tensors, run info_nce_loss,
        #           backward → accumulate grad into grad_final via scatter_add
        #   Step c: one fresh GCN forward + backward(grad_final)
        _saved_dropout    = raw_model.edge_dropout
        raw_model.edge_dropout = 0.0

        with torch.no_grad():
            final_det = raw_model._propagate(adj, noise_eps=0.0)
            u_det = final_det[:n_users]
            i_det = final_det[n_users:]

        grad_final     = torch.zeros_like(final_det)
        total_cl_loss  = 0.0
        total_bpr_loss = 0.0
        perm = torch.randperm(n_train)

        for start in range(0, n_train, batch_size):
            end   = min(start + batch_size, n_train)
            idx   = perm[start:end]
            bu    = all_users_tr[idx].to(DEVICE)
            bp    = all_pos_tr[idx].to(DEVICE)
            bn    = all_neg_tr[idx].to(DEVICE)

            u_b   = u_det[bu].detach().requires_grad_(True)
            i_b   = i_det[bp].detach().requires_grad_(True)
            i_n_b = i_det[bn].detach().requires_grad_(True)

            loss_cl   = info_nce_loss(u_b, i_b, cl_temp_curr)
            pos_score = (u_b * i_b).sum(dim=1)
            neg_score = (u_b * i_n_b).sum(dim=1)
            loss_bpr  = -F.logsigmoid(pos_score - neg_score).mean()
            loss = loss_cl + lambda_bpr * loss_bpr
            loss.backward()
            total_cl_loss  += loss_cl.item() * len(bu) / n_train
            total_bpr_loss += loss_bpr.item() * len(bu) / n_train

            if u_b.grad is not None:
                grad_final[:n_users].scatter_add_(
                    0, bu.unsqueeze(1).expand(-1, emb_dim), u_b.grad
                )
            if i_b.grad is not None:
                grad_final[n_users:].scatter_add_(
                    0, bp.unsqueeze(1).expand(-1, emb_dim), i_b.grad
                )
            if i_n_b.grad is not None:
                grad_final[n_users:].scatter_add_(
                    0, bn.unsqueeze(1).expand(-1, emb_dim), i_n_b.grad
                )

        del final_det, u_det, i_det

        final_grad = raw_model._propagate(adj, noise_eps=noise_eps)
        scalar = (final_grad * grad_final.detach()).sum()
        del final_grad  # not needed during checkpoint recomputation — free 2 GiB before backward
        scalar.backward()
        del grad_final, scalar
        raw_model.edge_dropout = _saved_dropout

        optimizer.step()
        scheduler.step()

        if _is_main_rank() and (epoch == 1 or epoch % 10 == 0):
            log.info(
                f"  Epoch {epoch}/{epochs}  "
                f"cl={total_cl_loss:.4f}  bpr={total_bpr_loss:.4f}  "
                f"temp={cl_temp_curr:.3f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if eval_every > 0 and epoch % eval_every == 0:
            raw_model.eval()
            val_recall = _eval_recall_at_k(
                raw_model, adj, test_df, train_df, n_items, k=10
            )
            raw_model.train()
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
    log.info("Computing embeddings for evaluation…")
    u_emb, i_emb = raw_model.get_embeddings(adj)

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

    print(f"\n=== InfoNCE Results ===")
    print(f"  Precision@10: {precision:.4f}")
    print(f"  Recall@10:    {recall:.4f}")
    print(f"  NDCG@10:      {ndcg:.4f}")

    skip_str = ",".join(skip_feature_groups) if skip_feature_groups else ""
    variant  = f"skip {skip_str}" if skip_str else "all features"
    save_result(
        model="InfoNCE", variant=variant,
        precision=precision, recall=recall, ndcg=ndcg,
        epochs=epochs, emb_dim=emb_dim, n_layers=n_layers,
        skip_groups=skip_str,
    )

    if save:
        if _is_main_rank():
            file_tag   = f"skip_{skip_str.replace(',', '_')}" if skip_str else "all"
            model_path = MODEL_OUT.parent / f"infonce_{file_tag}.pt"
            model_path.parent.mkdir(exist_ok=True)
            EMB_OUT.mkdir(exist_ok=True)
            torch.save(raw_model.state_dict(), model_path)
            user_dec = {v: k for k, v in user_enc.items()}
            item_dec = {v: k for k, v in item_enc.items()}
            torch.save({"embeddings": u_emb.cpu(), "id_map": user_dec},
                       EMB_OUT / f"infonce_{file_tag}_user_embeddings.pt")
            torch.save({"embeddings": i_emb.cpu(), "id_map": item_dec},
                       EMB_OUT / f"infonce_{file_tag}_item_embeddings.pt")
            log.info(f"Model → {model_path} | Embeddings → {EMB_OUT}")

    return recall


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="InfoNCE-only recommendation model")
    parser.add_argument("--epochs",        type=int,   default=300)
    parser.add_argument("--emb-dim",       type=int,   default=2048)
    parser.add_argument("--layers",        type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight-decay",  type=float, default=1e-4,
                        help="L2 weight decay (replaces BPR's implicit L2, default 1e-4)")
    parser.add_argument("--batch-size",    type=int,   default=8192)
    parser.add_argument("--edge-dropout",  type=float, default=0.0)
    parser.add_argument("--film",          action="store_true", default=False)
    parser.add_argument("--grad-checkpoint", action="store_true", default=False)
    parser.add_argument("--llm-feat-mode", default="fast",
                        choices=["none", "fast", "full"])
    parser.add_argument("--no-nlp-feat",  action="store_true")
    parser.add_argument("--eval-every",   type=int,   default=50)
    parser.add_argument("--no-save",      action="store_true")
    parser.add_argument("--noise-eps",    type=float, default=0.1,
                        help="Per-layer noise magnitude ε (default 0.1)")
    parser.add_argument("--cl-temp",      type=float, default=0.15,
                        help="InfoNCE temperature τ final value (default 0.15)")
    parser.add_argument("--cl-temp-init", type=float, default=0.5,
                        help="InfoNCE temperature τ initial value for curriculum (default 0.5)")
    parser.add_argument("--lambda-bpr",   type=float, default=1.0,
                        help="BPR auxiliary loss weight (default 1.0)")
    parser.add_argument("--hard-neg-refresh", type=int, default=50,
                        help="Refresh hard negative pool every N epochs (0=disabled, default 50)")
    parser.add_argument("--prebuilt-features", action="store_true")
    parser.add_argument("--skip-feature-groups", default="",
                        help="Comma-separated feature groups to drop")
    args, _ = parser.parse_known_args()
    init_distributed()

    skip_groups = [g.strip() for g in args.skip_feature_groups.split(",") if g.strip()] \
        if args.skip_feature_groups else None

    train(
        epochs=args.epochs,
        emb_dim=args.emb_dim,
        n_layers=args.layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        save=not args.no_save,
        grad_checkpoint=args.grad_checkpoint,
        edge_dropout=args.edge_dropout,
        use_film=args.film,
        llm_feat_mode=args.llm_feat_mode,
        use_nlp_feat=not args.no_nlp_feat,
        eval_every=args.eval_every,
        noise_eps=args.noise_eps,
        cl_temp=args.cl_temp,
        cl_temp_init=args.cl_temp_init,
        lambda_bpr=args.lambda_bpr,
        hard_neg_refresh=args.hard_neg_refresh,
        prebuilt_features=args.prebuilt_features,
        skip_feature_groups=skip_groups,
    )


if __name__ == "__main__":
    main()
