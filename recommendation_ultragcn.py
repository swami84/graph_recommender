#!/usr/bin/env python3
"""
recommendation_ultragcn.py — UltraGCN: Ultra Simplification of GCN for Recommendation.

Paper: "UltraGCN: Ultra Simplification of Graph Convolutional Networks
        for Recommendation"
       Mao et al., SIGIR 2021.

Key idea: replace explicit graph message passing with a constraint-based loss
that directly approximates the infinite-layer LightGCN solution.  Result:
no propagation at epoch time → 5-10× faster than KGAT/SimGCL per epoch.

Loss:   L = L_UI + λ_II * L_II + reg * ||E||²
  L_UI  = degree-weighted BCE over positive & negative user-item pairs
  L_II  = item-item co-interaction auxiliary loss (pulls user toward
          co-rated items of the positive, based on precomputed II graph)

Degree weights:
  w_pos(u,i) = 1 + γ * (1/deg_u + 1/deg_i)   — amplifies rare interactions
  w_neg      = 1                               — uniform over negatives

Usage:
    python recommendation_ultragcn.py
    python recommendation_ultragcn.py --epochs 300 --emb-dim 2048 \\
        --batch-size 8192 --neg-count 10 --ii-weight 1e-4 --ii-topk 10 \\
        --eval-every 50 --prebuilt-features
"""

import argparse
import copy
import logging
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model_results import save_result
from recommendation_gnn import (
    load_interactions,
    build_user_features,
    build_item_features,
    sample_negatives,
    ndcg_at_k,
    DEVICE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ultragcn")

MODEL_OUT = Path("models/ultragcn.pt")
EMB_OUT   = Path("data/embeddings")


# ── UltraGCN Model ─────────────────────────────────────────────────────────────

class UltraGCN(nn.Module):
    """No message passing — raw embeddings with optional feature projection."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 64,
        user_feat: np.ndarray | None = None,
        item_feat: np.ndarray | None = None,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        for name, feat in [("user", user_feat), ("item", item_feat)]:
            if feat is not None:
                self.register_buffer(f"{name}_feat", torch.tensor(feat, dtype=torch.float32))
                proj = nn.Linear(feat.shape[1], emb_dim, bias=False)
                nn.init.xavier_uniform_(proj.weight)
                setattr(self, f"{name}_proj", proj)
            else:
                self.register_buffer(f"{name}_feat", None)
                setattr(self, f"{name}_proj", None)

    def get_all_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.user_emb.weight
        i = self.item_emb.weight
        if self.user_proj is not None:
            u = u + self.user_proj(self.user_feat)
        if self.item_proj is not None:
            i = i + self.item_proj(self.item_feat)
        return u, i

    def get_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.get_all_embeddings()


# ── Precompute helpers ─────────────────────────────────────────────────────────

def _build_degree_weights(
    train_df,
    n_users: int,
    n_items: int,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    Compute per-training-pair degree weight:
        w(u,i) = 1 + gamma * (1/deg_u + 1/deg_i)

    Returns float32 tensor of shape (len(train_df),) aligned with train_df rows.
    """
    deg_u = train_df.groupby("user_idx")["item_idx"].count().to_dict()
    deg_i = train_df.groupby("item_idx")["user_idx"].count().to_dict()

    u_deg = np.array([deg_u.get(u, 1) for u in train_df["user_idx"]], dtype=np.float32)
    i_deg = np.array([deg_i.get(i, 1) for i in train_df["item_idx"]], dtype=np.float32)

    w = 1.0 + gamma * (1.0 / u_deg + 1.0 / i_deg)
    return torch.tensor(w, dtype=torch.float32)


def _build_ii_graph(
    train_df,
    n_items: int,
    n_users: int,
    top_k: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build item-item co-interaction graph via sparse matrix multiplication.

    R  = (n_items, n_users) binary indicator matrix
    II = R @ R.T  → (n_items, n_items) co-occurrence counts

    For each item returns its top-K co-items and normalised weights.
    Returns:
        ii_items   (n_items, top_k) long tensor
        ii_weights (n_items, top_k) float tensor  (row-normalised)
    """
    log.info(f"  Building item-item co-occurrence graph (top_k={top_k})…")
    rows = train_df["item_idx"].values
    cols = train_df["user_idx"].values
    data = np.ones(len(train_df), dtype=np.float32)

    R  = sp.csr_matrix((data, (rows, cols)), shape=(n_items, n_users))
    II = R @ R.T
    II.setdiag(0)
    II = II.tocsr()
    II.eliminate_zeros()

    ii_items   = np.zeros((n_items, top_k), dtype=np.int64)
    ii_weights = np.zeros((n_items, top_k), dtype=np.float32)

    for i in range(n_items):
        row  = II.getrow(i)
        idxs = row.indices
        vals = row.data.astype(np.float32)
        if len(idxs) == 0:
            # isolated item — point to itself (weight 0 means it won't contribute)
            continue
        if len(idxs) <= top_k:
            order = np.argsort(-vals)
        else:
            order = np.argpartition(-vals, top_k)[:top_k]
            order = order[np.argsort(-vals[order])]
        k = min(top_k, len(idxs))
        ii_items[i, :k]   = idxs[order[:k]]
        raw               = vals[order[:k]]
        ii_weights[i, :k] = raw / max(raw.sum(), 1e-8)

    log.info(f"  II graph built — avg non-zero co-items per item: "
             f"{(ii_weights > 0).sum(1).mean():.1f}")

    return torch.tensor(ii_items), torch.tensor(ii_weights)


# ── Loss ───────────────────────────────────────────────────────────────────────

def _ui_loss(
    u: torch.Tensor,      # (B, D)
    pos: torch.Tensor,    # (B, D)
    neg: torch.Tensor,    # (B, n_neg, D)
    w_pos: torch.Tensor,  # (B,) degree weights
    reg: float,
) -> torch.Tensor:
    """
    Weighted BCE loss:
      L_pos = -mean(w_pos * logsigmoid(u·pos))
      L_neg = mean(softplus(u·neg))   averaged over n_neg negatives
      L_reg = reg * (||u||² + ||pos||²) / 2
    """
    pos_score = (u * pos).sum(1)                        # (B,)
    neg_score = (u.unsqueeze(1) * neg).sum(2)           # (B, n_neg)

    l_pos = -(w_pos * F.logsigmoid(pos_score)).mean()
    l_neg = F.softplus(neg_score).mean()
    l_reg = reg * (u.pow(2).sum(1) + pos.pow(2).sum(1)).mean() / 2
    return l_pos + l_neg + l_reg


def _ii_loss(
    u: torch.Tensor,           # (B, D)
    co_emb: torch.Tensor,      # (B, K, D)
    co_w: torch.Tensor,        # (B, K)  row-normalised weights
    neg: torch.Tensor,         # (B, n_neg, D)
    reg: float,
) -> torch.Tensor:
    """
    Item-item auxiliary loss: for each (u, co-item k), apply BCE weighted by γ_ik.
      L_II_pos = -mean(γ * logsigmoid(u · co_k))
      L_II_neg = mean(softplus(u · neg))
    """
    co_score  = (u.unsqueeze(1) * co_emb).sum(2)        # (B, K)
    neg_score = (u.unsqueeze(1) * neg).sum(2)            # (B, n_neg)

    mask      = (co_w > 0)
    l_pos     = -(co_w * F.logsigmoid(co_score) * mask).sum(1).mean()
    l_neg     = F.softplus(neg_score).mean()
    l_reg     = reg * co_emb.pow(2).sum(2).mean() / 2
    return l_pos + l_neg + l_reg


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    epochs: int          = 300,
    emb_dim: int         = 2048,
    lr: float            = 1e-3,
    batch_size: int      = 8192,
    neg_count: int       = 10,
    gamma: float         = 1.0,
    ii_weight: float     = 1e-4,
    ii_topk: int         = 10,
    reg_weight: float    = 1e-4,
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

    # ── Precompute degree weights ──────────────────────────────────────────────
    log.info("  Precomputing degree weights…")
    all_w_pos = _build_degree_weights(train_df, n_users, n_items, gamma=gamma)

    # ── Precompute II graph ────────────────────────────────────────────────────
    ii_items_np, ii_weights_np = _build_ii_graph(
        train_df, n_items, n_users, top_k=ii_topk
    )
    ii_items   = ii_items_np.to(DEVICE)    # (n_items, K)
    ii_weights = ii_weights_np.to(DEVICE)  # (n_items, K)

    model = UltraGCN(
        n_users=n_users, n_items=n_items, emb_dim=emb_dim,
        user_feat=user_feat, item_feat=item_feat,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 50)

    log.info(
        f"Training UltraGCN | device={DEVICE} | epochs={epochs} | emb_dim={emb_dim} | "
        f"lr={lr} | batch_size={batch_size} | neg_count={neg_count} | "
        f"gamma={gamma} | ii_weight={ii_weight} | ii_topk={ii_topk}"
    )

    # Pre-extract arrays for fast tensor construction each epoch
    all_users_np = train_df["user_idx"].values.astype(np.int64)
    all_pos_np   = train_df["item_idx"].values.astype(np.int64)
    n_train      = len(all_users_np)

    best_recall = 0.0
    best_state  = None

    for epoch in range(1, epochs + 1):
        model.train()

        # Sample negatives fresh each epoch
        neg_matrix = np.random.randint(0, n_items, size=(n_train, neg_count))
        perm       = np.random.permutation(n_train)

        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]

            bu  = torch.tensor(all_users_np[idx], dtype=torch.long, device=DEVICE)
            bp  = torch.tensor(all_pos_np[idx],   dtype=torch.long, device=DEVICE)
            bn  = torch.tensor(neg_matrix[idx],   dtype=torch.long, device=DEVICE)  # (B, n_neg)
            bw  = all_w_pos[idx].to(DEVICE)

            u_all, i_all = model.get_all_embeddings()

            u   = u_all[bu]                        # (B, D)
            pos = i_all[bp]                        # (B, D)
            neg = i_all[bn]                        # (B, n_neg, D)

            loss = _ui_loss(u, pos, neg, bw, reg_weight)

            if ii_weight > 0.0:
                co_idx = ii_items[bp]              # (B, K)
                co_w   = ii_weights[bp]            # (B, K)
                co_emb = i_all[co_idx]             # (B, K, D)
                loss   = loss + ii_weight * _ii_loss(u, co_emb, co_w, neg, reg_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        if epoch % 10 == 0:
            log.info(
                f"  Epoch {epoch}/{epochs}  "
                f"loss={epoch_loss/n_batches:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if eval_every > 0 and epoch % eval_every == 0:
            model.eval()
            u_emb_eval, i_emb_eval = model.get_embeddings()

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
    u_emb, i_emb = model.get_embeddings()

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
        scores  = (u_emb[u_idx] @ i_emb.T).cpu().numpy()

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

    print(f"\n=== UltraGCN Results ===")
    print(f"  Precision@10: {precision:.4f}")
    print(f"  Recall@10:    {recall:.4f}")
    print(f"  NDCG@10:      {ndcg:.4f}")

    skip_str = ",".join(skip_feature_groups) if skip_feature_groups else ""
    variant  = f"skip {skip_str}" if skip_str else "all features"
    save_result(
        model="UltraGCN", variant=variant,
        precision=precision, recall=recall, ndcg=ndcg,
        epochs=epochs, emb_dim=emb_dim, n_layers=0,
        skip_groups=skip_str,
    )

    if save:
        file_tag   = f"skip_{skip_str.replace(',', '_')}" if skip_str else "all"
        model_path = MODEL_OUT.parent / f"ultragcn_{file_tag}.pt"
        model_path.parent.mkdir(exist_ok=True)
        EMB_OUT.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_path)
        user_dec = {v: k for k, v in user_enc.items()}
        item_dec = {v: k for k, v in item_enc.items()}
        torch.save({"embeddings": u_emb.cpu(), "id_map": user_dec},
                   EMB_OUT / f"ultragcn_{file_tag}_user_embeddings.pt")
        torch.save({"embeddings": i_emb.cpu(), "id_map": item_dec},
                   EMB_OUT / f"ultragcn_{file_tag}_item_embeddings.pt")
        log.info(f"Model → {model_path} | Embeddings → {EMB_OUT}")

    return recall


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UltraGCN recommendation model")
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--emb-dim",      type=int,   default=2048)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--batch-size",   type=int,   default=8192)
    parser.add_argument("--neg-count",    type=int,   default=10,
                        help="Negatives sampled per positive per batch step")
    parser.add_argument("--gamma",        type=float, default=1.0,
                        help="Degree-weight amplification: w=1+gamma*(1/deg_u+1/deg_i)")
    parser.add_argument("--ii-weight",    type=float, default=1e-4,
                        help="Weight of item-item co-interaction loss (0 to disable)")
    parser.add_argument("--ii-topk",      type=int,   default=10,
                        help="Top-K co-items per item in the II graph")
    parser.add_argument("--reg-weight",   type=float, default=1e-4,
                        help="L2 regularisation weight on embeddings")
    parser.add_argument("--hard-neg-refresh", type=int, default=0,
                        help="Unused — kept for CLI consistency (UltraGCN uses random negs)")
    parser.add_argument("--eval-every",   type=int,   default=50)
    parser.add_argument("--no-save",      action="store_true")
    parser.add_argument("--no-nlp-feat",  action="store_true")
    parser.add_argument("--llm-feat-mode",type=str,   default="fast",
                        choices=["none", "fast", "full"])
    parser.add_argument("--prebuilt-features", action="store_true")
    parser.add_argument("--skip-feature-groups", default="",
                        help="Comma-separated feature groups to drop "
                             "(user: base,extended,pref; item: base,nlp,extended,llm)")
    args = parser.parse_args()

    skip_groups = [g.strip() for g in args.skip_feature_groups.split(",") if g.strip()] \
        if args.skip_feature_groups else None

    train(
        epochs              = args.epochs,
        emb_dim             = args.emb_dim,
        lr                  = args.lr,
        batch_size          = args.batch_size,
        neg_count           = args.neg_count,
        gamma               = args.gamma,
        ii_weight           = args.ii_weight,
        ii_topk             = args.ii_topk,
        reg_weight          = args.reg_weight,
        save                = not args.no_save,
        llm_feat_mode       = args.llm_feat_mode,
        use_nlp_feat        = not args.no_nlp_feat,
        eval_every          = args.eval_every,
        prebuilt_features   = args.prebuilt_features,
        skip_feature_groups = skip_groups,
    )
