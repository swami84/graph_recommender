#!/usr/bin/env python3
"""
recommendation_dgcf.py — DGCF: Disentangled Graph Collaborative Filtering.

Paper: "Disentangled Graph Collaborative Filtering"
       Wang et al., SIGIR 2020.

Key idea: split the embedding space into K intent chunks.  For each chunk,
learn per-edge attention weights (softmax over K intents) so the graph
propagation decomposes by user intent.  A distance-correlation independence
loss pushes the K intent representations to be orthogonal.

Architecture
  - user_emb / item_emb: (N, emb_dim)
  - emb_dim split into K chunks of d_intent = emb_dim // K
  - For each edge (u, i): intent weight w_k = softmax_k(u_k · i_k)
  - Propagation (1 layer default):
      u_k' = Σ_{j in N(u)} w_k(u,j) · i_k[j] / sqrt(deg_u * deg_j)
      i_k' = Σ_{j in N(i)} w_k(j,i) · u_k[j] / sqrt(deg_i * deg_j)
  - Final: concat over K; layer-pooling over L layers
  - Independence: mean pairwise dCor over K intent vectors on a user batch

Loss:   L = L_BPR + α * L_ind
  L_BPR  = standard Bayesian Personalised Ranking
  L_ind  = mean pairwise distance correlation across intent dimensions

Usage:
    python recommendation_dgcf.py
    python recommendation_dgcf.py --epochs 300 --emb-dim 2048 --n-intents 4 \\
        --layers 1 --batch-size 8192 --ind-weight 0.1 --eval-every 50 --prebuilt-features
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
    bpr_loss,
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
log = logging.getLogger("dgcf")

MODEL_OUT = Path("models/dgcf.pt")
EMB_OUT   = Path("data/embeddings")


# ── Distance Correlation (dCor) ────────────────────────────────────────────────

def _dcor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Unbiased distance correlation between matrices a, b each (B, d).
    Returns scalar in [0, 1].
    """
    def _dcov2(x, y):
        n = x.shape[0]
        dx = torch.cdist(x, x)   # (B, B)
        dy = torch.cdist(y, y)
        # double-centering
        dx = dx - dx.mean(1, keepdim=True) - dx.mean(0, keepdim=True) + dx.mean()
        dy = dy - dy.mean(1, keepdim=True) - dy.mean(0, keepdim=True) + dy.mean()
        return (dx * dy).sum() / (n * n)

    dcov_ab = _dcov2(a, b)
    dcov_aa = _dcov2(a, a)
    dcov_bb = _dcov2(b, b)
    denom   = (dcov_aa * dcov_bb).clamp(min=1e-10).sqrt()
    return dcov_ab / denom


# ── DGCF Model ─────────────────────────────────────────────────────────────────

class DGCF(nn.Module):
    """
    Disentangled Graph Collaborative Filtering.

    Propagation decomposes the embedding into K intent chunks.  Each chunk
    has its own per-edge attention weight (intent-specific softmax).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int          = 64,
        n_intents: int        = 4,
        n_layers: int         = 1,
        user_feat: np.ndarray | None = None,
        item_feat: np.ndarray | None = None,
        grad_checkpoint: bool = True,
        edge_dropout: float   = 0.0,
    ):
        super().__init__()
        assert emb_dim % n_intents == 0, "emb_dim must be divisible by n_intents"
        self.n_users      = n_users
        self.n_items      = n_items
        self.emb_dim      = emb_dim
        self.n_intents    = n_intents
        self.d_intent     = emb_dim // n_intents
        self.n_layers     = n_layers
        self.grad_ckpt    = grad_checkpoint
        self.edge_dropout = edge_dropout

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

    # ── Intent weight computation ──────────────────────────────────────────────

    def _intent_weights(
        self,
        u_emb: torch.Tensor,   # (U, D)
        i_emb: torch.Tensor,   # (I, D)
        edge_u: torch.Tensor,  # (E,) user indices
        edge_i: torch.Tensor,  # (E,) item indices
    ) -> torch.Tensor:
        """
        Compute (E, K) softmax intent weights intent-by-intent to avoid
        materialising the full (E, K, d_intent) tensor.

        For each intent k: w_k = u[edge_u, k*dk:(k+1)*dk] · i[edge_i, k*dk:(k+1)*dk]
        Returns softmax over K dimension.
        """
        dk = self.d_intent
        scores = torch.zeros(edge_u.shape[0], self.n_intents, device=u_emb.device)
        for k in range(self.n_intents):
            s = k * dk
            e = s + dk
            scores[:, k] = (u_emb[edge_u, s:e] * i_emb[edge_i, s:e]).sum(1)
        return F.softmax(scores, dim=1)   # (E, K)

    # ── Single propagation layer ───────────────────────────────────────────────

    def _propagate_step(
        self,
        u_emb: torch.Tensor,     # (U, D)
        i_emb: torch.Tensor,     # (I, D)
        edge_u: torch.Tensor,    # (E,)
        edge_i: torch.Tensor,    # (E,)
        weights: torch.Tensor,   # (E, K)  softmax intent weights
        deg_u: torch.Tensor,     # (U,) user degree vector (for normalisation)
        deg_i: torch.Tensor,     # (I,) item degree vector
    ):
        """
        One DGCF propagation step.

        For each intent k:
          u_k' = Σ_{j in N(u)} w_k(u,j) / sqrt(deg_u * deg_j) * i_k[j]
          i_k' = Σ_{j in N(i)} w_k(j,i) / sqrt(deg_i * deg_j) * u_k[j]

        Symmetrically-normalised: divide by sqrt(deg_u[edge_u] * deg_i[edge_i]).
        """
        dk = self.d_intent
        norm = (deg_u[edge_u] * deg_i[edge_i]).clamp(min=1e-8).sqrt()   # (E,)

        new_u = torch.zeros_like(u_emb)
        new_i = torch.zeros_like(i_emb)

        for k in range(self.n_intents):
            s, e = k * dk, (k + 1) * dk
            wk = (weights[:, k] / norm).unsqueeze(1)   # (E, 1)

            # aggregate items → users
            msg_to_u = wk * i_emb[edge_i, s:e]                       # (E, dk)
            new_u[:, s:e] = new_u[:, s:e] + torch.zeros(
                u_emb.shape[0], dk, device=u_emb.device
            ).scatter_add(0, edge_u.unsqueeze(1).expand_as(msg_to_u), msg_to_u)

            # aggregate users → items
            msg_to_i = wk * u_emb[edge_u, s:e]                       # (E, dk)
            new_i[:, s:e] = new_i[:, s:e] + torch.zeros(
                i_emb.shape[0], dk, device=i_emb.device
            ).scatter_add(0, edge_i.unsqueeze(1).expand_as(msg_to_i), msg_to_i)

        return new_u, new_i

    # ── Multi-layer propagation ────────────────────────────────────────────────

    def _propagate_all(
        self,
        edge_u: torch.Tensor,
        edge_i: torch.Tensor,
        deg_u: torch.Tensor,
        deg_i: torch.Tensor,
        use_ckpt: bool = True,
    ):
        """
        Full DGCF propagation; returns (U+I, D) layer-averaged embeddings.
        Intent weights are computed from detached embeddings (like DGCF paper),
        so their computation does not incur extra gradient graph.
        """
        def _full_pass(u_w, i_w):
            u = u_w
            i = i_w
            if self.user_proj is not None:
                u = u + self.user_proj(self.user_feat)
            if self.item_proj is not None:
                i = i + self.item_proj(self.item_feat)

            # Intent weights from detached embeddings (paper §3.2)
            with torch.no_grad():
                weights = self._intent_weights(u.detach(), i.detach(), edge_u, edge_i)

            u_sum = u
            i_sum = i
            u_cur, i_cur = u, i
            for _ in range(self.n_layers):
                u_cur, i_cur = self._propagate_step(
                    u_cur, i_cur, edge_u, edge_i, weights, deg_u, deg_i
                )
                u_sum = u_sum + u_cur
                i_sum = i_sum + i_cur

            u_out = u_sum / (self.n_layers + 1)
            i_out = i_sum / (self.n_layers + 1)
            return torch.cat([u_out, i_out], dim=0)

        if use_ckpt and self.grad_ckpt and self.training:
            return grad_ckpt(_full_pass, self.user_emb.weight, self.item_emb.weight,
                             use_reentrant=False)
        return _full_pass(self.user_emb.weight, self.item_emb.weight)

    def get_embeddings(
        self,
        edge_u: torch.Tensor,
        edge_i: torch.Tensor,
        deg_u: torch.Tensor,
        deg_i: torch.Tensor,
    ):
        """Clean embeddings for evaluation — no noise, no grad."""
        with torch.no_grad():
            final = self._propagate_all(edge_u, edge_i, deg_u, deg_i, use_ckpt=False)
        return final[:self.n_users], final[self.n_users:]

    # ── Independence loss ──────────────────────────────────────────────────────

    def independence_loss(
        self,
        u_emb: torch.Tensor,   # (U, D) final user embeddings
        u_idx: torch.Tensor,   # (B,) sampled user indices
        ind_batch: int = 2048,
    ) -> torch.Tensor:
        """
        Mean pairwise distance correlation between intent chunks on a user batch.
        Lower dCor → intent representations are more independent.
        """
        dk   = self.d_intent
        K    = self.n_intents
        sub  = u_emb[u_idx[:ind_batch]]   # (B, D)
        chunks = [sub[:, k * dk:(k + 1) * dk] for k in range(K)]

        loss = torch.tensor(0.0, device=u_emb.device)
        n_pairs = 0
        for a in range(K):
            for b in range(a + 1, K):
                loss = loss + _dcor(chunks[a], chunks[b])
                n_pairs += 1
        return loss / max(n_pairs, 1)


# ── Edge index extraction from bipartite adj ──────────────────────────────────

def _extract_edges(adj: torch.Tensor, n_users: int, n_items: int):
    """
    Extract edge_u, edge_i (both 1-D long tensors) and degree vectors from the
    symmetric bipartite adj matrix that build_adj() produces.

    adj is (n_users+n_items) × (n_users+n_items).  Only upper-left block
    (user→item) edges are returned; the graph is undirected so we only need
    the edge set once.
    """
    # COO representation
    adj_coo = adj.coalesce()
    rows    = adj_coo.indices()[0]
    cols    = adj_coo.indices()[1]

    # Keep only user→item edges (upper-left quadrant)
    mask   = (rows < n_users) & (cols >= n_users)
    edge_u = rows[mask]
    edge_i = cols[mask] - n_users

    # Degree from the full adj
    deg_full = torch.zeros(n_users + n_items, device=adj.device)
    deg_full.scatter_add_(0, rows, torch.ones(len(rows), device=adj.device))

    deg_u = deg_full[:n_users].clamp(min=1)
    deg_i = deg_full[n_users:].clamp(min=1)

    return edge_u, edge_i, deg_u, deg_i


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    epochs: int        = 300,
    emb_dim: int       = 2048,
    n_intents: int     = 4,
    n_layers: int      = 1,
    lr: float          = 1e-3,
    batch_size: int    = 8192,
    ind_weight: float  = 0.1,
    ind_batch: int     = 2048,
    save: bool         = True,
    hard_neg_refresh: int = 50,
    llm_feat_mode: str    = "fast",
    use_nlp_feat: bool    = True,
    eval_every: int       = 50,
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

    edge_u, edge_i, deg_u, deg_i = _extract_edges(adj, n_users, n_items)
    log.info(f"Edges: {len(edge_u):,}  |  deg_u mean={deg_u.mean():.1f}  "
             f"deg_i mean={deg_i.mean():.1f}")

    model = DGCF(
        n_users=n_users, n_items=n_items, emb_dim=emb_dim,
        n_intents=n_intents, n_layers=n_layers,
        user_feat=user_feat, item_feat=item_feat,
        grad_checkpoint=True, edge_dropout=0.0,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 50)

    log.info(
        f"Training DGCF | device={DEVICE} | epochs={epochs} | emb_dim={emb_dim} | "
        f"n_intents={n_intents} | d_intent={emb_dim // n_intents} | layers={n_layers} | "
        f"lr={lr} | ind_weight={ind_weight}"
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

        # Refresh hard-negative pool
        if hard_neg_refresh > 0 and epoch > 1 and (epoch - 1) % hard_neg_refresh == 0:
            model.eval()
            with torch.no_grad():
                _, i_emb_eval = model.get_embeddings(edge_u, edge_i, deg_u, deg_i)
            emb_pool = build_hard_neg_pool(i_emb_eval, k=50)
            del i_emb_eval
            model.train()
            log.info(f"  [Epoch {epoch}] Hard neg pool refreshed")
            train_with_neg = sample_negatives(train_df, n_items, emb_pool=emb_pool)
            all_users = torch.tensor(train_with_neg["user_idx"].values, dtype=torch.long)
            all_pos   = torch.tensor(train_with_neg["item_idx"].values, dtype=torch.long)
            all_neg   = torch.tensor(train_with_neg["neg_idx"].values,  dtype=torch.long)

        model.train()
        optimizer.zero_grad()

        # ── Step 1: Independence loss ─────────────────────────────────────────
        ind_val = 0.0
        if ind_weight > 0.0:
            final_ind = model._propagate_all(edge_u, edge_i, deg_u, deg_i, use_ckpt=True)
            u_ind     = final_ind[:n_users]
            sub_u_idx = torch.randperm(n_users, device=DEVICE)[:ind_batch]
            loss_ind  = model.independence_loss(u_ind, sub_u_idx, ind_batch)
            (ind_weight * loss_ind).backward()
            ind_val = loss_ind.item()
            del final_ind, u_ind, loss_ind
            torch.cuda.empty_cache()

        # ── Step 2a: BPR gradient — no_grad chunked scatter_add ──────────────
        with torch.no_grad():
            final_det = model._propagate_all(edge_u, edge_i, deg_u, deg_i, use_ckpt=False)
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

                grad_final[:n_users].scatter_add_(
                    0, bu.unsqueeze(1).expand_as(gu), gu)
                grad_final[n_users:].scatter_add_(
                    0, bp.unsqueeze(1).expand_as(gp), gp)
                grad_final[n_users:].scatter_add_(
                    0, bn.unsqueeze(1).expand_as(gn), gn)

            del final_det, u_det, i_det

        # ── Step 2b: single ckpt forward + one backward with precomputed grad ─
        final_bpr = model._propagate_all(edge_u, edge_i, deg_u, deg_i, use_ckpt=True)
        final_bpr.backward(gradient=grad_final)
        del final_bpr, grad_final
        torch.cuda.empty_cache()

        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            log.info(
                f"  Epoch {epoch}/{epochs}  "
                f"bpr={bpr_val:.4f}  "
                f"ind={ind_val:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if eval_every > 0 and epoch % eval_every == 0:
            model.eval()
            # Wrap get_embeddings to match _eval_recall_at_k signature (expects adj)
            # by temporarily adapting the call
            with torch.no_grad():
                u_emb_eval, i_emb_eval = model.get_embeddings(edge_u, edge_i, deg_u, deg_i)

            # Manual recall evaluation
            import pandas as pd
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
    u_emb, i_emb = model.get_embeddings(edge_u, edge_i, deg_u, deg_i)

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

    print(f"\n=== DGCF Results ===")
    print(f"  Precision@10: {precision:.4f}")
    print(f"  Recall@10:    {recall:.4f}")
    print(f"  NDCG@10:      {ndcg:.4f}")

    skip_str = ",".join(skip_feature_groups) if skip_feature_groups else ""
    variant  = f"K={n_intents} skip {skip_str}" if skip_str else f"K={n_intents}"
    save_result(
        model="DGCF", variant=variant,
        precision=precision, recall=recall, ndcg=ndcg,
        epochs=epochs, emb_dim=emb_dim, n_layers=n_layers,
        skip_groups=skip_str,
    )

    if save:
        MODEL_OUT.parent.mkdir(exist_ok=True)
        EMB_OUT.mkdir(exist_ok=True)
        torch.save(model.state_dict(), MODEL_OUT)
        user_dec = {v: k for k, v in user_enc.items()}
        item_dec = {v: k for k, v in item_enc.items()}
        torch.save({"embeddings": u_emb.cpu(), "id_map": user_dec},
                   EMB_OUT / "dgcf_users.pt")
        torch.save({"embeddings": i_emb.cpu(), "id_map": item_dec},
                   EMB_OUT / "dgcf_items.pt")
        log.info(f"Saved model → {MODEL_OUT}  embeddings → {EMB_OUT}/dgcf_*.pt")

    return recall


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGCF recommendation model")
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--emb-dim",      type=int,   default=2048)
    parser.add_argument("--n-intents",    type=int,   default=4)
    parser.add_argument("--layers",       type=int,   default=1)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--batch-size",   type=int,   default=8192)
    parser.add_argument("--ind-weight",   type=float, default=0.1,
                        help="Weight of independence loss (dCor)")
    parser.add_argument("--ind-batch",    type=int,   default=2048,
                        help="User batch size for dCor computation (O(B^2))")
    parser.add_argument("--hard-neg-refresh", type=int, default=50)
    parser.add_argument("--eval-every",   type=int,   default=50)
    parser.add_argument("--no-save",      action="store_true")
    parser.add_argument("--no-nlp-feat",  action="store_true")
    parser.add_argument("--llm-feat-mode",type=str,   default="fast",
                        choices=["none", "fast", "full"])
    parser.add_argument("--prebuilt-features", action="store_true",
                        help="Load pre-joined features from data/*_features_prebuilt.parquet")
    parser.add_argument("--skip-feature-groups", nargs="*", default=None,
                        metavar="GROUP",
                        help="Feature groups to skip, e.g. --skip-feature-groups nlp extended")
    args = parser.parse_args()

    train(
        epochs              = args.epochs,
        emb_dim             = args.emb_dim,
        n_intents           = args.n_intents,
        n_layers            = args.layers,
        lr                  = args.lr,
        batch_size          = args.batch_size,
        ind_weight          = args.ind_weight,
        ind_batch           = args.ind_batch,
        save                = not args.no_save,
        hard_neg_refresh    = args.hard_neg_refresh,
        llm_feat_mode       = args.llm_feat_mode,
        use_nlp_feat        = not args.no_nlp_feat,
        eval_every          = args.eval_every,
        prebuilt_features   = args.prebuilt_features,
        skip_feature_groups = args.skip_feature_groups,
    )
