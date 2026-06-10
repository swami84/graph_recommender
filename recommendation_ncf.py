#!/usr/bin/env python3
"""
recommendation_ncf.py — Neural Collaborative Filtering (NeuMF) recommendation.

Architecture: NeuMF (He et al. 2017) combining:
  - GMF path: element-wise product of user/item embeddings (linear CF signal)
  - MLP path: deep MLP on concatenated user/item embeddings (non-linear signal)
  - FiLM feature gating for side features (same as LightGCN)

Uses identical data loading, feature pipeline, evaluation, and training setup
as LightGCN for direct apples-to-apples comparison.

Usage:
    python recommendation_ncf.py --epochs 500 --emb-dim 256 --film
    python recommendation_ncf.py --epochs 500 --emb-dim 512 --film --hard-neg-refresh 50
"""

import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recommendation_gnn import (
    FiLMFusion,
    bpr_loss,
    build_hard_neg_pool,
    build_item_features,
    build_user_features,
    load_interactions,
    ndcg_at_k,
    sample_negatives,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ncf")


# ── Model ──────────────────────────────────────────────────────────────────────

class NeuMF(nn.Module):
    """
    Neural Matrix Factorisation combining GMF and MLP paths.

    GMF:  element-wise product of gmf_user_emb × gmf_item_emb → captures
          linear bilinear interactions (equivalent to generalised MF).
    MLP:  concatenation of mlp_user_emb ∥ mlp_item_emb → deep non-linear
          interactions via stacked linear-ReLU layers.
    Output: Linear([gmf_out; mlp_out]) → scalar score.

    Side features modulated via FiLM: γ(feat)·emb + β(feat), initialised
    as identity so the model starts identically to no-feature NeuMF.
    """

    def __init__(
        self,
        n_users:       int,
        n_items:       int,
        emb_dim:       int   = 256,
        mlp_dims:      list  = None,
        dropout:       float = 0.1,
        user_feat_dim: int   = 0,
        item_feat_dim: int   = 0,
        use_film:      bool  = True,
    ):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [512, 256, 128]

        # Separate embedding tables: GMF and MLP paths are independent
        self.gmf_user = nn.Embedding(n_users, emb_dim)
        self.gmf_item = nn.Embedding(n_items, emb_dim)
        self.mlp_user = nn.Embedding(n_users, emb_dim)
        self.mlp_item = nn.Embedding(n_items, emb_dim)

        # FiLM side-feature gating (shared across GMF and MLP paths)
        self.use_film = use_film and (user_feat_dim > 0 or item_feat_dim > 0)
        if use_film and user_feat_dim > 0:
            self.user_film = FiLMFusion(user_feat_dim, emb_dim)
        if use_film and item_feat_dim > 0:
            self.item_film = FiLMFusion(item_feat_dim, emb_dim)

        # MLP tower
        layers: list[nn.Module] = []
        in_dim = emb_dim * 2
        for out_dim in mlp_dims:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # Output: combines GMF element-wise product + MLP final hidden
        self.output_layer = nn.Linear(emb_dim + mlp_dims[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item]:
            nn.init.normal_(emb.weight, std=0.01)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def _apply_film(self, emb: torch.Tensor, feat: torch.Tensor, module: nn.Module):
        return module(emb, feat)

    def _get_user_embs(self, user_ids: torch.Tensor,
                       user_feat: "torch.Tensor | None") -> tuple:
        gmf_u = self.gmf_user(user_ids)
        mlp_u = self.mlp_user(user_ids)
        if user_feat is not None and hasattr(self, "user_film"):
            f = user_feat[user_ids]
            gmf_u = self._apply_film(gmf_u, f, self.user_film)
            mlp_u = self._apply_film(mlp_u, f, self.user_film)
        return gmf_u, mlp_u

    def _get_item_embs(self, item_ids: torch.Tensor,
                       item_feat: "torch.Tensor | None") -> tuple:
        gmf_i = self.gmf_item(item_ids)
        mlp_i = self.mlp_item(item_ids)
        if item_feat is not None and hasattr(self, "item_film"):
            f = item_feat[item_ids]
            gmf_i = self._apply_film(gmf_i, f, self.item_film)
            mlp_i = self._apply_film(mlp_i, f, self.item_film)
        return gmf_i, mlp_i

    def _score(self, gmf_u, mlp_u, gmf_i, mlp_i) -> torch.Tensor:
        gmf_out = gmf_u * gmf_i                             # (B, emb)
        mlp_out = self.mlp(torch.cat([mlp_u, mlp_i], -1))  # (B, mlp_dims[-1])
        return self.output_layer(torch.cat([gmf_out, mlp_out], -1)).squeeze(-1)

    def forward(
        self,
        user_ids:  torch.Tensor,
        pos_ids:   torch.Tensor,
        neg_ids:   torch.Tensor,
        user_feat: "torch.Tensor | None" = None,
        item_feat: "torch.Tensor | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gmf_u, mlp_u = self._get_user_embs(user_ids, user_feat)
        gmf_p, mlp_p = self._get_item_embs(pos_ids,  item_feat)
        gmf_n, mlp_n = self._get_item_embs(neg_ids,  item_feat)
        pos_scores   = self._score(gmf_u, mlp_u, gmf_p, mlp_p)
        neg_scores   = self._score(gmf_u, mlp_u, gmf_n, mlp_n)
        return pos_scores, neg_scores

    @torch.no_grad()
    def score_all_items(
        self,
        user_ids:  torch.Tensor,
        item_feat: "torch.Tensor | None" = None,
        user_feat: "torch.Tensor | None" = None,
        chunk_size: int = 16,
    ) -> torch.Tensor:
        """
        Score user_ids against ALL items. Returns (len(user_ids), n_items) float32.

        Processes in chunks of `chunk_size` users to keep memory bounded.
        With emb_dim=256, chunk_size=16 uses ~300MB on GPU.
        """
        n_items  = self.gmf_item.num_embeddings
        all_ids  = torch.arange(n_items, device=user_ids.device)
        gmf_all, mlp_all = self._get_item_embs(all_ids, item_feat)  # (N, emb)

        scores_list = []
        for s in range(0, len(user_ids), chunk_size):
            e     = min(s + chunk_size, len(user_ids))
            u_ids = user_ids[s:e]
            B     = e - s
            N     = n_items

            gmf_u, mlp_u = self._get_user_embs(u_ids, user_feat)  # (B, emb)

            # GMF: element-wise product → (B, N, emb) — memory = B×N×emb
            gmf_out = gmf_u.unsqueeze(1) * gmf_all.unsqueeze(0)   # (B, N, emb)

            # MLP: concat → (B×N, 2*emb) → MLP → (B, N, mlp_last)
            mlp_u_e = mlp_u.unsqueeze(1).expand(B, N, -1)
            mlp_i_e = mlp_all.unsqueeze(0).expand(B, N, -1)
            mlp_in  = torch.cat([mlp_u_e, mlp_i_e], dim=-1).view(B * N, -1)
            mlp_out = self.mlp(mlp_in).view(B, N, -1)

            combined = torch.cat([gmf_out, mlp_out], dim=-1).view(B * N, -1)
            chunk_scores = self.output_layer(combined).view(B, N)
            scores_list.append(chunk_scores.cpu().float())

        return torch.cat(scores_list, dim=0)  # (n_test_users, n_items)


# ── BPR loss adapted for NeuMF (scores already computed, not dot-product) ─────

def bpr_loss_scores(pos_scores: torch.Tensor, neg_scores: torch.Tensor,
                    model: NeuMF, reg: float = 1e-4) -> torch.Tensor:
    bpr = -F.logsigmoid(pos_scores - neg_scores).mean()
    # L2 reg on embedding weights (not per-batch, scaled by n_params)
    reg_loss = reg * sum(
        p.norm(2).pow(2)
        for name, p in model.named_parameters()
        if "emb" in name and "weight" in name
    ) / (2 * model.gmf_user.num_embeddings)
    return bpr + reg_loss


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:     NeuMF,
    test_df:   "pd.DataFrame",
    excl_df:   "pd.DataFrame",
    n_items:   int,
    user_feat: "torch.Tensor | None",
    item_feat: "torch.Tensor | None",
    k:         int  = 10,
    device:    str  = "cuda",
    eval_chunk: int = 16,
) -> dict:
    import pandas as pd

    model.eval()
    excl_map = excl_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    test_map  = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    test_users = np.array(list(test_map.keys()))

    precision_list, recall_list, ndcg_list = [], [], []
    hits_total = 0
    CHUNK = 512

    for start in range(0, len(test_users), CHUNK):
        chunk     = test_users[start : start + CHUNK]
        chunk_t   = torch.tensor(chunk, dtype=torch.long, device=device)
        scores    = model.score_all_items(chunk_t, item_feat, user_feat,
                                          chunk_size=eval_chunk)  # (B, N) cpu float

        for i, uid in enumerate(chunk):
            relevant = test_map.get(int(uid), set())
            if not relevant:
                continue
            exclude = excl_map.get(int(uid), set())
            row = scores[i].numpy().copy()
            for it in exclude:
                row[it] = -np.inf
            top_idx = np.argsort(-row)[: k + len(exclude)]
            ranked  = [int(x) for x in top_idx if x not in exclude][:k]

            hit = sum(1 for x in ranked if x in relevant)
            hits_total += (hit > 0)
            precision_list.append(hit / k)
            recall_list.append(hit / len(relevant))
            ndcg_list.append(ndcg_at_k(relevant, ranked, k))

    return {
        "precision": float(np.mean(precision_list)),
        "recall":    float(np.mean(recall_list)),
        "ndcg":      float(np.mean(ndcg_list)),
        "hits":      hits_total,
        "n_users":   len(precision_list),
    }


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    epochs:          int   = 500,
    emb_dim:         int   = 256,
    mlp_dims:        list  = None,
    lr:              float = 1e-3,
    batch_size:      int   = 4096,
    dropout:         float = 0.1,
    reg:             float = 1e-4,
    use_film:        bool  = True,
    warm_restart:    bool  = True,
    t0:              int   = 200,
    hard_neg_refresh: int  = 50,
    llm_feat_mode:   str   = "none",
    eval_chunk:      int   = 16,
    save:            bool  = True,
):
    import pandas as pd
    from pathlib import Path

    if mlp_dims is None:
        mlp_dims = [512, 256, 128]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # ── Data ───────────────────────────────────────────────────────────────────
    train_df, test_df, user_enc, item_enc = load_interactions()
    n_users, n_items = len(user_enc), len(item_enc)
    log.info(f"Train {len(train_df):,} | Test {len(test_df):,} | "
             f"Users {n_users:,} | Items {n_items:,}")

    # ── Side features ──────────────────────────────────────────────────────────
    user_feat_np = build_user_features(train_df, user_enc)
    item_feat_np = build_item_features(item_enc, llm_feat_mode=llm_feat_mode)

    user_feat = (torch.tensor(user_feat_np, dtype=torch.float32, device=device)
                 if user_feat_np is not None else None)
    item_feat = (torch.tensor(item_feat_np, dtype=torch.float32, device=device)
                 if item_feat_np is not None else None)

    user_feat_dim = user_feat.shape[1] if user_feat is not None else 0
    item_feat_dim = item_feat.shape[1] if item_feat is not None else 0

    # ── Model ──────────────────────────────────────────────────────────────────
    model = NeuMF(
        n_users=n_users, n_items=n_items,
        emb_dim=emb_dim, mlp_dims=mlp_dims, dropout=dropout,
        user_feat_dim=user_feat_dim, item_feat_dim=item_feat_dim,
        use_film=use_film,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        f"NeuMF | emb_dim={emb_dim} | mlp_dims={mlp_dims} | "
        f"film={use_film} | params={n_params/1e6:.1f}M | "
        f"user_feat={user_feat_dim} | item_feat={item_feat_dim}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if warm_restart:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, T_mult=2
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

    # ── Training loop ──────────────────────────────────────────────────────────
    hard_neg_pool = None

    for epoch in range(1, epochs + 1):
        model.train()

        # Refresh hard negative pool
        if hard_neg_refresh > 0 and epoch % hard_neg_refresh == 1:
            with torch.no_grad():
                i_emb_for_pool = model.gmf_item.weight.detach()
                if item_feat is not None and hasattr(model, "item_film"):
                    all_ids = torch.arange(n_items, device=device)
                    i_emb_for_pool = model.item_film(
                        i_emb_for_pool, item_feat[all_ids]
                    )
            hard_neg_pool = build_hard_neg_pool(i_emb_for_pool)
            log.info(f"  [Epoch {epoch}] Hard neg pool refreshed")

        neg_df  = sample_negatives(train_df, n_items, emb_pool=hard_neg_pool)
        users_t = torch.tensor(neg_df["user_idx"].values, dtype=torch.long, device=device)
        pos_t   = torch.tensor(neg_df["item_idx"].values, dtype=torch.long, device=device)
        neg_t   = torch.tensor(neg_df["neg_idx"].values,  dtype=torch.long, device=device)

        n      = len(users_t)
        perm   = torch.randperm(n, device=device)
        users_t, pos_t, neg_t = users_t[perm], pos_t[perm], neg_t[perm]

        epoch_loss = 0.0
        n_batches  = max(1, n // batch_size)

        for b in range(n_batches):
            s = b * batch_size
            e = min(s + batch_size, n)
            u_b = users_t[s:e]
            p_b = pos_t[s:e]
            n_b = neg_t[s:e]

            optimizer.zero_grad()
            pos_scores, neg_scores = model(u_b, p_b, n_b, user_feat, item_feat)
            loss = bpr_loss_scores(pos_scores, neg_scores, model, reg=reg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if epoch % 10 == 0:
            log.info(f"  Epoch {epoch}/{epochs}  "
                     f"loss={epoch_loss/n_batches:.4f}  "
                     f"lr={scheduler.get_last_lr()[0]:.2e}")

    # ── Evaluation ─────────────────────────────────────────────────────────────
    log.info("Evaluating…")
    metrics = evaluate(
        model, test_df, train_df, n_items,
        user_feat, item_feat,
        device=device, eval_chunk=eval_chunk,
    )

    print(f"\n=== NeuMF Results ===")
    print(f"  Precision@10: {metrics['precision']:.4f}")
    print(f"  Recall@10:    {metrics['recall']:.4f}")
    print(f"  NDCG@10:      {metrics['ndcg']:.4f}")
    print(f"  Hits@10:      {metrics['hits']:,} / {metrics['n_users']:,} users")

    if save:
        Path("models").mkdir(exist_ok=True)
        torch.save(model.state_dict(), "models/ncf.pt")
        log.info("Model saved → models/ncf.pt")

    return metrics


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NeuMF recommendation")
    parser.add_argument("--epochs",           type=int,   default=500)
    parser.add_argument("--emb-dim",          type=int,   default=256)
    parser.add_argument("--mlp-dims",         type=int,   nargs="+", default=[512, 256, 128])
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--batch-size",       type=int,   default=4096)
    parser.add_argument("--dropout",          type=float, default=0.1)
    parser.add_argument("--reg",              type=float, default=1e-4)
    parser.add_argument("--film",             action="store_true")
    parser.add_argument("--warm-restart",     action="store_true", default=True)
    parser.add_argument("--no-warm-restart",  action="store_false", dest="warm_restart")
    parser.add_argument("--t0",               type=int,   default=200)
    parser.add_argument("--hard-neg-refresh", type=int,   default=50)
    parser.add_argument("--llm-feat-mode",    default="none",
                        choices=["none", "fast", "full"])
    parser.add_argument("--eval-chunk",       type=int,   default=16,
                        help="Users per chunk during full-rank evaluation (lower = less VRAM)")
    parser.add_argument("--no-save",          action="store_true")
    args = parser.parse_args()

    train(
        epochs=args.epochs, emb_dim=args.emb_dim, mlp_dims=args.mlp_dims,
        lr=args.lr, batch_size=args.batch_size, dropout=args.dropout,
        reg=args.reg, use_film=args.film,
        warm_restart=args.warm_restart, t0=args.t0,
        hard_neg_refresh=args.hard_neg_refresh,
        llm_feat_mode=args.llm_feat_mode,
        eval_chunk=args.eval_chunk,
        save=not args.no_save,
    )


if __name__ == "__main__":
    main()
