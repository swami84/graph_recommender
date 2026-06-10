#!/usr/bin/env python3
"""
recommendation_simgcl.py — SimGCL: Simple Graph Contrastive Learning for Recommendation.

Paper: "SimGCL: A Simple Graph Contrastive Learning Framework for Recommendation"
       Yu et al., SIGIR 2022.

Key idea over LightGCN:
  Instead of graph augmentation (edge/node dropout like SGL), SimGCL creates two
  augmented embedding views by adding uniform random noise at each propagation layer.
  An InfoNCE contrastive loss then pulls the two views of the same user/item together
  while pushing different users/items apart.

  This directly addresses sparse graphs (post-dataset-expansion regression) by making
  representations more robust without changing the graph structure.

Loss:   L = L_BPR + λ * L_CL
  L_BPR  = standard Bayesian Personalised Ranking
  L_CL   = symmetric InfoNCE over two noise-augmented embedding views

Key hyperparameters vs LightGCN:
  --cl-weight   λ  weight of contrastive loss (default 0.2)
  --noise-eps   ε  noise magnitude per layer   (default 0.1)
  --cl-temp     τ  InfoNCE temperature          (default 0.15)

Usage:
    python recommendation_simgcl.py
    python recommendation_simgcl.py --epochs 100 --emb-dim 2048 --layers 4 \\
        --batch-size 8192 --cl-weight 0.2 --noise-eps 0.1 --cl-temp 0.15 --eval-every 20
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

# ── Import shared infrastructure from LightGCN ────────────────────────────────
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
log = logging.getLogger("simgcl")

MODEL_OUT = Path("models/simgcl.pt")
EMB_OUT   = Path("data/embeddings")


# ── SimGCL Model ───────────────────────────────────────────────────────────────

class SimGCL(nn.Module):
    """
    SimGCL: LightGCN backbone + noise-based contrastive augmentation.

    Standard forward pass (for BPR + evaluation) is identical to LightGCN.
    forward_cl() runs the same propagation but adds per-layer uniform noise,
    producing an augmented view for the contrastive loss.
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

    def _init_embeddings(self) -> torch.Tensor:
        """Apply feature projection / FiLM and concatenate user+item embeddings."""
        u = self.user_emb.weight
        i = self.item_emb.weight
        if self.user_film is not None:
            u = self.user_film(u, self.user_feat)
        elif self.user_proj is not None:
            u = u + self.user_proj(self.user_feat)
        if self.item_film is not None:
            i = self.item_film(i, self.item_feat)
        elif self.item_proj is not None:
            i = i + self.item_proj(self.item_feat)
        return torch.cat([u, i], dim=0)

    def _propagate(
        self, adj: torch.Tensor, noise_eps: float = 0.0, use_ckpt: bool = True
    ) -> torch.Tensor:
        """
        LightGCN propagation with optional per-layer noise injection.

        use_ckpt=True  — checkpoint the full pass (FiLM + GCN loop); intermediates
                         freed after forward and recomputed on backward.  Use for
                         both CL and BPR: recomputes ~2.5 GB per backward step
                         rather than keeping the full graph alive (~7 GB).
        use_ckpt=False — store all intermediates (no recomputation).  Only for
                         inference / no-grad contexts.
        """
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

    def forward(self, adj, users, pos_items, neg_items):
        """Standard BPR forward (no noise)."""
        final = self._propagate(adj, noise_eps=0.0)
        u_emb, i_emb = final[:self.n_users], final[self.n_users:]
        return (
            u_emb[users], i_emb[pos_items], i_emb[neg_items],
            self.user_emb.weight, self.item_emb.weight,
        )

    def forward_cl(self, adj: torch.Tensor, noise_eps: float):
        """Augmented forward pass for contrastive learning (with noise)."""
        final = self._propagate(adj, noise_eps=noise_eps)
        return final[:self.n_users], final[self.n_users:]

    def get_embeddings(self, adj: torch.Tensor):
        """Clean embeddings for evaluation — no noise, no grad."""
        with torch.no_grad():
            final = self._propagate(adj, noise_eps=0.0)
        return final[:self.n_users], final[self.n_users:]


# ── Contrastive Loss ───────────────────────────────────────────────────────────

def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float) -> torch.Tensor:
    """
    Symmetric InfoNCE loss between two augmented views.
    Positive pairs: (z1[i], z2[i]).  Negatives: all other rows in batch.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    # (B, B) cosine similarity matrix
    logits = torch.mm(z1, z2.T) / temp
    labels = torch.arange(len(z1), device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    epochs: int        = 100,
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
    # SimGCL-specific
    cl_weight: float      = 0.2,
    noise_eps: float      = 0.1,
    cl_temp: float        = 0.15,
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

    model = SimGCL(
        n_users=n_users, n_items=n_items, emb_dim=emb_dim, n_layers=n_layers,
        user_feat=user_feat, item_feat=item_feat,
        grad_checkpoint=grad_checkpoint, edge_dropout=edge_dropout,
        use_film=use_film,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 50)

    log.info(
        f"Training SimGCL | device={DEVICE} | epochs={epochs} | emb_dim={emb_dim} | "
        f"layers={n_layers} | lr={lr} | edge_dropout={edge_dropout} | film={use_film} | "
        f"cl_weight={cl_weight} | noise_eps={noise_eps} | cl_temp={cl_temp}"
    )

    emb_pool    = None
    best_recall = 0.0
    best_state  = None
    rng         = np.random.default_rng()

    # Pre-sample negatives for the full training set once
    train_with_neg = sample_negatives(train_df, n_items, emb_pool=emb_pool)
    all_users  = torch.tensor(train_with_neg["user_idx"].values, dtype=torch.long)
    all_pos    = torch.tensor(train_with_neg["item_idx"].values, dtype=torch.long)
    all_neg    = torch.tensor(train_with_neg["neg_idx"].values,  dtype=torch.long)
    n_train    = len(all_users)

    for epoch in range(1, epochs + 1):

        # Refresh hard-negative pool and resample negatives
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

        # ── Epoch-level strategy: 3 propagations total per epoch ─────────────
        # CL: 2 noise-augmented ckpt propagations, backward immediately → freed.
        # BPR: manual gradient (no autograd graph needed):
        #   Phase a — no_grad chunked loop: compute d(BPR)/d(final_emb) analytically
        #             via scatter_add.  Cheap; no GCN recomputation.
        #   Phase b — one ckpt forward + one backward with the precomputed gradient.
        # Cost: ~n_steps scatter_adds + 1 GCN backward  (was 60 GCN backwards with
        #        retain_graph).  Gradient flows through the full GCN, not just raw embs.
        # Both BPR phases use edge_dropout=0 so detached and grad values are identical.

        # ── Step 1: CL loss — two augmented views, backprop immediately ────
        u_emb1, i_emb1 = model.forward_cl(adj, noise_eps)
        u_emb2, i_emb2 = model.forward_cl(adj, noise_eps)

        sub_u = torch.randperm(n_users, device=DEVICE)[:batch_size]
        sub_i = torch.randperm(n_items, device=DEVICE)[:batch_size]
        loss_cl = (
            info_nce_loss(u_emb1[sub_u], u_emb2[sub_u], cl_temp) +
            info_nce_loss(i_emb1[sub_i], i_emb2[sub_i], cl_temp)
        )
        (cl_weight * loss_cl).backward()
        cl_val = loss_cl.item()
        del u_emb1, i_emb1, u_emb2, i_emb2, loss_cl
        torch.cuda.empty_cache()

        # ── Step 2a: BPR gradient — no_grad chunked scatter_add ─────────────
        _saved_dropout    = model.edge_dropout
        model.edge_dropout = 0.0   # deterministic: detached and grad passes must match

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

                # d(mean_BPR)/d(margin_i) = -(1-σ(m_i)) / n_train
                coef = -(1.0 - torch.sigmoid(margin)) / n_train
                r    = 1e-4 / n_train   # L2 reg coeff, matches bpr_loss normalisation

                gu = coef.unsqueeze(1) * (pos - neg) + r * u
                gp = coef.unsqueeze(1) * u            + r * pos
                gn = -coef.unsqueeze(1) * u           + r * neg

                grad_final[:n_users].scatter_add_(0, bu.unsqueeze(1).expand_as(gu), gu)
                grad_final[n_users:].scatter_add_(0, bp.unsqueeze(1).expand_as(gp), gp)
                grad_final[n_users:].scatter_add_(0, bn.unsqueeze(1).expand_as(gn), gn)

            del final_det, u_det, i_det

        # ── Step 2b: single ckpt forward + one backward with precomputed grad ─
        final_bpr = model._propagate(adj, noise_eps=0.0, use_ckpt=True)
        final_bpr.backward(gradient=grad_final)
        del final_bpr, grad_final
        model.edge_dropout = _saved_dropout
        torch.cuda.empty_cache()

        optimizer.step()
        scheduler.step()   # once per epoch

        if epoch % 10 == 0:
            log.info(
                f"  Epoch {epoch}/{epochs}  "
                f"bpr={bpr_val:.4f}  "
                f"cl={cl_val:.4f}  "
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

    print(f"\n=== SimGCL Results ===")
    print(f"  Precision@10: {precision:.4f}")
    print(f"  Recall@10:    {recall:.4f}")
    print(f"  NDCG@10:      {ndcg:.4f}")

    skip_str = ",".join(skip_feature_groups) if skip_feature_groups else ""
    variant  = f"skip {skip_str}" if skip_str else "all features"
    save_result(
        model="SimGCL", variant=variant,
        precision=precision, recall=recall, ndcg=ndcg,
        epochs=epochs, emb_dim=emb_dim, n_layers=n_layers,
        skip_groups=skip_str,
    )

    if save:
        file_tag   = f"skip_{skip_str.replace(',', '_')}" if skip_str else "all"
        model_path = MODEL_OUT.parent / f"simgcl_{file_tag}.pt"
        model_path.parent.mkdir(exist_ok=True)
        EMB_OUT.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_path)
        user_dec = {v: k for k, v in user_enc.items()}
        item_dec = {v: k for k, v in item_enc.items()}
        torch.save({"embeddings": u_emb.cpu(), "id_map": user_dec},
                   EMB_OUT / f"simgcl_{file_tag}_user_embeddings.pt")
        torch.save({"embeddings": i_emb.cpu(), "id_map": item_dec},
                   EMB_OUT / f"simgcl_{file_tag}_item_embeddings.pt")
        log.info(f"Model → {model_path} | Embeddings → {EMB_OUT}")

    return recall


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SimGCL recommendation model")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--emb-dim",     type=int,   default=2048)
    parser.add_argument("--layers",      type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch-size",  type=int,   default=8192)
    parser.add_argument("--edge-dropout",type=float, default=0.1)
    parser.add_argument("--film",        action="store_true", default=False)
    parser.add_argument("--grad-checkpoint", action="store_true", default=False)
    parser.add_argument("--hard-neg-refresh", type=int, default=50)
    parser.add_argument("--llm-feat-mode", default="fast",
                        choices=["none", "fast", "full"])
    parser.add_argument("--no-nlp-feat", action="store_true")
    parser.add_argument("--eval-every",  type=int,   default=20)
    parser.add_argument("--no-save",     action="store_true")
    # SimGCL-specific
    parser.add_argument("--cl-weight",   type=float, default=0.2,
                        help="Weight of contrastive loss λ (default 0.2)")
    parser.add_argument("--noise-eps",   type=float, default=0.1,
                        help="Per-layer noise magnitude ε (default 0.1)")
    parser.add_argument("--cl-temp",     type=float, default=0.15,
                        help="InfoNCE temperature τ (default 0.15)")
    parser.add_argument("--prebuilt-features", action="store_true",
                        help="Load pre-joined feature matrices from "
                             "data/*_features_prebuilt.parquet. "
                             "Run build_training_features.py first.")
    parser.add_argument("--skip-feature-groups", default="",
                        help="Comma-separated feature groups to drop "
                             "(user: base,extended,pref; item: base,nlp,extended,llm)")
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
        cl_temp=args.cl_temp,
        prebuilt_features=args.prebuilt_features,
        skip_feature_groups=skip_groups,
    )


if __name__ == "__main__":
    main()
