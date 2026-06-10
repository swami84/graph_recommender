#!/usr/bin/env python3
"""
recommendation_ifl_gcl.py — IFL-GCL: InfoNCE as a Free Lunch for Semantically
guided Graph Contrastive Learning applied to restaurant recommendation.

Paper: "InfoNCE is a Free Lunch for Semantically guided Graph Contrastive
        Learning" — Wang et al., SIGIR 2025.

Core idea over SimGCL:
  Standard GCL treats every non-augmented node as a negative sample, causing
  sampling bias when semantically similar nodes (same cuisine, taste profile,
  neighbourhood) are pushed apart. IFL-GCL reframes GCL as Positive-Unlabeled
  (PU) learning: augmented pairs are confirmed positives (D_L^+) while some
  non-augmented pairs are unlabeled positives (D_U^+). InfoNCE's similarity
  score is proportional to the density ratio r(x) = p(x|y=+1)/p(x), providing
  a "free lunch" to identify D_U^+ without extra computation.

  Algorithm (Algorithm 1 from the paper):
    1. Warm up M epochs with standard SimGCL (BPR + symmetric InfoNCE).
    2. Every K epochs: use current embeddings to mine D_U^+
         — user-user pairs with cosine similarity > t_s
         — item-item pairs with cosine similarity > t_s
    3. Train with corrected loss (Eq. 27):
         L^corr = E_{D_L^+} [-log(P_{n,n'} * prod_{D_U^+} P_{n,n''}^{β·s̃(n,n'')})]
               ≈ L_InfoNCE_standard + β * weighted_InfoNCE_for_D_U+

Loss:  L = L_BPR + λ * L_CL^corrected
Views: two LightGCN propagations with per-layer uniform noise (same as SimGCL)

New hyperparameters vs SimGCL:
  --warmup-epochs   M   epochs of standard SimGCL before mining starts (default 50)
  --update-interval K   re-mine D_U^+ every K epochs                   (default 50)
  --threshold       t_s cosine similarity threshold for D_U^+           (default 0.90)
  --beta                weight exponent for D_U^+ correction             (default 1.0)
  --top-k               max similar neighbours mined per node            (default 20)

Usage:
    python recommendation_ifl_gcl.py
    python recommendation_ifl_gcl.py --epochs 300 --emb-dim 2048 --layers 4 \\
        --warmup-epochs 50 --update-interval 50 --threshold 0.90 --beta 1.0 \\
        --cl-weight 0.2 --noise-eps 0.1 --cl-temp 0.15 --eval-every 50
"""

import argparse
import copy
import logging
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
    _sparse_mm_f32,
    dropout_adj,
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
log = logging.getLogger("ifl_gcl")

MODEL_OUT = Path("models/ifl_gcl.pt")
EMB_OUT   = Path("data/embeddings")
PRED_DIR  = Path("data/predictions")


# ── Model (identical backbone to SimGCL) ──────────────────────────────────────

class IFLGCLModel(nn.Module):
    """
    LightGCN backbone with noise-based contrastive augmentation.
    Architecture is identical to SimGCL; IFL-GCL modifies only the loss.
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
            return grad_ckpt(
                _full_pass, self.user_emb.weight, self.item_emb.weight,
                use_reentrant=False,
            )
        return _full_pass(self.user_emb.weight, self.item_emb.weight)

    def forward(self, adj, users, pos_items, neg_items):
        final = self._propagate(adj, noise_eps=0.0)
        u_emb, i_emb = final[:self.n_users], final[self.n_users:]
        return (
            u_emb[users], i_emb[pos_items], i_emb[neg_items],
            self.user_emb.weight, self.item_emb.weight,
        )

    def forward_cl(self, adj: torch.Tensor, noise_eps: float):
        final = self._propagate(adj, noise_eps=noise_eps)
        return final[:self.n_users], final[self.n_users:]

    def get_embeddings(self, adj: torch.Tensor):
        with torch.no_grad():
            final = self._propagate(adj, noise_eps=0.0)
        return final[:self.n_users], final[self.n_users:]


# ── Standard InfoNCE ───────────────────────────────────────────────────────────

def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float) -> torch.Tensor:
    """Symmetric InfoNCE. Positive pairs: (z1[i], z2[i])."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.T) / temp
    labels = torch.arange(len(z1), device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ── IFL-GCL: D_U^+ mining ─────────────────────────────────────────────────────

def mine_unlabeled_positives(
    z: torch.Tensor,
    threshold: float = 0.90,
    top_k: int = 20,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Mine same-type unlabeled positive pairs (D_U^+) by thresholding cosine
    similarity in the current embedding space (the InfoNCE 'free lunch').

    Chunk-based to avoid materialising the full N×N similarity matrix on GPU.

    Returns (src, dst, weights) where weights are min-max normalised similarity
    scores s̃(n, n'') as in Eq.26 of the paper, or None if no pairs found.
    """
    z_norm = F.normalize(z, dim=1)
    N = len(z_norm)
    src_list, dst_list, sim_list = [], [], []

    for start in range(0, N, chunk_size):
        end   = min(start + chunk_size, N)
        chunk = z_norm[start:end]                        # (C, d)
        sims  = chunk @ z_norm.T                         # (C, N)

        # Exclude self-similarity
        for local_i, global_i in enumerate(range(start, end)):
            sims[local_i, global_i] = -1.0

        k      = min(top_k, N - 1)
        vals, idxs = sims.topk(k, dim=1)                # (C, k)
        mask   = vals >= threshold
        if not mask.any():
            continue

        anchors = (
            torch.arange(start, end, device=z.device)
            .unsqueeze(1)
            .expand_as(mask)
        )
        src_list.append(anchors[mask])
        dst_list.append(idxs[mask])
        sim_list.append(vals[mask])

    if not src_list:
        return None

    src  = torch.cat(src_list)
    dst  = torch.cat(dst_list)
    sims = torch.cat(sim_list)

    s_min = sims.min()
    s_max = sims.max()
    weights = (sims - s_min) / (s_max - s_min + 1e-8)   # Eq.26

    log.info(
        f"    D_U^+ mined: {len(src):,} pairs "
        f"(threshold={threshold}, sim range [{s_min:.3f}, {s_max:.3f}])"
    )
    return src, dst, weights


# ── IFL-GCL: corrected InfoNCE loss ──────────────────────────────────────────

def corrected_cl_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    node_ids: torch.Tensor,
    unlab_src: torch.Tensor | None,
    unlab_dst: torch.Tensor | None,
    unlab_weights: torch.Tensor | None,
    temp: float,
    beta: float,
) -> torch.Tensor:
    """
    Corrected InfoNCE loss (Eq.27).  For each anchor n in the batch:
      L^corr_n = -log P(n,n') - β * Σ_{n''∈D_U^+(n)} s̃(n,n'') * log P(n,n'')

    where P(n,n') = softmax of the (B×B) cosine similarity matrix.

    D_U^+ pairs outside the current batch contribute nothing (they would
    require a separate forward pass and are omitted for efficiency).
    """
    z1n = F.normalize(z1, dim=1)
    z2n = F.normalize(z2, dim=1)
    logits = z1n @ z2n.T / temp                          # (B, B)
    labels = torch.arange(len(z1), device=z1.device)
    loss   = (
        F.cross_entropy(logits, labels) +
        F.cross_entropy(logits.T, labels)
    ) / 2

    if unlab_src is None or len(unlab_src) == 0:
        return loss

    # Build global-id → local-batch-position mapping
    B      = len(z1)
    max_id = max(node_ids.max().item(), unlab_src.max().item(),
                 unlab_dst.max().item()) + 1
    id_map = torch.full((max_id,), -1, dtype=torch.long, device=z1.device)
    id_map[node_ids] = torch.arange(B, device=z1.device)

    # Filter D_U^+ to pairs where both endpoints are in this batch
    in_range = (unlab_src < max_id) & (unlab_dst < max_id)
    if not in_range.any():
        return loss

    s_filt = unlab_src[in_range]
    d_filt = unlab_dst[in_range]
    w_filt = unlab_weights[in_range]

    pos_s  = id_map[s_filt]
    pos_d  = id_map[d_filt]
    valid  = (pos_s >= 0) & (pos_d >= 0)
    if not valid.any():
        return loss

    pos_s = pos_s[valid]
    pos_d = pos_d[valid]
    w     = w_filt[valid]

    log_probs  = F.log_softmax(logits, dim=1)            # (B, B)
    correction = -(beta * w * log_probs[pos_s, pos_d]).mean()

    return loss + correction


# ── Prediction saving ──────────────────────────────────────────────────────────

def save_predictions(
    u_emb: torch.Tensor,
    i_emb: torch.Tensor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_enc: dict,
    item_enc: dict,
    file_tag: str,
) -> None:
    """Generate top-10 predictions for all test users and write to parquet."""
    user_id_map = {v: k for k, v in user_enc.items()}
    item_id_map = {v: k for k, v in item_enc.items()}

    train_sets = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    test_users = test_df["user_idx"].values
    test_items = test_df["item_idx"].values

    u_emb_cpu = u_emb.cpu().float()
    i_emb_cpu = i_emb.cpu().float()

    records = []
    for start in range(0, len(test_users), 2048):
        end    = min(start + 2048, len(test_users))
        u_idx  = test_users[start:end]
        i_idx  = test_items[start:end]
        scores = (u_emb_cpu[u_idx] @ i_emb_cpu.T).numpy()

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
    out_path = PRED_DIR / f"ifl_gcl_{file_tag}_predictions.parquet"
    pred_df.to_parquet(out_path, index=False)
    hit_rate = pred_df["rank"].notna().mean()
    log.info(f"Predictions → {out_path}  ({len(pred_df):,} rows | Hit@10={hit_rate:.4f})")


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    epochs: int             = 300,
    emb_dim: int            = 2048,
    n_layers: int           = 4,
    lr: float               = 1e-3,
    batch_size: int         = 8192,
    save: bool              = True,
    grad_checkpoint: bool   = False,
    edge_dropout: float     = 0.0,
    use_film: bool          = False,
    hard_neg_refresh: int   = 50,
    llm_feat_mode: str      = "fast",
    use_nlp_feat: bool      = True,
    eval_every: int         = 50,
    prebuilt_features: bool = False,
    skip_feature_groups: list[str] | None = None,
    # SimGCL base hyperparams
    cl_weight: float        = 0.2,
    noise_eps: float        = 0.1,
    cl_temp: float          = 0.15,
    # IFL-GCL hyperparams
    warmup_epochs: int      = 50,
    update_interval: int    = 50,
    threshold: float        = 0.90,
    beta: float             = 1.0,
    top_k: int              = 20,
    mine_chunk: int         = 4096,
):
    train_df, test_df, user_enc, item_enc = load_interactions()
    n_users, n_items = len(user_enc), len(item_enc)
    log.info(
        f"Train {len(train_df):,} | Test {len(test_df):,} | "
        f"Users {n_users:,} | Items {n_items:,}"
    )

    user_feat = build_user_features(
        train_df, user_enc,
        prebuilt=prebuilt_features,
        skip_groups=skip_feature_groups,
    )
    item_feat = build_item_features(
        item_enc,
        llm_feat_mode=llm_feat_mode,
        use_nlp_feat=use_nlp_feat,
        prebuilt=prebuilt_features,
        skip_groups=skip_feature_groups,
    )
    adj = build_adj(train_df, n_users, n_items)

    model = IFLGCLModel(
        n_users=n_users, n_items=n_items, emb_dim=emb_dim, n_layers=n_layers,
        user_feat=user_feat, item_feat=item_feat,
        grad_checkpoint=grad_checkpoint, edge_dropout=edge_dropout,
        use_film=use_film,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 50)

    log.info(
        f"Training IFL-GCL | device={DEVICE} | epochs={epochs} | emb_dim={emb_dim} | "
        f"layers={n_layers} | lr={lr} | cl_weight={cl_weight} | "
        f"noise_eps={noise_eps} | cl_temp={cl_temp} | "
        f"warmup={warmup_epochs} | update_interval={update_interval} | "
        f"threshold={threshold} | beta={beta} | top_k={top_k}"
    )

    # D_U^+ pair tensors — populated after warmup, refreshed every K epochs
    unlab_user_src: torch.Tensor | None = None
    unlab_user_dst: torch.Tensor | None = None
    unlab_user_w:   torch.Tensor | None = None
    unlab_item_src: torch.Tensor | None = None
    unlab_item_dst: torch.Tensor | None = None
    unlab_item_w:   torch.Tensor | None = None

    emb_pool    = None
    best_recall = 0.0
    best_state  = None

    train_with_neg = sample_negatives(train_df, n_items, emb_pool=emb_pool)
    all_users  = torch.tensor(train_with_neg["user_idx"].values, dtype=torch.long)
    all_pos    = torch.tensor(train_with_neg["item_idx"].values, dtype=torch.long)
    all_neg    = torch.tensor(train_with_neg["neg_idx"].values,  dtype=torch.long)
    n_train    = len(all_users)

    for epoch in range(1, epochs + 1):

        # ── Refresh hard-negative pool ─────────────────────────────────────────
        if hard_neg_refresh > 0 and epoch > 1 and (epoch - 1) % hard_neg_refresh == 0:
            model.eval()
            with torch.no_grad():
                _, i_emb_tmp = model.get_embeddings(adj)
            emb_pool = build_hard_neg_pool(i_emb_tmp, k=50)
            del i_emb_tmp
            model.train()
            log.info(f"  [Epoch {epoch}] Hard neg pool refreshed")
            train_with_neg = sample_negatives(train_df, n_items, emb_pool=emb_pool)
            all_users = torch.tensor(train_with_neg["user_idx"].values, dtype=torch.long)
            all_pos   = torch.tensor(train_with_neg["item_idx"].values, dtype=torch.long)
            all_neg   = torch.tensor(train_with_neg["neg_idx"].values,  dtype=torch.long)

        # ── Mine D_U^+ after warmup, refreshed every K epochs ─────────────────
        is_mining_epoch = (
            epoch > warmup_epochs and
            (epoch - warmup_epochs) % update_interval == 1
        )
        if is_mining_epoch:
            log.info(f"  [Epoch {epoch}] Mining D_U^+ (threshold={threshold}) …")
            model.eval()
            with torch.no_grad():
                u_emb_m, i_emb_m = model.get_embeddings(adj)

            result_u = mine_unlabeled_positives(
                u_emb_m, threshold=threshold, top_k=top_k, chunk_size=mine_chunk,
            )
            if result_u is not None:
                unlab_user_src, unlab_user_dst, unlab_user_w = (
                    t.to(DEVICE) for t in result_u
                )
            else:
                unlab_user_src = unlab_user_dst = unlab_user_w = None
                log.info("    No user D_U^+ pairs found at this threshold.")

            result_i = mine_unlabeled_positives(
                i_emb_m, threshold=threshold, top_k=top_k, chunk_size=mine_chunk,
            )
            if result_i is not None:
                unlab_item_src, unlab_item_dst, unlab_item_w = (
                    t.to(DEVICE) for t in result_i
                )
            else:
                unlab_item_src = unlab_item_dst = unlab_item_w = None
                log.info("    No item D_U^+ pairs found at this threshold.")

            del u_emb_m, i_emb_m
            torch.cuda.empty_cache()
            model.train()

        model.train()
        optimizer.zero_grad()

        # ── Step 1: CL loss — two augmented views, backward immediately ────────
        u_emb1, i_emb1 = model.forward_cl(adj, noise_eps)
        u_emb2, i_emb2 = model.forward_cl(adj, noise_eps)

        sub_u = torch.randperm(n_users, device=DEVICE)[:batch_size]
        sub_i = torch.randperm(n_items, device=DEVICE)[:batch_size]

        # Use standard loss during warmup; corrected loss thereafter
        if epoch <= warmup_epochs:
            loss_cl = (
                info_nce_loss(u_emb1[sub_u], u_emb2[sub_u], cl_temp) +
                info_nce_loss(i_emb1[sub_i], i_emb2[sub_i], cl_temp)
            )
        else:
            loss_cl = (
                corrected_cl_loss(
                    u_emb1[sub_u], u_emb2[sub_u],
                    node_ids=sub_u,
                    unlab_src=unlab_user_src,
                    unlab_dst=unlab_user_dst,
                    unlab_weights=unlab_user_w,
                    temp=cl_temp, beta=beta,
                ) +
                corrected_cl_loss(
                    i_emb1[sub_i], i_emb2[sub_i],
                    node_ids=sub_i,
                    unlab_src=unlab_item_src,
                    unlab_dst=unlab_item_dst,
                    unlab_weights=unlab_item_w,
                    temp=cl_temp, beta=beta,
                )
            )

        (cl_weight * loss_cl).backward()
        cl_val = loss_cl.item()
        del u_emb1, i_emb1, u_emb2, i_emb2, loss_cl
        torch.cuda.empty_cache()

        # ── Step 2a: BPR gradient — no_grad chunked scatter_add ───────────────
        _saved_dropout     = model.edge_dropout
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

        # ── Step 2b: one ckpt forward + backward with precomputed BPR grad ─────
        final_bpr = model._propagate(adj, noise_eps=0.0, use_ckpt=True)
        final_bpr.backward(gradient=grad_final)
        del final_bpr, grad_final
        model.edge_dropout = _saved_dropout
        torch.cuda.empty_cache()

        optimizer.step()
        scheduler.step()

        phase = "warmup" if epoch <= warmup_epochs else "corrected"
        if epoch % 10 == 0:
            log.info(
                f"  Epoch {epoch}/{epochs} [{phase}]  "
                f"bpr={bpr_val:.4f}  cl={cl_val:.4f}  "
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
                log.info(
                    f"  [Epoch {epoch}] val Recall@10={val_recall:.4f}  *** new best ***"
                )
            else:
                log.info(
                    f"  [Epoch {epoch}] val Recall@10={val_recall:.4f}  "
                    f"(best={best_recall:.4f})"
                )

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info(f"Loaded best checkpoint (val Recall@10={best_recall:.4f})")

    # ── Final evaluation ───────────────────────────────────────────────────────
    model.eval()
    log.info("Computing embeddings for final evaluation…")
    u_emb, i_emb = model.get_embeddings(adj)

    log.info(
        f"  Embedding norms — users mean={u_emb.norm(dim=1).mean():.4f}, "
        f"items mean={i_emb.norm(dim=1).mean():.4f}"
    )

    test_users_arr  = test_df["user_idx"].values
    test_items_arr  = test_df["item_idx"].values
    train_items_map = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    hits, ndcg_sum, total = 0, 0.0, 0
    chunk = 4096
    log.info(f"Evaluating {len(test_users_arr):,} test users in chunks of {chunk}…")

    for start in range(0, len(test_users_arr), chunk):
        end     = min(start + chunk, len(test_users_arr))
        u_idx   = test_users_arr[start:end]
        i_idx   = test_items_arr[start:end]
        u_batch = u_emb[u_idx]
        scores  = (u_batch @ i_emb.T).cpu().numpy()

        for k, (ui, ii) in enumerate(zip(u_idx, i_idx)):
            excl = train_items_map.get(ui, set())
            for ex in excl:
                scores[k, ex] = -np.inf
            top10 = np.argsort(-scores[k])[:10]
            if ii in top10:
                hits     += 1
                ndcg_sum += ndcg_at_k({ii}, top10.tolist(), 10)
            total += 1

    precision = hits / (total * 10)
    recall    = hits / total
    ndcg      = ndcg_sum / total

    print(f"\n=== IFL-GCL Results ===")
    print(f"  Precision@10: {precision:.4f}")
    print(f"  Recall@10:    {recall:.4f}")
    print(f"  NDCG@10:      {ndcg:.4f}")

    skip_str = ",".join(skip_feature_groups) if skip_feature_groups else ""
    variant  = f"skip {skip_str}" if skip_str else "all features"
    save_result(
        model="IFL-GCL", variant=variant,
        precision=precision, recall=recall, ndcg=ndcg,
        epochs=epochs, emb_dim=emb_dim, n_layers=n_layers,
        skip_groups=skip_str,
        notes=(
            f"warmup={warmup_epochs} interval={update_interval} "
            f"threshold={threshold} beta={beta} top_k={top_k}"
        ),
    )

    if save:
        file_tag   = f"skip_{skip_str.replace(',', '_')}" if skip_str else "all"
        model_path = MODEL_OUT.parent / f"ifl_gcl_{file_tag}.pt"
        model_path.parent.mkdir(exist_ok=True)
        EMB_OUT.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_path)
        user_dec = {v: k for k, v in user_enc.items()}
        item_dec = {v: k for k, v in item_enc.items()}
        torch.save(
            {"embeddings": u_emb.cpu(), "id_map": user_dec},
            EMB_OUT / f"ifl_gcl_{file_tag}_user_embeddings.pt",
        )
        torch.save(
            {"embeddings": i_emb.cpu(), "id_map": item_dec},
            EMB_OUT / f"ifl_gcl_{file_tag}_item_embeddings.pt",
        )
        log.info(f"Model → {model_path} | Embeddings → {EMB_OUT}")

        save_predictions(
            u_emb, i_emb,
            train_df=train_df, test_df=test_df,
            user_enc=user_enc, item_enc=item_enc,
            file_tag=file_tag,
        )

    return recall


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IFL-GCL recommendation model")

    # Shared training args
    parser.add_argument("--epochs",            type=int,   default=300)
    parser.add_argument("--emb-dim",           type=int,   default=2048)
    parser.add_argument("--layers",            type=int,   default=4)
    parser.add_argument("--lr",                type=float, default=1e-3)
    parser.add_argument("--batch-size",        type=int,   default=8192)
    parser.add_argument("--edge-dropout",      type=float, default=0.0)
    parser.add_argument("--film",              action="store_true", default=False)
    parser.add_argument("--grad-checkpoint",   action="store_true", default=False)
    parser.add_argument("--hard-neg-refresh",  type=int,   default=50)
    parser.add_argument("--llm-feat-mode",     default="fast",
                        choices=["none", "fast", "full"])
    parser.add_argument("--no-nlp-feat",       action="store_true")
    parser.add_argument("--eval-every",        type=int,   default=50)
    parser.add_argument("--no-save",           action="store_true")
    parser.add_argument("--prebuilt-features", action="store_true")
    parser.add_argument("--skip-feature-groups", default="",
                        help="Comma-separated feature groups to drop "
                             "(user: base,extended,pref; item: base,nlp,extended,llm)")

    # SimGCL base
    parser.add_argument("--cl-weight",  type=float, default=0.2,
                        help="Weight of contrastive loss λ (default 0.2)")
    parser.add_argument("--noise-eps",  type=float, default=0.1,
                        help="Per-layer noise magnitude ε (default 0.1)")
    parser.add_argument("--cl-temp",   type=float, default=0.15,
                        help="InfoNCE temperature τ (default 0.15)")

    # IFL-GCL specific
    parser.add_argument("--warmup-epochs",   type=int,   default=50,
                        help="Standard SimGCL warmup before mining D_U^+ (default 50)")
    parser.add_argument("--update-interval", type=int,   default=50,
                        help="Re-mine D_U^+ every K epochs after warmup (default 50)")
    parser.add_argument("--threshold",       type=float, default=0.90,
                        help="Cosine similarity threshold t_s for D_U^+ (default 0.90)")
    parser.add_argument("--beta",            type=float, default=1.0,
                        help="Exponential weight for D_U^+ correction (default 1.0)")
    parser.add_argument("--top-k",          type=int,   default=20,
                        help="Max similar neighbours mined per node (default 20)")
    parser.add_argument("--mine-chunk",     type=int,   default=4096,
                        help="Chunk size for GPU similarity mining (default 4096)")

    args = parser.parse_args()

    skip_groups = (
        [g.strip() for g in args.skip_feature_groups.split(",") if g.strip()]
        if args.skip_feature_groups else None
    )

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
        prebuilt_features=args.prebuilt_features,
        skip_feature_groups=skip_groups,
        cl_weight=args.cl_weight,
        noise_eps=args.noise_eps,
        cl_temp=args.cl_temp,
        warmup_epochs=args.warmup_epochs,
        update_interval=args.update_interval,
        threshold=args.threshold,
        beta=args.beta,
        top_k=args.top_k,
        mine_chunk=args.mine_chunk,
    )


if __name__ == "__main__":
    main()
