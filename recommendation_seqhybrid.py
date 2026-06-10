#!/usr/bin/env python3
"""
recommendation_seqhybrid.py — Sequential/Hybrid Recommendation.

Architecture: LightGCN graph branch + SASRec sequential branch, fused via a
learned scalar gate.

  final_u = graph_u + sigmoid(α) * seq_u
  score   = final_u · graph_i

Rationale:
  Our dataset is very sparse (median seq_len=3, mean=3.75), so the graph
  branch (LightGCN) carries the primary ranking signal.  The SASRec branch
  adds a recency bias — strongest for the ~20% of users with ≥5 interactions,
  weaker (close to a single-item embedding) for shorter histories.

Graph branch: LightGCN, 3 layers — same manual-gradient BPR pattern as
  SimGCL/KGAT (one ckpt backward per epoch).

Sequence branch: SASRec, 2 transformer layers, 4 heads — mini-batch backward
  using the precomputed BPR gradient w.r.t. seq_u (α * grad_final_u).

Fusion gate α: learned scalar parameter; gradient computed analytically via
  chain rule through the fusion function.

Usage:
    python recommendation_seqhybrid.py \\
        --epochs 300 --emb-dim 2048 --gcn-layers 3 \\
        --seq-layers 2 --seq-heads 4 --max-seq-len 20 \\
        --batch-size 8192 --eval-every 50 --prebuilt-features
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
    _sparse_mm_f32, dropout_adj,
    build_adj, load_interactions,
    build_user_features, build_item_features,
    sample_negatives, build_hard_neg_pool,
    ndcg_at_k, DEVICE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("seqhybrid")

SEQ_DATA  = Path("data/user_sequences.parquet")
MODEL_OUT = Path("models/seqhybrid.pt")
EMB_OUT   = Path("data/embeddings")


# ── Sequence data loading ──────────────────────────────────────────────────────

def load_sequences(n_users: int, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load user_sequences.parquet → padded tensors.

    Returns:
        seq_ids  (n_users, max_seq_len) long  — item indices, 0 = pad
        seq_lens (n_users,)             long  — actual sequence length
    """
    seq_df = pd.read_parquet(SEQ_DATA)

    seq_arr = np.zeros((n_users, max_seq_len), dtype=np.int32)
    seq_len = np.zeros(n_users, dtype=np.int32)

    for row in seq_df.itertuples(index=False):
        uid  = int(row.user_idx)
        seq  = list(row.seq)[-max_seq_len:]   # keep most-recent max_seq_len items
        L    = len(seq)
        # shift item ids by 1 so 0 is reserved as pad token
        seq_arr[uid, :L] = [x + 1 for x in seq]
        seq_len[uid]     = L

    # Users with no sequence data default to len=1 (single pad token → model returns pad emb)
    seq_len = np.maximum(seq_len, 1)

    log.info(f"Sequences loaded — mean len: {seq_len.mean():.2f}  "
             f"median: {int(np.median(seq_len))}  max: {seq_len.max()}")

    return (torch.tensor(seq_arr, dtype=torch.long),
            torch.tensor(seq_len, dtype=torch.long))


# ── SASRec Encoder ─────────────────────────────────────────────────────────────

class SASRecEncoder(nn.Module):
    """
    Self-Attentive Sequential Recommendation encoder.

    Input:  (B, L) item index tensor (0 = pad, 1-indexed items)
    Output: (B, D) user sequence embedding (last valid token's hidden state)

    The transformer runs in seq_emb_dim (default 256) for efficiency; a linear
    projection maps the output to the GCN emb_dim.  Running the transformer at
    the full GCN emb_dim (e.g. 2048) is ~64× slower with no accuracy benefit.
    """

    def __init__(
        self,
        n_items: int,
        emb_dim: int,
        max_seq_len: int,
        seq_emb_dim: int = 256,
        n_layers: int  = 2,
        n_heads: int   = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.emb_dim     = emb_dim
        self.seq_emb_dim = seq_emb_dim
        self.max_seq_len = max_seq_len

        # +1 for pad token (index 0), items are 1-indexed
        self.item_emb = nn.Embedding(n_items + 1, seq_emb_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_seq_len, seq_emb_dim)
        self.drop     = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=seq_emb_dim, nhead=n_heads,
            dim_feedforward=seq_emb_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(seq_emb_dim)

        # Project seq embedding up to GCN embedding dimension
        self.out_proj = nn.Linear(seq_emb_dim, emb_dim, bias=False)

        nn.init.xavier_uniform_(self.item_emb.weight[1:])  # skip pad row
        nn.init.xavier_uniform_(self.pos_emb.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        seq_ids:  torch.Tensor,   # (B, L) long, 0 = pad
        seq_lens: torch.Tensor,   # (B,)   long, actual lengths
    ) -> torch.Tensor:            # (B, D)
        B, L = seq_ids.shape
        pos  = torch.arange(L, device=seq_ids.device).unsqueeze(0)  # (1, L)

        x = self.drop(self.item_emb(seq_ids) + self.pos_emb(pos))   # (B, L, D)

        # Padding mask: True = ignore position
        pad_mask = seq_ids == 0                                      # (B, L)
        # Guarantee at least one unmasked position (transformer requirement)
        last_valid = (seq_lens - 1).clamp(min=0)
        pad_mask[torch.arange(B, device=seq_ids.device), last_valid] = False

        x = self.transformer(x, src_key_padding_mask=pad_mask)      # (B, L, S)
        x = self.norm(x)

        # Output: hidden state at the last valid position, projected to emb_dim
        out = x[torch.arange(B, device=seq_ids.device), last_valid] # (B, S)
        return self.out_proj(out)                                    # (B, D)


# ── SeqHybrid Model ────────────────────────────────────────────────────────────

class SeqHybrid(nn.Module):
    """
    LightGCN graph branch + SASRec sequence branch fused by a learned scalar gate.

    final_u = graph_u + sigmoid(α) * seq_u
    score   = final_u · graph_i
    """

    def __init__(
        self,
        n_users:     int,
        n_items:     int,
        emb_dim:     int,
        max_seq_len: int,
        n_gcn_layers: int  = 3,
        n_seq_layers: int  = 2,
        n_seq_heads:  int  = 4,
        seq_emb_dim:  int  = 256,
        seq_dropout:  float = 0.2,
        edge_dropout: float = 0.0,
        user_feat: np.ndarray | None = None,
        item_feat: np.ndarray | None = None,
    ):
        super().__init__()
        self.n_users      = n_users
        self.n_items      = n_items
        self.n_gcn_layers = n_gcn_layers
        self.edge_dropout = edge_dropout

        # ── GCN branch ────────────────────────────────────────────────────────
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

        # ── SASRec branch ─────────────────────────────────────────────────────
        self.seq_enc = SASRecEncoder(
            n_items=n_items, emb_dim=emb_dim, max_seq_len=max_seq_len,
            seq_emb_dim=seq_emb_dim,
            n_layers=n_seq_layers, n_heads=n_seq_heads, dropout=seq_dropout,
        )

        # ── Fusion gate ───────────────────────────────────────────────────────
        # sigmoid(-5) ≈ 0.007: GCN trains clean early; α grows only if
        # the sequence branch earns it through gradient pressure
        self.alpha_param = nn.Parameter(torch.full((1,), -5.0))

    # ── GCN propagation (same pattern as SimGCL) ──────────────────────────────

    def _propagate_gcn(self, adj: torch.Tensor, use_ckpt: bool = True) -> torch.Tensor:
        adj_used = dropout_adj(adj, self.edge_dropout, self.training)

        def _full_pass(u_w, i_w):
            u = u_w + self.user_proj(self.user_feat) if self.user_proj else u_w
            i = i_w + self.item_proj(self.item_feat) if self.item_proj else i_w
            e = torch.cat([u, i], dim=0)
            s = e
            for _ in range(self.n_gcn_layers):
                e = _sparse_mm_f32(adj_used, e)
                s = s + e
            return s / (self.n_gcn_layers + 1)

        if use_ckpt and self.training:
            return grad_ckpt(_full_pass, self.user_emb.weight, self.item_emb.weight,
                             use_reentrant=False)
        return _full_pass(self.user_emb.weight, self.item_emb.weight)

    # ── Sequence encoding ──────────────────────────────────────────────────────

    def encode_seq_batch(
        self,
        seq_ids:  torch.Tensor,   # (B, max_seq_len)
        seq_lens: torch.Tensor,   # (B,)
    ) -> torch.Tensor:            # (B, D)
        return self.seq_enc(seq_ids.to(DEVICE), seq_lens.to(DEVICE))

    def encode_all_seqs(
        self,
        all_seq_ids:  torch.Tensor,   # (n_users, L)
        all_seq_lens: torch.Tensor,   # (n_users,)
        chunk: int = 2048,
    ) -> torch.Tensor:                # (n_users, D)
        parts = []
        for s in range(0, self.n_users, chunk):
            e  = min(s + chunk, self.n_users)
            su = self.encode_seq_batch(all_seq_ids[s:e], all_seq_lens[s:e])
            parts.append(su)
        return torch.cat(parts, dim=0)

    # ── Full embeddings for evaluation ────────────────────────────────────────

    def get_embeddings(
        self,
        adj:          torch.Tensor,
        all_seq_ids:  torch.Tensor,
        all_seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            all_gcn = self._propagate_gcn(adj, use_ckpt=False)
            graph_u = all_gcn[:self.n_users]
            graph_i = all_gcn[self.n_users:]
            seq_u   = self.encode_all_seqs(all_seq_ids, all_seq_lens)
            alpha   = torch.sigmoid(self.alpha_param)
            final_u = graph_u + alpha * seq_u
        return final_u, graph_i


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    epochs: int          = 300,
    emb_dim: int         = 2048,
    n_gcn_layers: int    = 3,
    n_seq_layers: int    = 2,
    n_seq_heads: int     = 4,
    max_seq_len: int     = 20,
    seq_emb_dim: int     = 256,
    seq_dropout: float   = 0.2,
    lr: float            = 1e-3,
    lr_seq: float        = 1e-3,
    batch_size: int      = 8192,
    seq_chunk: int       = 8192,
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

    log.info("Loading user sequences…")
    all_seq_ids, all_seq_lens = load_sequences(n_users, max_seq_len)

    model = SeqHybrid(
        n_users=n_users, n_items=n_items, emb_dim=emb_dim,
        max_seq_len=max_seq_len,
        n_gcn_layers=n_gcn_layers, n_seq_layers=n_seq_layers,
        n_seq_heads=n_seq_heads, seq_emb_dim=seq_emb_dim,
        seq_dropout=seq_dropout,
        edge_dropout=0.0,
        user_feat=user_feat, item_feat=item_feat,
    ).to(DEVICE)

    # ── Separate optimisers: GCN branch vs. SASRec + fusion ───────────────────
    gcn_param_ids = set()
    gcn_params    = [{"params": [model.user_emb.weight]},
                     {"params": [model.item_emb.weight]}]
    for p in [model.user_proj, model.item_proj]:
        if p is not None:
            gcn_params.append({"params": list(p.parameters())})
    for pg in gcn_params:
        for p in pg["params"]:
            gcn_param_ids.add(id(p))

    seq_params = [
        {"params": list(model.seq_enc.parameters())},
        {"params": [model.alpha_param]},
    ]

    optimizer_gcn = Adam(gcn_params, lr=lr)
    optimizer_seq = Adam(seq_params, lr=lr_seq)
    scheduler_gcn = CosineAnnealingLR(optimizer_gcn, T_max=epochs, eta_min=lr / 50)
    scheduler_seq = CosineAnnealingLR(optimizer_seq, T_max=epochs, eta_min=lr_seq / 50)

    log.info(
        f"Training SeqHybrid | device={DEVICE} | epochs={epochs} | emb_dim={emb_dim} | "
        f"gcn_layers={n_gcn_layers} | seq_layers={n_seq_layers} | "
        f"seq_heads={n_seq_heads} | seq_emb_dim={seq_emb_dim} | max_seq_len={max_seq_len} | "
        f"lr={lr} | lr_seq={lr_seq}"
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

        # Hard negative refresh
        if hard_neg_refresh > 0 and epoch > 1 and (epoch - 1) % hard_neg_refresh == 0:
            model.eval()
            with torch.no_grad():
                _, i_emb_eval = model.get_embeddings(adj, all_seq_ids, all_seq_lens)
            emb_pool = build_hard_neg_pool(i_emb_eval, k=50)
            del i_emb_eval
            model.train()
            log.info(f"  [Epoch {epoch}] Hard neg pool refreshed")
            train_with_neg = sample_negatives(train_df, n_items, emb_pool=emb_pool)
            all_users = torch.tensor(train_with_neg["user_idx"].values, dtype=torch.long)
            all_pos   = torch.tensor(train_with_neg["item_idx"].values, dtype=torch.long)
            all_neg   = torch.tensor(train_with_neg["neg_idx"].values,  dtype=torch.long)

        model.train()

        # ── Phase 1: GCN + SASRec forward, no_grad ────────────────────────────
        # Compute detached fused embeddings for the BPR gradient calculation.
        with torch.no_grad():
            all_gcn = model._propagate_gcn(adj, use_ckpt=False)
            graph_u = all_gcn[:n_users]      # (n_users, D)
            graph_i = all_gcn[n_users:]      # (n_items, D)
            del all_gcn

            seq_u = model.encode_all_seqs(all_seq_ids, all_seq_lens, chunk=seq_chunk)  # (n_users, D)

            alpha   = torch.sigmoid(model.alpha_param)
            final_u = graph_u + alpha * seq_u  # (n_users, D)

        # ── Phase 2: BPR gradient w.r.t. final_u and graph_i ─────────────────
        # Same manual gradient formula as SimGCL/KGAT.
        with torch.no_grad():
            grad_all = torch.zeros(n_users + n_items, final_u.shape[1], device=DEVICE)
            bpr_val  = 0.0
            perm     = torch.randperm(n_train)

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                idx = perm[start:end]
                bu  = all_users[idx].to(DEVICE)
                bp  = all_pos[idx].to(DEVICE)
                bn  = all_neg[idx].to(DEVICE)

                u   = final_u[bu]
                pos = graph_i[bp]
                neg = graph_i[bn]

                margin   = (u * pos).sum(1) - (u * neg).sum(1)
                bpr_val += (-F.logsigmoid(margin)).sum().item() / n_train

                coef = -(1.0 - torch.sigmoid(margin)) / n_train
                r    = 1e-4 / n_train

                gu = coef.unsqueeze(1) * (pos - neg) + r * u
                gp = coef.unsqueeze(1) * u            + r * pos
                gn = -coef.unsqueeze(1) * u           + r * neg

                grad_all[:n_users].scatter_add_(0, bu.unsqueeze(1).expand_as(gu), gu)
                grad_all[n_users:].scatter_add_(0, bp.unsqueeze(1).expand_as(gp), gp)
                grad_all[n_users:].scatter_add_(0, bn.unsqueeze(1).expand_as(gn), gn)

            del final_u

        # ── Phase 3: GCN backward ─────────────────────────────────────────────
        # grad_all[:n_users] = ∂L/∂final_u = ∂L/∂graph_u  (pass-through from fusion)
        # grad_all[n_users:] = ∂L/∂graph_i
        optimizer_gcn.zero_grad()
        all_gcn_grad = model._propagate_gcn(adj, use_ckpt=True)
        all_gcn_grad.backward(gradient=grad_all)
        del all_gcn_grad
        optimizer_gcn.step()
        torch.cuda.empty_cache()

        # ── Phase 4: SASRec backward (mini-batch) ─────────────────────────────
        # ∂L/∂seq_u = α * ∂L/∂final_u   (chain rule through fusion)
        # ∂L/∂α_param = (∂L/∂final_u * seq_u * α * (1-α)).sum()  — set manually
        alpha = torch.sigmoid(model.alpha_param)
        grad_seq = alpha.detach() * grad_all[:n_users]   # (n_users, D)

        optimizer_seq.zero_grad()
        for s in range(0, n_users, seq_chunk):
            e    = min(s + seq_chunk, n_users)
            su   = model.encode_seq_batch(all_seq_ids[s:e], all_seq_lens[s:e])
            gsu  = grad_seq[s:e]
            su.backward(gradient=gsu)

        # Gradient for the fusion gate α (analytical)
        with torch.no_grad():
            dL_dalpha = (grad_all[:n_users] * seq_u).sum()
            dL_dparam = dL_dalpha * alpha * (1.0 - alpha)
        if model.alpha_param.grad is None:
            model.alpha_param.grad = dL_dparam.view(1)
        else:
            model.alpha_param.grad += dL_dparam.view(1)

        del seq_u, grad_all, grad_seq
        optimizer_seq.step()
        torch.cuda.empty_cache()

        scheduler_gcn.step()
        scheduler_seq.step()

        if epoch % 10 == 0:
            alpha_val = torch.sigmoid(model.alpha_param).item()
            log.info(
                f"  Epoch {epoch}/{epochs}  "
                f"bpr={bpr_val:.4f}  "
                f"α={alpha_val:.3f}  "
                f"lr={scheduler_gcn.get_last_lr()[0]:.2e}"
            )

        if eval_every > 0 and epoch % eval_every == 0:
            model.eval()
            u_emb_eval, i_emb_eval = model.get_embeddings(adj, all_seq_ids, all_seq_lens)

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
    u_emb, i_emb = model.get_embeddings(adj, all_seq_ids, all_seq_lens)

    log.info(f"  Embedding norms — users mean={u_emb.norm(dim=1).mean():.4f}, "
             f"items mean={i_emb.norm(dim=1).mean():.4f}")
    log.info(f"  Final α = {torch.sigmoid(model.alpha_param).item():.4f}")

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

    print(f"\n=== SeqHybrid Results ===")
    print(f"  Precision@10: {precision:.4f}")
    print(f"  Recall@10:    {recall:.4f}")
    print(f"  NDCG@10:      {ndcg:.4f}")
    print(f"  Final α:      {torch.sigmoid(model.alpha_param).item():.4f}")

    skip_str = ",".join(skip_feature_groups) if skip_feature_groups else ""
    variant  = f"gcn{n_gcn_layers}+seq{n_seq_layers}" + (f" skip {skip_str}" if skip_str else "")
    save_result(
        model="SeqHybrid", variant=variant,
        precision=precision, recall=recall, ndcg=ndcg,
        epochs=epochs, emb_dim=emb_dim,
        n_layers=n_gcn_layers,
        skip_groups=skip_str,
    )

    if save:
        file_tag   = f"skip_{skip_str.replace(',', '_')}" if skip_str else "all"
        model_path = MODEL_OUT.parent / f"seqhybrid_{file_tag}.pt"
        model_path.parent.mkdir(exist_ok=True)
        EMB_OUT.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_path)
        user_dec = {v: k for k, v in user_enc.items()}
        item_dec = {v: k for k, v in item_enc.items()}
        torch.save({"embeddings": u_emb.cpu(), "id_map": user_dec},
                   EMB_OUT / f"seqhybrid_{file_tag}_user_embeddings.pt")
        torch.save({"embeddings": i_emb.cpu(), "id_map": item_dec},
                   EMB_OUT / f"seqhybrid_{file_tag}_item_embeddings.pt")
        log.info(f"Model → {model_path} | Embeddings → {EMB_OUT}")

    return recall


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SeqHybrid: LightGCN + SASRec")
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--emb-dim",      type=int,   default=2048)
    parser.add_argument("--gcn-layers",   type=int,   default=3)
    parser.add_argument("--seq-layers",   type=int,   default=2)
    parser.add_argument("--seq-heads",    type=int,   default=4)
    parser.add_argument("--max-seq-len",  type=int,   default=20)
    parser.add_argument("--seq-emb-dim",  type=int,   default=256,
                        help="Transformer internal dimension (output projected to emb-dim)")
    parser.add_argument("--seq-dropout",  type=float, default=0.2)
    parser.add_argument("--lr",           type=float, default=1e-3,
                        help="LR for GCN branch")
    parser.add_argument("--lr-seq",       type=float, default=1e-3,
                        help="LR for SASRec + fusion branch")
    parser.add_argument("--batch-size",   type=int,   default=8192)
    parser.add_argument("--seq-chunk",    type=int,   default=8192,
                        help="Users per chunk when encoding all sequences")
    parser.add_argument("--hard-neg-refresh", type=int, default=50)
    parser.add_argument("--eval-every",   type=int,   default=50)
    parser.add_argument("--no-save",      action="store_true")
    parser.add_argument("--no-nlp-feat",  action="store_true")
    parser.add_argument("--llm-feat-mode",type=str,   default="fast",
                        choices=["none", "fast", "full"])
    parser.add_argument("--prebuilt-features", action="store_true")
    parser.add_argument("--skip-feature-groups", default="",
                        help="Comma-separated feature groups to drop")
    args = parser.parse_args()

    skip_groups = [g.strip() for g in args.skip_feature_groups.split(",") if g.strip()] \
        if args.skip_feature_groups else None

    train(
        epochs              = args.epochs,
        emb_dim             = args.emb_dim,
        n_gcn_layers        = args.gcn_layers,
        n_seq_layers        = args.seq_layers,
        n_seq_heads         = args.seq_heads,
        max_seq_len         = args.max_seq_len,
        seq_emb_dim         = args.seq_emb_dim,
        seq_dropout         = args.seq_dropout,
        lr                  = args.lr,
        lr_seq              = args.lr_seq,
        batch_size          = args.batch_size,
        seq_chunk           = args.seq_chunk,
        save                = not args.no_save,
        llm_feat_mode       = args.llm_feat_mode,
        use_nlp_feat        = not args.no_nlp_feat,
        eval_every          = args.eval_every,
        prebuilt_features   = args.prebuilt_features,
        skip_feature_groups = skip_groups,
    )
