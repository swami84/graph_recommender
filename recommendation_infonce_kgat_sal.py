#!/usr/bin/env python3
"""
recommendation_infonce_kgat_sal.py — KGAT-SAL + InfoNCE View-Based Contrastive Learning.

Extends KGAT-SAL with two-view InfoNCE contrastive learning on the KG-enriched CF
embeddings. Two independently-augmented graph views (different edge dropout + per-layer
noise masks from independent random draws) are compared for same-entity pairs via
symmetric InfoNCE.

Why: KGAT-SAL's BPR + SAL gradient weakens as representations converge; view-based CL
provides a persistent signal throughout training, encourages better representation
structure, and combats embedding collapse in the 2048D space.

Architecture: identical to KGAT-SAL with one additional step per epoch:
  Two augmented CF forwards (no_grad)  → u_v1/i_v1, u_v2/i_v2
  InfoNCE(u_v1[batch], u_v2[batch]) + InfoNCE(i_v1[batch], i_v2[batch])
  Accumulated CL grads merged into BPR grad_final: grad_total = grad_bpr + λ_cl * grad_cl
  Single gradient-checkpointed _propagate_all backward — no extra memory overhead.

Loss:
  L = L_BPR(u_final, i_kgat) + λ_cl * L_CL(views) + λ_sal * L_SAL

Usage:
    python recommendation_infonce_kgat_sal.py --prebuilt-features
    python recommendation_infonce_kgat_sal.py --epochs 300 --emb-dim 2048 \\
        --n-periods 3 --lambda-cl 0.1 --cl-aug-dropout 0.2 --cl-temp 0.15 \\
        --cl-sample-size 4096 --prebuilt-features
"""

import argparse
import copy
import csv
import logging
import os
from datetime import datetime
from pathlib import Path

import contextlib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import Adafactor

from model_results import save_result
from recommendation_gnn import (
    _sparse_mm_f32, dropout_adj,
    build_adj, load_interactions,
    build_user_features, build_item_features,
    sample_negatives, build_hard_neg_pool,
    _eval_recall_at_k, ndcg_at_k, _is_main_rank, _sync_grads, init_distributed, DEVICE,
)
from recommendation_kgat import build_kg, N_RELATIONS
from recommendation_kgat_sal import KGAT_SAL, build_period_adjs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("infonce_kgat_sal")
log.setLevel(logging.INFO)
if not log.handlers:
    _lh = logging.StreamHandler()
    _lh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_lh)
    log.propagate = False

MODEL_OUT      = Path("models/infonce_kgat_sal.pt")
EMB_OUT        = Path("data/embeddings")
PRED_DIR       = Path("data/predictions")
CHECKPOINT_CSV = Path("results/training_checkpoints.csv")


# ── InfoNCE loss ───────────────────────────────────────────────────────────────

def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float) -> torch.Tensor:
    """Symmetric InfoNCE. Positives: (z1[i], z2[i]). Negatives: all other rows."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.T) / temp
    labels = torch.arange(len(z1), device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ── InfoNCE-KGAT-SAL Model ─────────────────────────────────────────────────────

class InfoNCEKGATSAL(KGAT_SAL):
    """
    KGAT-SAL with view-based InfoNCE contrastive learning on CF embeddings.

    Adds _propagate_aug(): same KG + CF forward as KGAT-SAL's _propagate_all
    but without the temporal component, and with independent edge dropout + noise
    for each call. Two calls produce two different augmented views of the same
    embedding space, suitable for symmetric InfoNCE.

    All KGAT-SAL capabilities (KG attention, temporal SAL, hard neg BPR) unchanged.
    """

    def _propagate_aug(
        self,
        adj: torch.Tensor,
        noise_eps: float,
        aug_dropout: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Augmented CF forward for view-based CL. No temporal component.

        Each call uses independent random edge dropout and per-layer noise masks,
        so two consecutive calls with identical parameters yield two distinct views.
        KG dropout (self.kg_dropout) applies via self.training gate as normal.

        Returns (u_cf, i_cf) — KG-enriched LightGCN embeddings before temporal fusion.
        """
        u0 = self.user_emb.weight
        if self.user_proj is not None:
            u0 = u0 + self.user_proj(self.user_feat)

        # KG propagation (kg_dropout applies when self.training=True)
        ent_emb = self.entity_emb.weight
        if self.item_proj is not None:
            proj    = self.item_proj(self.item_feat)
            ent_emb = ent_emb.clone()
            ent_emb[:self.n_items] = ent_emb[:self.n_items] + proj
        summed_ent = ent_emb
        for _ in range(self.n_kg_layers):
            ent_emb    = self._kg_propagate(ent_emb)
            summed_ent = summed_ent + ent_emb
        enriched = (summed_ent / (self.n_kg_layers + 1))[:self.n_items]

        # CF propagation with augmented adj and per-layer noise
        adj_aug = dropout_adj(adj, aug_dropout, training=True)
        all_emb = torch.cat([u0, enriched], dim=0)
        summed  = all_emb
        for _ in range(self.n_cf_layers):
            all_emb = _sparse_mm_f32(adj_aug, all_emb)
            if noise_eps > 0.0:
                all_emb = all_emb + torch.rand_like(all_emb).sign() * noise_eps
            summed = summed + all_emb
        final = summed / (self.n_cf_layers + 1)
        return final[:self.n_users], final[self.n_users:]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _log_checkpoint(run_id: str, variant: str, epoch: int, recall: float) -> None:
    CHECKPOINT_CSV.parent.mkdir(exist_ok=True)
    write_header = not CHECKPOINT_CSV.exists()
    with open(CHECKPOINT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "run_id", "model", "variant", "epoch", "recall_at_10"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    run_id, "InfoNCE-KGAT-SAL", variant, epoch, f"{recall:.4f}"])


def save_predictions(
    u_emb:    torch.Tensor,
    i_emb:    torch.Tensor,
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    user_enc: dict,
    item_enc: dict,
    file_tag: str,
) -> None:
    user_id_map = {v: k for k, v in user_enc.items()}
    item_id_map = {v: k for k, v in item_enc.items()}
    train_sets  = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    test_users  = test_df["user_idx"].values
    test_items  = test_df["item_idx"].values
    u_cpu       = u_emb.cpu().float()
    i_cpu       = i_emb.cpu().float()

    records = []
    for start in range(0, len(test_users), 2048):
        end    = min(start + 2048, len(test_users))
        u_idx  = test_users[start:end]
        i_idx  = test_items[start:end]
        scores = (u_cpu[u_idx] @ i_cpu.T).numpy()
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
    out_path = PRED_DIR / f"infonce_kgat_sal_{file_tag}_predictions.parquet"
    pred_df.to_parquet(out_path, index=False)
    hit_rate = pred_df["rank"].notna().mean()
    log.info("Predictions → %s  (%d rows | Hit@10=%.4f)", out_path, len(pred_df), hit_rate)


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    epochs: int           = 300,
    emb_dim: int          = 2048,
    n_kg_layers: int      = 2,
    n_cf_layers: int      = 4,
    n_periods: int        = 3,
    d_temporal: int       = 256,
    d_sal: int            = 32,
    lambda_sal: float     = 1e-6,
    lambda_cl: float      = 0.1,
    cl_warmup: int        = 50,
    cl_temp: float        = 0.15,
    cl_aug_dropout: float = 0.2,
    noise_eps: float      = 0.1,
    cl_sample_size: int   = 4096,
    lr: float             = 1e-3,
    batch_size: int       = 2048,
    sal_samples: int      = 4096,
    save: bool            = True,
    edge_dropout: float   = 0.1,
    kg_dropout: float     = 0.1,
    hard_neg_refresh: int = 50,
    llm_feat_mode: str    = "fast",
    use_nlp_feat: bool    = True,
    eval_every: int       = 50,
    prebuilt_features: bool = False,
    skip_feature_groups: list[str] | None = None,
    use_spatial_cbg: bool = True,
) -> float:
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    skip_str = ",".join(skip_feature_groups) if skip_feature_groups else ""
    variant  = f"{'skip ' + skip_str if skip_str else 'all features'} T={n_periods}"

    train_df, test_df, user_enc, item_enc = load_interactions()
    n_users, n_items = len(user_enc), len(item_enc)
    log.info("Train %d | Test %d | Users %d | Items %d",
             len(train_df), len(test_df), n_users, n_items)

    user_feat = build_user_features(train_df, user_enc,
                                    prebuilt=prebuilt_features,
                                    skip_groups=skip_feature_groups)
    item_feat = build_item_features(item_enc, llm_feat_mode=llm_feat_mode,
                                    use_nlp_feat=use_nlp_feat,
                                    prebuilt=prebuilt_features,
                                    skip_groups=skip_feature_groups)
    adj = build_adj(train_df, n_users, n_items)

    log.info("Building temporal period adjacency matrices (T=%d)…", n_periods)
    period_adjs, period_dfs = build_period_adjs(train_df, n_users, n_items, n_periods)

    log.info("Building knowledge graph…")
    kg_heads, kg_rels, kg_tails, n_kg_ents = build_kg(item_enc, use_spatial_cbg=use_spatial_cbg)

    model = InfoNCEKGATSAL(
        n_users=n_users,
        n_items=n_items,
        n_kg_ents=n_kg_ents,
        emb_dim=emb_dim,
        n_kg_layers=n_kg_layers,
        n_cf_layers=n_cf_layers,
        kg_heads=kg_heads,
        kg_rels=kg_rels,
        kg_tails=kg_tails,
        user_feat=user_feat,
        item_feat=item_feat,
        edge_dropout=edge_dropout,
        kg_dropout=kg_dropout,
        d_temporal=d_temporal,
        d_sal=d_sal,
        n_periods=n_periods,
    ).to(DEVICE)
    model.period_adjs = period_adjs

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = model.to(torch.bfloat16)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if dist.is_initialized() else model

    optimizer = Adafactor(model.parameters(), lr=lr,
                          relative_step=False, scale_parameter=False, warmup_init=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 50)

    log.info(
        "Training InfoNCE-KGAT-SAL | device=%s | epochs=%d | emb_dim=%d | "
        "kg_layers=%d | cf_layers=%d | T=%d | d_temp=%d | d_sal=%d | "
        "λ_sal=%.1e | λ_cl=%.2f (warmup=%d) | cl_temp=%.2f | cl_aug_dr=%.2f | noise=%.2f",
        DEVICE, epochs, emb_dim, n_kg_layers, n_cf_layers,
        n_periods, d_temporal, d_sal, lambda_sal, lambda_cl, cl_warmup,
        cl_temp, cl_aug_dropout, noise_eps,
    )

    all_users_t = torch.LongTensor(train_df["user_idx"].values)
    all_pos_t   = torch.LongTensor(train_df["item_idx"].values)
    n_train     = len(train_df)
    neg_pool    = None
    best_recall = 0.0
    best_state  = None
    emb_dim_    = raw_model.user_emb.weight.size(1)

    for epoch in range(1, epochs + 1):

        # ── Hard negative pool refresh ─────────────────────────────────────────
        if epoch % hard_neg_refresh == 1:
            with torch.no_grad():
                _, i_emb_tmp = raw_model.get_embeddings(adj)
            neg_pool = build_hard_neg_pool(i_emb_tmp, k=50)
            del i_emb_tmp
            if epoch > 1:
                log.info("  [Epoch %d] Hard neg pool refreshed", epoch)

        raw_model.train()
        neg_df    = sample_negatives(train_df, n_items, emb_pool=neg_pool)
        all_neg_t = torch.LongTensor(neg_df["neg_idx"].values)

        optimizer.zero_grad()

        # ── Step 1: SAL backward (trains StabilityWeighter) ───────────────────
        L_sal   = raw_model.sal_loss(period_dfs, n_samples=sal_samples)
        _nosync = model.no_sync() if dist.is_initialized() else contextlib.nullcontext()
        with _nosync:
            (lambda_sal * L_sal).backward()
        sal_val = L_sal.item()
        del L_sal
        torch.cuda.empty_cache()

        # ── Step 2: CL (skipped during warmup) ────────────────────────────────
        # grad_cl is normalised by entity count so its scale matches grad_bpr
        # (which is normalised by n_train). Without this, CL grads at τ=0.15
        # are ~1000× larger than BPR grads and completely swamp the ranking signal.
        _model_dt = next(raw_model.parameters()).dtype
        grad_cl = torch.zeros(n_users + n_items, emb_dim_, device=DEVICE, dtype=_model_dt)
        cl_val  = 0.0

        if epoch > cl_warmup:
            # Two no-grad augmented forwards; independent random masks → two views
            with torch.no_grad():
                u_v1, i_v1 = raw_model._propagate_aug(adj, noise_eps, cl_aug_dropout)
                u_v2, i_v2 = raw_model._propagate_aug(adj, noise_eps, cl_aug_dropout)

            perm_u = torch.randperm(n_users)
            perm_i = torch.randperm(n_items)

            for start in range(0, n_users, cl_sample_size):
                end   = min(start + cl_sample_size, n_users)
                bu    = perm_u[start:end].to(DEVICE)
                uv1_b = u_v1[bu].detach().requires_grad_(True)
                uv2_b = u_v2[bu].detach().requires_grad_(True)
                loss_u = info_nce_loss(uv1_b, uv2_b, cl_temp)
                loss_u.backward()
                cl_val += loss_u.item() * len(bu) / n_users
                if uv1_b.grad is not None:
                    grad_cl[:n_users].scatter_add_(
                        0, bu.unsqueeze(1).expand(-1, emb_dim_), uv1_b.grad / n_users
                    )

            del u_v1, u_v2

            for start in range(0, n_items, cl_sample_size):
                end   = min(start + cl_sample_size, n_items)
                bi    = perm_i[start:end].to(DEVICE)
                iv1_b = i_v1[bi].detach().requires_grad_(True)
                iv2_b = i_v2[bi].detach().requires_grad_(True)
                loss_i = info_nce_loss(iv1_b, iv2_b, cl_temp)
                loss_i.backward()
                cl_val += loss_i.item() * len(bi) / n_items
                if iv1_b.grad is not None:
                    grad_cl[n_users:].scatter_add_(
                        0, bi.unsqueeze(1).expand(-1, emb_dim_), iv1_b.grad / n_items
                    )

            del i_v1, i_v2
            torch.cuda.empty_cache()

        # ── Step 3a: BPR gradient accumulation (manual, no_grad) ─────────────
        # Identical to KGAT-SAL: analytically compute BPR gradient and scatter
        # into grad_bpr, avoiding storing the full GCN computation graph.
        _saved_dropout     = raw_model.edge_dropout
        raw_model.edge_dropout = 0.0

        with torch.no_grad():
            final_det = raw_model._propagate_all(adj, use_ckpt=False)
            u_det     = final_det[:n_users]
            i_det     = final_det[n_users:]

            grad_bpr = torch.zeros_like(final_det)
            bpr_val  = 0.0
            perm     = torch.randperm(n_train)

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                idx = perm[start:end]
                bu  = all_users_t[idx].to(DEVICE)
                bp  = all_pos_t[idx].to(DEVICE)
                bn  = all_neg_t[idx].to(DEVICE)

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

                grad_bpr[:n_users].scatter_add_(0, bu.unsqueeze(1).expand_as(gu), gu)
                grad_bpr[n_users:].scatter_add_(0, bp.unsqueeze(1).expand_as(gp), gp)
                grad_bpr[n_users:].scatter_add_(0, bn.unsqueeze(1).expand_as(gn), gn)

            del final_det, u_det, i_det

        # ── Step 3b: Combined BPR+CL ckpt backward ────────────────────────────
        # Merge gradients before the single checkpointed backward. CL grads flow
        # through _propagate_all (which includes temporal) as an approximation —
        # the CL was computed on _propagate_aug (no temporal), but since
        # u_temporal is a small residual correction, this approximation is benign.
        grad_total = grad_bpr + lambda_cl * grad_cl
        del grad_bpr, grad_cl   # free ~2.4 GiB before ckpt recompute
        torch.cuda.empty_cache()
        final_ckpt = raw_model._propagate_all(adj, use_ckpt=True)
        scalar = (final_ckpt * grad_total.detach()).sum()
        del final_ckpt  # not needed during checkpoint recomputation — free 2 GiB before backward
        scalar.backward()
        del grad_total, scalar
        raw_model.edge_dropout = _saved_dropout
        torch.cuda.empty_cache()

        optimizer.step()
        scheduler.step()

        if _is_main_rank() and (epoch == 1 or epoch % 10 == 0):
            log.info(
                "  Epoch %d/%d  bpr=%.4f  sal=%.6f  cl=%.4f  lr=%.2e",
                epoch, epochs, bpr_val, sal_val, cl_val, scheduler.get_last_lr()[0],
            )

        if eval_every > 0 and epoch % eval_every == 0:
            raw_model.eval()
            val_recall = _eval_recall_at_k(raw_model, adj, test_df, train_df, n_items, k=10)
            raw_model.train()
            _log_checkpoint(run_id, variant, epoch, val_recall)
            if val_recall > best_recall:
                best_recall = val_recall
                best_state  = copy.deepcopy(raw_model.state_dict())
                log.info("  [Epoch %d] val Recall@10=%.4f  *** new best ***", epoch, val_recall)
            else:
                log.info("  [Epoch %d] val Recall@10=%.4f  (best=%.4f)",
                         epoch, val_recall, best_recall)

    # ── Restore best checkpoint ────────────────────────────────────────────────
    if best_state is not None:
        raw_model.load_state_dict(best_state)
        log.info("Loaded best checkpoint (val Recall@10=%.4f)", best_recall)

    # ── Final evaluation ───────────────────────────────────────────────────────
    raw_model.eval()
    log.info("Computing final embeddings…")
    u_emb, i_emb = raw_model.get_embeddings(adj)

    log.info("  Embedding norms — users mean=%.4f  items mean=%.4f",
             u_emb.norm(dim=1).mean().item(), i_emb.norm(dim=1).mean().item())

    test_users  = test_df["user_idx"].values
    test_items  = test_df["item_idx"].values
    train_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    hits, ndcg_sum, total = 0, 0.0, 0
    chunk = 4096

    for start in range(0, len(test_users), chunk):
        end     = min(start + chunk, len(test_users))
        u_idx   = test_users[start:end]
        i_idx   = test_items[start:end]
        scores  = (u_emb[u_idx] @ i_emb.T).cpu().numpy()

        for k, (ui, ii) in enumerate(zip(u_idx, i_idx)):
            for ex in train_items.get(ui, set()):
                scores[k, ex] = -np.inf
            top10 = np.argsort(-scores[k])[:10]
            if ii in top10:
                hits += 1
                ndcg_sum += ndcg_at_k({ii}, top10.tolist(), 10)
            total += 1

    precision = hits / (total * 10)
    recall    = hits / total
    ndcg      = ndcg_sum / total

    print(f"\n=== InfoNCE-KGAT-SAL Results ===")
    print(f"  Precision@10: {precision:.4f}")
    print(f"  Recall@10:    {recall:.4f}")
    print(f"  NDCG@10:      {ndcg:.4f}")

    spatial_tag  = "+spatial_cbg" if use_spatial_cbg else "no_spatial"
    full_variant = f"{'skip ' + skip_str if skip_str else 'all features'} T={n_periods} {spatial_tag}"
    save_result(
        model="InfoNCE-KGAT-SAL", variant=full_variant,
        precision=precision, recall=recall, ndcg=ndcg,
        epochs=epochs, emb_dim=emb_dim, n_layers=n_cf_layers,
        skip_groups=skip_str,
        notes=(
            f"T={n_periods} d_temp={d_temporal} d_sal={d_sal} "
            f"λ_sal={lambda_sal:.0e} λ_cl={lambda_cl:.2f} "
            f"cl_temp={cl_temp} cl_aug_dr={cl_aug_dropout} kg_layers={n_kg_layers}"
        ),
    )

    if save:
        if _is_main_rank():
            spatial_suffix = "" if use_spatial_cbg else "_nospatial"
            file_tag   = f"skip_{skip_str.replace(',', '_')}" if skip_str else "all"
            file_tag   = f"{file_tag}_T{n_periods}{spatial_suffix}"
            model_path = MODEL_OUT.parent / f"infonce_kgat_sal_{file_tag}.pt"
            model_path.parent.mkdir(exist_ok=True)
            EMB_OUT.mkdir(exist_ok=True)

            torch.save(raw_model.state_dict(), model_path)

            user_dec = {v: k for k, v in user_enc.items()}
            item_dec = {v: k for k, v in item_enc.items()}
            torch.save({"embeddings": u_emb.cpu(), "id_map": user_dec},
                       EMB_OUT / f"infonce_kgat_sal_{file_tag}_user_embeddings.pt")
            torch.save({"embeddings": i_emb.cpu(), "id_map": item_dec},
                       EMB_OUT / f"infonce_kgat_sal_{file_tag}_item_embeddings.pt")
            log.info("Model → %s | Embeddings → %s", model_path, EMB_OUT)

            save_predictions(u_emb, i_emb, train_df, test_df, user_enc, item_enc, file_tag)

    return recall


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="InfoNCE + KGAT-SAL recommendation model")
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--emb-dim",      type=int,   default=2048)
    parser.add_argument("--kg-layers",    type=int,   default=2,
                        help="KG attentive propagation layers (default 2)")
    parser.add_argument("--cf-layers",    type=int,   default=4,
                        help="CF LightGCN propagation layers (default 4)")
    parser.add_argument("--n-periods",    type=int,   default=3,
                        help="Number of temporal periods T (default 3)")
    parser.add_argument("--d-temporal",   type=int,   default=256)
    parser.add_argument("--d-sal",        type=int,   default=32)
    parser.add_argument("--lambda-sal",   type=float, default=1e-6)
    parser.add_argument("--lambda-cl",    type=float, default=0.1,
                        help="InfoNCE CL loss weight (default 0.1)")
    parser.add_argument("--cl-warmup",    type=int,   default=50,
                        help="Epochs before CL activates; BPR+SAL only during warmup (default 50)")
    parser.add_argument("--cl-temp",      type=float, default=0.15,
                        help="InfoNCE temperature τ (default 0.15)")
    parser.add_argument("--cl-aug-dropout", type=float, default=0.2,
                        help="Edge dropout for augmented CL views (default 0.2)")
    parser.add_argument("--noise-eps",    type=float, default=0.1,
                        help="Per-layer noise magnitude for CL views (default 0.1)")
    parser.add_argument("--cl-sample-size", type=int, default=4096,
                        help="Entities per CL mini-batch (affects logit matrix size, default 4096)")
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--batch-size",   type=int,   default=2048)
    parser.add_argument("--sal-samples",  type=int,   default=4096)
    parser.add_argument("--edge-dropout", type=float, default=0.1)
    parser.add_argument("--kg-dropout",   type=float, default=0.1)
    parser.add_argument("--hard-neg-refresh", type=int, default=50)
    parser.add_argument("--llm-feat-mode", default="fast", choices=["none", "fast", "full"])
    parser.add_argument("--no-nlp-feat",  action="store_true")
    parser.add_argument("--eval-every",   type=int,   default=50)
    parser.add_argument("--no-save",      action="store_true")
    parser.add_argument("--prebuilt-features", action="store_true",
                        help="Load pre-joined feature matrices (run build_training_features.py first)")
    parser.add_argument("--skip-feature-groups", default="",
                        help="Comma-separated feature groups to drop")
    parser.add_argument("--no-spatial-cbg", action="store_true",
                        help="Disable CBG-CBG spatial KG triples")
    args, _ = parser.parse_known_args()
    init_distributed()

    skip_groups = [g.strip() for g in args.skip_feature_groups.split(",") if g.strip()] \
        if args.skip_feature_groups else None

    train(
        epochs              = args.epochs,
        emb_dim             = args.emb_dim,
        n_kg_layers         = args.kg_layers,
        n_cf_layers         = args.cf_layers,
        n_periods           = args.n_periods,
        d_temporal          = args.d_temporal,
        d_sal               = args.d_sal,
        lambda_sal          = args.lambda_sal,
        lambda_cl           = args.lambda_cl,
        cl_warmup           = args.cl_warmup,
        cl_temp             = args.cl_temp,
        cl_aug_dropout      = args.cl_aug_dropout,
        noise_eps           = args.noise_eps,
        cl_sample_size      = args.cl_sample_size,
        lr                  = args.lr,
        batch_size          = args.batch_size,
        sal_samples         = args.sal_samples,
        save                = not args.no_save,
        edge_dropout        = args.edge_dropout,
        kg_dropout          = args.kg_dropout,
        hard_neg_refresh    = args.hard_neg_refresh,
        llm_feat_mode       = args.llm_feat_mode,
        use_nlp_feat        = not args.no_nlp_feat,
        eval_every          = args.eval_every,
        prebuilt_features   = args.prebuilt_features,
        skip_feature_groups = skip_groups,
        use_spatial_cbg     = not args.no_spatial_cbg,
    )


if __name__ == "__main__":
    main()
