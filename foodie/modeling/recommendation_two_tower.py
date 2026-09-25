#!/usr/bin/env python3
"""Feature-aware two-tower reference model for publication experiments.

This is deliberately non-graph: user and restaurant towers combine an ID
embedding with projected side features and are optimized with BPR. Evaluation
is full-catalog chronological leave-two-out, identical to the graph models.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from foodie.modeling.model_results import save_result
from foodie.modeling.publication_results import condition_from_skip, save_publication_result
from foodie.modeling.recommendation_gnn import (
    DEVICE, build_item_features, build_user_features, load_interaction_splits,
    ndcg_at_k, sample_negatives,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("two_tower")

MODEL_DIR = Path(os.environ.get("FOODIE_MODEL_DIR", "models"))
EMB_DIR = Path(os.environ.get("FOODIE_EMBEDDING_DIR", "data/embeddings"))
PRED_DIR = Path(os.environ.get("FOODIE_PREDICTION_DIR", "data/predictions"))


class SideTower(nn.Module):
    def __init__(self, n_entities: int, features: np.ndarray, emb_dim: int, dropout: float):
        super().__init__()
        self.id_embedding = nn.Embedding(n_entities, emb_dim)
        self.register_buffer("features", torch.from_numpy(features).float(), persistent=False)
        feat_dim = features.shape[1]
        self.feature_net = (nn.Sequential(
            nn.Linear(feat_dim, emb_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim), nn.LayerNorm(emb_dim),
        ) if feat_dim else None)
        self.feature_gate = nn.Parameter(torch.tensor(0.0)) if feat_dim else None
        nn.init.normal_(self.id_embedding.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        out = self.id_embedding(idx)
        if self.feature_net is not None:
            side = self.feature_net(self.features[idx])
            out = out + torch.sigmoid(self.feature_gate) * side
        return F.normalize(out, dim=-1)


class FeatureTwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, user_features: np.ndarray,
                 item_features: np.ndarray, emb_dim: int, dropout: float):
        super().__init__()
        self.user_tower = SideTower(n_users, user_features, emb_dim, dropout)
        self.item_tower = SideTower(n_items, item_features, emb_dim, dropout)

    def score_triplet(self, users, positives, negatives):
        u = self.user_tower(users)
        p = self.item_tower(positives)
        n = self.item_tower(negatives)
        return (u * p).sum(1), (u * n).sum(1)

    @torch.no_grad()
    def all_embeddings(self, chunk: int = 8192):
        self.eval()
        def encode(tower: SideTower):
            return torch.cat([
                tower(torch.arange(start, min(start + chunk, tower.id_embedding.num_embeddings),
                                   device=DEVICE))
                for start in range(0, tower.id_embedding.num_embeddings, chunk)
            ])
        return encode(self.user_tower), encode(self.item_tower)


def ranking_metrics(u_emb, i_emb, target_df, exclude_df, k=10):
    rel_map = target_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    excl_map = exclude_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    users = np.asarray(list(rel_map), dtype=np.int64)
    recalls, ndcgs = [], []
    for start in range(0, len(users), 2048):
        chunk = users[start:start + 2048]
        scores = (u_emb[torch.as_tensor(chunk, device=u_emb.device)] @ i_emb.T).float().cpu().numpy()
        for row_idx, uid in enumerate(chunk):
            for item in excl_map.get(int(uid), set()):
                scores[row_idx, item] = -np.inf
            top = np.argpartition(-scores[row_idx], k - 1)[:k]
            ranked = top[np.argsort(-scores[row_idx, top])].tolist()
            relevant = rel_map[int(uid)]
            recalls.append(len(relevant.intersection(ranked)) / len(relevant))
            ndcgs.append(ndcg_at_k(relevant, ranked, k))
    return {"hit": float(np.mean(np.asarray(recalls) > 0)),
            "recall": float(np.mean(recalls)), "ndcg": float(np.mean(ndcgs))}


def save_predictions(u_emb, i_emb, test_df, exclude_df, user_enc, item_enc, file_tag):
    user_dec, item_dec = ({v: k for k, v in user_enc.items()},
                          {v: k for k, v in item_enc.items()})
    exclude = exclude_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    records = []
    rows = test_df[["user_idx", "item_idx"]].to_numpy()
    for start in range(0, len(rows), 2048):
        chunk = rows[start:start + 2048]
        scores = (u_emb[torch.as_tensor(chunk[:, 0], device=u_emb.device)] @ i_emb.T).float().cpu().numpy()
        for row_idx, (uid, true_item) in enumerate(chunk):
            for item in exclude.get(int(uid), set()):
                scores[row_idx, item] = -np.inf
            top_idx = np.argpartition(-scores[row_idx], 9)[:10]
            top = top_idx[np.argsort(-scores[row_idx, top_idx])].tolist()
            records.append({"user_idx": int(uid), "true_item_idx": int(true_item),
                            "top10_item_idxs": top,
                            "contributor_id": user_dec[int(uid)],
                            "true_place_id": item_dec[int(true_item)],
                            "top10_place_ids": [item_dec[x] for x in top],
                            "rank": top.index(int(true_item)) + 1 if int(true_item) in top else None})
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(PRED_DIR / f"two_tower_{file_tag}_predictions.parquet",
                                     index=False)


def train(epochs=100, emb_dim=256, lr=1e-3, batch_size=4096, dropout=0.1,
          eval_every=10, skip_feature_groups=None, seed=42, save=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    train_df, val_df, test_df, user_enc, item_enc = load_interaction_splits()
    user_feat = build_user_features(train_df, user_enc, prebuilt=True,
                                    skip_groups=skip_feature_groups)
    item_feat = build_item_features(item_enc, llm_feat_mode="full", prebuilt=True,
                                    skip_groups=skip_feature_groups)
    model = FeatureTwoTower(len(user_enc), len(item_enc), user_feat, item_feat,
                            emb_dim, dropout).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 50)
    best_ndcg, best_state, best_val = -1.0, None, None

    log.info("Training feature Two-Tower | users=%s items=%s epochs=%d seed=%d",
             f"{len(user_enc):,}", f"{len(item_enc):,}", epochs, seed)
    for epoch in range(1, epochs + 1):
        model.train()
        sampled = sample_negatives(train_df, len(item_enc), rng=rng)
        order = rng.permutation(len(sampled))
        user_np = sampled["user_idx"].to_numpy(dtype=np.int64, copy=True)
        pos_np = sampled["item_idx"].to_numpy(dtype=np.int64, copy=True)
        neg_np = sampled["neg_idx"].to_numpy(dtype=np.int64, copy=True)
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            users = torch.as_tensor(user_np[idx], device=DEVICE)
            pos = torch.as_tensor(pos_np[idx], device=DEVICE)
            neg = torch.as_tensor(neg_np[idx], device=DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pos_score, neg_score = model.score_triplet(users, pos, neg)
            loss = -F.logsigmoid(pos_score - neg_score).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0:
            log.info("Epoch %d/%d loss=%.5f", epoch, epochs, epoch_loss / len(sampled))
        if eval_every and (epoch % eval_every == 0 or epoch == epochs):
            u_emb, i_emb = model.all_embeddings()
            metrics = ranking_metrics(u_emb, i_emb, val_df, train_df)
            log.info("  val Hit@10=%.4f NDCG@10=%.4f", metrics["hit"], metrics["ndcg"])
            if metrics["ndcg"] > best_ndcg:
                best_ndcg, best_val = metrics["ndcg"], metrics
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    u_emb, i_emb = model.all_embeddings()
    seen_df = pd.concat([train_df, val_df], ignore_index=True)
    metrics = ranking_metrics(u_emb, i_emb, test_df, seen_df)
    skip = ",".join(skip_feature_groups or [])
    condition = f"skip_{skip.replace(',', '_')}" if skip else "all"
    file_tag = f"{condition}_pubsplit_seed{seed}"
    print(f"\n=== Feature Two-Tower Results ===\n  Hit@10:  {metrics['hit']:.4f}"
          f"\n  NDCG@10: {metrics['ndcg']:.4f}")
    val_note = (f"val_hit@10={best_val['hit']:.6f}; val_ndcg@10={best_val['ndcg']:.6f}"
                if best_val else "no_periodic_validation")
    save_result("FeatureTwoTower", condition, metrics["hit"] / 10, metrics["recall"],
                metrics["ndcg"], epochs, emb_dim, 0, skip,
                notes=f"seed={seed}; checkpoint=val_ndcg@10; {val_note}")
    save_publication_result("FeatureTwoTower", condition_from_skip(skip),
                            seed, best_val, metrics, epochs, emb_dim)
    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        EMB_DIR.mkdir(exist_ok=True)
        torch.save(model.state_dict(), MODEL_DIR / f"two_tower_{file_tag}.pt")
        torch.save({"embeddings": u_emb.cpu(), "id_map": {v: k for k, v in user_enc.items()}},
                   EMB_DIR / f"two_tower_{file_tag}_user_embeddings.pt")
        torch.save({"embeddings": i_emb.cpu(), "id_map": {v: k for k, v in item_enc.items()}},
                   EMB_DIR / f"two_tower_{file_tag}_item_embeddings.pt")
        save_predictions(u_emb, i_emb, test_df, seen_df, user_enc, item_enc, file_tag)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-feature-groups", default="")
    parser.add_argument("--no-save", action="store_true")
    # Accepted for a uniform experiment-runner interface.
    parser.add_argument("--prebuilt-features", action="store_true")
    parser.add_argument("--llm-feat-mode", choices=["none", "fast", "full"], default="full")
    parser.add_argument("--publication-split", action="store_true")
    args = parser.parse_args()
    skip = [x.strip() for x in args.skip_feature_groups.split(",") if x.strip()]
    train(args.epochs, args.emb_dim, args.lr, args.batch_size, args.dropout,
          args.eval_every, skip, args.seed, not args.no_save)


if __name__ == "__main__":
    main()
