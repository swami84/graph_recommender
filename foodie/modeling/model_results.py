#!/usr/bin/env python3
"""
model_results.py — Shared utility for logging and comparing model metrics.

Appends one row per run to results/model_results.csv.
Call save_result() from any training script after final evaluation.
Call show_leaderboard() or run this file directly to print the table.

Usage:
    python -m foodie.modeling.model_results                    # print leaderboard
    python -m foodie.modeling.model_results --sort ndcg        # sort by NDCG@10
    python -m foodie.modeling.model_results --top 10           # show top N rows
"""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

RESULTS_FILE = Path(os.environ.get("FOODIE_MODEL_RESULTS", "results/model_results.csv"))

COLUMNS = [
    "timestamp",
    "model",
    "variant",
    "precision_at_10",
    "recall_at_10",
    "ndcg_at_10",
    "epochs",
    "emb_dim",
    "n_layers",
    "skip_groups",
    "notes",
]


def save_result(
    model: str,
    variant: str,
    precision: float,
    recall: float,
    ndcg: float,
    epochs: int       = 0,
    emb_dim: int      = 0,
    n_layers: int     = 4,
    skip_groups: str  = "",
    notes: str        = "",
) -> None:
    """Append one result row to results/model_results.csv."""
    RESULTS_FILE.parent.mkdir(exist_ok=True)

    write_header = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model":           model,
            "variant":         variant,
            "precision_at_10": f"{precision:.4f}",
            "recall_at_10":    f"{recall:.4f}",
            "ndcg_at_10":      f"{ndcg:.4f}",
            "epochs":          epochs,
            "emb_dim":         emb_dim,
            "n_layers":        n_layers,
            "skip_groups":     skip_groups,
            "notes":           notes,
        })


def load_results() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    with open(RESULTS_FILE, newline="") as f:
        return list(csv.DictReader(f))


def show_leaderboard(sort_by: str = "recall", top: int | None = None) -> None:
    rows = load_results()
    if not rows:
        print("No results saved yet.")
        return

    key_map = {
        "recall":    "recall_at_10",
        "ndcg":      "ndcg_at_10",
        "precision": "precision_at_10",
    }
    sort_col = key_map.get(sort_by, "recall_at_10")

    rows.sort(key=lambda r: float(r.get(sort_col, 0) or 0), reverse=True)
    if top:
        rows = rows[:top]

    best_recall = max(float(r["recall_at_10"]) for r in rows)

    col_w = {"model": 12, "variant": 26, "recall": 10, "ndcg": 9,
             "precision": 11, "epochs": 7, "emb": 8, "layers": 8,
             "skip": 22, "ts": 19}

    def hdr(s, w): return s.ljust(w)
    sep = "  "

    header = (
        sep.join([
            hdr("Model",      col_w["model"]),
            hdr("Variant",    col_w["variant"]),
            hdr("Recall@10",  col_w["recall"]),
            hdr("NDCG@10",    col_w["ndcg"]),
            hdr("Prec@10",    col_w["precision"]),
            hdr("Epochs",     col_w["epochs"]),
            hdr("EmbDim",     col_w["emb"]),
            hdr("Layers",     col_w["layers"]),
            hdr("SkipGroups", col_w["skip"]),
            hdr("Timestamp",  col_w["ts"]),
        ])
    )
    divider = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("  MODEL LEADERBOARD")
    print(f"{'='*len(header)}")
    print(header)
    print(divider)

    for r in rows:
        recall = float(r["recall_at_10"])
        marker = " ***" if recall == best_recall else ""
        print(sep.join([
            hdr(r["model"],                col_w["model"]),
            hdr(r["variant"],              col_w["variant"]),
            hdr(r["recall_at_10"],         col_w["recall"]),
            hdr(r["ndcg_at_10"],           col_w["ndcg"]),
            hdr(r["precision_at_10"],      col_w["precision"]),
            hdr(r["epochs"],               col_w["epochs"]),
            hdr(r["emb_dim"],              col_w["emb"]),
            hdr(r.get("n_layers", "4"),    col_w["layers"]),
            hdr(r.get("skip_groups", ""),  col_w["skip"]),
            hdr(r["timestamp"],            col_w["ts"]),
        ]) + marker)

    print(f"{'='*len(header)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show model results leaderboard")
    parser.add_argument("--sort", default="recall",
                        choices=["recall", "ndcg", "precision"],
                        help="Column to sort by (default: recall)")
    parser.add_argument("--top", type=int, default=None,
                        help="Show only top N rows")
    args = parser.parse_args()
    show_leaderboard(sort_by=args.sort, top=args.top)
