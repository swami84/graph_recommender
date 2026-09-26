#!/usr/bin/env python3
"""Generate README figures from the frozen expanded-experiment results."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from article.figures.publication_style import (  # noqa: E402
    CORAL,
    INK,
    MUTED,
    TEAL,
    apply_style,
    clean_axis,
    save,
    title,
)


RESULT_DIR = ROOT / "results" / "expanded_2026-09-21"
OUTPUT_DIR = ROOT / "docs" / "figures"
MODEL_ORDER = ["FeatureTwoTower", "LightGCN", "KGAT-SAL"]
DISPLAY_NAMES = {
    "FeatureTwoTower": "Two-Tower",
    "LightGCN": "LightGCN",
    "KGAT-SAL": "KGAT-SAL",
}


def model_feature_comparison() -> None:
    results = pd.read_csv(RESULT_DIR / "core_results.csv")
    summary = (
        results.groupby(["model", "condition"])
        .agg(
            hit_mean=("test_hit_at_10", "mean"),
            hit_std=("test_hit_at_10", "std"),
            ndcg_mean=("test_ndcg_at_10", "mean"),
            ndcg_std=("test_ndcg_at_10", "std"),
        )
        .reset_index()
    )

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    conditions = [
        ("non_llm", "Conventional features", "#AEB7C3"),
        ("full_llm", "+ LLM features", TEAL),
    ]

    for axis, metric, label in [
        (axes[0], "ndcg", "NDCG@10"),
        (axes[1], "hit", "Hit@10"),
    ]:
        for offset, (condition, legend, color) in zip((-width / 2, width / 2), conditions):
            subset = summary[summary.condition == condition].set_index("model")
            means = [subset.loc[model, f"{metric}_mean"] for model in MODEL_ORDER]
            stds = [subset.loc[model, f"{metric}_std"] for model in MODEL_ORDER]
            bars = axis.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=3,
                color=color,
                label=legend,
                edgecolor="white",
                linewidth=0.7,
            )
            axis.bar_label(bars, fmt="%.4f", padding=4, fontsize=9, color=INK)
        axis.set_xticks(x, [DISPLAY_NAMES[model] for model in MODEL_ORDER])
        axis.set_ylabel(label)
        axis.set_ylim(0, summary[f"{metric}_mean"].max() * 1.32)
        clean_axis(axis)

    axes[0].legend(loc="upper left")
    title(
        fig,
        "Expanded-dataset ranking performance",
        "Mean full-catalogue test metrics ± sample SD across seeds 42, 43, and 44",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save(fig, OUTPUT_DIR / "model_feature_comparison.png")


def proximity_uplift() -> None:
    results = pd.read_csv(RESULT_DIR / "core_results.csv")
    raw = results[
        (results.model == "KGAT-SAL") & (results.condition == "full_llm")
    ]
    proximity = pd.read_csv(RESULT_DIR / "expanded_proximity_test_by_seed.csv")
    values = {
        "NDCG@10": (
            raw.test_ndcg_at_10.mean(),
            proximity.ndcg_at_10.mean(),
            raw.test_ndcg_at_10.std(),
            proximity.ndcg_at_10.std(),
        ),
        "Hit@10": (
            raw.test_hit_at_10.mean(),
            proximity.hit_at_10.mean(),
            raw.test_hit_at_10.std(),
            proximity.hit_at_10.std(),
        ),
    }

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))
    for axis, (metric, (raw_mean, prox_mean, raw_std, prox_std)) in zip(axes, values.items()):
        bars = axis.bar(
            [0, 1],
            [raw_mean, prox_mean],
            yerr=[raw_std, prox_std],
            capsize=4,
            color=[TEAL, CORAL],
            width=0.62,
            edgecolor="white",
            linewidth=0.7,
        )
        axis.bar_label(bars, fmt="%.4f", padding=5, fontsize=10, color=INK)
        uplift = (prox_mean / raw_mean - 1) * 100
        axis.text(
            0.5,
            max(raw_mean, prox_mean) * 1.17,
            f"+{uplift:.1f}%",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=CORAL,
        )
        axis.set_xticks([0, 1], ["Raw KGAT-SAL", "+ proximity"])
        axis.set_ylabel(metric)
        axis.set_ylim(0, prox_mean * 1.38)
        clean_axis(axis)

    title(
        fig,
        "Effect of validation-tuned proximity reranking",
        "Expanded-dataset test metrics; hyperparameters selected without test-set access",
    )
    save(fig, OUTPUT_DIR / "proximity_uplift.png")


def main() -> None:
    model_feature_comparison()
    proximity_uplift()
    print(f"Wrote README figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
