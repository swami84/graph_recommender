#!/usr/bin/env python3
"""Summarize expanded results and compare them with the frozen earlier run."""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NEW_DIR = ROOT / "results/expanded_2026-09-21"
NEW = NEW_DIR / "core_results.csv"
OLD = ROOT / "results/publication_core_results.csv"


def aggregate(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path).sort_values("timestamp").drop_duplicates(
        ["model", "condition", "seed"], keep="last"
    )
    expected = {(m, c, s) for m in ["FeatureTwoTower", "LightGCN", "KGAT-SAL"]
                for c in ["non_llm", "full_llm"] for s in [42, 43, 44]}
    present = set(map(tuple, frame[["model", "condition", "seed"]].to_records(index=False)))
    if expected - present:
        raise SystemExit(f"Incomplete experiment {path}: {sorted(expected - present)}")
    metrics = ["val_hit_at_10", "val_ndcg_at_10", "test_hit_at_10", "test_ndcg_at_10"]
    summary = frame.groupby(["model", "condition"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(x).rstrip("_") for x in summary.columns]
    return frame, summary


def main() -> None:
    _, expanded = aggregate(NEW)
    _, earlier = aggregate(OLD)
    expanded.to_csv(NEW_DIR / "core_summary.csv", index=False)
    keys = ["model", "condition"]
    comparison = expanded.merge(earlier, on=keys, suffixes=("_expanded", "_earlier"))
    for metric in ("test_hit_at_10_mean", "test_ndcg_at_10_mean",
                   "val_hit_at_10_mean", "val_ndcg_at_10_mean"):
        comparison[f"delta_{metric}"] = comparison[f"{metric}_expanded"] - comparison[f"{metric}_earlier"]
    comparison.to_csv(NEW_DIR / "comparison_with_earlier.csv", index=False)
    report = ["# Expanded dataset model comparison", "",
              "The expanded run uses the same three models, two feature conditions, three seeds, hyperparameters, validation checkpoint selection, and metrics as the earlier frozen run.",
              "The logical chronological split policy is unchanged. The expanded pipeline adds a stable review-ID tie-break for equal timestamps; therefore, the comparison is reproducible, but a small part of the delta may reflect this split-integrity correction rather than dataset expansion alone.", "",
              "## Expanded results", "", expanded.to_markdown(index=False, floatfmt=".6f"), "",
              "## Expanded minus earlier", "", comparison[keys + [
                  "delta_test_hit_at_10_mean", "delta_test_ndcg_at_10_mean",
                  "delta_val_hit_at_10_mean", "delta_val_ndcg_at_10_mean",
              ]].to_markdown(index=False, floatfmt=".6f"), ""]
    (NEW_DIR / "comparison_report.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
