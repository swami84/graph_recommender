#!/usr/bin/env python3
"""Summarize three-seed core results and select the winner on validation NDCG."""

from pathlib import Path
import pandas as pd

SOURCE = Path("results/publication_core_results.csv")
OUT = Path("results/publication_core_summary.csv")
REPORT = Path("results/publication_core_summary.md")


def main():
    df = pd.read_csv(SOURCE)
    # If an interrupted condition was deliberately rerun, the newest row wins.
    df = df.sort_values("timestamp").drop_duplicates(
        ["model", "condition", "seed"], keep="last"
    )
    expected = {(m, c, s) for m in ["FeatureTwoTower", "LightGCN", "KGAT-SAL"]
                for c in ["non_llm", "full_llm"] for s in [42, 43, 44]}
    present = set(map(tuple, df[["model", "condition", "seed"]].to_records(index=False)))
    missing = expected - present
    if missing:
        raise SystemExit("Core experiment incomplete: " + ", ".join(map(str, sorted(missing))))

    metrics = ["val_hit_at_10", "val_ndcg_at_10", "test_hit_at_10", "test_ndcg_at_10"]
    summary = df.groupby(["model", "condition"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(x).rstrip("_") for x in summary.columns]
    summary.to_csv(OUT, index=False)

    paired = df.pivot(index=["model", "seed"], columns="condition", values=metrics)
    delta_rows = []
    for model in sorted(df.model.unique()):
        row = {"model": model}
        for metric in metrics:
            delta = paired.loc[model][metric]["full_llm"] - paired.loc[model][metric]["non_llm"]
            row[f"delta_{metric}_mean"] = delta.mean()
            row[f"delta_{metric}_std"] = delta.std()
        delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows)

    winner = (summary[summary.condition == "full_llm"]
              .sort_values("val_ndcg_at_10_mean", ascending=False).iloc[0])
    lines = ["# Publication core experiment summary", "",
             "Selection criterion: mean validation NDCG@10 across seeds 42, 43, and 44.", "",
             f"Selected model/feature set: **{winner['model']} / full_llm** "
             f"(validation NDCG@10 {winner['val_ndcg_at_10_mean']:.6f} ± "
             f"{winner['val_ndcg_at_10_std']:.6f}).", "", "## Mean ± SD", "",
             summary.to_markdown(index=False, floatfmt=".6f"), "",
             "## Paired LLM uplift (full_llm − non_llm)", "",
             deltas.to_markdown(index=False, floatfmt=".6f"), ""]
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines[:7]))
    print(f"\nCSV: {OUT}\nReport: {REPORT}")


if __name__ == "__main__":
    main()
