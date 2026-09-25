#!/usr/bin/env python3
"""Compare frozen, re-judged, regenerated, and match-withheld GraphRAG runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCORES = ["entailment", "personalization", "usefulness", "citation_correctness", "completeness"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def flatten(rows: list[dict], label: str) -> pd.DataFrame:
    values = []
    for row in rows:
        judge = row.get("judge") or {}
        values.append({
            "condition": label,
            "sample_id": int(row["sample_id"]),
            **{key: pd.to_numeric(judge.get(key), errors="coerce") for key in SCORES},
            "unsupported_claims": pd.to_numeric(judge.get("unsupported_claims"), errors="coerce"),
            "attribution_error": bool(judge.get("attribution_error", False)),
        })
    return pd.DataFrame(values)


def paired_ci(values: pd.Series, seed: int = 20260915) -> tuple[float, float]:
    clean = values.dropna().to_numpy(float)
    if not len(clean):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(clean, size=len(clean), replace=True).mean() for _ in range(10_000)
    ])
    return tuple(np.quantile(means, [.025, .975]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--rejudged", type=Path, required=True)
    parser.add_argument("--regenerated", type=Path, required=True)
    parser.add_argument("--withheld", type=Path, required=True)
    parser.add_argument("--legacy-evidence", type=Path, required=True)
    parser.add_argument("--strict-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frozen_rows = load_jsonl(args.frozen)
    rejudged_rows = load_jsonl(args.rejudged)
    regenerated_rows = load_jsonl(args.regenerated)
    withheld_rows = load_jsonl(args.withheld)
    frames = [
        flatten(frozen_rows, "Original rubric and serialization"),
        flatten(rejudged_rows, "Corrected rubric, original generation"),
        flatten(regenerated_rows, "Corrected rubric and serialization"),
        flatten(withheld_rows, "Personalization match withheld"),
    ]

    lines = ["# GraphRAG remediation comparison", "", "## Condition summary", "",
             "| Condition | N | Entailment | Personalization | Usefulness | Citation | Completeness | Unsupported | Attribution error | Low entailment |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for frame in frames:
        label = frame.condition.iloc[0]
        def mean(column): return frame[column].mean()
        lines.append(
            f"| {label} | {len(frame)} | {mean('entailment'):.3f} | "
            f"{mean('personalization'):.3f} | {mean('usefulness'):.3f} | "
            f"{mean('citation_correctness'):.3f} | {mean('completeness'):.3f} | "
            f"{mean('unsupported_claims'):.3f} | {mean('attribution_error'):.1%} | "
            f"{(frame.entailment <= 2).mean():.1%} |"
        )

    baseline = frames[2]
    withheld = frames[3]
    paired = baseline.merge(withheld, on="sample_id", suffixes=("_with_match", "_withheld"))
    lines.extend(["", "## Paired personalization-withheld comparison", "",
                  f"Paired cases: {len(paired)}", "",
                  "| Metric | Mean delta (withheld − match) | Bootstrap 95% CI |",
                  "|---|---:|---:|"])
    for metric in [*SCORES, "unsupported_claims"]:
        delta = paired[f"{metric}_withheld"] - paired[f"{metric}_with_match"]
        low, high = paired_ci(delta)
        lines.append(f"| {metric.replace('_', ' ').title()} | {delta.mean():+.3f} | [{low:+.3f}, {high:+.3f}] |")

    legacy = load_jsonl(args.legacy_evidence)
    strict = load_jsonl(args.strict_evidence)
    legacy_matches = {int(r["sample_id"]) for r in legacy if any(n["id"] == "M1" for n in r["evidence"])}
    strict_matches = {int(r["sample_id"]) for r in strict if any(n["id"] == "M1" for n in r["evidence"])}
    lines.extend(["", "## Polarity-aware match-policy check", "",
                  f"- Legacy supported-match coverage: {len(legacy_matches)}/{len(legacy)} ({len(legacy_matches)/len(legacy):.1%})",
                  f"- Strict supported-match coverage: {len(strict_matches)}/{len(strict)} ({len(strict_matches)/len(strict):.1%})",
                  f"- Removed ambiguous matches: {len(legacy_matches-strict_matches)}",
                  f"- Newly introduced matches: {len(strict_matches-legacy_matches)}", ""])
    for sample_id in sorted(legacy_matches - strict_matches):
        row = next(r for r in legacy if int(r["sample_id"]) == sample_id)
        match = next(n for n in row["evidence"] if n["id"] == "M1")
        lines.append(f"- Sample {sample_id}: {match['fact']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
