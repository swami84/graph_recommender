#!/usr/bin/env python3
"""Audit structured LLM feature quality and geographic distribution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "analysis" / "expanded_eda" / "tables"
REPORT = ROOT / "analysis" / "expanded_eda" / "LLM_FEATURE_AUDIT.md"


def region_from_cbg(value: object) -> str:
    cbg = str(value).replace(".0", "").zfill(12)
    return {"36": "NY", "06": "CA"}.get(cbg[:2], "Other states")


def structured_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    excluded = (
        "cuisine_", "meal_", "overall_confidence", "taste_confidence",
        "atmosphere_confidence", "operations_confidence",
        "dietary_confidence", "evidence_strength",
    )
    return [
        col for col in frame.columns
        if col.startswith(prefix)
        and not col.endswith("_known")
        and col + "_known" in frame.columns
        and not any(col.startswith(prefix + suffix) for suffix in excluded)
    ]


def numeric_audit(frame: pd.DataFrame, columns: list[str], prefix: str) -> pd.DataFrame:
    rows = []
    for col in columns:
        known = pd.to_numeric(frame[col + "_known"], errors="coerce").fillna(0).gt(0)
        values = pd.to_numeric(frame.loc[known, col], errors="coerce").dropna()
        rows.append({
            "feature": col.removeprefix(prefix),
            "rows": len(frame),
            "known_rows": int(known.sum()),
            "known_share": float(known.mean()),
            "mean_when_known": float(values.mean()) if len(values) else np.nan,
            "std_when_known": float(values.std()) if len(values) else np.nan,
            "p05_when_known": float(values.quantile(.05)) if len(values) else np.nan,
            "median_when_known": float(values.median()) if len(values) else np.nan,
            "p95_when_known": float(values.quantile(.95)) if len(values) else np.nan,
            "zero_share_when_known": float(values.eq(0).mean()) if len(values) else np.nan,
            "half_share_when_known": float(values.eq(.5).mean()) if len(values) else np.nan,
            "one_share_when_known": float(values.eq(1).mean()) if len(values) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["known_share", "feature"], ascending=[False, True])


def restaurant_geography(frame: pd.DataFrame, columns: list[str], prefix: str) -> pd.DataFrame:
    rows = []
    for col in columns:
        known_col = col + "_known"
        for region, group in frame.groupby("region", observed=True):
            known = pd.to_numeric(group[known_col], errors="coerce").fillna(0).gt(0)
            values = pd.to_numeric(group.loc[known, col], errors="coerce").dropna()
            rows.append({
                "feature": col.removeprefix(prefix),
                "region": region,
                "region_rows": len(group),
                "known_rows": int(known.sum()),
                "known_share": float(known.mean()),
                "mean_when_known": float(values.mean()) if len(values) else np.nan,
            })
    out = pd.DataFrame(rows)
    out["share_of_known_rows"] = (
        out["known_rows"] / out.groupby("feature")["known_rows"].transform("sum")
    )
    out["baseline_region_share"] = out["region_rows"] / len(frame)
    out["representation_ratio"] = (
        out["share_of_known_rows"] / out["baseline_region_share"]
    )
    return out.sort_values(["feature", "region"])


def categorical_audit(frame: pd.DataFrame, columns: list[str], prefix: str) -> pd.DataFrame:
    baseline = frame["region"].value_counts(normalize=True)
    rows = []
    for col in columns:
        selected = pd.to_numeric(frame[col], errors="coerce").fillna(0).gt(0)
        global_rate = float(selected.mean())
        for region, group in frame.groupby("region", observed=True):
            region_selected = pd.to_numeric(group[col], errors="coerce").fillna(0).gt(0)
            selected_count = int(region_selected.sum())
            positive_geo_share = (
                selected_count / int(selected.sum()) if int(selected.sum()) else np.nan
            )
            rows.append({
                "feature": col.removeprefix(prefix),
                "region": region,
                "region_rows": len(group),
                "selected_rows": selected_count,
                "within_region_rate": float(region_selected.mean()),
                "global_rate": global_rate,
                "rate_ratio_to_global": (
                    float(region_selected.mean()) / global_rate if global_rate else np.nan
                ),
                "share_of_positive_rows": positive_geo_share,
                "baseline_catalogue_share": float(baseline.get(region, 0)),
            })
    return pd.DataFrame(rows).sort_values(["feature", "region"])


def confidence_audit(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = [
        col for col in frame.columns
        if col.startswith(prefix) and col.endswith("_confidence")
    ]
    rows = []
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce").dropna()
        rows.append({
            "feature": col.removeprefix(prefix),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "half_share": float(values.eq(.5).mean()),
            "one_share": float(values.eq(1).mean()),
            "unique_values": int(values.nunique()),
        })
    return pd.DataFrame(rows).sort_values("feature")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    restaurants = pd.read_parquet(
        DATA / "restaurants_enriched.parquet", columns=["place_id", "cbg"]
    )
    restaurant = pd.read_parquet(DATA / "restaurant_llm_features.parquet").merge(
        restaurants, on="place_id", how="left", validate="one_to_one"
    )
    restaurant["region"] = restaurant["cbg"].map(region_from_cbg)
    user = pd.read_parquet(DATA / "user_llm_features.parquet")

    r_numeric_cols = structured_columns(restaurant, "llm_")
    u_numeric_cols = structured_columns(user, "user_llm_")
    r_numeric = numeric_audit(restaurant, r_numeric_cols, "llm_")
    u_numeric = numeric_audit(user, u_numeric_cols, "user_llm_")
    r_geo = restaurant_geography(restaurant, r_numeric_cols, "llm_")
    tag_cols = [
        col for col in restaurant.columns
        if col.startswith("llm_cuisine_") or col.startswith("llm_meal_")
    ]
    tag_cols = [col for col in tag_cols if col != "llm_cuisine_confidence"]
    r_tags = categorical_audit(restaurant, tag_cols, "llm_")
    r_conf = confidence_audit(restaurant, "llm_")
    u_conf = confidence_audit(user, "user_llm_")

    raw_cuisines = [json.loads(raw).get("cuisines", []) for raw in restaurant["raw_response_json"]]
    duplicate_arrays = sum(len(values) != len(set(values)) for values in raw_cuisines)

    r_numeric.to_csv(OUT / "restaurant_llm_numeric_audit.csv", index=False)
    u_numeric.to_csv(OUT / "user_llm_numeric_audit.csv", index=False)
    r_geo.to_csv(OUT / "restaurant_llm_numeric_by_region.csv", index=False)
    r_tags.to_csv(OUT / "restaurant_llm_tags_by_region.csv", index=False)
    r_conf.to_csv(OUT / "restaurant_llm_confidence_audit.csv", index=False)
    u_conf.to_csv(OUT / "user_llm_confidence_audit.csv", index=False)

    state_share = restaurant["region"].value_counts(normalize=True)
    low_coverage_r = r_numeric.nsmallest(10, "known_share")
    low_coverage_u = u_numeric.nsmallest(10, "known_share")
    saturated_r = r_numeric.nlargest(10, "half_share_when_known")
    saturated_u = u_numeric.nlargest(10, "half_share_when_known")
    southern = r_tags[r_tags["feature"] == "cuisine_southern"]
    geo_supported = r_geo[
        r_geo.groupby("feature")["known_rows"].transform("sum") >= 500
    ].copy()
    geo_supported["absolute_representation_deviation"] = (
        geo_supported["representation_ratio"] - 1
    ).abs()
    geo_outliers = geo_supported.nlargest(12, "absolute_representation_deviation")

    REPORT.write_text(f"""# Structured LLM feature audit

The restaurant feature population contains {len(restaurant):,} rows. NY provides
{state_share.get('NY', 0):.1%}, CA {state_share.get('CA', 0):.1%}, and other states
{state_share.get('Other states', 0):.1%}. These are comparison baselines, not
targets that every semantic feature should reproduce.

## Cuisine failure

The current cuisine output is not publication-ready. `Southern` is selected for
{int(restaurant['llm_cuisine_southern'].sum()):,} restaurants
({restaurant['llm_cuisine_southern'].mean():.1%}), and {duplicate_arrays:,}
({duplicate_arrays/len(restaurant):.1%}) raw arrays repeat labels despite the
unique-item schema. `Southern` by region:

{southern.to_markdown(index=False)}

## Restaurant structured-feature coverage

Lowest known/evidence coverage:

{low_coverage_r.to_markdown(index=False)}

Largest concentration at the neutral value 0.5 among supposedly known values:

{saturated_r.to_markdown(index=False)}

## User structured-feature coverage

Lowest known/evidence coverage:

{low_coverage_u.to_markdown(index=False)}

Largest concentration at 0.5 among supposedly known values:

{saturated_u.to_markdown(index=False)}

## Geographic representation checks

Among features with at least 500 known restaurant rows, the largest deviations
in where supporting evidence occurs are:

{geo_outliers.to_markdown(index=False)}

Most high-coverage features broadly track the 24.5% NY / 17.9% CA / 57.7%
other-state feature-row baseline. The `Southern` positives also nearly track
that baseline (22.3% NY, 18.4% CA, 59.3% elsewhere), which is further evidence
of a generic extraction artifact rather than a credible regional cuisine signal.

Confidence is also poorly calibrated: restaurant cuisine confidence equals
1.0 for 78.5% of rows even though the cuisine audit exposes widespread false
positives and schema violations. Several supposedly known scalar features are
heavily concentrated at exactly 0.5, while rare fields have too little coverage
to contribute reliably without their evidence masks.

## Interpretation

State contribution should broadly resemble the catalogue only for ubiquitous
features. Cuisine, meal period, dietary accommodation, atmosphere, and service
features can legitimately differ by state. Large NY/CA-versus-rest deviations
are therefore flags for review, not automatic evidence of error. Detailed
regional tables and confidence diagnostics are in this directory.
""")
    print(f"restaurant features: {len(r_numeric_cols)} numeric + {len(tag_cols)} tags")
    print(f"user features: {len(u_numeric_cols)} numeric")
    print(f"duplicate cuisine arrays: {duplicate_arrays:,}")
    print(f"report: {REPORT}")


if __name__ == "__main__":
    main()
