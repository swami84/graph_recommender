#!/usr/bin/env python3
"""Build a nationally balanced top-density CBG queue.

The default output contains exactly 10,000 CBGs: the densest CBG in every state
and DC is protected, and all remaining slots are filled by national population
density rank. Previously queried CBGs are identified by both exact legacy GEOID
and a spatial crosswalk of prior Google query coordinates to 2024 boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


def _geoid_column(frame: gpd.GeoDataFrame) -> str:
    for name in ("GEOID", "GEOID10"):
        if name in frame.columns:
            return name
    raise ValueError("CBG geometry has no GEOID/GEOID10 column")


def prior_coverage(
    query_log: Path, done_dir: Path, geometry_dir: Path, legacy_geometry_dir: Path
) -> set[str]:
    done = {path.stem.zfill(12) for path in done_dir.glob("*.done")}
    covered = set(done)

    # A prior terminal marker means the entire old-vintage CBG is terminal,
    # including the intentionally empty/early-stopped CBGs. Crosswalk those
    # polygons to 2024 using each current CBG's representative point. Query
    # points alone miss split/boundary-changed portions of large old CBGs.
    for state in sorted({cbg[:2] for cbg in done}):
        current_path = geometry_dir / f"cb_2024_{state}_bg_500k.zip"
        legacy_path = legacy_geometry_dir / f"cb_2019_{state}_bg_500k.zip"
        if not current_path.exists() or not legacy_path.exists():
            continue
        current = gpd.read_file(current_path)[["GEOID", "geometry"]].rename(
            columns={"GEOID": "cbg_str"}
        )
        legacy = gpd.read_file(legacy_path)
        geoid_col = _geoid_column(legacy)
        legacy[geoid_col] = legacy[geoid_col].astype(str).str.zfill(12)
        legacy = legacy[legacy[geoid_col].isin(done)][[geoid_col, "geometry"]]
        if legacy.empty:
            continue
        representative_points = gpd.GeoDataFrame(
            current[["cbg_str"]].copy(),
            geometry=current.geometry.representative_point(),
            crs=current.crs,
        ).to_crs(legacy.crs)
        joined = gpd.sjoin(
            representative_points, legacy, how="inner", predicate="intersects"
        )
        covered.update(joined["cbg_str"].astype(str).str.zfill(12))

    if not query_log.exists():
        return covered
    calls = pd.read_csv(
        query_log,
        usecols=["cbg", "h3_index", "lat", "lng"],
        dtype={"cbg": str},
    ).dropna(subset=["lat", "lng"]).drop_duplicates("h3_index")
    calls["state_fips"] = calls["cbg"].str.zfill(12).str[:2]

    for state, state_calls in calls.groupby("state_fips"):
        geometry_path = geometry_dir / f"cb_2024_{state}_bg_500k.zip"
        if not geometry_path.exists():
            continue
        cbgs = gpd.read_file(geometry_path)[["GEOID", "geometry"]].rename(
            columns={"GEOID": "cbg_str"}
        )
        points = gpd.GeoDataFrame(
            state_calls[["h3_index"]].copy(),
            geometry=gpd.points_from_xy(state_calls["lng"], state_calls["lat"]),
            crs="EPSG:4326",
        ).to_crs(cbgs.crs)
        joined = gpd.sjoin(points, cbgs, how="left", predicate="within")
        covered.update(joined["cbg_str"].dropna().astype(str).str.zfill(12))
    return covered


def select_top_density(core: pd.DataFrame, top_n: int, min_states: int) -> pd.DataFrame:
    core = core.sort_values("population_density", ascending=False).copy()
    core["national_density_rank"] = range(1, len(core) + 1)
    strict_top = core.head(top_n)
    if strict_top["state_fips"].nunique() >= min_states:
        selected = strict_top.copy()
        selected["state_champion"] = False
        return selected

    champions = core.groupby("state_fips", as_index=False).head(1)
    missing = champions[~champions["state_fips"].isin(strict_top["state_fips"])]
    needed = max(0, min_states - strict_top["state_fips"].nunique())
    protected = missing.nlargest(needed, "population_density")
    remaining = core[~core["cbg_str"].isin(protected["cbg_str"])].head(
        top_n - len(protected)
    )
    selected = pd.concat([protected, remaining], ignore_index=True).drop_duplicates("cbg_str")
    selected["state_champion"] = selected["cbg_str"].isin(protected["cbg_str"])
    return selected.sort_values("population_density", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=10_000)
    parser.add_argument("--min-states", type=int, default=51,
                        help="Minimum state/DC coverage; use 0 for a strict national top-N")
    parser.add_argument("--plan", type=Path, default=Path("data/national_density_cbgs.csv"))
    parser.add_argument("--query-log", type=Path, default=Path("data/hex_api_call_log.csv"))
    parser.add_argument("--done-dir", type=Path, default=Path("data/hex_restaurants/.done"))
    parser.add_argument("--geometry-dir", type=Path, default=Path("data/census_geometry_2024"))
    parser.add_argument(
        "--legacy-geometry-dir",
        type=Path,
        default=Path.home() / ".cache" / "pygris",
        help="Cached 2019 CBG polygons used to crosswalk terminal legacy CBGs",
    )
    parser.add_argument("--output", type=Path, default=Path("data/top10000_density_cbgs.csv"))
    parser.add_argument(
        "--incremental-output",
        type=Path,
        default=Path("data/top10000_density_cbgs_incremental.csv"),
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("data/top10000_density_cbgs_summary.json")
    )
    args = parser.parse_args()

    plan = pd.read_csv(args.plan, dtype={"cbg_str": str, "state_fips": str})
    plan["cbg_str"] = plan["cbg_str"].str.zfill(12)
    core = plan[plan["selection_reason"] == "high_density"].copy()
    selected = select_top_density(core, args.top_n, args.min_states)
    covered = prior_coverage(
        args.query_log, args.done_dir, args.geometry_dir, args.legacy_geometry_dir
    )
    selected["previously_queried"] = selected["cbg_str"].isin(covered)
    selected["pending"] = ~selected["previously_queried"]
    incremental = selected[selected["pending"]].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    incremental.to_csv(args.incremental_output, index=False)
    summary = {
        "top_n": int(len(selected)),
        "minimum_states_requested": args.min_states,
        "states_selected": int(selected["state_fips"].nunique()),
        "previously_queried": int(selected["previously_queried"].sum()),
        "incremental_cbgs": int(len(incremental)),
        "incremental_states": int(incremental["state_fips"].nunique()),
        "incremental_h3_calls_upper_bound": int(incremental["h3_cell_count"].sum()),
        "minimum_selected_density": float(selected["population_density"].min()),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output}, {args.incremental_output}, and {args.summary}")


if __name__ == "__main__":
    main()
