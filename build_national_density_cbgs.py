#!/usr/bin/env python3
"""Build a national high-population-density CBG coverage plan.

The plan contains every ACS 2019 block group above a configurable population
density threshold plus its directly adjacent block groups.  No Google API is
called.  The output also records whether each CBG was already collected and
the number of H3 resolution-9 cells that the current Places collector would
query in the worst case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd
import requests
from shapely.geometry import mapping


ACS_URL = (
    "https://www2.census.gov/programs-surveys/acs/summary_file/2024/"
    "table-based-SF/data/5YRData/acsdt5y2024-b01003.dat"
)
GEOMETRY_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2024/shp/"
    "cb_2024_{state}_bg_500k.zip"
)
DEFAULT_GEOMETRY_CACHE = Path("data/census_geometry_2024")
DEFAULT_POPULATION_FILE = Path("data/acsdt5y2024-b01003.dat")
DEFAULT_DONE_DIR = Path("data/hex_restaurants/.done")
DEFAULT_OUTPUT = Path("data/national_density_cbgs.csv")
DEFAULT_SUMMARY = Path("data/national_density_cbgs_summary.json")
HTTP_HEADERS = {
    "User-Agent": "foodie-revamp-density-audit/1.0 (research contact: local project)"
}

# Fifty states + DC. Puerto Rico is intentionally excluded from the stated
# national-coverage target, but can be included with --include-puerto-rico.
STATE_FIPS = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36",
    "37", "38", "39", "40", "41", "42", "44", "45", "46", "47", "48",
    "49", "50", "51", "53", "54", "55", "56",
]


def load_population_bulk(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Downloading ACS 2024 population table to {path}", flush=True)
        response = requests.get(ACS_URL, headers=HTTP_HEADERS, timeout=300)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    frame = pd.read_csv(path, sep="|", dtype={"GEO_ID": str})
    frame = frame[frame["GEO_ID"].str.startswith("1500000US")].copy()
    frame["cbg_str"] = frame["GEO_ID"].str.removeprefix("1500000US")
    frame["population"] = pd.to_numeric(frame["B01003_E001"], errors="coerce")
    return frame[["cbg_str", "population"]]


def load_state_geometry(cache_dir: Path, state_fips: str) -> gpd.GeoDataFrame:
    path = cache_dir / f"cb_2024_{state_fips}_bg_500k.zip"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = GEOMETRY_URL.format(state=state_fips)
        print(f"  downloading {url}", flush=True)
        response = requests.get(url, headers=HTTP_HEADERS, timeout=180)
        response.raise_for_status()
        path.write_bytes(response.content)
    frame = gpd.read_file(path)[["GEOID", "ALAND", "geometry"]].rename(
        columns={"GEOID": "cbg_str", "ALAND": "land_area_m2"}
    )
    frame["cbg_str"] = frame["cbg_str"].astype(str).str.zfill(12)
    frame["land_area_m2"] = pd.to_numeric(frame["land_area_m2"], errors="coerce")
    return frame


def directly_adjacent_cbgs(
    all_cbgs: gpd.GeoDataFrame, dense_cbgs: gpd.GeoDataFrame
) -> set[str]:
    if dense_cbgs.empty:
        return set()
    # Intersects includes each dense polygon itself; callers subtract the core.
    joined = gpd.sjoin(
        all_cbgs[["cbg_str", "geometry"]],
        dense_cbgs[["geometry"]],
        how="inner",
        predicate="intersects",
    )
    return set(joined["cbg_str"])


def h3_cell_count(geometry, resolution: int) -> int:
    if geometry is None or geometry.is_empty:
        return 0
    if geometry.geom_type == "Polygon":
        return max(1, len(h3.geo_to_cells(mapping(geometry), resolution)))
    if geometry.geom_type == "MultiPolygon":
        cells: set[str] = set()
        for polygon in geometry.geoms:
            cells.update(h3.geo_to_cells(mapping(polygon), resolution))
        return max(1, len(cells))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density-threshold", type=float, default=3000.0)
    parser.add_argument("--h3-resolution", type=int, default=9)
    parser.add_argument("--geometry-cache", type=Path, default=DEFAULT_GEOMETRY_CACHE)
    parser.add_argument("--population-file", type=Path, default=DEFAULT_POPULATION_FILE)
    parser.add_argument("--done-dir", type=Path, default=DEFAULT_DONE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--include-puerto-rico", action="store_true")
    args = parser.parse_args()

    state_fips = STATE_FIPS + (["72"] if args.include_puerto_rico else [])
    already_done = {path.stem for path in args.done_dir.glob("*.done")}
    state_outputs: list[gpd.GeoDataFrame] = []
    population_all = load_population_bulk(args.population_file)

    for state in state_fips:
        print(f"State {state}: calculating density and adjacency", flush=True)
        geometry = load_state_geometry(args.geometry_cache, state)
        population = population_all[population_all["cbg_str"].str[:2] == state]
        cbgs = geometry.merge(population, on="cbg_str", how="left")
        cbgs["population"] = cbgs["population"].fillna(0.0)
        cbgs["land_area_km2"] = cbgs["land_area_m2"] / 1_000_000.0
        cbgs["population_density"] = cbgs["population"] / cbgs[
            "land_area_km2"
        ].where(cbgs["land_area_km2"] > 0)

        dense = cbgs[cbgs["population_density"] >= args.density_threshold].copy()
        adjacent_ids = directly_adjacent_cbgs(cbgs, dense) - set(dense["cbg_str"])
        selected = cbgs[
            cbgs["cbg_str"].isin(set(dense["cbg_str"]) | adjacent_ids)
        ].copy()
        dense_ids = set(dense["cbg_str"])
        selected["selection_reason"] = selected["cbg_str"].map(
            lambda value: "high_density" if value in dense_ids else "adjacent"
        )
        selected["state_fips"] = state
        selected["already_collected"] = selected["cbg_str"].isin(already_done)
        selected["h3_cell_count"] = selected.geometry.map(
            lambda value: h3_cell_count(value, args.h3_resolution)
        )
        state_outputs.append(selected)
        print(
            f"  dense={len(dense):,} adjacent={len(adjacent_ids):,} "
            f"selected={len(selected):,}",
            flush=True,
        )

    plan = pd.concat(state_outputs, ignore_index=True)
    plan = plan.drop(columns=["geometry"]).sort_values(
        ["state_fips", "selection_reason", "population_density"],
        ascending=[True, False, False],
    )
    plan["pending"] = ~plan["already_collected"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(args.output, index=False)

    pending = plan[plan["pending"]]
    summary = {
        "acs_release": {"id": "acs2024_5yr", "name": "ACS 2024 5-year", "years": "2020-2024"},
        "geometry_year": 2024,
        "density_threshold_people_per_km2": args.density_threshold,
        "surrounding_definition": "direct polygon intersection/adjacency",
        "states_in_scope": int(plan["state_fips"].nunique()),
        "states_with_high_density_cbgs": int(
            plan.loc[plan["selection_reason"] == "high_density", "state_fips"].nunique()
        ),
        "high_density_cbgs": int((plan["selection_reason"] == "high_density").sum()),
        "adjacent_cbgs": int((plan["selection_reason"] == "adjacent").sum()),
        "selected_cbgs": int(len(plan)),
        "already_collected_cbgs": int(plan["already_collected"].sum()),
        "pending_cbgs": int(plan["pending"].sum()),
        "pending_h3_calls_upper_bound": int(pending["h3_cell_count"].sum()),
        "h3_resolution": args.h3_resolution,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output} and {args.summary}")


if __name__ == "__main__":
    main()
