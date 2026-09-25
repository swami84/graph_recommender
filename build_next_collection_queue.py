#!/usr/bin/env python3
"""Build the ordered, spend-bounded next Places collection queue."""

from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd

from build_national_density_cbgs import h3_cell_count


SOURCE = Path("data/national_density_budget_queue_2026-09.csv")
POPULATION_FILE = Path("data/acsdt5y2024-b01003.dat")
GEOMETRY_DIR = Path("data/census_geometry_2024")
INDEX_FILE = Path("data/hex_restaurants_index.csv")
DONE_DIR = Path("data/hex_restaurants/.done")
OUTPUT = Path("data/next_collection_queue_2026-09.csv")

# Keep a $6 buffer below the user's remaining $86 credit balance.
UNIT_COST_USD = 0.032
MAX_RUN_COST_USD = 80.0
MAX_PLANNED_CALLS = int(MAX_RUN_COST_USD / UNIT_COST_USD)
MIN_COUNTY_COVERAGE = 0.25
BALANCED_FILL_CEILING = 0.50

TOP_DENSITY_COUNTIES = {
    "36061": "New York County, NY (Manhattan)",
    "36047": "Kings County, NY (Brooklyn)",
    "36005": "Bronx County, NY",
    "36081": "Queens County, NY",
    "06075": "San Francisco County, CA",
    "34017": "Hudson County, NJ",
    "25025": "Suffolk County, MA",
    "42101": "Philadelphia County, PA",
    "11001": "District of Columbia",
    "51510": "Alexandria city, VA",
    "51013": "Arlington County, VA",
    "36085": "Richmond County, NY (Staten Island)",
    "51610": "Falls Church city, VA",
    "24510": "Baltimore city, MD",
    "34013": "Essex County, NJ",
    "34039": "Union County, NJ",
    "51685": "Manassas Park city, VA",
    "17031": "Cook County, IL",
    "36059": "Nassau County, NY",
    "08031": "Denver County, CO",
}

TOURISM_ROWS = [
    {
        "cbg_str": "230050003002", "land_area_m2": 162853,
        "population": 598.0, "land_area_km2": 0.162853,
        "population_density": 3672.0232356788024,
        "selection_reason": "tourism_priority", "state_fips": "23",
        "already_collected": False, "h3_cell_count": 1, "pending": True,
    },
    {
        "cbg_str": "230099659002", "land_area_m2": 14142699,
        "population": 1051.0, "land_area_km2": 14.142699,
        "population_density": 74.31396227834588,
        "selection_reason": "tourism_priority", "state_fips": "23",
        "already_collected": False, "h3_cell_count": 1, "pending": True,
    },
    {
        "cbg_str": "471550811012", "land_area_m2": 3427380,
        "population": 345.0, "land_area_km2": 3.42738,
        "population_density": 100.65997934282163,
        "selection_reason": "tourism_priority", "state_fips": "47",
        "already_collected": False, "h3_cell_count": 1, "pending": True,
    },
    {
        "cbg_str": "471550810022", "land_area_m2": 5100476,
        "population": 1587.0, "land_area_km2": 5.100476,
        "population_density": 311.14743016142023,
        "selection_reason": "tourism_priority", "state_fips": "47",
        "already_collected": False, "h3_cell_count": 1, "pending": True,
    },
]


def load_top_county_cbgs() -> pd.DataFrame:
    population = pd.read_csv(
        POPULATION_FILE, sep="|", dtype={"GEO_ID": str},
        usecols=["GEO_ID", "B01003_E001"],
    )
    population = population[
        population.GEO_ID.str.startswith("1500000US")
    ].copy()
    population["cbg_str"] = population.GEO_ID.str.removeprefix("1500000US")
    population["population"] = pd.to_numeric(
        population.B01003_E001, errors="coerce"
    ).fillna(0.0)

    frames = []
    states = sorted({fips[:2] for fips in TOP_DENSITY_COUNTIES})
    for state in states:
        path = GEOMETRY_DIR / f"cb_2024_{state}_bg_500k.zip"
        frame = gpd.read_file(f"zip://{path}")[["GEOID", "ALAND", "geometry"]]
        frame["county_fips"] = frame.GEOID.str[:5]
        frames.append(frame[frame.county_fips.isin(TOP_DENSITY_COUNTIES)])

    cbgs = pd.concat(frames, ignore_index=True).rename(
        columns={"GEOID": "cbg_str", "ALAND": "land_area_m2"}
    )
    cbgs = cbgs.merge(
        population[["cbg_str", "population"]], on="cbg_str", how="left"
    )
    cbgs["population"] = cbgs.population.fillna(0.0)
    cbgs["land_area_km2"] = cbgs.land_area_m2 / 1_000_000.0
    cbgs["population_density"] = cbgs.population / cbgs.land_area_km2.where(
        cbgs.land_area_km2 > 0
    )
    cbgs["state_fips"] = cbgs.cbg_str.str[:2]
    return cbgs


def select_county_gaps(
    cbgs: pd.DataFrame, covered: set[str], already_queued: set[str],
    call_budget: int,
) -> tuple[pd.DataFrame, dict]:
    cbgs = cbgs.copy()
    cbgs["covered"] = cbgs.cbg_str.isin(covered)
    cbgs["queued"] = cbgs.cbg_str.isin(already_queued)
    gaps = cbgs[~cbgs.covered & ~cbgs.queued].copy()
    gaps["h3_cell_count"] = gaps.geometry.map(
        lambda geometry: h3_cell_count(geometry, 9)
    )
    gaps = gaps.sort_values(
        ["county_fips", "population_density", "population"],
        ascending=[True, False, False],
    )

    selected_indices: list[int] = []
    selected_set: set[int] = set()
    coverage_counts: dict[str, int] = {}
    totals = cbgs.groupby("county_fips").size().to_dict()
    for county, frame in cbgs.groupby("county_fips"):
        coverage_counts[county] = int((frame.covered | frame.queued).sum())

    calls_used = 0
    # First establish a meaningful minimum in every undercovered county.
    for county in sorted(TOP_DENSITY_COUNTIES):
        target = math.ceil(MIN_COUNTY_COVERAGE * totals[county])
        needed = max(0, target - coverage_counts[county])
        candidates = gaps[gaps.county_fips == county].head(needed)
        candidate_calls = int(candidates.h3_cell_count.sum())
        if calls_used + candidate_calls > call_budget:
            raise RuntimeError(
                "The 25% county coverage floor exceeds the configured budget"
            )
        selected_indices.extend(candidates.index.tolist())
        selected_set.update(candidates.index.tolist())
        calls_used += candidate_calls
        coverage_counts[county] += len(candidates)

    # Spend remaining capacity on the currently lowest-coverage county, taking
    # its densest remaining CBG. Stop at 50% so already strong counties do not
    # absorb budget intended to repair geographic imbalance.
    while calls_used < call_budget:
        county_order = sorted(
            TOP_DENSITY_COUNTIES,
            key=lambda county: (
                coverage_counts[county] / totals[county], county
            ),
        )
        added = False
        for county in county_order:
            if coverage_counts[county] / totals[county] >= BALANCED_FILL_CEILING:
                continue
            candidates = gaps[
                (gaps.county_fips == county)
                & (~gaps.index.isin(selected_set))
                & (gaps.h3_cell_count <= call_budget - calls_used)
            ]
            if candidates.empty:
                continue
            index = int(candidates.index[0])
            selected_indices.append(index)
            selected_set.add(index)
            calls_used += int(gaps.at[index, "h3_cell_count"])
            coverage_counts[county] += 1
            added = True
            break
        if not added:
            break

    selected = gaps.loc[selected_indices].copy()
    selected["selection_reason"] = "top20_county_gap"
    selected["already_collected"] = False
    selected["pending"] = True
    selected = selected.drop(columns=["geometry", "covered", "queued"])
    audit = {
        county: {
            "county": TOP_DENSITY_COUNTIES[county],
            "total_cbgs": int(totals[county]),
            "covered_or_previously_queued": int(
                cbgs.loc[cbgs.county_fips == county, ["covered", "queued"]]
                .any(axis=1).sum()
            ),
            "county_gap_cbgs_added": int(
                (selected.county_fips == county).sum()
            ),
            "projected_coverage": round(
                coverage_counts[county] / totals[county], 6
            ),
        }
        for county in TOP_DENSITY_COUNTIES
    }
    return selected, {
        "calls_used": calls_used,
        "remaining_unallocated_calls": call_budget - calls_used,
        "county_audit": audit,
    }


def main() -> None:
    source = pd.read_csv(SOURCE, dtype={"cbg_str": str, "state_fips": str})
    source["cbg_str"] = source.cbg_str.str.zfill(12)
    done = {path.stem for path in DONE_DIR.glob("*.done")}
    index = pd.read_csv(INDEX_FILE, dtype={"cbg": str}, usecols=["cbg"])
    index_cbgs = set(index.cbg.str.zfill(12))
    covered = done | index_cbgs

    high_density = source[
        (source.selection_reason == "high_density")
        & (~source.cbg_str.isin(done))
    ].copy().sort_values("queue_position")
    if len(high_density) != 293:
        raise RuntimeError(
            f"Expected 293 pending high-density CBGs; found {len(high_density)}"
        )

    tourism = pd.DataFrame(TOURISM_ROWS)
    base = pd.concat(
        [high_density.drop(columns=["queue_position"]), tourism],
        ignore_index=True,
    )
    base_calls = int(base.h3_cell_count.sum())
    county_gaps, gap_audit = select_county_gaps(
        load_top_county_cbgs(), covered, set(base.cbg_str),
        MAX_PLANNED_CALLS - base_calls,
    )
    county_gaps = county_gaps[base.columns]
    ordered = pd.concat([base, county_gaps], ignore_index=True)
    if ordered.cbg_str.duplicated().any():
        duplicates = ordered.loc[ordered.cbg_str.duplicated(), "cbg_str"].tolist()
        raise RuntimeError(f"Duplicate CBGs in next queue: {duplicates}")
    ordered.insert(0, "queue_position", range(1, len(ordered) + 1))
    ordered.to_csv(OUTPUT, index=False)

    total_calls = int(ordered.h3_cell_count.sum())
    summary = {
        "queue": str(OUTPUT),
        "total_cbgs": int(len(ordered)),
        "pending_high_density_cbgs": int(len(high_density)),
        "downtown_tourism_cbgs": int(len(tourism)),
        "top20_county_gap_cbgs": int(len(county_gaps)),
        "planned_api_calls_upper_bound": total_calls,
        "planned_cost_upper_bound_usd": round(total_calls * UNIT_COST_USD, 2),
        "credit_balance_before_run_usd": 86.0,
        "credit_buffer_usd": round(86.0 - total_calls * UNIT_COST_USD, 2),
        "county_gap_policy": (
            "Bring every top-20-density county to at least 25% projected CBG "
            "coverage, then balance the remainder toward 50% by repeatedly "
            "selecting the densest CBG in the lowest-coverage county."
        ),
        "unallocated_call_capacity": gap_audit["remaining_unallocated_calls"],
        "ordering": [
            "293 pending high-density CBGs in existing queue order",
            "Portland, ME downtown",
            "Bar Harbor, ME downtown",
            "Gatlinburg, TN downtown",
            "Pigeon Forge, TN downtown",
            "budget-balanced top-20 county coverage gaps",
        ],
        "county_audit": gap_audit["county_audit"],
    }
    OUTPUT.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "county_audit"}, indent=2))


if __name__ == "__main__":
    main()
