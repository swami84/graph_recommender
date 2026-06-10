#!/usr/bin/env python3
"""
Build an expansion CBG list using two complementary signals:

  1. Productive neighbors  — uncollected CBGs within 3 km of any CBG that
                             already yielded ≥1 restaurant (guaranteed density)
  2. Population density    — uncollected CBGs with >500 people/km² within
                             25 km of any collected CBG (catches dense
                             residential neighbourhoods we haven't visited)

Output: data/expansion_cbgs_combined.csv  (column: cbg_str, 12-digit zero-padded)
"""
import time
from pathlib import Path

import pandas as pd
import geopandas as gpd
import pygris
import requests

ROOT      = Path(__file__).parent
INDEX_CSV = ROOT / "data" / "hex_restaurants_index.csv"
DONE_DIR  = ROOT / "data" / "hex_restaurants" / ".done"
OUTPUT    = ROOT / "data" / "expansion_cbgs_combined.csv"

PROD_BUFFER_M  =  1_500   # 1.5 km — immediate neighbours of productive CBGs
DENS_BUFFER_M  = 10_000   # 10 km  — proximity check for density pass
POP_DENSITY_THRESHOLD = 3_000   # people per km² (dense urban only)
GEO_YEAR = 2019

# ── Step 1: already-collected and productive CBGs ─────────────────────────────
collected = set(f.stem for f in DONE_DIR.glob("*.done"))
print(f"Already-collected CBGs: {len(collected):,}")

index_df = pd.read_csv(INDEX_CSV, usecols=["cbg"])
index_df["cbg"] = index_df["cbg"].astype(str).str.zfill(12)
productive = set(index_df["cbg"].unique())
print(f"Productive CBGs (≥1 restaurant): {len(productive):,}")

# ── Step 2: download CBG geometries for all relevant states ───────────────────
def _state_set(cbg_iter) -> set:
    return {c[:2] for c in cbg_iter}

needed_states = _state_set(collected) | _state_set(productive)
print(f"\nDownloading geometries for {len(needed_states)} states …")

frames = []
for st in sorted(needed_states):
    try:
        gdf = pygris.block_groups(state=st, year=GEO_YEAR, cache=True, cb=True)
        frames.append(gdf[["GEOID", "ALAND", "geometry"]])
    except Exception as exc:
        print(f"  WARNING: state {st} — {exc}")

all_bg = gpd.GeoDataFrame(
    pd.concat(frames, ignore_index=True).rename(columns={"GEOID": "cbg_str"}),
    geometry="geometry",
).to_crs("EPSG:5070")
print(f"  Loaded {len(all_bg):,} block groups.")

uncollected_bg = all_bg[~all_bg["cbg_str"].isin(collected)].copy()
print(f"  Uncollected block groups: {len(uncollected_bg):,}")

# ── Step 3: signal A — productive-CBG neighbours (3 km buffer) ───────────────
prod_bg = all_bg[all_bg["cbg_str"].isin(productive)]
prod_buffer = prod_bg.geometry.centroid.buffer(PROD_BUFFER_M).union_all()

signal_a = set(
    uncollected_bg.loc[
        uncollected_bg.geometry.centroid.within(prod_buffer), "cbg_str"
    ]
)
print(f"\nSignal A (productive neighbours, {PROD_BUFFER_M//1000} km): {len(signal_a):,} CBGs")

# ── Step 4: fetch ACS population for needed states ────────────────────────────
print(f"\nFetching ACS 2019 population for {len(needed_states)} states …")

pop_rows = []
for st in sorted(needed_states):
    try:
        r = requests.get(
            "https://api.census.gov/data/2019/acs/acs5",
            params={"get": "B01003_001E", "for": "block group:*", "in": f"state:{st} county:*"},
            timeout=20,
        )
        data = r.json()
        header, *rows = data
        for row in rows:
            pop, state, county, tract, blkgrp = row
            cbg_str = f"{state.zfill(2)}{county.zfill(3)}{tract.zfill(6)}{blkgrp.zfill(1)}"
            pop_rows.append({"cbg_str": cbg_str, "population": int(pop) if pop else 0})
        time.sleep(0.1)
    except Exception as exc:
        print(f"  WARNING: population fetch failed for state {st} — {exc}")

pop_df = pd.DataFrame(pop_rows)
print(f"  Fetched population for {len(pop_df):,} block groups.")

# ── Step 5: signal B — high-density CBGs within 25 km ────────────────────────
# Compute density: population / land area in km²
all_bg = all_bg.merge(pop_df, on="cbg_str", how="left")
all_bg["area_km2"] = all_bg["ALAND"] / 1_000_000   # ALAND is in m²
all_bg["pop_density"] = all_bg["population"] / all_bg["area_km2"].replace(0, float("nan"))

collected_bg = all_bg[all_bg["cbg_str"].isin(collected)]
dens_buffer = collected_bg.geometry.centroid.buffer(DENS_BUFFER_M).union_all()

dense_uncollected = uncollected_bg.merge(
    all_bg[["cbg_str", "pop_density"]], on="cbg_str", how="left"
)
signal_b = set(
    dense_uncollected.loc[
        dense_uncollected.geometry.centroid.within(dens_buffer)
        & (dense_uncollected["pop_density"] > POP_DENSITY_THRESHOLD),
        "cbg_str",
    ]
)
print(f"Signal B (dense + within {DENS_BUFFER_M//1000} km, >{POP_DENSITY_THRESHOLD} ppl/km²): {len(signal_b):,} CBGs")

# ── Step 6: stratified by state, ranked by density within each, cap at TOP_K ──
TOP_K = 2_000

combined = (signal_a & signal_b) | (signal_a - signal_b)
print(f"\nCombined (A∩B + A-only): {len(combined):,} CBGs  "
      f"(A∩B: {len(signal_a & signal_b):,}, A-only: {len(signal_a - signal_b):,})")

density_lookup = all_bg.set_index("cbg_str")["pop_density"]

# Count productive CBGs per state to weight the per-state allocation
prod_state_counts = pd.Series(list(productive)).str[:2].value_counts()
total_prod = prod_state_counts.sum()

candidates = (
    pd.DataFrame({"cbg_str": sorted(combined)})
    .assign(
        pop_density=lambda d: d["cbg_str"].map(density_lookup),
        state=lambda d: d["cbg_str"].str[:2],
    )
)

# Allocate TOP_K slots proportionally to each state's share of productive CBGs,
# minimum 1 slot per state, then take the densest CBGs within each state's quota
state_quota = (
    (prod_state_counts / total_prod * TOP_K)
    .clip(lower=1)
    .round()
    .astype(int)
)
# Trim to exactly TOP_K after rounding
excess = state_quota.sum() - TOP_K
if excess > 0:
    state_quota = state_quota.sort_values(ascending=True)
    for st in state_quota.index:
        if excess <= 0:
            break
        if state_quota[st] > 1:
            state_quota[st] -= 1
            excess -= 1

pieces = []
for state, quota in state_quota.items():
    state_cbgs = candidates[candidates["state"] == state]
    pieces.append(
        state_cbgs.sort_values("pop_density", ascending=False).head(quota)
    )

result = (
    pd.concat(pieces)
    .sort_values("pop_density", ascending=False)
    [["cbg_str"]]
    .reset_index(drop=True)
)

kept_density = density_lookup[result["cbg_str"]]
states_covered = result["cbg_str"].str[:2].nunique()
print(f"Saved {len(result):,} CBGs across {states_covered} states")
print(f"  Density range: {kept_density.max():,.0f} → {kept_density.min():,.0f} ppl/km²")
print(f"  Top states by allocation:")
print(result["cbg_str"].str[:2].value_counts().head(10).to_string())
result.to_csv(OUTPUT, index=False)
print(f"\nSaved → {OUTPUT}")
