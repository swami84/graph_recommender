#!/usr/bin/env python3
"""
Select the next 2,000 CBGs by visitor count (ranks 2001–4000 among uncollected),
then filter to those within 50 km of any already-collected CBG centroid.

Output: data/expansion_cbgs_50km.csv  (column: cbg_str, 12-digit zero-padded)
        sorted by visitor count descending
"""
from pathlib import Path

import pandas as pd
import geopandas as gpd
import pygris

ROOT       = Path(__file__).parent
CBG_PAT    = ROOT / "data" / "cbg_patterns.csv"
HEX_DIR    = ROOT / "data" / "hex_restaurants"
OUTPUT     = ROOT / "data" / "expansion_cbgs_50km.csv"

BUFFER_M   = 50_000   # 50 km in metres (EPSG:5070)
RANK_START = 2001
RANK_END   = 4000
GEO_YEAR   = 2019

# ── Step 1: rank all CBGs by total visitor count ───────────────────────────────
print("Loading cbg_patterns.csv …")
df = pd.read_csv(CBG_PAT, usecols=["census_block_group", "raw_visitor_count"])
df["cbg_str"] = df["census_block_group"].astype("Int64").astype(str).str.zfill(12)
totals = (
    df.groupby("cbg_str", sort=False)["raw_visitor_count"]
    .sum()
    .reset_index()
    .sort_values("raw_visitor_count", ascending=False)
    .reset_index(drop=True)
)
print(f"  Total unique CBGs in patterns: {len(totals):,}")

# ── Step 2: exclude already-collected CBGs ─────────────────────────────────────
collected = set(f.name.split("_")[0] for f in HEX_DIR.glob("*.json"))
print(f"  Already-collected CBGs: {len(collected):,}")

uncollected = totals[~totals["cbg_str"].isin(collected)].reset_index(drop=True)
uncollected["rank"] = uncollected.index + 1
print(f"  Uncollected CBGs: {len(uncollected):,}")

# ── Step 3: take ranks 2001–4000 ──────────────────────────────────────────────
candidates = uncollected[
    (uncollected["rank"] >= RANK_START) & (uncollected["rank"] <= RANK_END)
].copy()
print(f"  Candidate CBGs (ranks {RANK_START}–{RANK_END}): {len(candidates):,}")

# ── Step 4: download CBG geometries for relevant states ───────────────────────
def _state_set(cbg_series: pd.Series) -> set:
    return set(cbg_series.str[:2].unique())

needed_states = _state_set(pd.Series(list(collected))) | _state_set(candidates["cbg_str"])
print(f"\nDownloading cartographic block-group boundaries for {len(needed_states)} states …")

frames = []
for st in sorted(needed_states):
    try:
        gdf = pygris.block_groups(state=st, year=GEO_YEAR, cache=True, cb=True)
        frames.append(gdf[["GEOID", "geometry"]])
    except Exception as exc:
        print(f"  WARNING: state {st} failed — {exc}")

all_bg = pd.concat(frames, ignore_index=True)
all_bg = all_bg.rename(columns={"GEOID": "cbg_str"})

# Project to US Albers Equal Area (metres) for accurate distance calculations
all_bg = gpd.GeoDataFrame(all_bg, geometry="geometry").to_crs("EPSG:5070")
print(f"  Loaded {len(all_bg):,} block groups across {len(needed_states)} states.")

# ── Step 5: 50 km buffer around collected CBG centroids ───────────────────────
collected_bg = all_bg[all_bg["cbg_str"].isin(collected)].copy()
print(f"  Collected CBGs matched to geometry: {len(collected_bg):,}")

buffer_union = collected_bg.geometry.centroid.buffer(BUFFER_M).union_all()

# ── Step 6: filter candidates to those inside the buffer ──────────────────────
cand_bg = all_bg[all_bg["cbg_str"].isin(candidates["cbg_str"])].copy()
cand_bg["within_50km"] = cand_bg.geometry.centroid.within(buffer_union)

passing_cbgs = set(cand_bg.loc[cand_bg["within_50km"], "cbg_str"])
print(
    f"\n  {len(passing_cbgs)} of {len(candidates)} candidate CBGs "
    f"pass the 50 km proximity filter."
)

# ── Step 7: save output ───────────────────────────────────────────────────────
result = (
    candidates[candidates["cbg_str"].isin(passing_cbgs)]
    .sort_values("raw_visitor_count", ascending=False)
    [["cbg_str"]]
    .reset_index(drop=True)
)
result.to_csv(OUTPUT, index=False)
print(f"Saved {len(result):,} CBGs → {OUTPUT}")
