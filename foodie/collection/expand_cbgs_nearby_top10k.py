#!/usr/bin/env python3
"""
Find all uncollected CBGs within 50 km of any already-collected CBG centroid,
then keep only those that rank in the top 10,000 by visitor count overall.

This complements the existing expansion_cbgs_50km.csv (ranks 2001–4000) —
already-collected CBGs (including those from the first expansion run) are
automatically excluded via .done sentinel files.

Output: data/expansion_cbgs_nearby_top10k.csv  (column: cbg_str, 12-digit zero-padded)
        sorted by visitor count descending
"""
from pathlib import Path

import pandas as pd
import geopandas as gpd
import pygris

ROOT     = Path(__file__).resolve().parents[2]
CBG_PAT  = ROOT / "data" / "cbg_patterns.csv"
HEX_DIR  = ROOT / "data" / "hex_restaurants"
DONE_DIR = HEX_DIR / ".done"
OUTPUT   = ROOT / "data" / "expansion_cbgs_nearby_top10k.csv"

BUFFER_M  = 25_000   # 25 km in metres (EPSG:5070)
TOP_N     = 2_000    # keep only CBGs ranked within top 2k by visitor count
GEO_YEAR  = 2019

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
totals["rank"] = totals.index + 1
print(f"  Total unique CBGs in patterns: {len(totals):,}")

# ── Step 2: exclude already-collected CBGs (.done files) ──────────────────────
collected = set(f.stem for f in DONE_DIR.glob("*.done"))
print(f"  Already-collected CBGs (.done): {len(collected):,}")

uncollected = totals[~totals["cbg_str"].isin(collected)].copy()
print(f"  Uncollected CBGs: {len(uncollected):,}")

# ── Step 3: keep only top-10k uncollected by visitor count ────────────────────
top10k = uncollected[uncollected["rank"] <= TOP_N].copy()
print(f"  Top-{TOP_N:,} uncollected candidates: {len(top10k):,}")

# ── Step 4: download CBG geometries for relevant states ───────────────────────
def _state_set(cbg_series: pd.Series) -> set:
    return set(cbg_series.str[:2].unique())

needed_states = _state_set(pd.Series(list(collected))) | _state_set(top10k["cbg_str"])
print(f"\nDownloading cartographic block-group boundaries for {len(needed_states)} states …")

frames = []
for st in sorted(needed_states):
    try:
        gdf = pygris.block_groups(state=st, year=GEO_YEAR, cache=True, cb=True)
        frames.append(gdf[["GEOID", "geometry"]])
    except Exception as exc:
        print(f"  WARNING: state {st} failed — {exc}")

all_bg = gpd.GeoDataFrame(
    pd.concat(frames, ignore_index=True).rename(columns={"GEOID": "cbg_str"}),
    geometry="geometry",
).to_crs("EPSG:5070")
print(f"  Loaded {len(all_bg):,} block groups.")

# ── Step 5: 50 km buffer around all collected CBG centroids ───────────────────
collected_bg = all_bg[all_bg["cbg_str"].isin(collected)]
print(f"  Collected CBGs matched to geometry: {len(collected_bg):,}")

buffer_union = collected_bg.geometry.centroid.buffer(BUFFER_M).union_all()

# ── Step 6: filter top-10k candidates to those inside the buffer ──────────────
cand_bg = all_bg[all_bg["cbg_str"].isin(top10k["cbg_str"])].copy()
cand_bg["within_50km"] = cand_bg.geometry.centroid.within(buffer_union)

passing_cbgs = set(cand_bg.loc[cand_bg["within_50km"], "cbg_str"])
print(
    f"\n  {len(passing_cbgs)} of {len(top10k)} top-{TOP_N:,} candidates "
    f"are within {BUFFER_M // 1000} km of collected CBGs."
)

# ── Step 7: save output ───────────────────────────────────────────────────────
result = (
    top10k[top10k["cbg_str"].isin(passing_cbgs)]
    .sort_values("raw_visitor_count", ascending=False)
    [["cbg_str"]]
    .reset_index(drop=True)
)
result.to_csv(OUTPUT, index=False)
print(f"Saved {len(result):,} CBGs → {OUTPUT}")
