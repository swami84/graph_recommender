#!/usr/bin/env python3
"""
hexagon_places.py — Tile each CBG with H3 hexagons and query Google Places API.

H3 resolution 9 gives hexagons with ~174m edge length (~350m diameter).
This is the closest H3 resolution to the requested ~500m diameter.
The Places API search radius is set to 250m to cover each hexagon.

Note: New Places API caps at 20 results per request (user requested 50;
      20 is the hard API limit for searchNearby).

Usage:
    python hexagon_places.py                    # process all CBGs
    python hexagon_places.py --cbg 360610031001 # single CBG
    python hexagon_places.py --limit 5          # first 5 CBGs only

Output:
    data/hex_restaurants/{cbg}_{h3_index}.json  — one file per hexagon
    data/hex_restaurants_index.csv              — flat list of all restaurants found
    data/hex_api_call_log.csv                   — API call cost log
"""

import argparse
import csv
import datetime
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd
import pygris
import requests
from dotenv import load_dotenv
from shapely.geometry import mapping

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
PLACES_NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
FIELD_MASK = (
    "places.id,places.displayName,places.formattedAddress,"
    "places.location,places.types,places.businessStatus,"
    "places.rating,places.userRatingCount,places.priceLevel,"
    "places.nationalPhoneNumber,places.websiteUri"
)

H3_RESOLUTION   = 9    # edge ~174m, diameter ~350m — closest H3 res to 500m
SEARCH_RADIUS_M = 250  # covers hexagon from centroid to edge with small buffer
MAX_RESULTS     = 20   # hard cap from New Places API (searchNearby max)

# ── Paths ──────────────────────────────────────────────────────────────────────
CBG_LIST_FILE   = Path("data/top100_cbgs.csv")
HEX_DIR         = Path("data/hex_restaurants")
DONE_DIR        = Path("data/hex_restaurants/.done")   # sentinel files per completed CBG
INDEX_FILE      = Path("data/hex_restaurants_index.csv")
API_LOG_FILE    = Path("data/hex_api_call_log.csv")

HEX_DIR.mkdir(parents=True, exist_ok=True)
DONE_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hex_places")

# ── CSV column definitions ─────────────────────────────────────────────────────
INDEX_COLS = [
    "place_id", "name", "cbg", "h3_index",
    "lat", "lng", "rating", "user_rating_count",
    "price_level", "business_status", "types",
    "address", "phone", "website",
]
LOG_COLS = [
    "timestamp", "cbg", "h3_index", "lat", "lng",
    "radius", "response_status", "num_results", "estimated_cost_usd",
]


def _init_csv(path: Path, cols: list):
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()


def _log_api_call(cbg, h3_idx, lat, lng, status, n):
    with open(API_LOG_FILE, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_COLS).writerow({
            "timestamp":        datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "cbg":              cbg,
            "h3_index":         h3_idx,
            "lat":              lat,
            "lng":              lng,
            "radius":           SEARCH_RADIUS_M,
            "response_status":  status,
            "num_results":      n,
            "estimated_cost_usd": 0.032,
        })


def _append_index(places: list, cbg: str, h3_idx: str):
    if not places:
        return
    rows = []
    for p in places:
        loc = p.get("location", {})
        rows.append({
            "place_id":          p.get("id", ""),
            "name":              p.get("displayName", {}).get("text", ""),
            "cbg":               cbg,
            "h3_index":          h3_idx,
            "lat":               loc.get("latitude"),
            "lng":               loc.get("longitude"),
            "rating":            p.get("rating"),
            "user_rating_count": p.get("userRatingCount"),
            "price_level":       p.get("priceLevel"),
            "business_status":   p.get("businessStatus"),
            "types":             ",".join(p.get("types", [])),
            "address":           p.get("formattedAddress", ""),
            "phone":             p.get("nationalPhoneNumber", ""),
            "website":           p.get("websiteUri", ""),
        })
    with open(INDEX_FILE, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=INDEX_COLS).writerows(rows)


# ── Places API ─────────────────────────────────────────────────────────────────
def fetch_restaurants(cbg: str, h3_idx: str, lat: float, lng: float) -> list:
    headers = {
        "Content-Type":     "application/json",
        "X-Goog-Api-Key":   GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": FIELD_MASK,
    }
    body = {
        "includedTypes": ["restaurant"],
        "maxResultCount": MAX_RESULTS,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": float(SEARCH_RADIUS_M),
            }
        },
    }
    try:
        resp = requests.post(PLACES_NEARBY_URL, headers=headers, json=body, timeout=15)
        data = resp.json()
        if "error" in data:
            log.warning(f"  API error [{h3_idx}]: {data['error'].get('message')}")
            _log_api_call(cbg, h3_idx, lat, lng, "ERROR", 0)
            return []
        places = data.get("places", [])
        _log_api_call(cbg, h3_idx, lat, lng, "OK", len(places))
        return places
    except Exception as e:
        log.error(f"  Request failed [{h3_idx}]: {e}")
        _log_api_call(cbg, h3_idx, lat, lng, "ERROR", 0)
        return []


# ── Geometry helpers ───────────────────────────────────────────────────────────
GEOMETRY_YEARS = [2010, 2019, 2018]  # SafeGraph uses 2010 CBGs; fallback to later vintages for mid-decade additions


def load_geometries(cbg_list: list) -> dict:
    """Batch-fetch CBG geometries from Census TIGER, one pygris call per state per year.

    Tries GEOMETRY_YEARS in order. CBGs not found in 2010 may exist in later
    vintages due to mid-decade Census updates.
    """
    by_state = defaultdict(list)
    for cbg in cbg_list:
        by_state[cbg[:2]].append(cbg)

    geom_map = {}
    for state_fips, cbgs in sorted(by_state.items()):
        remaining = set(cbgs)
        for year in GEOMETRY_YEARS:
            if not remaining:
                break
            log.info(f"Fetching geometries for state {state_fips} year={year} ({len(remaining)} CBGs)…")
            try:
                gdf = pygris.block_groups(state=state_fips, year=year, cache=True)
                geoid_col = "GEOID10" if "GEOID10" in gdf.columns else "GEOID"
                found = set()
                for cbg in remaining:
                    row = gdf[gdf[geoid_col] == cbg]
                    if not row.empty:
                        geom_map[cbg] = row.geometry.iloc[0]
                        found.add(cbg)
                remaining -= found
                if found:
                    log.info(f"  Found {len(found)} CBGs in {year} vintage")
            except Exception as e:
                log.error(f"  Failed fetching state {state_fips} year={year}: {e}")

        for cbg in remaining:
            log.warning(f"  GEOID {cbg} not found in any vintage")

    return geom_map


def cbg_to_hexagons(geom) -> list:
    """Return H3 cell IDs that cover the CBG polygon (WGS84 input)."""
    geo_dict = mapping(geom)  # Shapely → GeoJSON dict (lng, lat coordinate order)
    return list(h3.geo_to_cells(geo_dict, H3_RESOLUTION))


# ── Per-CBG processing ─────────────────────────────────────────────────────────
def process_cbg(cbg_str: str, geom):
    hexes = cbg_to_hexagons(geom)
    pending = [h for h in hexes if not (HEX_DIR / f"{cbg_str}_{h}.json").exists()]
    log.info(f"CBG {cbg_str}: {len(hexes)} hexagons, {len(pending)} pending")

    consecutive_empty = 0
    for h3_idx in pending:
        lat, lng = h3.cell_to_latlng(h3_idx)
        places = fetch_restaurants(cbg_str, h3_idx, lat, lng)

        payload = {
            "cbg":             cbg_str,
            "h3_index":        h3_idx,
            "h3_resolution":   H3_RESOLUTION,
            "centroid_lat":    lat,
            "centroid_lng":    lng,
            "search_radius_m": SEARCH_RADIUS_M,
            "scraped_at":      datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_results":   len(places),
            "places":          places,
        }
        (HEX_DIR / f"{cbg_str}_{h3_idx}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
        _append_index(places, cbg_str, h3_idx)
        time.sleep(0.2)

        if len(places) == 0:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                log.info(f"CBG {cbg_str}: first 2 hexagons empty — skipping remaining {len(pending) - consecutive_empty} hexagons")
                break
        else:
            consecutive_empty = 0

    # Mark CBG as fully complete
    (DONE_DIR / f"{cbg_str}.done").touch()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Tile CBGs with H3 hexagons, query Places API")
    parser.add_argument("--cbg",      help="Process a single CBG FIPS")
    parser.add_argument("--cbg-file", default=str(CBG_LIST_FILE),
                        help=f"CSV with cbg_str column (default: {CBG_LIST_FILE})")
    parser.add_argument("--limit",    type=int, help="Cap number of CBGs to process")
    args = parser.parse_args()

    _init_csv(INDEX_FILE, INDEX_COLS)
    _init_csv(API_LOG_FILE, LOG_COLS)

    if args.cbg:
        cbg_list = [args.cbg]
    else:
        cbg_file = Path(args.cbg_file)
        if not cbg_file.exists():
            log.error(f"CBG list not found: {cbg_file}. Export top100['cbg_str'].to_csv('data/top100_cbgs.csv', index=False) from the notebook first.")
            return
        cbg_list = pd.read_csv(cbg_file, dtype=str)["cbg_str"].str.zfill(12).tolist()
        if args.limit:
            cbg_list = cbg_list[:args.limit]

    # Filter out already-completed CBGs before fetching any geometries
    pending_cbgs = [c for c in cbg_list if not (DONE_DIR / f"{c}.done").exists()]
    skipped = len(cbg_list) - len(pending_cbgs)
    log.info(f"{len(cbg_list)} CBGs total — {skipped} already done, {len(pending_cbgs)} to process")

    if not pending_cbgs:
        log.info("Nothing to do.")
        return

    geom_map = load_geometries(pending_cbgs)

    for cbg_str in pending_cbgs:
        geom = geom_map.get(cbg_str)
        if geom is None:
            log.warning(f"No geometry for {cbg_str} — skipping")
            continue
        process_cbg(cbg_str, geom)

    log.info("Done.")


if __name__ == "__main__":
    main()
