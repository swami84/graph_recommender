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
import fcntl
import json
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
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
    # Keep discovery on Nearby Search Pro. Rating, rating count, price, phone,
    # and website are Enterprise fields and made every tile materially dearer.
    # Those attributes can be enriched after discovery or obtained while
    # scraping the public Maps page.
    "places.primaryType"
)

H3_RESOLUTION   = 9    # edge ~174m, diameter ~350m — closest H3 res to 500m
SEARCH_RADIUS_M = 250  # covers hexagon from centroid to edge with small buffer
MAX_RESULTS     = 20   # hard cap from New Places API (searchNearby max)
PLACES_NEARBY_PRO_UNIT_COST_USD = 0.032

# ── Paths ──────────────────────────────────────────────────────────────────────
CBG_LIST_FILE   = Path("data/top100_cbgs.csv")
SEARCH_SEEDS_FILE = Path("data/cbg_search_seeds.csv")
HEX_DIR         = Path("data/hex_restaurants")
DONE_DIR        = Path("data/hex_restaurants/.done")   # sentinel files per completed CBG
PRIORITY_DONE_DIR = Path("data/hex_restaurants/.priority_done")
STATUS_DIR      = Path("data/hex_restaurants/.status") # completion reason/scan counts
INDEX_FILE      = Path("data/hex_restaurants_index.csv")
API_LOG_FILE    = Path("data/hex_api_call_log.csv")
BUDGET_DIR      = Path("data/places_budget")
LOCAL_GEOMETRY_DIR = Path("data/census_geometry_2024")

HEX_DIR.mkdir(parents=True, exist_ok=True)
DONE_DIR.mkdir(parents=True, exist_ok=True)
PRIORITY_DONE_DIR.mkdir(parents=True, exist_ok=True)
STATUS_DIR.mkdir(parents=True, exist_ok=True)
BUDGET_DIR.mkdir(parents=True, exist_ok=True)

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
            "estimated_cost_usd": 0.032,  # Pro first paid tier; free cap handled monthly
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
@dataclass(frozen=True)
class FetchResult:
    places: list
    ok: bool


class PlacesBudgetExceeded(RuntimeError):
    """Raised before an HTTP request that would exceed either hard guard."""


@dataclass
class PlacesBudgetGuard:
    """Persistent, fail-closed request and spend guard.

    Reservations are recorded before HTTP transmission and are never refunded.
    That deliberately over-counts failed requests and protects the budget after
    process or power failures.
    """

    budget_id: str
    billing_month: str
    max_monthly_calls: int
    max_run_cost_usd: float
    unit_cost_usd: float = PLACES_NEARBY_PRO_UNIT_COST_USD
    api_log_file: Path = API_LOG_FILE
    budget_dir: Path = BUDGET_DIR

    def __post_init__(self):
        if not re_fullmatch_budget_id(self.budget_id):
            raise ValueError("budget_id may contain only letters, numbers, dots, dashes, and underscores")
        if not re_fullmatch_month(self.billing_month):
            raise ValueError("billing_month must use YYYY-MM")
        if self.max_monthly_calls < 1 or self.max_run_cost_usd <= 0:
            raise ValueError("budget limits and unit cost must be positive")
        if self.unit_cost_usd < PLACES_NEARBY_PRO_UNIT_COST_USD:
            raise ValueError(
                f"unit cost cannot be below the published Pro rate "
                f"${PLACES_NEARBY_PRO_UNIT_COST_USD:.3f} per request"
            )
        self.budget_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.budget_dir / f"{self.budget_id}.json"
        self.lock_path = self.budget_dir / f"{self.budget_id}.lock"

    def _logged_month_calls(self) -> int:
        if not self.api_log_file.exists():
            return 0
        count = 0
        try:
            with self.api_log_file.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if str(row.get("timestamp", "")).startswith(self.billing_month):
                        count += 1
        except (OSError, csv.Error) as error:
            raise PlacesBudgetExceeded(
                f"Cannot safely read API ledger {self.api_log_file}: {error}"
            ) from error
        return count

    def _initial_state(self) -> dict:
        return {
            "budget_id": self.budget_id,
            "billing_month": self.billing_month,
            "baseline_month_calls": self._logged_month_calls(),
            "reserved_calls": 0,
            "max_monthly_calls": self.max_monthly_calls,
            "max_run_cost_usd": self.max_run_cost_usd,
            "unit_cost_usd": self.unit_cost_usd,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return self._initial_state()
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise PlacesBudgetExceeded(
                f"Cannot safely read budget state {self.state_path}: {error}"
            ) from error
        expected = {
            "billing_month": self.billing_month,
            "max_monthly_calls": self.max_monthly_calls,
            "max_run_cost_usd": self.max_run_cost_usd,
            "unit_cost_usd": self.unit_cost_usd,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise PlacesBudgetExceeded(
                    f"Budget state {self.budget_id} has {key}={state.get(key)!r}, "
                    f"not requested {value!r}; use the original limits or a new budget ID"
                )
        return state

    def _write_state(self, state: dict) -> None:
        state["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.state_path)

    def _snapshot_locked(self, state: dict) -> dict:
        logged = self._logged_month_calls()
        baseline = int(state["baseline_month_calls"])
        reserved = int(state["reserved_calls"])
        accounted_month_calls = max(logged, baseline + reserved)
        run_cost = reserved * self.unit_cost_usd
        return {
            "logged_month_calls": logged,
            "accounted_month_calls": accounted_month_calls,
            "reserved_calls": reserved,
            "estimated_run_cost_usd": run_cost,
            "remaining_month_calls": max(0, self.max_monthly_calls - accounted_month_calls),
            "remaining_run_calls": max(
                0, math.floor((self.max_run_cost_usd + 1e-9) / self.unit_cost_usd) - reserved
            ),
        }

    def status(self) -> dict:
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self._load_state()
            self._write_state(state)
            return {**state, **self._snapshot_locked(state)}

    def reserve(self, cbg: str, h3_idx: str) -> dict:
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self._load_state()
            snapshot = self._snapshot_locked(state)
            if snapshot["accounted_month_calls"] + 1 > self.max_monthly_calls:
                raise PlacesBudgetExceeded(
                    f"monthly request ceiling reached ({self.max_monthly_calls:,} calls for "
                    f"{self.billing_month}); blocked before {cbg}/{h3_idx}"
                )
            projected_cost = (snapshot["reserved_calls"] + 1) * self.unit_cost_usd
            if projected_cost > self.max_run_cost_usd + 1e-9:
                raise PlacesBudgetExceeded(
                    f"run spend ceiling reached (${self.max_run_cost_usd:,.2f}); "
                    f"blocked before {cbg}/{h3_idx}"
                )
            state["reserved_calls"] = snapshot["reserved_calls"] + 1
            state["last_reservation"] = {"cbg": cbg, "h3_index": h3_idx}
            self._write_state(state)
            return self._snapshot_locked(state)


def re_fullmatch_budget_id(value: str) -> bool:
    return bool(value) and all(character.isalnum() or character in "._-" for character in value)


def re_fullmatch_month(value: str) -> bool:
    try:
        datetime.datetime.strptime(value, "%Y-%m")
        return len(value) == 7
    except ValueError:
        return False


def fetch_restaurants(
    cbg: str, h3_idx: str, lat: float, lng: float,
    budget_guard: PlacesBudgetGuard,
) -> FetchResult:
    """Query one cell, keeping a valid empty response distinct from an error."""
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
    # Reserve before network I/O. A failed request remains charged against the
    # local budget because Google can bill requests that do not yield results.
    budget_guard.reserve(cbg, h3_idx)
    try:
        resp = requests.post(PLACES_NEARBY_URL, headers=headers, json=body, timeout=15)
        data = resp.json()
        if not resp.ok or "error" in data:
            message = data.get("error", {}).get("message", f"HTTP {resp.status_code}")
            log.warning(f"  API error [{h3_idx}]: {message}")
            _log_api_call(cbg, h3_idx, lat, lng, "ERROR", 0)
            return FetchResult([], False)
        places = data.get("places", [])
        _log_api_call(cbg, h3_idx, lat, lng, "OK", len(places))
        return FetchResult(places, True)
    except Exception as e:
        log.error(f"  Request failed [{h3_idx}]: {e}")
        _log_api_call(cbg, h3_idx, lat, lng, "ERROR", 0)
        return FetchResult([], False)


# ── Geometry helpers ───────────────────────────────────────────────────────────
GEOMETRY_YEARS = [2024, 2023, 2020, 2010, 2019, 2018]


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
        local_path = LOCAL_GEOMETRY_DIR / f"cb_2024_{state_fips}_bg_500k.zip"
        if local_path.exists():
            try:
                local = gpd.read_file(local_path)
                geoid_col = "GEOID" if "GEOID" in local.columns else "GEOID10"
                local[geoid_col] = local[geoid_col].astype(str).str.zfill(12)
                matches = local[local[geoid_col].isin(remaining)]
                for geoid, geom in zip(matches[geoid_col], matches.geometry):
                    geom_map[geoid] = geom
                remaining -= set(matches[geoid_col])
                log.info(
                    f"Loaded {len(matches)} of {len(cbgs)} state {state_fips} "
                    "CBGs from local 2024 geometry"
                )
            except Exception as e:
                log.error(f"Failed reading local geometry {local_path}: {e}")

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


def cbg_to_hexagons(
    geom, search_center: tuple[float, float] | None = None
) -> list:
    """Return deterministic H3 cells ordered from the requested search center."""
    geo_dict = mapping(geom)  # Shapely → GeoJSON dict (lng, lat coordinate order)
    cells = list(h3.geo_to_cells(geo_dict, H3_RESOLUTION))
    if search_center is None:
        center = geom.representative_point()
        center_lat, center_lng = center.y, center.x
    else:
        center_lat, center_lng = search_center
        # H3 polygon filling uses cell-center containment, so the cell that
        # contains an explicitly requested downtown point can be omitted when
        # its own center falls just outside the CBG boundary. Always include
        # the requested point's cell for priority point searches.
        priority_cell = h3.latlng_to_cell(
            center_lat, center_lng, H3_RESOLUTION
        )
        if priority_cell not in cells:
            cells.append(priority_cell)
    if not cells:
        # H3 center containment can return no cells for very small/sliver CBGs.
        cells = [h3.latlng_to_cell(center_lat, center_lng, H3_RESOLUTION)]
    return sorted(
        cells,
        key=lambda cell: (
            (h3.cell_to_latlng(cell)[0] - center_lat) ** 2
            + (h3.cell_to_latlng(cell)[1] - center_lng) ** 2,
            cell,
        ),
    )


# ── Per-CBG processing ─────────────────────────────────────────────────────────
def _write_cbg_status(cbg_str: str, **status):
    payload = {
        "cbg": cbg_str,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **status,
    }
    (STATUS_DIR / f"{cbg_str}.json").write_text(json.dumps(payload, indent=2) + "\n")


def build_h3_cache() -> dict[str, Path]:
    """Index successful cell payloads so identical searches are never repaid."""
    cache = {}
    for path in HEX_DIR.glob("*.json"):
        parts = path.stem.rsplit("_", 1)
        if len(parts) == 2:
            cache.setdefault(parts[1], path)
    return cache


def process_cbg(
    cbg_str: str, geom, budget_guard: PlacesBudgetGuard, empty_stop: int = 2,
    h3_cache: dict[str, Path] | None = None,
    search_center: tuple[float, float] | None = None,
    max_h3_cells: int | None = None,
    completion_marker_dir: Path = DONE_DIR,
) -> bool:
    """Process a CBG; only successful responses can create its done marker."""
    hexes = cbg_to_hexagons(geom, search_center=search_center)
    polygon_cell_count = len(hexes)
    if max_h3_cells is not None:
        if max_h3_cells < 1:
            raise ValueError("max_h3_cells must be positive")
        hexes = hexes[:max_h3_cells]
    if search_center is not None:
        log.info(
            "CBG %s: using priority search center %.6f, %.6f",
            cbg_str, search_center[0], search_center[1],
        )
    if max_h3_cells is not None:
        log.info(
            "CBG %s: priority scope limited to %s H3 cell(s)",
            cbg_str, len(hexes),
        )
    pending = [h for h in hexes if not (HEX_DIR / f"{cbg_str}_{h}.json").exists()]
    log.info(f"CBG {cbg_str}: {len(hexes)} hexagons, {len(pending)} pending")

    consecutive_empty = 0
    queried = 0
    reused = 0
    results = 0
    completion_reason = (
        "priority_cell_limit" if max_h3_cells is not None
        else "all_cells_queried"
    )
    for h3_idx in pending:
        lat, lng = h3.cell_to_latlng(h3_idx)
        reused_from = h3_cache.get(h3_idx) if h3_cache is not None else None
        if reused_from is not None:
            try:
                places = json.loads(reused_from.read_text()).get("places", [])
                reused += 1
            except Exception as e:
                log.warning(f"Could not reuse {reused_from}: {e}; querying API")
                reused_from = None

        if reused_from is None:
            fetch = fetch_restaurants(cbg_str, h3_idx, lat, lng, budget_guard)
        else:
            fetch = FetchResult(places, True)
        if not fetch.ok:
            # Do not write a normal cell file and do not mark the CBG done. A
            # corrected key/quota/network condition can safely resume this cell.
            _write_cbg_status(
                cbg_str,
                state="incomplete_error",
                total_cells=len(hexes),
                pending_at_start=len(pending),
                successful_calls=queried,
                reused_cells=reused,
                results=results,
                failed_h3_index=h3_idx,
            )
            log.error(f"CBG {cbg_str}: API failure; leaving CBG pending")
            return False
        places = fetch.places
        if reused_from is None:
            queried += 1
        results += len(places)

        payload = {
            "cbg":             cbg_str,
            "h3_index":        h3_idx,
            "h3_resolution":   H3_RESOLUTION,
            "centroid_lat":    lat,
            "centroid_lng":    lng,
            "search_radius_m": SEARCH_RADIUS_M,
            "scraped_at":      datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_results":   len(places),
            "api_response_reused": reused_from is not None,
            "reused_from": str(reused_from) if reused_from is not None else None,
            "places":          places,
        }
        (HEX_DIR / f"{cbg_str}_{h3_idx}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
        _append_index(places, cbg_str, h3_idx)
        if h3_cache is not None:
            h3_cache.setdefault(h3_idx, HEX_DIR / f"{cbg_str}_{h3_idx}.json")
        if reused_from is None:
            time.sleep(0.2)

        if len(places) == 0:
            consecutive_empty += 1
            if empty_stop > 0 and consecutive_empty >= empty_stop:
                completion_reason = "consecutive_empty_stop"
                remaining = len(pending) - queried - reused
                log.info(
                    f"CBG {cbg_str}: {empty_stop} consecutive center-out cells "
                    f"empty — intentionally skipping {remaining} remaining cells"
                )
                break
        else:
            consecutive_empty = 0

    # Both a full scan and an intentional empty-stop are terminal. Existing
    # empty CBGs therefore remain excluded on every subsequent run.
    _write_cbg_status(
        cbg_str,
        state="done",
        completion_reason=completion_reason,
        total_cells=len(hexes),
        polygon_cells=polygon_cell_count,
        pending_at_start=len(pending),
        successful_calls=queried,
        reused_cells=reused,
        results=results,
        empty_stop=empty_stop,
        search_scope=(
            "priority_cells" if max_h3_cells is not None else "full_cbg"
        ),
    )
    completion_marker_dir.mkdir(parents=True, exist_ok=True)
    (completion_marker_dir / f"{cbg_str}.done").touch()
    return True


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Tile CBGs with H3 hexagons, query Places API")
    parser.add_argument("--cbg",      help="Process a single CBG FIPS")
    parser.add_argument("--cbg-file", default=str(CBG_LIST_FILE),
                        help=f"CSV with cbg_str column (default: {CBG_LIST_FILE})")
    parser.add_argument("--limit",    type=int, help="Cap number of CBGs to process")
    parser.add_argument(
        "--empty-stop", type=int, default=2,
        help="Mark a CBG done after this many consecutive empty center-out cells; 0 disables (default: 2)",
    )
    parser.add_argument("--budget-id", help="Persistent identifier for this capped run")
    parser.add_argument("--billing-month", default=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m"))
    parser.add_argument("--max-monthly-api-calls", type=int,
                        help="Hard ceiling across the local monthly API ledger")
    parser.add_argument("--max-run-cost-usd", type=float,
                        help="Hard estimated list-price ceiling for this budget ID")
    parser.add_argument(
        "--unit-cost-usd", type=float, default=PLACES_NEARBY_PRO_UNIT_COST_USD,
        help="May be raised for conservatism but not set below the published Pro rate",
    )
    parser.add_argument("--budget-status-only", action="store_true",
                        help="Print both guard balances and exit without API calls")
    args = parser.parse_args()

    if args.empty_stop < 0:
        parser.error("--empty-stop must be zero or positive")
    if not args.budget_id or args.max_monthly_api_calls is None or args.max_run_cost_usd is None:
        parser.error(
            "--budget-id, --max-monthly-api-calls, and --max-run-cost-usd are required; "
            "Places collection fails closed without both safeguards"
        )
    try:
        budget_guard = PlacesBudgetGuard(
            budget_id=args.budget_id,
            billing_month=args.billing_month,
            max_monthly_calls=args.max_monthly_api_calls,
            max_run_cost_usd=args.max_run_cost_usd,
            unit_cost_usd=args.unit_cost_usd,
        )
        budget_status = budget_guard.status()
    except (ValueError, PlacesBudgetExceeded) as error:
        parser.error(str(error))
    log.info(
        "Budget guards armed: month %s/%s calls; run %s calls reserved ($%.2f/$%.2f)",
        f"{budget_status['accounted_month_calls']:,}", f"{args.max_monthly_api_calls:,}",
        f"{budget_status['reserved_calls']:,}", budget_status["estimated_run_cost_usd"],
        args.max_run_cost_usd,
    )
    if args.budget_status_only:
        log.info(
            "No-call status: at most %s additional requests allowed by both guards",
            f"{min(budget_status['remaining_month_calls'], budget_status['remaining_run_calls']):,}",
        )
        return
    if not GOOGLE_PLACES_API_KEY:
        parser.error("GOOGLE_PLACES_API_KEY is not set or is empty")

    _init_csv(INDEX_FILE, INDEX_COLS)
    _init_csv(API_LOG_FILE, LOG_COLS)

    if args.cbg:
        cbg_list = [args.cbg]
    else:
        cbg_file = Path(args.cbg_file)
        if not cbg_file.exists():
            log.error(f"CBG list not found: {cbg_file}. Export top100['cbg_str'].to_csv('data/top100_cbgs.csv', index=False) from the notebook first.")
            return
        queue = pd.read_csv(cbg_file, dtype=str)
        # The density planner spatially crosswalks legacy done CBGs to current
        # boundaries. Honor its flag even if the caller passes the full plan
        # instead of the already-filtered incremental file.
        if "pending" in queue.columns:
            pending_flag = queue["pending"].str.lower().isin({"true", "1", "yes"})
            queue = queue[pending_flag]
        elif "previously_queried" in queue.columns:
            prior_flag = queue["previously_queried"].str.lower().isin(
                {"true", "1", "yes"}
            )
            queue = queue[~prior_flag]
        cbg_list = queue["cbg_str"].str.zfill(12).tolist()
        if args.limit:
            cbg_list = cbg_list[:args.limit]

    search_centers: dict[str, tuple[float, float]] = {}
    search_cell_limits: dict[str, int] = {}
    if SEARCH_SEEDS_FILE.exists():
        seed_rows = pd.read_csv(SEARCH_SEEDS_FILE, dtype={"cbg_str": str})
        required = {"cbg_str", "search_lat", "search_lng"}
        if not required.issubset(seed_rows.columns):
            parser.error(
                f"{SEARCH_SEEDS_FILE} must contain {sorted(required)}"
            )
        for row in seed_rows.itertuples(index=False):
            search_centers[str(row.cbg_str).zfill(12)] = (
                float(row.search_lat), float(row.search_lng)
            )
            raw_limit = getattr(row, "max_h3_cells", None)
            if raw_limit is not None and not pd.isna(raw_limit):
                search_cell_limits[str(row.cbg_str).zfill(12)] = int(raw_limit)

    # Filter out already-completed CBGs before fetching any geometries
    def completion_marker(cbg: str) -> Path:
        directory = (
            PRIORITY_DONE_DIR if cbg in search_cell_limits else DONE_DIR
        )
        return directory / f"{cbg}.done"

    pending_cbgs = [c for c in cbg_list if not completion_marker(c).exists()]
    skipped = len(cbg_list) - len(pending_cbgs)
    log.info(f"{len(cbg_list)} CBGs total — {skipped} already done, {len(pending_cbgs)} to process")

    if not pending_cbgs:
        log.info("Nothing to do.")
        return

    geom_map = load_geometries(pending_cbgs)

    h3_cache = build_h3_cache()
    log.info(f"Loaded {len(h3_cache)} prior H3 responses for cross-CBG reuse")
    consecutive_failures = 0
    for cbg_str in pending_cbgs:
        geom = geom_map.get(cbg_str)
        if geom is None:
            log.warning(f"No geometry for {cbg_str} — skipping")
            continue
        completed = process_cbg(
            cbg_str, geom, budget_guard, empty_stop=args.empty_stop,
            h3_cache=h3_cache, search_center=search_centers.get(cbg_str),
            max_h3_cells=search_cell_limits.get(cbg_str),
            completion_marker_dir=(
                PRIORITY_DONE_DIR if cbg_str in search_cell_limits else DONE_DIR
            ),
        )
        if completed:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                log.error(
                    "Three consecutive API failures; stopping collection so a "
                    "configuration or network problem cannot churn through the queue"
                )
                raise SystemExit(2)

    log.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except PlacesBudgetExceeded as error:
        log.error("BUDGET GUARD STOP: %s", error)
        raise SystemExit(78)
