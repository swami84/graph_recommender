#!/usr/bin/env python3
"""
build_graph_data.py — Phase 1 data preparation for the recommendation graph.

Steps:
  1a. Cuisine taxonomy: map 320 Google types → 17 canonical categories
      → data/restaurants_enriched.parquet
  1b. Review flattening: parse meta fields, add text features, join demographics
      → data/reviews_flat.parquet
  1c. Dish extraction: meta["Recommended dishes"] + spaCy NLP on review text
      → data/dishes.parquet, data/review_dishes.parquet

Usage:
    python build_graph_data.py              # all steps
    python build_graph_data.py --step 1a   # single step
"""

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_graph_data")

# ── Paths ──────────────────────────────────────────────────────────────────────
HEX_INDEX       = Path("data/hex_restaurants_index.csv")
REVIEWS_DIR     = Path("data/reviews")
NAME_PREDS      = Path("data/name_predictions.parquet")
OUT_RESTAURANTS = Path("data/restaurants_enriched.parquet")
OUT_REVIEWS     = Path("data/reviews_flat.parquet")
OUT_DISHES      = Path("data/dishes.parquet")
OUT_REV_DISHES  = Path("data/review_dishes.parquet")

# ── Cuisine taxonomy ───────────────────────────────────────────────────────────
# Noise types that carry no cuisine signal — skip when walking types list
NOISE_TYPES = {
    "restaurant", "food", "point_of_interest", "establishment",
    "service", "store", "food_store", "meal_delivery", "meal_takeaway",
    "food_delivery", "catering_service",
}

# Maps a Google Places type token → canonical cuisine category.
# For each restaurant, walk its types list, skip noise, return the first match.
CUISINE_MAP = {
    # American & Comfort
    "american_restaurant":       "American & Comfort",
    "brunch_restaurant":         "American & Comfort",
    "diner":                     "American & Comfort",
    "bar_and_grill":             "American & Comfort",
    "chicken_restaurant":        "American & Comfort",
    "chicken_wings_restaurant":  "American & Comfort",
    "southern_us_restaurant":    "American & Comfort",
    # Fast Food & Burgers
    "fast_food_restaurant":      "Fast Food & Burgers",
    "hamburger_restaurant":      "Fast Food & Burgers",
    "hot_dog_stand":             "Fast Food & Burgers",
    # Mexican & Latin
    "mexican_restaurant":        "Mexican & Latin",
    "tex_mex_restaurant":        "Mexican & Latin",
    "taco_restaurant":           "Mexican & Latin",
    "latin_american_restaurant": "Mexican & Latin",
    "salvadoran_restaurant":     "Mexican & Latin",
    "cuban_restaurant":          "Mexican & Latin",
    "colombian_restaurant":      "Mexican & Latin",
    "peruvian_restaurant":       "Mexican & Latin",
    "brazilian_restaurant":      "Mexican & Latin",
    # Chinese
    "chinese_restaurant":        "Chinese",
    "dim_sum_restaurant":        "Chinese",
    "cantonese_restaurant":      "Chinese",
    "hot_pot_restaurant":        "Chinese",
    # Japanese & Sushi
    "japanese_restaurant":       "Japanese & Sushi",
    "sushi_restaurant":          "Japanese & Sushi",
    "ramen_restaurant":          "Japanese & Sushi",
    "noodle_restaurant":         "Japanese & Sushi",
    # Asian (other)
    "thai_restaurant":           "Asian (Other)",
    "vietnamese_restaurant":     "Asian (Other)",
    "korean_restaurant":         "Asian (Other)",
    "asian_restaurant":          "Asian (Other)",
    "pan_asian_restaurant":      "Asian (Other)",
    "cambodian_restaurant":      "Asian (Other)",
    "philippine_restaurant":     "Asian (Other)",
    "indonesian_restaurant":     "Asian (Other)",
    "malaysian_restaurant":      "Asian (Other)",
    # Indian & South Asian
    "indian_restaurant":         "Indian & South Asian",
    "pakistani_restaurant":      "Indian & South Asian",
    "bangladeshi_restaurant":    "Indian & South Asian",
    "sri_lankan_restaurant":     "Indian & South Asian",
    # Mediterranean & Middle Eastern
    "mediterranean_restaurant":  "Mediterranean & Middle Eastern",
    "greek_restaurant":          "Mediterranean & Middle Eastern",
    "turkish_restaurant":        "Mediterranean & Middle Eastern",
    "lebanese_restaurant":       "Mediterranean & Middle Eastern",
    "middle_eastern_restaurant": "Mediterranean & Middle Eastern",
    "israeli_restaurant":        "Mediterranean & Middle Eastern",
    "persian_restaurant":        "Mediterranean & Middle Eastern",
    "afghani_restaurant":        "Mediterranean & Middle Eastern",
    # Italian & Pizza
    "italian_restaurant":        "Italian & Pizza",
    "pizza_restaurant":          "Italian & Pizza",
    # BBQ & Steakhouse
    "steak_house":               "BBQ & Steakhouse",
    "bbq_restaurant":            "BBQ & Steakhouse",
    "barbecue_restaurant":       "BBQ & Steakhouse",
    # Seafood
    "seafood_restaurant":        "Seafood",
    "fish_and_chips_restaurant": "Seafood",
    "sushi_restaurant":          "Japanese & Sushi",  # prefer Japanese bucket
    # Soul Food
    "soul_food_restaurant":      "Soul Food",
    # Café & Bakery
    "cafe":                      "Café & Bakery",
    "coffee_shop":               "Café & Bakery",
    "bakery":                    "Café & Bakery",
    "dessert_shop":              "Café & Bakery",
    "ice_cream_shop":            "Café & Bakery",
    "donut_shop":                "Café & Bakery",
    "sandwich_shop":             "Sandwiches & Deli",
    # Bar & Pub
    "bar":                       "Bar & Pub",
    "pub":                       "Bar & Pub",
    "brewery":                   "Bar & Pub",
    "cocktail_bar":              "Bar & Pub",
    "wine_bar":                  "Bar & Pub",
    "beer_garden":               "Bar & Pub",
    "sports_bar":                "Bar & Pub",
    # Fine Dining
    "fine_dining_restaurant":    "Fine Dining",
    # Sandwiches & Deli
    "sandwich_shop":             "Sandwiches & Deli",
    "deli":                      "Sandwiches & Deli",
    "sub_sandwich_shop":         "Sandwiches & Deli",
}

FALLBACK_CUISINE = "Other"


def classify_cuisine(types_str: str) -> str:
    """Walk a comma-separated types string and return the first mapped category."""
    if not isinstance(types_str, str) or not types_str.strip():
        return FALLBACK_CUISINE
    for t in types_str.split(","):
        t = t.strip()
        if t in NOISE_TYPES:
            continue
        if t in CUISINE_MAP:
            return CUISINE_MAP[t]
    return FALLBACK_CUISINE


# ── 1a. Cuisine normalisation ──────────────────────────────────────────────────
def step_1a():
    log.info("1a: Loading restaurant index…")
    df = pd.read_csv(HEX_INDEX, dtype={"cbg": str})
    log.info(f"    {len(df):,} rows, {df['place_id'].nunique():,} unique restaurants")

    df["cuisine_category"] = df["types"].apply(classify_cuisine)

    # Keep one row per restaurant (first occurrence wins — includes CBG context)
    restaurants = (
        df.drop_duplicates("place_id")
        [[
            "place_id", "name", "cbg", "h3_index",
            "lat", "lng", "rating", "user_rating_count",
            "price_level", "business_status", "cuisine_category",
            "types", "address", "phone", "website",
        ]]
        .copy()
    )

    log.info(f"    Cuisine distribution:\n{restaurants['cuisine_category'].value_counts().to_string()}")
    restaurants.to_parquet(OUT_RESTAURANTS, index=False)
    log.info(f"    Saved {len(restaurants):,} restaurants → {OUT_RESTAURANTS}")


# ── 1b. Review flattening ──────────────────────────────────────────────────────
_PRICE_RE = re.compile(r"\$(\d+)\D+(\d+)?")


def _parse_price(s) -> float | None:
    """'$10–20' → 15.0,  '$100+' → 100.0,  '' → None"""
    if not isinstance(s, str):
        return None
    m = _PRICE_RE.search(s)
    if not m:
        return None
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else lo
    return (lo + hi) / 2.0


_TS_PATTERNS = [
    (re.compile(r"(\d+)\s+year"),    365),
    (re.compile(r"a\s+year"),         365),
    (re.compile(r"(\d+)\s+month"),    30),
    (re.compile(r"a\s+month"),         30),
    (re.compile(r"(\d+)\s+week"),      7),
    (re.compile(r"a\s+week"),           7),
    (re.compile(r"(\d+)\s+day"),        1),
    (re.compile(r"yesterday"),          1),
    (re.compile(r"today|just now"),     0),
]


def _parse_days_ago(s) -> int | None:
    if not isinstance(s, str):
        return None
    s = s.lower()
    for pat, unit in _TS_PATTERNS:
        m = pat.search(s)
        if m:
            n = int(m.group(1)) if m.lastindex and m.group(1) else 1
            return n * unit
    return None


def step_1b():
    log.info("1b: Flattening reviews…")
    files = list(REVIEWS_DIR.glob("*.json"))
    log.info(f"    {len(files):,} review files")

    records = []
    for fpath in files:
        try:
            data = json.loads(fpath.read_text())
        except Exception:
            continue
        place_id = data.get("place_id", "")
        for r in data.get("reviews", []):
            meta = r.get("meta") or {}
            text = (r.get("text") or "").strip()
            records.append({
                "review_id":         r.get("review_id", ""),
                "place_id":          place_id,
                "contributor_id":    r.get("contributor_id", ""),
                "reviewer_name":     (r.get("reviewer_name") or "").strip(),
                "is_local_guide":    bool(r.get("is_local_guide", False)),
                "reviewer_reviews":  r.get("reviewer_reviews"),
                "reviewer_photos":   r.get("reviewer_photos"),
                "rating":            r.get("rating"),
                "timestamp":         r.get("timestamp", ""),
                "timestamp_days_ago": _parse_days_ago(r.get("timestamp", "")),
                "text":              text,
                "text_len":          len(text),
                "has_content":       len(text) >= 50,
                "attached_photos":   r.get("attached_photos", 0),
                # parsed meta
                "meal_type":         meta.get("Meal type"),
                "price_per_person":  _parse_price(meta.get("Price per person")),
                "food_score":        meta.get("Food"),
                "service_score":     meta.get("Service"),
                "atmosphere_score":  meta.get("Atmosphere"),
                "recommended_dishes": meta.get("Recommended dishes", ""),
            })

    df = pd.DataFrame(records)
    df = df[df["reviewer_name"] != ""].reset_index(drop=True)

    # Join demographics if available
    if NAME_PREDS.exists():
        preds = pd.read_parquet(NAME_PREDS)[
            ["reviewer_name", "predicted_race", "race_score", "predicted_gender",
             "race_HL", "race_API", "race_BLACK", "race_WHITE"]
        ]
        df = df.merge(preds, on="reviewer_name", how="left")
        log.info(f"    Demographics joined for {df['predicted_race'].notna().sum():,} reviews")

    df.to_parquet(OUT_REVIEWS, index=False)
    log.info(f"    Saved {len(df):,} reviews → {OUT_REVIEWS}")
    return df


# ── 1c. Dish extraction ────────────────────────────────────────────────────────
def _normalise_dish(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def step_1c(reviews_df: pd.DataFrame | None = None):
    log.info("1c: Extracting dishes…")

    if reviews_df is None:
        if OUT_REVIEWS.exists():
            reviews_df = pd.read_parquet(OUT_REVIEWS)
        else:
            log.error("Run step 1b first.")
            return

    dish_rows = []
    rev_dish_rows = []

    # Pass 1: meta["Recommended dishes"]
    for _, row in reviews_df.iterrows():
        raw = row.get("recommended_dishes", "") or ""
        if not raw.strip():
            continue
        for d in raw.split(","):
            name = _normalise_dish(d)
            if len(name) < 3:
                continue
            dish_rows.append({"dish_name": name, "place_id": row["place_id"]})
            rev_dish_rows.append({
                "review_id": row["review_id"],
                "dish_name": name,
                "place_id":  row["place_id"],
                "source":    "meta",
            })

    log.info(f"    Pass 1 (meta): {len(dish_rows):,} dish mentions")

    dishes_df = (
        pd.DataFrame(dish_rows)
        .drop_duplicates(["dish_name", "place_id"])
        .reset_index(drop=True)
    )
    dishes_df["dish_id"] = dishes_df.apply(
        lambda r: hashlib.md5(f"{r['place_id']}::{r['dish_name']}".encode()).hexdigest()[:12],
        axis=1,
    )

    rev_dishes_df = pd.DataFrame(rev_dish_rows).drop_duplicates(
        ["review_id", "dish_name", "place_id"]
    ).reset_index(drop=True)

    dishes_df.to_parquet(OUT_DISHES, index=False)
    rev_dishes_df.to_parquet(OUT_REV_DISHES, index=False)
    log.info(f"    Saved {len(dishes_df):,} unique dishes → {OUT_DISHES}")
    log.info(f"    Saved {len(rev_dishes_df):,} review-dish edges → {OUT_REV_DISHES}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build graph data artefacts")
    parser.add_argument("--step", choices=["1a", "1b", "1c"], help="Run a single step only")
    args = parser.parse_args()

    if args.step == "1a":
        step_1a()
    elif args.step == "1b":
        step_1b()
    elif args.step == "1c":
        step_1c()
    else:
        step_1a()
        reviews_df = step_1b()
        step_1c(reviews_df)


if __name__ == "__main__":
    main()
