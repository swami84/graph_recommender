#!/usr/bin/env python3
"""
build_extended_features.py — Rich user and restaurant side features for the GNN.
Polars-native implementation for fast multi-threaded computation.

Computes features not derivable from the interaction matrix:
  User: cuisine affinity vector, cuisine entropy, price sensitivity,
        geographic range, unique CBG/state/county/tract count, CBG entropy,
        mean CBG distance, rating pickiness, sequential signals, meal type
        preferences, sub-score emphasis (food/service/ambiance), photo
        engagement, rating distribution (pct_5star, pct_1star, rating_entropy),
        price deviation.
  Restaurant: dish diversity, location density, review trend, text engagement,
              pct_5star, rating bimodality, avg review length, avg photos/review.

Outputs:
  data/user_extended_features.parquet
  data/item_extended_features.parquet

Usage:
    python -m foodie.features.build_extended_features
"""

import argparse
import logging
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import BallTree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ext_feat")

REVIEWS_FLAT    = Path("data/reviews_flat.parquet")
RESTAURANTS_ENR = Path("data/restaurants_enriched.parquet")
DISHES_FILE     = Path("data/dishes.parquet")
PROFILES_FILE   = Path("data/dish_profiles.parquet")
CBG_PATTERNS    = Path("data/cbg_patterns.csv")
USER_EXT_OUT    = Path("data/user_extended_features.parquet")
ITEM_EXT_OUT    = Path("data/item_extended_features.parquet")

_REVIEWS_COLS = [
    "contributor_id", "place_id", "rating", "timestamp_days_ago",
    "text_len", "has_content", "food_score", "service_score", "atmosphere_score",
    "meal_type", "price_per_person", "attached_photos", "is_local_guide",
    "predicted_race",
]

PRICE_MAP = {
    "PRICE_LEVEL_INEXPENSIVE": 1.0,
    "PRICE_LEVEL_MODERATE":    2.0,
    "PRICE_LEVEL_EXPENSIVE":   3.0,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4.0,
}
_PRICE_TIER_MID = {1.0: 12.5, 2.0: 22.5, 3.0: 37.5, 4.0: 62.5}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _norm(col: str, log1p: bool = False) -> pl.Expr:
    """Normalize a column to [0, 1]. Returns 0.5 when max == min."""
    e = pl.col(col).cast(pl.Float64)
    if log1p:
        e = (e + 1.0).log()
    mn = e.min()
    mx = e.max()
    return (
        pl.when(mx == mn)
        .then(pl.lit(0.5))
        .otherwise((e - mn) / (mx - mn))
        .alias(col)
    )


def _norm_arr(arr: np.ndarray, log1p: bool = False) -> np.ndarray:
    arr = arr.astype(float)
    if log1p:
        arr = np.log1p(arr)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.full_like(arr, 0.5)
    return (arr - mn) / (mx - mn)


# ── Dish diversity helpers (numpy-based, used in item features) ────────────────

def _category_entropy(feat_mat: np.ndarray) -> float:
    if feat_mat.shape[0] == 0:
        return 0.0
    cat_counts = feat_mat[:, :18].sum(axis=0)
    total = cat_counts.sum()
    if total == 0:
        return 0.0
    probs = cat_counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _flavor_diversity(feat_mat: np.ndarray) -> float:
    if feat_mat.shape[0] < 2:
        return 0.0
    flavor = feat_mat[:, 54:66].astype(float)
    norms = np.linalg.norm(flavor, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return float(np.mean(np.std(flavor / norms, axis=0)))


# ── User extended features ─────────────────────────────────────────────────────

def build_user_extended(reviews: pl.DataFrame, restaurants: pl.DataFrame) -> pl.DataFrame:
    import time as _t
    _t0 = _t.time()
    def _step(name): log.info(f"  user feat — {name} ({_t.time()-_t0:.0f}s elapsed)")

    # ── Prep: extract numeric sub-scores ──────────────────────────────────────
    for sc in ["food_score", "service_score", "atmosphere_score"]:
        col_num = sc.replace("score", "score_num")
        if sc in reviews.columns:
            reviews = reviews.with_columns(
                pl.col(sc).cast(pl.Utf8)
                  .str.extract(r"(\d+(?:\.\d+)?)")
                  .cast(pl.Float64, strict=False)
                  .alias(col_num)
            )
        else:
            reviews = reviews.with_columns(pl.lit(None, dtype=pl.Float64).alias(col_num))

    reviews = reviews.with_columns(
        pl.col("rating").cast(pl.Float64, strict=False).alias("rating_num")
    )

    pm_keys = list(PRICE_MAP.keys())
    pm_vals = [float(v) for v in PRICE_MAP.values()]
    restaurants = restaurants.with_columns(
        pl.col("price_level").replace_strict(pm_keys, pm_vals, default=2.0)
          .cast(pl.Float64).alias("price_num"),
        pl.col("lat").cast(pl.Float64, strict=False),
        pl.col("lng").cast(pl.Float64, strict=False),
        pl.col("rating").cast(pl.Float64, strict=False).alias("restaurant_rating_num"),
    )

    result = reviews.select("contributor_id").unique()

    _step("1 cuisine affinity")
    # ── Feature 1 — Cuisine affinity vector ───────────────────────────────────
    rest_cuisine = restaurants.select(
        "place_id",
        pl.col("cuisine_category").fill_null("other").alias("cuisine_category"),
    )
    cuis_merged = (
        reviews.select(["contributor_id", "place_id"])
        .join(rest_cuisine, on="place_id", how="left")
        .with_columns(pl.col("cuisine_category").fill_null("other"))
    )
    cuis_counts = (
        cuis_merged.group_by(["contributor_id", "cuisine_category"])
        .agg(pl.len().alias("cnt"))
        .with_columns(
            (pl.col("cnt").cast(pl.Float64) / pl.col("cnt").sum().over("contributor_id"))
            .alias("frac")
        )
    )
    cuis_pivot = cuis_counts.pivot(
        values="frac", index="contributor_id", on="cuisine_category",
        aggregate_function="first",
    ).fill_null(0.0)
    cuis_pivot = cuis_pivot.rename({
        c: "cuisine_aff_" + c.lower().replace(" ", "_").replace("/", "_")
        for c in cuis_pivot.columns if c != "contributor_id"
    })
    cuisine_aff_cols = [c for c in cuis_pivot.columns if c != "contributor_id"]
    result = result.join(cuis_pivot, on="contributor_id", how="left")
    result = result.with_columns([pl.col(c).fill_null(0.0) for c in cuisine_aff_cols])

    _step("2 price sensitivity")
    # ── Feature 2 — Price sensitivity ─────────────────────────────────────────
    price_sens = (
        reviews.select(["contributor_id", "place_id"])
        .join(restaurants.select(["place_id", "price_num"]), on="place_id", how="left")
        .group_by("contributor_id")
        .agg(pl.col("price_num").median().alias("price_sensitivity"))
        .with_columns(
            ((pl.col("price_sensitivity") - 1.0) / 3.0).clip(0.0, 1.0)
        )
    )
    result = result.join(price_sens, on="contributor_id", how="left")
    result = result.with_columns(pl.col("price_sensitivity").fill_null(0.5))

    _step("3 geographic range")
    # ── Feature 3 — Geographic range ─────────────────────────────────────────
    geo_agg = (
        reviews.select(["contributor_id", "place_id"])
        .join(restaurants.select(["place_id", "lat", "lng"]), on="place_id", how="left")
        .group_by("contributor_id")
        .agg(
            pl.col("lat").std().fill_null(0.0).alias("lat_std"),
            pl.col("lng").std().fill_null(0.0).alias("lng_std"),
        )
        .with_columns(
            ((pl.col("lat_std") ** 2 + pl.col("lng_std") ** 2).sqrt()).alias("geo_range")
        )
        .with_columns(_norm("geo_range"))
        .select(["contributor_id", "geo_range"])
    )
    result = result.join(geo_agg, on="contributor_id", how="left")
    result = result.with_columns(pl.col("geo_range").fill_null(0.5))

    _step("3b cuisine entropy")
    # ── Feature 3b — Cuisine diversity (entropy) ─────────────────────────────
    aff_mat = result.select(cuisine_aff_cols).to_numpy() + 1e-9
    aff_norm = aff_mat / aff_mat.sum(axis=1, keepdims=True)
    raw_ent = -np.sum(aff_norm * np.log(aff_norm), axis=1)
    max_ent = np.log(len(cuisine_aff_cols))
    result = result.with_columns(
        pl.Series("cuisine_entropy", (raw_ent / max_ent).clip(0, 1))
    )

    _step("3c CBG diversity")
    # ── Feature 3c — CBG diversity features ──────────────────────────────────
    cbg_centroids = (
        restaurants
        .filter(pl.col("lat").is_not_null() & pl.col("lng").is_not_null() & pl.col("cbg").is_not_null())
        .group_by("cbg")
        .agg(pl.col("lat").mean(), pl.col("lng").mean())
    )
    cbg_merged = (
        reviews.select(["contributor_id", "place_id"])
        .join(restaurants.select(["place_id", "cbg"]), on="place_id", how="left")
        .drop_nulls(subset=["cbg"])
        .with_columns(pl.col("cbg").cast(pl.Utf8).str.zfill(12).alias("cbg_str"))
        .with_columns([
            pl.col("cbg_str").str.slice(0, 2).alias("state_fips"),
            pl.col("cbg_str").str.slice(0, 5).alias("county_fips"),
            pl.col("cbg_str").str.slice(0, 11).alias("tract_fips"),
        ])
    )

    for agg_col, out_col in [
        ("cbg",         "unique_cbg_count"),
        ("state_fips",  "unique_state_count"),
        ("county_fips", "unique_county_count"),
        ("tract_fips",  "unique_tract_count"),
    ]:
        uniq = (
            cbg_merged.group_by("contributor_id")
            .agg(pl.col(agg_col).n_unique().cast(pl.Float64).alias(out_col))
            .with_columns(_norm(out_col))
        )
        result = result.join(uniq, on="contributor_id", how="left")
        result = result.with_columns(pl.col(out_col).fill_null(0.0))

    # cbg_entropy: vectorized, no apply()
    cbg_cnt = (
        cbg_merged.group_by(["contributor_id", "cbg"])
        .agg(pl.len().alias("cnt"))
        .with_columns(
            (pl.col("cnt").cast(pl.Float64) / pl.col("cnt").sum().over("contributor_id")).alias("p")
        )
        .with_columns(-(pl.col("p") * (pl.col("p") + 1e-9).log()).alias("h"))
    )
    cbg_ent = (
        cbg_cnt.group_by("contributor_id")
        .agg(pl.col("h").sum().alias("cbg_entropy"))
        .with_columns(_norm("cbg_entropy"))
    )
    result = result.join(cbg_ent, on="contributor_id", how="left")
    result = result.with_columns(pl.col("cbg_entropy").fill_null(0.0))

    # mean_cbg_distance: precomputed haversine matrix + vectorized self-join
    rel_cbgs = (
        cbg_centroids
        .filter(pl.col("cbg").is_in(cbg_merged["cbg"].unique()))
        .sort("cbg")
    )
    if len(rel_cbgs) >= 2:
        rel_cbgs_pd = rel_cbgs.to_pandas()
        cbg_list   = rel_cbgs_pd["cbg"].tolist()
        cbg_to_idx = {cbg: i for i, cbg in enumerate(cbg_list)}
        dist_matrix = haversine_distances(
            np.radians(rel_cbgs_pd[["lat", "lng"]].values)
        ) * 6371.0

        idx_keys = list(cbg_to_idx.keys())
        idx_vals = [int(v) for v in cbg_to_idx.values()]
        user_cbg = (
            cbg_merged.select(["contributor_id", "cbg"]).unique()
            .with_columns(
                pl.col("cbg").replace_strict(idx_keys, idx_vals, default=None)
                  .cast(pl.Int32, strict=False).alias("cbg_idx")
            )
            .drop_nulls(subset=["cbg_idx"])
        )
        pairs = (
            user_cbg.join(user_cbg, on="contributor_id", suffix="_r")
            .filter(pl.col("cbg_idx") < pl.col("cbg_idx_r"))
        )
        if len(pairs) > 0:
            dists = dist_matrix[pairs["cbg_idx"].to_numpy(), pairs["cbg_idx_r"].to_numpy()]
            pairs = pairs.with_columns(pl.Series("dist", dists))
            cbg_dist = (
                pairs.group_by("contributor_id")
                .agg(pl.col("dist").mean().alias("mean_cbg_distance"))
                .with_columns(_norm("mean_cbg_distance"))
            )
            result = result.join(cbg_dist, on="contributor_id", how="left")

    if "mean_cbg_distance" not in result.columns:
        result = result.with_columns(pl.lit(0.0).alias("mean_cbg_distance"))
    else:
        result = result.with_columns(pl.col("mean_cbg_distance").fill_null(0.0))

    _step("4 rating pickiness")
    # ── Feature 4 — Rating pickiness ─────────────────────────────────────────
    pickiness = (
        reviews.select(["contributor_id", "place_id", "food_score_num"])
        .join(restaurants.select(["place_id", "restaurant_rating_num"]), on="place_id", how="left")
        .with_columns((pl.col("food_score_num") - pl.col("restaurant_rating_num")).alias("diff"))
        .group_by("contributor_id")
        .agg(pl.col("diff").mean().fill_null(0.0).alias("rating_pickiness"))
        .with_columns(_norm("rating_pickiness"))
    )
    result = result.join(pickiness, on="contributor_id", how="left")
    result = result.with_columns(pl.col("rating_pickiness").fill_null(0.5))

    _step("5 sequential ratings")
    # ── Feature 5 — Sequential rating features ───────────────────────────────
    seq = reviews.filter(pl.col("rating_num").is_not_null()).sort("timestamp_days_ago")

    last_rating = (
        seq.group_by("contributor_id")
        .agg(pl.col("rating_num").first().alias("last_rating"))
        .with_columns(((pl.col("last_rating") - 1.0) / 4.0).clip(0.0, 1.0))
    )
    last3 = (
        seq.group_by("contributor_id")
        .agg(pl.col("rating_num").head(3).mean().alias("last3_rating_mean"))
        .with_columns(((pl.col("last3_rating_mean") - 1.0) / 4.0).clip(0.0, 1.0))
    )
    rat_var = (
        seq.group_by("contributor_id")
        .agg(pl.col("rating_num").std().fill_null(0.0).alias("rating_variance"))
        .with_columns(_norm("rating_variance"))
    )
    days_since = (
        reviews.group_by("contributor_id")
        .agg(pl.col("timestamp_days_ago").min().alias("days_since_last"))
        .with_columns(_norm("days_since_last", log1p=True))
    )
    for df, col, fill in [
        (last_rating, "last_rating",       0.5),
        (last3,       "last3_rating_mean", 0.5),
        (rat_var,     "rating_variance",   0.0),
        (days_since,  "days_since_last",   0.5),
    ]:
        result = result.join(df, on="contributor_id", how="left")
        result = result.with_columns(pl.col(col).fill_null(fill))

    _step("6 meal type")
    # ── Feature 6 — Meal type preferences ────────────────────────────────────
    if "meal_type" in reviews.columns:
        meal_valid = (
            reviews.select(["contributor_id", "meal_type"])
            .with_columns(
                pl.col("meal_type").cast(pl.Utf8)
                  .str.extract(r"(Dinner|Lunch|Brunch|Breakfast)")
                  .alias("meal_clean")
            )
            .drop_nulls(subset=["meal_clean"])
        )
        if len(meal_valid) > 0:
            meal_counts = (
                meal_valid.group_by(["contributor_id", "meal_clean"])
                .agg(pl.len().alias("cnt"))
                .with_columns(
                    (pl.col("cnt").cast(pl.Float64) / pl.col("cnt").sum().over("contributor_id"))
                    .alias("frac")
                )
            )
            meal_pivot = meal_counts.pivot(
                values="frac", index="contributor_id", on="meal_clean",
                aggregate_function="first",
            ).fill_null(0.0)
            meal_pivot = meal_pivot.rename({
                c: f"meal_pref_{c.lower()}"
                for c in meal_pivot.columns if c != "contributor_id"
            })
            result = result.join(meal_pivot, on="contributor_id", how="left")

    for col in ["meal_pref_dinner", "meal_pref_lunch", "meal_pref_brunch", "meal_pref_breakfast"]:
        if col not in result.columns:
            result = result.with_columns(pl.lit(0.0).alias(col))
        else:
            result = result.with_columns(pl.col(col).fill_null(0.0))

    _step("7 sub-score emphasis")
    # ── Feature 7 — Sub-score emphasis ───────────────────────────────────────
    sub_rev = reviews.filter(
        pl.col("food_score_num").is_not_null() &
        pl.col("service_score_num").is_not_null() &
        pl.col("atmosphere_score_num").is_not_null()
    ).select(["contributor_id", "food_score_num", "service_score_num", "atmosphere_score_num"]).with_columns(
        ((pl.col("food_score_num") + pl.col("service_score_num") + pl.col("atmosphere_score_num")) / 3.0)
        .alias("_sub_mean")
    )

    if len(sub_rev) > 0:
        for feat_name, score_col in [
            ("food_emphasis",     "food_score_num"),
            ("service_emphasis",  "service_score_num"),
            ("ambiance_emphasis", "atmosphere_score_num"),
        ]:
            emp = (
                sub_rev.with_columns((pl.col(score_col) - pl.col("_sub_mean")).alias(feat_name))
                .group_by("contributor_id")
                .agg(pl.col(feat_name).mean())
                .with_columns(_norm(feat_name))
            )
            result = result.join(emp, on="contributor_id", how="left")
            result = result.with_columns(pl.col(feat_name).fill_null(0.5))
    else:
        for col in ["food_emphasis", "service_emphasis", "ambiance_emphasis"]:
            result = result.with_columns(pl.lit(0.5).alias(col))

    _step("8 photo engagement")
    # ── Feature 8 — Photo engagement ──────────────────────────────────────────
    _photo_col = next(
        (c for c in ["attached_photos", "num_photos", "photo_count"] if c in reviews.columns), None
    )
    if _photo_col:
        photo_df = (
            reviews.group_by("contributor_id")
            .agg(pl.col(_photo_col).cast(pl.Float64).mean().alias("photo_posting_rate"))
            .with_columns(_norm("photo_posting_rate", log1p=True))
        )
        result = result.join(photo_df, on="contributor_id", how="left")
        result = result.with_columns(pl.col("photo_posting_rate").fill_null(0.0))
    else:
        result = result.with_columns(pl.lit(0.0).alias("photo_posting_rate"))

    _step("9 rating distribution")
    # ── Feature 9 — Rating distribution ───────────────────────────────────────
    rv = reviews.filter(pl.col("rating_num").is_not_null()).with_columns(
        pl.col("rating_num").clip(1, 5).round().cast(pl.Int32).alias("_ri")
    )
    if len(rv) > 0:
        pct5 = (
            rv.with_columns((pl.col("_ri") == 5).cast(pl.Float64).alias("_is5"))
            .group_by("contributor_id").agg(pl.col("_is5").mean().alias("pct_5star"))
        )
        pct1 = (
            rv.with_columns((pl.col("_ri") == 1).cast(pl.Float64).alias("_is1"))
            .group_by("contributor_id").agg(pl.col("_is1").mean().alias("pct_1star"))
        )
        rat_ent = (
            rv.group_by(["contributor_id", "_ri"])
            .agg(pl.len().alias("_cnt"))
            .with_columns(
                (pl.col("_cnt").cast(pl.Float64) / pl.col("_cnt").sum().over("contributor_id"))
                .alias("_p")
            )
            .with_columns(-(pl.col("_p") * (pl.col("_p") + 1e-9).log()).alias("_h"))
            .group_by("contributor_id")
            .agg(pl.col("_h").sum().alias("rating_entropy"))
            .with_columns(_norm("rating_entropy"))
        )
        for df, col in [(pct5, "pct_5star"), (pct1, "pct_1star"), (rat_ent, "rating_entropy")]:
            result = result.join(df, on="contributor_id", how="left")
            result = result.with_columns(pl.col(col).fill_null(0.0))
    else:
        for col in ["pct_5star", "pct_1star", "rating_entropy"]:
            result = result.with_columns(pl.lit(0.0).alias(col))

    _step("10 price deviation")
    # ── Feature 10 — Price deviation ──────────────────────────────────────────
    if "price_per_person" in reviews.columns:
        tm_keys = [float(k) for k in _PRICE_TIER_MID.keys()]
        tm_vals = [float(v) for v in _PRICE_TIER_MID.values()]
        pdev = (
            reviews.select(
                "contributor_id", "place_id",
                pl.col("price_per_person").cast(pl.Float64, strict=False),
            )
            .join(restaurants.select(["place_id", "price_num"]), on="place_id", how="left")
            .with_columns(
                pl.col("price_num").replace_strict(tm_keys, tm_vals, default=None).alias("tier_mid")
            )
            .with_columns((pl.col("price_per_person") - pl.col("tier_mid")).alias("price_dev"))
            .drop_nulls(subset=["price_dev"])
        )
        if len(pdev) > 0:
            pdev_df = (
                pdev.group_by("contributor_id")
                .agg(pl.col("price_dev").mean().alias("price_deviation"))
                .with_columns(_norm("price_deviation"))
            )
            result = result.join(pdev_df, on="contributor_id", how="left")

    if "price_deviation" not in result.columns:
        result = result.with_columns(pl.lit(0.5).alias("price_deviation"))
    else:
        result = result.with_columns(pl.col("price_deviation").fill_null(0.5))

    _step("combine + save")
    result = result.fill_null(0.0).with_columns(pl.col("contributor_id").cast(pl.Utf8))
    n_feat = len([c for c in result.columns if c != "contributor_id"])
    log.info(f"User extended features: {len(result):,} users × {n_feat} features")
    return result


# ── Item extended features ─────────────────────────────────────────────────────

def build_item_extended(
    reviews: pl.DataFrame,
    restaurants: pl.DataFrame,
    dishes=None,
    profiles=None,
) -> pl.DataFrame:
    import time as _t
    _t0 = _t.time()
    def _step(name): log.info(f"  item feat — {name} ({_t.time()-_t0:.0f}s elapsed)")

    rest_pd = restaurants.to_pandas()
    rest_pd["lat"] = pd.to_numeric(rest_pd["lat"], errors="coerce")
    rest_pd["lng"] = pd.to_numeric(rest_pd["lng"], errors="coerce")

    result = restaurants.select("place_id").unique()

    # ── Feature group A — Dish diversity ──────────────────────────────────────
    _step("A dish diversity")
    dishes_pd = None
    if dishes is not None:
        dishes_pd = dishes.to_pandas() if isinstance(dishes, pl.DataFrame) else dishes
        dishes_pd["dish_norm"] = dishes_pd["dish_name"].str.lower().str.strip()
        dish_counts = (
            dishes_pd.groupby("place_id")["dish_norm"]
            .nunique()
            .rename("dish_count")
            .reset_index()
        )
        values = np.log1p(dish_counts["dish_count"].to_numpy(dtype=float))
        mn, mx = values.min(), values.max()
        dish_counts["dish_count"] = (
            (values - mn) / (mx - mn)
            if mx > mn
            else np.full_like(values, 0.5)
        )
        result = result.join(
            pl.from_pandas(dish_counts), on="place_id", how="left"
        )

    if dishes_pd is not None and profiles is not None:
        profiles_pd = profiles.to_pandas() if isinstance(profiles, pl.DataFrame) else profiles

        profiles_pd["dish_norm"] = profiles_pd["dish_name"].str.lower().str.strip()
        feat_cols = [c for c in profiles_pd.columns if c.startswith("f")]
        dish_prof = dishes_pd.merge(
            profiles_pd[["dish_norm"] + feat_cols], on="dish_norm", how="inner"
        )

        div_records = []
        for place_id, grp in dish_prof.groupby("place_id"):
            feat_mat = grp[feat_cols].values.astype(float)
            cuisine_breadth = int((feat_mat[:, 18:34].sum(axis=0) > 0).sum()) if feat_mat.shape[1] > 33 else 0
            div_records.append({
                "place_id":           place_id,
                "dish_cat_entropy":   _category_entropy(feat_mat),
                "dish_flavor_div":    _flavor_diversity(feat_mat),
                "dish_cuisine_breadth": cuisine_breadth,
            })
        if div_records:
            div_pd = pd.DataFrame(div_records)
            for col in ["dish_cat_entropy", "dish_flavor_div", "dish_cuisine_breadth"]:
                arr = div_pd[col].values.astype(float)
                mn, mx = arr.min(), arr.max()
                div_pd[col] = (arr - mn) / (mx - mn) if mx > mn else np.full_like(arr, 0.5)
            div_pl = pl.from_pandas(div_pd)
            result = result.join(div_pl, on="place_id", how="left")

    # When no leakage-safe dish source is supplied, omit the columns entirely.
    # Constant-zero placeholders falsely imply that the old dish features were
    # retained and add no learnable signal.
    if dishes_pd is not None:
        for col in ["dish_count", "dish_cat_entropy", "dish_flavor_div", "dish_cuisine_breadth"]:
            if col not in result.columns:
                result = result.with_columns(pl.lit(0.0).alias(col))
            else:
                result = result.with_columns(pl.col(col).fill_null(0.0))

    # ── Feature group B — Location density ────────────────────────────────────
    _step("B location density")
    rest_valid = rest_pd.dropna(subset=["lat", "lng"]).copy()
    if len(rest_valid) > 0:
        coords_rad = np.radians(rest_valid[["lat", "lng"]].values)
        tree       = BallTree(coords_rad, metric="haversine")
        R          = 6371.0
        counts_500m = tree.query_radius(coords_rad, r=0.5 / R, count_only=True)
        counts_2km  = tree.query_radius(coords_rad, r=2.0 / R, count_only=True)

        same_cuisine = np.zeros(len(rest_valid), dtype=float)
        rest_reset = rest_valid.reset_index(drop=True)
        if "cuisine_category" in rest_reset.columns:
            for cuisine, grp in rest_reset.groupby(rest_reset["cuisine_category"].fillna("")):
                if not cuisine or len(grp) < 2:
                    continue
                ci = grp.index.tolist()
                ct = BallTree(np.radians(grp[["lat", "lng"]].values), metric="haversine")
                cc = ct.query_radius(np.radians(grp[["lat", "lng"]].values), r=2.0 / R, count_only=True)
                for k, idx in enumerate(ci):
                    same_cuisine[idx] = max(cc[k] - 1, 0)

        density_pd = pd.DataFrame({
            "place_id":               rest_valid["place_id"].values,
            "density_500m":           _norm_arr(np.maximum(counts_500m - 1, 0).astype(float), log1p=True),
            "density_2km":            _norm_arr(np.maximum(counts_2km  - 1, 0).astype(float), log1p=True),
            "density_same_cuisine_2km": _norm_arr(same_cuisine, log1p=True),
        })
        result = result.join(pl.from_pandas(density_pd), on="place_id", how="left")
    for col in ["density_500m", "density_2km", "density_same_cuisine_2km"]:
        if col not in result.columns:
            result = result.with_columns(pl.lit(0.0).alias(col))
        else:
            result = result.with_columns(pl.col(col).fill_null(0.0))

    # ── Feature group C — Review trend + rating distribution ──────────────────
    _step("C review trend")
    reviews = reviews.with_columns(
        pl.col("rating").cast(pl.Float64, strict=False).alias("rating_num")
    )
    _photo_col_r = next(
        (c for c in ["attached_photos", "num_photos", "photo_count"] if c in reviews.columns), None
    )

    # Add row rank (0-based) within each place_id, sorted by timestamp_days_ago
    rev_sorted = reviews.sort(["place_id", "timestamp_days_ago"], nulls_last=True)
    rev_sorted = rev_sorted.with_columns(
        (pl.int_range(pl.len()).over("place_id")).alias("_rank")
    )

    trend_agg_exprs = [
        pl.col("rating_num").drop_nulls().head(20).mean().alias("_recent20_mean"),
        pl.col("rating_num").drop_nulls().mean().alias("_overall_mean"),
        pl.col("rating_num").drop_nulls().count().alias("_n"),
        pl.col("rating_num").drop_nulls().std().fill_null(0.0).alias("rating_bimodality"),
        (pl.col("rating_num").drop_nulls() == 5).cast(pl.Float64).mean().alias("pct_5star_restaurant"),
        # velocity_ratio: recent / older
        (pl.col("timestamp_days_ago") <= 180).sum().alias("_recent_cnt"),
        ((pl.col("timestamp_days_ago") > 180) & (pl.col("timestamp_days_ago") <= 360)).sum().alias("_older_cnt"),
        pl.col("has_content").fill_null(False).cast(pl.Float64).mean().alias("text_engagement"),
        pl.col("text_len").fill_null(0).cast(pl.Float64).mean().alias("avg_review_length"),
        # For polyfit slope: need sum_xy = sum(rank * rating), handled below
        (pl.col("_rank").cast(pl.Float64) * pl.col("rating_num")).drop_nulls().sum().alias("_sum_xy"),
        pl.col("_rank").cast(pl.Float64).drop_nulls().sum().alias("_sum_x"),
    ]
    if _photo_col_r:
        trend_agg_exprs.append(
            pl.col(_photo_col_r).fill_null(0).cast(pl.Float64).mean().alias("avg_photos_per_review")
        )

    trend_df = rev_sorted.group_by("place_id").agg(trend_agg_exprs)

    # recency_boost: mean of first 20 - overall mean (only if n >= 5)
    trend_df = trend_df.with_columns(
        pl.when(pl.col("_n") >= 5)
        .then(pl.col("_recent20_mean") - pl.col("_overall_mean"))
        .otherwise(0.0)
        .alias("recency_boost")
    )

    # velocity_ratio
    trend_df = trend_df.with_columns(
        (pl.col("_recent_cnt").cast(pl.Float64) / (pl.col("_older_cnt").cast(pl.Float64) + 1.0))
        .alias("velocity_ratio")
    )

    # rating_trend via analytical least-squares slope: x = rank (0..n-1)
    # slope = (n*sum_xy - sum_x*sum_y) / (n*n*(n-1)*(2n-1)/6 - (n*(n-1)/2)^2)
    trend_df = trend_df.with_columns([
        pl.col("_n").cast(pl.Float64).alias("_nf"),
        (pl.col("_n").cast(pl.Float64) * (pl.col("_n").cast(pl.Float64) - 1) / 2).alias("_sum_x_formula"),
    ]).with_columns(
        pl.when(pl.col("_n") >= 3)
        .then(
            (pl.col("_nf") * pl.col("_sum_xy") - pl.col("_sum_x") * (pl.col("_overall_mean") * pl.col("_nf"))) /
            ((pl.col("_nf") * pl.col("_nf") * (pl.col("_nf") - 1) * (2 * pl.col("_nf") - 1) / 6)
             - pl.col("_sum_x_formula") ** 2 + 1e-9)
        )
        .otherwise(0.0)
        .alias("rating_trend")
    )

    # Normalize
    norm_cols = ["recency_boost", "rating_trend", "rating_bimodality", "avg_review_length"]
    log1p_cols = ["velocity_ratio"]
    if _photo_col_r:
        log1p_cols.append("avg_photos_per_review")
    else:
        trend_df = trend_df.with_columns(pl.lit(0.0).alias("avg_photos_per_review"))

    trend_df = trend_df.with_columns([_norm(c) for c in norm_cols])
    trend_df = trend_df.with_columns([_norm(c, log1p=True) for c in log1p_cols])

    keep = ["place_id", "recency_boost", "rating_trend", "velocity_ratio", "text_engagement",
            "pct_5star_restaurant", "rating_bimodality", "avg_review_length", "avg_photos_per_review"]
    result = result.join(trend_df.select(keep), on="place_id", how="left")
    for col in keep[1:]:
        result = result.with_columns(pl.col(col).fill_null(0.0))

    # ── Legacy analytical contract ───────────────────────────────────────────
    # These six features were present in the previous publication files. Keep
    # their exact semantics while computing them from training reviews only.
    _step("C2 legacy analytical contract")
    result = result.with_columns(pl.col("rating_bimodality").alias("rating_std"))

    if _photo_col_r:
        photo_rate = reviews.group_by("place_id").agg(
            (pl.col(_photo_col_r).fill_null(0).cast(pl.Float64) > 0)
            .mean().alias("photo_rate")
        )
        result = result.join(photo_rate, on="place_id", how="left")
    else:
        result = result.with_columns(pl.lit(0.0).alias("photo_rate"))

    visits = reviews.group_by(["place_id", "contributor_id"]).agg(
        pl.len().alias("_visits")
    )
    repeat_rate = visits.group_by("place_id").agg(
        (pl.col("_visits") > 1).cast(pl.Float64).mean().alias("repeat_visitor_rate")
    )
    result = result.join(repeat_rate, on="place_id", how="left")

    if "is_local_guide" in reviews.columns:
        local_guides = reviews.group_by("place_id").agg(
            pl.col("is_local_guide").fill_null(False).cast(pl.Float64)
            .mean().alias("local_guide_pct")
        )
        result = result.join(local_guides, on="place_id", how="left")
    else:
        result = result.with_columns(pl.lit(0.0).alias("local_guide_pct"))

    age = reviews.group_by("place_id").agg(
        pl.col("timestamp_days_ago").cast(pl.Float64, strict=False)
        .max().alias("restaurant_age_norm")
    ).with_columns(_norm("restaurant_age_norm", log1p=True))
    result = result.join(age, on="place_id", how="left")

    if "predicted_race" in reviews.columns:
        race_gap = (
            reviews.filter(
                pl.col("predicted_race").is_not_null() &
                pl.col("rating_num").is_not_null()
            )
            .group_by(["place_id", "predicted_race"])
            .agg(pl.col("rating_num").mean().alias("_race_mean"))
            .group_by("place_id")
            .agg(pl.col("_race_mean").std().fill_null(0.0).alias("race_rating_gap"))
            .with_columns(_norm("race_rating_gap"))
        )
        result = result.join(race_gap, on="place_id", how="left")
    else:
        result = result.with_columns(pl.lit(0.0).alias("race_rating_gap"))

    for col in ["rating_std", "photo_rate", "repeat_visitor_rate",
                "local_guide_pct", "restaurant_age_norm", "race_rating_gap"]:
        result = result.with_columns(pl.col(col).fill_null(0.0))

    # ── Feature group D — CBG foot traffic (SafeGraph mobility) ──────────────
    if CBG_PATTERNS.exists() and "cbg" in restaurants.columns:
        _step("D CBG foot traffic")
        import json as _json

        cbg_raw = pl.read_csv(CBG_PATTERNS, columns=[
            "census_block_group", "raw_visit_count", "raw_visitor_count",
            "distance_from_home", "popularity_by_hour",
            "popularity_by_day", "visitor_home_cbgs",
        ]).with_columns(
            pl.col("census_block_group").cast(pl.Utf8).str.zfill(12).alias("cbg")
        )

        def _parse_hour(s):
            try:
                a = np.array(_json.loads(s), dtype=float); t = a.sum()
                if t == 0: return 0.0, 0.0, 0.0
                return float(a[11:15].sum()/t), float(a[17:23].sum()/t), float(a[22:].sum()/t)
            except: return 0.0, 0.0, 0.0

        def _parse_weekend(s):
            try:
                d = _json.loads(s); t = sum(d.values())
                return float((d.get("Saturday",0)+d.get("Sunday",0))/t) if t>0 else 0.0
            except: return 0.0

        def _parse_origin_div(s):
            try: return float(len(_json.loads(s)))
            except: return 0.0

        cbg_pd = cbg_raw.to_pandas()
        hr = cbg_pd["popularity_by_hour"].fillna("[]").apply(_parse_hour)
        cbg_pd["cbg_lunch_ratio"]     = [x[0] for x in hr]
        cbg_pd["cbg_evening_ratio"]   = [x[1] for x in hr]
        cbg_pd["cbg_latenight_ratio"] = [x[2] for x in hr]
        cbg_pd["cbg_weekend_ratio"]   = cbg_pd["popularity_by_day"].fillna("{}").apply(_parse_weekend)
        cbg_pd["cbg_origin_diversity"]= cbg_pd["visitor_home_cbgs"].fillna("{}").apply(_parse_origin_div)
        cbg_pd["cbg_log_visits"]      = np.log1p(cbg_pd["raw_visit_count"].fillna(0))
        cbg_pd["cbg_distance_from_home"] = cbg_pd["distance_from_home"].fillna(0)

        cbg_feat_cols = [
            "cbg_log_visits", "cbg_distance_from_home", "cbg_weekend_ratio",
            "cbg_lunch_ratio", "cbg_evening_ratio", "cbg_latenight_ratio",
            "cbg_origin_diversity",
        ]
        for col in cbg_feat_cols:
            arr = cbg_pd[col].values.astype(float)
            mn, mx = arr.min(), arr.max()
            cbg_pd[col] = (arr - mn) / (mx - mn) if mx > mn else np.full_like(arr, 0.5)

        cbg_pl = pl.from_pandas(cbg_pd[["cbg"] + cbg_feat_cols]).unique(subset=["cbg"])
        result = result.join(
            restaurants.select(["place_id", "cbg"]).join(cbg_pl, on="cbg", how="left").select(["place_id"] + cbg_feat_cols),
            on="place_id", how="left",
        )
        for col in cbg_feat_cols:
            result = result.with_columns(pl.col(col).fill_null(0.5))
        log.info(f"CBG features joined: {len(cbg_feat_cols)} features")

    # ── Combine ────────────────────────────────────────────────────────────────
    result = result.fill_null(0.0)
    n_feat = len([c for c in result.columns if c != "place_id"])
    log.info(f"Item extended features: {len(result):,} restaurants × {n_feat} features")
    return result


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-dish-profiles",
        action="store_true",
        help="Exclude existing LLM-derived dish-profile features",
    )
    args = parser.parse_args()
    import time as _t
    log.info("Loading data…")
    t0 = _t.time()

    log.info("  Step 1: pq.read_table …")
    import pyarrow.parquet as pq
    table = pq.read_table(REVIEWS_FLAT, columns=_REVIEWS_COLS)
    log.info(f"  Step 1 done in {_t.time()-t0:.1f}s — {len(table):,} rows")

    t1 = _t.time()
    log.info("  Step 2: to Polars …")
    reviews = pl.from_arrow(table)
    log.info(f"  Step 2 done in {_t.time()-t1:.1f}s — mem≈{reviews.estimated_size()/1e9:.2f} GB")

    t2 = _t.time()
    log.info("  Step 3: read restaurants …")
    restaurants = pl.read_parquet(RESTAURANTS_ENR)
    log.info(f"  Step 3 done in {_t.time()-t2:.1f}s")

    t3 = _t.time()
    log.info("  Step 4: read dishes/profiles …")
    dishes = pl.read_parquet(DISHES_FILE) if DISHES_FILE.exists() else None
    profiles = (
        pl.read_parquet(PROFILES_FILE)
        if PROFILES_FILE.exists() and not args.no_dish_profiles
        else None
    )
    log.info(f"  Step 4 done in {_t.time()-t3:.1f}s")
    log.info(f"Loading complete in {_t.time()-t0:.1f}s total")

    user_ext = build_user_extended(reviews, restaurants)
    user_ext.write_parquet(USER_EXT_OUT)
    log.info(f"Saved → {USER_EXT_OUT}")

    item_ext = build_item_extended(reviews, restaurants, dishes, profiles)
    item_ext.write_parquet(ITEM_EXT_OUT)
    log.info(f"Saved → {ITEM_EXT_OUT}")


if __name__ == "__main__":
    main()
