#!/usr/bin/env python3
"""
build_training_features.py — Pre-compute all user + item feature matrices once.

Training scripts load the cached files instead of re-joining on every run.
Feature groups are tagged in feature_groups.json so columns can be dropped
at training time via --skip-feature-groups.

Outputs
-------
  data/user_features_prebuilt.parquet   contributor_id + all user feature cols
  data/item_features_prebuilt.parquet   place_id + all item feature cols
  data/feature_groups.json              {user: {group: [cols]}, item: {group: [cols]}}

Usage
-----
    python build_training_features.py
    python build_training_features.py --output-dir data/ --llm-feat-mode full
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_feat")

REVIEWS_FLAT          = Path("data/reviews_flat.parquet")
RESTAURANTS_ENR       = Path("data/restaurants_enriched.parquet")
NLP_FEATURES          = Path("data/restaurant_nlp_features.parquet")
USER_EXT_FEATURES     = Path("data/user_extended_features.parquet")
USER_PREF_FEATURES    = Path("data/user_preference_features.parquet")
USER_DIETARY_FEATURES = Path("data/user_dietary_features.parquet")
ITEM_EXT_FEATURES     = Path("data/item_extended_features.parquet")
RESTAURANT_LLM_FEAT   = Path("data/restaurant_llm_features.parquet")

ITEM_DIETARY_COLS = [
    "llm_vegan_options", "llm_vegetarian_options", "llm_gluten_free_options",
    "llm_halal", "llm_kosher",
]


def _build_user_feature_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Build user features indexed by contributor_id.
    Returns (feature_df, group_map) where group_map = {group_name: [col_list]}.
    """
    cols = ["contributor_id", "predicted_race", "predicted_gender",
            "is_local_guide", "reviewer_reviews"]
    user_meta = (
        df.sort_values("timestamp_days_ago", ascending=True, na_position="last")
        .drop_duplicates("contributor_id")[cols].copy()
    )
    user_meta["predicted_race"]   = user_meta["predicted_race"].fillna("unknown").astype(str)
    user_meta["predicted_gender"] = user_meta["predicted_gender"].fillna("unknown").astype(str)
    user_meta["is_local_guide"]   = user_meta["is_local_guide"].fillna(False).astype(float)
    user_meta["log_reviews"] = np.log1p(
        user_meta["reviewer_reviews"].fillna(0).clip(lower=0).astype(float)
    )
    mx = user_meta["log_reviews"].max()
    if mx > 0:
        user_meta["log_reviews"] /= mx

    loc_cols: list[str] = []
    if RESTAURANTS_ENR.exists():
        rest_loc = pd.read_parquet(RESTAURANTS_ENR, columns=["place_id", "lat", "lng"])
        rest_loc["lat"] = pd.to_numeric(rest_loc["lat"], errors="coerce")
        rest_loc["lng"] = pd.to_numeric(rest_loc["lng"], errors="coerce")
        user_loc = (
            df[["contributor_id", "place_id"]]
            .merge(rest_loc, on="place_id", how="left")
            .groupby("contributor_id")[["lat", "lng"]].median()
        )
        lat_min, lat_max = user_loc["lat"].min(), user_loc["lat"].max()
        lng_min, lng_max = user_loc["lng"].min(), user_loc["lng"].max()
        user_loc["lat_norm"] = (user_loc["lat"] - lat_min) / (lat_max - lat_min + 1e-8)
        user_loc["lng_norm"] = (user_loc["lng"] - lng_min) / (lng_max - lng_min + 1e-8)
        user_meta = user_meta.merge(
            user_loc[["lat_norm", "lng_norm"]], left_on="contributor_id",
            right_index=True, how="left"
        )
        user_meta[["lat_norm", "lng_norm"]] = user_meta[["lat_norm", "lng_norm"]].fillna(0.5)
        loc_cols = ["lat_norm", "lng_norm"]

    race_d   = pd.get_dummies(user_meta["predicted_race"],   prefix="race",   dtype=float)
    gender_d = pd.get_dummies(user_meta["predicted_gender"], prefix="gender", dtype=float)
    feat_df  = pd.concat(
        [user_meta[["contributor_id", "is_local_guide", "log_reviews"] + loc_cols],
         race_d, gender_d],
        axis=1,
    )
    base_cols = [c for c in feat_df.columns if c != "contributor_id"]
    log.info(f"  User base features: {len(base_cols)} cols "
             f"(race={len(race_d.columns)}, gender={len(gender_d.columns)}, "
             f"loc={len(loc_cols)}, other=2)")

    groups: dict[str, list[str]] = {"base": base_cols}

    if USER_EXT_FEATURES.exists():
        ext = pd.read_parquet(USER_EXT_FEATURES)
        ext_cols = [c for c in ext.columns if c != "contributor_id"]
        feat_df = feat_df.merge(ext, on="contributor_id", how="left")
        feat_df[ext_cols] = feat_df[ext_cols].fillna(0.0)
        groups["extended"] = ext_cols
        log.info(f"  User extended features: +{len(ext_cols)} cols")
    else:
        groups["extended"] = []

    if USER_PREF_FEATURES.exists():
        pref = pd.read_parquet(USER_PREF_FEATURES)
        pref_cols = [c for c in pref.columns if c != "contributor_id"]
        feat_df = feat_df.merge(pref, on="contributor_id", how="left")
        feat_df[pref_cols] = feat_df[pref_cols].fillna(0.5)
        groups["pref"] = pref_cols
        coverage = feat_df["contributor_id"].isin(pref["contributor_id"]).mean()
        log.info(f"  User pref features: +{len(pref_cols)} cols (coverage={coverage:.1%})")
    else:
        groups["pref"] = []

    if USER_DIETARY_FEATURES.exists():
        dietary = pd.read_parquet(USER_DIETARY_FEATURES)
        dietary_cols = [c for c in dietary.columns if c != "contributor_id"]
        feat_df = feat_df.merge(dietary, on="contributor_id", how="left")
        feat_df[dietary_cols] = feat_df[dietary_cols].fillna(0.0)
        groups["dietary"] = dietary_cols
        coverage = feat_df["contributor_id"].isin(dietary["contributor_id"]).mean()
        log.info(f"  User dietary+protein features: +{len(dietary_cols)} cols (coverage={coverage:.1%})")
    else:
        groups["dietary"] = []

    all_feat_cols = [c for c in feat_df.columns if c != "contributor_id"]
    feat_df[all_feat_cols] = feat_df[all_feat_cols].astype(np.float32)
    total = sum(len(v) for v in groups.values())
    log.info(f"  User total: {total} feature columns across {len(feat_df):,} contributors")
    return feat_df, groups


def _build_item_feature_df(
    llm_feat_mode: str = "fast",
    use_nlp_feat: bool = True,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Build item features indexed by place_id.
    Returns (feature_df, group_map).
    """
    if not RESTAURANTS_ENR.exists():
        raise FileNotFoundError(f"{RESTAURANTS_ENR} not found — run build_graph_data.py first")

    _price_map = {
        "PRICE_LEVEL_INEXPENSIVE":    1.0,
        "PRICE_LEVEL_MODERATE":       2.0,
        "PRICE_LEVEL_EXPENSIVE":      3.0,
        "PRICE_LEVEL_VERY_EXPENSIVE": 4.0,
    }

    rest = pd.read_parquet(RESTAURANTS_ENR)
    rest["cuisine_category"] = rest["cuisine_category"].fillna("Other").astype(str)
    rest["price_num"]  = rest["price_level"].map(_price_map).fillna(2.0)
    rest["rating"]     = pd.to_numeric(rest["rating"], errors="coerce")
    rest["rating"]     = rest["rating"].fillna(rest["rating"].median())
    rest["rating_count"] = pd.to_numeric(rest["user_rating_count"], errors="coerce").fillna(0)
    rest["lat"] = pd.to_numeric(rest["lat"], errors="coerce")
    rest["lng"] = pd.to_numeric(rest["lng"], errors="coerce")
    rest["lat_norm"] = (rest["lat"] - rest["lat"].min()) / (rest["lat"].max() - rest["lat"].min() + 1e-8)
    rest["lng_norm"] = (rest["lng"] - rest["lng"].min()) / (rest["lng"].max() - rest["lng"].min() + 1e-8)
    rest[["lat_norm", "lng_norm"]] = rest[["lat_norm", "lng_norm"]].fillna(0.5)
    rest["price_norm"]       = (rest["price_num"].clip(1, 4) - 1) / 3.0
    rest["rating_norm"]      = (rest["rating"].clip(1, 5) - 1) / 4.0
    rest["log_rating_count"] = np.log1p(rest["rating_count"])
    mx = rest["log_rating_count"].max()
    if mx > 0:
        rest["log_rating_count"] /= mx

    review_cols: list[str] = []
    if REVIEWS_FLAT.exists():
        rev = pd.read_parquet(REVIEWS_FLAT, columns=["place_id", "food_score",
                                                      "service_score", "atmosphere_score",
                                                      "meal_type"])
        for col in ["food_score", "service_score", "atmosphere_score"]:
            rev[col] = pd.to_numeric(
                rev[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
            )
        score_agg = (
            rev.groupby("place_id")[["food_score", "service_score", "atmosphere_score"]]
            .mean()
            .rename(columns={"food_score": "food_score_mean",
                             "service_score": "service_score_mean",
                             "atmosphere_score": "atmosphere_score_mean"})
        )
        for col in ["food_score_mean", "service_score_mean", "atmosphere_score_mean"]:
            score_agg[col] = (score_agg[col].clip(1, 5) - 1) / 4.0

        _meal_clean = rev["meal_type"].astype(str).str.extract(
            r"(Dinner|Lunch|Brunch|Breakfast)", expand=False
        )
        meal_counts = (
            rev.assign(meal_clean=_meal_clean)
            .groupby("place_id")["meal_clean"]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
            .rename(columns=lambda c: f"meal_pct_{c.lower()}")
        )
        for col in ["meal_pct_dinner", "meal_pct_lunch", "meal_pct_brunch", "meal_pct_breakfast"]:
            if col not in meal_counts.columns:
                meal_counts[col] = 0.0

        rest = (rest.set_index("place_id")
                    .join(score_agg, how="left")
                    .join(meal_counts[["meal_pct_dinner", "meal_pct_lunch",
                                       "meal_pct_brunch", "meal_pct_breakfast"]], how="left")
                    .reset_index())
        for col in ["food_score_mean", "service_score_mean", "atmosphere_score_mean",
                    "meal_pct_dinner", "meal_pct_lunch", "meal_pct_brunch", "meal_pct_breakfast"]:
            rest[col] = rest[col].fillna(0.0)
        review_cols = ["food_score_mean", "service_score_mean", "atmosphere_score_mean",
                       "meal_pct_dinner", "meal_pct_lunch", "meal_pct_brunch", "meal_pct_breakfast"]

    cuisine_d  = pd.get_dummies(rest["cuisine_category"], prefix="cuisine", dtype=float)
    scalar_cols = (["price_norm", "rating_norm", "log_rating_count", "lat_norm", "lng_norm"]
                   + review_cols)
    base_cols  = scalar_cols + list(cuisine_d.columns)

    feat_df = pd.concat([rest[["place_id"] + scalar_cols], cuisine_d], axis=1)
    log.info(f"  Item base features: {len(base_cols)} cols "
             f"(scalar={len(scalar_cols)}, cuisine={len(cuisine_d.columns)})")

    groups: dict[str, list[str]] = {"base": base_cols}

    nlp_cols: list[str] = []
    if use_nlp_feat and NLP_FEATURES.exists():
        try:
            from build_nlp_features import ATTRIBUTES as NLP_ATTRS
            nlp = pd.read_parquet(NLP_FEATURES)
            feat_df = feat_df.merge(nlp, on="place_id", how="left")
            for col in NLP_ATTRS:
                feat_df[col] = feat_df[col].fillna(0.5)
            nlp_cols = NLP_ATTRS
            groups["nlp"] = nlp_cols
            log.info(f"  Item NLP features: +{len(nlp_cols)} cols for "
                     f"{nlp['place_id'].nunique():,} restaurants")
        except ImportError:
            log.warning("  build_nlp_features not importable — NLP features skipped")
            groups["nlp"] = []
    else:
        groups["nlp"] = []

    if ITEM_EXT_FEATURES.exists():
        ext = pd.read_parquet(ITEM_EXT_FEATURES)
        ext_cols = [c for c in ext.columns if c != "place_id"]
        feat_df = feat_df.merge(ext, on="place_id", how="left")
        feat_df[ext_cols] = feat_df[ext_cols].fillna(0.0)
        groups["extended"] = ext_cols
        log.info(f"  Item extended features: +{len(ext_cols)} cols")
    else:
        groups["extended"] = []

    llm_cols: list[str] = []
    if RESTAURANT_LLM_FEAT.exists() and llm_feat_mode != "none":
        llm_ext = pd.read_parquet(RESTAURANT_LLM_FEAT)
        if llm_feat_mode == "fast":
            llm_cols = [c for c in llm_ext.columns if c != "place_id" and not c.startswith("llm_")]
        else:
            llm_cols = [c for c in llm_ext.columns if c != "place_id"
                        and c not in ITEM_DIETARY_COLS]
        feat_df = feat_df.merge(llm_ext[["place_id"] + llm_cols], on="place_id", how="left")
        feat_df[llm_cols] = feat_df[llm_cols].fillna(0.0)
        groups["llm"] = llm_cols
        log.info(f"  Item LLM features ({llm_feat_mode}): +{len(llm_cols)} cols")
    else:
        groups["llm"] = []

    # Restaurant dietary flags — always loaded from LLM features regardless of llm_feat_mode
    item_dietary_cols: list[str] = []
    if RESTAURANT_LLM_FEAT.exists():
        llm_df = pd.read_parquet(RESTAURANT_LLM_FEAT, columns=["place_id"] + ITEM_DIETARY_COLS)
        # Avoid double-join if already loaded in full mode above
        already = [c for c in ITEM_DIETARY_COLS if c in feat_df.columns]
        to_load = [c for c in ITEM_DIETARY_COLS if c not in feat_df.columns]
        if to_load:
            feat_df = feat_df.merge(llm_df[["place_id"] + to_load], on="place_id", how="left")
        item_dietary_cols = ITEM_DIETARY_COLS
        feat_df[item_dietary_cols] = feat_df[item_dietary_cols].fillna(0.0)
        groups["dietary"] = item_dietary_cols
        log.info(f"  Item dietary features: +{len(to_load)} new cols "
                 f"({len(already)} already present from llm group)")
    else:
        groups["dietary"] = []

    all_feat_cols = [c for c in feat_df.columns if c != "place_id"]
    feat_df[all_feat_cols] = feat_df[all_feat_cols].astype(np.float32)
    total = sum(len(v) for v in groups.values())
    log.info(f"  Item total: {total} feature columns across {len(feat_df):,} restaurants")
    return feat_df, groups


def build(output_dir: Path = Path("data"), llm_feat_mode: str = "fast",
          use_nlp_feat: bool = True):
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading reviews_flat …")
    df = pd.read_parquet(REVIEWS_FLAT)
    df = df[df["contributor_id"].notna() & (df["contributor_id"] != "") &
            df["place_id"].notna()].copy()
    log.info(f"  {len(df):,} rows, {df['contributor_id'].nunique():,} contributors, "
             f"{df['place_id'].nunique():,} restaurants")

    log.info("Building user features …")
    user_df, user_groups = _build_user_feature_df(df)

    log.info("Building item features …")
    item_df, item_groups = _build_item_feature_df(
        llm_feat_mode=llm_feat_mode, use_nlp_feat=use_nlp_feat
    )

    user_out = output_dir / "user_features_prebuilt.parquet"
    item_out  = output_dir / "item_features_prebuilt.parquet"
    groups_out = output_dir / "feature_groups.json"

    log.info(f"Saving {user_out} …")
    user_df.to_parquet(user_out, index=False)

    log.info(f"Saving {item_out} …")
    item_df.to_parquet(item_out, index=False)

    feature_groups = {"user": user_groups, "item": item_groups}
    log.info(f"Saving {groups_out} …")
    with open(groups_out, "w") as f:
        json.dump(feature_groups, f, indent=2)

    # Summary
    u_cols = sum(len(v) for v in user_groups.values())
    i_cols = sum(len(v) for v in item_groups.values())
    log.info("Done.")
    log.info(f"  User features : {u_cols:3d} cols  ({user_out})")
    log.info(f"  Item features : {i_cols:3d} cols  ({item_out})")
    log.info(f"  Feature groups: {groups_out}")
    log.info("")
    log.info("User groups:")
    for g, cols in user_groups.items():
        log.info(f"  {g:12s}: {len(cols):3d} cols")
    log.info("Item groups:")
    for g, cols in item_groups.items():
        log.info(f"  {g:12s}: {len(cols):3d} cols")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/",
                        help="Directory to write prebuilt files (default: data/)")
    parser.add_argument("--llm-feat-mode", default="fast",
                        choices=["none", "fast", "full"],
                        help="LLM feature columns to include (default: fast)")
    parser.add_argument("--no-nlp", action="store_true",
                        help="Skip NLP feature join")
    args = parser.parse_args()

    build(
        output_dir=Path(args.output_dir),
        llm_feat_mode=args.llm_feat_mode,
        use_nlp_feat=not args.no_nlp,
    )


if __name__ == "__main__":
    main()
