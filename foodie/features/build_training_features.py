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
    python -m foodie.features.build_training_features
    python -m foodie.features.build_training_features --output-dir data/ --llm-feat-mode full
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_feat")

REVIEWS_FLAT       = Path(os.environ.get("FOODIE_REVIEWS_FLAT", "data/reviews_flat.parquet"))
RESTAURANTS_ENR    = Path(os.environ.get("FOODIE_RESTAURANTS_ENR", "data/restaurants_enriched.parquet"))
USER_EXT_FEATURES  = Path(os.environ.get("FOODIE_USER_EXT_FEATURES", "data/user_extended_features.parquet"))
USER_LLM_FEATURES  = Path(os.environ.get(
    "FOODIE_USER_LLM_FEATURES", "data/user_llm_features.parquet"
))
USER_LLM_EMBEDDING = Path(os.environ.get("FOODIE_USER_LLM_EMBEDDINGS", "data/user_llm_embeddings.parquet"))
ITEM_EXT_FEATURES  = Path(os.environ.get("FOODIE_ITEM_EXT_FEATURES", "data/item_extended_features.parquet"))
REVIEW_DISHES      = Path(os.environ.get("FOODIE_REVIEW_DISHES", "data/review_dishes.parquet"))
DISH_PROFILES      = Path(os.environ.get("FOODIE_DISH_PROFILES", "data/dish_profiles.parquet"))
RESTAURANT_LLM_FEAT = Path(os.environ.get(
    "FOODIE_RESTAURANT_LLM_FEATURES", "data/restaurant_llm_features.parquet"
))
ITEM_LLM_EMBEDDING = Path(os.environ.get("FOODIE_RESTAURANT_LLM_EMBEDDINGS", "data/restaurant_llm_embeddings.parquet"))


def _numeric_feature_columns(df: pd.DataFrame, id_col: str) -> list[str]:
    """Return only model-ready numeric columns, excluding provenance metadata."""
    provenance = {"source_review_count"}
    return [
        c for c in df.columns
        if c != id_col and c not in provenance and pd.api.types.is_numeric_dtype(df[c])
    ]


def _fill_llm_missing(df: pd.DataFrame, columns: list[str]) -> None:
    """Keep unknownness distinct from a neutral 0.5 semantic score."""
    zero_markers = ("_known", "_confidence", "source_review_count", "cuisine", "meal_")
    for col in columns:
        default = 0.0 if any(marker in col for marker in zero_markers) else 0.5
        df[col] = df[col].fillna(default)


def _build_user_feature_df(
    df: pd.DataFrame,
    llm_feat_mode: str = "fast",
    include_llm_derived: bool = True,
    user_extended: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
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

    if user_extended is not None or USER_EXT_FEATURES.exists():
        ext = user_extended if user_extended is not None else pd.read_parquet(USER_EXT_FEATURES)
        ext_cols = [c for c in ext.columns if c != "contributor_id"]
        feat_df = feat_df.merge(ext, on="contributor_id", how="left")
        feat_df[ext_cols] = feat_df[ext_cols].fillna(0.0)
        groups["extended"] = ext_cols
        log.info(f"  User extended features: +{len(ext_cols)} cols")
    else:
        groups["extended"] = []

    if include_llm_derived and llm_feat_mode != "none" and USER_LLM_FEATURES.exists():
        user_llm = pd.read_parquet(USER_LLM_FEATURES)
        llm_cols = _numeric_feature_columns(user_llm, "contributor_id")
        feat_df = feat_df.merge(
            user_llm[["contributor_id"] + llm_cols], on="contributor_id", how="left"
        )
        _fill_llm_missing(feat_df, llm_cols)
        groups["llm"] = llm_cols
        coverage = feat_df["contributor_id"].isin(user_llm["contributor_id"]).mean()
        log.info(f"  User LLM features: +{len(llm_cols)} cols (coverage={coverage:.1%})")
    else:
        groups["llm"] = []

    if (
        include_llm_derived and llm_feat_mode == "full"
        and USER_LLM_EMBEDDING.exists()
    ):
        emb = pd.read_parquet(USER_LLM_EMBEDDING)
        emb_cols = [c for c in emb.columns if c.startswith("llm_emb_")]
        feat_df = feat_df.merge(
            emb[["contributor_id"] + emb_cols], on="contributor_id", how="left"
        )
        feat_df[emb_cols] = feat_df[emb_cols].fillna(0.0)
        groups["llm_embedding"] = emb_cols
        log.info(f"  User LLM embeddings: +{len(emb_cols)} cols")
    else:
        groups["llm_embedding"] = []

    all_feat_cols = [c for c in feat_df.columns if c != "contributor_id"]
    feat_df[all_feat_cols] = feat_df[all_feat_cols].astype(np.float32)
    total = sum(len(v) for v in groups.values())
    log.info(f"  User total: {total} feature columns across {len(feat_df):,} contributors")
    return feat_df, groups


def _build_item_feature_df(
    llm_feat_mode: str = "fast",
    use_nlp_feat: bool = True,
    include_llm_derived: bool = True,
    reviews: pd.DataFrame | None = None,
    item_extended: pd.DataFrame | None = None,
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
    if reviews is not None or REVIEWS_FLAT.exists():
        review_columns = ["place_id", "food_score", "service_score", "atmosphere_score",
                          "meal_type"]
        rev = (reviews[review_columns].copy() if reviews is not None
               else pd.read_parquet(REVIEWS_FLAT, columns=review_columns))
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

    if item_extended is not None or ITEM_EXT_FEATURES.exists():
        ext = item_extended if item_extended is not None else pd.read_parquet(ITEM_EXT_FEATURES)
        ext_cols = [c for c in ext.columns if c != "place_id"]
        dish_cols = [c for c in ext_cols if c in {
            "dish_count", "dish_cat_entropy", "dish_flavor_div", "dish_cuisine_breadth"
        }]
        conventional_ext_cols = [c for c in ext_cols if c not in dish_cols]
        feat_df = feat_df.merge(ext, on="place_id", how="left")
        feat_df[ext_cols] = feat_df[ext_cols].fillna(0.0)
        groups["extended"] = conventional_ext_cols
        groups["dish_llm"] = dish_cols
        log.info(f"  Item extended features: +{len(conventional_ext_cols)} conventional, "
                 f"+{len(dish_cols)} LLM-derived dish cols")
    else:
        groups["extended"] = []
        groups["dish_llm"] = []

    llm_cols: list[str] = []
    if (
        include_llm_derived
        and RESTAURANT_LLM_FEAT.exists()
        and llm_feat_mode != "none"
    ):
        llm_ext = pd.read_parquet(RESTAURANT_LLM_FEAT)
        llm_cols = _numeric_feature_columns(llm_ext, "place_id")
        feat_df = feat_df.merge(llm_ext[["place_id"] + llm_cols], on="place_id", how="left")
        _fill_llm_missing(feat_df, llm_cols)
        groups["llm"] = llm_cols
        log.info(f"  Item LLM features ({llm_feat_mode}): +{len(llm_cols)} cols")
    else:
        groups["llm"] = []

    if (
        include_llm_derived and llm_feat_mode == "full"
        and ITEM_LLM_EMBEDDING.exists()
    ):
        emb = pd.read_parquet(ITEM_LLM_EMBEDDING)
        emb_cols = [c for c in emb.columns if c.startswith("llm_emb_")]
        feat_df = feat_df.merge(emb[["place_id"] + emb_cols], on="place_id", how="left")
        feat_df[emb_cols] = feat_df[emb_cols].fillna(0.0)
        groups["llm_embedding"] = emb_cols
        log.info(f"  Item LLM embeddings: +{len(emb_cols)} cols")
    else:
        groups["llm_embedding"] = []

    all_feat_cols = [c for c in feat_df.columns if c != "place_id"]
    feat_df[all_feat_cols] = feat_df[all_feat_cols].astype(np.float32)
    total = sum(len(v) for v in groups.values())
    log.info(f"  Item total: {total} feature columns across {len(feat_df):,} restaurants")
    return feat_df, groups


def build(
    output_dir: Path = Path("data"),
    llm_feat_mode: str = "fast",
    use_nlp_feat: bool = True,
    include_llm_derived: bool = True,
    publication_split: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading reviews_flat …")
    df = pd.read_parquet(REVIEWS_FLAT)
    df = df[df["contributor_id"].notna() & (df["contributor_id"] != "") &
            df["place_id"].notna()].copy()
    user_extended = item_extended = None
    if publication_split:
        # Behavioral features must not see the held-out validation/test pairs.
        # Keep every raw review belonging to a training user-item pair, so repeat
        # reviews of an observed restaurant remain legitimate training evidence.
        from recommendation_gnn import load_interaction_splits
        from build_extended_features import build_user_extended, build_item_extended
        import polars as pl

        train_df, _, _, _, _ = load_interaction_splits()
        train_pairs = train_df[["contributor_id", "place_id"]].drop_duplicates()
        df = df.merge(train_pairs.assign(_publication_train=1),
                      on=["contributor_id", "place_id"], how="inner")
        df = df.drop(columns="_publication_train")
        restaurants_pl = pl.read_parquet(RESTAURANTS_ENR)
        reviews_pl = pl.from_pandas(df)
        user_extended = build_user_extended(reviews_pl, restaurants_pl).to_pandas()
        # Restrict restaurant-dish links to training review IDs. Flavor profiles
        # are static descriptors joined only for dishes observed in those links.
        train_dishes = None
        dish_profiles = None
        if REVIEW_DISHES.exists():
            review_dishes = pd.read_parquet(REVIEW_DISHES)
            train_review_ids = df[["review_id"]].drop_duplicates()
            train_dishes_pd = (
                review_dishes.merge(train_review_ids, on="review_id", how="inner")
                [["dish_name", "place_id"]].drop_duplicates()
            )
            train_dishes = pl.from_pandas(train_dishes_pd)
            dishes_out = output_dir / "dishes_train.parquet"
            train_dishes_pd.to_parquet(dishes_out, index=False)
            log.info("Publication split: %s training-only restaurant-dish links", f"{len(train_dishes_pd):,}")
        if DISH_PROFILES.exists():
            dish_profiles = pl.read_parquet(DISH_PROFILES)
        item_extended = build_item_extended(
            reviews_pl, restaurants_pl, dishes=train_dishes, profiles=dish_profiles
        ).to_pandas()
        log.info("Publication split: behavioral features use %s training-review rows", f"{len(df):,}")
    log.info(f"  {len(df):,} rows, {df['contributor_id'].nunique():,} contributors, "
             f"{df['place_id'].nunique():,} restaurants")

    log.info("Building user features …")
    user_df, user_groups = _build_user_feature_df(
        df, llm_feat_mode=llm_feat_mode, include_llm_derived=include_llm_derived,
        user_extended=user_extended,
    )

    log.info("Building item features …")
    item_df, item_groups = _build_item_feature_df(
        llm_feat_mode=llm_feat_mode,
        use_nlp_feat=use_nlp_feat,
        include_llm_derived=include_llm_derived,
        reviews=df,
        item_extended=item_extended,
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
    parser.add_argument(
        "--non-llm-only",
        action="store_true",
        help=(
            "Exclude all LLM-derived item NLP/semantic/dietary features and "
            "user preference/dietary features"
        ),
    )
    parser.add_argument(
        "--publication-split", action="store_true",
        help="Build leakage-safe behavioral aggregates from training interactions only",
    )
    args = parser.parse_args()

    build(
        output_dir=Path(args.output_dir),
        llm_feat_mode="none" if args.non_llm_only else args.llm_feat_mode,
        use_nlp_feat=not (args.no_nlp or args.non_llm_only),
        include_llm_derived=not args.non_llm_only,
        publication_split=args.publication_split,
    )


if __name__ == "__main__":
    main()
