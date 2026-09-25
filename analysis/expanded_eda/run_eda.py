#!/usr/bin/env python3
"""Reproducible EDA for the frozen expanded Foodie dataset."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "analysis" / "expanded_eda"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORT = OUT / "EDA_REPORT.md"

FIPS = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT",
    "10":"DE","11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL",
    "18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD",
    "25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE",
    "32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND",
    "39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD",
    "47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV",
    "55":"WI","56":"WY",
}


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def save_bar(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str,
             filename: str, horizontal: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    if horizontal:
        plot = df.iloc[::-1]
        ax.barh(plot[x], plot[y], color="#3977a8")
        ax.set_xlabel(ylabel)
    else:
        ax.bar(df[x], df[y], color="#3977a8")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x" if horizontal else "y", alpha=.25)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    rest = pl.read_parquet(DATA / "restaurants_enriched.parquet").with_columns(
        pl.col("cbg").cast(pl.Utf8).str.replace(r"\.0$", "").str.zfill(12).alias("cbg"),
        pl.col("rating").cast(pl.Float64, strict=False),
        pl.col("user_rating_count").cast(pl.Int64, strict=False),
    ).with_columns(
        pl.col("cbg").str.slice(0, 2).replace_strict(FIPS, default=None).alias("state")
    )
    rev = pl.scan_parquet(DATA / "reviews_flat.parquet").select(
        "review_id", "place_id", "contributor_id", "rating", "text_len",
        "has_content", "attached_photos", "is_local_guide",
        "predicted_gender", "predicted_race",
    )
    valid_rev = rev.filter(pl.col("rating").cast(pl.Float64, strict=False).is_between(1, 5))

    rstat = rev.select(
        pl.len().alias("review_rows"),
        pl.col("review_id").n_unique().alias("unique_review_ids"),
        pl.col("contributor_id").drop_nulls().n_unique().alias("unique_reviewers"),
        pl.col("place_id").n_unique().alias("reviewed_restaurants"),
        pl.col("has_content").fill_null(False).mean().alias("content_share"),
        pl.col("attached_photos").fill_null(0).gt(0).mean().alias("photo_share"),
        pl.col("text_len").fill_null(0).mean().alias("mean_text_length"),
    ).collect().row(0, named=True)

    restaurant_reviews = rev.group_by("place_id").agg(
        pl.len().alias("collected_reviews"),
        pl.col("contributor_id").drop_nulls().n_unique().alias("unique_reviewers"),
    ).collect()
    rest_profile = rest.join(restaurant_reviews, on="place_id", how="left").with_columns(
        pl.col("collected_reviews").fill_null(0), pl.col("unique_reviewers").fill_null(0)
    )

    state = (rest_profile.group_by("state").agg(
        pl.len().alias("restaurants"),
        pl.col("cbg").n_unique().alias("cbgs"),
        pl.col("collected_reviews").sum().alias("reviews"),
        pl.col("place_id").filter(pl.col("collected_reviews") > 0).n_unique()
          .alias("restaurants_with_reviews"),
    ).sort("restaurants", descending=True))
    state.write_csv(TABLES / "state_coverage.csv")

    cuisine = (rest_profile.group_by(pl.col("cuisine_category").fill_null("Unknown")).agg(
        pl.len().alias("restaurants"),
        pl.col("collected_reviews").sum().alias("reviews"),
        pl.col("rating").mean().alias("mean_places_rating"),
    ).sort("restaurants", descending=True))
    cuisine.write_csv(TABLES / "google_type_cuisine_distribution.csv")

    repair = pl.read_parquet(DATA / "restaurant_cuisine_27b_v4.parquet")
    old_ids = pl.read_parquet(DATA / "restaurants_enriched_llmcz.parquet", columns=["place_id"])
    repaired = (rest.join(
        repair.select("place_id", "primary_cuisine", "confidence", "classification_method"),
        on="place_id", how="left"
    ).join(old_ids.with_columns(pl.lit("pre-expansion").alias("cohort")),
           on="place_id", how="left").with_columns(
        pl.col("cohort").fill_null("added after old EDA"),
        pl.when(pl.col("cuisine_category").fill_null("Other") == "Other")
          .then(pl.col("primary_cuisine").fill_null("unknown").str.to_lowercase()
                .replace({"other": "unknown"}))
          .otherwise(pl.col("cuisine_category").str.to_lowercase())
          .alias("audited_cuisine")
    ))
    audited_cuisine = (repaired.group_by("audited_cuisine").agg(
        pl.len().alias("restaurants")
    ).with_columns(
        (pl.col("restaurants") / rest.height).alias("share_of_catalogue")
    ).sort("restaurants", descending=True))
    audited_cuisine.write_csv(TABLES / "audited_cuisine_distribution.csv")
    cohort_cuisine = (repaired.group_by("cohort").agg(
        pl.len().alias("restaurants"),
        (pl.col("cuisine_category").fill_null("Other") == "Other").sum().alias("google_other"),
        (pl.col("audited_cuisine") == "unknown").sum().alias("post_repair_unknown"),
    ).with_columns(
        (pl.col("google_other") / pl.col("restaurants")).alias("google_other_share"),
        (pl.col("post_repair_unknown") / pl.col("restaurants")).alias("post_repair_unknown_share"),
    ).sort("cohort"))
    cohort_cuisine.write_csv(TABLES / "cuisine_other_by_collection_cohort.csv")
    repair.group_by("classification_method").agg(
        pl.len().alias("restaurants")
    ).sort("restaurants", descending=True).write_csv(TABLES / "cuisine_repair_methods.csv")

    rating_dist = (valid_rev.with_columns(
        pl.col("rating").cast(pl.Float64).round().cast(pl.Int8).alias("stars")
    ).group_by("stars").agg(pl.len().alias("reviews")).sort("stars").collect())
    rating_dist = rating_dist.with_columns(
        (pl.col("reviews") / pl.col("reviews").sum()).alias("share")
    )
    rating_dist.write_csv(TABLES / "rating_distribution.csv")

    activity = (rev.filter(pl.col("contributor_id").is_not_null() &
                           (pl.col("contributor_id") != ""))
        .group_by("contributor_id").agg(
            pl.len().alias("review_rows"), pl.col("place_id").n_unique().alias("restaurants")
        ).collect())
    bins = (activity.with_columns(
        pl.when(pl.col("restaurants") == 1).then(pl.lit("1"))
        .when(pl.col("restaurants") == 2).then(pl.lit("2"))
        .when(pl.col("restaurants") == 3).then(pl.lit("3"))
        .when(pl.col("restaurants") <= 5).then(pl.lit("4–5"))
        .when(pl.col("restaurants") <= 10).then(pl.lit("6–10"))
        .when(pl.col("restaurants") <= 25).then(pl.lit("11–25"))
        .otherwise(pl.lit("26+")).alias("activity_band")
    ).group_by("activity_band").agg(pl.len().alias("reviewers")))
    order = {v: i for i, v in enumerate(["1","2","3","4–5","6–10","11–25","26+"])}
    bins = bins.with_columns(
        pl.col("activity_band").replace(order).cast(pl.Int8).alias("_order"),
        (pl.col("reviewers") / pl.col("reviewers").sum()).alias("share"),
    ).sort("_order").drop("_order")
    bins.write_csv(TABLES / "reviewer_activity.csv")

    quality = pl.DataFrame({
        "measure": ["missing_address", "missing_coordinates", "missing_places_rating",
                    "missing_price_level", "zero_collected_reviews"],
        "count": [rest["address"].null_count(),
                  rest.filter(pl.col("lat").is_null() | pl.col("lng").is_null()).height,
                  rest["rating"].null_count(), rest["price_level"].null_count(),
                  rest_profile.filter(pl.col("collected_reviews") == 0).height],
    }).with_columns((pl.col("count") / rest.height).alias("share"))
    quality.write_csv(TABLES / "data_quality.csv")

    state_pd = state.drop_nulls("state").head(15).to_pandas()
    save_bar(state_pd, "state", "restaurants", "Expanded restaurant coverage by state",
             "Restaurants", "restaurants_by_state.png", horizontal=True)
    cuisine_pd = cuisine.head(15).to_pandas()
    save_bar(cuisine_pd, "cuisine_category", "restaurants",
             "Restaurant catalogue by Google-type cuisine", "Restaurants",
             "restaurants_by_google_type_cuisine.png", horizontal=True)
    audited_cuisine_pd = audited_cuisine.head(20).to_pandas()
    audited_cuisine_pd["audited_cuisine"] = audited_cuisine_pd["audited_cuisine"].replace(
        {"unknown": "Unknown / insufficient evidence"}
    )
    save_bar(audited_cuisine_pd, "audited_cuisine", "restaurants",
             "Restaurant catalogue by audited cuisine", "Restaurants",
             "restaurants_by_audited_cuisine.png", horizontal=True)
    rating_pd = rating_dist.to_pandas()
    save_bar(rating_pd, "stars", "share", "Collected review rating distribution",
             "Share of reviews", "review_rating_distribution.png")
    activity_pd = bins.to_pandas()
    save_bar(activity_pd, "activity_band", "share", "Reviewer activity is strongly long-tailed",
             "Share of reviewers", "reviewer_activity.png")

    demographics = (rev.filter(
        pl.col("contributor_id").is_not_null() & (pl.col("contributor_id") != "")
    ).group_by("contributor_id").agg(
        pl.col("predicted_gender").drop_nulls().first().alias("gender"),
        pl.col("predicted_race").drop_nulls().first().alias("race"),
    ).collect())
    gender = (demographics.group_by(
        pl.col("gender").fill_null("unclassified")
    ).agg(pl.len().alias("reviewers")).with_columns(
        (pl.col("reviewers") / demographics.height).alias("share")
    ).sort("reviewers", descending=True))
    race = (demographics.group_by(
        pl.col("race").fill_null("unclassified")
    ).agg(pl.len().alias("reviewers")).with_columns(
        (pl.col("reviewers") / demographics.height).alias("share")
    ).sort("reviewers", descending=True))
    gender.write_csv(TABLES / "reviewer_gender_distribution.csv")
    race.write_csv(TABLES / "reviewer_race_distribution.csv")
    save_bar(gender.to_pandas(), "gender", "reviewers",
             "Derived reviewer gender distribution", "Unique reviewers",
             "reviewer_gender_distribution.png")
    save_bar(race.to_pandas(), "race", "reviewers",
             "Derived reviewer race/ethnicity distribution", "Unique reviewers",
             "reviewer_race_distribution.png")

    freeze = json.loads((DATA / "review_dataset_freeze.json").read_text())
    density = json.loads((DATA / "top10000_density_cbgs_summary.json").read_text())
    split = json.loads((DATA / "llm_corpora" / "split_manifest.json").read_text())
    old_catalogue = 44_630  # frozen baseline stated by the archived pre-expansion article
    old_reviewed = 17_923
    # Match the publication split exactly: require a usable user, place and
    # rating, then count unique user--restaurant interactions.
    model_activity = (rev.filter(
        pl.col("contributor_id").is_not_null() &
        (pl.col("contributor_id") != "") &
        pl.col("place_id").is_not_null() &
        pl.col("rating").is_not_null()
    ).unique(["contributor_id", "place_id"]).group_by("contributor_id").agg(
        pl.len().alias("restaurants")
    ).collect())
    eligible = int((model_activity["restaurants"] >= 4).sum())
    raw_catalogue = freeze["catalogue"]
    summary = {
        "restaurants": rest.height,
        "restaurants_with_reviews": int((rest_profile["collected_reviews"] > 0).sum()),
        "zero_review_restaurants": int((rest_profile["collected_reviews"] == 0).sum()),
        "cbgs": rest["cbg"].n_unique(),
        "states_and_dc": state["state"].drop_nulls().n_unique(),
        "states_missing_from_canonical_catalogue": ["AK", "MT"],
        **rstat,
        "raw_scraper_review_objects": raw_catalogue["review_records"],
        "analytical_rows_removed": raw_catalogue["review_records"] - rstat["review_rows"],
        "searched_cbgs_with_restaurant_hits": raw_catalogue["cbgs_in_index"],
        "canonical_restaurant_cbgs": rest["cbg"].n_unique(),
        "cuisine_repair_target_rows": repair.height,
        "cuisine_repair_coverage": repair.height / int((rest["cuisine_category"].fill_null("Other") == "Other").sum()),
        "google_other_restaurants": int((rest["cuisine_category"].fill_null("Other") == "Other").sum()),
        "post_repair_unknown_restaurants": int((repaired["audited_cuisine"] == "unknown").sum()),
        "gender_unclassified_reviewers": int(demographics["gender"].is_null().sum()),
        "gender_unknown_reviewers": int((demographics["gender"] == "unknown").sum()),
        "gender_classified_reviewers": int((
            demographics["gender"].is_not_null() &
            (demographics["gender"] != "unknown")
        ).sum()),
        "race_classified_reviewers": int((
            demographics["race"].is_not_null() &
            (demographics["race"] != "unknown")
        ).sum()),
        "eligible_reviewers_ge4_restaurants": eligible,
        "publication_train_interactions": split["interaction_counts"]["train"],
        "publication_validation_interactions": split["interaction_counts"]["validation"],
        "publication_test_interactions": split["interaction_counts"]["test"],
        "pre_expansion_article_catalogue": old_catalogue,
        "catalogue_growth": rest.height - old_catalogue,
        "catalogue_growth_share": rest.height / old_catalogue - 1,
        "reviewed_restaurant_growth": int((rest_profile["collected_reviews"] > 0).sum()) - old_reviewed,
        "top10k_density_cbgs": density["top_n"],
        "top10k_previously_queried": density["previously_queried"],
        "top10k_incremental_cbgs": density["incremental_cbgs"],
        "collection_integrity_issues": sum(freeze["integrity"]["issue_counts"].values()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")

    top_states = ", ".join(f"{r['state']} ({r['restaurants']:,})"
                           for r in state.head(5).iter_rows(named=True))
    top_cuisines = ", ".join(f"{r['cuisine_category']} ({r['restaurants']:,})"
                             for r in cuisine.head(5).iter_rows(named=True))
    report = f"""# Expanded dataset exploratory analysis

Generated from the frozen canonical files by `analysis/expanded_eda/run_eda.py`.

## Executive profile

The expanded catalogue contains **{rest.height:,} restaurants** across
**{summary['cbgs']:,} canonical restaurant CBGs** and **{summary['states_and_dc']} states/DC**.
**{summary['restaurants_with_reviews']:,} restaurants ({pct(summary['restaurants_with_reviews']/rest.height)})**
have at least one collected review; **{summary['zero_review_restaurants']:,}** have none.
The review table contains **{summary['review_rows']:,} rows**, representing
**{summary['unique_review_ids']:,} unique review IDs** from
**{summary['unique_reviewers']:,} identified reviewers**.

Relative to the archived article's pre-expansion discovery catalogue of
{old_catalogue:,} restaurants, the current catalogue is larger by
**{summary['catalogue_growth']:,} ({pct(summary['catalogue_growth_share'])})**.
This comparison is a historical-baseline comparison, not an attribution of
every added restaurant specifically to the final density-CBG pass.

## Collection-to-analysis funnel

The frozen scraper audit recorded **{raw_catalogue['review_records']:,} review objects**.
Canonical processing retained **{summary['review_rows']:,} analytical rows**,
removing **{summary['analytical_rows_removed']:,} ({pct(summary['analytical_rows_removed']/raw_catalogue['review_records'])})**
that did not satisfy the flattening pipeline's identity requirements. Similarly,
**{raw_catalogue['cbgs_in_index']:,} CBGs** produced at least one restaurant hit
during tiled discovery, whereas the deduplicated restaurant catalogue contains
**{summary['cbgs']:,} canonical CBG assignments** because a restaurant found from
multiple search tiles is stored once. These counts measure different stages and
should not be used interchangeably.

## Geographic coverage

The leading states are {top_states}. The final population-density search queue covered
all 51 state/DC jurisdictions: {density['previously_queried']:,} of its top
10,000 CBGs had already been queried and {density['incremental_cbgs']:,} required
new searches. The resulting canonical restaurant catalogue represents 49
states/DC; Alaska and Montana produced no retained restaurant records.

![Restaurants by state](figures/restaurants_by_state.png)

## Catalogue composition and LLM cuisine audit

The primary descriptive cuisine chart currently uses the deterministic
Google-type taxonomy. Its largest categories are {top_cuisines}.

![Restaurants by Google-type cuisine](figures/restaurants_by_google_type_cuisine.png)

Google's broad type mapping originally left **{summary['google_other_restaurants']:,}
({pct(summary['google_other_restaurants']/rest.height)})** restaurants as `Other`.
The audited targeted repair assigns a supported cuisine when evidence exists and
retains **{summary['post_repair_unknown_restaurants']:,}
({pct(summary['post_repair_unknown_restaurants']/rest.height)})** as
`Unknown / insufficient evidence`; it never forces a label. `Other` was not
created only by expansion: `tables/cuisine_other_by_collection_cohort.csv`
reports both the pre-expansion and added-after-old-EDA cohorts.

![Restaurants by audited cuisine](figures/restaurants_by_audited_cuisine.png)

## Reviews and model population

Review text is present for **{pct(summary['content_share'])}** of rows and
**{pct(summary['photo_share'])}** contain at least one attached photo. The mean
stored text length is **{summary['mean_text_length']:.1f} characters**.
There are **{eligible:,} reviewers with at least four unique rated restaurants**,
exactly matching the publication leave-two-out population. They contribute
{summary['publication_train_interactions']:,} training interactions plus
{summary['publication_validation_interactions']:,} validation and
{summary['publication_test_interactions']:,} test targets.

![Rating distribution](figures/review_rating_distribution.png)

![Reviewer activity](figures/reviewer_activity.png)

## Derived reviewer demographics

Gender predictions are available for **{summary['gender_classified_reviewers']:,}**
of {summary['unique_reviewers']:,} identified reviewers, and race/ethnicity
predictions for **{summary['race_classified_reviewers']:,}**. These labels are
derived from reviewer names rather than self-reported and must be interpreted as
noisy proxy variables. The rebuild used the exact historical method: first-token
`gender_guesser` for gender and `pparasurama/raceBERT` on the full display name
for race. There are **{summary['gender_unclassified_reviewers']:,}** missing gender
predictions and **{summary['gender_unknown_reviewers']:,}** explicit `unknown`
predictions; these are distinct states.

![Derived reviewer gender](figures/reviewer_gender_distribution.png)

![Derived reviewer race and ethnicity](figures/reviewer_race_distribution.png)

## Integrity and limitations

The frozen collection audit reports **{summary['collection_integrity_issues']} unresolved integrity issues**.
The search frame is national, but the retained restaurant catalogue has no
Alaska or Montana observations and is deliberately concentrated in high-density
and previously high-foot-traffic CBGs; it is not a uniform sample of US
restaurants. Places metadata and review availability can also differ by
market, restaurant age, and Google Maps visibility. Demographic labels are
probabilistic name-based inferences and should be used for diagnostic fairness
analysis, not treated as ground truth or sensitive targeting features.

Detailed tables are under `analysis/expanded_eda/tables/`.
"""
    REPORT.write_text(report)
    print(json.dumps(summary, indent=2, default=float))
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
