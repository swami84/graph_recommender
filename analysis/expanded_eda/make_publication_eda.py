#!/usr/bin/env python3
"""Create publication-grade EDA figures from the frozen expanded dataset."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from article.figures.publication_style import (  # noqa: E402
    apply_style, clean_axis, save, title, NAVY, TEAL, CORAL, GOLD, PURPLE,
    LIGHT, MUTED, RATING_COLORS,
)

DATA = ROOT / "data"
OUT = ROOT / "analysis" / "publication_eda"
TABLES = OUT / "tables"
FIGURES = ROOT / "article" / "figures"

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

REPAIR_MAP = {
    "unknown": None,
    "other": None,
    "mexican": "Mexican & Latin",
    "latin_american": "Mexican & Latin",
    "south_american": "Mexican & Latin",
    "japanese": "Japanese & Sushi",
    "cafe_bakery": "Café & Bakery",
    "burgers": "Fast Food & Burgers",
    "middle_eastern": "Mediterranean & Middle Eastern",
    "mediterranean": "Mediterranean & Middle Eastern",
    "pizza": "Italian & Pizza",
    "italian": "Italian & Pizza",
    "bbq": "BBQ & Steakhouse",
    "steakhouse": "BBQ & Steakhouse",
    "indian": "Indian & South Asian",
    "american": "American & Comfort",
    "southern": "American & Comfort",
    "bar_pub": "Bar & Pub",
    "british_irish": "Bar & Pub",
    "chinese": "Chinese",
    "seafood": "Seafood",
}


def display_label(value: str | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    key = str(value).strip().lower()
    if key in REPAIR_MAP:
        return REPAIR_MAP[key]
    replacements = {
        "asian (other)": "Asian",
        "bbq & steakhouse": "BBQ & Steakhouse",
        "café & bakery": "Café & Bakery",
    }
    return replacements.get(key, key.replace("_", " ").title())


def audited_restaurants() -> pd.DataFrame:
    rest = pd.read_parquet(DATA / "restaurants_enriched.parquet", columns=["place_id", "cbg", "cuisine_category"])
    repair = pd.read_parquet(DATA / "restaurant_cuisine_27b_v4.parquet", columns=["place_id", "primary_cuisine"])
    rest = rest.merge(repair, on="place_id", how="left")
    generic = rest.cuisine_category.fillna("Other").eq("Other")
    source = rest.cuisine_category.where(~generic, rest.primary_cuisine)
    rest["cuisine"] = source.map(display_label)
    rest["cbg"] = rest.cbg.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)
    rest["state"] = rest.cbg.str[:2].map(FIPS)
    return rest


def annotate_bars(axis, bars, formatter=lambda value: f"{value:,.0f}"):
    labels = [formatter(bar.get_width() if bar.get_width() > bar.get_height() else bar.get_height()) for bar in bars]
    axis.bar_label(bars, labels=labels, padding=4, fontsize=9, color=MUTED)


def main() -> None:
    apply_style()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rest = audited_restaurants()
    classified = rest[rest.cuisine.notna()].copy()
    excluded = len(rest) - len(classified)

    state_counts = rest.state.value_counts().rename_axis("state").reset_index(name="restaurants")
    cuisine_counts = classified.cuisine.value_counts().rename_axis("cuisine").reset_index(name="restaurants")
    state_counts.to_csv(TABLES / "restaurant_state_counts.csv", index=False)
    cuisine_counts.to_csv(TABLES / "classified_cuisine_counts.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"wspace": 0.38})
    state_plot = state_counts.head(12).iloc[::-1]
    bars = axes[0].barh(state_plot.state, state_plot.restaurants, color=PURPLE, height=0.72)
    annotate_bars(axes[0], bars)
    axes[0].set_title("Restaurant counts by state")
    axes[0].set_xlabel("Restaurants")
    axes[0].set_xlim(0, state_plot.restaurants.max() * 1.18)
    clean_axis(axes[0], "x")
    cuisine_plot = cuisine_counts.head(14).iloc[::-1]
    share = 100 * cuisine_plot.restaurants / len(classified)
    bars = axes[1].barh(cuisine_plot.cuisine, share, color=NAVY, height=0.72)
    annotate_bars(axes[1], bars, lambda value: f"{value:.1f}%")
    axes[1].set_title("Cuisine mix among classified venues")
    axes[1].set_xlabel("Share of classified restaurants")
    axes[1].set_xlim(0, share.max() * 1.22)
    clean_axis(axes[1], "x")
    title(fig, "Restaurant catalogue coverage and supported cuisine composition",
          f"75,203 restaurants; cuisine panel excludes {excluded:,} venues with insufficient evidence")
    save(fig, FIGURES / "eda_catalogue_coverage.png")

    top_states = state_counts.head(8).state.tolist()
    top_cuisines = cuisine_counts.head(10).cuisine.tolist()
    counts = pd.crosstab(classified.state, classified.cuisine).reindex(index=top_states, columns=top_cuisines, fill_value=0)
    denominators = classified.groupby("state").size().reindex(top_states)
    matrix = counts.div(denominators, axis=0) * 100
    matrix.to_csv(TABLES / "top_state_cuisine_shares.csv")
    fig, ax = plt.subplots(figsize=(13, 6.1))
    image = ax.imshow(matrix.values, cmap="Blues", aspect="auto", vmin=0, vmax=max(18, matrix.to_numpy().max()))
    ax.set_xticks(range(len(top_cuisines)), top_cuisines, rotation=32, ha="right")
    ax.set_yticks(range(len(top_states)), top_states)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iat[row, column]
            ax.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if value > matrix.to_numpy().max() * 0.58 else "#263238")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Share of state's classified restaurants (%)")
    ax.grid(False)
    title(fig, "State-level distribution of supported cuisine categories",
          "Top ten specific cuisines; no generic or insufficient-evidence category is displayed")
    save(fig, FIGURES / "eda_state_cuisine_heatmap.png")

    # Legacy-style paired state figure: catalogue size plus stacked cuisine mix.
    state_mix_counts = pd.crosstab(classified.state, classified.cuisine).reindex(
        index=top_states, columns=top_cuisines, fill_value=0
    )
    # Normalize over the displayed, classified cuisines so no synthetic "Other" segment is needed.
    state_mix = state_mix_counts.div(state_mix_counts.sum(axis=1), axis=0)
    state_mix.to_csv(TABLES / "top_state_displayed_cuisine_composition.csv")
    cuisine_palette = [NAVY, CORAL, TEAL, "#C65D57", PURPLE,
                       "#8C6D5A", "#D16BA5", "#7B8794", "#B8B51B", "#2FB5C4"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [0.9, 1.25], "wspace": 0.25})
    state_plot = state_counts.head(12).iloc[::-1]
    bars = axes[0].barh(state_plot.state, state_plot.restaurants, color=PURPLE, height=0.72)
    annotate_bars(axes[0], bars)
    axes[0].set_title("Restaurants by state")
    axes[0].set_xlabel("Restaurants")
    axes[0].set_xlim(0, state_plot.restaurants.max() * 1.18)
    clean_axis(axes[0], "x")
    state_mix.plot(kind="bar", stacked=True, ax=axes[1], color=cuisine_palette,
                   width=0.78, edgecolor="white", linewidth=0.3)
    axes[1].set_title("Cuisine mix in the largest state catalogues")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Share across displayed cuisines")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    clean_axis(axes[1])
    title(fig, "State distribution and cuisine composition of the modeling catalogue",
          "Stacked bars use the ten most common supported cuisines; insufficient-evidence venues are omitted")
    save(fig, FIGURES / "eda_state.png")

    scan = pl.scan_parquet(DATA / "reviews_flat.parquet")
    ratings = (scan.filter(pl.col("rating").cast(pl.Float64, strict=False).is_between(1, 5))
        .with_columns(pl.col("rating").cast(pl.Float64).round().cast(pl.Int8).alias("stars"))
        .group_by("stars").agg(
            pl.len().alias("reviews"),
            pl.col("text_len").fill_null(0).mean().alias("mean_text_length"),
            pl.col("attached_photos").fill_null(0).mean().alias("mean_photos"),
        ).sort("stars").collect().to_pandas())
    ratings["share"] = ratings.reviews / ratings.reviews.sum()
    activity = pd.read_csv(ROOT / "analysis/expanded_eda/tables/reviewer_activity.csv")
    ratings.to_csv(TABLES / "review_behavior_by_rating.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"wspace": 0.30})
    x = np.arange(1, 6)
    bars = axes[0].bar(x, 100 * ratings.share, color=RATING_COLORS, width=0.72)
    axes[0].bar_label(bars, labels=[f"{100*v:.1f}%" for v in ratings.share], padding=4)
    axes[0].set_xticks(x, [f"{value}★" for value in x])
    axes[0].set_ylabel("Share of collected reviews")
    axes[0].set_title("Review-rating distribution")
    axes[0].set_ylim(0, 72)
    clean_axis(axes[0])
    bars = axes[1].bar(activity.activity_band, 100 * activity.share,
                       color=[LIGHT, LIGHT, LIGHT, TEAL, TEAL, TEAL, TEAL], width=0.72,
                       edgecolor=[MUTED] * 3 + [TEAL] * 4)
    axes[1].bar_label(bars, labels=[f"{100*v:.1f}%" if v >= .001 else f"{100*v:.2f}%" for v in activity.share], padding=4)
    axes[1].axvline(2.5, color=CORAL, linewidth=1.6, linestyle="--")
    axes[1].text(2.65, 66, "modeling population\nstarts at 4 restaurants", color=CORAL, fontsize=9)
    axes[1].set_ylabel("Share of reviewers")
    axes[1].set_xlabel("Distinct restaurants reviewed")
    axes[1].set_title("Reviewer-activity distribution")
    axes[1].set_ylim(0, 91)
    clean_axis(axes[1])
    title(fig, "Review-rating distribution and modeling-population threshold",
          "The leave-two-out experiment retains 156,261 users with at least four unique restaurants")
    save(fig, FIGURES / "eda_ratings_and_activity.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"wspace": 0.30})
    bars = axes[0].bar(x, ratings.mean_text_length, color=RATING_COLORS, width=0.72)
    axes[0].bar_label(bars, labels=[f"{v:.0f}" for v in ratings.mean_text_length], padding=4)
    axes[0].set_xticks(x, [f"{value}★" for value in x])
    axes[0].set_ylabel("Mean characters")
    axes[0].set_title("Mean review length by rating")
    axes[0].set_ylim(0, ratings.mean_text_length.max() * 1.18)
    clean_axis(axes[0])
    bars = axes[1].bar(x, ratings.mean_photos, color=RATING_COLORS, width=0.72)
    axes[1].bar_label(bars, labels=[f"{v:.2f}" for v in ratings.mean_photos], padding=4)
    axes[1].set_xticks(x, [f"{value}★" for value in x])
    axes[1].set_ylabel("Mean attached photos")
    axes[1].set_title("Mean attached-photo count by rating")
    axes[1].set_ylim(0, ratings.mean_photos.max() * 1.22)
    clean_axis(axes[1])
    title(fig, "Review text and photo availability by star rating",
          "Long complaints and photo-heavy praise give the LLM different kinds of evidence")
    save(fig, FIGURES / "eda_review_richness.png")

    # Cuisine-conditioned demographic diagnostics use only classified cuisine rows and
    # name-inference labels with adequate support. They are descriptive, not identity claims.
    restaurant_lookup = pl.from_pandas(
        classified[["place_id", "state", "cuisine"]], include_index=False
    ).lazy()
    review_base = (pl.scan_parquet(DATA / "reviews_flat.parquet").select(
        "review_id", "contributor_id", "place_id", "rating", "is_local_guide",
        "predicted_gender", "predicted_race",
    ).filter(
        pl.col("contributor_id").is_not_null() & (pl.col("contributor_id") != "") &
        pl.col("place_id").is_not_null() &
        pl.col("rating").cast(pl.Float64, strict=False).is_between(1, 5)
    ).with_columns(pl.col("rating").cast(pl.Float64)))
    review_with_cuisine = review_base.join(restaurant_lookup, on="place_id", how="inner")
    cuisine_review_counts = (review_with_cuisine.group_by("cuisine").agg(
        pl.len().alias("reviews")
    ).sort("reviews", descending=True).collect())
    top_review_cuisines = cuisine_review_counts.head(14)["cuisine"].to_list()

    gender_means = (review_with_cuisine.filter(
        pl.col("cuisine").is_in(top_review_cuisines) &
        pl.col("predicted_gender").is_in(["female", "male"])
    ).group_by("cuisine", "predicted_gender").agg(
        pl.col("rating").mean().alias("mean_rating"), pl.len().alias("reviews")
    ).collect().to_pandas())
    gender_pivot = gender_means.pivot(index="cuisine", columns="predicted_gender", values="mean_rating")
    gender_pivot = gender_pivot.reindex(top_review_cuisines[::-1])
    gender_means.to_csv(TABLES / "gender_cuisine_mean_ratings.csv", index=False)
    fig, ax = plt.subplots(figsize=(10.5, 7.6))
    y = np.arange(len(gender_pivot))
    height = 0.36
    female = ax.barh(y - height / 2, gender_pivot["female"], height,
                     color=CORAL, label="Female")
    male = ax.barh(y + height / 2, gender_pivot["male"], height,
                   color=NAVY, label="Male")
    ax.bar_label(female, labels=[f"{value:.2f}★" for value in gender_pivot["female"]],
                 padding=3, fontsize=7.8)
    ax.bar_label(male, labels=[f"{value:.2f}★" for value in gender_pivot["male"]],
                 padding=3, fontsize=7.8)
    ax.set_yticks(y, gender_pivot.index)
    ax.set_xlim(0, 5.35)
    ax.set_xticks(range(0, 6), ["0", "1★", "2★", "3★", "4★", "5★"])
    ax.set_xlabel("Mean rating")
    ax.set_title("Mean rating by inferred gender and cuisine")
    ax.legend(loc="lower right")
    clean_axis(ax, "x")
    title(fig, "Mean rating by inferred gender and restaurant cuisine",
          "Overall difference is statistically detectable but negligible (reviewer-level Welch p<10⁻¹⁹⁸; d=0.035); labels are name-inferred")
    save(fig, FIGURES / "eda_rating_gender.png")

    race_names = {
        "nh_white": "White", "api": "Asian",
        "nh_black": "Black", "hispanic": "Hispanic",
    }
    race_means = (review_with_cuisine.filter(
        pl.col("cuisine").is_in(top_review_cuisines) &
        pl.col("predicted_race").is_in(list(race_names))
    ).group_by("cuisine", "predicted_race").agg(
        pl.col("rating").mean().alias("mean_rating"), pl.len().alias("reviews")
    ).collect().to_pandas())
    race_means["race"] = race_means.predicted_race.map(race_names)
    race_order = list(race_names.values())
    race_pivot = race_means.pivot(index="cuisine", columns="race", values="mean_rating")
    race_pivot = race_pivot.reindex(index=top_review_cuisines, columns=race_order)
    race_means.to_csv(TABLES / "race_cuisine_mean_ratings.csv", index=False)
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    values = race_pivot.to_numpy()
    finite = values[np.isfinite(values)]
    vmin, vmax = np.quantile(finite, [0.02, 0.98])
    image = ax.imshow(values, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(race_order)), race_order, rotation=0, ha="center")
    ax.set_yticks(range(len(top_review_cuisines)), top_review_cuisines)
    ax.set_xlabel("Inferred race/ethnicity")
    ax.set_ylabel("Restaurant cuisine")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                ax.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Mean star rating")
    ax.grid(False)
    title(fig, "Mean ratings by inferred race/ethnicity and cuisine",
          "Descriptive name-inference diagnostic; AIAN omitted because the sample is very small")
    save(fig, FIGURES / "eda_race_cuisine_heatmap.png")

    gender_counts = (review_with_cuisine.filter(
        pl.col("cuisine").is_in(top_review_cuisines) &
        pl.col("predicted_gender").is_in(["female", "male"])
    ).group_by("cuisine", "predicted_gender").agg(pl.len().alias("reviews"))
      .collect().to_pandas())
    race_counts = (review_with_cuisine.filter(
        pl.col("cuisine").is_in(top_review_cuisines) &
        pl.col("predicted_race").is_in(list(race_names))
    ).group_by("cuisine", "predicted_race").agg(pl.len().alias("reviews"))
      .collect().to_pandas())
    gender_comp = gender_counts.pivot(index="cuisine", columns="predicted_gender", values="reviews").fillna(0)
    gender_comp = gender_comp.reindex(index=top_review_cuisines[::-1], columns=["female", "male"]).fillna(0)
    gender_comp = gender_comp.div(gender_comp.sum(axis=1), axis=0)
    race_comp = race_counts.pivot(index="cuisine", columns="predicted_race", values="reviews").fillna(0)
    race_comp = race_comp.reindex(index=top_review_cuisines[::-1], columns=list(race_names)).fillna(0)
    race_comp = race_comp.div(race_comp.sum(axis=1), axis=0)
    gender_comp.to_csv(TABLES / "gender_review_share_by_cuisine.csv")
    race_comp.rename(columns=race_names).to_csv(TABLES / "race_review_share_by_cuisine.csv")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.8), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    race_comp.rename(columns=race_names).plot(
        kind="barh", stacked=True, ax=axes[0], width=0.78,
        color=[NAVY, GOLD, TEAL, CORAL], edgecolor="white", linewidth=0.25,
    )
    axes[0].set_title("Inferred race/ethnicity mix of reviews")
    axes[0].set_xlabel("Share of classified reviews")
    axes[0].set_ylabel("")
    axes[0].set_xlim(0, 1)
    axes[0].legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.13),
                   ncol=2, fontsize=8)
    clean_axis(axes[0], "x")
    gender_comp.rename(columns={"female":"Female", "male":"Male"}).plot(
        kind="barh", stacked=True, ax=axes[1], width=0.78,
        color=[CORAL, NAVY], edgecolor="white", linewidth=0.25,
    )
    axes[1].set_title("Inferred gender mix of reviews")
    axes[1].set_xlabel("Share of classified reviews")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].set_xlim(0, 1)
    axes[1].legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.13),
                   ncol=2, fontsize=8)
    clean_axis(axes[1], "x")
    title(fig, "Review composition by inferred demographic group and cuisine",
          "Shares are calculated within classified demographic subsets and should not be read as population estimates")
    fig.subplots_adjust(bottom=0.18)
    save(fig, FIGURES / "eda_composition_by_cuisine.png")

    # Recreate the legacy active-reviewer profile using the current >=4-restaurant threshold.
    active_ids = (review_base.select("contributor_id", "place_id").unique()
        .group_by("contributor_id").agg(pl.len().alias("restaurants"))
        .filter(pl.col("restaurants") >= 4).select("contributor_id").collect())
    active_reviews = review_base.join(active_ids.lazy(), on="contributor_id", how="inner")
    active_with_restaurant = active_reviews.join(
        pl.from_pandas(rest[["place_id", "state", "cuisine"]], include_index=False).lazy(),
        on="place_id", how="left",
    )
    active_state = (active_with_restaurant.group_by("state").agg(pl.len().alias("reviews"))
                    .drop_nulls("state").sort("reviews", descending=True).collect().to_pandas())
    active_rating = (active_reviews.with_columns(
        pl.col("rating").round().cast(pl.Int8).alias("stars")
    ).group_by("stars").agg(pl.len().alias("reviews")).sort("stars").collect().to_pandas())
    active_rating["share"] = active_rating.reviews / active_rating.reviews.sum()
    all_rating = ratings[["stars", "reviews"]].copy()
    all_rating["share"] = all_rating.reviews / all_rating.reviews.sum()
    diversity = (active_with_restaurant.drop_nulls("cuisine").group_by("contributor_id").agg(
        pl.col("cuisine").n_unique().alias("cuisines")
    ).collect().to_pandas())
    diversity["display_cuisines"] = diversity.cuisines.clip(upper=8)
    diversity_dist = diversity.display_cuisines.value_counts(normalize=True).sort_index() * 100
    user_flags = (active_reviews.group_by("contributor_id").agg(
        pl.col("is_local_guide").fill_null(False).max().alias("is_local_guide")
    ).collect())
    mean_states = (active_with_restaurant.drop_nulls("state").group_by("contributor_id").agg(
        pl.col("state").n_unique().alias("states")
    ).select(pl.col("states").mean()).collect().item())
    active_rows = active_reviews.select(pl.len()).collect().item()
    all_rows = review_base.select(pl.len()).collect().item()
    active_mean_rating = active_reviews.select(pl.col("rating").mean()).collect().item()
    all_mean_rating = review_base.select(pl.col("rating").mean()).collect().item()
    local_guide_share = user_flags["is_local_guide"].mean()
    five_plus_share = (diversity.cuisines >= 5).mean()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"wspace": 0.30})
    state_view = active_state.head(8).iloc[::-1]
    state_share = 100 * state_view.reviews / active_state.reviews.sum()
    bars = axes[0].barh(state_view.state, state_share, color=PURPLE, height=0.72)
    axes[0].bar_label(bars, labels=[f"{value:.1f}%" for value in state_share], padding=4)
    axes[0].set_title("State distribution of modeling-user reviews")
    axes[0].set_xlabel("Share of their reviews")
    axes[0].set_xlim(0, state_share.max() * 1.20)
    clean_axis(axes[0], "x")
    stars = np.arange(1, 6)
    width = 0.36
    axes[1].bar(stars - width/2, 100 * all_rating.set_index("stars").reindex(stars).share,
                width, color="#AEB7C3", label="All reviewers")
    axes[1].bar(stars + width/2, 100 * active_rating.set_index("stars").reindex(stars).share,
                width, color=TEAL, label="Modeling users")
    axes[1].set_xticks(stars, [f"{value}★" for value in stars])
    axes[1].set_ylabel("Share of ratings")
    axes[1].set_title(f"Rating distribution ({active_mean_rating:.2f} vs {all_mean_rating:.2f})")
    axes[1].legend(loc="upper left")
    clean_axis(axes[1])
    bars = axes[2].bar(range(len(diversity_dist)), diversity_dist.values, color=TEAL, width=0.74)
    axes[2].bar_label(bars, labels=[f"{value:.1f}%" for value in diversity_dist.values], padding=4, fontsize=8)
    axes[2].set_xticks(range(len(diversity_dist)),
                       [f"{int(value)}" if value < 8 else "8+" for value in diversity_dist.index])
    axes[2].axvline(diversity.cuisines.mean() - 1, linestyle="--", color=CORAL, linewidth=1.4)
    axes[2].set_title("Cuisine variety")
    axes[2].set_xlabel("Distinct supported cuisines visited")
    axes[2].set_ylabel("Share of modeling users")
    clean_axis(axes[2])
    title(fig, f"Distributional characteristics of {len(active_ids):,} modeling users",
          f"{active_rows/all_rows:.1%} of reviews · {local_guide_share:.0%} Local Guides · "
          f"{mean_states:.1f} states each on average · {five_plus_share:.0%} sample 5+ cuisines")
    save(fig, FIGURES / "eda_active_reviewers.png")

    gender = pd.read_csv(ROOT / "analysis/expanded_eda/tables/reviewer_gender_distribution.csv")
    race = pd.read_csv(ROOT / "analysis/expanded_eda/tables/reviewer_race_distribution.csv")
    gender = gender[gender.gender.isin(["female", "male"])].copy()
    gender["label"] = gender.gender.str.title()
    gender = gender.set_index("gender").reindex(["female", "male"]).reset_index()
    race_names = {"nh_white":"White", "api":"Asian",
                  "nh_black":"Black", "hispanic":"Hispanic"}
    race = race[race.race.isin(race_names)].copy()
    race["label"] = race.race.map(race_names)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"wspace": 0.30})
    bars = axes[0].bar(gender.label, gender.reviewers / 1e6, color=[CORAL, NAVY], width=0.62)
    axes[0].bar_label(bars, labels=[f"{v/1e6:.2f}M" for v in gender.reviewers], padding=4)
    axes[0].set_ylabel("Unique reviewers (millions)")
    axes[0].set_title("Name-inferred gender, classified subset")
    axes[0].set_ylim(0, gender.reviewers.max() / 1e6 * 1.18)
    clean_axis(axes[0])
    bars = axes[1].bar(race.label, race.reviewers / 1e6,
                       color=[NAVY, GOLD, TEAL, CORAL], width=0.68)
    axes[1].bar_label(bars, labels=[f"{v/1e6:.2f}M" for v in race.reviewers], padding=4)
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].set_xlabel("Inferred race/ethnicity")
    axes[1].set_ylabel("Unique reviewers (millions)")
    axes[1].set_title("Name-inferred race/ethnicity")
    axes[1].set_ylim(0, race.reviewers.max() / 1e6 * 1.20)
    clean_axis(axes[1])
    title(fig, "Distribution of inferred demographic labels",
          "Unknown gender and the very small AIAN category are omitted from this descriptive view")
    save(fig, FIGURES / "eda_reviewer_demographics.png")

    lines = [
        "# Publication EDA figure notes", "",
        "These figures use the frozen 75,203-restaurant and 5,544,947-review analytical tables.", "",
        "## Display treatment of cuisine", "",
        f"The cuisine figures exclude {excluded:,} restaurants ({excluded/len(rest):.1%}) with "
        "insufficient supported evidence. No restaurant is deleted from the dataset. Clear repair-label "
        "synonyms are consolidated into the publication taxonomy; genuinely distinct cuisines remain separate.", "",
        "## Figures", "",
        "- `eda_catalogue_coverage.png`: state coverage and classified cuisine distribution.",
        "- `eda_state.png`: restaurant counts and stacked cuisine mix for the largest state catalogues.",
        "- `eda_state_cuisine_heatmap.png`: optional numerical version of the state-cuisine comparison.",
        "- `eda_ratings_and_activity.png`: review sentiment and reviewer activity funnel.",
        "- `eda_review_richness.png`: text and photo evidence by star rating.",
        "- `eda_reviewer_demographics.png`: qualified name-inference diagnostics.", "",
        "- `eda_rating_gender.png`: grouped-bar comparison of mean cuisine ratings by inferred gender.",
        "- `eda_race_cuisine_heatmap.png`: mean cuisine ratings by inferred race/ethnicity.",
        "- `eda_composition_by_cuisine.png`: demographic composition of reviews by cuisine.",
        "- `eda_active_reviewers.png`: geography, ratings, and cuisine breadth of modeling users.", "",
        "Cuisine percentages use classified restaurants as their denominator. State totals, review totals, "
        "and model-population counts retain the entire eligible dataset.", "",
    ]
    (OUT / "PUBLICATION_EDA.md").write_text("\n".join(lines))
    print(f"Wrote ten EDA figures; excluded {excluded:,} unclassified venues only from cuisine views")


if __name__ == "__main__":
    main()
