#!/usr/bin/env python3
"""Compare rating distributions for classified inferred-gender groups."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "reviews_flat.parquet"
OUT = ROOT / "analysis" / "publication_eda" / "tables"


def summarize(values: np.ndarray) -> dict:
    return {
        "observations": len(values),
        "mean_rating": values.mean(),
        "median_rating": np.median(values),
        "standard_deviation": values.std(ddof=1),
    }


def main() -> None:
    frame = pd.read_parquet(
        DATA,
        columns=["contributor_id", "rating", "timestamp_days_ago", "predicted_gender"],
    )
    frame["rating"] = pd.to_numeric(frame.rating, errors="coerce")
    frame = frame[
        frame.contributor_id.notna() & frame.contributor_id.ne("") &
        frame.rating.between(1, 5)
    ].copy()
    classified = frame[frame.predicted_gender.isin(["female", "male"])].copy()

    review_rows = []
    for gender in ["female", "male"]:
        review_rows.append({"level": "review", "gender": gender,
                            **summarize(classified.loc[classified.predicted_gender == gender, "rating"].to_numpy())})

    latest_gender = (
        classified.sort_values("timestamp_days_ago", ascending=True, na_position="last")
        .drop_duplicates("contributor_id", keep="first")
        .set_index("contributor_id").predicted_gender
    )
    reviewer = frame.groupby("contributor_id").rating.mean().to_frame("mean_rating").join(
        latest_gender.rename("gender"), how="inner"
    )
    reviewer_rows = []
    for gender in ["female", "male"]:
        reviewer_rows.append({"level": "reviewer", "gender": gender,
                              **summarize(reviewer.loc[reviewer.gender == gender, "mean_rating"].to_numpy())})

    female = reviewer.loc[reviewer.gender == "female", "mean_rating"].to_numpy()
    male = reviewer.loc[reviewer.gender == "male", "mean_rating"].to_numpy()
    welch = stats.ttest_ind(female, male, equal_var=False)
    difference = female.mean() - male.mean()
    standard_error = np.sqrt(female.var(ddof=1) / len(female) + male.var(ddof=1) / len(male))
    pooled_sd = np.sqrt(
        ((len(female) - 1) * female.var(ddof=1) + (len(male) - 1) * male.var(ddof=1)) /
        (len(female) + len(male) - 2)
    )
    test = pd.DataFrame([{
        "analysis_level": "one mean rating per reviewer",
        "female_minus_male": difference,
        "ci_95_low": difference - 1.96 * standard_error,
        "ci_95_high": difference + 1.96 * standard_error,
        "welch_t": welch.statistic,
        "welch_df": welch.df,
        "welch_p": welch.pvalue,
        "cohens_d": difference / pooled_sd,
    }])
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(review_rows + reviewer_rows).to_csv(OUT / "gender_rating_summary.csv", index=False)
    test.to_csv(OUT / "gender_rating_significance.csv", index=False)
    print(pd.DataFrame(review_rows + reviewer_rows).to_string(index=False))
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
