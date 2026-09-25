# Cuisine, demographic, and legacy-feature provenance audit

## Cuisine: why the old EDA showed no `Other`

The absence of `Other` in the previous charts was not evidence that every restaurant had
been classified. Two separate implementation choices removed it from view:

1. `archive/legacy_llm_pipeline_20260904/scripts/build_llmcz_restaurants.py` applied
   `argmax` to 17 one-hot cuisine columns. Of 44,630 restaurant feature rows, 6,878 had
   all 17 cuisine values equal to zero. NumPy `argmax` returns index zero for an all-zero
   row, so those restaurants were assigned `American & Comfort`. The old feature file had
   5,295 positive American predictions, but the merged catalogue reported 12,173 American
   restaurants: exactly 5,295 + 6,878. The legacy `llm_cuisine_other` column summed to zero.
2. `archive/pre_publication_2026-09-09/article_previous/figures/make_eda.py` explicitly
   excluded `Other` when constructing `KEEP`/`CORDER`, mapped excluded labels to `Other`,
   and then reindexed the principal plots to a category order that did not contain it.

In the current 75,203-restaurant catalogue, Google's broad mapping has 15,747 `Other`
restaurants (20.94%). They span both collection cohorts:

| Cohort | Restaurants | Google `Other` | Share | Unknown after audited repair | Share |
|---|---:|---:|---:|---:|---:|
| Pre-expansion catalogue | 44,630 | 8,545 | 19.15% | 3,972 | 8.90% |
| Added after old EDA | 30,573 | 7,202 | 23.56% | 3,024 | 9.89% |

Therefore, `Other` is somewhat more common in the added cohort, but it is not a recent-data
problem: 8,545 of the 15,747 original `Other` rows belong to the old catalogue. The audited
v4 repair covers every target row, recovered 3,195 labels with high-precision evidence rules,
and adjudicated 2,762 multi-candidate cases with the constrained 27B model. It leaves 6,996
restaurants as `Unknown / insufficient evidence` rather than fabricating a label.

## Reviewer demographics

The repaired pipeline uses the exact historical algorithms recovered from
`archive/pre_publication_2026-09-09/notebooks/review_demographics_analysis.ipynb`:

- gender: `gender_guesser.Detector(case_sensitive=False)` on the first name token;
  `male`/`mostly_male` map to male, `female`/`mostly_female` map to female, and all other
  outcomes map to the explicit value `unknown`;
- race/ethnicity: `pparasurama/raceBERT` on the full display name, maximum input length 64,
  using the highest-probability output class.

The current canonical review table has zero null gender predictions and zero null race
predictions. Across non-empty unique reviewer IDs, gender counts are 1,574,659 male,
1,348,076 female, and 1,007,299 explicit unknown. `Unknown` means the name algorithm was not confident enough
to assign male/female; it is not an unprocessed or unmatched reviewer. RaceBERT assigned one
of its five classes to every canonical review row.

These are name-derived proxy labels, not self-reported demographics. They are suitable for
aggregate EDA and cautious fairness diagnostics, not sensitive targeting or claims about an
individual's identity.

## Legacy feature accounting

`legacy_feature_coverage.csv` accounts for all 78 feature columns in the four archived
feature-source parquets, plus four dish-derived columns from the old canonical matrix and
the old `race_unknown` dummy. The current matrix retains or supersedes 65 entries. The 17
old cuisine one-hots are replaced because their closed taxonomy and all-zero `argmax` bug
made them unsafe. `race_unknown` is omitted with justification because complete RaceBERT
coverage makes it constant zero.

The rebuild also corrected two silent omissions:

- six deterministic restaurant features (`rating_std`, `photo_rate`,
  `repeat_visitor_rate`, `local_guide_pct`, `restaurant_age_norm`, `race_rating_gap`) and
  two venue attributes (`has_bar`, `byob`) are restored;
- dish diversity features and KG dish edges now use only dish links from training reviews.
  They belong to the `dish_llm` group and are excluded from the non-LLM condition.

The rebuilt publication matrices contain 205 user features and 262 item features, with no
nulls, infinities, or constant columns. Model training remains paused pending acceptance of
this corrected data/feature contract.
