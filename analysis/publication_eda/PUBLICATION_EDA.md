# Publication EDA figure notes

These figures use the frozen 75,203-restaurant and 5,544,947-review analytical tables.

## Display treatment of cuisine

The cuisine figures exclude 6,996 restaurants (9.3%) with insufficient supported evidence. No restaurant is deleted from the dataset. Clear repair-label synonyms are consolidated into the publication taxonomy; genuinely distinct cuisines remain separate.

## Figures

- `eda_catalogue_coverage.png`: state coverage and classified cuisine distribution.
- `eda_state.png`: restaurant counts and stacked cuisine mix for the largest state catalogues.
- `eda_state_cuisine_heatmap.png`: optional numerical version of the state-cuisine comparison.
- `eda_ratings_and_activity.png`: review sentiment and reviewer activity funnel.
- `eda_review_richness.png`: text and photo evidence by star rating.
- `eda_reviewer_demographics.png`: qualified name-inference diagnostics.

- `eda_rating_gender.png`: grouped-bar comparison of mean cuisine ratings by inferred gender.
- `eda_race_cuisine_heatmap.png`: mean cuisine ratings by inferred race/ethnicity.
- `eda_composition_by_cuisine.png`: demographic composition of reviews by cuisine.
- `eda_active_reviewers.png`: geography, ratings, and cuisine breadth of modeling users.

Cuisine percentages use classified restaurants as their denominator. State totals, review totals, and model-population counts retain the entire eligible dataset.
