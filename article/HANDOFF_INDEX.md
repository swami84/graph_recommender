# Technical article handoff index

## Start here

- Medium editorial handoff and final checks: `MEDIUM_EDITORIAL_HANDOFF.md`
- Part 1 — data collection and EDA: `drafts/PART_1_DATA_COLLECTION_AND_EDA.md`
- Part 2 — LLM feature engineering: `drafts/PART_2_LLM_FEATURE_ENGINEERING.md`
- Part 3 — model evaluation: `drafts/PART_3_MODEL_EVALUATION.md`
- Part 4 — GraphRAG explanations: `drafts/PART_4_GRAPHRAG_EXPLANATIONS.md`
- Combined working draft: `drafts/TECHNICAL_ARTICLE_SERIES_DRAFT.md`
- Writing-expert instructions and frozen facts: `WRITING_EXPERT_PROMPT.md`
- Medium editorial memo, fact-check, figure inventory, requests, and publishing checklist:
  `MEDIUM_EDITORIAL_HANDOFF.md`
- Concise final numerical report: `../results/publication_final_results.md`
- EDA methodology and display exclusions: `../analysis/publication_eda/PUBLICATION_EDA.md`
- Frozen study design: `PUBLICATION_STUDY_DESIGN.md`

## Opening the articles in Google Docs

The edited Markdown files are the canonical Medium sources. Google Docs does not resolve their local
relative image paths. The existing files under `google_docs_ready/` predate the final Medium edit;
regenerate them with `article/build_google_docs_ready.py` after the required human-audit placeholder
is resolved. The resulting DOCX files embed the figures, and the matching PDFs provide fixed-layout
visual previews.

| Article | Google Docs upload file | Embedded figures |
|---|---|---:|
| Part 1 — Data collection and EDA | `google_docs_ready/PART_1_DATA_COLLECTION_AND_EDA.docx` | 9 |
| Part 2 — LLM feature engineering | `google_docs_ready/PART_2_LLM_FEATURE_ENGINEERING.docx` | 1 |
| Part 3 — Model evaluation | `google_docs_ready/PART_3_MODEL_EVALUATION.docx` | 5 |
| Part 4 — GraphRAG explanations | `google_docs_ready/PART_4_GRAPHRAG_EXPLANATIONS.docx` | 1 |

## Completed figure set

| Figure | Intended use |
|---|---|
| `figures/eda_catalogue_coverage.png` | Part 1: national coverage and classified cuisine mix |
| `figures/collection_coverage_map.png` | Part 1: searched CBG footprint across the US |
| `figures/eda_state.png` | Part 1: state counts and stacked cuisine distribution |
| `figures/eda_ratings_and_activity.png` | Part 1: ratings and modeling-population funnel |
| `figures/eda_review_richness.png` | Part 1/2: evidence content by sentiment |
| `figures/eda_reviewer_demographics.png` | Optional Part 1 diagnostic with inference caveat |
| `figures/eda_rating_gender.png` | Part 1: inferred-gender mean ratings by cuisine |
| `figures/eda_race_cuisine_heatmap.png` | Part 1: inferred-race/ethnicity mean ratings by cuisine |
| `figures/eda_composition_by_cuisine.png` | Part 1: inferred demographic review shares by cuisine |
| `figures/eda_active_reviewers.png` | Part 1: current four-restaurant modeling-user profile |
| `figures/graph_schema.png` | Part 2: typed KGAT-SAL graph and temporal views |
| `figures/feature_tree.png` | Part 2: exact frozen analytical and LLM feature taxonomy |
| `figures/feature_contract_schematic.png` | Part 2: chronological source and feature-construction contract |
| `figures/publication_model_llm_comparison.png` | Part 3: asymmetric LLM-feature benefit |
| `figures/publication_proximity_uplift.png` | Part 3: proximity gain |
| `figures/fairness_performance.png` | Part 3: final-model metrics by inferred demographic groups |
| `figures/segment_performance.png` | Part 3: final-model metrics by held-out cuisine |
| `figures/publication_rating_aware_hits.png` | Part 3: liked versus disliked recovered targets |
| `figures/graphrag_explanation_example.png` | Part 4: real evidence-to-explanation example |
| `figures/publication_graphrag_audit.png` | Optional appendix: automated quality/provenance audit |

All figures are rasterized at 240 DPI and use the shared visual system in
`figures/publication_style.py`.

## Regeneration

```bash
cd /home/swami/Work/Projects/foodie_revamp
export MPLCONFIGDIR=/tmp/foodie-mpl
/home/swami/venv/dev_env/bin/python analysis/expanded_eda/make_publication_eda.py
/home/swami/venv/dev_env/bin/python analysis/evaluation/make_segment_performance.py
/home/swami/venv/dev_env/bin/python -m foodie.evaluation.analyze_rating_aware_recommendations
/home/swami/venv/dev_env/bin/python -m foodie.evaluation.make_publication_results
/home/swami/venv/dev_env/bin/python article/figures/make_graph_schema.py
/home/swami/venv/dev_env/bin/python article/figures/make_feature_taxonomy.py
/home/swami/venv/dev_env/bin/python article/figures/make_feature_contract_schematic.py
/home/swami/venv/dev_env/bin/python article/figures/make_graphrag_example.py
/home/swami/venv/dev_env/bin/python article/build_google_docs_ready.py
```

## Remaining publication gates

1. Complete the blinded human audit in
   `../results/graphrag/publication_graphrag_human_audit.csv`.
2. Have the writing expert fact-check every edited numerical claim against the artifacts listed in
   `WRITING_EXPERT_PROMPT.md`.
3. The planned structured-only versus embeddings-only focused decomposition has not been run. Do
   not describe either component as the sole cause of the full-LLM gain without those six runs.
4. If stronger product claims are desired, retrain with a rating-aware objective; the current
   primary metrics evaluate held-out visit retrieval, not satisfaction.
