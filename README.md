# Foodie Revamp

Foodie Revamp is a restaurant recommendation study built from Google Places
metadata and public Google Maps reviews. It evaluates conventional and
LLM-derived user and restaurant features in a Two-Tower reference model,
LightGCN, and KGAT-SAL, followed by validation-tuned geographic reranking and a
GraphRAG explanation layer.

## Current experiment

The current frozen experiment is the September 2026 expanded dataset:

- 158,981 eligible users;
- 85,381 catalogued restaurants;
- 67,947 restaurants in the full-catalogue ranking population;
- 789,505 training, 158,981 validation, and 158,981 test interactions;
- three seeds (42, 43, and 44), 300 epochs, and 1,024-dimensional embeddings.

KGAT-SAL with LLM features is the strongest raw model. Its mean test metrics
are Hit@10 0.02754 and NDCG@10 0.01367. Validation-only proximity tuning
selected alpha 0.6, a 2.5 km bandwidth, and a candidate depth of 1,600,
increasing mean test performance to Hit@10 0.04597 and NDCG@10 0.02206.

See [the expanded result summary](results/expanded_2026-09-21/expanded_proximity_summary.md)
and [experiment manifest](results/expanded_2026-09-21/experiment_manifest.json).

## Repository layout

| Path | Purpose |
|---|---|
| `article/` | Current technical-article drafts, figures, Google Docs-ready exports, and editorial handoff material |
| `analysis/` | Reproducible EDA, segment evaluation, and supporting tables |
| `foodie/collection/` | Places discovery, review scraping, budget controls, and dataset freezing |
| `foodie/features/` | Canonical data preparation, feature generation, audits, and feature contracts |
| `foodie/modeling/` | Recommender implementations, experiment runners, reranking, and summaries |
| `foodie/explanations/` | GraphRAG generation, remediation, and explanation audits |
| `foodie/evaluation/` | Publication tables and rating-aware evaluation |
| `results/expanded_2026-09-21/` | Current per-seed metrics, aggregate comparisons, proximity grid, and frozen selection |
| `tests/` | Regression tests for split integrity, feature generation, collection safeguards, and GraphRAG remediation |
| `systemd/` | User-service definitions for long-running collection and experiment jobs |
| `literature/` | Literature index and cited research papers, organized by topic |
| `data/`, `models/` | Large generated artifacts, excluded from Git |
| `archive/` | Recoverable superseded artifacts, excluded from Git; see [ARCHIVE_INDEX.md](ARCHIVE_INDEX.md) |

## Main pipelines

Run project entry points as modules from the repository root. For example:

```bash
/home/swami/venv/dev_env/bin/python -m foodie.collection.run_places_then_reviews --help
/home/swami/venv/dev_env/bin/python -m foodie.features.build_training_features --help
/home/swami/venv/dev_env/bin/python -m foodie.modeling.run_expanded_model_experiments
/home/swami/venv/dev_env/bin/python -m foodie.explanations.run_publication_graphrag --help
```

See [SCRIPT_LAYOUT.md](docs/SCRIPT_LAYOUT.md) for the categorized inventory,
entry-point convention, and archived-script decisions.

## Environment

The project has been run with Python 3.12 in
`/home/swami/venv/dev_env`. Large data and checkpoint files are intentionally
not stored in Git. Paths can be redirected through the `FOODIE_*` environment
variables used by the pipeline scripts.

## Reproducibility note

The interaction split is chronological leave-two-out after user-restaurant
deduplication. Behavioral and LLM-derived features are constructed from
training evidence only. Validation selects checkpoints and proximity
hyperparameters; the test split is evaluated after those choices are frozen.
