# Foodie Revamp — Restaurant Recommendation from Google Reviews

A GNN-based restaurant recommender built on ~2.5M Google reviews of high-traffic US Census Block Groups, with locally-hosted-LLM feature engineering and spatial re-ranking. The full 4-part technical write-up is in [`article/`](article/).

## Repository layout

| Path | Contents |
|---|---|
| [`article/`](article/) | The 4-part write-up (data → features → architectures → results), the series outline, and `RECONCILIATION.md` (results provenance / canonical-numbers decision) |
| [`figures/`](figures/) | Publication figures and the scripts that generate them (`make_*.py`) |
| [`results/`](results/) | Experiment logs — `model_results.csv`, `proximity_grid_search.csv`, `training_checkpoints.csv` |
| `data/`, `models/` | Large artifacts — **git-ignored**; regenerate from the pipeline |
| Literature PDFs (root) | The six reimplemented papers (InfoNCE-GCL, RaDAR, HEK-CL, Self-GNN, HGNN-AR, LIT-GRAPH) — see Part 3 references |

## Code (project root)

- **Data collection** — `hexagon_places.py`, `scrape_reviews*.py`, `expand_cbgs*.py`, `run_expansion_pipeline.py`, `build_graph*.py`
- **Feature engineering** — `build_*_features.py`, `extract_dishes_llm.py`, `build_dish_embeddings.py`, `build_training_features.py`
- **Models** — `recommendation_*.py` (matrix factorization, LightGCN, KGAT, contrastive, the InfoNCE-KGAT-SAL hybrid); training drivers `run_*.py`
- **Post-processing** — `rerank_proximity.py` (spatial re-ranking), `explain_recommendations.py` (recommendation explanations)
- **Analysis** — exploratory notebooks are **git-ignored** (their cell outputs may contain reviewer-level data); the reproducible EDA lives in [`figures/make_eda.py`](figures/make_eda.py)

## Canonical results

All published numbers use the **unextended** dataset at embedding dimension 2048. A later dataset expansion degraded results and is excluded — see [`article/RECONCILIATION.md`](article/RECONCILIATION.md).
