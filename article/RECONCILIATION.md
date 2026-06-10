# Results Reconciliation — Source of Truth for the Article Series

_Generated 2026-06-06 during pre-publication review._

## Decision

**Canonical = the *unextended* dataset, emb_dim = 2048 (Apr 24–26 runs).**

A later dataset extension (more reviews scraped via `run_expansion_pipeline.py`, Apr 26–28)
**degraded** results, and those reruns were done at emb_dim = 1024. Per author decision, the
extended/1024 runs are **excluded** from all published numbers.

| | Canonical (publish) | Excluded |
|---|---|---|
| Dataset | unextended (pre Apr 28) | extended (Apr 28 rebuild) |
| emb_dim | 2048 (HEK-CL: 512) | 1024 |
| Timestamps in CSV | 2026-04-24 18:19 → 2026-04-26 12:17 | 2026-04-28 20:35 → 2026-05-03 03:48 |
| Row count (`model_results.csv`) | 71 | 25 |

## Canonical dataset profile (`data/graph_stats.txt`, Apr 17 — unextended graph)

| Node / Edge | Count |
|---|---|
| CBGs | 666 |
| Cuisines | 17 |
| Dishes | 517,486 |
| Restaurants | 18,879 |
| Reviewers | 1,964,125 |
| Reviews — REVIEWED edges (raw) | 10,010,512 |
| Reviews — distinct (deduped on `review_id`) | **2,502,628** |

> **Review-count note (added 2026-06-09).** `graph.pkl` is a NetworkX `MultiDiGraph` whose REVIEWED
> edges are **4× duplicated** — the same review (identical `review_id`) is stored as ~4 parallel
> edges, because each review was re-captured across multiple scrape passes. The honest distinct-review
> count is **~2.5M**, and the article cites 2.5M throughout (not 10M). This does **not** affect any model
> result: `load_interactions` in `recommendation_gnn.py` (lines ~848–865) deduplicates to one row per
> `(contributor_id, place_id)` before building the adjacency or sampling BPR triples, collapsing the
> 10.0M rows to 2,502,622 distinct interactions. Verified 2026-06-09; no retraining needed.

> Note: `data/restaurant_llm_features.parquet` covers ~44,630 restaurants — that is the
> **extended** superset. Canonical models used the 18,879-restaurant subset. Cite counts
> consistently per dataset when writing. (In the reconstructed canonical reviews file, only
> **17,923** restaurants carry REVIEWED edges; 18,879 is the count with reviews at scrape time.)

## Frozen results of record (unextended, best config per model)

| Model | dim | base NDCG@10 | base Recall@10 | +proximity NDCG@10 | proximity config |
|---|---|---|---|---|---|
| **InfoNCE-KGAT-SAL** (hybrid) | 2048 | 0.0588 | 0.1107 | **0.0735** | α=0.7, bw=20 km |
| **KGAT-SAL** | 2048 | 0.0602 | 0.1136 | **0.0734** | α=0.6, bw=10 km |
| **KGAT** | 2048 | 0.0658 | 0.1243 | 0.0727 | α=0.4, bw=5 km |
| RaDAR | 2048 | 0.0520 | 0.0987 | 0.0676 | α=0.7, bw=10 km |
| InfoNCE | 2048 | 0.0631 | 0.1230 | 0.0671 | α=0.4, bw=5 km |
| SimGCL-HRCL | 2048 | 0.0509 | 0.0966 | 0.0614 | α=0.3, bw=10 km |
| LightGCN | 2048 | 0.0517 | 0.1005 | 0.0561 | α=0.4, bw=5 km |
| HEK-CL | 512 | 0.0406 | 0.0788 | 0.0517 | α=0.3, bw=10 km |
| SimGCL | 2048 | 0.0536 | 0.1015 | 0.0478 | α=0.3, bw=10 km |
| IFL-GCL-KG | 2048 | 0.0319 | 0.0619 | 0.0448 | α=0.3, bw=10 km |
| SeqHybrid | 2048 | 0.0522 | 0.0986 | — | (no prox run) |
| UltraGCN | 2048 | 0.0492 | 0.0927 | — | (no prox run) |
| IFL-GCL | 2048 | 0.0333 | 0.0651 | — | (no prox run) |

Baselines (ALS `recommendation_baseline.py`, NeuMF `recommendation_ncf.py`, DGCF
`recommendation_dgcf.py`) exist in code but are **not** in the canonical CSV — confirm/log
their numbers if needed for the Part 3/4 leaderboard.

**Headline story:** proximity re-ranking lifts NDCG@10 by ~0.01 and precision@10 from ~0.012
to ~0.14 across every model. KGAT-family + spatial re-ranking is the winner (~0.073).

## Artifact-survival manifest (the consequential part)

The Apr 28 extended rerun **overwrote** the top-5 models' 2048 artifacts (same filenames).
No `.bak`/backups exist. Re-running at 2048-unextended is **not possible** (unextended
features + `reviews_flat` were also overwritten). For a blog series this is fine — we publish
the surviving metrics and anchor deep analysis on surviving artifacts.

| Artifact | Status |
|---|---|
| `results/model_results.csv` (71 canonical rows) | ✅ intact |
| `results/proximity_grid_search.csv` (756 rows) | ⚠️ **MIXED** — unextended + extended rows both present (May 3 append). Each model has two base NDCGs: unextended (KGAT 0.0652, KGAT-SAL 0.0602, hybrid 0.0588, RaDAR 0.0513, InfoNCE 0.0631/0.0594, LightGCN 0.0517) and extended (KGAT 0.0392, KGAT-SAL 0.0352, hybrid 0.0345, RaDAR 0.0263). **Filter to the unextended base NDCG per model** to recover the canonical α×bandwidth grid (no overwritten embeddings needed — the rows survive). |
| `results/training_checkpoints.csv` | ✅ intact (KGAT-SAL convergence curve) |
| Top-5 `.pt` checkpoints (kgat_all, kgat_sal, infonce_kgat_sal, radar, infonce) | ❌ overwritten → extended-1024 (Apr 28) |
| Top-5 embeddings + base predictions | ❌ overwritten → extended (Apr 28) |
| **KGAT-SAL** unextended predictions (`kgat_sal_all_T3_nospatial_prox_a30`, Apr 26 09:08) | ✅ intact |
| **KGAT** unextended predictions (`kgat_all_prox_a30`, Apr 25 23:58) | ✅ intact |
| Unextended predictions: lightgcn_3l, infonce_llm, hek_cl, ifl_gcl_kg | ✅ intact |
| Unextended checkpoints/embeddings: hek_cl, ifl_gcl(_kg), kgat_skip_llm, simgcl_hrcl, infonce_skip_llm | ✅ intact |

**→ Anchor the Part 4 per-user / per-segment deep-dive on KGAT-SAL** (co-best at 0.0734, and
its unextended predictions survive). Present InfoNCE-KGAT-SAL (0.0735) as the top *leaderboard*
line (metrics only — its per-user unextended artifacts are gone, and it's only +0.0001).

## Existing analysis assets (reusable for the article)

- `prediction_analysis.ipynb` (Apr 26, 13 sections) — model comparison, proximity analysis,
  per-location / activity / demographic / cuisine / cold-start / popularity-bias error analysis.
  **~80% of Part 4.** ✅ **FIXED (2026-06-06):** §4 repointed to the surviving unextended
  `kgat_sal_all_T3_nospatial_prox_a30_predictions.parquet` (131,632 rows, NDCG@10=0.0699, α=0.3 —
  the α=0.6 optimum was overwritten). §2 leaderboard filtered `emb_dim != 1024` (96→71 rows). §3 grid
  filtered to per-model max base_ndcg (756→168 rows). §12 baseline repointed to unextended
  `kgat_all_prox_a30` (now KGAT-SAL+prox vs KGAT+prox; its prose still says "no proximity" — update when
  drafting). All stale outputs cleared — **re-run the notebook** to regenerate canonical figures. Backup:
  `prediction_analysis.ipynb.bak`.

**Feature-usage finding (2026-06-06):** the canonical models trained with default `--llm-feat-mode fast`;
the prebuilt item parquet has **no semantic LLM columns** (authenticity/occasion/ambiance). So
`build_restaurant_llm_features.py`'s ~43 semantic attributes were **built but unused**; `item.llm` was
only 6 cheap analytics (skip-llm ablation ≈ 0.000 Δ). LLM signal that mattered: `item.nlp` (12),
`item.dietary` (5), `user.pref` (12), dish-derived diversity. Frame Part 2 accordingly.
- `review_demographics_analysis.ipynb` (Apr 22) — race/gender inference + rating fairness → Part 1/4.
- `foodie_revamp_analysis.ipynb` (Apr 11) — CBG selection, Census centroids, Places API, cost → Part 1.

## Open verification items (resolve during drafting)

1. `kgat.pt` (Apr 24, unextended) vs `kgat_all.pt` (Apr 28, extended) — naming ambiguity; confirm
   which is the canonical unextended KGAT checkpoint if a checkpoint is needed.
2. Exact train/test split + min-interaction user/item filter (not surfaced by grep of
   `build_training_features.py`; likely in `build_graph_data.py`) — needed for Part 1 dataset stats.
3. LLM serving stack for canonical features: `logs/vllm_*.log` is Apr 27 (extension era). Confirm
   whether unextended features were built with vLLM and/or llama.cpp for Part 2.
