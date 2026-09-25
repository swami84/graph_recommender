# Python module layout

Run active project code from the repository root with Python's module form:

```bash
/home/swami/venv/dev_env/bin/python -m foodie.<category>.<module> [arguments]
```

This keeps imports stable regardless of a module's directory and ensures that
all relative data paths resolve from the project root.

## Categories

| Package | Responsibility |
|---|---|
| `foodie.collection` | CBG queue construction, Places collection, review scraping, cost controls, and dataset freezing |
| `foodie.features` | Canonical tables, deterministic and LLM feature generation, feature audits, and feature-contract assembly |
| `foodie.modeling` | Two-Tower, LightGCN, KGAT/KGAT-SAL, experiment orchestration, proximity reranking, and result summaries |
| `foodie.explanations` | GraphRAG generation, hit examples, human-audit preparation, and remediation analysis |
| `foodie.evaluation` | Publication result generation and rating-aware recommendation evaluation |

The unit tests, systemd units, shell monitoring tools, and internal subprocess
commands all use these package-qualified module names.

## Archived files reviewed on 2026-09-25

These files had no live imports, subprocess callers, systemd callers, or test
references. They are retained locally under
`archive/repo_cleanup_2026-09-25/`, which is excluded from Git.

| Former file | Decision |
|---|---|
| `build_dish_embeddings.py` | Superseded one-off local-vLLM dish profiler; current runs consume already-built dish artifacts |
| `check_parquet_quality.py` | One-off audit with hard-coded local paths; current audits live under `analysis/` and `tests/` |
| `ds_config.json` | Unused DeepSpeed ZeRO configuration; current model runners use native PyTorch |
| `ds_config_2acc.json` | Unused DeepSpeed gradient-accumulation variant |
| `expand_cbgs_50km.py` | Superseded by national-density, budgeted, and next-collection queue builders |
| `graphrag_recommend.py` | Superseded by `foodie.explanations.run_publication_graphrag`, which enforces split-safe evidence and auditing |
| `reconstruct_unextended_reviews.py` | Historical recovery tool for the overwritten pre-expansion interaction table |

The archive is recoverable locally but intentionally not published to GitHub.
