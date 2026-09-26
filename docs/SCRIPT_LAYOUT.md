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
