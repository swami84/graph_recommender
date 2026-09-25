# Local archive index

Large or superseded artifacts are retained locally under `archive/`, which is
excluded from Git. The current experiment outputs remain under
`results/expanded_2026-09-21/`.

## Active archive sets

- `archive/pre_expansion_2026-09-25/`: pre-expansion model checkpoints,
  experiment results, GraphRAG remediation outputs, and duplicate external
  image-generation outputs.
- `archive/pre_publication_2026-09-09/`: legacy article drafts, figures,
  literature files, notebooks, model implementations, and result snapshots
  retained during the publication cleanup.
- `archive/pre_feature_contract_v2_20260909/`: feature-contract predecessor
  embeddings, models, predictions, and results.
- `archive/legacy_llm_pipeline_20260904/`: superseded LLM feature pipeline.
- `archive/graphrag_pilot_20260909/`: earlier GraphRAG pilot generations.
- `archive/demographics_pre_expansion_20260909/`: pre-expansion demographic
  inference output.
- `archive/remote_main_before_cleanup_2026-09-25.tar.gz`: exact snapshot of
  the GitHub `main` tree immediately before the repository cleanup.

The archive is intentionally local because it contains large generated files
and historical artifacts. It is not deleted and can be restored by moving the
required files back to their former paths.
