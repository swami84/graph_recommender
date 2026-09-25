# Publication study design: LLM feature generation and GraphRAG recommendation

## Research question

Do LLM-derived structured semantics and dense semantic representations improve
full-catalog restaurant recommendation beyond conventional metadata and
behavioral features, and can graph-grounded retrieval turn the selected GNN's
ranked results into faithful, evidence-backed explanations?

The defensible claim is not that conventional NLP *cannot* produce these
signals. It is that approximating the same combined representation would
normally require several separately trained extractors, taxonomies, rules, and
aggregation stages, while one local LLM pipeline produces a unified schema.

## Core comparison

- Models: feature-aware Two-Tower (non-graph reference), LightGCN, KGAT-SAL.
- Feature conditions: conventional non-LLM; conventional + structured LLM +
  LLM embeddings.
- Seeds: 42, 43, 44.
- Primary metrics: raw full-catalog NDCG@10 and Hit@10.
- Split: per-user chronological leave-two-out; newest unique restaurant is test,
  second-newest is validation, older restaurants are training.
- Checkpoint selection: validation NDCG@10 only.
- Test exclusions: both training and validation interactions.
- Reporting: mean ± standard deviation across seeds and paired LLM uplift.

This is 18 core runs. KGAT-SAL was chosen over InfoNCE-KGAT-SAL before the new
experiment because it was consistently stronger in the earlier raw comparison;
the choice is therefore not based on the new test set.

## Focused feature decomposition

For the model selected by mean validation NDCG@10, compare:

1. conventional non-LLM features;
2. structured LLM features only;
3. LLM embeddings only;
4. both structured features and embeddings.

The core experiment already supplies conditions 1 and 4, leaving six additional
runs (conditions 2 and 3 across three seeds).

## Proximity stage

Apply proximity only to the selected model and feature condition. Select the
blend weight, distance bandwidth, and candidate depth on validation data, freeze
them, and evaluate once on test data. Report raw and proximity-filtered metrics
side by side; do not tune proximity on test results.

## GraphRAG explanation stage

The GNN remains the ranker. For sampled top-ranked recommendations, retrieve a
small evidence subgraph containing the user's training-only dining history,
restaurant metadata, cuisine/CBG relations, LLM preference and restaurant
attributes, and training-only review evidence. The LLM verbalizes only this
retrieved evidence and may use short review excerpts. Explanations are post-hoc
and must be described as grounded rationales, not causal accounts of the GNN.

Evaluate explanations separately for evidence entailment, unsupported claims,
personalization, usefulness, and quote fidelity. The held-out validation/test
reviews must never enter retrieval context.

## Execution

Core plan check (no training):

```bash
/home/swami/venv/dev_env/bin/python -m foodie.modeling.run_focused_llm_experiments --dry-run
```

Core training (resumes completed conditions automatically):

```bash
/home/swami/venv/dev_env/bin/python -m foodie.modeling.run_focused_llm_experiments
```

Core summary and validation winner:

```bash
/home/swami/venv/dev_env/bin/python -m foodie.modeling.summarize_publication_experiments
```

After selecting a winner (example shown for KGAT-SAL):

```bash
/home/swami/venv/dev_env/bin/python -m foodie.modeling.run_focused_llm_experiments \
  --models kgat_sal --conditions structured_only,embeddings_only
```
