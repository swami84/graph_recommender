# Frozen publication feature contract

Frozen at `2026-09-09T14:44:11.615977-04:00` under schema `publication-feature-contract-v2`.

- Users: **156,261** rows × **205** features.
- Restaurants: **75,203** rows × **262** features.
- Cuisine repair: **8,751** named and
  **6,996** unknown among 15,747 original Google `Other` rows.
- Demographic join: zero missing gender and race predictions in the canonical review table.
- Legacy audit: all **83** entries retained, superseded, or excluded with justification;
  zero unaccounted.
- Matrix checks: unique identifiers, no nulls, no infinities, no constant columns, and exact,
  non-overlapping feature-group coverage.

## Experimental conditions

- `non_llm`: excludes `llm`, `llm_embedding`, and `dish_llm` on both applicable sides;
  KG dish relations are disabled.
- `full_llm`: includes structured LLM features, LLM embeddings, training-only dish features,
  and training-only KG dish relations.
- Both conditions use identical publication train/validation/test interactions and spatial CBG
  edges. Proximity re-ranking is reserved for the validation-selected winner.

The machine-readable manifest with SHA-256 hashes is
`data/publication_features/feature_contract_freeze.json`.
