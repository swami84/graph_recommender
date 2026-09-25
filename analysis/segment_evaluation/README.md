# Final-model segment evaluation

These tables and figures evaluate the frozen KGAT-SAL + full LLM feature model after the frozen
proximity reranker (alpha 0.7, 5 km bandwidth, candidate depth 100). Hit@10 and NDCG@10 are
calculated for each of seeds 42, 43, and 44 and then averaged; error bars are the sample standard
deviation across those three seeds.

- Gender and race/ethnicity are probabilistic name-inference labels attached to the held-out user's
  review record. Unknown labels are excluded from the corresponding view. These plots are
  descriptive diagnostics, not a fairness certification and not measurements of self-reported
  identity.
- Cuisine is the corrected, supported cuisine label of the held-out target restaurant. The figure
  displays the fourteen most frequent supported target cuisines. Restaurants without supported
  cuisine evidence are excluded from this view only.
- Each user has one test target, so subgroup Hit@10 and NDCG@10 remain user-level ranking metrics.

Regenerate from the project root with:

```bash
export MPLCONFIGDIR=/tmp/foodie-mpl
/home/swami/venv/dev_env/bin/python analysis/evaluation/make_segment_performance.py
```
