# Publication proximity reranking

Hyperparameters were selected exclusively by mean validation NDCG@10 across seeds 42, 43, and 44. User centroids use training interactions only. Test ranking excludes both training and validation restaurants.

## Frozen configuration

- Blend weight (alpha): 0.6
- Distance bandwidth: 2.5 km
- Candidate depth: 1600
- Validation Hit@10: 0.060619 ± 0.000250
- Validation NDCG@10: 0.029759 ± 0.000188

## One-time test evaluation

|      seed |   hit_at_10 |   ndcg_at_10 |   test_hit_at_10 |   test_ndcg_at_10 |   delta_hit_at_10 |   delta_ndcg_at_10 |
|----------:|------------:|-------------:|-----------------:|------------------:|------------------:|-------------------:|
| 42.000000 |    0.046043 |     0.022130 |         0.027903 |          0.013907 |          0.018141 |           0.008223 |
| 43.000000 |    0.046282 |     0.022165 |         0.027972 |          0.013897 |          0.018310 |           0.008268 |
| 44.000000 |    0.045578 |     0.021891 |         0.026733 |          0.013202 |          0.018845 |           0.008689 |

Proximity Hit@10: **0.045968 ± 0.000358**
Proximity NDCG@10: **0.022062 ± 0.000149**
Mean Hit@10 change: **+0.018432**
Mean NDCG@10 change: **+0.008393**
