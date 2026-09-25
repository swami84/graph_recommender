# Expanded dataset model comparison

The expanded run uses the same three models, two feature conditions, three seeds, hyperparameters, validation checkpoint selection, and metrics as the earlier frozen run.
The logical chronological split policy is unchanged. The expanded pipeline adds a stable review-ID tie-break for equal timestamps; therefore, the comparison is reproducible, but a small part of the delta may reflect this split-integrity correction rather than dataset expansion alone.

## Expanded results

| model           | condition   |   val_hit_at_10_mean |   val_hit_at_10_std |   val_ndcg_at_10_mean |   val_ndcg_at_10_std |   test_hit_at_10_mean |   test_hit_at_10_std |   test_ndcg_at_10_mean |   test_ndcg_at_10_std |
|:----------------|:------------|---------------------:|--------------------:|----------------------:|---------------------:|----------------------:|---------------------:|-----------------------:|----------------------:|
| FeatureTwoTower | full_llm    |             0.018667 |            0.000162 |              0.008958 |             0.000120 |              0.015304 |             0.000104 |               0.007441 |              0.000007 |
| FeatureTwoTower | non_llm     |             0.018126 |            0.000113 |              0.008723 |             0.000016 |              0.014870 |             0.000122 |               0.007265 |              0.000064 |
| KGAT-SAL        | full_llm    |             0.038023 |            0.000823 |              0.019119 |             0.000455 |              0.027536 |             0.000696 |               0.013669 |              0.000404 |
| KGAT-SAL        | non_llm     |             0.033845 |            0.000629 |              0.016918 |             0.000335 |              0.024751 |             0.000661 |               0.012336 |              0.000306 |
| LightGCN        | full_llm    |             0.029731 |            0.000296 |              0.014845 |             0.000095 |              0.021665 |             0.000118 |               0.010677 |              0.000079 |
| LightGCN        | non_llm     |             0.028054 |            0.000131 |              0.013936 |             0.000047 |              0.021013 |             0.000128 |               0.010337 |              0.000059 |

## Expanded minus earlier

| model           | condition   |   delta_test_hit_at_10_mean |   delta_test_ndcg_at_10_mean |   delta_val_hit_at_10_mean |   delta_val_ndcg_at_10_mean |
|:----------------|:------------|----------------------------:|-----------------------------:|---------------------------:|----------------------------:|
| FeatureTwoTower | full_llm    |                   -0.001184 |                    -0.000521 |                  -0.001549 |                   -0.000850 |
| FeatureTwoTower | non_llm     |                   -0.001470 |                    -0.000584 |                  -0.001736 |                   -0.000960 |
| KGAT-SAL        | full_llm    |                   -0.004464 |                    -0.002475 |                  -0.004896 |                   -0.002723 |
| KGAT-SAL        | non_llm     |                   -0.003872 |                    -0.002035 |                  -0.003567 |                   -0.001821 |
| LightGCN        | full_llm    |                   -0.005102 |                    -0.002707 |                  -0.006175 |                   -0.003141 |
| LightGCN        | non_llm     |                   -0.004551 |                    -0.002404 |                  -0.005766 |                   -0.003007 |
