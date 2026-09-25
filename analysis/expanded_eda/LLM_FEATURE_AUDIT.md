# Structured LLM feature audit (superseded diagnostic)

> This document records the failed pre-repair cuisine audit. Its `Southern`
> findings motivated the targeted 27B rebuild and must not be treated as the
> current distribution. See `FEATURE_PROVENANCE_AUDIT.md` and `EDA_REPORT.md`
> for the corrected, current results.

The restaurant feature population contains 68,918 rows. NY provides
24.5%, CA 17.9%, and other states
57.7%. These are comparison baselines, not
targets that every semantic feature should reproduce.

## Cuisine failure

The current cuisine output is not publication-ready. `Southern` is selected for
12,471 restaurants
(18.1%), and 7,934
(11.5%) raw arrays repeat labels despite the
unique-item schema. `Southern` by region:

| feature          | region       |   region_rows |   selected_rows |   within_region_rate |   global_rate |   rate_ratio_to_global |   share_of_positive_rows |   baseline_catalogue_share |
|:-----------------|:-------------|--------------:|----------------:|---------------------:|--------------:|-----------------------:|-------------------------:|---------------------------:|
| cuisine_southern | CA           |         12308 |            2300 |             0.18687  |      0.180954 |               1.03269  |                 0.184428 |                   0.178589 |
| cuisine_southern | NY           |         16871 |            2778 |             0.164661 |      0.180954 |               0.909961 |                 0.222757 |                   0.244798 |
| cuisine_southern | Other states |         39739 |            7393 |             0.186039 |      0.180954 |               1.0281   |                 0.592815 |                   0.576613 |

## Restaurant structured-feature coverage

Lowest known/evidence coverage:

| feature             |   rows |   known_rows |   known_share |   mean_when_known |   std_when_known |   p05_when_known |   median_when_known |   p95_when_known |   zero_share_when_known |   half_share_when_known |   one_share_when_known |
|:--------------------|-------:|-------------:|--------------:|------------------:|-----------------:|-----------------:|--------------------:|-----------------:|------------------------:|------------------------:|-----------------------:|
| nightlife           |  68918 |           70 |    0.0010157  |          0.981429 |        0.0766941 |            0.845 |                 1   |                1 |              0          |              0.0142857  |               0.928571 |
| richness            |  68918 |          264 |    0.00383064 |          0.924242 |        0.106444  |            0.7   |                 1   |                1 |              0          |              0.00378788 |               0.609848 |
| reservation         |  68918 |          300 |    0.004353   |          0.652333 |        0.430634  |            0     |                 1   |                1 |              0.273333   |              0.0833333  |               0.523333 |
| business            |  68918 |          515 |    0.00747265 |          0.577282 |        0.356114  |            0     |                 0.5 |                1 |              0.201942   |              0.347573   |               0.295146 |
| poultry             |  68918 |          587 |    0.00851737 |          0.982112 |        0.0727587 |            0.9   |                 1   |                1 |              0.00170358 |              0.00511073 |               0.908007 |
| plant_based         |  68918 |          769 |    0.0111582  |          0.891027 |        0.259025  |            0     |                 1   |                1 |              0.0585176  |              0.0546164  |               0.784135 |
| seafood             |  68918 |         1296 |    0.018805   |          0.66412  |        0.416273  |            0     |                 1   |                1 |              0.24537    |              0.0949074  |               0.510031 |
| local_independent   |  68918 |         1504 |    0.021823   |          0.96988  |        0.128261  |            0.9   |                 1   |                1 |              0.012633   |              0.00864362 |               0.886303 |
| review_disagreement |  68918 |         1717 |    0.0249137  |          0.654426 |        0.312004  |            0     |                 0.5 |                1 |              0.101922   |              0.404193   |               0.355853 |
| parking             |  68918 |         2495 |    0.0362024  |          0.627221 |        0.340893  |            0     |                 0.6 |                1 |              0.146693   |              0.314228   |               0.341884 |

Largest concentration at the neutral value 0.5 among supposedly known values:

| feature             |   rows |   known_rows |   known_share |   mean_when_known |   std_when_known |   p05_when_known |   median_when_known |   p95_when_known |   zero_share_when_known |   half_share_when_known |   one_share_when_known |
|:--------------------|-------:|-------------:|--------------:|------------------:|-----------------:|-----------------:|--------------------:|-----------------:|------------------------:|------------------------:|-----------------------:|
| wait                |  68918 |        19672 |    0.285441   |          0.590602 |         0.18607  |              0.5 |                 0.5 |                1 |               0.0336011 |                0.469042 |              0.0719805 |
| formality           |  68918 |         9875 |    0.143286   |          0.460911 |         0.316049 |              0   |                 0.5 |                1 |               0.233722  |                0.424709 |              0.137418  |
| review_disagreement |  68918 |         1717 |    0.0249137  |          0.654426 |         0.312004 |              0   |                 0.5 |                1 |               0.101922  |                0.404193 |              0.355853  |
| noise               |  68918 |        28160 |    0.408602   |          0.626117 |         0.232973 |              0   |                 0.6 |                1 |               0.0591619 |                0.380717 |              0.130114  |
| order_accuracy      |  68918 |         4706 |    0.068284   |          0.424341 |         0.309492 |              0   |                 0.5 |                1 |               0.287718  |                0.351041 |              0.078198  |
| business            |  68918 |          515 |    0.00747265 |          0.577282 |         0.356114 |              0   |                 0.5 |                1 |               0.201942  |                0.347573 |              0.295146  |
| parking             |  68918 |         2495 |    0.0362024  |          0.627221 |         0.340893 |              0   |                 0.6 |                1 |               0.146693  |                0.314228 |              0.341884  |
| offers_gluten_free  |  68918 |         5686 |    0.0825038  |          0.580056 |         0.402271 |              0   |                 0.5 |                1 |               0.262399  |                0.288779 |              0.395005  |
| transit             |  68918 |         4361 |    0.0632781  |          0.631025 |         0.362216 |              0   |                 0.6 |                1 |               0.182527  |                0.279982 |              0.370328  |
| sweetness           |  68918 |         6896 |    0.100061   |          0.620164 |         0.29265  |              0   |                 0.6 |                1 |               0.109919  |                0.267546 |              0.224478  |

## User structured-feature coverage

Lowest known/evidence coverage:

| feature                       |   rows |   known_rows |   known_share |   mean_when_known |   std_when_known |   p05_when_known |   median_when_known |   p95_when_known |   zero_share_when_known |   half_share_when_known |   one_share_when_known |
|:------------------------------|-------:|-------------:|--------------:|------------------:|-----------------:|-----------------:|--------------------:|-----------------:|------------------------:|------------------------:|-----------------------:|
| restriction_kosher            | 150767 |           46 |   0.000305107 |          0.180435 |         0.376899 |              0   |                   0 |                1 |               0.804348  |              0.0217391  |               0.152174 |
| restriction_halal             | 150767 |          138 |   0.00091532  |          0.791304 |         0.403358 |              0   |                   1 |                1 |               0.202899  |              0.00724638 |               0.775362 |
| restriction_gluten_free       | 150767 |          230 |   0.00152553  |          0.813478 |         0.347217 |              0   |                   1 |                1 |               0.130435  |              0.0826087  |               0.713043 |
| nightlife                     | 150767 |          261 |   0.00173115  |          0.963985 |         0.161024 |              0.8 |                   1 |                1 |               0.0191571 |              0.0306513  |               0.942529 |
| business                      | 150767 |          289 |   0.00191687  |          0.797578 |         0.36963  |              0   |                   1 |                1 |               0.15917   |              0.0519031  |               0.719723 |
| restriction_vegetarian        | 150767 |          334 |   0.00221534  |          0.620659 |         0.439995 |              0   |                   1 |                1 |               0.302395  |              0.122754   |               0.502994 |
| restriction_vegan             | 150767 |          335 |   0.00222197  |          0.629254 |         0.442141 |              0   |                   1 |                1 |               0.301493  |              0.110448   |               0.513433 |
| review_preference_consistency | 150767 |         1147 |   0.00760777  |          0.758675 |         0.370323 |              0   |                   1 |                1 |               0.1517    |              0.168265   |               0.657367 |
| poultry                       | 150767 |         1550 |   0.0102808   |          0.892129 |         0.250208 |              0   |                   1 |                1 |               0.0522581 |              0.0709677  |               0.769032 |
| accessibility                 | 150767 |         1767 |   0.0117201   |          0.629598 |         0.434629 |              0   |                   1 |                1 |               0.290323  |              0.102999   |               0.508206 |

Largest concentration at 0.5 among supposedly known values:

| feature       |   rows |   known_rows |   known_share |   mean_when_known |   std_when_known |   p05_when_known |   median_when_known |   p95_when_known |   zero_share_when_known |   half_share_when_known |   one_share_when_known |
|:--------------|-------:|-------------:|--------------:|------------------:|-----------------:|-----------------:|--------------------:|-----------------:|------------------------:|------------------------:|-----------------------:|
| noise         | 150767 |        60178 |     0.399146  |          0.498838 |         0.276294 |                0 |                0.5  |                1 |               0.184486  |                0.421383 |              0.0729336 |
| wait          | 150767 |        37290 |     0.247335  |          0.528709 |         0.314226 |                0 |                0.5  |                1 |               0.189193  |                0.393886 |              0.173827  |
| formality     | 150767 |        26700 |     0.177094  |          0.63959  |         0.302164 |                0 |                0.6  |                1 |               0.110524  |                0.349213 |              0.265131  |
| solo          | 150767 |        19627 |     0.130181  |          0.51835  |         0.346966 |                0 |                0.5  |                1 |               0.226066  |                0.340093 |              0.239007  |
| sweetness     | 150767 |         4673 |     0.0309948 |          0.553013 |         0.294461 |                0 |                0.5  |                1 |               0.146159  |                0.321849 |              0.165632  |
| portion_size  | 150767 |        63908 |     0.423886  |          0.697996 |         0.292261 |                0 |                0.75 |                1 |               0.0838393 |                0.253349 |              0.332165  |
| service_speed | 150767 |        83774 |     0.555652  |          0.692478 |         0.317244 |                0 |                0.8  |                1 |               0.107599  |                0.235694 |              0.369255  |
| red_meat      | 150767 |         1905 |     0.0126354 |          0.706588 |         0.342208 |                0 |                0.8  |                1 |               0.122835  |                0.220472 |              0.462467  |
| romantic      | 150767 |        34922 |     0.231629  |          0.420673 |         0.387129 |                0 |                0.5  |                1 |               0.391215  |                0.21717  |              0.200218  |
| parking       | 150767 |         3567 |     0.023659  |          0.589179 |         0.418359 |                0 |                0.5  |                1 |               0.277264  |                0.214186 |              0.437903  |

## Geographic representation checks

Among features with at least 500 known restaurant rows, the largest deviations
in where supporting evidence occurs are:

| feature              | region   |   region_rows |   known_rows |   known_share |   mean_when_known |   share_of_known_rows |   baseline_region_share |   representation_ratio |   absolute_representation_deviation |
|:---------------------|:---------|--------------:|-------------:|--------------:|------------------:|----------------------:|------------------------:|-----------------------:|------------------------------------:|
| review_disagreement  | NY       |         16871 |          606 |    0.0359196  |          0.629538 |              0.352941 |                0.244798 |               1.44176  |                            0.441764 |
| parking              | CA       |         12308 |          627 |    0.0509425  |          0.650505 |              0.251303 |                0.178589 |               1.40716  |                            0.407156 |
| parking              | NY       |         16871 |          370 |    0.0219311  |          0.575946 |              0.148297 |                0.244798 |               0.605791 |                            0.394209 |
| poultry              | NY       |         16871 |          191 |    0.0113212  |          0.983246 |              0.325383 |                0.244798 |               1.32919  |                            0.32919  |
| review_disagreement  | CA       |         12308 |          224 |    0.0181995  |          0.682589 |              0.13046  |                0.178589 |               0.730505 |                            0.269495 |
| order_accuracy       | CA       |         12308 |          617 |    0.05013    |          0.472609 |              0.131109 |                0.178589 |               0.734139 |                            0.265861 |
| outdoor              | NY       |         16871 |          850 |    0.0503823  |          0.788412 |              0.185105 |                0.244798 |               0.756152 |                            0.243848 |
| offers_halal         | NY       |         16871 |         2151 |    0.127497   |          0.357741 |              0.303171 |                0.244798 |               1.23845  |                            0.238454 |
| order_accuracy       | NY       |         16871 |         1419 |    0.0841088  |          0.368851 |              0.30153  |                0.244798 |               1.23175  |                            0.231749 |
| business             | NY       |         16871 |           97 |    0.00574951 |          0.545361 |              0.18835  |                0.244798 |               0.769407 |                            0.230593 |
| offers_gluten_free   | NY       |         16871 |         1092 |    0.0647265  |          0.540018 |              0.192051 |                0.244798 |               0.784527 |                            0.215473 |
| atmosphere_sentiment | CA       |         12308 |         1082 |    0.0879103  |          0.956747 |              0.142537 |                0.178589 |               0.79813  |                            0.20187  |

Most high-coverage features broadly track the 24.5% NY / 17.9% CA / 57.7%
other-state feature-row baseline. The `Southern` positives also nearly track
that baseline (22.3% NY, 18.4% CA, 59.3% elsewhere), which is further evidence
of a generic extraction artifact rather than a credible regional cuisine signal.

Confidence is also poorly calibrated: restaurant cuisine confidence equals
1.0 for 78.5% of rows even though the cuisine audit exposes widespread false
positives and schema violations. Several supposedly known scalar features are
heavily concentrated at exactly 0.5, while rare fields have too little coverage
to contribute reliably without their evidence masks.

## Interpretation

State contribution should broadly resemble the catalogue only for ubiquitous
features. Cuisine, meal period, dietary accommodation, atmosphere, and service
features can legitimately differ by state. Large NY/CA-versus-rest deviations
are therefore flags for review, not automatic evidence of error. Detailed
regional tables and confidence diagnostics are in this directory.
