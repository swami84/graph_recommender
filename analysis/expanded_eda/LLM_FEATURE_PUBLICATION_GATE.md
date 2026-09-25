# LLM feature publication gate

The old cuisine one-hots were rejected because their closed taxonomy forced invalid labels.
They are replaced by the v4 evidence-rule + constrained-27B repair (15,747/15,747
target restaurants currently available). Old user cuisine preferences are excluded because
they share the rejected taxonomy.

The old vegan/vegetarian availability scores also failed: absence of a mention had often been
treated as evidence of absence. They are replaced with conservative positive-only indicators.

Spice, outdoor seating, value, service sentiment, and service speed are retained only when the
source types/reviews contain direct lexical evidence. Other non-cuisine structured fields remain
temporarily retained, as requested, pending later article audit.

| feature           | decision                            |   known_before |   known_after |   evidence_share_known |   evidence_share_high |   evidence_share_low |
|:------------------|:------------------------------------|---------------:|--------------:|-----------------------:|----------------------:|---------------------:|
| spice             | retain_with_evidence_gate           |       7588.000 |          6326 |                  0.834 |                 0.839 |                0.295 |
| outdoor           | retain_with_evidence_gate           |       4592.000 |          2951 |                  0.643 |                 0.760 |                0.053 |
| value             | retain_with_evidence_gate           |      57619.000 |         46995 |                  0.816 |                 0.814 |                0.687 |
| service_sentiment | retain_with_evidence_gate           |      33520.000 |         31471 |                  0.939 |                 0.952 |                0.710 |
| service_speed     | retain_with_evidence_gate           |      23736.000 |         20185 |                  0.850 |                 0.827 |                0.830 |
| offers_vegan      | replace_with_positive_evidence_only |      10855.000 |          2675 |                  0.112 |               nan     |              nan     |
| offers_vegetarian | replace_with_positive_evidence_only |      10678.000 |          4435 |                  0.114 |               nan     |              nan     |
| has_bar           | restore_as_positive_evidence_only   |        nan     |         13084 |                nan     |               nan     |              nan     |
| byob              | restore_as_positive_evidence_only   |        nan     |           475 |                nan     |               nan     |              nan     |
