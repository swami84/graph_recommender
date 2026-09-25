# Auditing Post-Ranking GraphRAG Explanations for Provenance and Grounding

*An evaluation of training-safe evidence retrieval, citation provenance, personalization, and post-hoc explanation quality.*

**Series:** Part 4 of 4
**Previous:** [Part 3: Evaluating LLM-Derived Features, Graph Architectures, and Proximity Reranking](PART_3_MODEL_EVALUATION.md)

Parts 2 and 3 developed a recommendation pipeline that combines review-derived attributes, collaborative interactions, typed restaurant relations, and geographic reranking. This final part evaluates whether the selected recommendations can be accompanied by concise rationales grounded in inspectable evidence.

The explanation stage is intentionally separated from ranking. The Knowledge Graph Attention Network with Stability-Aware Learning (KGAT-SAL) scores the catalogue, the frozen proximity function reranks its top 100 candidates, and GraphRAG—graph-based retrieval-augmented generation—then assembles evidence for the selected restaurant. The explanation layer cannot change the recommendation order. Its output is therefore a grounded post-hoc rationale, not a reconstruction of the ranker's internal computation.

## Purpose and scope of the explanation layer

A ranked list provides limited information for evaluating whether a recommendation fits a diner's immediate circumstances. A cited rationale can identify supporting evidence, such as a shared cuisine or attribute, while also reporting relevant caveats from restaurant attributes or review text. The purpose of the layer is decision support: it makes the available evidence inspectable without claiming that the explanation caused the ranking.

The design also supports auditing. Each factual statement is expected to resolve to a supplied evidence node, and each review-derived node retains its source identifier. This allows the evaluation to test whether citations are valid, whether held-out reviews were excluded, and whether personalized statements are supported by an explicit relationship between the diner and restaurant.

Figure 1 contrasts the graph-based design with passage retrieval from a vector index. A vector index can retrieve semantically similar review passages, but it does not natively represent a diner-to-restaurant relationship. It also requires additional record-level controls to exclude held-out reviews and identify the entity described by each passage. In the graph representation, these properties are encoded directly through typed nodes, edges, entity identifiers, and review identifiers.

![Two-column architecture comparison. Left, chunk-and-embed retrieval: a review corpus split into chunks and embedded into vectors, the nearest to a query highlighted as top-k. Right, typed graph retrieval: a chain of nodes from diner through a visited restaurant and a cuisine node to the recommendation, with attribute and review nodes hanging off it and a dashed derived match edge arcing between diner and recommendation. A five-row table underneath contrasts retrieval unit, selection mechanism, the diner-to-restaurant link, held-out exclusion, and the provenance of a claim.](../figures/graphrag_vs_vector_rag.png)

*Figure 1. Passage-based and graph-based retrieval for the explanation task. Both supply evidence to a language model, but the graph representation explicitly records relationships, source identities, and the entity associated with each fact.*

Graph retrieval is also consistent with the ranking architecture used in this study. KGAT-SAL already operates over typed relations among diners, restaurants, cuisine categories, price tiers, dishes, and Census block groups. The explanation layer retrieves a subgraph anchored on the selected diner–restaurant pair rather than maintaining a separate representation of the same evidence.

## Graph-based evidence architecture

For each selected recommendation, the retriever assembles a compact evidence subgraph containing:

- restaurants in the diner's training history and their observed ratings;
- diner and restaurant attributes derived from training-period reviews;
- restaurant metadata and Census block-group location;
- short review excerpts whose identifiers occur in neither the validation nor test split; and
- an explicit match node when the diner and restaurant share a supported attribute or when the restaurant shares a cuisine category with a previously highly rated restaurant.

Each node records an identifier, type, associated entity, source, and natural-language fact. The generator must cite factual claims using these identifiers and cannot access private profile fields, held-out review text, or restaurants outside the selected result. The match node supports personalization because it records a relationship between the diner and recommendation rather than a property of only one entity.

Across the 100 audited cases, evidence bundles contain a mean of 14.3 nodes, ranging from 8 to 19. Generated rationales average 262 characters; 97 contain two sentences and 93 use exactly two inline citations. The generator therefore compresses approximately fourteen retrieved facts into two sentences, making evidence selection an important part of the explanation process.

Figure 2 presents one audited case. A diner with three five-star training interactions at sandwich or comfort-food restaurants receives Ru San's Kennesaw, a sushi restaurant. No cuisine relationship connects the history to the recommendation. Instead, a match node links a diner value preference of 0.8 with a restaurant value attribute of 0.8. The bundle also contains a restaurant healthiness attribute of 0.9.

![Three-stage schematic. Stage one shows KGAT-SAL scoring, proximity reranking, and the selected restaurant. Stage two shows diner evidence, a shared-attribute match node, and restaurant evidence inside a training-safe retrieval panel. A middle band breaks the match node into its identifier, kind, entity, source, and fact fields. Stage three shows the cited rationale beside an audit card listing six judge outcomes, all at the top of the scale.](../figures/graphrag_explanation_example.png)

*Figure 2. Post-ranking explanation for one audited recommendation. Evidence is assembled after restaurant selection and therefore does not affect the ranking.*

The generator produced:

> The evidence indicates a shared attribute where both the diner's preference for value and Ru San's Kennesaw's corresponding value attribute are on the high end [M1]. Additionally, Ru San's Kennesaw has a high training-review-derived healthiness attribute of 0.9 [A1].

The first statement resolves to the match node and the second to a restaurant attribute derived from training reviews. This example illustrates the intended function of the layer: converting a selected recommendation into a short rationale with identifiable supporting evidence.

## Discussion

The evaluation includes 100 deterministic top-1 recommendations from the seed-42 frozen proximity output, with 25 diners sampled from each activity quartile. A 27-billion-parameter language model generated the rationales and scored them in a separate pass. Deterministic provenance checks were evaluated independently of the model-judged measures.

Within this sample, every inline citation resolves to a supplied evidence node, every explanation cites restaurant evidence, and no bundle contains a validation- or test-review identifier. A supported diner-to-restaurant match is available in 79 cases and is cited in all 79. These results show that the implemented provenance and temporal-exclusion controls operate as intended in the audited cases.

Figure 3 summarizes the model-judged results. Mean scores are 4.970 for entailment, 4.950 for citation correctness, 4.900 for completeness, 4.290 for usefulness, and 3.670 for personalization. One explanation contains one unsupported claim. These scores indicate strong support under the applied rubric, although the concentration of entailment and citation scores near the maximum limits their ability to distinguish small differences in quality.

![Two panels. Left: stacked horizontal bars giving the score distribution for each of five judged dimensions across 100 explanations, with entailment, citation correctness and completeness concentrated at 5 and personalization spread across the scale. Right: grouped bars comparing mean scores for the 21 cases with no available diner match against the 79 cases with one.](../figures/graphrag_quality_profile.png)

*Figure 3. Model-judged explanation quality across 100 audited cases and stratified by the availability of a supported diner-to-restaurant match.*

Personalization depends on whether the retriever identifies a supported relationship between the diner and restaurant. Among the 79 matched cases, mean personalization is 4.380 and usefulness is 4.557. In the 21 cases without a match, the corresponding scores are 1.000 and 3.286, while entailment remains similar. The layer can therefore provide a supported restaurant description when no match is available, but it has limited evidence for explaining why that restaurant is relevant to the individual diner.

The explanations also include potentially decision-relevant caveats. Contrastive wording appears in 27 cases, and manual review identifies a substantive drawback in 22, including price, waiting time, noise, crowding, cleanliness, menu changes, and parking conditions. This is a descriptive review of the generated text rather than a predefined audit measure.

Because matched and unmatched diners differ in activity and evidence-bundle size, a paired ablation regenerated the 79 matched cases after removing only the match node. Withholding this relationship reduced mean usefulness by 1.418 points and personalization by 3.367 points. Changes in entailment, citation correctness, and completeness remained close to zero, with bootstrap intervals spanning zero. Within this audit, the explicit relationship therefore improved perceived relevance without a measurable reduction in model-judged grounding.

The current evidence is limited by the 100-case, single-seed audit, sparse interaction histories for unmatched diners, and use of the same language model for generation and evaluation. A larger and more diverse evaluation sample would provide more precise estimates, while additional review and interaction data could increase diner-side feature coverage. A larger independent model may provide a more discriminating quality assessment and potentially improve generation, but this should be tested against blinded human review rather than assumed from model size alone. The layer remains a post-hoc evidence mechanism and does not reveal the internal cause of KGAT-SAL's ranking decision.

## Key findings

- Ranking and explanation remain separate: KGAT-SAL and proximity reranking select the restaurant before GraphRAG retrieves evidence.
- In the 100-case audit, every inline citation resolves and no evidence bundle contains a validation- or test-review identifier.
- A supported diner-to-restaurant match is available in 79% of cases and is cited whenever available.
- Model-judged entailment, citation correctness, and completeness are high, but ceiling effects and use of the same model for generation and judging limit interpretation.
- In the paired ablation, removing the match node reduces usefulness by 1.418 points and personalization by 3.367 points, with little change in grounding measures.
- The explanation layer can report caveats as well as supporting evidence, but match-rule calibration, broader sampling, and blinded human evaluation remain necessary.
- The resulting explanations are evidence-grounded post-hoc rationales and should not be interpreted as traces of the ranker's internal reasoning.

**Series navigation:** [Introduction](PART_0_SERIES_INTRODUCTION.md) · [Part 1](PART_1_DATA_COLLECTION_AND_EDA.md) · [Part 2](PART_2_LLM_FEATURE_ENGINEERING.md) · [Part 3](PART_3_MODEL_EVALUATION.md)
