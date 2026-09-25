# From raw reviews to explained recommendations

## Editorial status

This is the handoff-ready master draft for a four-part technical series. Bracketed figure callouts
refer to completed assets under `article/figures/`. The prose deliberately distinguishes measured
results from hypotheses and avoids claiming that conventional NLP is incapable of extracting the
same signals.

---

# Part 1 — Building a national restaurant evidence base

## The recommendation problem starts before model training

A restaurant recommender can only learn from the places and people represented in its data. We
therefore constructed a combined sampling frame covering census block groups (CBGs) with high foot
traffic, the 10,000 highest population densities with national safeguards, and surrounding areas.
The design represents both active commercial districts and dense residential or mixed-use areas
without treating either signal as a complete proxy for restaurant availability.

Each CBG polygon was covered with H3 resolution-9 search cells. Google Places API (New) Nearby
Search supplied restaurant identity and metadata; Place ID was the deduplication key. A completed
empty search was checkpointed just like a successful restaurant-bearing search, which prevented
large or sparsely commercial CBGs from being queried repeatedly.

The resulting frozen catalogue contains 75,203 unique restaurants across 49 state/DC jurisdictions.
New York and California remain the largest components,
followed by Texas, Florida, and New Jersey. This is broad national coverage, not a population-
representative sample: venue visibility still depends on the discovery strategy and Google Maps.

> **Coverage map.** Insert `collection_coverage_map.png` before the catalogue summary. It shows the
> combined searched-CBG footprint rather than a sequence of collection phases.

> **Figure 1 — Catalogue coverage.** Insert `eda_catalogue_coverage.png`.
>
> Suggested caption: *The expanded catalogue covers most US jurisdictions but remains anchored in
> the largest urban restaurant markets. Cuisine shares use only venues with supported specific
> labels; 6,996 insufficient-evidence venues remain in the dataset but are omitted from that panel.*

## Collecting reviews without confusing failure with absence

Public Google Maps review pages were collected with Camoufox. Stable browser contexts were reused,
while a fresh page was opened for each restaurant. Browser starts were staggered, globally rate
limited, and separated by randomized pauses. Pagination scrolling was deliberately paced.

These controls were not merely about throughput. During collection, a missing review panel could
mean that a restaurant truly had no reviews, that the page had not scrolled far enough to render
the panel, that connectivity had failed, or that Google had presented an automated-traffic
challenge. A known-positive probe and retryable status classes prevented those cases from being
collapsed into a false “no reviews” outcome.

The scraper freeze contains 5,545,260 raw review objects. Canonical processing retained 5,544,947
unique analytical review rows from 3,930,034 identified reviewers. The 313-row difference—0.006%
of raw objects—comes from identity and flattening requirements, not unfinished scraping. Of the
75,203 restaurants, 70,625 retain at least one analytical review.

## What the review corpus looks like

The corpus is strongly positive: 63.2% of reviews have five stars and another 12.5% have four.
One- and two-star reviews account for 17.8%. Reviewer activity is even more skewed: 82.0% of
reviewers appear at only one restaurant, while 156,261 users have the four or more unique
restaurants required by the publication experiment.

> **Figure 2 — Ratings and activity.** Insert `eda_ratings_and_activity.png`.

The modeling subset is not simply a smaller copy of the corpus. Its 156,261 diners contribute
19.6% of valid review rows, are more geographically concentrated in New York and California, and
rate differently from the full reviewer population. Their cuisine breadth also determines how
much cross-cuisine preference evidence is available for feature generation.

> **Figure 3 — Modeling-user profile.** Insert `eda_active_reviewers.png`.

Review content also changes with sentiment. Two-star reviews average 406 characters and one-star
reviews 386, compared with 223 characters for five-star reviews. Attached photos move in the
opposite direction: four-star reviews average 0.96 photos and five-star reviews 0.81, versus 0.24
for one-star reviews. An LLM feature extractor therefore sees different evidence regimes: detailed
complaints at the negative end and visually documented experiences at the positive end.

> **Figure 4 — Review richness.** Insert `eda_review_richness.png`.

## Cuisine without forcing unsupported labels

Google’s broad type mapping left 15,747 restaurants under a generic `Other` label. A targeted 27B
local model revisited only those rows using restaurant names, types, summaries, and review
evidence. It assigned 8,751 supported specific labels and retained 6,996 as insufficient evidence.

For publication EDA, the generic category is removed from cuisine plots—not from the dataset.
Clear label synonyms such as `mexican` and `latin_american`, or `pizza` and `italian`, are combined
into stable display categories. Distinct supported cuisines such as French, Ethiopian, Filipino,
Korean, and Cajun/Creole remain distinct. Cuisine percentages therefore use classified venues as
their denominator, while state totals and model training retain every eligible row.

> **Figure 5 — State and cuisine distribution.** Insert `eda_state.png`.

Name-inferred demographic labels provide a descriptive diagnostic—not self-reported identity.
Within that limitation, the data allow two separate questions: whether mean ratings differ across
cuisine and inferred groups, and whether the composition of reviews differs across cuisines. These
views should not be interpreted as population estimates or used for sensitive targeting.

> **Figure 6 — Mean ratings by inferred gender and cuisine.** Insert `eda_rating_gender.png`.
>
> **Figure 7 — Mean ratings by inferred race/ethnicity and cuisine.** Insert
> `eda_race_cuisine_heatmap.png`.
>
> **Figure 8 — Review composition by cuisine.** Insert `eda_composition_by_cuisine.png`.

## From EDA to a leakage-safe experiment

The model treats each unique user–restaurant pair as one interaction. For every reviewer with at
least four restaurants, the newest visit is held out for test, the second-newest for validation,
and all older visits form training. This yields 772,170 training interactions, 156,261 validation
targets, and 156,261 test targets. The same split governs behavioral aggregates, LLM corpora, graph
edges, model selection, and explanation evidence.

That boundary is the bridge to Part 2. Once review text becomes a model feature, ordinary train/test
separation is no longer enough: the text used to construct the feature must also respect the split.

---

# Part 2 — Using an LLM as a unified feature engineer

## The useful comparison is not “LLM versus impossible”

Restaurant reviews contain signals that star ratings and place types do not express cleanly:
spice tolerance, perceived value, portion size, noise, formality, date-night suitability, family
fit, outdoor seating, wait expectations, service speed, dietary accommodation, and repeat-visit
intent. Conventional NLP can extract many of these. But reproducing the complete representation
would usually require a collection of separate classifiers, lexicons, weak-supervision rules,
taxonomies, aggregation logic, and missing-evidence policies.

The more defensible question is whether one local LLM pipeline can build this heterogeneous feature
contract efficiently enough to improve recommendation—and whether every architecture can exploit
the result equally well.

## Two representations from the same training-safe evidence

The pipeline produces structured features and dense embeddings for both users and restaurants.
Structured fields cover five broad families:

- taste and texture, including spice, sweetness, richness, healthiness, novelty, and portions;
- atmosphere and occasion, including noise, formality, romantic, family, group, solo, business,
  outdoor, and nightlife suitability;
- operations, including waits, service speed, reservations, takeout, parking, transit,
  accessibility, consistency, and value;
- dietary and protein evidence, including vegan, vegetarian, gluten-free, halal, kosher, seafood,
  red meat, poultry, and plant-based signals; and
- cuisine and meal-period distributions.

Restaurant features additionally represent sentiment toward food, service, atmosphere, value,
cleanliness, order accuracy, authenticity, review disagreement, and repeat-visit intent. User
features represent preferences and restrictions inferred from the diner’s own training history.

Every sparse field is paired with a `known` indicator or evidence-strength signal. This matters.
“No evidence that a restaurant offers gluten-free accommodation” is not the same claim as “the
restaurant does not accommodate gluten-free diners.” Encoding missing evidence separately avoids
turning model uncertainty into a false negative.

Dense 64-dimensional text embeddings preserve signals that do not fit neatly into the schema.
Training-only dish mentions also enter the full LLM condition as features and knowledge-graph
relations. The frozen matrices contain 205 user features for 156,261 eligible users and 262 item
features for all 75,203 restaurants.

> **Suggested schematic — Feature contract.** A two-column tree with User and Restaurant roots.
> Show conventional features in blue and LLM-derived structured fields, embeddings, and dish
> relations in amber. Pair several structured fields with small “known?” shields to emphasize
> missing-evidence handling. Do not claim that all 205/262 dimensions are LLM-generated.

## Leakage prevention has to operate at the source-review level

Features derived from a held-out review leak more than a label. They can expose the target
restaurant, sentiment, dishes, and preferences that the model is supposed to predict. The LLM
corpora were therefore built after the chronological split. User profiles use training reviews
only. Restaurant corpora also exclude validation and test review text.

The source-ID audit compared 1,048,459 review IDs used by the structured LLM features against all
312,522 validation and test review IDs and found zero overlap. GraphRAG later uses the same safety
boundary. This source-level audit is stronger than checking only that feature-table rows are
separate, because a restaurant row can otherwise aggregate text from multiple split periods.

## The asymmetric-benefit hypothesis

A feature-aware Two-Tower model can compare user and restaurant vectors directly, but it cannot
propagate a semantic clue through neighboring users, restaurants, cuisines, dishes, time periods,
and locations. LightGCN can propagate collaborative structure, but it uses fewer typed semantic
relations. KGAT-SAL combines collaborative propagation with restaurant knowledge relations,
training-only dish edges, temporal periods, self-augmented learning, and spatial CBG edges.

This creates the central hypothesis for Part 3: the same LLM representation should provide little
benefit when treated as isolated side information and more benefit when an architecture can route
it through relevant graph structure.

---

# Part 3 — Semantic features help asymmetrically; geography helps differently

## A controlled three-model comparison

We compare a feature-aware Two-Tower reference, LightGCN, and KGAT-SAL. Each receives either the
conventional feature contract or the conventional contract plus structured LLM features, dense LLM
embeddings, and training-only dish information. Both conditions retain identical spatial CBG
edges, interactions, and non-LLM features.

All models train for 300 epochs with 1,024-dimensional embeddings under seeds 42, 43, and 44.
LightGCN uses three propagation layers. KGAT-SAL uses four collaborative-filtering layers, two
knowledge-graph layers, and three temporal periods. Validation NDCG@10 selects checkpoints and the
winning configuration; the test split is evaluated once. Results rank against the full catalogue
of 61,204 restaurants that occur in eligible interactions.

| Model | Feature condition | Hit@10 | NDCG@10 |
|---|---|---:|---:|
| Two-Tower | Conventional | 0.01634 ± 0.00015 | 0.00785 ± 0.00010 |
| Two-Tower | Conventional + LLM | 0.01649 ± 0.00015 | 0.00796 ± 0.00007 |
| LightGCN | Conventional | 0.02556 ± 0.00035 | 0.01274 ± 0.00007 |
| LightGCN | Conventional + LLM | 0.02677 ± 0.00005 | 0.01338 ± 0.00008 |
| KGAT-SAL | Conventional | 0.02862 ± 0.00026 | 0.01437 ± 0.00018 |
| **KGAT-SAL** | **Conventional + LLM** | **0.03200 ± 0.00059** | **0.01614 ± 0.00029** |

> **Figure 9 — Model and feature comparison.** Insert `publication_model_llm_comparison.png`.

## The same features produce different gains

The Two-Tower model changes little: LLM features improve NDCG@10 by 1.4% and Hit@10 by 0.9%.
LightGCN gains 5.0% in NDCG and 4.7% in Hit Rate. KGAT-SAL gains 12.3% in NDCG and 11.8% in Hit
Rate, with both metrics improving under every seed.

This does not prove why KGAT-SAL improves. The experiment did not intervene on individual message-
passing paths. It does show a consistent architecture–feature interaction: a unified semantic
representation is most useful in the model with the richest set of typed relations through which
to propagate it. Feature generation and model design should therefore be evaluated as a coupled
system rather than as independent choices.

## What the LLM still misses: where the diner can realistically go

KGAT-SAL already contains spatial edges between each CBG and its five nearest CBGs. Yet explicit
distance remains powerful after ranking. We use the diner’s training-history centroid and tune a
simple exponential distance blend on validation predictions only. The frozen configuration uses a
0.7 proximity weight, a 5 km bandwidth, and the model’s top 100 candidates.

On the one-time test evaluation, proximity reranking raises NDCG@10 from 0.01614 to 0.02378 and
Hit@10 from 0.03200 to 0.04803—relative gains of 47.3% and 50.1%. Proximity does not replace the
GNN; it reorders only candidates that the GNN already considered plausible.

> **Figure 10 — Proximity uplift.** Insert `publication_proximity_uplift.png`.

## Performance across people and cuisines

The final KGAT-SAL, LLM-feature, and proximity pipeline is nearly level across the classified
inferred demographic groups. Female and male Hit@10 values are 0.04859 and 0.04954; their
NDCG@10 values are 0.02385 and 0.02456. Across the four sufficiently represented inferred
race/ethnicity groups, Hit@10 ranges from 0.04663 to 0.04874. These labels are name-inferred,
not self-reported, and the result is a descriptive subgroup diagnostic rather than evidence of
individual fairness or equal treatment.

> **Figure 11 — Final-model performance by inferred demographic group.** Insert
> `fairness_performance.png`.

Performance varies much more across the held-out restaurant's cuisine. Hit@10 is highest for Fast
Food & Burgers (0.07220), Sandwiches & Deli (0.06827), and Café & Bakery (0.06053), but falls to
0.02896 for Mediterranean & Middle Eastern restaurants. NDCG@10 follows a similar ordering. This
gap may reflect differences in catalogue frequency, interaction density, geographic concentration,
chain repetition, and cuisine-label breadth; the segmented result alone does not identify a cause.

> **Figure 12 — Final-model performance by held-out cuisine.** Insert
> `segment_performance.png`.

The result illustrates an important systems lesson. LLM-derived semantics answer “what kind of
place fits this diner?” Distance answers “which plausible place is useful from here?” Encoding
nearby CBG relationships inside the graph does not guarantee that the final top ten will respect
the strength and immediacy of geographic utility.

## A hit is not automatically a positive experience

The primary experiment uses implicit feedback: a visit is an interaction regardless of its rating.
Consequently, Hit@10 answers whether the held-out restaurant was recovered, not whether the diner
liked it. We added a diagnostic that labels four- and five-star targets as liked, three-star targets
as neutral, and one- and two-star targets as disliked.

After proximity reranking, positive Hit@10 is 0.03764 and disliked-target Hit@10 is 0.00600.
Among recovered targets, 78.4% were liked and 12.5% were disliked. In the complete test population,
80.1% were liked and 11.4% disliked. The recovered set is therefore not more positive than the
baseline target distribution; it is slightly less so. Rating-signed discounted utility@10 is
0.01553.

> **Figure 13 — Rating-aware hits.** Insert `publication_rating_aware_hits.png`.

This does not invalidate the architecture comparison, because every condition used the same
implicit objective and split. It does limit the product interpretation. A future model should
either threshold relevance, weight interactions by sentiment, or optimize a utility-aware ranking
loss if “liked restaurant” rather than “visited restaurant” is the intended outcome.

---

# Part 4 — GraphRAG as an explanation layer, not a replacement ranker

## Separate deciding from explaining

The final system gives the LLM a deliberately narrow job. KGAT-SAL scores the catalogue. The frozen
proximity stage reorders the top 100. Only after a restaurant is selected does GraphRAG retrieve a
compact evidence subgraph and ask the LLM to verbalize it.

The retrieved bundle can contain the diner’s training-only restaurant history, supported user and
restaurant LLM attributes, cuisine and CBG relations, restaurant metadata, and training-safe
review excerpts. It excludes validation/test review text and private profile fields. Every factual
claim must cite an evidence node.

> **Figure 14 — GraphRAG example.** Insert `graphrag_explanation_example.png`.

## A concrete recommendation explanation

In one audited example, the ranker and proximity stage selected Elizabeth's, a fine-dining
restaurant in Washington, DC. The graph connected a training-review-derived plant-based user
attribute of 0.9 with the restaurant's corresponding attribute of 1.0 `[M1]`. Training-safe reviews
then supplied evidence about its vegan tasting menu, atmosphere, and service `[Q1][Q2]`.

The constrained model produced:

> The evidence shows a shared attribute where the user's plant-based score is 0.9 and the
> restaurant's corresponding score is 1.0 [M1]. Reviewers describe the establishment as offering
> a fully vegan tasting menu with a fantastic atmosphere and prompt service [Q1][Q2].

The local judge assigned this example 4/5 for entailment, 5/5 for personalization, 4/5 for
usefulness, and 5/5 for citation correctness, with zero unsupported claims. It illustrates the
intended mechanism: `[M1]` supports the personalized semantic connection, while `[Q1][Q2]`
support claims about the restaurant.

## Traceability is necessary but not sufficient

The full audit uses 100 deterministic top-1 recommendations, with 25 users from each activity
quartile. Supported personalization evidence was available for 79% of examples and cited in every
case where it was available. All explanations used valid evidence identifiers, all cited restaurant
evidence, and all evidence bundles were free of held-out review IDs. Declared short quotations
matched retrieved evidence and the length constraint in 95% of cases.

However, the independent local-LLM judge scored average entailment at 3.39/5, personalization at
2.34/5, usefulness at 2.71/5, and citation correctness at 3.75/5. It flagged 0.81 unsupported claims
per explanation. A valid citation identifier does not guarantee that every phrase surrounding the
citation is entailed; some summaries broadened or combined evidence too aggressively.

> **Optional appendix figure — GraphRAG audit.** `publication_graphrag_audit.png`. Prefer the
> disaggregated audit table in the main article.

The defensible conclusion is therefore narrower than “GraphRAG solved explainability.” The system
demonstrates a leakage-safe, auditable explanation architecture and produces strong individual
examples, but average verbalization quality is not yet publication-grade. A blinded human audit,
followed by error-driven prompt and retrieval refinement, is required before presenting it as a
reliable user-facing layer.

## What the complete study shows

The experiment supports three connected findings.

First, a local LLM can replace a patchwork of task-specific extraction components with one unified
semantic feature contract. Second, the value of that contract is asymmetric: it produces its
largest gain in a knowledge-graph recommender able to propagate semantic relations. Third, neither
the LLM nor the graph makes explicit geography obsolete, and neither ranking stage automatically
produces a faithful natural-language explanation.

The broader design pattern is modular. Use LLMs where language understanding is the bottleneck.
Use graph learning where relationships and propagation are the bottleneck. Use explicit proximity
where physical utility is the bottleneck. Use GraphRAG after ranking when the bottleneck is a
human-readable, evidence-traceable explanation—and audit that explanation as its own model output.
