# Data collection, processing, and exploratory analysis

> Working article section, updated after completion of the publication runs.
> Counts come from the frozen collection manifest, canonical Parquet files, and
> leakage-safe split manifest. Final model and GraphRAG results are developed in
> `TECHNICAL_ARTICLE_SERIES_DRAFT.md`.

## Study setting

This study asks whether a local large language model can turn restaurant-review
text into structured preference and venue attributes that improve restaurant
recommendation beyond conventional metadata and behavioral signals. The final
system separates three jobs: a recommender produces the ranking, a geographic
stage constrains or re-ranks candidates by proximity, and GraphRAG retrieves
evidence for an LLM-generated explanation. The LLM is therefore evaluated both
as a feature generator before ranking and as a grounded verbalizer after
ranking; it does not replace the recommender.

## Restaurant discovery

The discovery frame combines two complementary sampling strategies. The
original collection emphasized census block groups (CBGs) with high observed
foot traffic and nearby CBG expansions. Because that strategy can miss densely
populated residential areas, a second pass constructed a nationally balanced
queue from population density. The queue retained 10,000 CBGs, protected the
densest CBG in every state and the District of Columbia, and filled the
remaining positions by national density rank. Of these 10,000 CBGs, 2,466 were
already covered by prior terminal searches and 7,534 required new collection.

Each CBG polygon was tiled with H3 resolution-9 cells. A representative-point
fallback handled polygons too small or narrow to contain an H3 cell center.
Each cell was queried using the Google Places API (New) Nearby Search endpoint,
with `restaurant` as the included type, a 250-m radius, and the API maximum of
20 results. The discovery field mask requested identifiers, names, addresses,
coordinates, types, business status, and primary type. Responses were
checkpointed per H3 cell, and a terminal CBG marker prevented a valid empty
search from being repeated. Restaurant identity was deduplicated by Google
Place ID.

The density queue included all 50 states and DC. The final deduplicated
restaurant catalogue contains observations in 49 states/DC; Alaska and Montana
produced no retained restaurants. This exceeds the prespecified target of at
least 45 states but should be described as broad national coverage—not a
uniform or population-representative national sample.

## Review collection and quality controls

Reviews were collected from public Google Maps place pages using Camoufox. The
collector used stable browser contexts, fresh pages between restaurants,
staggered launches, globally rate-limited starts, randomized inter-restaurant
pauses, and deliberately paced pagination scrolling. These choices reduced
network bursts and allowed page state to persist without carrying a restaurant
page forward to the next task.

Collection outcomes were checkpointed by Place ID. A known-positive probe
distinguished a system-wide rendering or challenge problem from a true
restaurant-level absence of reviews. Network failures, automated-traffic
challenges, and pages that could not be validated remained retryable instead of
being classified as zero-review restaurants. A final integrity pass required
every eligible restaurant to have a checkpoint; it also checked invalid JSON,
declared-versus-observed review counts, retryable outcomes, contradictory
no-review classifications, and orphan files. The frozen audit reported zero
unresolved integrity issues.

The frozen collection contains 75,203 unique restaurants and 5,545,260 raw
review objects. Of the restaurants, 70,628 had at least one raw collected review
and 4,575 had none at the checkpoint stage.

## Canonical analytical data

The analytical pipeline assigned each Place ID one restaurant row, normalized
Google place types into a deterministic cuisine taxonomy, and flattened nested
review JSON. Review-level fields include rating, relative timestamp, text and
text length, local-guide status, photo count, meal type, price per person, and
food, service, and atmosphere scores when present. Duplicate nonempty review
IDs were removed and records without the reviewer identity required by the
flattening pipeline were excluded.

This creates an important distinction between collection and analysis counts:

| Stage | Restaurants | Reviews | CBG interpretation |
|---|---:|---:|---|
| Frozen scraper checkpoints | 75,203 | 5,545,260 | 8,942 CBGs produced restaurant hits across tiled searches |
| Canonical analytical tables | 75,203 | 5,544,947 | 7,199 single CBG assignments after Place-ID deduplication |

The 313-row difference is a 0.006% analytical exclusion, not missing scraper
output. The CBG difference likewise does not indicate lost geography: a venue
can be returned by tiles associated with more than one CBG, but is stored once
with one canonical CBG in the restaurant table.

## Leakage-safe recommendation population

The recommendation experiment treats a user–restaurant pair as an implicit
interaction. It requires a nonempty contributor ID, restaurant ID, and rating,
then retains one interaction for each unique user–restaurant pair. Users need
at least four unique rated restaurants. This yields 156,261 eligible users.

Interactions are split chronologically per user: the newest unique restaurant
is the test target, the second-newest is the validation target, and all older
restaurants are training history. The resulting split contains 772,170 training
interactions, 156,261 validation targets, and 156,261 test targets. User LLM
features are generated only from training-history reviews. Held-out validation
and test text is excluded from user and restaurant LLM corpora, including the
evidence later exposed to GraphRAG.

## Exploratory findings

The canonical catalogue has 75,203 restaurants, 70,625 of which (93.9%) retain
at least one analytical review; 4,578 have none. Relative to the 44,630-venue
catalogue described in the archived pre-expansion article, the new catalogue is
larger by 30,573 restaurants (68.5%). This is a historical before/after
comparison and should not be interpreted as 30,573 restaurants attributable
only to the final population-density pass.

The analytical review table contains 5,544,947 unique review IDs from 3,930,034
identified reviewers. Review text is available for 83.6% of rows, at least one
attached photo for 30.4%, and the mean stored text length is 265.2 characters.
Ratings are highly positive: 63.2% are five-star and 12.5% are four-star,
compared with 12.9% one-star, 4.8% two-star, and 6.5% three-star. The reviewer
population is also strongly long-tailed: 82.0% appear at only one restaurant in
the collected graph, while 4.0% qualify for the four-or-more-restaurant modeling
population.

Coverage is concentrated. New York contains 18,575 catalogue restaurants,
California 13,483, Texas 6,883, Florida 4,489, and New Jersey 4,467.

The original LLM cuisine output failed quality control: duplicated labels and
implausible `Southern` assignments made it unsuitable for publication. A targeted
27B repair has now processed all 15,747 restaurants assigned `Other` by Google's
broad type mapping. It assigns a cuisine only when supported by names, Google
types, or review evidence; 6,996 restaurants (9.3% of the full catalogue)
remain `Unknown / insufficient evidence`. This is preferable to forcing a label.
The unresolved group spans both the pre-expansion and new cohorts, so it is not
merely an artifact of the recent collection.

The archived EDA's absence of `Other` was itself a pipeline artifact. Its merger
used `argmax` on cuisine one-hots, mapping 6,878 all-zero rows to the first class,
`American & Comfort`, and its plotting code then omitted `Other` from the displayed
category order. The analytical audit reports Google categories and audited cuisine
labels separately. The new publication figures omit the unresolved category only as
an explicit display choice: their subtitles disclose the 6,996 excluded venues and
their cuisine percentages use classified venues as the denominator. No rows are
removed from modeling or non-cuisine totals.

Places ratings are missing for 32,819 restaurants (43.6%) and price level for
44,483 (59.2%); addresses and coordinates are complete. These missingness rates
also motivate review-derived attributes, but missing values must be handled
identically across feature conditions so that the LLM comparison remains fair.

## Threats to validity

The sample overrepresents high-foot-traffic and high-density CBGs, and venue
visibility is mediated by Google Maps. The API's per-query result cap can create
additional undercoverage in unusually dense cells. Review availability,
restaurant age, language, and reviewer participation vary geographically.
Relative timestamps are less precise than event timestamps, so chronological
ties require deterministic handling. Google ratings and price levels are
substantially incomplete in the discovery metadata.

Name-based demographic predictions, where present, are probabilistic and must
not be described as self-reported identity or used for sensitive targeting.
They are suitable only for carefully qualified diagnostic analysis. Finally,
LLM features are derived from observed review text and may encode popularity,
reviewer-selection, geographic, or linguistic biases. The article should report
coverage of each LLM field and include a later feature audit even if model
training initially assumes the generated values are usable.

## Reproducibility pointers

- Queue construction: `build_top_density_queue.py`
- Places collection: `hexagon_places.py`
- Review freeze audit: `finalize_review_dataset.py`
- Canonical tables: `build_graph_data.py`
- LLM corpus and split manifest: `build_llm_features_ollama.py`
- Reproducible EDA: `analysis/expanded_eda/run_eda.py`
- EDA report and figures: `analysis/expanded_eda/EDA_REPORT.md`

All three seeds have completed for Two-Tower, LightGCN, and KGAT-SAL under both
non-LLM and full-LLM conditions. The validation-selected proximity evaluation,
rating-aware hit diagnostic, and GraphRAG audit are reported in
`results/publication_final_results.md` and the master series draft.
