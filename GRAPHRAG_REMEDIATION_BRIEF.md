# GraphRAG explanation layer: remediation brief

**Hand this file to the implementing agent as its task specification.** It is self-contained. Everything below was derived from the frozen audit artifacts in this repository; the diagnosis is settled, the work is not.

---

## 1. Context

This repository (`foodie_revamp`) contains a restaurant recommendation pipeline. A KGAT-SAL graph model ranks the catalogue, a frozen proximity function reranks its top 100 candidates, and only then does a GraphRAG layer assemble a training-safe evidence subgraph and ask a local 27B model to write a two-sentence cited rationale. A second pass of the same model judges each rationale on four 1–5 dimensions plus an unsupported-claim count.

An audit of 100 explanations produced these results:

| Deterministic provenance check | Result |
|---|---:|
| Valid inline citations | 100% |
| Declared evidence resolvable | 100% |
| Restaurant evidence cited | 100% |
| Held-out-safe bundles | 100% |
| Supported user match available | 79% |
| Available match cited when present | 100% |
| Exact short-quote fidelity | 95% |

| Local-judge score | Mean | Share 1–2 | Share 4–5 |
|---|---:|---:|---:|
| Entailment | 3.39 | 48% | 51% |
| Personalization | 2.34 | 57% | 23% |
| Usefulness | 2.71 | 51% | 32% |
| Citation correctness | 3.75 | 28% | 64% |
| Unsupported claims | 0.81 per explanation | 57% have none | 36% have two or more |

**Retrieval is not the problem.** Held-out safety and citation resolution are both perfect. Do not rebuild the retriever, the subgraph assembly, the ranker, or the proximity configuration.

**The problem is in evidence serialization and in the judging rubric.** Reading all 48 judge reasons behind the 1 and 2 entailment scores:

- Roughly half (22 to 26 of 48) turn on an entity-attribution dispute that the evidence format cannot settle.
- 7 of 48 record no unsupported claim at all and penalize an omission instead; 4 of those concern a contradicting attribute.
- 8 of 48 criticize generic wording.
- The residual is genuine calibration failure, for example rendering one cited reviewer as "Reviewers".

So the 48% low-entailment rate is an upper bound on generation failure, not a measurement of it. Tasks 1 and 2 below are what make it measurable. **They must be completed before Tasks 4 and 5**, because scaling the audit first would buy precision on a biased estimate.

---

## 2. Relevant files

| Path | Role |
|---|---|
| `foodie/explanations/run_publication_graphrag.py` | Everything below lives here: evidence assembly, prompts, judging, summary |
| `foodie.explanations.run_publication_graphrag.prepare_evidence()` (~line 118) | Builds one evidence bundle per recommendation |
| `foodie.explanations.run_publication_graphrag.add_evidence()` (~line 106) | Appends a node as `{id, kind, fact, source}` |
| `foodie.explanations.run_publication_graphrag.generation_prompt()` (~line 312) | Explanation prompt |
| `foodie.explanations.run_publication_graphrag.judge_prompt()` (~line 341) | Judging prompt |
| `foodie.explanations.run_publication_graphrag.summarize()` (~line 445) | Aggregates the audit |
| `results/graphrag/publication_graphrag_evidence.jsonl` | Frozen bundles, 100 rows |
| `results/graphrag/publication_graphrag_generations.jsonl` | Frozen generations plus judge output, 100 rows |
| `results/graphrag/publication_graphrag_results.parquet` | Flattened per-case scores and deterministic checks |
| `results/graphrag/publication_graphrag_human_audit.csv` | 20 prepared rows, unscored, for a separate blinded human audit |

Node id prefixes currently in use: `R` restaurant metadata, `H` training history, `U` LLM user attribute, `A` LLM restaurant attribute, `M` supported personalization match, `Q` training-safe review excerpt. Mean bundle size is 14.3 nodes, range 8 to 19.

Default model is `hf.co/unsloth/Qwen3.5-27B-GGUF:Q4_K_M` served over Ollama at `http://127.0.0.1:11434`. CLI flags already exist: `--sample-size`, `--concurrency`, `--url`, `--model`, `--prepare-only`.

---

## 3. Task 1 (blocking): make every evidence node state which entity it describes

### The defect

In `prepare_evidence()`, review excerpts are serialized as:

```python
f"A training-safe reviewer rated it {rating:g} stars and wrote: \"{text[:350]}\""
```

The word **"it"** has no antecedent anywhere in the bundle. All 300 review nodes across the 100 audited bundles use this scaffold and none names the restaurant; the quoted review text happens to mention it in only 25 of 300 cases, by accident of what the reviewer wrote. By pipeline design these excerpts describe the recommended restaurant, but nothing in the evidence says so, and 47 of the 48 low-entailment explanations cite at least one of them.

The consequence is visible in the judge reasons. One correctly identifies the gap: *"the evidence [Q1] does not explicitly state which restaurant the quote refers to, only that a 'training-safe reviewer' rated 'it' 5 stars; without a direct link in the text to Sub Station Downtown, this is an assumption."* Others resolve the ambiguity the other way and assert an attribution the bundle does not contain, for example claiming the evidence "explicitly states these reviews are for Big Cheese Pizza" when no node names any restaurant.

A second, related ambiguity: user and restaurant attribute nodes are near-identically worded, differing only by one word.

```python
f"Training-review-derived user attribute: {attribute} score {score} on a 0–1 scale."
f"Training-review-derived restaurant attribute: {attribute} score {score} on a 0–1 scale."
```

At least one judge reason complains that the explanation attributed user scores (U1, U3) to the restaurant.

### What to change

1. Add an `entity` field to every node. Extend `add_evidence()` to take and store it. Populate it with the entity's display name, plus an entity type (`recommended_restaurant`, `history_restaurant`, `diner`).
2. Rewrite the `Q` fact template to name the restaurant inline, for example: `A training-safe reviewer of {restaurant_name} rated it {rating:g} stars and wrote: "..."`.
3. Rewrite the `U` and `A` templates so they cannot be confused by a reader who has lost track of the id prefix, for example `Diner's own preference, derived from their training reviews: {attribute} {score} on a 0–1 scale` versus `{restaurant_name}, derived from training reviews: {attribute} {score} on a 0–1 scale`.
4. Confirm `H` nodes already name their restaurant (they do) and give them the same `entity` treatment for consistency.
5. Update `generation_prompt()` to state that every evidence line names the entity it describes and that claims must not transfer a fact from one entity to another.

### Acceptance criteria

- Every node in every regenerated bundle has a non-empty `entity` and `entity_type`.
- No `Q` fact matches the regex `rated it \d` without a preceding restaurant name.
- Running with `--prepare-only` and diffing bundle counts against the frozen run shows the same node ids and the same held-out filtering; only fact strings and the new fields change.
- Do not change which nodes are retrieved. This task is serialization only.

---

## 4. Task 2 (blocking): separate entailment from completeness in the rubric

### The defect

The current judge prompt supplies no definition for any dimension:

```
Return JSON with integer scores from 1 (poor) to 5 (excellent):
{"entailment":1,"personalization":1,"usefulness":1,"citation_correctness":1,"unsupported_claims":0,"reason":"brief"}
Count every claim not directly supported as an unsupported claim. Do not reward fluent unsupported text.
```

The judge therefore invents its own criteria, and in 7 of 48 low-entailment cases it lowers entailment for something the explanation *omitted* rather than for anything it asserted. The clearest example: *"it fails to mention the restaurant's low value score (0.5) despite the user's high value preference (1.0) ... The explanation is factually supported but lacks completeness."* That is a fair criticism scored under the wrong heading, and it silently inflates the headline failure rate.

### What to change

1. Define each dimension explicitly in `judge_prompt()`. Entailment must be restricted to: *do the claims actually stated follow from the cited evidence, without contradiction or unsupported extension?* State that omissions are out of scope for this dimension.
2. Add a fifth judged dimension, `completeness` or `contradiction_disclosure`: *does the explanation surface any retrieved evidence that contradicts or materially qualifies its own claims?* This turns currently discarded observations into a measurement of how often compression from ~14 nodes to two sentences drops a contradicting node.
3. Make the `unsupported_claims` counting rule concrete: one count per asserted proposition not supported by a cited node, with an explicit instruction not to count omissions.
4. Add an `attribution_error` boolean: *does the explanation attach a fact to the wrong entity?* After Task 1 this becomes checkable, and it isolates the failure class that currently dominates.
5. Extend `summarize()` and the parquet schema for the new fields.

### Acceptance criteria

- Re-judge the **existing 100 generations** with the new rubric before regenerating anything, so the rubric change is measured in isolation.
- Report the old and new entailment distributions side by side, and the count of cases whose entailment score moves by 2 or more.
- Expected direction: the 48% low-entailment share should fall. Report the actual figure; do not assume it.

---

## 5. Task 3: use a judge that is not the generator

The generator and the judge are currently the same model in two passes. Internal consistency is good (unsupported-claim counts correlate −0.81 with entailment, citation correctness +0.75), but consistency is not accuracy, and a model grading its own output is the weakest link in the audit.

Run the new rubric over the same generations with a *different* local model as judge. Report per-dimension agreement (exact-match rate and quadratic-weighted kappa, or Spearman correlation on the 1–5 scales) and the disagreement rate on `attribution_error`. Add `--judge-model` as a separate CLI flag so generator and judge can be set independently.

---

## 6. Task 4: retain the 100-case audit

The study owner elected not to expand the automated explanation audit to 500 or 1,000 cases. The corrected evaluation therefore remains a deterministic 100-case audit. A rate near 50% carries a 95% interval of about ±10 points, so activity-quartile estimates should not be emphasized.

Do not launch a 500- or 1,000-case run. Use the separate balanced 12-case Hit@10 illustration set only for qualitative examples; do not treat its liked, neutral, or disliked proportions as population estimates.

Also run the same audit on the seed-43 and seed-44 proximity outputs. The current audit is single-seed (42), so it cannot distinguish a property of the system from a property of one ranking.

---

## 7. Task 5: personalization-withheld ablation

The audit's largest effect is that explanations with a supported user match score far worse on support than those without:

| | No match available (n = 21) | Match available (n = 79) |
|---|---:|---:|
| Entailment | 4.81 | 3.01 |
| Citation correctness | 4.71 | 3.49 |
| Usefulness | 3.43 | 2.52 |
| Unsupported claims | 0.10 | 1.00 |
| Evidence nodes | 12.0 | 14.9 |
| Training interactions | 2.8 | 5.6 |

This is confounded. Matched cases also have larger bundles and more active users, so "attempting personalization degrades support" and "larger bundles degrade support" predict the same pattern and the observational audit cannot separate them.

Regenerate explanations for the **same 79 matched users** with the `M` node removed from the bundle, holding everything else fixed. This is a paired design: with a matched-group entailment standard deviation of 1.37 at n = 79, the paired standard error is roughly 0.18 against an observed gap of 1.80, so it is amply powered. Report paired deltas on entailment, citation correctness, and unsupported claims. Where practical, also report a bundle-size-matched comparison to separate the two explanations further.

---

## 8. Minor item worth fixing in passing

`generation_prompt()` instructs the model to paraphrase and return `"quotes": []`, yet 6 of 100 generations returned quotes anyway and 5 of those failed the exact-quote-fidelity check. Either enforce the empty-quotes contract in post-processing, or drop the prohibition and require verbatim quotes with a fidelity check. The current state is a prompt rule the model ignores 6% of the time, which makes the 95% quote-fidelity figure hard to interpret.

---

## 9. Constraints

- **Do not change the ranker, the proximity configuration (alpha 0.7, 5 km bandwidth, candidate depth 100), or the chronological train/validation/test split.** Those are frozen and their results are published.
- **Do not weaken held-out filtering.** Every regenerated bundle must still exclude every validation and test review identifier. Assert this, do not assume it.
- **Do not tune anything on test data.**
- **Do not replace free-text generation with templates** without flagging it first. That is a live design option, but it changes what the study is about and is a separate decision.
- Preserve every existing deterministic check. Add to them; remove none.
- Keep the frozen artifacts intact. Write new outputs under a new suffix or directory so the published numbers remain reproducible.

---

## 10. Deliverables

1. A diff of `foodie/explanations/run_publication_graphrag.py` covering Tasks 1 to 3, plus any new CLI flags.
2. A short report comparing, for the same 100 cases: original rubric and original serialization, new rubric on original generations, new rubric on regenerated generations. State how much of the 48% low-entailment rate was schema ambiguity, how much was the rubric, and what the residual genuine failure rate is.
3. The scaled audit results with confidence intervals, including the activity-stratum breakdown that the small sample previously could not support.
4. Inter-judge agreement statistics.
5. The paired personalization-withheld comparison.
6. A note on anything in this brief that turned out to be wrong when you looked at the code. The diagnosis was derived from audit outputs rather than from running the pipeline, so treat it as a strong prior, not as ground truth.
