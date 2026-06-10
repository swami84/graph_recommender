# Foodie Revamp — Technical Blog Series Outline

_Drafted 2026-06-06. Companion to `RECONCILIATION.md` (results source-of-truth)._

**Series framing:** *Building a restaurant recommender from 2.5 million Google reviews — graphs, local LLMs, and geography.*
**Format:** technical blog series (accessible, narrative, code snippets + figures). 4 parts, ~2,000–3,500 words each.
**Through-line:** We collected ~2.5M distinct reviews across 666 high-traffic Census Block Groups, engineered ~200 features with a locally-hosted 9B LLM, benchmarked 13 GNN architectures (6 reimplemented from recent papers, plus a custom hybrid) — and found a simple geographic re-ranking step beat every architectural refinement. (The raw scrape captured ~10M review-rows, but each review was re-fetched ~4× across passes; deduplicated, that is ~2.5M distinct reviews.)

**Canonical-data rule (applies to every part):** use the **unextended / emb_dim=2048** numbers only. The Apr 28 dataset extension is excluded (it degraded results and overwrote artifacts). See `RECONCILIATION.md`.

---

## Part 1 — Data & Collection
### "Building the Dataset: From Foot-Traffic Priors to a 2.5M-Review Graph"

**Hook:** No Netflix-style ratings matrix exists for restaurants. So we started from where people *actually go* — SafeGraph foot-traffic — and scraped the reviews ourselves.

**Sections**
1. **Why Census Block Groups?** The cold-start framing; SafeGraph Patterns as a foot-traffic prior. (`cbg_patterns.csv`, Oct–Nov 2018, ranked by `raw_visitor_count`.)
2. **Selecting the densest blocks.** Top-2,000 by visitor count → proximity/density expansions (within 50 km, population-density signals). The 12-digit FIPS `zfill` gotcha. Funnel: 4,477 CBGs queried → 2,705 productive → **666** with reviewed restaurants.
3. **Finding the restaurants.** H3 hexagon tiling (res 9, 250 m radius) over Census TIGER/Line CBG polygons (via pygris) + Google Places (New) `searchNearby`. 20-result cap, early-stop heuristic, Advanced field mask. **Cost: 23,479 calls, $751.33.** 44,630 unique restaurants found → **18,879** reviewed (canonical).
4. **Scraping the reviews.** Browser automation: camoufox (anti-detection Firefox) + Playwright + injected Google cookies; 200 reviews/restaurant cap, 50 concurrent sessions. Per-review fields: rating, meal type, food/service/atmosphere sub-scores, recommended dishes, photos, local-guide flag, contributor id. **Ethics/ToS note required.** Volume: **~2.5M distinct reviews / 1.96M reviewers** (≈10M raw scrape-rows before 4× dedup).
5. **Who's reviewing? (demographics + the caveat).** RaceBERT (`pparasurama/raceBERT`) + `gender_guesser` on display names → `name_predictions.parquet`. **Strong caveat block:** probabilistic, known demographic bias, names ≠ identity, binary+unknown gender. Distribution (unique reviewers): nh_white 720K / api 589K / nh_black 521K / hispanic 133K. Finding: mean rating **nearly identical across race (~4.07–4.08)**.
6. **Building the graph.** 5 node types (restaurant, reviewer, dish, cbg, cuisine) / edges (REVIEWED ~2.5M distinct, stored as 10M 4×-duplicated parallel edges, plus SERVES, LOCATED_IN, HAS_CUISINE); NetworkX MultiDiGraph (2.3 GB). The training loader dedups interactions to one row per (reviewer, restaurant), so the duplication doesn't affect results. Split = **leave-one-out by recency**; filter = **users with ≥3 unique restaurants**. Metrics: Precision/Recall/NDCG@10.

**Figures** — *existing:* `race_gender_distribution.png`, `rating_by_demographics.png`, `rating_heatmap_race.png`, `race_by_cbg.png`, `engagement_by_demographics.png`, Folium CBG map. *To create:* US map of 666 CBGs; data funnel diagram; H3-tiling illustration over one CBG; graph schema diagram.

**Gaps to close**
- ~~Demographics PNGs were generated on the **extended** scrape — regenerate on canonical set.~~ **DONE (2026-06-09):** all `eda_*` figures now regenerated on `reviews_flat_unextended` (canonical, deduped on review_id); composition/activity/active-core numbers updated in Part 1.
- Post-filter train user/item counts aren't persisted — compute once and report.
- Race-label vocabulary mismatch (`nh_white/api/...` vs `WHITE/BLACK/...`) — standardize.

---

## Part 2 — LLM Feature Engineering
### "Turning Review Text into ~200 Features with a Local 9B Model"

**Hook:** Reviews are unstructured prose. We turned ~700K–900K local-LLM calls into structured, *interpretable* features — no cloud API, no opaque embeddings.

**Sections**
1. **The local-LLM stack.** Qwen3.5-9B-AWQ via **vLLM** (tensor-parallel on 2×RTX 3090, port 8082), `temperature=0`, `enable_thinking=False` (the Qwen3 token-budget trick), OpenAI-compatible endpoint, constrained JSON decoding. Why local: cost + scale + privacy.
2. **Restaurant attributes from text.** `build_nlp_features.py` → 12 scored attributes (spice, noise, formality, family-friendly, …) using a 0↔1 anchoring guide; `build_restaurant_llm_features.py` → authenticity, occasion, ambiance, dietary flags. Top-K review selection per restaurant.
3. **Dishes as first-class entities.** `extract_dishes_llm.py` (batched 30 reviews/call, `json_schema`-constrained, heavy negative-list prompt) → **254K dishes / 750K edges**. `build_dish_embeddings.py` → **66-dim structured flavor descriptors (no sentence-transformer)** → cosine similarity graph (862K pairs). Design choice: interpretable similarity over black-box embeddings.
4. **User preferences mirror item attributes.** `build_user_preference_features.py` → 12 dims 1:1 with item NLP attributes (enables user↔item alignment in the GNN). `build_user_dietary_features.py` → 6 affinities (5,000 users).
5. **Non-LLM analytical features.** `build_extended_features.py` (Polars): cuisine-affinity vector + entropy, spatial density (BallTree 500 m/2 km), CBG foot-traffic ratios, recency/velocity, rating-distribution stats.
6. **Assembling the matrix.** `build_training_features.py` → `*_features_prebuilt.parquet` + `feature_groups.json`. Group taxonomy (user: base/extended/pref/dietary; item: base/nlp/extended/llm/dietary). Ablation = `--skip-feature-groups` (drop-by-name at load time in `recommendation_gnn.py`).

**Figures** — *to create:* feature-taxonomy tree; LLM pipeline diagram (reviews → vLLM → JSON → parquet); example prompt + JSON-output box; a dish flavor-vector / similarity-neighborhood example; per-group coverage table.

**Gaps to close (important for honest reporting)**
- ✅ **RESOLVED (verified 2026-06-06).** The prebuilt item parquet has **zero semantic LLM columns** (no `authenticity_score`/occasion/ambiance — confirmed), and the drivers use default `--llm-feat-mode fast`. So `build_restaurant_llm_features.py`'s ~43 semantic attributes were **built but NOT used** by the canonical models; `item.llm` was only the 6 cheap analytics, and the "skip llm" ablation moved NDCG by ~0.000 (corroborates negligible contribution). **The LLM signal that mattered entered via `item.nlp` (12 LLM-scored attributes) + `item.dietary` (5) + `user.pref` (12 LLM mirrors) + dish-derived diversity features** — all present and used. *Part 2 framing:* the value came from the NLP attributes, dishes, and preferences, not the bespoke semantic-attribute extractor (an honest, interesting result).
- ⚠️ **`user.dietary` is empty in `feature_groups.json`** though the parquet exists and `build_training_features.py` wires it (stale JSON). Decide: regenerate + include (re-run?) or document as "built, not used." *(Deferred — not blocking; revisit when drafting Part 2.)*

---

## Part 3 — Model Architectures
### "Thirteen Graph Recommenders, From LightGCN to a Custom Hybrid"

**Hook:** We benchmarked the GNN-CF zoo, reimplemented six recent papers, and built a hybrid that fuses knowledge-graph attention with contrastive learning.

**Sections**
1. **The setup.** Implicit-feedback BPR ranking on the user–item graph; feature injection via **FiLM**; `_SparseMmChunkedBF16` to fit D=2048 in memory.
2. **Baselines.** ALS (`implicit`), NeuMF/NCF.
3. **Standard GNN CF.** LightGCN (workhorse), UltraGCN (constraint-based ∞-layer approx), SimGCL (noise-augmented CL), DGCF (disentangled intents).
4. **Knowledge-graph models.** KGAT (attentive propagation over cuisine/price/CBG/dish KG); KGAT-SAL (+ Self-Augmented Learning: temporal periods + stability weighting).
5. **Paper reimplementations (contrastive / self-supervised).** Paper→model map:
   | PDF | Idea | Model file |
   |---|---|---|
   | INFONCE_GCL | GCL as positive-unlabeled learning | `ifl_gcl` / `ifl_gcl_kg` |
   | HEK-CL | Hyperbolic (Poincaré) embeddings + denoising + HRCL | `hek_cl` / `simgcl_hrcl` |
   | RaDAR | Diffusion denoiser + asymmetric predictor + edge scorer | `radar` |
   | Self-GNN | Self-augmented short-term + long-term | SAL in `kgat_sal` |
   | HGNN-AR | Adaptive KG edge reconstruction | KG component in `kgat`/hybrid |
   | LIT-GRAPH GCN | Deep relational KG embeddings | rationale for `kgat` |
6. **The custom hybrid: InfoNCE-KGAT-SAL.** KGAT-SAL backbone + two-view InfoNCE CL on KG-enriched embeddings. `L = L_BPR + λ_cl·L_CL + λ_sal·L_SAL`. The **InfoNCE plateau fix** story (BPR aux loss + hard-negative mining + temperature curriculum). SeqHybrid (LightGCN+SASRec) as an aside.

**Figures** — *to create:* architecture diagrams (LightGCN, KGAT, KGAT-SAL, hybrid); model family tree; paper→model table; loss-decomposition diagram for the hybrid.

**Gaps to close**
- Confirm baseline metrics (ALS/NCF/DGCF) for the leaderboard — not in the canonical CSV.
- `kgat.pt` (Apr 24) vs `kgat_all.pt` naming — identify canonical checkpoint if needed.
- Decide depth per model (full treatment vs one-paragraph).

---

## Part 4 — Results, Spatial Re-ranking & Fairness
### "Geography Beats Architecture"

**Hook:** The punchline — a simple distance-aware re-ranking lifted precision@10 from ~0.012 to ~0.14, dwarfing the gaps between architectures. Then: where the model is fair, and where it isn't.

**Sections**
1. **The leaderboard.** Frozen table (unextended/2048), base vs +proximity; ablations (skip-llm). Top: InfoNCE-KGAT-SAL 0.0735 / KGAT-SAL 0.0734 / KGAT 0.0727 (+prox NDCG@10).
2. **Geography beats architecture.** `rerank_proximity.py`: home = median train lat/lng; kernel `exp(-dist/bw)`; `blended = (1-α)·norm(score) + α·norm(prox)`. α×bandwidth heatmaps. The 0.012→0.14 precision finding (and why: P@10==R@10 under leave-one-out). Best config per model.
3. **Where does it work?** Per-location & per-CBG (§5), per-cuisine (§8), per-restaurant-characteristic — popularity/rating/price (§9).
4. **Who does it work for? (fairness).** Per-activity-quintile (§6), per race/gender (§7), cross cuisine×race & race×activity (§10).
5. **Cold-start & popularity bias** (§11): recommended-vs-catalog popularity mix; hit/miss by restaurant popularity.
6. **Summary dashboard** (§13) + model-vs-model across segments (§12).

**Figures** — mostly *exist* in `prediction_analysis.ipynb` (§2,3,5–13). *To create:* the headline "geography beats architecture" chart.

**Gaps to close — ✅ RESOLVED (2026-06-06, `prediction_analysis.ipynb` now runs on canonical data)**
- ✅ **§4 repointed** to surviving unextended `kgat_sal_all_T3_nospatial_prox_a30_predictions.parquet` (131,632 rows; verified Hit@10=0.1321, NDCG@10=0.0699). Single-file load replaces the old base+a60 merge. Note: α=0.3 (the α=0.6 optimum was overwritten) — fine for per-segment analysis; leaderboard still reports the 0.0734 α=0.6 optimum.
- ✅ **§2 leaderboard** filtered (`emb_dim != 1024`): 96→71 canonical rows.
- ✅ **§3 grid** filtered to per-model max `base_ndcg`: 756→168 canonical rows.
- ✅ **§12 baseline** repointed to unextended `kgat_all_prox_a30` (Hit@10=0.1360) — now a KGAT-SAL+prox **vs** KGAT+prox comparison (both reranked). ⚠️ Its markdown still says "no proximity" — **update §12 prose when drafting**.
- All stale (extended-data) figure outputs cleared — **the notebook must be re-run top-to-bottom** to regenerate canonical figures. Backup at `prediction_analysis.ipynb.bak`.

---

## Cross-cutting

- **Publishing order:** 1 → 2 → 3 → 4 (chronological = pedagogical).
- **Transparency box** (every part or an appendix): the unextended/2048 canonical decision + why the extension was excluded.
- **Optional Part 5 — "What didn't work":** the InfoNCE plateau; the dataset-extension regression (more data ≠ better — lower-traffic CBGs added noise); dietary features built-but-unwired; `llm-feat-mode=fast` leaving semantic LLM features unused. A strong honest-retrospective closer; or fold the extension regression into Part 4 as a sidebar.
- **Recurring assets:** code in repo root; data in `data/`; results in `results/`; frozen numbers in `RECONCILIATION.md`.
