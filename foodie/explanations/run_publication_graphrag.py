#!/usr/bin/env python3
"""Generate and audit leakage-safe GraphRAG explanations.

The ranker remains frozen. This stage samples either top-1 recommendations or
known Hit@10 cases from the seed-42 proximity output, retrieves a compact graph
using training-only user history and review evidence, and asks a local LLM to
verbalize the evidence.
Every generated claim must cite an evidence ID. A separately configurable
judge scores the result; deterministic citation and quote checks are reported.

The original publication artifacts are immutable by default. Remediation runs
must provide ``--output-dir`` and write versioned outputs elsewhere.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd

from foodie.modeling.recommendation_gnn import load_interaction_splits


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("publication_graphrag")

PREDICTIONS = Path(
    "data/predictions/kgat_sal_full_llm_pubsplit_seed42_proximity_predictions.parquet"
)
RESTAURANTS = Path("data/restaurants_enriched.parquet")
REVIEWS = Path("data/reviews_flat.parquet")
USER_LLM = Path("data/audited_llm_features/user_llm_features.parquet")
ITEM_LLM = Path("data/audited_llm_features/restaurant_llm_features.parquet")
OUT_DIR = Path("results/graphrag")
EVIDENCE_OUT = OUT_DIR / "publication_graphrag_evidence.jsonl"
GENERATIONS_OUT = OUT_DIR / "publication_graphrag_generations.jsonl"
RESULTS_OUT = OUT_DIR / "publication_graphrag_results.parquet"
SUMMARY_OUT = OUT_DIR / "publication_graphrag_summary.md"

DEFAULT_MODEL = "hf.co/unsloth/Qwen3.5-27B-GGUF:Q4_K_M"
DEFAULT_SECONDARY_JUDGE = "llama3.1:8b"
ATTRIBUTES = (
    "spice", "sweetness", "richness", "healthiness", "novelty",
    "portion_size", "noise", "formality", "romantic", "family", "group",
    "solo", "business", "outdoor", "nightlife", "wait", "service_speed",
    "reservation", "takeout", "parking", "transit", "accessibility",
    "consistency", "value", "local_independent", "seafood", "red_meat",
    "poultry", "plant_based",
)

# Low-low alignment is semantically interpretable for these bipolar preference
# dimensions. Operational attributes are only surfaced as matches when both
# sides are high; ``wait`` is excluded because the user-side direction does not
# distinguish tolerance from a preference for a long wait.
BIPOLAR_MATCH_FIELDS = {
    "spice", "sweetness", "richness", "healthiness", "novelty",
    "portion size", "noise", "formality", "romantic", "family", "group",
    "solo", "business", "outdoor", "nightlife", "seafood", "red meat",
    "poultry", "plant based",
}
HIGH_ONLY_MATCH_FIELDS = {
    "service speed", "reservation", "takeout", "parking", "transit",
    "accessibility", "consistency", "value", "local independent",
}

GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "quotes": {"type": "array", "items": {"type": "string"}, "maxItems": 0},
    },
    "required": ["explanation", "evidence_ids", "quotes"],
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "entailment": {"type": "integer", "minimum": 1, "maximum": 5},
        "personalization": {"type": "integer", "minimum": 1, "maximum": 5},
        "usefulness": {"type": "integer", "minimum": 1, "maximum": 5},
        "citation_correctness": {"type": "integer", "minimum": 1, "maximum": 5},
        "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
        "unsupported_claims": {"type": "integer", "minimum": 0},
        "attribution_error": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": [
        "entailment", "personalization", "usefulness", "citation_correctness",
        "completeness", "unsupported_claims", "attribution_error", "reason",
    ],
    "additionalProperties": False,
}


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)


def parse_json_list(value) -> list[str]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else []
        return [str(x) for x in parsed]
    except (json.JSONDecodeError, TypeError):
        return []


def cuisine_from_llm(row: pd.Series | None, fallback: str) -> str:
    if row is not None:
        matches = [
            column.removeprefix("llm_cuisine27_").replace("_", " ")
            for column, value in row.items()
            if column.startswith("llm_cuisine27_")
            and column != "llm_cuisine27_confidence"
            and pd.notna(value) and float(value) > 0.5
        ]
        matches = [value for value in matches if value != "unknown"]
        if matches:
            return matches[0]
    return fallback or "unknown"


def top_attributes(row: pd.Series | None, prefix: str, limit: int = 4) -> list[dict]:
    if row is None:
        return []
    values = []
    for attribute in ATTRIBUTES:
        value_col = f"{prefix}{attribute}"
        known_col = f"{value_col}_known"
        if value_col not in row or known_col not in row:
            continue
        if pd.isna(row[known_col]) or float(row[known_col]) <= 0:
            continue
        value = float(row[value_col])
        values.append((abs(value - 0.5), attribute, value))
    values.sort(reverse=True)
    return [
        {"attribute": attribute.replace("_", " "), "score": round(value, 2)}
        for _, attribute, value in values[:limit]
    ]


def add_evidence(
    evidence: list[dict], eid: str, kind: str, fact: str, source: str,
    entity: str, entity_type: str, entity_id: str = "",
):
    if not entity or not entity_type:
        raise ValueError(f"Evidence {eid} requires entity and entity_type")
    evidence.append({
        "id": eid,
        "kind": kind,
        "entity": entity,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "fact": fact,
        "source": source,
    })


def attributes_align(
    attribute: str, user_value: float, item_value: float, policy: str = "strict"
) -> bool:
    """Return only explanation-safe, directionally explicit alignments."""
    if policy == "legacy":
        return (
            (user_value >= 0.6 and item_value >= 0.6)
            or (user_value <= 0.4 and item_value <= 0.4)
        )
    close = abs(user_value - item_value) <= 0.25
    high = user_value >= 0.65 and item_value >= 0.65
    low = user_value <= 0.35 and item_value <= 0.35
    if attribute in HIGH_ONLY_MATCH_FIELDS:
        return close and high
    if attribute in BIPOLAR_MATCH_FIELDS:
        return close and (high or low)
    return False


def cuisine_history_aligns(shared: bool, rating: float, policy: str = "strict") -> bool:
    """Treat shared cuisine as preference evidence only after a liked visit."""
    return bool(shared and (policy == "legacy" or float(rating) >= 4.0))


def validate_evidence_bundle(bundle: dict) -> None:
    evidence = bundle.get("evidence", [])
    if not evidence:
        raise ValueError("Evidence bundle is empty")
    ids = [item.get("id") for item in evidence]
    if len(ids) != len(set(ids)):
        raise ValueError("Evidence IDs must be unique within a bundle")
    for item in evidence:
        if not item.get("entity") or not item.get("entity_type"):
            raise ValueError(f"Evidence {item.get('id')} lacks entity attribution")
        if item.get("kind") == "training_review_excerpt":
            entity = str(item["entity"])
            fact = str(item.get("fact", ""))
            if entity not in fact or fact.startswith("A training-safe reviewer rated it "):
                raise ValueError(f"Review evidence {item.get('id')} has ambiguous attribution")


def cuisine_tokens(value: str) -> set[str]:
    ignored = {"and", "restaurant", "restaurants", "food", "unknown", "other"}
    return {
        token for token in re.findall(r"[a-z]+", str(value).lower().replace("&", " and "))
        if token not in ignored
    }


def prepare_evidence(
    sample_size: int,
    evidence_out: Path,
    predictions_path: Path = PREDICTIONS,
    withhold_match: bool = False,
    only_matched: bool = False,
    match_policy: str = "strict",
    target_policy: str = "top1",
    sample_strategy: str = "activity",
) -> list[dict]:
    train, validation, test, user_enc, item_enc = load_interaction_splits()
    user_dec = {value: key for key, value in user_enc.items()}
    item_dec = {value: key for key, value in item_enc.items()}
    predictions = pd.read_parquet(predictions_path)
    restaurants = pd.read_parquet(RESTAURANTS).set_index("place_id", drop=False)
    user_llm = pd.read_parquet(USER_LLM).set_index("contributor_id", drop=False)
    item_llm = pd.read_parquet(ITEM_LLM).set_index("place_id", drop=False)
    heldout_ids = set(validation.review_id.dropna().astype(str)) | set(
        test.review_id.dropna().astype(str)
    )
    train_counts = train.groupby("user_idx").size()
    candidates = predictions[["user_idx", "true_item_idx", "top10_place_ids"]].copy()
    candidates["true_place_id"] = candidates.true_item_idx.map(item_dec)
    candidates["hit_rank"] = candidates.apply(
        lambda row: (
            list(row.top10_place_ids).index(row.true_place_id) + 1
            if row.true_place_id in list(row.top10_place_ids) else np.nan
        ), axis=1,
    )
    test_metadata = (
        test[["user_idx", "rating", "review_id", "place_id"]]
        .drop_duplicates("user_idx", keep="last")
        .rename(columns={
            "rating": "heldout_rating", "review_id": "heldout_review_id",
            "place_id": "heldout_place_id",
        })
    )
    candidates = candidates.merge(test_metadata, on="user_idx", how="left")
    candidates["heldout_outcome"] = np.select(
        [candidates.heldout_rating >= 4, candidates.heldout_rating <= 2],
        ["liked", "disliked"], default="neutral",
    )
    if target_policy == "hit":
        candidates = candidates[candidates.hit_rank.notna()].copy()
    candidates["activity"] = candidates.user_idx.map(train_counts)
    candidates = candidates[candidates.activity.notna()].copy()
    candidates["stratum"] = pd.qcut(
        candidates.activity.rank(method="first"), 4,
        labels=["low", "medium_low", "medium_high", "high"],
    )
    candidates["hash"] = candidates.user_idx.map(lambda x: stable_hash(str(int(x))))
    if sample_strategy == "hit_outcome":
        if target_policy != "hit":
            raise ValueError("hit_outcome sampling requires target_policy='hit'")
        per_outcome = int(np.ceil(sample_size / 3))
        sampled = (
            candidates.sort_values("hash")
            .groupby("heldout_outcome", observed=True, group_keys=False)
            .head(per_outcome).head(sample_size)
        )
    else:
        per_stratum = int(np.ceil(sample_size / 4))
        sampled = (
            candidates.sort_values("hash").groupby("stratum", observed=True, group_keys=False)
            .head(per_stratum).head(sample_size)
        )

    sampled_users = set(sampled.user_idx.astype(int))
    train_by_user = {
        int(user): group.sort_values("timestamp_days_ago", ascending=True)
        for user, group in train[train.user_idx.isin(sampled_users)].groupby(
            "user_idx", sort=False
        )
    }
    target_places = {
        str(row.true_place_id) if target_policy == "hit" else str(list(row.top10_place_ids)[0])
        for row in sampled.itertuples(index=False)
    }
    needed_review_ids = set()
    for place_id in target_places:
        if place_id in item_llm.index:
            needed_review_ids.update(
                parse_json_list(item_llm.loc[place_id].get("source_review_ids_json", "[]"))
            )
    reviews = pd.read_parquet(
        REVIEWS,
        columns=["review_id", "place_id", "contributor_id", "text", "rating"],
        filters=[("review_id", "in", sorted(needed_review_ids))],
    ).drop_duplicates("review_id", keep="first").set_index("review_id", drop=False)

    bundles = []
    for sample_index, row in enumerate(sampled.itertuples(index=False), start=1):
        user_idx = int(row.user_idx)
        contributor_id = str(user_dec[user_idx])
        target_place = (
            str(row.true_place_id) if target_policy == "hit"
            else str(list(row.top10_place_ids)[0])
        )
        target = restaurants.loc[target_place] if target_place in restaurants.index else None
        user_features = user_llm.loc[contributor_id] if contributor_id in user_llm.index else None
        item_features = item_llm.loc[target_place] if target_place in item_llm.index else None
        evidence = []
        target_cuisine = "unknown"
        target_name = "Unknown restaurant"

        if target is not None:
            target_name = str(target.get("name", "Unknown restaurant"))
            target_cuisine = cuisine_from_llm(
                item_features, str(target.get("cuisine_category", ""))
            )
            fact = (
                f"{target_name} is a {target_cuisine} restaurant at "
                f"{target.get('address', 'an unknown address')}; Google rating "
                f"{target.get('rating', 'unknown')} from {target.get('user_rating_count', 'unknown')} ratings."
            )
            add_evidence(
                evidence, "R1", "restaurant_metadata", fact, "Google Places metadata",
                target_name, "recommended_restaurant", target_place,
            )

        history = train_by_user.get(user_idx, pd.DataFrame())
        history_places = []
        history_candidates = []
        for history_row in history.itertuples(index=False):
            place_id = str(history_row.place_id)
            metadata = restaurants.loc[place_id] if place_id in restaurants.index else None
            if metadata is None:
                continue
            history_cuisine = str(metadata.get("cuisine_category", "unknown cuisine"))
            shared = bool(cuisine_tokens(target_cuisine) & cuisine_tokens(history_cuisine))
            history_candidates.append((shared, float(history_row.rating), history_row, metadata))
        history_candidates.sort(key=lambda value: (value[0], value[1]), reverse=True)
        shared_history = None
        for position, (shared, _, history_row, metadata) in enumerate(
            history_candidates[:6], start=1
        ):
            place_id = str(history_row.place_id)
            history_places.append(place_id)
            history_name = str(metadata.get("name", place_id))
            fact = (
                f"The diner previously reviewed {history_name} "
                f"({metadata.get('cuisine_category', 'unknown cuisine')}) and gave it "
                f"{history_row.rating} stars."
            )
            add_evidence(
                evidence, f"H{position}", "training_history", fact,
                str(history_row.review_id), history_name, "history_restaurant", place_id,
            )
            # Under the strict explanation policy, cuisine overlap is evidence
            # of preference only when the diner previously liked that cuisine.
            # A low-rated same-cuisine visit is a relation, not a rationale.
            supported_shared = cuisine_history_aligns(
                shared, float(history_row.rating), match_policy
            )
            if supported_shared and shared_history is None:
                shared_history = (position, metadata.get("name", place_id), history_row.rating)

        user_attributes = top_attributes(user_features, "user_llm_")
        item_attributes = top_attributes(item_features, "llm_")
        for position, attribute in enumerate(user_attributes, start=1):
            add_evidence(
                evidence, f"U{position}", "llm_user_attribute",
                f"The diner's own training-review-derived preference signal for "
                f"{attribute['attribute']} is {attribute['score']} on a 0–1 scale.",
                "local LLM structured feature",
                "Diner", "diner", contributor_id,
            )
        for position, attribute in enumerate(item_attributes, start=1):
            add_evidence(
                evidence, f"A{position}", "llm_restaurant_attribute",
                f"{target_name}'s training-review-derived {attribute['attribute']} attribute is "
                f"{attribute['score']} on a 0–1 scale.",
                "local LLM structured feature",
                target_name, "recommended_restaurant", target_place,
            )

        match_fact = None
        match_source = None
        if shared_history is not None:
            history_position, history_name, history_rating = shared_history
            match_fact = (
                f"Shared-cuisine evidence: the diner gave {history_name} {history_rating} stars, "
                f"and {history_name} and {target_name} share the cuisine category {target_cuisine}."
            )
            match_source = f"exact relation match from H{history_position} and R1"
        else:
            user_by_name = {item["attribute"]: item["score"] for item in user_attributes}
            for item in item_attributes:
                user_value = user_by_name.get(item["attribute"])
                item_value = item["score"]
                compatible = user_value is not None and attributes_align(
                    item["attribute"], user_value, item_value, match_policy
                )
                if compatible:
                    direction = "high" if user_value >= 0.65 else "low"
                    match_fact = (
                        f"Shared-attribute evidence: the diner's {item['attribute']} preference "
                        f"is on the {direction} end ({user_value}), and {target_name}'s corresponding "
                        f"attribute is also on the {direction} end ({item_value})."
                    )
                    match_source = "exact compatible structured-attribute match"
                    break
        had_match = bool(match_fact)
        if match_fact and not withhold_match:
            add_evidence(
                evidence, "M1", "supported_personalization_match", match_fact, match_source,
                f"Diner and {target_name}", "recommendation_relation", target_place,
            )

        source_ids = parse_json_list(
            item_features.get("source_review_ids_json", "[]") if item_features is not None else "[]"
        )
        safe_reviews = []
        for review_id in source_ids:
            if review_id in heldout_ids or review_id not in reviews.index:
                continue
            review = reviews.loc[review_id]
            text = " ".join(str(review.get("text", "")).split())
            if len(text) < 20:
                continue
            safe_reviews.append((float(review.get("rating", 0) or 0), review_id, text))
        quote_position = 1
        for rating, review_id, text in sorted(safe_reviews, reverse=True):
            add_evidence(
                evidence, f"Q{quote_position}", "training_review_excerpt",
                f"A training-safe reviewer of {target_name} rated {target_name} "
                f"{rating:g} stars and wrote: "
                f"\"{text[:350]}\"",
                review_id,
                target_name, "recommended_restaurant", target_place,
            )
            quote_position += 1
            if quote_position > 3:
                break

        relations = [
            {"source": "USER", "relation": "VISITED_DURING_TRAINING", "target": place}
            for place in history_places
        ]
        relations.extend([
            {"source": target_place, "relation": "HAS_METADATA", "target": "R1"},
            {"source": target_place, "relation": "LOCATED_IN", "target": str(target.get("cbg", "")) if target is not None else ""},
        ])
        bundle = {
            "sample_id": sample_index,
            "user_idx": user_idx,
            "contributor_id": contributor_id,
            "activity_stratum": str(row.stratum),
            "training_interactions": int(row.activity),
            "recommended_place_id": target_place,
            "recommended_restaurant": target_name,
            "target_policy": target_policy,
            "evaluation_hit_at_10": bool(pd.notna(row.hit_rank)),
            "recommendation_rank": int(row.hit_rank) if pd.notna(row.hit_rank) else None,
            # Evaluation-only outcome metadata. It is intentionally outside the
            # evidence list and therefore never enters generation_prompt().
            "heldout_rating": float(row.heldout_rating) if pd.notna(row.heldout_rating) else None,
            "heldout_outcome": str(row.heldout_outcome),
            "match_available_before_withholding": had_match,
            "evidence": evidence,
            "graph_relations": relations,
            "heldout_review_ids_present": sorted(
                {str(item["source"]) for item in evidence} & heldout_ids
            ),
        }
        validate_evidence_bundle(bundle)
        if only_matched and not had_match:
            continue
        bundles.append(bundle)

    evidence_out.parent.mkdir(parents=True, exist_ok=True)
    evidence_out.write_text("".join(json.dumps(row) + "\n" for row in bundles))
    log.info("Prepared %d evidence subgraphs; held-out evidence occurrences=%d", bundles.__len__(), sum(bool(x["heldout_review_ids_present"]) for x in bundles))
    return bundles


def generation_prompt(bundle: dict) -> str:
    facts = "\n".join(
        f"[{x['id']}] ENTITY_TYPE={x['entity_type']} | ENTITY={x['entity']} | FACT={x['fact']}"
        for x in bundle["evidence"]
    )
    has_match = any(item["id"] == "M1" for item in bundle["evidence"])
    match_rule = (
        "- Sentence 1 MUST use [M1] to neutrally state the supported shared cuisine or "
        "attribute. Say that the evidence is shared; do not call it a great/perfect match."
        if has_match else
        "- No supported personalized match was retrieved. Do NOT claim that the diner will like "
        "the restaurant or connect unrelated history; give a factual recommendation summary instead."
    )
    return f"""Write a concise two-sentence personalized restaurant recommendation explanation.

Rules:
- Use only the evidence below. Do not infer amenities, dishes, quality, or preferences not stated.
{match_rule}
- Every evidence line names the entity it describes. Never transfer a fact, rating, attribute, or
  review from one entity to another.
- Sentence 2 should add a useful restaurant detail using R, A, or Q evidence. Do not suppress a
  material qualification merely to make the recommendation sound positive.
- Never say "great match", "perfect fit", "you will enjoy", "you might enjoy", or otherwise
  predict satisfaction. A shared cuisine or attribute is evidence of similarity, not proof of liking.
- Cite every factual claim using its evidence ID in square brackets.
- Paraphrase review evidence without quotation marks; do not use verbatim quotes. The `quotes`
  array must be empty.
- Do not mention private identifiers, graph mechanics, scores, or that this is an experiment.
- If evidence is sparse, state only what is supported.
- Return JSON: {{"explanation":"...", "evidence_ids":["R1"], "quotes":[]}}

EVIDENCE:
{facts}
"""


def judge_prompt(bundle: dict, generation: dict) -> str:
    facts = "\n".join(
        f"[{x['id']}] ENTITY_TYPE={x.get('entity_type', 'unspecified')} | "
        f"ENTITY={x.get('entity', 'unspecified')} | FACT={x['fact']}"
        for x in bundle["evidence"]
    )
    return f"""Audit the explanation strictly against the supplied evidence.

EVIDENCE:
{facts}

EXPLANATION:
{generation.get('explanation', '')}

Score each dimension independently from 1 (poor) to 5 (excellent):
- Entailment: only whether claims actually stated follow from their cited evidence without
  contradiction, entity transfer, or unsupported extension. Do not penalize omitted facts here.
- Personalization: whether the explanation correctly and specifically uses supported diner history,
  preferences, or an explicit match. Factual generic text may entail evidence but score low here.
- Usefulness: whether the supported content gives a reader decision-relevant information.
- Citation correctness: whether citations are attached to the propositions their evidence supports.
- Completeness: whether the explanation discloses evidence that materially contradicts or qualifies
  its own claims. Do not require it to repeat every evidence node.
- Unsupported claims: count one per asserted proposition not supported by a cited node. Never count
  omissions as unsupported claims.
- Attribution error: true only if a claim assigns evidence to the wrong diner or restaurant entity.

Return one JSON object with the fields entailment, personalization, usefulness,
citation_correctness, completeness, unsupported_claims, attribution_error, and reason. Use the
required schema; do not copy a fixed example or default all scores to the minimum. The reason must
identify the decisive evidence and explain any deduction.
Do not reward fluent unsupported text. Do not treat an omission as an entailment failure.
"""


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            return json.loads(match.group())
        raise


async def ollama_chat(session, semaphore, url, model, prompt, num_predict, schema):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": schema,
        "think": False,
        "options": {"temperature": 0.2, "num_predict": num_predict, "num_ctx": 8192},
        "keep_alive": "30m",
    }
    last_error = None
    for attempt in range(1, 5):
        try:
            async with semaphore:
                async with session.post(url.rstrip("/") + "/api/chat", json=payload) as response:
                    response.raise_for_status()
                    body = await response.json()
                    return extract_json(body["message"]["content"])
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            log.warning("Ollama response failed attempt %d/4: %s", attempt, error)
            if attempt < 4:
                await asyncio.sleep(1.5 * attempt)
    raise RuntimeError(f"Ollama response failed after four attempts: {last_error}")


async def process_bundle(
    session, semaphore, url, model, judge_model, secondary_judge_model, bundle
):
    generation = await ollama_chat(
        session, semaphore, url, model, generation_prompt(bundle), 300, GENERATION_SCHEMA
    )
    # Schema-constrained output should make this unnecessary, but enforce the
    # no-verbatim-quote contract defensively and retain the violation signal.
    quote_contract_violation = bool(generation.get("quotes"))
    generation["quotes"] = []
    judge = await ollama_chat(
        session, semaphore, url, judge_model, judge_prompt(bundle, generation), 420, JUDGE_SCHEMA
    )
    secondary_judge = None
    if secondary_judge_model:
        secondary_judge = await ollama_chat(
            session, semaphore, url, secondary_judge_model,
            judge_prompt(bundle, generation), 420, JUDGE_SCHEMA,
        )
    valid_ids = {item["id"] for item in bundle["evidence"]}
    cited_ids = set(re.findall(r"\[([A-Z]\d+)\]", str(generation.get("explanation", ""))))
    declared_ids = set(map(str, generation.get("evidence_ids", [])))
    evidence_text = " ".join(item["fact"] for item in bundle["evidence"])
    quotes = [str(value).strip() for value in generation.get("quotes", [])]
    deterministic = {
        "citations_valid": bool(cited_ids) and cited_ids <= valid_ids,
        "declared_evidence_valid": declared_ids <= valid_ids,
        "user_evidence_cited": any(value.startswith(("H", "U", "M")) for value in cited_ids),
        "restaurant_evidence_cited": any(
            value.startswith(("R", "A", "Q", "M")) for value in cited_ids
        ),
        "match_available": "M1" in valid_ids,
        "match_cited_when_available": "M1" not in valid_ids or "M1" in cited_ids,
        "quote_fidelity": all(quote in evidence_text and len(quote.split()) <= 12 for quote in quotes),
        "quote_contract_valid": not quote_contract_violation,
        "heldout_safe": not bundle["heldout_review_ids_present"],
    }
    return {
        **bundle, "generation": generation, "judge": judge,
        "secondary_judge": secondary_judge, "deterministic": deterministic,
    }


async def run_generation(
    bundles, url, model, judge_model, secondary_judge_model, concurrency,
    generations_out: Path,
):
    completed = {}
    if generations_out.exists():
        for line in generations_out.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed[int(row["sample_id"])] = row
    pending = [row for row in bundles if int(row["sample_id"]) not in completed]
    log.info("GraphRAG generation: %d complete, %d pending", len(completed), len(pending))
    timeout = aiohttp.ClientTimeout(total=900)
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            process_bundle(
                session, semaphore, url, model, judge_model, secondary_judge_model, row
            )
            for row in pending
        ]
        for future in asyncio.as_completed(tasks):
            result = await future
            completed[int(result["sample_id"])] = result
            generations_out.parent.mkdir(parents=True, exist_ok=True)
            with generations_out.open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            if len(completed) % 10 == 0 or len(completed) == len(bundles):
                log.info("Completed %d/%d explanations", len(completed), len(bundles))
    return [completed[key] for key in sorted(completed)]


async def rejudge_rows(
    rows: list[dict], url: str, judge_model: str,
    secondary_judge_model: str | None, concurrency: int, output: Path,
) -> list[dict]:
    """Apply the corrected rubric without regenerating any explanation."""
    completed = {}
    if output.exists():
        for line in output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed[int(row["sample_id"])] = row
    pending = [row for row in rows if int(row["sample_id"]) not in completed]
    timeout = aiohttp.ClientTimeout(total=900)
    semaphore = asyncio.Semaphore(concurrency)

    async def audit_one(session, row):
        primary = await ollama_chat(
            session, semaphore, url, judge_model,
            judge_prompt(row, row["generation"]), 420, JUDGE_SCHEMA,
        )
        secondary = None
        if secondary_judge_model:
            secondary = await ollama_chat(
                session, semaphore, url, secondary_judge_model,
                judge_prompt(row, row["generation"]), 420, JUDGE_SCHEMA,
            )
        return {**row, "original_judge": row.get("judge"), "judge": primary,
                "secondary_judge": secondary}

    output.parent.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [audit_one(session, row) for row in pending]
        for future in asyncio.as_completed(tasks):
            result = await future
            completed[int(result["sample_id"])] = result
            with output.open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            if len(completed) % 10 == 0 or len(completed) == len(rows):
                log.info("Rejudged %d/%d explanations", len(completed), len(rows))
    return [completed[key] for key in sorted(completed)]


async def secondary_rejudge_rows(
    rows: list[dict], url: str, secondary_judge_model: str,
    concurrency: int, output: Path,
) -> list[dict]:
    """Replace only the secondary judgment, preserving the primary judgment."""
    completed = {}
    if output.exists():
        for line in output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed[int(row["sample_id"])] = row
    pending = [row for row in rows if int(row["sample_id"]) not in completed]
    timeout = aiohttp.ClientTimeout(total=900)
    semaphore = asyncio.Semaphore(concurrency)

    async def audit_one(session, row):
        secondary = await ollama_chat(
            session, semaphore, url, secondary_judge_model,
            judge_prompt(row, row["generation"]), 420, JUDGE_SCHEMA,
        )
        return {**row, "secondary_judge": secondary}

    output.parent.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [audit_one(session, row) for row in pending]
        for future in asyncio.as_completed(tasks):
            result = await future
            completed[int(result["sample_id"])] = result
            with output.open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            if len(completed) % 10 == 0 or len(completed) == len(rows):
                log.info("Secondary-rejudged %d/%d explanations", len(completed), len(rows))
    return [completed[key] for key in sorted(completed)]


def quadratic_weighted_kappa(a: pd.Series, b: pd.Series) -> float:
    pairs = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"),
                          "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if pairs.empty:
        return float("nan")
    values = np.arange(1, 6)
    observed = np.zeros((5, 5), dtype=float)
    for left, right in zip(pairs.a.astype(int), pairs.b.astype(int)):
        if 1 <= left <= 5 and 1 <= right <= 5:
            observed[left - 1, right - 1] += 1
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    weights = ((values[:, None] - values[None, :]) / 4) ** 2
    denominator = (weights * expected).sum()
    return float(1 - (weights * observed).sum() / denominator) if denominator else float("nan")


def summarize(
    rows: list[dict], model: str, judge_model: str,
    secondary_judge_model: str | None, results_out: Path, summary_out: Path,
    run_label: str,
):
    flat = []
    for row in rows:
        flat.append({
            "sample_id": row["sample_id"],
            "user_idx": row["user_idx"],
            "recommended_place_id": row["recommended_place_id"],
            "target_policy": row.get("target_policy", "top1"),
            "evaluation_hit_at_10": row.get("evaluation_hit_at_10"),
            "recommendation_rank": row.get("recommendation_rank"),
            "heldout_rating": row.get("heldout_rating"),
            "heldout_outcome": row.get("heldout_outcome"),
            "activity_stratum": row["activity_stratum"],
            "training_interactions": row["training_interactions"],
            "evidence_count": len(row["evidence"]),
            "explanation": row["generation"].get("explanation", ""),
            "entailment": row["judge"].get("entailment"),
            "personalization": row["judge"].get("personalization"),
            "usefulness": row["judge"].get("usefulness"),
            "citation_correctness": row["judge"].get("citation_correctness"),
            "completeness": row["judge"].get("completeness"),
            "unsupported_claims": row["judge"].get("unsupported_claims"),
            "attribution_error": row["judge"].get("attribution_error"),
            "original_entailment": (row.get("original_judge") or {}).get("entailment"),
            "secondary_entailment": (row.get("secondary_judge") or {}).get("entailment"),
            "secondary_personalization": (row.get("secondary_judge") or {}).get("personalization"),
            "secondary_usefulness": (row.get("secondary_judge") or {}).get("usefulness"),
            "secondary_citation_correctness": (row.get("secondary_judge") or {}).get("citation_correctness"),
            "secondary_completeness": (row.get("secondary_judge") or {}).get("completeness"),
            "secondary_attribution_error": (row.get("secondary_judge") or {}).get("attribution_error"),
            **row["deterministic"],
        })
    frame = pd.DataFrame(flat)
    results_out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(results_out, index=False)
    score_cols = ["entailment", "personalization", "usefulness", "citation_correctness", "completeness"]
    for column in score_cols + ["unsupported_claims"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    lines = [
        f"# GraphRAG remediation audit: {run_label}", "",
        f"Generator: `{model}`  ",
        f"Primary judge: `{judge_model}`  ",
        f"Secondary judge: `{secondary_judge_model or 'not run'}`", "",
        f"Sample: {len(frame)} deterministic recommendations from the seed-42 frozen "
        f"proximity output; target policy `{frame.target_policy.iloc[0]}`.", "",
        "The ranker was not changed. Evidence contains training-only user history, "
        "training-safe review excerpts, restaurant metadata, cuisine/CBG relations, and "
        "LLM attributes generated from corpora with held-out review text excluded.", "",
        "## Automated audit", "",
        *[f"- {column.replace('_', ' ').title()}: {frame[column].mean():.3f} / 5" for column in score_cols],
        f"- Unsupported claims: {frame.unsupported_claims.mean():.3f} per explanation",
        f"- Attribution-error rate: {frame.attribution_error.fillna(False).astype(bool).mean():.1%}",
        f"- Valid inline citations: {frame.citations_valid.mean():.1%}",
        f"- User evidence cited: {frame.user_evidence_cited.mean():.1%}",
        f"- Restaurant evidence cited: {frame.restaurant_evidence_cited.mean():.1%}",
        f"- Supported personalization-match coverage: {frame.match_available.mean():.1%}",
        f"- Match cited when available: {frame.match_cited_when_available.mean():.1%}",
        f"- Exact short-quote fidelity: {frame.quote_fidelity.mean():.1%}",
        f"- Held-out-safe evidence bundles: {frame.heldout_safe.mean():.1%}", "",
        "The quality scores are local-LLM judgments and must be supplemented by a blinded "
        "human audit before publication. Citation validity, quote-contract compliance, and held-out "
        "safety are deterministic checks.", "",
    ]
    if frame.original_entailment.notna().any():
        original = pd.to_numeric(frame.original_entailment, errors="coerce")
        current = pd.to_numeric(frame.entailment, errors="coerce")
        lines.extend([
            "## Rubric-only comparison", "",
            f"- Original low-entailment share (1–2): {(original <= 2).mean():.1%}",
            f"- Corrected-rubric low-entailment share (1–2): {(current <= 2).mean():.1%}",
            f"- Cases moving by at least two points: {((current-original).abs() >= 2).sum()}", "",
        ])
    if secondary_judge_model and frame.secondary_entailment.notna().any():
        lines.extend(["## Inter-judge agreement", ""])
        for column in score_cols:
            other = frame[f"secondary_{column}"]
            valid = frame[[column, f"secondary_{column}"]].dropna()
            exact = (valid[column] == valid[f"secondary_{column}"]).mean()
            rho = valid[column].corr(valid[f"secondary_{column}"], method="spearman")
            kappa = quadratic_weighted_kappa(valid[column], valid[f"secondary_{column}"])
            lines.append(
                f"- {column.replace('_', ' ').title()}: exact={exact:.1%}, "
                f"Spearman={rho:.3f}, quadratic-weighted kappa={kappa:.3f}"
            )
        valid_attr = frame[["attribution_error", "secondary_attribution_error"]].dropna()
        disagreement = (
            valid_attr.attribution_error.astype(bool)
            != valid_attr.secondary_attribution_error.astype(bool)
        ).mean()
        lines.extend([f"- Attribution-error disagreement: {disagreement:.1%}", ""])
    lines.extend(["## Examples", ""])
    for row in rows[:8]:
        lines.extend([
            f"### Sample {row['sample_id']}", "",
            row["generation"].get("explanation", ""), "",
        ])
    summary_out.write_text("\n".join(lines))
    log.info("Wrote %s and %s", results_out, summary_out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["prepare", "generate", "rejudge", "secondary-rejudge"],
        default="generate",
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--secondary-judge-model", default=DEFAULT_SECONDARY_JUDGE)
    parser.add_argument("--no-secondary-judge", action="store_true")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--source-generations", type=Path, default=GENERATIONS_OUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-label", default="remediation-v2")
    parser.add_argument("--withhold-match", action="store_true")
    parser.add_argument("--only-matched", action="store_true")
    parser.add_argument("--match-policy", choices=["legacy", "strict"], default="strict")
    parser.add_argument("--target-policy", choices=["top1", "hit"], default="top1")
    parser.add_argument(
        "--sample-strategy", choices=["activity", "hit_outcome"], default="activity",
    )
    # Backward-compatible alias; it now selects the versioned prepare mode.
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    if args.prepare_only:
        args.mode = "prepare"
    output_dir = args.output_dir.resolve()
    frozen_dir = OUT_DIR.resolve()
    if output_dir == frozen_dir:
        parser.error("--output-dir must not be the frozen results/graphrag directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_out = output_dir / "evidence.jsonl"
    generations_out = output_dir / "generations.jsonl"
    results_out = output_dir / "results.parquet"
    summary_out = output_dir / "summary.md"
    secondary = None if args.no_secondary_judge else args.secondary_judge_model

    if args.mode == "rejudge":
        rows = [
            json.loads(line) for line in args.source_generations.read_text().splitlines()
            if line.strip()
        ]
        rows = asyncio.run(rejudge_rows(
            rows, args.url, args.judge_model, secondary, args.concurrency, generations_out
        ))
        summarize(
            rows, "frozen original generations", args.judge_model, secondary,
            results_out, summary_out, args.run_label,
        )
        return
    if args.mode == "secondary-rejudge":
        if not secondary:
            parser.error("secondary-rejudge requires --secondary-judge-model")
        rows = [
            json.loads(line) for line in args.source_generations.read_text().splitlines()
            if line.strip()
        ]
        rows = asyncio.run(secondary_rejudge_rows(
            rows, args.url, secondary, args.concurrency, generations_out
        ))
        summarize(
            rows, "frozen original generations", args.judge_model, secondary,
            results_out, summary_out, args.run_label,
        )
        return

    bundles = prepare_evidence(
        args.sample_size, evidence_out, args.predictions,
        withhold_match=args.withhold_match, only_matched=args.only_matched,
        match_policy=args.match_policy, target_policy=args.target_policy,
        sample_strategy=args.sample_strategy,
    )
    if args.mode == "prepare":
        return
    rows = asyncio.run(run_generation(
        bundles, args.url, args.model, args.judge_model, secondary,
        args.concurrency, generations_out,
    ))
    summarize(
        rows, args.model, args.judge_model, secondary,
        results_out, summary_out, args.run_label,
    )


if __name__ == "__main__":
    main()
