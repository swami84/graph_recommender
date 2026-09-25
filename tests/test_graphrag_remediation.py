import pytest

from foodie.explanations.run_publication_graphrag import (
    add_evidence,
    attributes_align,
    cuisine_history_aligns,
    generation_prompt,
    judge_prompt,
    validate_evidence_bundle,
)


def _bundle():
    evidence = []
    add_evidence(
        evidence, "R1", "restaurant_metadata", "Cafe A is in Boston.", "metadata",
        "Cafe A", "recommended_restaurant", "place-a",
    )
    add_evidence(
        evidence, "Q1", "training_review_excerpt",
        'A training-safe reviewer of Cafe A rated Cafe A 5 stars and wrote: "Good food."',
        "review-1", "Cafe A", "recommended_restaurant", "place-a",
    )
    return {"evidence": evidence}


def test_evidence_requires_entity_attribution():
    with pytest.raises(ValueError):
        add_evidence([], "R1", "metadata", "fact", "source", "", "")


def test_review_evidence_names_recommended_restaurant():
    bundle = _bundle()
    validate_evidence_bundle(bundle)
    bundle["evidence"][1]["fact"] = 'A training-safe reviewer rated it 5 stars and wrote: "Good."'
    with pytest.raises(ValueError):
        validate_evidence_bundle(bundle)


def test_prompts_encode_entity_and_rubric_boundaries():
    bundle = _bundle()
    generated = {"explanation": "Cafe A is in Boston [R1].", "evidence_ids": ["R1"], "quotes": []}
    gp = generation_prompt(bundle)
    jp = judge_prompt(bundle, generated)
    assert "ENTITY_TYPE=recommended_restaurant" in gp
    assert "Never transfer a fact" in gp
    assert '"quotes":[]' in gp
    assert "Do not penalize omitted facts" in jp
    assert "Attribution error" in jp
    assert "fields entailment, personalization, usefulness" in jp
    assert "do not copy a fixed example" in jp


def test_heldout_outcome_metadata_is_not_serialized_into_prompt():
    bundle = _bundle() | {
        "heldout_rating": 1.0,
        "heldout_outcome": "disliked",
        "recommendation_rank": 4,
    }
    prompt = generation_prompt(bundle)
    assert "heldout" not in prompt.lower()
    assert "disliked" not in prompt.lower()
    assert "recommendation_rank" not in prompt


def test_strict_attribute_policy_excludes_ambiguous_wait_and_large_gaps():
    assert attributes_align("value", 0.9, 0.8, "strict")
    assert not attributes_align("value", 0.2, 0.2, "strict")
    assert attributes_align("noise", 0.2, 0.3, "strict")
    assert not attributes_align("wait", 0.8, 0.8, "strict")
    assert not attributes_align("value", 1.0, 0.6, "strict")
    assert attributes_align("value", 1.0, 0.6, "legacy")


def test_strict_cuisine_match_requires_a_liked_history_visit():
    assert cuisine_history_aligns(True, 4.0, "strict")
    assert cuisine_history_aligns(True, 5.0, "strict")
    assert not cuisine_history_aligns(True, 3.0, "strict")
    assert not cuisine_history_aligns(True, 1.0, "strict")
    assert cuisine_history_aligns(True, 1.0, "legacy")
    assert not cuisine_history_aligns(False, 5.0, "strict")
