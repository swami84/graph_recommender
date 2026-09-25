from foodie.features.build_llm_features_ollama import CONFIDENCE_FIELDS, _validate_raw


def test_exact_duplicate_labels_are_normalized() -> None:
    raw = {
        "scores": {},
        "confidences": {name: 0.5 for name in CONFIDENCE_FIELDS},
        "cuisines": ["indian", "indian", "cafe_bakery"],
        "meal_periods": ["lunch", "lunch"],
    }
    _validate_raw(raw)
    assert raw["cuisines"] == ["indian", "cafe_bakery"]
    assert raw["meal_periods"] == ["lunch"]
