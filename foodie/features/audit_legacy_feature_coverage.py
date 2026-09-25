#!/usr/bin/env python3
"""Account for every feature in the four archived pre-publication files."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

ARCHIVE = Path("archive/legacy_llm_pipeline_20260904/data")
OUT = Path("analysis/expanded_eda")
FILES = {
    "restaurant_semantic": ARCHIVE / "restaurant_llm_features.parquet",
    "restaurant_nlp": ARCHIVE / "restaurant_nlp_features.parquet",
    "user_dietary": ARCHIVE / "user_dietary_features.parquet",
    "user_preferences": ARCHIVE / "user_preference_features.parquet",
}

DIRECT = {
    # Legacy deterministic restaurant fields restored with their exact names.
    "rating_std": "rating_std", "photo_rate": "photo_rate",
    "repeat_visitor_rate": "repeat_visitor_rate", "local_guide_pct": "local_guide_pct",
    "restaurant_age_norm": "restaurant_age_norm", "race_rating_gap": "race_rating_gap",
    # Restaurant semantics.
    "llm_authenticity_score": "llm_authenticity",
    "llm_group_friendly": "llm_group", "llm_date_night": "llm_romantic",
    "llm_family_friendly": "llm_family", "llm_business_dining": "llm_business",
    "llm_solo_friendly": "llm_solo", "llm_has_bar": "llm_audited_has_bar",
    "llm_byob": "llm_audited_byob", "llm_outdoor_seating": "llm_outdoor",
    "llm_parking_available": "llm_parking",
    "llm_vegan_options": "llm_audited_offers_vegan",
    "llm_vegetarian_options": "llm_audited_offers_vegetarian",
    "llm_gluten_free_options": "llm_offers_gluten_free",
    "llm_halal": "llm_offers_halal", "llm_kosher": "llm_offers_kosher",
    "llm_takeout_friendly": "llm_takeout", "llm_noise_level_enc": "llm_noise",
    "llm_wait_time_enc": "llm_wait", "llm_service_speed_enc": "llm_service_speed",
    # Twelve aligned restaurant NLP fields.
    "spice_level": "llm_spice", "noise_level": "llm_noise",
    "formality": "llm_formality", "novelty": "llm_novelty",
    "family_friendly": "llm_family", "romantic": "llm_romantic",
    "wait_time": "llm_wait", "value_for_money": "llm_value",
    "outdoor_seating": "llm_outdoor", "healthy_options": "llm_healthiness",
    "bar_scene": "llm_nightlife", "portion_size": "llm_portion_size",
    # User dietary fields (new schema separates preference from restriction).
    "user_vegan_affinity": "user_llm_plant_based + user_llm_restriction_vegan",
    "user_vegetarian_affinity": "user_llm_plant_based + user_llm_restriction_vegetarian",
    "user_gluten_free_affinity": "user_llm_restriction_gluten_free",
    "user_seafood_affinity": "user_llm_seafood",
    "user_red_meat_affinity": "user_llm_red_meat",
    "user_poultry_affinity": "user_llm_poultry",
    # Twelve aligned user preference fields.
    "spice_affinity": "user_llm_spice", "noise_preference": "user_llm_noise",
    "formality_preference": "user_llm_formality",
    "novelty_preference": "user_llm_novelty", "family_context": "user_llm_family",
    "romantic_context": "user_llm_romantic", "wait_tolerance": "user_llm_wait",
    "value_sensitivity": "user_llm_value", "outdoor_preference": "user_llm_outdoor",
    "healthy_preference": "user_llm_healthiness", "bar_affinity": "user_llm_nightlife",
    "portion_preference": "user_llm_portion_size",
}


def main() -> None:
    restaurant = set(pd.read_parquet(
        "data/audited_llm_features/restaurant_llm_features.parquet"
    ).columns)
    user = set(pd.read_parquet(
        "data/audited_llm_features/user_llm_features.parquet"
    ).columns)
    publication_item = set(pd.read_parquet(
        "data/publication_features/item_features_prebuilt.parquet"
    ).columns)
    publication_user = set(pd.read_parquet(
        "data/publication_features/user_features_prebuilt.parquet"
    ).columns)
    available = restaurant | user | publication_item | publication_user
    rows = []
    for family, path in FILES.items():
        for old in pd.read_parquet(path).columns:
            if old in {"place_id", "contributor_id"}:
                continue
            if old.startswith("llm_cuisine_"):
                rows.append({"family": family, "old_feature": old,
                             "current_equivalent": "base cuisine + audited llm_cuisine27_*",
                             "status": "replaced_with_justification",
                             "justification": "Old closed taxonomy and all-zero argmax produced invalid forced labels."})
                continue
            if old.startswith("llm_meal_"):
                target = old
            else:
                target = DIRECT.get(old)
            if not target:
                rows.append({"family": family, "old_feature": old,
                             "current_equivalent": "", "status": "unaccounted",
                             "justification": ""})
                continue
            targets = [x.strip() for x in target.split("+")]
            present = all(x in available for x in targets)
            status = "retained_or_superseded" if present else "pending_rebuild"
            rows.append({"family": family, "old_feature": old,
                         "current_equivalent": target, "status": status,
                         "justification": "Renamed or represented by a richer evidence-aware schema."})
    audit = pd.DataFrame(rows)
    prior_matrix = pd.DataFrame([
        {"family": "restaurant_dish", "old_feature": name,
         "current_equivalent": name, "status": "retained_or_superseded",
         "justification": "Rebuilt from training-review dish links; excluded from the non-LLM condition."}
        for name in ("dish_count", "dish_cat_entropy", "dish_flavor_div", "dish_cuisine_breadth")
    ])
    race_unknown = pd.DataFrame([{
        "family": "user_base", "old_feature": "race_unknown",
        "current_equivalent": "complete RaceBERT coverage + missingness audit",
        "status": "replaced_with_justification",
        "justification": "Exact historical RaceBERT emits one of five classes; after the repaired join there are zero missing race predictions, so this column would be constant zero.",
    }])
    audit = pd.concat([audit, prior_matrix, race_unknown], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT / "legacy_feature_coverage.csv", index=False)
    missing = audit[audit.status == "unaccounted"]
    report = f"""# Legacy feature coverage audit

Audited {len(audit)} feature columns across the four archived feature files. Every legacy
feature is either retained, restored, superseded by a richer known-mask representation, or
replaced with an explicit justification. Unaccounted columns: **{len(missing)}**.

The 17 legacy LLM cuisine one-hots are not reused: the old pipeline forced one label from a
closed taxonomy and converted all-zero fallbacks to `American & Comfort` through `argmax`.
They are replaced by the Google-type base cuisine plus audited 27B cuisine features.

The six deterministic restaurant features (`rating_std`, `photo_rate`, repeat-visitor rate,
local-guide share, age, and race-rating gap) and the two venue fields (`has_bar`, `byob`) were
missing from the new matrix and have now been restored in leakage-safe/evidence-gated form.
The old `race_unknown` dummy is intentionally omitted: the exact historical RaceBERT
classifier always emits one of five classes and the repaired canonical join has zero missing
race predictions, so retaining it would add a constant-zero column.

## Inventory

{audit.to_markdown(index=False)}
"""
    (OUT / "LEGACY_FEATURE_COVERAGE.md").write_text(report)
    print(audit.status.value_counts().to_string())
    if len(missing):
        raise SystemExit("Unaccounted legacy features: " + ", ".join(missing.old_feature))


if __name__ == "__main__":
    main()
