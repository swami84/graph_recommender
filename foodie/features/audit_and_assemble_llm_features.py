#!/usr/bin/env python3
"""Audit evidence-sensitive LLM fields and assemble publication-safe features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DATA = Path("data")
ANALYSIS = Path("analysis/expanded_eda")
OLD_RESTAURANT = DATA / "restaurant_llm_features.parquet"
OLD_USER = DATA / "user_llm_features.parquet"
CORPUS = DATA / "llm_corpora/restaurant_train_corpus.parquet"
REPAIRED_CUISINE = DATA / "restaurant_cuisine_27b_v4.parquet"
OUT_DIR = DATA / "audited_llm_features"

# Conservative lexical evidence gates. They are intentionally precision-first:
# unsupported scores become unknown rather than neutral 0.5 values.
EVIDENCE_PATTERNS = {
    "spice": r"\bspic(?:e|y|ier|iest|iness)|\bhot sauce\b|\bchili\b|\bchilli\b|\bjalape",
    "outdoor": r"\boutdoor|\bpatio|\bsidewalk seating|\brooftop|\bal fresco",
    "value": r"\bprice|\bpriced|\bpricing|\bvalue|\bafford|\bcheap|\bexpensive|\boverpriced|\bworth\b|\bcost\b|\bportion",
    "service_sentiment": r"\bservice|\bstaff|\bserver|\bwaiter|\bwaitress|\bhostess|\bcashier|\bemployee|\bmanager",
    "service_speed": r"\bquick|\bfast|\bslow|\bwait(?:ed|ing)?\b|\bminute|\bspeed|\bprompt|\befficient",
}
DIETARY_PATTERNS = {
    "vegan": r"vegan_restaurant|\bvegan[- ]friendly\b|\bvegan (?:option|menu|dish|food|pizza|cheese|pastr|dessert)|\b(?:option|menu|dish|food|pizza|cheese|pastr|dessert)[^.!?]{0,30}\bvegan\b|\bis vegan\b",
    "vegetarian": r"vegetarian_restaurant|\bvegetarian[- ]friendly\b|\bvegetarian (?:option|menu|dish|food|meal)|\b(?:option|menu|dish|food|meal)[^.!?]{0,30}\bvegetarian\b|\bveggie (?:option|menu|dish|burger)|\bmeatless (?:option|menu|dish|meal)",
}
VENUE_PATTERNS = {
    "has_bar": r"(?:^|,)(?:bar|pub|cocktail_bar|wine_bar|sports_bar)(?:,|$)|\bfull bar\b|\bhas (?:a |the )?bar\b|\bbar (?:area|seating|menu)\b|\bat the bar\b",
    "byob": r"\bbyob\b|\bbring your own (?:beer|wine|bottle)\b",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial-cuisine", action="store_true")
    args = parser.parse_args()

    restaurant = pd.read_parquet(OLD_RESTAURANT)
    user = pd.read_parquet(OLD_USER)
    corpus = pd.read_parquet(CORPUS, columns=["place_id", "name", "types", "corpus"])
    restaurants = pd.read_parquet(
        DATA / "restaurants_enriched.parquet", columns=["place_id", "cuisine_category"]
    )
    cuisine = pd.read_parquet(REPAIRED_CUISINE)
    expected = set(restaurants.loc[
        restaurants["cuisine_category"].fillna("Other").eq("Other"), "place_id"
    ].astype(str))
    actual = set(cuisine["place_id"].astype(str))
    missing = expected - actual
    if missing and not args.allow_partial_cuisine:
        raise SystemExit(f"Cuisine repair incomplete: {len(missing):,} of {len(expected):,} missing")

    # Use the complete catalogue as the spine. Some restaurants have no general
    # semantic row, but targeted cuisine repair can still cover them; dropping
    # those rows here silently discarded valid repaired cuisines.
    joined = (restaurants[["place_id"]]
              .merge(restaurant, on="place_id", how="left", validate="one_to_one")
              .merge(corpus, on="place_id", how="left", validate="one_to_one"))
    text = (joined["types"].fillna("") + " " + joined["corpus"].fillna("")).str.lower()
    audit_rows = []
    sample_rows = []
    for feature, pattern in EVIDENCE_PATTERNS.items():
        value_col, known_col = f"llm_{feature}", f"llm_{feature}_known"
        evidence = text.str.contains(pattern, regex=True)
        old_known = joined[known_col].fillna(0).gt(0)
        high = old_known & joined[value_col].ge(.75)
        low = old_known & joined[value_col].le(.25)
        audit_rows.append({
            "feature": feature, "decision": "retain_with_evidence_gate",
            "known_before": int(old_known.sum()), "known_after": int((old_known & evidence).sum()),
            "evidence_share_known": float(evidence[old_known].mean()),
            "evidence_share_high": float(evidence[high].mean()) if high.any() else None,
            "evidence_share_low": float(evidence[low].mean()) if low.any() else None,
        })
        joined[known_col] = (old_known & evidence).astype("float32")
        for band, mask in (("high", high), ("low", low)):
            sample = joined.loc[mask].sample(n=min(5, int(mask.sum())), random_state=42)
            for _, row in sample.iterrows():
                sample_rows.append({
                    "feature": feature, "band": band, "name": row["name"],
                    "score": row[value_col], "has_lexical_evidence": bool(evidence.loc[row.name]),
                    "source_excerpt": " ".join(str(row["corpus"]).split())[:500],
                })

    # The old binary availability fields fail the audit: most asserted zeros and
    # many asserted ones lack direct evidence. Replace them with positive-only,
    # evidence-backed indicators rather than treating silence as absence.
    for dietary, pattern in DIETARY_PATTERNS.items():
        evidence = text.str.contains(pattern, regex=True)
        joined[f"llm_audited_offers_{dietary}"] = evidence.astype("float32")
        joined[f"llm_audited_offers_{dietary}_known"] = evidence.astype("float32")
        audit_rows.append({
            "feature": f"offers_{dietary}", "decision": "replace_with_positive_evidence_only",
            "known_before": int(joined[f"llm_offers_{dietary}_known"].fillna(0).gt(0).sum()),
            "known_after": int(evidence.sum()),
            "evidence_share_known": float(evidence[
                joined[f"llm_offers_{dietary}_known"].fillna(0).gt(0)
            ].mean()), "evidence_share_high": None, "evidence_share_low": None,
        })

    for feature, pattern in VENUE_PATTERNS.items():
        evidence = text.str.contains(pattern, regex=True)
        joined[f"llm_audited_{feature}"] = evidence.astype("float32")
        joined[f"llm_audited_{feature}_known"] = evidence.astype("float32")
        audit_rows.append({
            "feature": feature, "decision": "restore_as_positive_evidence_only",
            "known_before": None, "known_after": int(evidence.sum()),
            "evidence_share_known": None, "evidence_share_high": None,
            "evidence_share_low": None,
        })

    old_cuisine_cols = [c for c in joined if c.startswith("llm_cuisine_")]
    failed_cols = old_cuisine_cols + ["llm_cuisine_confidence"]
    failed_cols += [
        "llm_offers_vegan", "llm_offers_vegan_known",
        "llm_offers_vegetarian", "llm_offers_vegetarian_known",
    ]
    joined = joined.drop(columns=[c for c in failed_cols if c in joined])

    cuisine = cuisine.copy()
    cuisine["place_id"] = cuisine["place_id"].astype(str)
    cuisine["primary_cuisine"] = (
        cuisine["primary_cuisine"].astype(str).str.strip().str.lower()
        .replace({"other": "unknown", "": "unknown", "none": "unknown", "nan": "unknown"})
    )
    dummies = pd.get_dummies(
        cuisine.set_index("place_id")["primary_cuisine"], prefix="llm_cuisine27", dtype=float
    ).reset_index()
    cuisine_numeric = cuisine[["place_id", "confidence"]].rename(
        columns={"confidence": "llm_cuisine27_confidence"}
    ).merge(dummies, on="place_id", how="left", validate="one_to_one")
    joined = joined.merge(cuisine_numeric, on="place_id", how="left", validate="one_to_one")
    new_cuisine_cols = [c for c in joined if c.startswith("llm_cuisine27_")]
    joined[new_cuisine_cols] = joined[new_cuisine_cols].fillna(0).astype("float32")

    provenance = {"name", "types", "corpus"}
    restaurant_out = joined.drop(columns=[c for c in provenance if c in joined])

    # User cuisine preferences came from the same invalid fixed taxonomy and are
    # excluded until a separate user-preference audit is performed.
    user_failed = [c for c in user if c.startswith("user_llm_cuisine_")]
    user_failed += ["user_llm_cuisine_confidence"]
    user_out = user.drop(columns=[c for c in user_failed if c in user])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    restaurant_path = OUT_DIR / "restaurant_llm_features.parquet"
    user_path = OUT_DIR / "user_llm_features.parquet"
    restaurant_out.to_parquet(restaurant_path, index=False)
    user_out.to_parquet(user_path, index=False)

    audit = pd.DataFrame(audit_rows)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    audit.to_csv(ANALYSIS / "llm_feature_evidence_spot_audit.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(
        ANALYSIS / "llm_feature_evidence_spot_samples.csv", index=False
    )
    decisions = {
        "status": "publication_feature_gate",
        "retained_with_evidence_gate": list(EVIDENCE_PATTERNS),
        "replaced": ["restaurant cuisine", "offers_vegan", "offers_vegetarian"],
        "restored_from_legacy": ["has_bar", "byob"],
        "excluded": ["old restaurant cuisine one-hots", "old user cuisine preferences"],
        "temporarily_retained_unaudited": "all other non-cuisine v5 structured fields",
        "cuisine_expected": len(expected), "cuisine_available": len(actual),
        "restaurant_output": str(restaurant_path), "user_output": str(user_path),
    }
    (OUT_DIR / "audit_decisions.json").write_text(json.dumps(decisions, indent=2) + "\n")
    report = f"""# LLM feature publication gate

The old cuisine one-hots were rejected because their closed taxonomy forced invalid labels.
They are replaced by the v4 evidence-rule + constrained-27B repair ({len(actual):,}/{len(expected):,}
target restaurants currently available). Old user cuisine preferences are excluded because
they share the rejected taxonomy.

The old vegan/vegetarian availability scores also failed: absence of a mention had often been
treated as evidence of absence. They are replaced with conservative positive-only indicators.

Spice, outdoor seating, value, service sentiment, and service speed are retained only when the
source types/reviews contain direct lexical evidence. Other non-cuisine structured fields remain
temporarily retained, as requested, pending later article audit.

{audit.to_markdown(index=False, floatfmt='.3f')}
"""
    (ANALYSIS / "LLM_FEATURE_PUBLICATION_GATE.md").write_text(report)
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
