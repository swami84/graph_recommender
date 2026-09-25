#!/usr/bin/env python3
"""Create a blinded, activity-stratified worksheet for GraphRAG human review."""

import json
from pathlib import Path

import pandas as pd


SOURCE = Path("results/graphrag/publication_graphrag_generations.jsonl")
OUTPUT = Path("results/graphrag/publication_graphrag_human_audit.csv")
SEED = 20260909


def main():
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    if len(rows) != 100 or len({row["sample_id"] for row in rows}) != 100:
        raise RuntimeError("Human audit requires the complete 100-example GraphRAG run")

    frame = pd.DataFrame(rows)
    selected = (
        frame.groupby("activity_stratum", group_keys=False)
        .sample(n=5, random_state=SEED)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    audit_rows = []
    for index, row in selected.iterrows():
        evidence = "\n".join(
            f"[{item['id']}] {item['fact']}" for item in row["evidence"]
        )
        audit_rows.append({
            "audit_id": f"A{index + 1:02d}",
            "activity_stratum": row["activity_stratum"],
            "explanation": row["generation"].get("explanation", ""),
            "evidence": evidence,
            "entailment_1_to_5": "",
            "personalization_1_to_5": "",
            "usefulness_1_to_5": "",
            "citation_correctness_1_to_5": "",
            "unsupported_claim_count": "",
            "notes": "",
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT} with {len(audit_rows)} blinded examples")


if __name__ == "__main__":
    main()
