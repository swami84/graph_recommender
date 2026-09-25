#!/usr/bin/env python3
"""Build a readable audit of genuine Hit@10 GraphRAG examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evidence_by_prefix(row: dict, prefix: str) -> list[dict]:
    return [item for item in row["evidence"] if str(item["id"]).startswith(prefix)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-per-outcome", type=int, default=2)
    args = parser.parse_args()

    rows = sorted(load_jsonl(args.input), key=lambda row: int(row["sample_id"]))
    if not rows or not all(row.get("evaluation_hit_at_10") for row in rows):
        raise ValueError("Input must contain genuine Hit@10 examples")

    counts = {
        outcome: sum(row.get("heldout_outcome") == outcome for row in rows)
        for outcome in ("liked", "neutral", "disliked")
    }
    # Retain the first deterministic cases in each outcome group rather than
    # choosing examples based on favorable audit scores.
    selected = [
        row
        for outcome in ("liked", "neutral", "disliked")
        for row in [r for r in rows if r.get("heldout_outcome") == outcome][
            : args.details_per_outcome
        ]
    ]

    lines = [
        "# Explanations for genuine Hit@10 cases", "",
        "These are illustrative cases from the frozen KGAT-SAL + LLM + proximity ranking. "
        "They are not an additional 1,000-case evaluation. The held-out restaurant appears "
        "within the model's top 10 for every case below.", "",
        "The held-out star rating is shown only after generation to classify the observed "
        "outcome. It was excluded from the retrieved evidence and from the LLM prompt. Thus, "
        "the explanation answers why the system could support the suggestion from prior "
        "evidence; it does not use the future rating to rationalize the result.", "",
        f"Balanced illustration set: {len(rows)} cases — {counts['liked']} liked (4–5 stars), "
        f"{counts['neutral']} neutral (3 stars), and {counts['disliked']} disliked (1–2 stars).", "",
        "## Case index", "",
        "| Case | Restaurant | Rank | Held-out rating | Outcome | Match evidence |",
        "|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        match = next((item for item in row["evidence"] if item["id"] == "M1"), None)
        lines.append(
            f"| {row['sample_id']} | {row['recommended_restaurant']} | "
            f"{row['recommendation_rank']} | {row['heldout_rating']:.0f} stars | "
            f"{row['heldout_outcome'].title()} | {'Yes' if match else 'No'} |"
        )

    lines.extend(["", "## Detailed examples", ""])
    for row in selected:
        evidence = row["evidence"]
        metadata = next((item for item in evidence if item["id"] == "R1"), None)
        match = next((item for item in evidence if item["id"] == "M1"), None)
        history = evidence_by_prefix(row, "H")[:3]
        restaurant = [
            item for item in evidence
            if item["id"] != "R1" and item["id"].startswith(("A", "Q"))
        ][:3]
        judge = row.get("judge") or {}
        lines.extend([
            f"### Case {row['sample_id']}: {row['recommended_restaurant']}", "",
            f"- Ranking result: **Hit@10 at rank {row['recommendation_rank']}**.",
            f"- Observed held-out outcome: **{row['heldout_rating']:.0f} stars "
            f"({row['heldout_outcome']})**.",
            f"- Restaurant: {metadata['fact'] if metadata else 'Metadata unavailable.'}", "",
            "Prior user evidence:", "",
        ])
        lines.extend(f"- [{item['id']}] {item['fact']}" for item in history)
        lines.extend(["", "Retrieved recommendation evidence:", ""])
        if match:
            lines.append(f"- [M1] {match['fact']}")
        else:
            lines.append("- No directionally supported cuisine/attribute match was retrieved.")
        lines.extend(f"- [{item['id']}] {item['fact']}" for item in restaurant)
        lines.extend([
            "", "Generated explanation:", "",
            f"> {row['generation'].get('explanation', '')}", "",
            "Automated audit: "
            f"entailment {judge.get('entailment')}/5; personalization "
            f"{judge.get('personalization')}/5; usefulness {judge.get('usefulness')}/5; "
            f"citation correctness {judge.get('citation_correctness')}/5; unsupported claims "
            f"{judge.get('unsupported_claims')}.", "",
        ])

    lines.extend([
        "## Interpretation", "",
        "A Hit@10 records retrieval, not satisfaction. The liked examples show cases where the "
        "ranking hit and the later rating was favorable. Neutral and disliked examples are equally "
        "important: they demonstrate that a well-grounded explanation can faithfully describe why "
        "a restaurant was surfaced without claiming that the diner necessarily enjoyed it.", "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
