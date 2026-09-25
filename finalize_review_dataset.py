#!/usr/bin/env python3
"""Validate the completed review catalogue and write a freeze manifest."""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path,
                        default=Path("data/hex_restaurants_index.csv"))
    parser.add_argument("--reviews-dir", type=Path,
                        default=Path("data/reviews"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/review_dataset_freeze.json"))
    args = parser.parse_args()

    eligible: set[str] = set()
    all_restaurants: set[str] = set()
    known_rating_count: dict[str, int] = {}
    cbgs: set[str] = set()
    index_rows = 0
    with args.index.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index_rows += 1
            place_id = (row.get("place_id") or "").strip()
            if not place_id:
                continue
            all_restaurants.add(place_id)
            cbg = (row.get("cbg") or "").strip()
            if cbg:
                cbgs.add(cbg.zfill(12))
            raw_count = (row.get("user_rating_count") or "").strip()
            if raw_count:
                try:
                    count = int(float(raw_count))
                    known_rating_count[place_id] = max(
                        count, known_rating_count.get(place_id, 0)
                    )
                    if count == 0:
                        continue
                except (TypeError, ValueError):
                    pass
            eligible.add(place_id)

    outcomes: Counter[str] = Counter()
    invalid_json: list[str] = []
    review_count_mismatches: list[str] = []
    retryable: list[str] = []
    zero_scraped: list[str] = []
    zero_legacy: list[str] = []
    contradictory_no_tab: list[str] = []
    checkpoint_ids: set[str] = set()
    review_records = 0
    positive_restaurants = 0
    unavailable_restaurants = 0
    digest = hashlib.sha256()

    for place_id in sorted(eligible):
        path = args.reviews_dir / f"{place_id}.json"
        if not path.exists():
            continue
        checkpoint_ids.add(place_id)
        raw = path.read_bytes()
        digest.update(place_id.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(raw).digest())
        try:
            payload = json.loads(raw)
        except Exception:
            invalid_json.append(place_id)
            continue

        reviews = payload.get("reviews") or []
        declared = int(payload.get("total_reviews") or 0)
        if declared != len(reviews):
            review_count_mismatches.append(place_id)
        review_records += len(reviews)
        if reviews:
            positive_restaurants += 1

        outcome = payload.get("outcome") or "legacy"
        outcomes[outcome] += 1
        if outcome == "review_cards_unavailable":
            unavailable_restaurants += 1
        if outcome in {"error", "bot_detection"}:
            retryable.append(place_id)
        if outcome == "scraped" and not reviews:
            zero_scraped.append(place_id)
        if outcome == "legacy" and not reviews:
            zero_legacy.append(place_id)
        if (
            outcome == "no_reviews_tab"
            and known_rating_count.get(place_id, 0) > 0
        ):
            contradictory_no_tab.append(place_id)

    missing = sorted(eligible - checkpoint_ids)
    orphan_files = sum(
        1 for path in args.reviews_dir.glob("*.json")
        if path.stem not in all_restaurants
    )
    issues = {
        "missing_checkpoints": missing,
        "invalid_json": invalid_json,
        "review_count_mismatches": review_count_mismatches,
        "retryable_outcomes": retryable,
        "scraped_with_zero_reviews": zero_scraped,
        "legacy_with_zero_reviews": zero_legacy,
        "no_reviews_tab_with_known_positive_rating_count": contradictory_no_tab,
    }
    ready = not any(issues.values())
    manifest = {
        "status": "frozen" if ready else "validation_failed",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "catalogue": {
            "index_rows": index_rows,
            "unique_restaurants": len(all_restaurants),
            "eligible_restaurants": len(eligible),
            "cbgs_in_index": len(cbgs),
            "review_checkpoints": len(checkpoint_ids),
            "positive_review_restaurants": positive_restaurants,
            "zero_review_restaurants": len(checkpoint_ids) - positive_restaurants,
            "review_cards_unavailable_restaurants": unavailable_restaurants,
            "review_records": review_records,
            "outcomes": dict(outcomes),
            "orphan_review_files": orphan_files,
        },
        "integrity": {
            "review_files_sha256": digest.hexdigest(),
            "issue_counts": {key: len(value) for key, value in issues.items()},
            "issues": issues,
        },
    }
    write_json_atomic(args.output, manifest)
    print(json.dumps({
        "status": manifest["status"],
        **manifest["catalogue"],
        "issue_counts": manifest["integrity"]["issue_counts"],
        "manifest": str(args.output),
    }, indent=2))
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
