#!/usr/bin/env python3
"""Run capped Places discovery, then scrape reviews for newly found restaurants."""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUDGET_STOP_EXIT = 78


def log(message: str) -> None:
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{stamp}  {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cbg-file", type=Path, required=True)
    parser.add_argument("--budget-id", required=True)
    parser.add_argument("--billing-month", required=True)
    parser.add_argument("--max-monthly-api-calls", type=int, required=True)
    parser.add_argument("--max-run-cost-usd", type=float, required=True)
    parser.add_argument("--review-concurrency", type=int, default=12)
    parser.add_argument("--review-workers", type=int, default=4)
    parser.add_argument("--starts-per-minute", type=float, default=24.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    places = [
        sys.executable, "hexagon_places.py",
        "--cbg-file", str(args.cbg_file),
        "--budget-id", args.budget_id,
        "--billing-month", args.billing_month,
        "--max-monthly-api-calls", str(args.max_monthly_api_calls),
        "--max-run-cost-usd", str(args.max_run_cost_usd),
        "--empty-stop", "2",
    ]
    reviews = [
        sys.executable, "run_review_collection_guarded.py",
        "--concurrency", str(args.review_concurrency),
        "--workers", str(args.review_workers),
        "--starts-per-minute", str(args.starts_per_minute),
        "--chunk-size", "100",
        "--max-reviews", "200",
    ]
    log("Places command: " + " ".join(places))
    log("Review command: " + " ".join(reviews))
    if args.dry_run:
        log("DRY RUN: neither command was started")
        return

    log("START capped CBG Places collection")
    places_result = subprocess.run(places, cwd=ROOT)
    if places_result.returncode not in (0, BUDGET_STOP_EXIT):
        raise SystemExit(places_result.returncode)
    if places_result.returncode == BUDGET_STOP_EXIT:
        log("Places budget ceiling reached safely; continuing to reviews")
    else:
        log("Places queue completed within budget")

    log("START guarded review collection for newly discovered restaurants")
    review_result = subprocess.run(reviews, cwd=ROOT)
    raise SystemExit(review_result.returncode)


if __name__ == "__main__":
    main()
