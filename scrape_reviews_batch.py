#!/usr/bin/env python3
"""
scrape_reviews_batch.py — Concurrently scrape Google Maps reviews for all
restaurants in the hexagon index.

Reads data/hex_restaurants_index.csv, deduplicates by place_id, then runs
N concurrent camoufox browser sessions to scrape up to --max-reviews reviews
per restaurant.

Usage:
    python scrape_reviews_batch.py                         # all restaurants, 50 concurrent
    python scrape_reviews_batch.py --concurrency 50
    python scrape_reviews_batch.py --max-reviews 200       # default
    python scrape_reviews_batch.py --limit 20              # first 20 unique restaurants
    python scrape_reviews_batch.py --retry-failed          # re-scrape 0-review results

Output:
    data/reviews/{place_id}.json

Skip logic:
    Skips any place_id that already has a non-empty reviews file.

Progress:
    Aggregate stats logged every 250 restaurants processed.
"""

import argparse
import asyncio
import csv
import datetime
import json
import logging
from pathlib import Path

import scrape_reviews
from scrape_reviews import NoReviewsTabError, BotDetectionError

INDEX_FILE  = Path("data/hex_restaurants_index.csv")
REVIEWS_DIR = Path("data/reviews")
REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

LOG_INTERVAL = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch")
logging.getLogger("scraper").setLevel(logging.ERROR)


def _log_stats(counter: dict, force: bool = False) -> None:
    processed = counter["scraped"] + counter["no_reviews_tab"] + counter["bot_detection"] + counter["error"]
    if not force and processed % LOG_INTERVAL != 0:
        return
    total    = counter["total"]
    skipped  = counter["skipped"]
    log.info(
        f"[{processed + skipped}/{total}]  "
        f"scraped={counter['scraped']}  "
        f"no_reviews_tab={counter['no_reviews_tab']}  "
        f"bot_detection={counter['bot_detection']}  "
        f"error={counter['error']}  "
        f"skipped={skipped}"
    )


async def scrape_one(
    place_id: str,
    name: str,
    max_reviews: int,
    headless: bool,
    retry_failed: bool,
    sem: asyncio.Semaphore,
    counter: dict,
) -> None:
    out_file = REVIEWS_DIR / f"{place_id}.json"

    if out_file.exists():
        try:
            cached = json.loads(out_file.read_text())
            if cached.get("total_reviews", 0) > 0 or not retry_failed:
                counter["skipped"] += 1
                return
        except Exception:
            pass  # corrupted file — re-scrape

    async with sem:
        reviews = []
        outcome = "error"
        try:
            reviews = await scrape_reviews.scrape_reviews(
                place_id, target=max_reviews, headless=headless
            )
            outcome = "scraped"
        except NoReviewsTabError:
            outcome = "no_reviews_tab"
        except BotDetectionError:
            outcome = "bot_detection"
        except Exception as e:
            log.debug(f"FAIL  {name} ({place_id}): {e}")
            outcome = "error"

        payload = {
            "place_id":      place_id,
            "name":          name,
            "scraped_at":    datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "outcome":       outcome,
            "total_reviews": len(reviews),
            "reviews":       reviews,
        }
        out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        counter[outcome] += 1
        _log_stats(counter)


async def run(
    concurrency: int,
    limit: int | None,
    max_reviews: int,
    headless: bool,
    retry_failed: bool,
    shard_index: int = 0,
    shard_total: int = 1,
) -> None:
    if not INDEX_FILE.exists():
        log.error(f"Index not found: {INDEX_FILE}. Run hexagon_places.py first.")
        return

    with open(INDEX_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Deduplicate, drop zero-rating restaurants (no reviews to fetch)
    seen, unique = set(), []
    for row in rows:
        pid = row.get("place_id", "").strip()
        if not pid or pid in seen:
            continue
        try:
            if int(float(row.get("user_rating_count") or 0)) == 0:
                continue
        except (ValueError, TypeError):
            pass
        seen.add(pid)
        unique.append(row)

    # Assign this shard: take every shard_total-th restaurant starting at shard_index
    unique = unique[shard_index::shard_total]

    if limit:
        unique = unique[:limit]

    shard_label = f"shard {shard_index + 1}/{shard_total}" if shard_total > 1 else "all"
    counter = {
        "total":          len(unique),
        "scraped":        0,
        "no_reviews_tab": 0,
        "bot_detection":  0,
        "error":          0,
        "skipped":        0,
    }
    log.info(
        f"[{shard_label}] Starting: {len(unique)} restaurants, "
        f"concurrency={concurrency}, max_reviews={max_reviews}"
        + (" [retry-failed]" if retry_failed else "")
    )

    sem = asyncio.Semaphore(concurrency)
    tasks = [
        scrape_one(r["place_id"], r["name"], max_reviews, headless, retry_failed, sem, counter)
        for r in unique
    ]
    await asyncio.gather(*tasks)

    log.info(f"[{shard_label}] Complete:")
    _log_stats(counter, force=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-scrape Google Maps reviews")
    parser.add_argument(
        "--concurrency", type=int, default=50,
        help="Number of concurrent browser instances (default: 50)",
    )
    parser.add_argument(
        "--max-reviews", type=int, default=200,
        help="Max reviews to scrape per restaurant (default: 200)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap total restaurants to scrape",
    )
    parser.add_argument(
        "--no-headless", action="store_true",
        help="Run browsers headed (default: headless)",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Re-scrape restaurants that previously returned 0 reviews",
    )
    parser.add_argument(
        "--shard", type=int, nargs=2, metavar=("INDEX", "TOTAL"), default=[0, 1],
        help="Process shard INDEX of TOTAL (0-indexed). E.g. --shard 0 3 for first of 3 shards.",
    )
    args = parser.parse_args()

    asyncio.run(run(
        args.concurrency, args.limit, args.max_reviews,
        headless=not args.no_headless, retry_failed=args.retry_failed,
        shard_index=args.shard[0], shard_total=args.shard[1],
    ))


if __name__ == "__main__":
    main()
