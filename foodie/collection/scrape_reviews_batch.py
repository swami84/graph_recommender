#!/usr/bin/env python3
"""
scrape_reviews_batch.py — Concurrently scrape Google Maps reviews for all
restaurants in the hexagon index.

Reads data/hex_restaurants_index.csv, deduplicates by place_id, then runs
N concurrent camoufox browser sessions to scrape up to --max-reviews reviews
per restaurant.

Usage:
    python -m foodie.collection.scrape_reviews_batch                         # all restaurants
    python -m foodie.collection.scrape_reviews_batch --concurrency 50
    python -m foodie.collection.scrape_reviews_batch --max-reviews 200       # default
    python -m foodie.collection.scrape_reviews_batch --limit 20              # first 20
    python -m foodie.collection.scrape_reviews_batch --retry-failed          # failures

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
import fcntl
import json
import logging
import os
import random
import time
from pathlib import Path

from foodie.collection import camoufox_env  # must precede any camoufox import
from foodie.collection.camoufox_env import open_camoufox

from foodie.collection import scrape_reviews
from foodie.collection.scrape_reviews import (
    BotDetectionError,
    GoogleChallengeError,
    NoReviewsTabError,
)

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
log.setLevel(logging.INFO)
logging.getLogger("scraper").setLevel(logging.ERROR)

MORE_REMAINING_EXIT = 75
GOOGLE_CHALLENGE_EXIT = 76
RESOURCE_EXHAUSTED_EXIT = 77
RATE_LIMIT_STATE = Path("/tmp/foodie_review_global_start_rate.lock")

# Hard ceiling on one restaurant.  Every Playwright call already has its own
# timeout, but a page with hundreds of truncated reviews multiplies those
# per-call waits into hours, which wedges the worker, its shard, and the
# whole run behind asyncio.gather.  Exceeding this marks the restaurant
# retryable rather than blocking the pipeline.
RESTAURANT_TIMEOUT = 300.0


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


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Commit one restaurant checkpoint without leaving a partial JSON file."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def _reserve_global_start(starts_per_minute: float, jitter: float = 0.0) -> float:
    """Reserve one globally spaced start time across every shard process.

    ``jitter`` spreads each gap uniformly over
    ``interval * [1 - jitter, 1 + jitter]``.  The mean gap is unchanged, so
    the long-run rate still tracks ``starts_per_minute``, but the starts no
    longer land on an exact clock-like cadence.
    """
    interval = 60.0 / starts_per_minute
    if jitter:
        interval *= random.uniform(1.0 - jitter, 1.0 + jitter)
    now = time.time()
    RATE_LIMIT_STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(RATE_LIMIT_STATE, "a+", encoding="utf-8") as state:
        fcntl.flock(state.fileno(), fcntl.LOCK_EX)
        state.seek(0)
        try:
            next_start = float(state.read().strip())
        except ValueError:
            next_start = now
        reserved = max(now, next_start)
        state.seek(0)
        state.truncate()
        state.write(f"{reserved + interval:.6f}")
        state.flush()
        fcntl.flock(state.fileno(), fcntl.LOCK_UN)
    return max(0.0, reserved - now)


async def wait_for_global_start(
    starts_per_minute: float, jitter: float = 0.0
) -> None:
    delay = await asyncio.to_thread(
        _reserve_global_start, starts_per_minute, jitter
    )
    if delay:
        await asyncio.sleep(delay)


def _should_process(
    place_id: str,
    retry_failed: bool,
    retry_outcomes: set[str],
    retry_before: datetime.datetime | None = None,
) -> bool:
    out_file = REVIEWS_DIR / f"{place_id}.json"
    if not out_file.exists():
        return not retry_outcomes
    try:
        cached = json.loads(out_file.read_text())
    except Exception:
        return True  # A prior crash may have left a corrupt legacy checkpoint.
    if retry_outcomes:
        if cached.get("outcome", "") not in retry_outcomes:
            return False
        # Resource-exhaustion checkpoints did not make a valid request and
        # should be eligible immediately after the worker process is recycled.
        message = str(cached.get("error_message") or "")
        if "Too many open files" in message or "Cannot allocate memory" in message:
            return True
        if retry_before:
            try:
                attempted_at = datetime.datetime.fromisoformat(
                    str(cached.get("scraped_at") or "")
                )
                if attempted_at.tzinfo is None:
                    attempted_at = attempted_at.replace(
                        tzinfo=datetime.timezone.utc
                    )
                if attempted_at > retry_before:
                    return False
            except (TypeError, ValueError):
                pass
        return True
    if retry_failed:
        return cached.get("total_reviews", 0) == 0
    return False


async def scrape_one(
    place_id: str,
    name: str,
    max_reviews: int,
    headless: bool,
    context,
    retry_failed: bool,
    retry_outcomes: set[str],
    sem: asyncio.Semaphore,
    counter: dict,
    google_challenge: asyncio.Event,
    resource_exhausted: asyncio.Event,
) -> None:
    out_file = REVIEWS_DIR / f"{place_id}.json"

    if retry_outcomes and not out_file.exists():
        counter["skipped"] += 1
        return

    if out_file.exists():
        try:
            cached = json.loads(out_file.read_text())
            cached_outcome = cached.get("outcome", "")
            if retry_outcomes:
                should_retry = cached_outcome in retry_outcomes
            else:
                should_retry = retry_failed and cached.get("total_reviews", 0) == 0
            if not should_retry:
                counter["skipped"] += 1
                return
        except Exception:
            pass  # corrupted file — re-scrape

    # Preserve all pending checkpoints once Google blocks the public IP.
    if google_challenge.is_set() or resource_exhausted.is_set():
        counter["skipped"] += 1
        return

    async with sem:
        if google_challenge.is_set() or resource_exhausted.is_set():
            counter["skipped"] += 1
            return
        reviews = []
        outcome = "error"
        error_type = None
        error_message = None
        try:
            reviews = await asyncio.wait_for(
                scrape_reviews.scrape_reviews(
                    place_id,
                    target=max_reviews,
                    headless=headless,
                    context=context,
                ),
                timeout=RESTAURANT_TIMEOUT,
            )
            outcome = "scraped"
        except NoReviewsTabError as e:
            outcome = "no_reviews_tab"
            error_type = type(e).__name__
            error_message = str(e)
        except GoogleChallengeError as e:
            outcome = "bot_detection"
            error_type = type(e).__name__
            error_message = str(e)
            google_challenge.set()
            log.error(
                "Google automated-traffic challenge detected; halting new "
                "browser work and preserving the remaining queue"
            )
        except BotDetectionError as e:
            outcome = "bot_detection"
            error_type = type(e).__name__
            error_message = str(e)
        except asyncio.TimeoutError:
            outcome = "error"
            error_type = "RestaurantTimeout"
            error_message = (
                f"Exceeded {RESTAURANT_TIMEOUT:.0f}s for one restaurant; "
                "abandoned so the worker keeps moving"
            )
            log.warning(f"TIMEOUT  {name} ({place_id}) after {RESTAURANT_TIMEOUT:.0f}s")
        except OSError as e:
            outcome = "error"
            error_type = type(e).__name__
            error_message = str(e)
            if e.errno in {12, 24}:
                resource_exhausted.set()
                log.error(
                    f"Local resource exhaustion detected ({e}); halting new "
                    "browser work so this worker can be recycled"
                )
        except Exception as e:
            log.debug(f"FAIL  {name} ({place_id}): {e}")
            outcome = "error"
            error_type = type(e).__name__
            error_message = str(e)

        payload = {
            "place_id":      place_id,
            "name":          name,
            "scraped_at":    datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "outcome":       outcome,
            "error_type":    error_type,
            "error_message": error_message,
            "total_reviews": len(reviews),
            "reviews":       reviews,
        }
        _write_json_atomic(out_file, payload)
        counter[outcome] += 1
        _log_stats(counter)


async def browser_worker(
    queue: asyncio.Queue,
    max_reviews: int,
    headless: bool,
    retry_failed: bool,
    retry_outcomes: set[str],
    sem: asyncio.Semaphore,
    counter: dict,
    google_challenge: asyncio.Event,
    resource_exhausted: asyncio.Event,
    starts_per_minute: float,
    start_jitter: float,
    min_pause: float,
    max_pause: float,
    launch_delay: float,
) -> None:
    """Reuse one browser session while opening a fresh page per restaurant."""
    try:
        if launch_delay:
            await asyncio.sleep(launch_delay)
        async with open_camoufox(headless=headless, os="linux") as browser:
            context = await browser.new_context()
            try:
                await scrape_reviews.initialize_maps_context(context)
                completed = 0
                while (
                    not google_challenge.is_set()
                    and not resource_exhausted.is_set()
                ):
                    try:
                        row = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    if completed:
                        await asyncio.sleep(random.uniform(min_pause, max_pause))
                    await wait_for_global_start(
                        starts_per_minute, start_jitter
                    )
                    await scrape_one(
                        row["place_id"], row["name"], max_reviews, headless,
                        context, retry_failed, retry_outcomes, sem, counter,
                        google_challenge, resource_exhausted,
                    )
                    completed += 1
                    if not browser.is_connected():
                        resource_exhausted.set()
                        log.error(
                            "Persistent Camoufox process disconnected; "
                            "preserving the remaining queue so this shard can "
                            "be recycled"
                        )
                        return
            finally:
                await context.close()
    except OSError as exc:
        if exc.errno in {12, 24}:
            resource_exhausted.set()
            log.error(
                f"Local resource exhaustion while launching browser ({exc}); "
                "preserving the remaining queue"
            )
            return
        raise


async def run(
    concurrency: int,
    limit: int | None,
    max_reviews: int,
    headless: bool,
    retry_failed: bool,
    retry_outcomes: set[str] | None = None,
    dry_run: bool = False,
    shard_index: int = 0,
    shard_total: int = 1,
    skip_recent_seconds: int = 0,
    starts_per_minute: float = 24.0,
    start_jitter: float = 0.0,
    min_pause: float = 1.0,
    max_pause: float = 3.0,
    launch_stagger: float = 1.5,
) -> tuple[bool, bool, bool]:
    if not INDEX_FILE.exists():
        log.error(f"Index not found: {INDEX_FILE}. Run hexagon_places.py first.")
        return False, False, False

    with open(INDEX_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Deduplicate, drop zero-rating restaurants (no reviews to fetch)
    seen, unique = set(), []
    for row in rows:
        pid = row.get("place_id", "").strip()
        if not pid or pid in seen:
            continue
        try:
            # ID/Pro-only discovery may not populate the Enterprise
            # userRatingCount field. Skip only an explicit numeric zero.
            raw_count = str(row.get("user_rating_count", "")).strip()
            if raw_count and int(float(raw_count)) == 0:
                continue
        except (ValueError, TypeError):
            pass
        seen.add(pid)
        unique.append(row)

    # Assign this shard: take every shard_total-th restaurant starting at shard_index
    unique = unique[shard_index::shard_total]
    retry_outcomes = retry_outcomes or set()
    retry_before = None
    if skip_recent_seconds:
        retry_before = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(seconds=skip_recent_seconds)
    candidates = [
        row for row in unique
        if _should_process(
            row["place_id"], retry_failed, retry_outcomes, retry_before
        )
    ]
    candidate_total = len(candidates)
    more_remaining = bool(limit and candidate_total > limit)
    if limit:
        candidates = candidates[:limit]

    shard_label = f"shard {shard_index + 1}/{shard_total}" if shard_total > 1 else "all"
    counter = {
        "total":          len(candidates),
        "scraped":        0,
        "no_reviews_tab": 0,
        "bot_detection":  0,
        "error":          0,
        "skipped":        0,
    }
    log.info(
        f"[{shard_label}] Starting chunk: {len(candidates)} of "
        f"{candidate_total} pending restaurants, "
        f"concurrency={concurrency}, max_reviews={max_reviews}, "
        f"global_start_cap={starts_per_minute:g}/min "
        f"jitter=+/-{start_jitter:.0%}"
        + (" [retry-failed]" if retry_failed else "")
        + (f" [retry-outcomes={','.join(sorted(retry_outcomes))}]" if retry_outcomes else "")
    )
    if dry_run:
        log.info("Dry run complete; no browsers launched")
        return more_remaining, False, False

    sem = asyncio.Semaphore(concurrency)
    google_challenge = asyncio.Event()
    resource_exhausted = asyncio.Event()
    queue: asyncio.Queue = asyncio.Queue()
    for candidate in candidates:
        queue.put_nowait(candidate)
    tasks = [
        browser_worker(
            queue, max_reviews, headless, retry_failed, retry_outcomes, sem,
            counter, google_challenge, resource_exhausted,
            starts_per_minute, start_jitter, min_pause, max_pause,
            (shard_index * concurrency + worker_index) * launch_stagger,
        )
        for worker_index in range(min(concurrency, len(candidates)))
    ]
    await asyncio.gather(*tasks)

    log.info(f"[{shard_label}] Complete:")
    _log_stats(counter, force=True)
    return (
        more_remaining,
        google_challenge.is_set(),
        resource_exhausted.is_set(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-scrape Google Maps reviews")
    parser.add_argument(
        "--concurrency", type=int, default=12,
        help="Number of concurrent browser instances (default: 12)",
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
        "--retry-outcomes", nargs="+", default=[],
        choices=["error", "bot_detection", "no_reviews_tab"],
        help="Re-scrape only cached files with these outcomes",
    )
    parser.add_argument(
        "--shard", type=int, nargs=2, metavar=("INDEX", "TOTAL"), default=[0, 1],
        help="Process shard INDEX of TOTAL (0-indexed). E.g. --shard 0 3 for first of 3 shards.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the selected restaurant count without launching browsers",
    )
    parser.add_argument(
        "--skip-recent-seconds", type=int, default=0,
        help="During outcome retries, skip non-resource errors attempted recently",
    )
    parser.add_argument(
        "--starts-per-minute", type=float, default=24.0,
        help="Global restaurant-start cap shared across all shard processes",
    )
    parser.add_argument(
        "--restaurant-timeout", type=float, default=300.0,
        help="Abandon a single restaurant after this many seconds",
    )
    parser.add_argument(
        "--start-jitter", type=float, default=0.4,
        help="Randomize each global start gap by +/- this fraction, keeping "
             "the average rate but avoiding an exact clock-like cadence",
    )
    parser.add_argument("--min-pause", type=float, default=1.0)
    parser.add_argument("--max-pause", type=float, default=3.0)
    parser.add_argument(
        "--launch-stagger", type=float, default=1.5,
        help="Seconds between successive browser launches across shards",
    )
    args = parser.parse_args()
    if args.retry_failed and args.retry_outcomes:
        parser.error("Use either --retry-failed or --retry-outcomes, not both")
    if not 0.0 <= args.start_jitter < 1.0:
        parser.error("--start-jitter must be in [0, 1)")
    if args.restaurant_timeout <= 0:
        parser.error("--restaurant-timeout must be positive")
    global RESTAURANT_TIMEOUT
    RESTAURANT_TIMEOUT = args.restaurant_timeout
    if args.starts_per_minute <= 0:
        parser.error("--starts-per-minute must be positive")
    if args.min_pause < 0 or args.max_pause < args.min_pause:
        parser.error("Require 0 <= --min-pause <= --max-pause")
    if args.launch_stagger < 0:
        parser.error("--launch-stagger cannot be negative")

    more_remaining, google_challenge, resource_exhausted = asyncio.run(run(
        args.concurrency, args.limit, args.max_reviews,
        headless=not args.no_headless, retry_failed=args.retry_failed,
        retry_outcomes=set(args.retry_outcomes),
        dry_run=args.dry_run,
        shard_index=args.shard[0], shard_total=args.shard[1],
        skip_recent_seconds=args.skip_recent_seconds,
        starts_per_minute=args.starts_per_minute,
        start_jitter=args.start_jitter,
        min_pause=args.min_pause,
        max_pause=args.max_pause,
        launch_stagger=args.launch_stagger,
    ))
    if google_challenge:
        raise SystemExit(GOOGLE_CHALLENGE_EXIT)
    if resource_exhausted:
        raise SystemExit(RESOURCE_EXHAUSTED_EXIT)
    if more_remaining and not args.dry_run:
        raise SystemExit(MORE_REMAINING_EXIT)


if __name__ == "__main__":
    main()
