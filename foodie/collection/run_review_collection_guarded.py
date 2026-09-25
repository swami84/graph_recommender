#!/usr/bin/env python3
"""Resume review collection conservatively, cooling down on Google challenges."""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GOOGLE_CHALLENGE_EXIT = 76
KNOWN_POSITIVE_PLACE_ID = "ChIJye4iijK3t4kR1e9LNbNnqms"


def log(message: str) -> None:
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{now}  {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of isolated collection shards")
    parser.add_argument("--cooldown", type=int, default=3600,
                        help="Seconds to wait after a Google challenge")
    parser.add_argument("--max-reviews", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="Restaurants processed before recycling each shard")
    parser.add_argument("--starts-per-minute", type=float, default=24.0)
    parser.add_argument("--start-jitter", type=float, default=0.4,
                        help="Randomize each global start gap by +/- this "
                             "fraction of the average interval")
    parser.add_argument("--min-pause", type=float, default=1.0)
    parser.add_argument("--max-pause", type=float, default=3.0)
    parser.add_argument("--launch-stagger", type=float, default=1.5)
    args = parser.parse_args()
    if (
        args.concurrency < 1
        or args.workers < 1
        or args.cooldown < 1
        or args.chunk_size < 1
        or args.starts_per_minute <= 0
        or args.min_pause < 0
        or args.max_pause < args.min_pause
        or args.launch_stagger < 0
        or not 0.0 <= args.start_jitter < 1.0
    ):
        parser.error(
            "--concurrency, --workers, --cooldown, and --chunk-size must be "
            "positive; pacing values must be valid"
        )
    if args.concurrency % args.workers:
        parser.error("--concurrency must be divisible by --workers")

    probe = [
        sys.executable,
        "-m", "foodie.collection.scrape_reviews",
        "--place-id", KNOWN_POSITIVE_PLACE_ID,
        "--target", "1",
        "--headless",
    ]
    collection = [
        sys.executable,
        "-m", "foodie.collection.run_review_collection",
        "--concurrency", str(args.concurrency),
        "--workers", str(args.workers),
        "--retry-concurrency", str(args.concurrency),
        "--retry-workers", str(args.workers),
        "--max-reviews", str(args.max_reviews),
        "--chunk-size", str(args.chunk_size),
        "--starts-per-minute", str(args.starts_per_minute),
        "--start-jitter", str(args.start_jitter),
        "--min-pause", str(args.min_pause),
        "--max-pause", str(args.max_pause),
        "--launch-stagger", str(args.launch_stagger),
        "--max-restarts", "0",
    ]
    while True:
        log("Running a one-browser Google availability probe")
        result = subprocess.run(probe, cwd=ROOT)
        if result.returncode == GOOGLE_CHALLENGE_EXIT:
            log(
                "Google challenge remains active; collection is safely paused "
                f"for {args.cooldown} seconds"
            )
            time.sleep(args.cooldown)
            continue
        if result.returncode != 0:
            log(
                "Known-positive probe could not load a review card; "
                f"collection is safely paused for {args.cooldown} seconds"
            )
            time.sleep(args.cooldown)
            continue

        log(
            "Probe passed; starting guarded collection at "
            f"concurrency={args.concurrency}"
        )
        result = subprocess.run(collection, cwd=ROOT)
        if result.returncode == 0:
            break
        if result.returncode != GOOGLE_CHALLENGE_EXIT:
            raise SystemExit(result.returncode)
        log(
            "Google challenge returned during collection; safely paused for "
            f"{args.cooldown} seconds"
        )
        time.sleep(args.cooldown)

    log("Collection complete; validating and freezing the review dataset")
    subprocess.run(
        [sys.executable, "-m", "foodie.collection.finalize_review_dataset"],
        cwd=ROOT,
        check=True,
    )
    log("REVIEW DATASET FROZEN")


if __name__ == "__main__":
    main()
