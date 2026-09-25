#!/usr/bin/env python3
"""Run the final review collection and one selective retry pass."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    timestamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{timestamp}  {message}", flush=True)


CHUNK_COMPLETE_MORE_REMAIN = 75
GOOGLE_CHALLENGE_EXIT = 76
RESOURCE_EXHAUSTED_EXIT = 77


class GoogleChallengeBlocked(RuntimeError):
    """The public IP is currently on Google's automated-traffic challenge."""


def run_resilient(
    name: str,
    command: list[str],
    max_restarts: int,
    restart_delay: int,
    repeat_chunks: bool = False,
) -> None:
    failures = 0
    chunk = 1
    while True:
        log(f"START {name} chunk={chunk}: {' '.join(command)}")
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode == 0:
            log(f"DONE {name}")
            return
        if result.returncode == GOOGLE_CHALLENGE_EXIT:
            log(f"PAUSED {name}: Google automated-traffic challenge detected")
            raise GoogleChallengeBlocked(name)
        if result.returncode == RESOURCE_EXHAUSTED_EXIT:
            log(
                f"RECYCLE {name}: local browser resources exhausted; "
                "starting a clean worker process"
            )
            chunk += 1
            time.sleep(5)
            continue
        if repeat_chunks and result.returncode == CHUNK_COMPLETE_MORE_REMAIN:
            log(f"RECYCLE {name}: chunk={chunk} complete; more checkpoints remain")
            chunk += 1
            failures = 0
            time.sleep(2)
            continue
        failures += 1
        if failures > max_restarts:
            log(f"FAILED {name}: exit={result.returncode}, retries exhausted")
            raise RuntimeError(f"{name} failed with exit {result.returncode}")
        delay = min(restart_delay * (2 ** (failures - 1)), 300)
        log(
            f"RESTART {name}: exit={result.returncode}, attempt={failures}/"
            f"{max_restarts}, backoff={delay}s"
        )
        time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=12,
                        help="Total browser concurrency across isolated workers")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Restaurants per worker process before recycling")
    parser.add_argument("--retry-concurrency", type=int, default=12)
    parser.add_argument("--retry-workers", type=int, default=4,
                        help="Isolated workers used for the error retry stage")
    parser.add_argument("--starts-per-minute", type=float, default=24.0)
    parser.add_argument("--start-jitter", type=float, default=0.4,
                        help="Randomize each global start gap by +/- this "
                             "fraction of the average interval")
    parser.add_argument("--min-pause", type=float, default=1.0)
    parser.add_argument("--max-pause", type=float, default=3.0)
    parser.add_argument("--launch-stagger", type=float, default=1.5)
    parser.add_argument("--max-reviews", type=int, default=200)
    parser.add_argument("--max-restarts", type=int, default=5)
    parser.add_argument("--restart-delay", type=int, default=30)
    args = parser.parse_args()

    if args.workers < 1 or args.concurrency < 1:
        parser.error("--workers and --concurrency must be positive")
    if args.retry_workers < 1 or args.retry_concurrency < 1:
        parser.error("--retry-workers and --retry-concurrency must be positive")
    if args.concurrency % args.workers:
        parser.error("--concurrency must be divisible by --workers")
    if args.retry_concurrency % args.retry_workers:
        parser.error("--retry-concurrency must be divisible by --retry-workers")
    if args.starts_per_minute <= 0:
        parser.error("--starts-per-minute must be positive")
    if not 0.0 <= args.start_jitter < 1.0:
        parser.error("--start-jitter must be in [0, 1)")
    if args.min_pause < 0 or args.max_pause < args.min_pause:
        parser.error("Require 0 <= --min-pause <= --max-pause")
    if args.launch_stagger < 0:
        parser.error("--launch-stagger cannot be negative")
    per_worker = args.concurrency // args.workers

    base = [sys.executable, "-m", "foodie.collection.scrape_reviews_batch"]
    log(
        f"Launching {args.workers} isolated workers x {per_worker} browsers "
        f"= {args.concurrency} total; recycle every {args.chunk_size} restaurants"
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for shard in range(args.workers):
            command = base + [
                "--concurrency", str(per_worker),
                "--max-reviews", str(args.max_reviews),
                "--limit", str(args.chunk_size),
                "--starts-per-minute", str(args.starts_per_minute),
                "--start-jitter", str(args.start_jitter),
                "--min-pause", str(args.min_pause),
                "--max-pause", str(args.max_pause),
                "--launch-stagger", str(args.launch_stagger),
                "--shard", str(shard), str(args.workers),
            ]
            futures.append(executor.submit(
                run_resilient,
                f"main shard {shard + 1}/{args.workers}",
                command,
                args.max_restarts,
                args.restart_delay,
                True,
            ))
        for future in futures:
            future.result()

    retry_per_worker = args.retry_concurrency // args.retry_workers
    log(
        f"Launching retry stage: {args.retry_workers} isolated workers x "
        f"{retry_per_worker} browsers = {args.retry_concurrency} total"
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.retry_workers
    ) as executor:
        futures = []
        for shard in range(args.retry_workers):
            command = base + [
                "--concurrency", str(retry_per_worker),
                "--max-reviews", str(args.max_reviews),
                "--limit", str(args.chunk_size),
                "--skip-recent-seconds", "28800",
                "--starts-per-minute", str(args.starts_per_minute),
                "--start-jitter", str(args.start_jitter),
                "--min-pause", str(args.min_pause),
                "--max-pause", str(args.max_pause),
                "--launch-stagger", str(args.launch_stagger),
                "--retry-outcomes", "error", "bot_detection",
                "--shard", str(shard), str(args.retry_workers),
            ]
            futures.append(executor.submit(
                run_resilient,
                f"error/bot-detection retry shard {shard + 1}/"
                f"{args.retry_workers}",
                command,
                args.max_restarts,
                args.restart_delay,
                True,
            ))
        for future in futures:
            future.result()
    log("ALL REVIEW COLLECTION STAGES COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except GoogleChallengeBlocked:
        raise SystemExit(GOOGLE_CHALLENGE_EXIT)
