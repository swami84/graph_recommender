#!/usr/bin/env python3
"""
run_at_6am.py — Sleep until 06:00, kill VLLM, then launch GNN training.

Usage:
    python run_at_6am.py
    python run_at_6am.py --time 07:30   # override target time (HH:MM)
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sleep_until(target: datetime):
    now = datetime.now()
    delta = (target - now).total_seconds()
    if delta <= 0:
        log(f"Target time {target.strftime('%H:%M')} already passed — running immediately")
        return
    log(f"Sleeping {delta/3600:.2f}h until {target.strftime('%H:%M:%S')} …")
    time.sleep(delta)


def run(cmd: str, check: bool = False) -> int:
    log(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        log(f"  WARNING: exit code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", default="06:00", help="Target time HH:MM (default 06:00)")
    args = parser.parse_args()

    hh, mm = map(int, args.time.split(":"))
    target = datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= datetime.now():
        target += timedelta(days=1)

    sleep_until(target)

    # ── Step 1: graceful VLLM server shutdown ────────────────────────────────
    log("Killing VLLM server (SIGTERM) …")
    run("pkill -TERM -f 'vllm.entrypoints'", check=False)
    run("pkill -TERM -f 'vllm serve'",        check=False)

    log("Waiting 60 s for VLLM to shut down …")
    time.sleep(60)

    # ── Step 2: force-kill any remaining VLLM GPU workers ────────────────────
    log("Force-killing VLLM GPU workers …")
    run("pkill -9 -f 'VLLM::Worker_TP0'", check=False)
    run("pkill -9 -f 'VLLM::Worker_TP1'", check=False)
    time.sleep(5)

    # ── Step 3: confirm GPU memory freed ─────────────────────────────────────
    log("GPU memory after cleanup:")
    run("nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader")

    # ── Step 4: launch GNN training ──────────────────────────────────────────
    log("Starting GNN training …")
    gnn_cmd = (
        "python recommendation_gnn.py"
        " --epochs 300"
        " --emb-dim 2048"
        " --layers 4"
        " --grad-checkpoint"
        " --edge-dropout 0.1"
        " --film"
        " --llm-feat-mode fast"
        " --eval-every 50"
    )
    result = subprocess.run(gnn_cmd, shell=True)
    log(f"GNN training finished — exit code {result.returncode}")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
