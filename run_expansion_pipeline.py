#!/usr/bin/env python3
"""
Orchestrates the restaurant data expansion pipeline (Steps 2–5):

  Step 2: hexagon_places.py          — tile CBGs, fetch Google Places results
  Step 3: scrape_reviews_batch.py    — scrape Google Maps reviews
  Step 4: vLLM lifecycle             — start server, run LLM feature scripts, kill
  Step 5: feature rebuild            — graph / NLP / extended / preference / training

Usage:
  python run_expansion_pipeline.py              # run all steps
  python run_expansion_pipeline.py --start-step 3   # skip step 2, resume from step 3
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT       = Path(__file__).parent
VLLM_PORT  = 8082
VLLM_MODEL = "cyankiwi/Qwen3.5-9B-AWQ-BF16-INT8"
MODEL_NAME = "qwen3.5-9b"

CBG_FILE   = ROOT / "data" / "expansion_cbgs_50km.csv"

# ── Utilities ─────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_step(name: str, cmd: list[str]) -> float:
    """Run a subprocess step, log wall time, exit on failure."""
    log(f"START  {name}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        log(f"FAILED {name} (exit {result.returncode}, {elapsed:.0f}s)")
        sys.exit(result.returncode)
    log(f"DONE   {name}  ({elapsed:.0f}s)")
    return elapsed


parser = argparse.ArgumentParser()
parser.add_argument(
    "--start-step", type=int, default=2, choices=[2, 3, 4, 5],
    help="Step to start from (default: 2). Use 3 to skip hexagon_places, 4 to skip reviews, etc.",
)
parser.add_argument(
    "--cbg-file", default=str(ROOT / "data" / "expansion_cbgs_50km.csv"),
    help="CBG list CSV to pass to hexagon_places.py (default: data/expansion_cbgs_50km.csv)",
)
parser.add_argument(
    "--concurrency", type=int, default=10,
    help="Concurrent browser instances per scraper worker (default: 10)",
)
parser.add_argument(
    "--scraper-workers", type=int, default=3,
    help="Number of parallel scraper processes (default: 3)",
)
args = parser.parse_args()
CBG_FILE = Path(args.cbg_file)

# ── Step 2: fetch restaurants ──────────────────────────────────────────────────

if args.start_step <= 2:
    run_step(
        "hexagon_places.py",
        ["python", "hexagon_places.py", "--cbg-file", str(CBG_FILE)],
    )

# ── Step 3: scrape reviews ─────────────────────────────────────────────────────

if args.start_step <= 3:
    n = args.scraper_workers
    log(f"Launching {n} parallel scraper workers (concurrency={args.concurrency} each) …")
    t0 = time.time()
    procs = [
        subprocess.Popen(
            [
                "python", "scrape_reviews_batch.py",
                "--max-reviews",  "200",
                "--concurrency",  str(args.concurrency),
                "--shard",        str(i), str(n),
            ],
            cwd=ROOT,
        )
        for i in range(n)
    ]
    for p in procs:
        p.wait()
    elapsed = time.time() - t0
    exit_codes = [p.returncode for p in procs]
    if any(c != 0 for c in exit_codes):
        log(f"FAILED scrape_reviews_batch.py — exit codes: {exit_codes} ({elapsed:.0f}s)")
        sys.exit(1)
    log(f"DONE   scrape_reviews_batch.py  ({elapsed:.0f}s)")

# ── Step 4: vLLM lifecycle + LLM feature scripts ──────────────────────────────

def _start_vllm() -> subprocess.Popen:
    vllm_log = ROOT / "logs" / f"vllm_{time.strftime('%Y%m%d_%H%M%S')}.log"
    vllm_log.parent.mkdir(exist_ok=True)
    log(f"Starting vLLM server … (logs → {vllm_log.name})")
    vllm_out = open(vllm_log, "w")
    proc = subprocess.Popen(
        [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model",                  VLLM_MODEL,
            "--tensor-parallel-size",   "2",
            "--max-model-len",          "16384",
            "--max-num-seqs",           "64",
            "--gpu-memory-utilization", "0.90",
            "--port",                   str(VLLM_PORT),
            "--served-model-name",      MODEL_NAME,
            "--enable-prefix-caching",
        ],
        cwd=ROOT,
        start_new_session=True,
        stdout=vllm_out,
        stderr=vllm_out,
        env={**os.environ, "VLLM_ENGINE_READY_TIMEOUT_S": "1200"},
    )
    log(f"  vLLM PID={proc.pid}")
    return proc


def _wait_healthy(timeout: int = 300) -> None:
    """Poll GET /health every 5 s until 200 OK or timeout."""
    url = f"http://localhost:{VLLM_PORT}/health"
    t0  = time.time()
    while True:
        try:
            if requests.get(url, timeout=4).status_code == 200:
                log(f"  vLLM healthy after {time.time() - t0:.0f}s")
                return
        except Exception:
            pass
        if time.time() - t0 > timeout:
            raise TimeoutError(f"vLLM did not become healthy within {timeout}s")
        time.sleep(5)


def _kill_vllm(proc: subprocess.Popen) -> None:
    log("Terminating vLLM server …")
    pgid = os.getpgid(proc.pid)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
        log("  vLLM terminated (SIGTERM)")
    except subprocess.TimeoutExpired:
        log("  Grace period expired — sending SIGKILL")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        log("  vLLM killed (SIGKILL)")


def _vram_used_mib() -> list[int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [int(x.strip()) for x in out.strip().splitlines()]


def _wait_vram_freed(timeout: int = 60) -> None:
    log("Waiting for VRAM to clear …")
    t0 = time.time()
    while True:
        used = _vram_used_mib()
        if all(v < 1000 for v in used):
            log(f"  VRAM cleared: {used} MiB  ({time.time() - t0:.0f}s)")
            return
        if time.time() - t0 > timeout:
            raise RuntimeError(
                f"VRAM not freed within {timeout}s; still using: {used} MiB"
            )
        log(f"  GPU VRAM: {used} MiB — waiting …")
        time.sleep(3)


if args.start_step <= 4:
    # Rebuild restaurants_enriched.parquet before LLM scripts so they see new restaurants
    run_step("build_graph_data.py", ["python", "build_graph_data.py"])

    vllm_proc = _start_vllm()
    try:
        _wait_healthy(timeout=300)

        run_step(
            "build_restaurant_llm_features.py",
            [
                "python", "build_restaurant_llm_features.py",
                "--local-url",   f"http://localhost:{VLLM_PORT}",
                "--local-model", MODEL_NAME,
                "--resume",
            ],
        )

        run_step(
            "extract_dishes_llm.py",
            [
                "python", "extract_dishes_llm.py",
                "--url",        f"http://localhost:{VLLM_PORT}",
                "--model-name", MODEL_NAME,
                "--resume",
            ],
        )
    finally:
        _kill_vllm(vllm_proc)
        _wait_vram_freed(timeout=60)

# ── Step 5: rebuild training features ─────────────────────────────────────────

if args.start_step <= 5:
    for script in [
        "build_graph_data.py",
        "build_nlp_features.py",
        "build_extended_features.py",
        "build_user_preference_features.py",
        "build_training_features.py",
    ]:
        run_step(script, ["python", script])

log("Pipeline complete.")
