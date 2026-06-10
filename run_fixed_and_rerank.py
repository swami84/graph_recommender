#!/usr/bin/env python3
"""
run_fixed_and_rerank.py

Step 1 — Re-run the three previously-failing models on both GPUs:
    GPU 0: InfoNCE 4L skip llm
    GPU 1: SimGCL-HRCL 4L skip llm
    (then)
    GPU 0: HEK-CL cf4/kg2 skip llm  (solo)

Step 2 — Run proximity re-ranker for every model that has saved embeddings.

Usage:
    python run_fixed_and_rerank.py
    python run_fixed_and_rerank.py --gpu0 0 --gpu1 1
    python run_fixed_and_rerank.py --alpha 0.3 --bandwidth 10
    python run_fixed_and_rerank.py --skip-training   # only run proximity re-ranking
"""

import argparse
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_fixed")


def run_training(gpu0: int, gpu1: int) -> None:
    log.info("=" * 65)
    log.info("STEP 1 — Training fixed models (InfoNCE, SimGCL-HRCL, HEK-CL)")
    log.info("=" * 65)

    # Runs 9=InfoNCE, 10=SimGCL-HRCL, 11=HEK-CL in run_all_models.py
    cmd = [
        sys.executable, "run_all_models.py",
        "--runs", "9", "10", "11",
        "--gpu0", str(gpu0),
        "--gpu1", str(gpu1),
    ]
    log.info(f"  Command: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        log.warning(f"run_all_models.py exited with rc={rc} — some models may have failed; continuing to re-rank")
    else:
        log.info("Training step complete.")


def run_proximity(alpha: float, bandwidth: float, top_k: int) -> None:
    log.info("")
    log.info("=" * 65)
    log.info("STEP 2 — Proximity re-ranking for all saved models")
    log.info("=" * 65)

    cmd = [
        sys.executable, "rerank_proximity.py",
        "--alpha",     str(alpha),
        "--bandwidth", str(bandwidth),
        "--top-k",     str(top_k),
    ]
    log.info(f"  Command: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        log.error(f"rerank_proximity.py exited with rc={rc}")
    else:
        log.info("Proximity re-ranking complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu0",          type=int,   default=0)
    parser.add_argument("--gpu1",          type=int,   default=1)
    parser.add_argument("--alpha",         type=float, default=0.3,
                        help="Proximity blend weight (default 0.3)")
    parser.add_argument("--bandwidth",     type=float, default=10.0,
                        help="Distance decay scale in km (default 10)")
    parser.add_argument("--top-k",         type=int,   default=50,
                        help="Candidate pool size for re-ranking (default 50)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training step, only run proximity re-ranking")
    args = parser.parse_args()

    if not args.skip_training:
        run_training(args.gpu0, args.gpu1)

    run_proximity(args.alpha, args.bandwidth, args.top_k)


if __name__ == "__main__":
    main()
