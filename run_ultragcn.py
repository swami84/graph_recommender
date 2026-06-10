#!/usr/bin/env python3
"""
run_ultragcn.py — Run UltraGCN with two feature configurations sequentially.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        python run_ultragcn.py

    # Specific runs only (0 = all features, 1 = skip llm):
    python run_ultragcn.py --runs 0
    python run_ultragcn.py --runs 1
"""

import argparse
import re
import subprocess
import sys
import time
from datetime import timedelta

RUNS = [
    {
        "name":        "all features",
        "skip_groups": "",
    },
    {
        "name":        "skip llm",
        "skip_groups": "llm",
    },
]

_RE_RECALL    = re.compile(r"Recall@10:\s+([\d.]+)")
_RE_NDCG      = re.compile(r"NDCG@10:\s+([\d.]+)")
_RE_PRECISION = re.compile(r"Precision@10:\s+([\d.]+)")


def run_experiment_streaming(run: dict, base_args: list[str]) -> dict:
    cmd = [sys.executable, "recommendation_ultragcn.py"] + base_args
    if run["skip_groups"]:
        cmd += ["--skip-feature-groups", run["skip_groups"]]

    print(f"\n{'='*60}", flush=True)
    print(f"  RUN:  {run['name']}", flush=True)
    if run["skip_groups"]:
        print(f"  SKIP: {run['skip_groups']}", flush=True)
    print(f"  CMD:  {' '.join(cmd)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    t0   = time.time()
    lines = []
    proc  = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    proc.wait()
    elapsed = time.time() - t0

    output    = "".join(lines)
    recall    = float(m.group(1)) if (m := _RE_RECALL.search(output))    else None
    ndcg      = float(m.group(1)) if (m := _RE_NDCG.search(output))      else None
    precision = float(m.group(1)) if (m := _RE_PRECISION.search(output)) else None

    return {
        "name":      run["name"],
        "skip":      run["skip_groups"] or "(none)",
        "recall":    recall,
        "ndcg":      ndcg,
        "precision": precision,
        "elapsed":   elapsed,
        "rc":        proc.returncode,
    }


def print_summary(results: list[dict]) -> None:
    print(f"\n{'='*65}")
    print("  ULTRAGCN SUMMARY")
    print(f"{'='*65}")
    header = f"  {'Run':<20}  {'Skip':<12}  {'Recall@10':>10}  {'NDCG@10':>9}  {'Time':>8}"
    print(header)
    print(f"  {'-'*18}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*8}")

    best = max((r["recall"] for r in results if r["recall"] is not None), default=0.0)
    for r in results:
        recall_str = f"{r['recall']:.4f}" if r["recall"] is not None else "FAILED"
        ndcg_str   = f"{r['ndcg']:.4f}"   if r["ndcg"]   is not None else "FAILED"
        time_str   = str(timedelta(seconds=int(r["elapsed"])))
        marker     = " ***" if r["recall"] == best else ""
        print(f"  {r['name']:<20}  {r['skip']:<12}  {recall_str:>10}  {ndcg_str:>9}  {time_str:>8}{marker}")
    print(f"{'='*65}\n", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=300)
    parser.add_argument("--emb-dim",    type=int,   default=2048)
    parser.add_argument("--batch-size", type=int,   default=8192)
    parser.add_argument("--neg-count",  type=int,   default=10)
    parser.add_argument("--ii-weight",  type=float, default=1e-4)
    parser.add_argument("--ii-topk",    type=int,   default=10)
    parser.add_argument("--eval-every", type=int,   default=50)
    parser.add_argument("--runs", nargs="*", type=int, default=None,
                        help="Indices of runs to execute (0=all features, 1=skip llm)")
    args = parser.parse_args()

    base_args = [
        "--epochs",     str(args.epochs),
        "--emb-dim",    str(args.emb_dim),
        "--batch-size", str(args.batch_size),
        "--neg-count",  str(args.neg_count),
        "--ii-weight",  str(args.ii_weight),
        "--ii-topk",    str(args.ii_topk),
        "--eval-every", str(args.eval_every),
        "--prebuilt-features",
    ]

    selected = RUNS if args.runs is None else [RUNS[i] for i in args.runs]

    print(f"\nUltraGCN: {len(selected)} run(s) | epochs={args.epochs} | "
          f"emb_dim={args.emb_dim} | neg_count={args.neg_count}", flush=True)

    results  = []
    total_t0 = time.time()
    for run in selected:
        res = run_experiment_streaming(run, base_args)
        results.append(res)
        print_summary(results)

    print(f"Total wall time: {timedelta(seconds=int(time.time() - total_t0))}\n")


if __name__ == "__main__":
    main()
