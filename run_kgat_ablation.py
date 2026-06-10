#!/usr/bin/env python3
"""
run_kgat_ablation.py — Run KGAT feature-group ablation experiments sequentially.

Each run trains KGAT with a different subset of feature groups and records
the final Recall@10 / NDCG@10.  Results are printed as a summary table at the end.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python run_kgat_ablation.py

    # Custom epochs or emb-dim:
    python run_kgat_ablation.py --epochs 100 --emb-dim 1024
"""

import argparse
import re
import subprocess
import sys
import time
from datetime import timedelta

RUNS = [
    {
        "name":        "all features (baseline)",
        "skip_groups": "",
        "note":        "baseline — all 68u + 69i dims",
    },
    {
        "name":        "skip llm",
        "skip_groups": "llm",
        "note":        "drop 6 noisy item cols (rating_std, photo_rate, …)",
    },
    {
        "name":        "skip pref + nlp",
        "skip_groups": "pref,nlp",
        "note":        "drop LLM-derived alignment pair (user pref ↔ item nlp)",
    },
    {
        "name":        "skip extended",
        "skip_groups": "extended",
        "note":        "drop user behavioral stats + item dish/density/CBG features",
    },
    {
        "name":        "base only",
        "skip_groups": "extended,pref,nlp,llm",
        "note":        "price, rating, cuisine, demographics only",
    },
]

# Regex to extract metrics from KGAT stdout
_RE_RECALL    = re.compile(r"Recall@10:\s+([\d.]+)")
_RE_NDCG      = re.compile(r"NDCG@10:\s+([\d.]+)")
_RE_PRECISION = re.compile(r"Precision@10:\s+([\d.]+)")


def run_experiment(run: dict, base_args: list[str]) -> dict:
    cmd = [sys.executable, "recommendation_kgat.py"] + base_args
    if run["skip_groups"]:
        cmd += ["--skip-feature-groups", run["skip_groups"]]

    print(f"\n{'='*70}")
    print(f"  RUN: {run['name']}")
    print(f"  NOTE: {run['note']}")
    if run["skip_groups"]:
        print(f"  SKIP: {run['skip_groups']}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'='*70}\n", flush=True)

    t0     = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    # Re-run capturing output to parse metrics (stdout was already shown live above,
    # so we run once more with capture; alternatively parse from a log file)
    # ── Instead: run once with Tee so we capture AND stream ──────────────────
    # The above already ran and printed live.  We parse from a second invocation
    # only if the first failed; otherwise we rely on the grep below.
    # Better approach: capture output while streaming it.
    return {
        "name":    run["name"],
        "elapsed": elapsed,
        "rc":      result.returncode,
    }


def run_experiment_streaming(run: dict, base_args: list[str]) -> dict:
    """Run subprocess, stream stdout/stderr live, and capture for metric parsing."""
    cmd = [sys.executable, "recommendation_kgat.py"] + base_args
    if run["skip_groups"]:
        cmd += ["--skip-feature-groups", run["skip_groups"]]

    print(f"\n{'='*70}", flush=True)
    print(f"  RUN:  {run['name']}", flush=True)
    print(f"  NOTE: {run['note']}", flush=True)
    if run["skip_groups"]:
        print(f"  SKIP: {run['skip_groups']}", flush=True)
    print(f"  CMD:  {' '.join(cmd)}", flush=True)
    print(f"{'='*70}\n", flush=True)

    t0      = time.time()
    lines   = []
    proc    = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    proc.wait()
    elapsed = time.time() - t0

    output = "".join(lines)
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
    print(f"\n{'='*70}")
    print("  KGAT ABLATION SUMMARY")
    print(f"{'='*70}")
    header = f"  {'Run':<30}  {'Skip':<25}  {'Recall@10':>10}  {'NDCG@10':>9}  {'Time':>8}"
    print(header)
    print(f"  {'-'*28}  {'-'*23}  {'-'*10}  {'-'*9}  {'-'*8}")

    best_recall = max((r["recall"] for r in results if r["recall"] is not None), default=0.0)
    for r in results:
        recall_str = f"{r['recall']:.4f}" if r["recall"] is not None else "FAILED"
        ndcg_str   = f"{r['ndcg']:.4f}"   if r["ndcg"]   is not None else "FAILED"
        time_str   = str(timedelta(seconds=int(r["elapsed"])))
        marker     = " ***" if r["recall"] == best_recall else ""
        print(f"  {r['name']:<30}  {r['skip']:<25}  {recall_str:>10}  {ndcg_str:>9}  {time_str:>8}{marker}")

    print(f"{'='*70}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="KGAT feature-group ablation runner")
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--emb-dim",      type=int,   default=2048)
    parser.add_argument("--batch-size",   type=int,   default=4096)
    parser.add_argument("--eval-every",   type=int,   default=50)
    parser.add_argument("--runs",         nargs="*",  type=int, default=None,
                        help="Indices of runs to execute (0-based). Default: all.")
    args = parser.parse_args()

    base_args = [
        "--epochs",     str(args.epochs),
        "--emb-dim",    str(args.emb_dim),
        "--batch-size", str(args.batch_size),
        "--eval-every", str(args.eval_every),
        "--prebuilt-features",
    ]

    selected = RUNS if args.runs is None else [RUNS[i] for i in args.runs]

    print(f"\nKGAT ablation: {len(selected)} run(s) | "
          f"epochs={args.epochs} | emb_dim={args.emb_dim} | "
          f"batch_size={args.batch_size}", flush=True)

    results = []
    total_t0 = time.time()
    for run in selected:
        res = run_experiment_streaming(run, base_args)
        results.append(res)
        print_summary(results)   # print running summary after each run

    total_elapsed = time.time() - total_t0
    print(f"Total wall time: {timedelta(seconds=int(total_elapsed))}\n", flush=True)


if __name__ == "__main__":
    main()
