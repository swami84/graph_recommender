#!/usr/bin/env python3
"""Run the publication core: 3 models × 2 feature sets × 3 seeds.

Models: feature-aware Two-Tower (non-graph), LightGCN, and KGAT-SAL.
Conditions: conventional non-LLM features, and conventional + structured LLM
features + LLM embeddings. Proximity is intentionally a later, validation-only
stage applied to the selected winner—not part of this core runner.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("publication_experiments")

FEATURE_DIR = Path("data/publication_features")
MANIFEST = Path("results/publication_experiment_manifest.json")
PROJECT_PYTHON = Path("/home/swami/venv/dev_env/bin/python")
PYTHON = str(PROJECT_PYTHON if PROJECT_PYTHON.exists() else Path(sys.executable))
REQUIRED_LLM = [
    Path("data/audited_llm_features/restaurant_llm_features.parquet"),
    Path("data/audited_llm_features/user_llm_features.parquet"),
    Path("data/restaurant_llm_embeddings.parquet"), Path("data/user_llm_embeddings.parquet"),
]
MODEL_SPECS = {
    "two_tower": ("recommendation_two_tower.py", ["--batch-size", "32768"]),
    "lightgcn": ("recommendation_gnn.py", ["--layers", "3", "--batch-size", "4096"]),
    "kgat_sal": ("recommendation_kgat_sal.py",
                 ["--cf-layers", "4", "--kg-layers", "2", "--n-periods", "3",
                  "--batch-size", "4096"]),
}
CONDITIONS = {
    "non_llm": "llm,llm_embedding,dish_llm",
    "structured_only": "llm_embedding",
    "embeddings_only": "llm",
    "full_llm": "",
}


def prepare_features(dry_run: bool, rebuild: bool) -> None:
    missing = [str(p) for p in REQUIRED_LLM if not p.exists()]
    if missing:
        raise FileNotFoundError("Feature generation incomplete: " + ", ".join(missing))
    outputs = [FEATURE_DIR / "user_features_prebuilt.parquet",
               FEATURE_DIR / "item_features_prebuilt.parquet",
               FEATURE_DIR / "feature_groups.json"]
    if rebuild or not all(p.exists() for p in outputs):
        cmd = [PYTHON, "build_training_features.py", "--llm-feat-mode", "full",
               "--publication-split", "--output-dir", str(FEATURE_DIR)]
        log.info("Leakage-safe feature assembly: %s", " ".join(cmd))
        if not dry_run:
            env = os.environ.copy()
            env.update({
                "FOODIE_RESTAURANT_LLM_FEATURES": str(REQUIRED_LLM[0]),
                "FOODIE_USER_LLM_FEATURES": str(REQUIRED_LLM[1]),
            })
            subprocess.run(cmd, check=True, env=env)
    elif not dry_run:
        groups = json.loads(outputs[2].read_text())
        required_groups = {
            "user": ("base", "extended", "llm", "llm_embedding"),
            "item": ("base", "extended", "dish_llm", "llm", "llm_embedding"),
        }
        for side, side_groups in required_groups.items():
            for group in side_groups:
                if not groups.get(side, {}).get(group):
                    raise ValueError(f"Publication feature group is empty: {side}.{group}")
        log.info("Publication feature cache passed group preflight: %s", FEATURE_DIR)


def stop_dedicated_foodie_ollama() -> None:
    """Release only Foodie's private Ollama workers; never touch shared port 11434."""
    services = ["foodie-ollama-gpu0.service", "foodie-ollama-gpu1.service"]
    subprocess.run(["systemctl", "--user", "stop", *services], check=False)
    log.info("Stopped dedicated Foodie Ollama workers (shared Ollama left untouched)")


def make_runs(models: list[str], conditions: list[str], seeds: list[int],
              epochs: int, emb_dim: int):
    runs = []
    for seed in seeds:
        for condition in conditions:
            skipped = CONDITIONS[condition]
            for model_name in models:
                script, specific = MODEL_SPECS[model_name]
                args = ["--epochs", str(epochs), "--emb-dim", str(emb_dim),
                        "--prebuilt-features", "--llm-feat-mode", "full",
                        "--eval-every", "25", "--publication-split", "--seed", str(seed),
                        *specific]
                if skipped:
                    args += ["--skip-feature-groups", skipped]
                runs.append({"model": model_name, "condition": condition, "seed": seed,
                             "script": script, "args": args})
    return runs


def prediction_path(run: dict) -> Path:
    skip = CONDITIONS[run["condition"]]
    condition = "all" if not skip else f"skip_{skip.replace(',', '_')}"
    seed = run["seed"]
    names = {
        "two_tower": f"two_tower_{condition}_pubsplit_seed{seed}_predictions.parquet",
        "lightgcn": f"lightgcn_3l_{condition}_pubsplit_seed{seed}_predictions.parquet",
        "kgat_sal": f"kgat_sal_{condition}_T3_pubsplit_seed{seed}_predictions.parquet",
    }
    return Path("data/predictions") / names[run["model"]]


def run_on_two_gpus(runs, dry_run: bool, force: bool):
    queue = list(runs)
    completed = [] if force or dry_run else [r for r in queue if prediction_path(r).exists()]
    if completed:
        done_keys = {(r["model"], r["condition"], r["seed"]) for r in completed}
        queue = [r for r in queue if (r["model"], r["condition"], r["seed"]) not in done_keys]
        log.info("Resume: skipping %d completed runs; %d remain", len(completed), len(queue))
    failed = []
    if dry_run:
        for index, run in enumerate(queue):
            gpu = index % 2
            cmd = [PYTHON, run["script"], *run["args"]]
            log.info("GPU %d | %s/%s seed=%d | %s", gpu, run["model"],
                     run["condition"], run["seed"], " ".join(cmd))
            completed.append(run)
        return completed, failed

    # Work-conserving scheduler: when one model finishes, immediately launch
    # the next run on that GPU instead of waiting for the slower paired model.
    active: dict[int, tuple[dict, subprocess.Popen]] = {}
    while queue or active:
        for gpu in (0, 1):
            if gpu in active or not queue:
                continue
            run = queue.pop(0)
            cmd = [PYTHON, run["script"], *run["args"]]
            log.info("GPU %d | %s/%s seed=%d | %s", gpu, run["model"],
                     run["condition"], run["seed"], " ".join(cmd))
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": str(gpu),
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                        "FOODIE_FEATURE_DIR": str(FEATURE_DIR),
                        "FOODIE_DISHES_FILE": str(FEATURE_DIR / "dishes_train.parquet")})
            active[gpu] = (run, subprocess.Popen(cmd, env=env))

        finished = []
        for gpu, (run, proc) in active.items():
            code = proc.poll()
            if code is None:
                continue
            (completed if code == 0 else failed).append(run)
            finished.append(gpu)
        for gpu in finished:
            del active[gpu]
        if finished:
            write_manifest(runs, completed, failed, dry_run=False)
        if failed:
            for _, proc in active.values():
                proc.terminate()
            for _, proc in active.values():
                proc.wait()
            break
        if active and not finished:
            time.sleep(2)
    return completed, failed


def write_manifest(runs, completed, failed, dry_run):
    MANIFEST.parent.mkdir(exist_ok=True)
    payload = {"updated_at": datetime.now().astimezone().isoformat(),
               "design": f"{len(set(x['model'] for x in runs))} models x "
                         f"{len(set(x['condition'] for x in runs))} feature conditions x "
                         f"{len(set(x['seed'] for x in runs))} seeds",
               "primary_metrics": ["NDCG@10", "Hit@10"],
               "selection": "validation NDCG@10; test evaluated once",
               "feature_dir": str(FEATURE_DIR), "dry_run": dry_run,
               "planned": runs, "completed": completed, "failed": failed}
    if not dry_run:
        MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--emb-dim", type=int, default=1024)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--models", default="two_tower,lightgcn,kgat_sal")
    parser.add_argument("--conditions", default="non_llm,full_llm",
                        help="Comma-separated: non_llm,structured_only,embeddings_only,full_llm")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--keep-foodie-ollama-loaded", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Rerun conditions even when their prediction artifact exists")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    unknown = set(models) - set(MODEL_SPECS)
    if unknown:
        parser.error("Unknown models: " + ", ".join(sorted(unknown)))
    unknown_conditions = set(conditions) - set(CONDITIONS)
    if unknown_conditions:
        parser.error("Unknown conditions: " + ", ".join(sorted(unknown_conditions)))
    prepare_features(args.dry_run, args.rebuild_features)
    runs = make_runs(models, conditions, seeds, args.epochs, args.emb_dim)
    if not args.keep_foodie_ollama_loaded and not args.dry_run:
        stop_dedicated_foodie_ollama()
    completed, failed = run_on_two_gpus(runs, args.dry_run, args.force)
    write_manifest(runs, completed, failed, args.dry_run)
    if failed:
        raise SystemExit("Experiment failed: " + ", ".join(
            f"{x['model']}/{x['condition']}/seed{x['seed']}" for x in failed))
    log.info("Core experiment %s: %d/%d runs", "plan" if args.dry_run else "complete",
             len(completed), len(runs))


if __name__ == "__main__":
    main()
