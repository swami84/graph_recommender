#!/usr/bin/env python3
"""Run the frozen three-model experiment on the deterministic expanded batch."""

from __future__ import annotations

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
log = logging.getLogger("expanded_experiments")

ROOT = Path(__file__).resolve().parent
BATCH = ROOT / "data/feature_batches/places_expansion_2026-09-16_split_v2"
CANONICAL = ROOT / "data/feature_batches/places_expansion_2026-09-16"
ASSEMBLED = BATCH / "assembled"
FEATURE_DIR = BATCH / "training_features"
RESULT_DIR = ROOT / "results/expanded_2026-09-21"
ARTIFACT_DIR = ROOT / "data/expanded_experiment_2026-09-21"
MODEL_DIR = ROOT / "models/expanded_2026-09-21"
EMBEDDING_DIR = ARTIFACT_DIR / "model_embeddings"
PREDICTION_DIR = ARTIFACT_DIR / "predictions"
PYTHON = str(Path("/home/swami/venv/dev_env/bin/python") if Path(
    "/home/swami/venv/dev_env/bin/python").exists() else Path(sys.executable))
MANIFEST = RESULT_DIR / "experiment_manifest.json"

MODEL_SPECS = {
    "two_tower": ("recommendation_two_tower.py", ["--batch-size", "32768"]),
    "lightgcn": ("recommendation_gnn.py", ["--layers", "3", "--batch-size", "4096"]),
    "kgat_sal": ("recommendation_kgat_sal.py",
                 ["--cf-layers", "4", "--kg-layers", "2", "--n-periods", "3",
                  "--batch-size", "4096"]),
}
CONDITIONS = {"non_llm": "llm,llm_embedding,dish_llm", "full_llm": ""}


def common_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "FOODIE_REVIEWS_FLAT": str(CANONICAL / "reviews_flat.parquet"),
        "FOODIE_RESTAURANTS_ENR": str(CANONICAL / "restaurants_enriched.parquet"),
        "FOODIE_RESTAURANT_LLM_FEATURES": str(ASSEMBLED / "restaurant_llm_features.parquet"),
        "FOODIE_USER_LLM_FEATURES": str(ASSEMBLED / "user_llm_features.parquet"),
        "FOODIE_RESTAURANT_LLM_EMBEDDINGS": str(ASSEMBLED / "restaurant_llm_embeddings.parquet"),
        "FOODIE_USER_LLM_EMBEDDINGS": str(ASSEMBLED / "user_llm_embeddings.parquet"),
        "FOODIE_FEATURE_DIR": str(FEATURE_DIR),
        "FOODIE_DISHES_FILE": str(FEATURE_DIR / "dishes_train.parquet"),
        "FOODIE_PUBLICATION_RESULTS": str(RESULT_DIR / "core_results.csv"),
        "FOODIE_MODEL_RESULTS": str(RESULT_DIR / "model_results.csv"),
        "FOODIE_TRAINING_CHECKPOINTS": str(RESULT_DIR / "training_checkpoints.csv"),
        "FOODIE_MODEL_DIR": str(MODEL_DIR),
        "FOODIE_EMBEDDING_DIR": str(EMBEDDING_DIR),
        "FOODIE_PREDICTION_DIR": str(PREDICTION_DIR),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    return env


def prepare_features() -> None:
    required = [
        CANONICAL / "reviews_flat.parquet", CANONICAL / "restaurants_enriched.parquet",
        ASSEMBLED / "restaurant_llm_features.parquet", ASSEMBLED / "user_llm_features.parquet",
        ASSEMBLED / "restaurant_llm_embeddings.parquet", ASSEMBLED / "user_llm_embeddings.parquet",
        ASSEMBLED / "assembly_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Expanded contract incomplete: " + ", ".join(missing))
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, "build_training_features.py", "--llm-feat-mode", "full",
           "--publication-split", "--output-dir", str(FEATURE_DIR)]
    log.info("Building expanded split-safe feature matrices")
    subprocess.run(cmd, cwd=ROOT, env=common_env(), check=True)
    groups = json.loads((FEATURE_DIR / "feature_groups.json").read_text())
    expected = {
        "user": {"base", "extended", "llm", "llm_embedding"},
        "item": {"base", "extended", "dish_llm", "llm", "llm_embedding"},
    }
    for side, names in expected.items():
        if set(groups[side]) != names or any(not groups[side][name] for name in names):
            raise ValueError(f"Incomplete expanded feature groups for {side}")


def make_runs() -> list[dict]:
    runs = []
    for seed in (42, 43, 44):
        for condition, skipped in CONDITIONS.items():
            for model, (script, specific) in MODEL_SPECS.items():
                args = ["--epochs", "300", "--emb-dim", "1024", "--prebuilt-features",
                        "--llm-feat-mode", "full", "--eval-every", "25",
                        "--publication-split", "--seed", str(seed), *specific]
                if skipped:
                    args += ["--skip-feature-groups", skipped]
                runs.append({"model": model, "condition": condition, "seed": seed,
                             "script": script, "args": args})
    return runs


def prediction_path(run: dict) -> Path:
    condition = "all" if not CONDITIONS[run["condition"]] else "skip_llm_llm_embedding_dish_llm"
    names = {
        "two_tower": f"two_tower_{condition}_pubsplit_seed{run['seed']}_predictions.parquet",
        "lightgcn": f"lightgcn_3l_{condition}_pubsplit_seed{run['seed']}_predictions.parquet",
        "kgat_sal": f"kgat_sal_{condition}_T3_pubsplit_seed{run['seed']}_predictions.parquet",
    }
    return PREDICTION_DIR / names[run["model"]]


def write_manifest(runs: list[dict], completed: list[dict], failed: list[dict]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    prior = ROOT / "results/publication_experiment_manifest.json"
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(),
        "dataset": "places_expansion_2026-09-16_split_v2",
        "design": "3 models x 2 feature conditions x 3 seeds",
        "comparability": "Same models, conditions, seeds, epochs, dimensions, layers, validation selection, and full-catalog metrics as earlier publication run",
        "earlier_results_untouched": True,
        "earlier_manifest": str(prior),
        "feature_dir": str(FEATURE_DIR), "results_dir": str(RESULT_DIR),
        "planned": runs, "completed": completed, "failed": failed,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")


def run_models(runs: list[dict]) -> None:
    completed = [run for run in runs if prediction_path(run).exists()]
    done = {(run["model"], run["condition"], run["seed"]) for run in completed}
    queue = [run for run in runs if (run["model"], run["condition"], run["seed"]) not in done]
    failed: list[dict] = []
    active: dict[int, tuple[dict, subprocess.Popen]] = {}
    write_manifest(runs, completed, failed)
    while queue or active:
        for gpu in (0, 1):
            if gpu in active or not queue:
                continue
            run = queue.pop(0)
            env = common_env()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = [PYTHON, run["script"], *run["args"]]
            log.info("GPU %d | %s/%s seed=%d", gpu, run["model"], run["condition"], run["seed"])
            active[gpu] = (run, subprocess.Popen(cmd, cwd=ROOT, env=env))
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
            write_manifest(runs, completed, failed)
        if failed:
            for _, proc in active.values():
                proc.terminate()
            for _, proc in active.values():
                proc.wait()
            raise SystemExit("Expanded experiment failed: " + ", ".join(
                f"{x['model']}/{x['condition']}/seed{x['seed']}" for x in failed))
        if active and not finished:
            time.sleep(2)


def main() -> None:
    for path in (RESULT_DIR, ARTIFACT_DIR, MODEL_DIR, EMBEDDING_DIR, PREDICTION_DIR):
        path.mkdir(parents=True, exist_ok=True)
    prepare_features()
    # Dedicated Ollama workers are no longer needed and occupy both GPUs.
    subprocess.run(["systemctl", "--user", "stop", "foodie-ollama-gpu0.service",
                    "foodie-ollama-gpu1.service"], check=False)
    runs = make_runs()
    run_models(runs)
    subprocess.run([PYTHON, "summarize_expanded_experiments.py"], cwd=ROOT,
                   env=common_env(), check=True)


if __name__ == "__main__":
    main()
