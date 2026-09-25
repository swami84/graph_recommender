#!/usr/bin/env python3
"""Run cuisine repair, strict assembly, then hand off to model reruns."""

from __future__ import annotations

import subprocess
from pathlib import Path

PYTHON = "/home/swami/venv/dev_env/bin/python"
PROJECT = Path(__file__).resolve().parent


def run(*args: str) -> None:
    subprocess.run(args, cwd=PROJECT, check=True)


def main() -> None:
    run(
        PYTHON, "classify_other_cuisines_27b.py",
        "--urls", "http://127.0.0.1:11436,http://127.0.0.1:11437",
        "--model", "qwen3.8-27b-24k:latest", "--concurrency", "12",
        "--log-every", "100",
    )
    run(PYTHON, "audit_and_assemble_llm_features.py")
    run("systemctl", "--user", "start", "--no-block",
        "foodie-publication-experiments.service")


if __name__ == "__main__":
    main()
