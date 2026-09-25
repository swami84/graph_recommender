#!/usr/bin/env python3
"""Run the frozen 3-model x 2-condition x 3-seed publication experiment."""

from __future__ import annotations

import subprocess
from pathlib import Path

PYTHON = "/home/swami/venv/dev_env/bin/python"
PROJECT = Path(__file__).resolve().parent


def main() -> None:
    subprocess.run(
        [PYTHON, "run_focused_llm_experiments.py",
         "--conditions", "non_llm,full_llm"],
        cwd=PROJECT, check=True,
    )
    subprocess.run(
        [PYTHON, "summarize_publication_experiments.py"],
        cwd=PROJECT, check=True,
    )


if __name__ == "__main__":
    main()
