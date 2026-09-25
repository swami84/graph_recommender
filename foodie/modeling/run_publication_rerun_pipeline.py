#!/usr/bin/env python3
"""Run the frozen 3-model x 2-condition x 3-seed publication experiment."""

from __future__ import annotations

import subprocess
from pathlib import Path

PYTHON = "/home/swami/venv/dev_env/bin/python"
PROJECT = Path(__file__).resolve().parents[2]


def main() -> None:
    subprocess.run(
        [PYTHON, "-m", "foodie.modeling.run_focused_llm_experiments",
         "--conditions", "non_llm,full_llm"],
        cwd=PROJECT, check=True,
    )
    subprocess.run(
        [PYTHON, "-m", "foodie.modeling.summarize_publication_experiments"],
        cwd=PROJECT, check=True,
    )


if __name__ == "__main__":
    main()
