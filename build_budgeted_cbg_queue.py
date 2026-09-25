#!/usr/bin/env python3
"""Build a nationally seeded, density-first CBG queue for a capped API run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("data/national_density_cbgs.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planned-h3-cells", type=int, default=20_000)
    parser.add_argument("--high-density-share", type=int, default=4,
                        help="High-density rows per adjacent row after state seeding")
    args = parser.parse_args()
    if args.planned_h3_cells < 1 or args.high_density_share < 1:
        parser.error("queue limits must be positive")

    plan = pd.read_csv(args.plan, dtype={"cbg_str": str, "state_fips": str})
    pending = plan[plan["pending"].astype(bool)].copy()
    pending["cbg_str"] = pending.cbg_str.str.zfill(12)
    pending["state_fips"] = pending.state_fips.str.zfill(2)
    pending = pending.sort_values(
        ["population_density", "population"], ascending=False
    )

    high = pending[pending.selection_reason == "high_density"].copy()
    adjacent = pending[pending.selection_reason == "adjacent"].copy()
    selected: list[pd.Series] = []
    seen: set[str] = set()
    planned_cells = 0

    def add(row) -> None:
        nonlocal planned_cells
        cbg = str(row.cbg_str)
        if cbg not in seen:
            selected.append(row)
            seen.add(cbg)
            planned_cells += int(row.h3_cell_count)

    # Seed every state with its densest core and adjacent CBG before the global
    # density ranking can concentrate the early budget in a few metros.
    for frame in (high, adjacent):
        for _, row in frame.groupby("state_fips", sort=True).head(1).iterrows():
            add(row)

    high_rows = [row for _, row in high.iterrows() if row.cbg_str not in seen]
    adjacent_rows = [row for _, row in adjacent.iterrows() if row.cbg_str not in seen]
    hi = adj = 0
    while planned_cells < args.planned_h3_cells:
        progressed = False
        for _ in range(args.high_density_share):
            if hi < len(high_rows):
                add(high_rows[hi]); hi += 1; progressed = True
        if adj < len(adjacent_rows):
            add(adjacent_rows[adj]); adj += 1; progressed = True
        if not progressed:
            break

    queue = pd.DataFrame(selected)
    queue.insert(0, "queue_position", range(1, len(queue) + 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.output, index=False)
    summary = {
        "queue": str(args.output),
        "cbgs": len(queue),
        "states": int(queue.state_fips.nunique()),
        "high_density_cbgs": int((queue.selection_reason == "high_density").sum()),
        "adjacent_cbgs": int((queue.selection_reason == "adjacent").sum()),
        "planned_h3_cells": int(queue.h3_cell_count.sum()),
        "ordering": "one high-density and one adjacent seed per state, then 4:1 density-first",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
