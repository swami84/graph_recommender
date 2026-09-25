#!/usr/bin/env python3
"""Estimate Google Places discovery costs from the national CBG plan.

This script is read-only. It applies Google's monthly volume tiers to the H3
request upper bound and reports an empirical early-stop estimate based on the
existing request log.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TIERS = {
    "pro": {
        "free": 5_000,
        "rates": [(100_000, 32.00), (500_000, 25.60), (1_000_000, 19.20),
                  (5_000_000, 9.60), (float("inf"), 2.40)],
    },
    "enterprise": {
        "free": 1_000,
        "rates": [(100_000, 35.00), (500_000, 28.00), (1_000_000, 21.00),
                  (5_000_000, 10.50), (float("inf"), 2.63)],
    },
}


def tiered_cost(requests: int, sku: str) -> float:
    config = TIERS[sku]
    previous = min(requests, config["free"])
    total = 0.0
    for cap, dollars_per_thousand in config["rates"]:
        quantity = max(0, min(requests, cap) - previous)
        total += quantity * dollars_per_thousand / 1_000.0
        previous = max(previous, min(requests, cap))
        if requests <= cap:
            break
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("data/national_density_cbgs.csv"))
    parser.add_argument("--log", type=Path, default=Path("data/hex_api_call_log.csv"))
    args = parser.parse_args()

    plan = pd.read_csv(args.plan, dtype={"cbg_str": str, "state_fips": str})
    pending = plan[plan["pending"]].copy()
    upper = int(pending["h3_cell_count"].sum())

    empirical_requests = None
    empirical_ratio = None
    if args.log.exists():
        calls = pd.read_csv(args.log, dtype={"cbg": str})
        calls["cbg"] = calls["cbg"].str.zfill(12)
        unique_calls = calls.groupby("cbg")["h3_index"].nunique().rename("actual_calls")
        historical = plan.merge(unique_calls, left_on="cbg_str", right_index=True, how="inner")
        historical = historical[historical["already_collected"]].copy()
        historical["effective_calls"] = np.minimum(
            historical["actual_calls"], historical["h3_cell_count"].clip(lower=1)
        )
        empirical_ratio = historical["effective_calls"].sum() / historical["h3_cell_count"].sum()
        empirical_requests = int(round(upper * empirical_ratio))

    print(f"Pending CBGs: {len(pending):,}")
    print(f"Full-tiling request upper bound: {upper:,}")
    if empirical_requests is not None:
        print(f"Historical early-stop ratio: {empirical_ratio:.1%}")
        print(f"Empirical request estimate: {empirical_requests:,}")

    for sku in ["pro", "enterprise"]:
        print(f"\n{sku.title()} SKU")
        print(f"  upper-bound monthly cost: ${tiered_cost(upper, sku):,.2f}")
        if empirical_requests is not None:
            print(f"  empirical monthly cost:   ${tiered_cost(empirical_requests, sku):,.2f}")

    core = pending[pending["selection_reason"] == "high_density"]
    core_calls = int(core["h3_cell_count"].sum())
    print(f"\nHigh-density core only: {len(core):,} CBGs / {core_calls:,} calls")
    print(f"  Pro upper-bound cost: ${tiered_cost(core_calls, 'pro'):,.2f}")
    pilot_calls = int(core.nlargest(1_000, "population_density")["h3_cell_count"].sum())
    print(f"Top-1,000-density pilot: {pilot_calls:,} calls / "
          f"${tiered_cost(pilot_calls, 'pro'):,.2f} Pro")


if __name__ == "__main__":
    main()
