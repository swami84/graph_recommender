import csv

import pytest
import h3
from shapely.geometry import Polygon

from hexagon_places import PlacesBudgetExceeded, PlacesBudgetGuard, cbg_to_hexagons


FIELDS = [
    "timestamp", "cbg", "h3_index", "lat", "lng", "radius",
    "response_status", "num_results", "estimated_cost_usd",
]


def write_log(path, timestamps):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for number, timestamp in enumerate(timestamps):
            writer.writerow({
                "timestamp": timestamp, "cbg": "1", "h3_index": str(number),
                "response_status": "OK", "estimated_cost_usd": "0.032",
            })


def test_monthly_call_guard_blocks_before_reservation(tmp_path):
    log = tmp_path / "calls.csv"
    write_log(log, ["2026-09-01T00:00:00+00:00"] * 3)
    guard = PlacesBudgetGuard(
        "monthly-test", "2026-09", 3, 10.0,
        api_log_file=log, budget_dir=tmp_path / "budget",
    )
    with pytest.raises(PlacesBudgetExceeded, match="monthly request ceiling"):
        guard.reserve("cbg", "hex")
    assert guard.status()["reserved_calls"] == 0


def test_run_spend_guard_is_persistent_and_counts_failed_calls(tmp_path):
    log = tmp_path / "calls.csv"
    write_log(log, [])
    kwargs = dict(
        budget_id="run-test", billing_month="2026-09",
        max_monthly_calls=100, max_run_cost_usd=0.064,
        api_log_file=log, budget_dir=tmp_path / "budget",
    )
    first = PlacesBudgetGuard(**kwargs)
    first.reserve("cbg", "hex-1")
    first.reserve("cbg", "hex-2")
    restarted = PlacesBudgetGuard(**kwargs)
    with pytest.raises(PlacesBudgetExceeded, match="run spend ceiling"):
        restarted.reserve("cbg", "hex-3")
    status = restarted.status()
    assert status["reserved_calls"] == 2
    assert status["remaining_run_calls"] == 0


def test_priority_search_center_controls_first_h3_cell():
    geometry = Polygon([
        (-68.22, 44.30), (-68.17, 44.30),
        (-68.17, 44.39), (-68.22, 44.39),
    ])
    search_center = (44.3876, -68.2043)
    cells = cbg_to_hexagons(geometry, search_center=search_center)
    distances = [
        (h3.cell_to_latlng(cell)[0] - search_center[0]) ** 2
        + (h3.cell_to_latlng(cell)[1] - search_center[1]) ** 2
        for cell in cells
    ]
    assert distances[0] == min(distances)
    assert cells[0] == h3.latlng_to_cell(*search_center, 9)
