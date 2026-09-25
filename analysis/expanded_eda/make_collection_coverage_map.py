#!/usr/bin/env python3
"""Map the terminally searched US Census block groups used for discovery."""

from pathlib import Path
import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from article.figures.publication_style import (  # noqa: E402
    apply_style, save, INK, MUTED, GRID, TEAL, PURPLE, LIGHT,
)


GEOMETRY = ROOT / "data" / "census_geometry_2024"
DONE = ROOT / "data" / "hex_restaurants" / ".done"
DENSITY = ROOT / "data" / "top10000_density_cbgs.csv"
OUTPUT = ROOT / "article" / "figures" / "collection_coverage_map.png"
TABLE = ROOT / "analysis" / "publication_eda" / "tables" / "collection_map_coverage.csv"

STATE_ABBR = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE",
    "11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA",
    "20":"KS","21":"KY","22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN",
    "28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM",
    "36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI",
    "45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA",
    "54":"WV","55":"WI","56":"WY",
}

MAJOR_CITIES = [
    ("Seattle", -122.33, 47.61, 0.35, 0.20), ("San Francisco", -122.42, 37.77, 0.45, 0.15),
    ("Los Angeles", -118.24, 34.05, 0.45, -0.28), ("Phoenix", -112.07, 33.45, 0.35, -0.25),
    ("Denver", -104.99, 39.74, 0.35, 0.18), ("Dallas", -96.80, 32.78, 0.35, -0.25),
    ("Houston", -95.37, 29.76, 0.35, -0.25), ("Minneapolis", -93.27, 44.98, 0.30, 0.22),
    ("Chicago", -87.63, 41.88, 0.30, 0.22), ("Detroit", -83.05, 42.33, 0.25, 0.22),
    ("Atlanta", -84.39, 33.75, 0.30, -0.28), ("Miami", -80.19, 25.76, 0.30, -0.25),
    ("Washington, DC", -77.04, 38.91, -0.40, -0.34), ("Philadelphia", -75.17, 39.95, 0.28, 0.18),
    ("New York", -74.01, 40.71, 0.30, 0.25), ("Boston", -71.06, 42.36, 0.22, 0.25),
]


def load_geography(done_ids: set[str]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    selected, states = [], []
    for path in sorted(GEOMETRY.glob("cb_2024_*_bg_500k.zip")):
        state_fips = path.name.split("_")[2]
        frame = gpd.read_file(path)[["GEOID", "geometry"]].rename(columns={"GEOID": "cbg"})
        frame["cbg"] = frame.cbg.astype(str).str.zfill(12)
        states.append(gpd.GeoDataFrame(
            {"state_fips": [state_fips]}, geometry=[frame.geometry.union_all()], crs=frame.crs
        ))
        subset = frame[frame.cbg.isin(done_ids)].copy()
        if not subset.empty:
            selected.append(subset)
    return (
        gpd.GeoDataFrame(pd.concat(selected, ignore_index=True), crs=selected[0].crs).to_crs(4326),
        gpd.GeoDataFrame(pd.concat(states, ignore_index=True), crs=states[0].crs).to_crs(4326),
    )


def draw_region(ax, states: gpd.GeoDataFrame, coverage: gpd.GeoDataFrame,
                bounds: tuple[float, float, float, float], label: str | None = None,
                annotate_mainland: bool = False) -> None:
    xmin, xmax, ymin, ymax = bounds
    state_view = states.cx[xmin:xmax, ymin:ymax]
    coverage_view = coverage.cx[xmin:xmax, ymin:ymax]
    ax.set_facecolor("#EAF3F8")
    state_view.plot(ax=ax, color="#F4F1E8", edgecolor="white", linewidth=0.75, zorder=0)
    for category, color in [
        ("Additional high-foot-traffic / surrounding coverage", PURPLE),
        ("Top population-density coverage", TEAL),
    ]:
        layer = coverage_view[coverage_view.coverage == category]
        if not layer.empty:
            layer.plot(ax=ax, color=color, edgecolor="none", alpha=0.92, zorder=2)
            # CBG polygons become sub-pixel at national scale. Representative-point
            # overlays preserve the actual locations without implying larger coverage.
            points = layer.geometry.representative_point()
            ax.scatter(points.x, points.y, s=4.0 if category.startswith("Top") else 3.0,
                       color=color, alpha=0.72, linewidths=0, zorder=3)
    state_view.boundary.plot(ax=ax, color="#9AA8B4", linewidth=0.62, zorder=4)
    if annotate_mainland:
        # Postal labels retain legibility at national scale, especially in the Northeast.
        for row in state_view.itertuples(index=False):
            point = row.geometry.representative_point()
            abbr = STATE_ABBR.get(str(row.state_fips).zfill(2))
            if abbr:
                ax.text(point.x, point.y, abbr, ha="center", va="center", fontsize=6.2,
                        fontweight="bold", color="#475467", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.68))
        for city, lng, lat, dx, dy in MAJOR_CITIES:
            if xmin <= lng <= xmax and ymin <= lat <= ymax:
                ax.scatter([lng], [lat], s=13, color=INK, edgecolor="white", linewidth=0.5, zorder=7)
                ax.text(lng + dx, lat + dy, city, fontsize=7.2, color=INK,
                        fontweight="bold", zorder=7,
                        ha="left" if dx >= 0 else "right",
                        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.82))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if label:
        ax.text(0.04, 0.91, label, transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", color=INK)


def main() -> None:
    apply_style()
    done_ids = {path.stem.zfill(12) for path in DONE.glob("*.done")}
    density_ids = set(pd.read_csv(DENSITY, dtype={"cbg_str": str}).cbg_str.str.zfill(12))
    coverage, states = load_geography(done_ids)
    coverage["coverage"] = coverage.cbg.map(
        lambda value: "Top population-density coverage" if value in density_ids
        else "Additional high-foot-traffic / surrounding coverage"
    )
    summary = coverage.coverage.value_counts().rename_axis("coverage").reset_index(name="mapped_cbgs")
    summary["terminal_marker_cbgs"] = len(done_ids)
    summary["unmatched_to_2024_geometry"] = len(done_ids - set(coverage.cbg))
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE, index=False)

    fig = plt.figure(figsize=(15.5, 8.5))
    main_ax = fig.add_axes([0.04, 0.11, 0.75, 0.73])
    alaska_ax = fig.add_axes([0.76, 0.44, 0.21, 0.25])
    hawaii_ax = fig.add_axes([0.76, 0.18, 0.21, 0.18])
    draw_region(main_ax, states, coverage, (-126, -66, 24, 50.5), annotate_mainland=True)
    draw_region(alaska_ax, states, coverage, (-180, -129, 50, 72), "Alaska")
    draw_region(hawaii_ax, states, coverage, (-161.2, -154.5, 18.5, 22.5), "Hawaii")

    fig.suptitle("Geographic coverage of US restaurant discovery", fontsize=19,
                 fontweight="bold", color=INK, y=0.97)
    fig.text(
        0.5, 0.91,
        f"{len(coverage):,} terminally searched CBGs mapped to 2024 boundaries; "
        "coverage combines population density, foot traffic, and surrounding areas",
        ha="center", fontsize=11, color=MUTED,
    )
    fig.legend(
        handles=[
            Patch(facecolor=TEAL, label="Top population-density coverage"),
            Patch(facecolor=PURPLE, label="Additional high-foot-traffic / surrounding coverage"),
            Patch(facecolor="#F4F1E8", edgecolor="#9AA8B4", label="State basemap"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=3, frameon=False, fontsize=10,
    )
    fig.text(0.965, 0.105, "Shading represents searched CBGs—not restaurant counts",
             ha="right", fontsize=8.5, color=MUTED)
    save(fig, OUTPUT)
    print(summary.to_string(index=False))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
