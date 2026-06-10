#!/usr/bin/env python3
"""
build_graph.py — Phase 2: construct a heterogeneous knowledge graph.

Nodes:  Restaurant, Reviewer, Dish, CBG, Cuisine
Edges:  REVIEWED, SERVES, MENTIONED_IN, LOCATED_IN, HAS_CUISINE

Outputs:
    data/graph.pkl          — NetworkX MultiDiGraph (all metadata on nodes/edges)
    data/graph_stats.txt    — node/edge counts by type

Usage:
    python build_graph.py
"""

import json
import logging
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_graph")

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT_RESTAURANTS = Path("data/restaurants_enriched.parquet")
OUT_REVIEWS     = Path("data/reviews_flat.parquet")
OUT_DISHES      = Path("data/dishes.parquet")
OUT_REV_DISHES  = Path("data/review_dishes.parquet")
GRAPH_FILE      = Path("data/graph.pkl")
STATS_FILE      = Path("data/graph_stats.txt")


def node_id(ntype: str, key: str) -> str:
    return f"{ntype}::{key}"


def build():
    # ── Load artefacts ─────────────────────────────────────────────────────────
    for p in [OUT_RESTAURANTS, OUT_REVIEWS]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p} — run build_graph_data.py first")

    restaurants = pd.read_parquet(OUT_RESTAURANTS)
    reviews     = pd.read_parquet(OUT_REVIEWS)
    dishes      = pd.read_parquet(OUT_DISHES) if OUT_DISHES.exists() else pd.DataFrame()
    rev_dishes  = pd.read_parquet(OUT_REV_DISHES) if OUT_REV_DISHES.exists() else pd.DataFrame()

    log.info(f"Loaded {len(restaurants):,} restaurants, {len(reviews):,} reviews, "
             f"{len(dishes):,} dishes")

    G = nx.MultiDiGraph()

    # ── Restaurant nodes ───────────────────────────────────────────────────────
    log.info("Adding Restaurant nodes…")
    for _, r in restaurants.iterrows():
        nid = node_id("restaurant", r["place_id"])
        G.add_node(nid,
            ntype="restaurant",
            place_id=r["place_id"],
            name=r.get("name", ""),
            cuisine=r.get("cuisine_category", "Other"),
            price_level=r.get("price_level"),
            rating=r.get("rating"),
            review_count=r.get("user_rating_count"),
            lat=r.get("lat"),
            lng=r.get("lng"),
            cbg=r.get("cbg", ""),
        )

    # ── Reviewer nodes ─────────────────────────────────────────────────────────
    log.info("Adding Reviewer nodes…")
    reviewer_agg = (
        reviews[reviews["contributor_id"].notna() & (reviews["contributor_id"] != "")]
        .drop_duplicates("contributor_id")
        [["contributor_id", "reviewer_name", "is_local_guide",
          "reviewer_reviews", "reviewer_photos",
          "predicted_race", "predicted_gender"]]
    )
    for _, r in reviewer_agg.iterrows():
        nid = node_id("reviewer", r["contributor_id"])
        G.add_node(nid,
            ntype="reviewer",
            contributor_id=r["contributor_id"],
            name=r.get("reviewer_name", ""),
            is_local_guide=bool(r.get("is_local_guide", False)),
            total_reviews=r.get("reviewer_reviews"),
            total_photos=r.get("reviewer_photos"),
            predicted_race=r.get("predicted_race"),
            predicted_gender=r.get("predicted_gender"),
        )

    # ── Dish nodes ─────────────────────────────────────────────────────────────
    if not dishes.empty:
        log.info("Adding Dish nodes…")
        for _, d in dishes.iterrows():
            nid = node_id("dish", d["dish_id"])
            G.add_node(nid,
                ntype="dish",
                dish_id=d["dish_id"],
                dish_name=d["dish_name"],
                place_id=d["place_id"],
            )

    # ── CBG nodes ──────────────────────────────────────────────────────────────
    log.info("Adding CBG nodes…")
    cbgs = restaurants["cbg"].dropna().unique()
    for cbg in cbgs:
        nid = node_id("cbg", str(cbg))
        G.add_node(nid,
            ntype="cbg",
            cbg=str(cbg),
            state_fips=str(cbg)[:2] if len(str(cbg)) >= 2 else "",
        )

    # ── Cuisine nodes ──────────────────────────────────────────────────────────
    log.info("Adding Cuisine nodes…")
    cuisines = restaurants["cuisine_category"].dropna().unique()
    for c in cuisines:
        nid = node_id("cuisine", c)
        G.add_node(nid, ntype="cuisine", cuisine=c)

    # ── REVIEWED edges  (Reviewer → Restaurant) ────────────────────────────────
    log.info("Adding REVIEWED edges…")
    valid = reviews[
        reviews["contributor_id"].notna() &
        (reviews["contributor_id"] != "") &
        reviews["place_id"].notna()
    ]
    for _, r in valid.iterrows():
        src = node_id("reviewer",    r["contributor_id"])
        dst = node_id("restaurant",  r["place_id"])
        if src in G and dst in G:
            G.add_edge(src, dst,
                etype="REVIEWED",
                review_id=r.get("review_id", ""),
                rating=r.get("rating"),
                has_content=bool(r.get("has_content", False)),
                meal_type=r.get("meal_type"),
                price_per_person=r.get("price_per_person"),
                food_score=r.get("food_score"),
                service_score=r.get("service_score"),
                atmosphere_score=r.get("atmosphere_score"),
                timestamp_days_ago=r.get("timestamp_days_ago"),
            )

    # ── SERVES edges  (Restaurant → Dish) ─────────────────────────────────────
    if not dishes.empty:
        log.info("Adding SERVES edges…")
        # review_dishes has dish_name+place_id but not dish_id — join to get it
        if not rev_dishes.empty and "dish_id" not in rev_dishes.columns:
            rev_dishes = rev_dishes.merge(
                dishes[["dish_name", "place_id", "dish_id"]],
                on=["dish_name", "place_id"],
                how="left",
            )
        mention_counts = (
            rev_dishes.groupby(["place_id", "dish_id"])
            .size()
            .reset_index(name="mention_count")
            if not rev_dishes.empty else pd.DataFrame()
        )
        for _, d in dishes.iterrows():
            src = node_id("restaurant", d["place_id"])
            dst = node_id("dish",       d["dish_id"])
            if src in G and dst in G:
                count = 1
                if not mention_counts.empty:
                    row = mention_counts[
                        (mention_counts["place_id"] == d["place_id"]) &
                        (mention_counts["dish_id"]  == d["dish_id"])
                    ]
                    if not row.empty:
                        count = int(row["mention_count"].iloc[0])
                G.add_edge(src, dst, etype="SERVES", mention_count=count)

    # ── LOCATED_IN edges  (Restaurant → CBG) ──────────────────────────────────
    log.info("Adding LOCATED_IN edges…")
    for _, r in restaurants[restaurants["cbg"].notna()].iterrows():
        src = node_id("restaurant", r["place_id"])
        dst = node_id("cbg",        str(r["cbg"]))
        if src in G and dst in G:
            G.add_edge(src, dst, etype="LOCATED_IN")

    # ── HAS_CUISINE edges  (Restaurant → Cuisine) ─────────────────────────────
    log.info("Adding HAS_CUISINE edges…")
    for _, r in restaurants[restaurants["cuisine_category"].notna()].iterrows():
        src = node_id("restaurant", r["place_id"])
        dst = node_id("cuisine",    r["cuisine_category"])
        if src in G and dst in G:
            G.add_edge(src, dst, etype="HAS_CUISINE")

    # ── Persist ────────────────────────────────────────────────────────────────
    log.info(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    with open(GRAPH_FILE, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"Graph saved → {GRAPH_FILE}")

    # Stats breakdown
    node_counts = {}
    for _, d in G.nodes(data=True):
        t = d.get("ntype", "unknown")
        node_counts[t] = node_counts.get(t, 0) + 1

    edge_counts = {}
    for _, _, d in G.edges(data=True):
        t = d.get("etype", "unknown")
        edge_counts[t] = edge_counts.get(t, 0) + 1

    stats_lines = ["=== Graph Statistics ===\n", "Nodes:\n"]
    for t, c in sorted(node_counts.items()):
        stats_lines.append(f"  {t:<20} {c:>10,}\n")
    stats_lines.append(f"  {'TOTAL':<20} {sum(node_counts.values()):>10,}\n\n")
    stats_lines.append("Edges:\n")
    for t, c in sorted(edge_counts.items()):
        stats_lines.append(f"  {t:<20} {c:>10,}\n")
    stats_lines.append(f"  {'TOTAL':<20} {sum(edge_counts.values()):>10,}\n")

    stats_str = "".join(stats_lines)
    STATS_FILE.write_text(stats_str)
    print(stats_str)
    log.info(f"Stats saved → {STATS_FILE}")


if __name__ == "__main__":
    build()
