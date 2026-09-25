"""Figures added during the technical-editing pass.

Every value is read from a frozen artifact under analysis/ or results/. No new model
runs and no synthetic data. Regenerate from the project root with:

    export MPLCONFIGDIR=/tmp/foodie-mpl
    python article/figures/make_added_analysis_figures.py
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from publication_style import (INK, MUTED, GRID, NAVY, TEAL, CORAL, GOLD, PURPLE,
                               LIGHT, RATING_COLORS, apply_style, clean_axis, save, title)

ROOT = Path(__file__).resolve().parents[2]
# Data root defaults to the repository root; override with FOODIE_DATA_ROOT when the
# frozen artifacts live elsewhere (for example a staged copy).
IN = Path(os.environ.get("FOODIE_DATA_ROOT", ROOT))
OUT = Path(__file__).resolve().parent
apply_style()

N_RESTAURANTS = 75_203
N_TEST = 156_261
N_CANDIDATES = 61_204


# --------------------------------------------------------------------------- Part 1
def evidence_mass():
    rb = pd.read_csv(IN / "analysis/publication_eda/tables/review_behavior_by_rating.csv")
    rb["chars"] = rb.reviews * rb.mean_text_length
    rb["photos"] = rb.reviews * rb.mean_photos

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5),
                             gridspec_kw={"width_ratios": [1, 1, 1.35]})

    ax = axes[0]
    bars = ax.bar(rb.stars, rb.mean_text_length, color=RATING_COLORS, width=0.72)
    for b, v in zip(bars, rb.mean_text_length):
        ax.text(b.get_x() + b.get_width() / 2, v + 9, f"{v:.0f}", ha="center",
                fontsize=10, color=INK, fontweight="bold")
    ax.set_title("Mean review length", pad=10)
    ax.set_xlabel("Star rating"); ax.set_ylabel("Characters")
    ax.set_ylim(0, 470); ax.set_xticks([1, 2, 3, 4, 5])
    clean_axis(ax)
    ax.annotate("highest at 2 stars", xy=(2, 406), xytext=(3.2, 452), fontsize=9.5,
                color=MUTED, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))

    ax = axes[1]
    bars = ax.bar(rb.stars, rb.mean_photos, color=RATING_COLORS, width=0.72)
    for b, v in zip(bars, rb.mean_photos):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.022, f"{v:.2f}", ha="center",
                fontsize=10, color=INK, fontweight="bold")
    ax.set_title("Mean attached photos", pad=10)
    ax.set_xlabel("Star rating"); ax.set_ylabel("Photos per review")
    ax.set_ylim(0, 1.18); ax.set_xticks([1, 2, 3, 4, 5])
    clean_axis(ax)
    ax.annotate("highest at 4 stars", xy=(4, 0.964), xytext=(2.5, 1.10), fontsize=9.5,
                color=MUTED, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))

    ax = axes[2]
    rows = [("Attached photos", rb.photos / rb.photos.sum()),
            ("Text characters", rb.chars / rb.chars.sum()),
            ("Review rows", rb.reviews / rb.reviews.sum())]
    ypos = np.arange(len(rows))
    for y, (_, shares) in zip(ypos, rows):
        left = 0.0
        for star, share in zip(rb.stars, shares):
            ax.barh(y, share, left=left, color=RATING_COLORS[star - 1], height=0.62)
            if share > 0.055:
                ax.text(left + share / 2, y, f"{share*100:.0f}%", ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")
            left += share
    ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    ax.set_xlim(0, 1); ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("Share of the 5,531,203 rated reviews")
    ax.set_title("Share of the rated corpus", pad=10)
    clean_axis(ax, grid_axis="x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=RATING_COLORS[i]) for i in range(5)]
    ax.legend(handles, ["1 star", "2", "3", "4", "5 stars"], ncol=5, fontsize=9.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.20), handlelength=1.1,
              columnspacing=1.0)

    title(fig, "Review evidence by star rating",
          "Mean length and photo count per review, and each rating's share of the "
          "5,531,203 rated reviews.")
    save(fig, OUT / "eda_evidence_mass.png")


def catalogue_sparsity():
    dq = pd.read_csv(IN / "analysis/expanded_eda/tables/data_quality.csv").set_index("measure")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4),
                            gridspec_kw={"width_ratios": [1.45, 1]})

    ax = axes[0]
    items = [("Google price level", dq.loc["missing_price_level", "share"]),
             ("Google rating", dq.loc["missing_places_rating", "share"]),
             ("Supported cuisine label", 6996 / N_RESTAURANTS),
             ("Any collected review", dq.loc["zero_collected_reviews", "share"]),
             ("Coordinates", dq.loc["missing_coordinates", "share"]),
             ("Street address", dq.loc["missing_address", "share"])]
    labels = [i[0] for i in items]
    vals = np.array([i[1] for i in items])
    y = np.arange(len(items))[::-1]
    colors = [CORAL if v > 0.3 else (GOLD if v > 0.02 else TEAL) for v in vals]
    ax.barh(y, vals * 100, color=colors, height=0.62)
    for yy, v in zip(y, vals):
        ax.text(v * 100 + 1.1, yy, f"{v*100:.1f}%", va="center", fontsize=10.5,
                color=INK, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 72); ax.set_xlabel("Share of 75,203 catalogue restaurants with the field missing")
    ax.set_title("Fields missing, by share of catalogue", pad=10)
    clean_axis(ax, grid_axis="x")

    ax = axes[1]
    stages = ["Google type\n= `Other`", "Supported name\nafter repair", "Unknown:\ninsufficient evidence"]
    vals2 = [15747, 8751, 6996]
    cols = [MUTED, TEAL, CORAL]
    bars = ax.bar(stages, vals2, color=cols, width=0.62)
    for b, v in zip(bars, vals2):
        ax.text(b.get_x() + b.get_width() / 2, v + 380, f"{v:,}", ha="center",
                fontsize=11, color=INK, fontweight="bold")
    ax.set_ylim(0, 18800)
    ax.set_ylabel("Restaurants")
    ax.set_title("Cuisine label resolution", pad=10)
    ax.tick_params(axis="x", labelsize=10)
    clean_axis(ax)
    title(fig, "Catalogue field completeness and cuisine label resolution",
          "Price level is absent for 59.2% of restaurants and an aggregate rating for 43.6%; "
          "6,996 restaurants retain no supported cuisine label.")
    save(fig, OUT / "eda_catalogue_sparsity.png")


# --------------------------------------------------------------------------- Part 2
def attribute_coverage():
    g = pd.read_csv(IN / "analysis/expanded_eda/llm_feature_evidence_spot_audit.csv")
    pretty = {"offers_vegan": "Vegan offering", "offers_vegetarian": "Vegetarian offering",
              "outdoor": "Outdoor seating", "spice": "Spice level", "value": "Value",
              "service_speed": "Service speed", "service_sentiment": "Service sentiment",
              "has_bar": "Bar on site", "byob": "BYOB"}
    g["label"] = g.feature.map(pretty)
    g["share"] = g.known_after / N_RESTAURANTS * 100
    g = g.sort_values("share")

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    y = np.arange(len(g))
    ax.barh(y, g.share, color=NAVY, height=0.62)
    for yy, row in zip(y, g.itertuples()):
        ax.text(row.share + 1.0, yy, f"{row.share:.1f}%   ({row.known_after:,})",
                va="center", fontsize=10, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(g.label, fontsize=11)
    ax.set_xlim(0, 88)
    ax.set_xlabel("Share of 75,203 restaurants carrying a value (%)")
    ax.set_title("Restaurants carrying a value for the attribute", pad=10)
    clean_axis(ax, grid_axis="x")

    ax = axes[1]
    ev = g[g.evidence_share_high.notna()].sort_values("evidence_share_known")
    ypos = np.arange(len(ev))
    ax.barh(ypos, ev.evidence_share_high * 100, height=0.30, color=TEAL,
            label="High-confidence assertions")
    ax.barh(ypos - 0.34, ev.evidence_share_low * 100, height=0.30, color=GOLD,
            label="Low-confidence assertions")
    for yy, row in zip(ypos, ev.itertuples()):
        ax.text(row.evidence_share_high * 100 + 1.5, yy, f"{row.evidence_share_high*100:.0f}%",
                va="center", fontsize=9.5, color=INK)
        ax.text(row.evidence_share_low * 100 + 1.5, yy - 0.34, f"{row.evidence_share_low*100:.0f}%",
                va="center", fontsize=9.5, color=INK)
    ax.set_yticks(ypos - 0.17)
    ax.set_yticklabels(ev.label, fontsize=10.5)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Assertions backed by direct lexical evidence (%)")
    ax.set_title("Evidence backing by asserted confidence", pad=10)
    clean_axis(ax, grid_axis="x")
    ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2)

    title(fig, "Structured attribute coverage and evidence backing",
          "A value is stored only where the source text or place types carry a direct mention, "
          "so coverage ranges from 62.5% of the catalogue to 0.6%.")
    save(fig, OUT / "feature_attribute_coverage.png")


def confidence_coverage():
    r = pd.read_csv(IN / "analysis/expanded_eda/tables/restaurant_llm_confidence_audit.csv")
    u = pd.read_csv(IN / "analysis/expanded_eda/tables/user_llm_confidence_audit.csv")
    m = r.merge(u, on="feature", suffixes=("_r", "_u"))
    m["label"] = m.feature.str.replace("_confidence", "").str.capitalize()
    m = m.sort_values("mean_r")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4))
    x = np.arange(len(m)); w = 0.36

    ax = axes[0]
    ax.bar(x - w / 2, m.mean_r, w, color=NAVY, label="Restaurant side (75,203 rows)")
    ax.bar(x + w / 2, m.mean_u, w, color=TEAL, label="User side (156,261 rows)")
    for xi, (vr, vu) in enumerate(zip(m.mean_r, m.mean_u)):
        ax.text(xi - w / 2, vr + 0.015, f"{vr:.2f}", ha="center", fontsize=9.5, color=INK)
        ax.text(xi + w / 2, vu + 0.015, f"{vu:.2f}", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(m.label, fontsize=10.5)
    ax.set_ylim(0, 1.06); ax.set_ylabel("Mean asserted confidence")
    ax.set_title("Mean asserted confidence", pad=10)
    clean_axis(ax)
    ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.annotate("lowest dimension:\nuser dietary preference", xy=(0 + w / 2, 0.665),
                xytext=(1.75, 0.42), fontsize=9.5, color=MUTED, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))

    ax = axes[1]
    ax.bar(x - w / 2, m.one_share_r * 100, w, color=NAVY, label="Restaurant side")
    ax.bar(x + w / 2, m.one_share_u * 100, w, color=TEAL, label="User side")
    ax.set_xticks(x); ax.set_xticklabels(m.label, fontsize=10.5)
    for xi, (vr, vu) in enumerate(zip(m.one_share_r, m.one_share_u)):
        ax.text(xi - w / 2, vr * 100 + 1.6, f"{vr*100:.0f}", ha="center", fontsize=9.5, color=INK)
        ax.text(xi + w / 2, vu * 100 + 1.6, f"{vu*100:.0f}", ha="center", fontsize=9.5, color=INK)
    ax.set_ylim(0, 100); ax.set_ylabel("Share of rows at confidence 1.0 (%)")
    ax.set_title("Rows at confidence 1.0", pad=10)
    clean_axis(ax)
    ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    title(fig, "Extraction confidence by semantic dimension",
          "Mean asserted confidence and the share of rows asserted at full confidence, "
          "restaurant side against user side.")
    save(fig, OUT / "feature_confidence_coverage.png")


# --------------------------------------------------------------------------- Part 3
def uplift_decomposition():
    s = pd.read_csv(IN / "results/publication_core_summary.csv").set_index(["model", "condition"])
    h = lambda m, c: s.loc[(m, c), "test_hit_at_10_mean"]
    prox = 0.048032885151552

    items = [
        ("LLM features on Two-Tower", h("FeatureTwoTower", "full_llm") - h("FeatureTwoTower", "non_llm"), NAVY),
        ("LLM features on LightGCN", h("LightGCN", "full_llm") - h("LightGCN", "non_llm"), NAVY),
        ("Architecture: LightGCN → KGAT-SAL", h("KGAT-SAL", "non_llm") - h("LightGCN", "non_llm"), MUTED),
        ("LLM features on KGAT-SAL", h("KGAT-SAL", "full_llm") - h("KGAT-SAL", "non_llm"), NAVY),
        ("Architecture: Two-Tower → LightGCN", h("LightGCN", "non_llm") - h("FeatureTwoTower", "non_llm"), MUTED),
        ("Proximity rerank on KGAT-SAL + LLM", prox - h("KGAT-SAL", "full_llm"), CORAL),
    ]
    items.sort(key=lambda t: t[1])
    ref = h("KGAT-SAL", "full_llm") - h("KGAT-SAL", "non_llm")

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.7),
                            gridspec_kw={"width_ratios": [1.6, 1]})

    ax = axes[0]
    y = np.arange(len(items))
    ax.barh(y, [i[1] for i in items], color=[i[2] for i in items], height=0.6)
    for yy, (_, v, _) in zip(y, items):
        mult = v / ref
        mtxt = f"{mult:.2f}×" if mult < 1 else f"{mult:.1f}×"
        ax.text(v + 0.00035, yy, f"+{v:.5f}   ({mtxt})", va="center",
                fontsize=10, color=INK, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=10.5)
    ax.set_xlim(0, 0.0215)
    ax.set_xlabel("Change in full-catalog test Hit@10 (mean of seeds 42, 43, 44)")
    ax.set_title("Change in test Hit@10", pad=10)
    clean_axis(ax, grid_axis="x")
    ax.text(0.0, -1.28, "multiples are relative to the KGAT-SAL LLM feature bundle",
            fontsize=9, color=MUTED, style="italic")

    ax = axes[1]
    levels = [("Uniform random", 10 / N_CANDIDATES, GRID),
              ("Two-Tower, conventional", h("FeatureTwoTower", "non_llm"), MUTED),
              ("KGAT-SAL, conventional", h("KGAT-SAL", "non_llm"), TEAL),
              ("KGAT-SAL + LLM", h("KGAT-SAL", "full_llm"), NAVY),
              ("+ proximity rerank", prox, CORAL)]
    y2 = np.arange(len(levels))[::-1]
    ax.barh(y2, [l[1] for l in levels], color=[l[2] for l in levels], height=0.6)
    for yy, (_, v, _) in zip(y2, levels):
        ax.text(v * 1.35, yy, f"{v:.5f}  ({v/(10/N_CANDIDATES):.0f}×)", va="center",
                fontsize=9.5, color=INK)
    ax.set_xscale("log")
    ax.set_yticks(y2); ax.set_yticklabels([l[0] for l in levels], fontsize=10.5)
    ax.set_xlim(1e-4, 0.55)
    ax.set_xlabel("Test Hit@10, log scale")
    ax.set_title("Absolute level, log scale", pad=10)
    clean_axis(ax, grid_axis="x")

    title(fig, "Change in test Hit@10 by intervention",
          "Absolute change contributed by each intervention, and the resulting levels against "
          "a uniform-random floor of 0.00016.")
    save(fig, OUT / "publication_uplift_decomposition.png")


def proximity_sensitivity():
    grid = pd.read_csv(IN / "results/publication_proximity_validation_grid.csv")
    gm = grid.groupby(["alpha", "bandwidth_km", "candidate_depth"])[["hit_at_10", "ndcg_at_10"]].mean().reset_index()
    no_rerank = 0.0218415  # KGAT-SAL + LLM validation NDCG@10, publication_core_summary.csv

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0),
                            gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    d100 = gm[gm.candidate_depth == 100]
    piv = d100.pivot(index="alpha", columns="bandwidth_km", values="ndcg_at_10").sort_index(ascending=False)
    im = ax.imshow(piv.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:.0f} km" for c in piv.columns], fontsize=10.5)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{a:.1f}" for a in piv.index], fontsize=10.5)
    ax.set_xlabel("Distance bandwidth"); ax.set_ylabel("Blend weight α")
    ax.set_title("Mean validation NDCG@10 at candidate depth 100", pad=10)
    vmax = piv.values.max()
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.5f}", ha="center", va="center", fontsize=9.5,
                    color="white" if v > vmax * 0.93 else INK)
    bi, bj = np.unravel_index(np.argmax(piv.values), piv.shape)
    ax.add_patch(Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor=CORAL, lw=3))
    ax.text(-0.80, bi, "frozen ▸", ha="right", va="center", fontsize=10.5, color=CORAL,
            fontweight="bold", clip_on=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)

    ax = axes[1]
    for depth, col, mk in [(25, GOLD, "o"), (50, TEAL, "s"), (100, NAVY, "D")]:
        sub = gm[(gm.bandwidth_km == 5.0) & (gm.candidate_depth == depth)].sort_values("alpha")
        ax.plot(sub.alpha, sub.ndcg_at_10, marker=mk, color=col, lw=2.2, ms=6,
                label=f"candidate depth {depth}")
    ax.axhline(no_rerank, color=CORAL, ls="--", lw=1.8)
    ax.text(0.86, no_rerank - 0.00060, "no reranking: KGAT-SAL + LLM validation NDCG@10",
            fontsize=9.5, color=CORAL, ha="right", va="top")
    ax.set_xlabel("Blend weight α  (bandwidth 5 km)")
    ax.set_ylabel("Mean validation NDCG@10")
    ax.set_title("Validation NDCG@10 against blend weight", pad=10)
    ax.set_xlim(0.05, 0.90); ax.set_ylim(0.0203, 0.0348)
    clean_axis(ax); ax.legend(fontsize=9.5, loc="upper left")
    ax.annotate("α = 0.7 is the largest value searched",
                xy=(0.705, 0.031526), xytext=(0.52, 0.0268), fontsize=9.5, color=MUTED,
                ha="center", arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))

    title(fig, "Validation sensitivity of the proximity reranker",
          "All 84 configurations exceed the unreranked model; the selected α = 0.7, 5 km, "
          "depth 100 lies at a boundary of the searched range on all three axes.")
    save(fig, OUT / "publication_proximity_sensitivity.png")


def hit_by_star():
    hr = pd.read_csv(IN / "results/publication_hit_rate_by_rating.csv")
    m = hr.groupby(["stage", "test_rating"]).agg(
        targets=("targets", "first"), hit=("hit_at_10", "mean"), sd=("hit_at_10", "std")).reset_index()
    raw = m[m.stage == "Raw KGAT-SAL + LLM"].sort_values("test_rating")
    prox = m[m.stage == "+ proximity reranking"].sort_values("test_rating")

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.7),
                            gridspec_kw={"width_ratios": [1.45, 1]})

    ax = axes[0]
    x = np.arange(5); w = 0.36
    ax.bar(x - w / 2, raw.hit, w, yerr=raw.sd, color=NAVY, capsize=3,
           error_kw=dict(ecolor=MUTED, lw=1.2), label="KGAT-SAL + LLM")
    ax.bar(x + w / 2, prox.hit, w, yerr=prox.sd, color=CORAL, capsize=3,
           error_kw=dict(ecolor=MUTED, lw=1.2), label="+ proximity rerank")
    for xi, v in zip(x, prox.hit):
        ax.text(xi + w / 2, v + 0.0028, f"{v:.5f}", ha="center", fontsize=9, color=INK)
    for xi, v in zip(x, raw.hit):
        ax.text(xi - w / 2, v + 0.0028, f"{v:.5f}", ha="center", fontsize=9, color=MUTED)
    band = prox[prox.test_rating <= 4].hit.mean()
    ax.hlines(band, -0.5, 3.5, color=CORAL, ls=":", lw=1.8)
    ax.text(1.5, band + 0.0060, "1–4 star mean 0.05217  (range 3.4%)",
            fontsize=9.5, color=MUTED, ha="center")
    ax.annotate("5-star band: 12.7% below the 1–4 star mean,\n62.1% of test targets",
                xy=(4 + w / 2, prox.hit.iloc[4] - 0.0015), xytext=(2.85, 0.0195), fontsize=9.5,
                color=MUTED, ha="center", arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{t:,} targets" for s, t in zip([1, 2, 3, 4, 5], prox.targets)],
                       fontsize=10)
    ax.set_xlabel("Star rating the diner later gave the held-out restaurant")
    ax.set_ylabel("Hit@10"); ax.set_ylim(0, 0.078)
    ax.set_title("Hit@10 by star rating", pad=10)
    clean_axis(ax); ax.legend(fontsize=10, loc="upper center", ncol=2)

    ax = axes[1]
    lift = prox.hit.values / raw.hit.values
    bars = ax.bar(x, lift, color=TEAL, width=0.62)
    for b, v in zip(bars, lift):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}×", ha="center",
                fontsize=10.5, color=INK, fontweight="bold")
    ax.axhline(1.0, color=MUTED, lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels([1, 2, 3, 4, 5])
    ax.set_xlabel("Star rating"); ax.set_ylabel("Hit@10 multiple from reranking")
    ax.set_ylim(0, 1.92)
    ax.set_title("Reranking multiple", pad=10)
    clean_axis(ax)

    title(fig, "Hit@10 by held-out star rating",
          "Recovery rate before and after proximity reranking, with the reranking multiple "
          "for each rating band.")
    save(fig, OUT / "publication_hit_by_star.png")


# --------------------------------------------------------------------------- Part 4
def graphrag_quality():
    gr = pd.read_parquet(IN / "results/graphrag/publication_graphrag_results.parquet")
    dims = [("entailment", "Entailment"), ("personalization", "Personalization"),
            ("usefulness", "Usefulness"), ("citation_correctness", "Citation correctness")]
    palette = ["#C65D57", "#D98A62", "#D8B65C", "#75A879", "#2A9D8F"]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9),
                            gridspec_kw={"width_ratios": [1.55, 1]})

    ax = axes[0]
    y = np.arange(len(dims))[::-1]
    for yy, (col, lab) in zip(y, dims):
        counts = gr[col].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        left = 0
        for score, c in zip([1, 2, 3, 4, 5], counts):
            if c:
                ax.barh(yy, c, left=left, color=palette[score - 1], height=0.6)
                if c >= 7:
                    ax.text(left + c / 2, yy, str(c), ha="center", va="center",
                            fontsize=10, color="white", fontweight="bold")
            left += c
        ax.text(101.5, yy, f"mean {gr[col].mean():.2f}", va="center", fontsize=10.5,
                color=INK, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([d[1] for d in dims], fontsize=11)
    ax.set_ylim(-0.55, 4.25)
    ax.set_xlim(0, 118); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Explanations (of 100 audited cases)")
    ax.set_title("Score counts across 100 explanations", pad=10)
    clean_axis(ax, grid_axis="x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[i]) for i in range(5)]
    ax.legend(handles, ["score 1", "2", "3", "4", "5"], ncol=5, fontsize=9.5,
              loc="upper center", bbox_to_anchor=(0.43, -0.19), handlelength=1.1,
              columnspacing=1.0)
    ax.annotate("one case scores 3 on entailment",
                xy=(49.5, 3.33), xytext=(66, 3.95), fontsize=9.5, color=MUTED, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))

    ax = axes[1]
    a = gr[gr.match_available]; b = gr[~gr.match_available]
    x = np.arange(len(dims)); w = 0.36
    ax.bar(x - w / 2, [b[c].mean() for c, _ in dims], w, color=TEAL,
           label=f"No user match available (n={len(b)})")
    ax.bar(x + w / 2, [a[c].mean() for c, _ in dims], w, color=NAVY,
           label=f"User match available (n={len(a)})")
    for xi, (c, _) in enumerate(dims):
        ax.text(xi - w / 2, b[c].mean() + 0.09, f"{b[c].mean():.2f}", ha="center", fontsize=9.5)
        ax.text(xi + w / 2, a[c].mean() + 0.09, f"{a[c].mean():.2f}", ha="center", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Entail-\nment", "Personal-\nization", "Useful-\nness", "Citation\ncorrectness"],
                       fontsize=9.5)
    ax.set_ylim(0, 5.9); ax.set_yticks([1, 2, 3, 4, 5]); ax.set_ylabel("Mean judge score (1–5)")
    ax.set_title("Mean score by match availability", pad=10)
    clean_axis(ax); ax.legend(fontsize=9, loc="upper center", ncol=1)
    ax.text(1.5, -1.35, "unsupported claims per explanation:  0.10 without a match,  1.00 with one",
            fontsize=9.2, color=MUTED, ha="center", style="italic")

    title(fig, "Local-judge score distributions and evidence availability",
          "Score counts across the 100 audited explanations, and mean scores split by whether "
          "a supported user match was available.")
    save(fig, OUT / "graphrag_quality_distribution.png")


if __name__ == "__main__":
    evidence_mass()
    catalogue_sparsity()
    attribute_coverage()
    confidence_coverage()
    uplift_decomposition()
    proximity_sensitivity()
    hit_by_star()
    graphrag_quality()
    print("done ->", OUT)
