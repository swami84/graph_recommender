#!/usr/bin/env python3
"""Create final publication tables and figures from frozen experiment outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from article.figures.publication_style import (
    apply_style, clean_axis, save, title, NAVY, TEAL, CORAL, GOLD, LIGHT,
)


RESULTS = Path("results")
FIGURES = Path("article/figures")
FINAL_CSV = RESULTS / "publication_final_model_comparison.csv"
FINAL_MD = RESULTS / "publication_final_results.md"

COLORS = {"non_llm": "#aeb7c3", "full_llm": TEAL, "proximity": CORAL}


def load_complete_graphrag_audit():
    """Return the frozen GraphRAG audit only after all 100 rows are available."""
    path = RESULTS / "graphrag" / "publication_graphrag_results.parquet"
    if not path.exists():
        return None
    audit = pd.read_parquet(path)
    if len(audit) != 100 or audit["sample_id"].nunique() != 100:
        return None
    return audit


def main():
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RESULTS / "publication_core_summary.csv")
    proximity = pd.read_csv(RESULTS / "publication_proximity_test_by_seed.csv")
    core = pd.read_csv(RESULTS / "publication_core_results.csv")
    core = core.sort_values("timestamp").drop_duplicates(
        ["model", "condition", "seed"], keep="last"
    )

    table = summary[[
        "model", "condition", "test_hit_at_10_mean", "test_hit_at_10_std",
        "test_ndcg_at_10_mean", "test_ndcg_at_10_std",
    ]].copy()
    table.to_csv(FINAL_CSV, index=False)

    models = ["FeatureTwoTower", "LightGCN", "KGAT-SAL"]
    labels = ["Two-Tower", "LightGCN", "KGAT-SAL"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(models))
    width = 0.36
    for axis, metric, ylabel in [
        (axes[0], "test_ndcg_at_10", "NDCG@10"),
        (axes[1], "test_hit_at_10", "Hit@10"),
    ]:
        for offset, condition, display in [
            (-width / 2, "non_llm", "Conventional features"),
            (width / 2, "full_llm", "+ LLM features"),
        ]:
            values, errors = [], []
            for model in models:
                row = summary[(summary.model == model) & (summary.condition == condition)].iloc[0]
                values.append(row[f"{metric}_mean"])
                errors.append(row[f"{metric}_std"])
            bars = axis.bar(
                x + offset, values, width, yerr=errors, capsize=3,
                label=display, color=COLORS[condition], edgecolor="white",
            )
            axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=4, fontsize=9)
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        clean_axis(axis)
        axis.set_ylim(0, max(axis.get_ylim()[1], max(values) * 1.28))
    axes[0].legend(frameon=False, loc="upper left")
    title(fig, "Full-catalog ranking performance by model and feature condition",
          "Full-catalogue test metrics; mean ± sample SD across three seeds")
    save(fig, FIGURES / "publication_model_llm_comparison.png")

    raw = core[(core.model == "KGAT-SAL") & (core.condition == "full_llm")]
    raw_hit = raw.test_hit_at_10.mean()
    raw_ndcg = raw.test_ndcg_at_10.mean()
    prox_hit = proximity.hit_at_10.mean()
    prox_ndcg = proximity.ndcg_at_10.mean()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for axis, values, panel_title in [
        (axes[0], [raw_ndcg, prox_ndcg], "NDCG@10"),
        (axes[1], [raw_hit, prox_hit], "Hit@10"),
    ]:
        bars = axis.bar(
            ["Raw KGAT-SAL", "+ proximity"], values,
            color=[COLORS["full_llm"], COLORS["proximity"]], width=0.62,
        )
        axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=4, fontsize=10)
        axis.set_title(panel_title, fontweight="bold")
        axis.set_ylim(0, max(values) * 1.22)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        clean_axis(axis)
    title(fig, "Effect of validation-tuned proximity reranking",
          "The geographic stage only reorders the model's top 100 candidates")
    save(fig, FIGURES / "publication_proximity_uplift.png")

    rating_path = RESULTS / "publication_rating_aware_summary.csv"
    if rating_path.exists():
        rating = pd.read_csv(rating_path)
        stages = ["Raw KGAT-SAL + LLM", "+ proximity reranking"]
        rating = rating.set_index("stage").reindex(stages)
        labels = ["Raw KGAT-SAL\n+ LLM", "+ proximity\nreranking"]
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"wspace": 0.30})
        x = np.arange(2)
        bottom = np.zeros(2)
        components = [
            ("positive_hit_at_10_mean", "Liked (4–5 stars)", TEAL),
            ("neutral_hit_at_10_mean", "Neutral (3 stars)", GOLD),
            ("disliked_hit_at_10_mean", "Disliked (1–2 stars)", CORAL),
        ]
        for column, label, color in components:
            values = 100 * rating[column].to_numpy()
            bars = axes[0].bar(x, values, bottom=bottom, color=color, width=0.62, label=label)
            for index, value in enumerate(values):
                if value >= 0.25:
                    axes[0].text(index, bottom[index] + value / 2, f"{value:.2f}%",
                                 ha="center", va="center", fontsize=9, color="white")
            bottom += values
        axes[0].bar_label(bars, labels=[f"total {value:.2f}%" for value in bottom], padding=5)
        axes[0].set_xticks(x, labels)
        axes[0].set_ylabel("Share of all test users recovered in top 10")
        axes[0].set_title("Hit@10 includes liked and disliked visits")
        axes[0].set_ylim(0, bottom.max() * 1.24)
        axes[0].legend(loc="upper left")
        clean_axis(axes[0])

        baseline = rating.iloc[0]
        groups = ["All test\ntargets", "Raw hits", "Proximity hits"]
        liked = 100 * np.array([
            baseline.liked_share_all_targets_mean,
            rating.iloc[0].liked_share_among_hits_mean,
            rating.iloc[1].liked_share_among_hits_mean,
        ])
        disliked = 100 * np.array([
            baseline.disliked_share_all_targets_mean,
            rating.iloc[0].disliked_share_among_hits_mean,
            rating.iloc[1].disliked_share_among_hits_mean,
        ])
        neutral = 100 - liked - disliked
        axes[1].bar(groups, liked, color=TEAL, width=0.62, label="Liked")
        axes[1].bar(groups, neutral, bottom=liked, color=GOLD, width=0.62, label="Neutral")
        axes[1].bar(groups, disliked, bottom=liked + neutral, color=CORAL, width=0.62, label="Disliked")
        for index in range(3):
            axes[1].text(index, liked[index] / 2, f"{liked[index]:.1f}% liked",
                         ha="center", va="center", color="white", fontsize=9)
            axes[1].text(index, 100 - disliked[index] / 2, f"{disliked[index]:.1f}%",
                         ha="center", va="center", color="white", fontsize=8)
        axes[1].set_ylim(0, 100)
        axes[1].set_ylabel("Outcome composition (%)")
        axes[1].set_title("Outcome composition relative to the test population")
        clean_axis(axes[1])
        title(fig, "Rating-aware composition of retrieved targets",
              "Liked = 4–5 stars; disliked = 1–2 stars on each user's held-out visit")
        save(fig, FIGURES / "publication_rating_aware_hits.png")

    graphrag = load_complete_graphrag_audit()
    if graphrag is not None:
        judged_columns = ["entailment", "personalization", "usefulness", "citation_correctness"]
        judged_labels = ["Entailment", "Personalization", "Usefulness", "Citation\ncorrectness"]
        deterministic_columns = [
            "citations_valid", "restaurant_evidence_cited",
            "match_cited_when_available", "quote_fidelity", "heldout_safe",
        ]
        deterministic_labels = [
            "Valid\ncitations", "Restaurant\nevidence", "Match cited\nwhen available",
            "Quote\nfidelity", "Held-out\nsafety",
        ]
        judged_values = [graphrag[column].mean() for column in judged_columns]
        deterministic_values = [100 * graphrag[column].mean() for column in deterministic_columns]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        bars = axes[0].bar(judged_labels, judged_values, color=COLORS["full_llm"], width=0.68)
        axes[0].bar_label(bars, labels=[f"{value:.2f}" for value in judged_values], padding=4)
        axes[0].set_ylabel("Local-LLM judge score (1–5)")
        axes[0].set_ylim(0, 5.5)
        bars = axes[1].bar(deterministic_labels, deterministic_values, color="#457b9d", width=0.68)
        axes[1].bar_label(bars, labels=[f"{value:.0f}%" for value in deterministic_values], padding=4)
        axes[1].set_ylabel("Explanations passing check (%)")
        axes[1].set_ylim(0, 112)
        for axis in axes:
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
        clean_axis(axes[0])
        clean_axis(axes[1])
        title(fig, "GraphRAG explanations remain traceable to training-safe evidence",
              "Local-LLM quality judgments (left) and deterministic provenance checks (right)")
        save(fig, FIGURES / "publication_graphrag_audit.png")

    display = table.copy()
    display["features"] = display.condition.map({
        "non_llm": "Conventional", "full_llm": "Conventional + LLM"
    })
    display["Hit@10"] = display.apply(
        lambda row: f"{row.test_hit_at_10_mean:.5f} ± {row.test_hit_at_10_std:.5f}", axis=1
    )
    display["NDCG@10"] = display.apply(
        lambda row: f"{row.test_ndcg_at_10_mean:.5f} ± {row.test_ndcg_at_10_std:.5f}", axis=1
    )
    display = display[["model", "features", "Hit@10", "NDCG@10"]]
    kg_non = table[(table.model == "KGAT-SAL") & (table.condition == "non_llm")].iloc[0]
    kg_full = table[(table.model == "KGAT-SAL") & (table.condition == "full_llm")].iloc[0]
    lines = [
        "# Final publication recommendation results", "",
        "All raw metrics are full-catalog test results over 61,204 interacted restaurants. "
        "Values are mean ± sample SD across seeds 42, 43, and 44. Model and feature-set "
        "selection used mean validation NDCG@10.", "", "## Core comparison", "",
        display.to_markdown(index=False), "", "## Main findings", "",
        f"- LLM features improve KGAT-SAL NDCG@10 by "
        f"{(kg_full.test_ndcg_at_10_mean / kg_non.test_ndcg_at_10_mean - 1):.1%} and "
        f"Hit@10 by {(kg_full.test_hit_at_10_mean / kg_non.test_hit_at_10_mean - 1):.1%}.",
        f"- Validation-selected KGAT-SAL + LLM achieves raw NDCG@10 "
        f"{raw_ndcg:.5f} and Hit@10 {raw_hit:.5f}.",
        f"- Frozen proximity reranking achieves NDCG@10 {prox_ndcg:.5f} and "
        f"Hit@10 {prox_hit:.5f}, gains of {(prox_ndcg / raw_ndcg - 1):.1%} and "
        f"{(prox_hit / raw_hit - 1):.1%}, respectively.", "",
        "Proximity hyperparameters were selected on validation only; test was evaluated "
        "once after freezing alpha=0.7, bandwidth=5 km, and candidate depth=100.", "",
    ]
    if rating_path.exists():
        prox_rating = rating.iloc[1]
        lines.extend([
            "## Rating-aware interpretation of hits", "",
            "Hit@10 is an implicit-feedback retrieval metric: it counts a held-out visit even when "
            "the reviewer gave the restaurant a low rating. With ratings 4–5 treated as liked and "
            "1–2 as disliked:", "",
            f"- Proximity positive Hit@10 is {prox_rating.positive_hit_at_10_mean:.5f}; "
            f"disliked-target Hit@10 is {prox_rating.disliked_hit_at_10_mean:.5f}.",
            f"- {prox_rating.liked_share_among_hits_mean:.1%} of recovered proximity targets were "
            f"liked and {prox_rating.disliked_share_among_hits_mean:.1%} were disliked.",
            f"- The corresponding full test-target shares are "
            f"{prox_rating.liked_share_all_targets_mean:.1%} and "
            f"{prox_rating.disliked_share_all_targets_mean:.1%}; therefore the current objective "
            "does not preferentially retrieve positive experiences.",
            f"- Rating-signed discounted utility@10 is "
            f"{prox_rating.signed_utility_at_10_mean:.5f}.", "",
            "This motivates a future rating-aware objective or a relevance threshold, while the "
            "implicit-feedback metrics remain the primary preregistered comparison.", "",
        ])
    if graphrag is not None:
        match_coverage = graphrag.match_available.mean()
        match_citation = graphrag.loc[
            graphrag.match_available, "match_cited_when_available"
        ].mean()
        lines.extend([
            "## GraphRAG explanation audit", "",
            "The ranker is unchanged. A local 27B LLM generated and independently judged 100 "
            "activity-stratified explanations from compact, training-safe evidence subgraphs.", "",
            f"- Entailment: {graphrag.entailment.mean():.2f}/5; personalization: "
            f"{graphrag.personalization.mean():.2f}/5; usefulness: "
            f"{graphrag.usefulness.mean():.2f}/5; citation correctness: "
            f"{graphrag.citation_correctness.mean():.2f}/5.",
            f"- Unsupported claims: {graphrag.unsupported_claims.mean():.2f} per explanation.",
            f"- Valid inline citations: {graphrag.citations_valid.mean():.1%}; exact quote "
            f"fidelity: {graphrag.quote_fidelity.mean():.1%}; held-out-safe bundles: "
            f"{graphrag.heldout_safe.mean():.1%}.",
            f"- Supported personalization evidence was available for {match_coverage:.1%} of "
            f"recommendations and cited in {match_citation:.1%} of those explanations.", "",
            "Judge scores are diagnostic local-LLM assessments; deterministic provenance checks "
            "and a planned blinded human audit provide separate safeguards.", "",
        ])
    FINAL_MD.write_text("\n".join(lines))
    figure_count = 3 if graphrag is not None else 2
    if rating_path.exists():
        figure_count += 1
    print(f"Wrote {FINAL_CSV}, {FINAL_MD}, and {figure_count} figures under {FIGURES}")


if __name__ == "__main__":
    main()
