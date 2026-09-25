import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ENTRY_MODULES = [
    "foodie.collection.hexagon_places",
    "foodie.collection.run_places_then_reviews",
    "foodie.collection.run_review_collection_guarded",
    "foodie.collection.scrape_reviews_batch",
    "foodie.features.build_llm_features_ollama",
    "foodie.features.build_training_features",
    "foodie.features.run_incremental_feature_batch",
    "foodie.modeling.recommendation_gnn",
    "foodie.modeling.recommendation_kgat_sal",
    "foodie.modeling.recommendation_two_tower",
    "foodie.modeling.run_expanded_model_experiments",
    "foodie.modeling.run_publication_proximity",
    "foodie.explanations.run_publication_graphrag",
]


def test_active_entry_modules_are_discoverable():
    missing = [name for name in ENTRY_MODULES if importlib.util.find_spec(name) is None]
    assert not missing


def test_project_root_has_no_python_scripts():
    assert list(ROOT.glob("*.py")) == []


def test_systemd_units_use_package_modules():
    for unit in (ROOT / "systemd").glob("foodie-*.service"):
        text = unit.read_text()
        python_lines = [line for line in text.splitlines() if line.startswith("ExecStart=") and "python" in line]
        assert all(" -m foodie." in line for line in python_lines), unit
