import pandas as pd

import build_llm_features_ollama as llm


def test_model_and_feature_split_helper_is_row_order_invariant() -> None:
    rows = pd.DataFrame([
        {"review_id": f"r{i}", "place_id": f"p{i}", "contributor_id": "u",
         "rating": 4, "timestamp_days_ago": 365, "text": f"review {i}"}
        for i in range(5)
    ])
    a = llm.assign_interaction_splits(rows)[["place_id", "split"]].sort_values("place_id")
    b = llm.assign_interaction_splits(rows.sample(frac=1, random_state=7))[
        ["place_id", "split"]
    ].sort_values("place_id")
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
