import pandas as pd

import build_llm_features_ollama as llm


def _reviews() -> pd.DataFrame:
    rows = [
        {
            "review_id": f"review-{i}", "place_id": f"place-{i}",
            "contributor_id": "same-user", "rating": 4,
            "timestamp_days_ago": 365, "text": f"Long review text for place {i}",
            "text_len": 28, "has_content": True, "reviewer_reviews": 4,
        }
        for i in range(5)
    ]
    rows.append({
        "review_id": "unrelated", "place_id": "other-place",
        "contributor_id": "other-user", "rating": 5,
        "timestamp_days_ago": 365, "text": "Another unrelated review text",
        "text_len": 29, "has_content": True, "reviewer_reviews": 1,
    })
    return pd.DataFrame(rows)


def test_split_is_invariant_to_row_order_and_unrelated_rows() -> None:
    reviews = _reviews()
    first = llm.assign_interaction_splits(reviews)
    shuffled = llm.assign_interaction_splits(reviews.sample(frac=1, random_state=18))
    columns = ["contributor_id", "place_id", "split"]
    left = first[columns].sort_values(columns).reset_index(drop=True)
    right = shuffled[columns].sort_values(columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    assert left["split"].value_counts().to_dict() == {
        "train": 3, "validation": 1, "test": 1,
    }


def test_corpora_are_invariant_to_row_order(tmp_path, monkeypatch) -> None:
    reviews = _reviews()
    restaurants = pd.DataFrame({
        "place_id": reviews["place_id"],
        "name": reviews["place_id"],
        "cuisine_category": "American & Comfort",
        "types": "restaurant",
    })
    outputs = []
    for seed in (1, 2):
        root = tmp_path / str(seed)
        root.mkdir()
        reviews.sample(frac=1, random_state=seed).to_parquet(root / "reviews.parquet", index=False)
        restaurants.to_parquet(root / "restaurants.parquet", index=False)
        monkeypatch.setattr(llm, "REVIEWS_FILE", root / "reviews.parquet")
        monkeypatch.setattr(llm, "RESTAURANTS_FILE", root / "restaurants.parquet")
        monkeypatch.setattr(llm, "RESTAURANT_CORPUS", root / "restaurant_corpus.parquet")
        monkeypatch.setattr(llm, "USER_CORPUS", root / "user_corpus.parquet")
        monkeypatch.setattr(llm, "SPLIT_MANIFEST", root / "split_manifest.json")
        llm.prepare_corpora()
        outputs.append((
            pd.read_parquet(root / "restaurant_corpus.parquet")[["place_id", "source_review_ids_json"]]
            .sort_values("place_id").reset_index(drop=True),
            pd.read_parquet(root / "user_corpus.parquet")[["contributor_id", "source_review_ids_json"]]
            .sort_values("contributor_id").reset_index(drop=True),
        ))
    pd.testing.assert_frame_equal(outputs[0][0], outputs[1][0])
    pd.testing.assert_frame_equal(outputs[0][1], outputs[1][1])
