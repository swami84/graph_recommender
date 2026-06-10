"""Build a corrected restaurants_enriched where cuisine_category is the LLM-inferred cuisine
(from restaurant_llm_features.parquet's llm_cuisine_* one-hots) instead of Google's coarse
types mapping. Both the item cuisine one-hot AND KGAT's knowledge-graph cuisine triples read
this column, so this is the single source to correct.
Output: data/restaurants_enriched_llmcz.parquet   (does NOT touch the original)
Run: python build_llmcz_restaurants.py"""
import re
import pandas as pd

CUISINE_CATS = ["American & Comfort", "Asian (Other)", "BBQ & Steakhouse", "Bar & Pub",
    "Café & Bakery", "Chinese", "Fast Food & Burgers", "Fine Dining",
    "Indian & South Asian", "Italian & Pizza", "Japanese & Sushi",
    "Mediterranean & Middle Eastern", "Mexican & Latin", "Other",
    "Sandwiches & Deli", "Seafood", "Soul Food"]
KEY2CAT = {re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_"): c for c in CUISINE_CATS}

rest = pd.read_parquet("data/restaurants_enriched.parquet")
llm = pd.read_parquet("data/restaurant_llm_features.parquet")
cz_cols = [c for c in llm.columns if c.startswith("llm_cuisine_")]
# argmax one-hot -> key -> display name
keys = [c.replace("llm_cuisine_", "") for c in cz_cols]
import numpy as np
idx = llm[cz_cols].values.argmax(1)
llm_label = pd.Series([KEY2CAT.get(keys[i], "Other") for i in idx], index=llm.index)
llm_map = dict(zip(llm["place_id"], llm_label))

orig = rest["cuisine_category"].copy()
rest["cuisine_category"] = rest["place_id"].map(llm_map).fillna(rest["cuisine_category"])

n_changed = (orig.values != rest["cuisine_category"].values).sum()
print(f"restaurants: {len(rest):,} | cuisine labels changed: {n_changed:,} ({n_changed/len(rest)*100:.1f}%)")
print("Other share — before:", round((orig == 'Other').mean()*100, 1), "%  after:",
      round((rest['cuisine_category'] == 'Other').mean()*100, 1), "%")
out = "data/restaurants_enriched_llmcz.parquet"
rest.to_parquet(out, index=False)
print("wrote", out)
