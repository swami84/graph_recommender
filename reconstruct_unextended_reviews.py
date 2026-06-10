"""Reconstruct the UNEXTENDED interaction set from graph.pkl (the Apr-17 unextended graph),
in the reviews_flat schema the training pipeline needs. The original unextended reviews_flat was
overwritten by the May extended rebuild; graph.pkl's 10.0M REVIEWED edges are the surviving source.
Output: data/reviews_flat_unextended.parquet   (does NOT touch reviews_flat.parquet)
Run: python reconstruct_unextended_reviews.py"""
import pickle, time
import pandas as pd

t = time.time()
print("loading graph.pkl …")
G = pickle.load(open("data/graph.pkl", "rb"))
print(f"  loaded in {time.time()-t:.0f}s")

# reviewer + restaurant node attribute lookups
rev_attr, place_of = {}, {}
for n, d in G.nodes(data=True):
    nt = d.get("ntype")
    if nt == "reviewer":
        rev_attr[n] = (d.get("contributor_id"), d.get("is_local_guide"),
                       d.get("total_reviews"), d.get("predicted_race"), d.get("predicted_gender"))
    elif nt == "restaurant":
        place_of[n] = d.get("place_id")
print(f"  reviewers={len(rev_attr):,}  restaurants={len(place_of):,}")

rows = []
t = time.time()
for u, v, d in G.edges(data=True):
    if d.get("etype") != "REVIEWED":
        continue
    ra = rev_attr.get(u)
    if ra is None:
        continue
    cid, lg, nrev, prace, pgen = ra
    rows.append((cid, place_of.get(v), d.get("rating"), d.get("timestamp_days_ago"),
                 d.get("has_content"), d.get("meal_type"), d.get("price_per_person"),
                 d.get("food_score"), d.get("service_score"), d.get("atmosphere_score"),
                 d.get("review_id"), lg, nrev, prace, pgen))
print(f"  collected {len(rows):,} REVIEWED edges in {time.time()-t:.0f}s")

df = pd.DataFrame(rows, columns=[
    "contributor_id", "place_id", "rating", "timestamp_days_ago", "has_content",
    "meal_type", "price_per_person", "food_score", "service_score", "atmosphere_score",
    "review_id", "is_local_guide", "reviewer_reviews", "predicted_race", "predicted_gender"])
# score columns are strings like "5" in the source — keep as string (loader extracts digits)
for c in ["food_score", "service_score", "atmosphere_score"]:
    df[c] = df[c].astype("string")
df["reviewer_photos"] = 0      # not stored on edges; unused by training
df["text_len"] = 0             # text-derived features come from separate parquets

out = "data/reviews_flat_unextended.parquet"
df.to_parquet(out, index=False)
print(f"wrote {out}  rows={len(df):,}  unique users={df.contributor_id.nunique():,}  "
      f"restaurants={df.place_id.nunique():,}")
print("rating non-null:", df.rating.notna().mean().round(4),
      "| ts non-null:", df.timestamp_days_ago.notna().mean().round(4))
