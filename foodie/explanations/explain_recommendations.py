"""Explainability layer: post-hoc, human-readable reasons for why each restaurant was recommended.

For a small sample of test diners, we explain the top recommendations using four signals the model
implicitly combines:
  • proximity      — is it near where the diner usually eats?
  • cuisine affinity — does it match cuisines they've rated highly?
  • taste profile  — does it match their LLM-derived preference fingerprint (spice, formality, ...)?
  • quality        — is it well-rated / popular?

Anchored on the surviving unextended KGAT-SAL + proximity predictions. Writes a markdown report.
Run: python -m foodie.explanations.explain_recommendations
"""
import pandas as pd, numpy as np

PRED = "data/predictions/kgat_sal_all_T3_nospatial_prox_a30_predictions.parquet"
N_USERS, TOP_K = 6, 3
OUT = "article/explainability_examples.md"

# user preference dim -> (item attribute, human phrase)
PREF_MAP = {
    "spice_affinity": ("spice_level", "spicy food"),
    "formality_preference": ("formality", "upscale, formal dining"),
    "romantic_context": ("romantic", "romantic spots"),
    "healthy_preference": ("healthy_options", "healthy options"),
    "bar_affinity": ("bar_scene", "a lively bar scene"),
    "outdoor_preference": ("outdoor_seating", "outdoor seating"),
    "family_context": ("family_friendly", "family-friendly places"),
    "novelty_preference": ("novelty", "trendy, novel spots"),
}

def haversine(a, b):
    (la1, lo1), (la2, lo2) = np.radians(a), np.radians(b)
    d = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(d))

print("loading…")
pred = pd.read_parquet(PRED, columns=["contributor_id", "true_place_id", "top10_place_ids"])
rest = pd.read_parquet("data/restaurants_enriched.parquet",
                       columns=["place_id", "name", "cuisine_category", "lat", "lng", "rating", "user_rating_count"]).set_index("place_id")
prefs = pd.read_parquet("data/user_features_prebuilt.parquet",
                        columns=["contributor_id"] + list(PREF_MAP)).set_index("contributor_id")
items = pd.read_parquet("data/item_features_prebuilt.parquet",
                        columns=["place_id"] + [v[0] for v in PREF_MAP.values()]).set_index("place_id")
hist = pd.read_parquet("data/reviews_flat.parquet", columns=["contributor_id", "place_id", "rating"])

# pick a DIVERSE sample of diners: a preference profile + rich, locatable history
cand = pred[pred.contributor_id.isin(prefs.index)].drop_duplicates("contributor_id")
hsel = hist[hist.contributor_id.isin(cand.contributor_id) & hist.place_id.isin(rest.index)]
counts = hsel.groupby("contributor_id").place_id.nunique()
rich = counts[counts >= 8].index
pool = cand[cand.contributor_id.isin(rich)]
sample_ids = pool.sample(min(N_USERS * 4, len(pool)), random_state=7).contributor_id.tolist()
hist_by = {u: g for u, g in hsel[hsel.contributor_id.isin(sample_ids)].groupby("contributor_id")}
cand = cand[cand.contributor_id.isin(sample_ids)]

def tolist(v):
    return list(v) if isinstance(v, (list, np.ndarray)) else []

def explain(uid, rec_pid, home, top_cuisines, liked_by_cuisine):
    reasons = []
    if rec_pid not in rest.index:
        return reasons
    r = rest.loc[rec_pid]
    # 1) proximity
    if home is not None and pd.notna(r.lat):
        km = haversine(home, (r.lat, r.lng))
        if km <= 10:
            reasons.append(f"📍 only ~{km:.1f} km from where you usually eat")
    # 2) cuisine affinity
    cz = r.cuisine_category
    if cz in top_cuisines and cz in liked_by_cuisine:
        ex_name, ex_rt = liked_by_cuisine[cz]
        reasons.append(f"🍽️ same cuisine ({cz}) as **{ex_name}**, which you rated {ex_rt:.0f}★")
    elif cz in top_cuisines:
        reasons.append(f"🍽️ {cz} — one of the cuisines you review most")
    # 3) taste-profile alignment (strongest matching dimension)
    if uid in prefs.index and rec_pid in items.index:
        best, bestv = None, 0.0
        up, it = prefs.loc[uid], items.loc[rec_pid]
        for pcol, (icol, phrase) in PREF_MAP.items():
            u, i = up.get(pcol, np.nan), it.get(icol, np.nan)
            if pd.notna(u) and pd.notna(i) and u > 0.6 and i > 0.6 and min(u, i) > bestv:
                best, bestv = phrase, min(u, i)
        if best:
            reasons.append(f"🎯 fits your taste for {best}")
    # 4) quality
    if pd.notna(r.rating) and r.rating >= 4.3 and r.user_rating_count >= 100:
        reasons.append(f"⭐ highly rated — {r.rating:.1f}★ over {int(r.user_rating_count):,} reviews")
    return reasons[:3]

lines = ["# Explaining the recommendations\n",
         "*Post-hoc explanations for why each restaurant was recommended, for a sample of test diners. "
         "Each reason is grounded in the diner's own history, location, and taste profile.*\n"]
done = 0
for _, row in cand.iterrows():
    uid = row.contributor_id
    g = hist_by.get(uid)
    if g is None:
        continue
    g = g[g.place_id.isin(rest.index)]
    if g.place_id.nunique() < 5:
        continue
    coords = rest.loc[g.place_id, ["lat", "lng"]].dropna()
    home = (coords.lat.median(), coords.lng.median()) if len(coords) else None
    gj = g.join(rest[["cuisine_category", "name"]], on="place_id")
    top_cuisines = gj.cuisine_category.value_counts().head(3).index.tolist()
    liked = gj[gj.rating >= 4].sort_values("rating", ascending=False)
    liked_by_cuisine = {}
    for _, lr in liked.iterrows():
        liked_by_cuisine.setdefault(lr.cuisine_category, (lr["name"], lr.rating))
    recs = list(dict.fromkeys(tolist(row.top10_place_ids)))[:TOP_K]
    if not recs:
        continue
    # profile blurb
    strong = [PREF_MAP[p][1] for p in PREF_MAP if uid in prefs.index and prefs.loc[uid].get(p, 0) > 0.65]
    lines.append(f"\n---\n\n## Diner {done+1}\n")
    lines.append(f"**Profile:** reviews mostly {', '.join(top_cuisines[:3])} · "
                 f"{len(g)} restaurants visited" + (f" · leans toward {', '.join(strong[:2])}" if strong else "") + "\n")
    for pid in recs:
        if pid not in rest.index:
            continue
        r = rest.loc[pid]
        hit = " ✓ *(this was their actual next visit)*" if pid == row.true_place_id else ""
        rs = explain(uid, pid, home, top_cuisines, liked_by_cuisine)
        lines.append(f"\n**→ {r['name']}** ({r.cuisine_category}){hit}")
        for x in (rs or ["recommended by the model's learned similarity to your history"]):
            lines.append(f"  - {x}")
    done += 1
    if done >= N_USERS:
        break

open(OUT, "w").write("\n".join(lines))
print(f"wrote {OUT}  ({done} diners explained)")
print("\n".join(lines[:60]))
