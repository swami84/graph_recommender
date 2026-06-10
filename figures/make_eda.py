"""EDA figures for Part 1: who reviews what, and where.
- Unknown gender / unclassified race dropped; AIAN excluded from all race analyses.
- Cuisine breakdowns use the 15 most-reviewed cuisines + an "Other" bucket (16 total).
- User composition counts UNIQUE users (deduped on contributor_id), shown with %.
- Rating differences tested for significance (Welch t / ANOVA) AND effect size.
Regenerate: python figures/make_eda.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
from scipy import stats

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.axisbelow": True, "font.size": 10})

FIPS = {"01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE",
"11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS",
"21":"KY","22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN","28":"MS","29":"MO",
"30":"MT","31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND",
"39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD","47":"TN","48":"TX",
"49":"UT","50":"VT","51":"VA","53":"WA","54":"WV","55":"WI","56":"WY"}
RACE = {"nh_white":"White (NH)","nh_black":"Black (NH)","api":"Asian/PI","hispanic":"Hispanic"}  # AIAN excluded
RORDER = ["White (NH)", "Asian/PI", "Black (NH)", "Hispanic"]
RCOL = dict(zip(RORDER, ["#4393c3", "#e6822e", "#5ab4ac", "#d6604d"]))
GCOL = {"Female": "#d6604d", "Male": "#4393c3"}

def state_of(cbg):
    s = str(cbg).split(".")[0].zfill(12); return FIPS.get(s[:2])
def pfmt(p): return "p < 1e-300" if p < 1e-300 else f"p = {p:.1e}"
def cohend(a, b):
    n1, n2 = len(a), len(b); s = np.sqrt(((n1-1)*a.var(ddof=1)+(n2-1)*b.var(ddof=1))/(n1+n2-2))
    return (a.mean()-b.mean())/s
def interp_d(d):
    d = abs(d); return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"
def interp_eta(e):
    return "negligible" if e < 0.01 else "small" if e < 0.06 else "medium" if e < 0.14 else "large"

# Canonical (unextended) reviews, reconstructed from graph.pkl's edges. Those edges are 4x-duplicated,
# so we deduplicate on review_id to recover the true ~2.5M distinct interactions.
rev = (pd.read_parquet("data/reviews_flat_unextended.parquet",
        columns=["review_id","place_id","predicted_race","predicted_gender","rating"])
       .drop_duplicates("review_id"))
rest = pd.read_parquet("data/restaurants_enriched_llmcz.parquet", columns=["place_id","cuisine_category","cbg"])
rev = rev.merge(rest[["place_id","cuisine_category"]], on="place_id", how="left")
rev["race"] = rev["predicted_race"].map(RACE)
rev["gender"] = rev["predicted_gender"].str.capitalize()
rev = rev[rev["rating"].between(1, 5)]

# LLM-inferred cuisine (no generic "Other" bucket) — show the 16 specific cuisines
KEEP = [c for c in rev["cuisine_category"].value_counts().index
        if isinstance(c, str) and c != "Other"][:16]
CORDER = KEEP
def gcz(c): return c if c in KEEP else "Other"
rev["cz"] = rev["cuisine_category"].map(gcz)
rest["cz"] = rest["cuisine_category"].map(gcz)
KEEP10 = KEEP[:10]                                  # state cuisine-mix: top 10 + "Other"
CORDER10 = KEEP10 + ["Other"]
rest["cz10"] = rest["cuisine_category"].map(lambda c: c if c in KEEP10 else "Other")

rev_g = rev[rev["gender"].isin(["Female", "Male"])]
rev_r = rev[rev["race"].isin(RORDER)]

u = pd.read_parquet("data/reviews_flat_unextended.parquet",
        columns=["contributor_id","predicted_race","predicted_gender"]).drop_duplicates("contributor_id")
u["race"] = u["predicted_race"].map(RACE); u["gender"] = u["predicted_gender"].str.capitalize()

# ================= SIGNIFICANCE =================
f = rev_g[rev_g.gender == "Female"].rating.to_numpy(); m = rev_g[rev_g.gender == "Male"].rating.to_numpy()
t, p = stats.ttest_ind(f, m, equal_var=False); d = cohend(f, m)
GTXT = (f"Female {f.mean():.3f} vs Male {m.mean():.3f}\nWelch t={t:.0f}, {pfmt(p)}\n"
        f"Cohen's d = {d:.3f} ({interp_d(d)})")
groups = [rev_r[rev_r.race == r].rating.to_numpy() for r in RORDER]
F, pa = stats.f_oneway(*groups)
allx = rev_r.rating.to_numpy(); gm = allx.mean()
ssb = sum(len(g)*(g.mean()-gm)**2 for g in groups); eta = ssb/((allx-gm)**2).sum()
RTXT = f"ANOVA F={F:.0f}, {pfmt(pa)}\nη² = {eta:.4f} ({interp_eta(eta)})"

# ================= FIG A: unique-user composition =================
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
gc = u[u.gender.isin(["Female", "Male"])].gender.value_counts().reindex(["Female", "Male"])
ax[0].bar(gc.index, gc.values/1e3, color=[GCOL[x] for x in gc.index])
for i, (k, v) in enumerate(gc.items()):
    ax[0].text(i, v/1e3, f"{v/1e3:.0f}K\n{v/gc.sum()*100:.0f}%", ha="center", va="bottom", fontsize=9)
ax[0].set_title("Unique users by gender"); ax[0].set_ylabel("users (thousands)"); ax[0].margins(y=0.18)
rc = u[u.race.isin(RORDER)].race.value_counts().reindex(RORDER).dropna()
ax[1].bar(rc.index, rc.values/1e3, color=[RCOL[x] for x in rc.index])
for i, (k, v) in enumerate(rc.items()):
    ax[1].text(i, v/1e3, f"{v/1e3:.0f}K\n{v/rc.sum()*100:.0f}%", ha="center", va="bottom", fontsize=9)
ax[1].set_title("Unique users by race/ethnicity"); ax[1].set_ylabel("users (thousands)"); ax[1].margins(y=0.18)
fig.suptitle(f"Reviewer composition  ({len(u[u.gender.isin(['Female','Male'])]):,} gender-classified, "
             f"{len(u[u.race.isin(RORDER)]):,} race-classified unique users)", y=1.02, fontsize=10)
plt.tight_layout(); plt.savefig("figures/eda_user_composition.png", bbox_inches="tight"); plt.close()

# ================= FIG B: gender rating by cuisine + sig =================
fig, ax = plt.subplots(figsize=(9, 7.2))
piv = rev_g.pivot_table("rating", "cz", "gender", "mean").reindex(CORDER)
piv[["Female", "Male"]].plot(kind="barh", ax=ax, color=GCOL, width=0.8)
ax.set_title("Mean rating by gender, by cuisine"); ax.set_xlabel("mean rating"); ax.set_ylabel("")
ax.set_xlim(3.3, 4.5); ax.legend(title="")
ax.text(0.015, 0.02, GTXT, transform=ax.transAxes, fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round", fc="#fffbe6", ec="#999"))
plt.tight_layout(); plt.savefig("figures/eda_rating_gender.png", bbox_inches="tight"); plt.close()

# ================= FIG C: race x cuisine heatmap + ANOVA =================
H = rev_r.pivot_table("rating", "cz", "race", "mean").reindex(CORDER)[RORDER]
fig, ax = plt.subplots(figsize=(8.5, 7.4))
im = ax.imshow(H.values, cmap="RdYlGn", aspect="auto", vmin=3.5, vmax=4.4)
ax.set_xticks(range(len(H.columns))); ax.set_xticklabels(H.columns, rotation=20, ha="right")
ax.set_yticks(range(len(H.index))); ax.set_yticklabels(H.index)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        if np.isfinite(H.values[i, j]):
            ax.text(j, i, f"{H.values[i,j]:.2f}", ha="center", va="center", fontsize=7.5)
ax.set_title("Mean rating by race × cuisine"); fig.colorbar(im, label="mean rating", fraction=0.046); ax.grid(False)
ax.text(1.18, -0.02, RTXT, transform=ax.transAxes, fontsize=8.3, va="top",
        bbox=dict(boxstyle="round", fc="#fffbe6", ec="#999"))
plt.tight_layout(); plt.savefig("figures/eda_race_cuisine_heatmap.png", bbox_inches="tight"); plt.close()

# ================= FIG D: composition of reviewers by cuisine =================
fig, ax = plt.subplots(1, 2, figsize=(14, 7.4))
rcomp = rev_r.pivot_table(index="cz", columns="race", aggfunc="size", fill_value=0).reindex(CORDER)[RORDER]
rcomp = rcomp.div(rcomp.sum(1), axis=0)
rcomp.plot(kind="barh", stacked=True, ax=ax[0], color=[RCOL[r] for r in RORDER], width=0.82)
ax[0].set_title("Race mix of reviewers, by cuisine"); ax[0].set_xlabel("share of reviews"); ax[0].set_ylabel("")
ax[0].legend(title="", fontsize=8, loc="lower right"); ax[0].set_xlim(0, 1)
gcomp = rev_g.pivot_table(index="cz", columns="gender", aggfunc="size", fill_value=0).reindex(CORDER)[["Female", "Male"]]
gcomp = gcomp.div(gcomp.sum(1), axis=0)
gcomp.plot(kind="barh", stacked=True, ax=ax[1], color=[GCOL["Female"], GCOL["Male"]], width=0.82)
ax[1].set_title("Gender mix of reviewers, by cuisine"); ax[1].set_xlabel("share of reviews"); ax[1].set_ylabel("")
ax[1].axvline(0.5, color="#333", ls="--", lw=1); ax[1].legend(title="", fontsize=8, loc="lower right"); ax[1].set_xlim(0, 1)
plt.tight_layout(); plt.savefig("figures/eda_composition_by_cuisine.png", bbox_inches="tight"); plt.close()

# ================= FIG E: cuisine % distribution (catalogue, overall) =================
cc = (rest["cz"].value_counts(normalize=True) * 100).reindex(CORDER).dropna()[::-1]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(cc.index, cc.values, color="#3182bd")
for i, v in enumerate(cc.values):
    ax.text(v + 0.25, i, f"{v:.1f}%", va="center", fontsize=9)
ax.set_title("Restaurants by cuisine (% of catalogue)"); ax.set_xlabel("% of restaurants")
ax.set_xlim(0, cc.max() * 1.12)
plt.tight_layout(); plt.savefig("figures/eda_cuisine_dist.png", bbox_inches="tight"); plt.close()

# ================= FIG F: by state =================
rest["state"] = rest["cbg"].map(state_of)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
sc = rest["state"].value_counts().head(12)[::-1]
ax[0].barh(sc.index, sc.values, color="#756bb1"); ax[0].set_title("Restaurants by state (top 12)"); ax[0].set_xlabel("restaurants")
tops = rest["state"].value_counts().head(8).index.tolist()
comp = rest[rest["state"].isin(tops)].pivot_table(index="state", columns="cz10", aggfunc="size", fill_value=0).reindex(tops)
comp = comp.reindex(columns=[c for c in CORDER10 if c in comp.columns])
comp = comp.div(comp.sum(1), axis=0)
comp.plot(kind="bar", stacked=True, ax=ax[1], colormap="tab20", width=0.8)
ax[1].set_title("Cuisine mix by state (top states)"); ax[1].set_ylabel("share of restaurants"); ax[1].set_xlabel("")
ax[1].legend(title="", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig("figures/eda_state.png", bbox_inches="tight"); plt.close()

# ================= FIG G: distribution of users by number of reviews =================
rc = (pd.read_parquet("data/reviews_flat_unextended.parquet", columns=["contributor_id", "review_id"])
        .drop_duplicates("review_id").contributor_id.value_counts())
top10 = rc.sort_values(ascending=False).head(max(1, round(len(rc) * 0.1))).sum() / rc.sum() * 100
bins = [0, 1, 2, 3, 5, 10, 25, 1e9]; lab = ["1", "2", "3", "4–5", "6–10", "11–25", "26+"]
pct = (pd.cut(rc, bins=bins, labels=lab, right=True).value_counts(normalize=True) * 100).reindex(lab)
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#c2c9d3", "#c2c9d3"] + ["#3182bd"] * 5      # 1–2 below cutoff (grey), 3+ kept (blue)
ax.bar(range(len(pct)), pct.values, color=colors, width=0.78)
for i, v in enumerate(pct.values):
    ax.text(i, v + 1.0, f"{v:.1f}%", ha="center", fontsize=9.5)
ax.axvline(1.5, ls="--", color="#c75d62", lw=1.6)
ax.text(1.65, 60, "model keeps the denser tail\n(diners with ≥3 restaurants)  →", color="#c75d62", fontsize=9)
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab)
ax.set_xlabel("reviews per user"); ax.set_ylabel("% of users"); ax.set_ylim(0, 92)
ax.set_title(f"How active are reviewers?   ({len(rc):,} users · median {int(rc.median())} review · top 10% write {top10:.0f}%)")
plt.tight_layout(); plt.savefig("figures/eda_user_activity.png", bbox_inches="tight"); plt.close()

# ================= FIG H: the active reviewers (>3 reviews) =================
av = (pd.read_parquet("data/reviews_flat_unextended.parquet",
        columns=["contributor_id", "place_id", "review_id", "rating", "is_local_guide"])
      .drop_duplicates("review_id"))
nrest = av.groupby("contributor_id").place_id.nunique()
hv = set(nrest[nrest >= 3].index)          # the model's training population: ≥3 distinct restaurants
a = av[av.contributor_id.isin(hv)].merge(rest[["place_id", "cbg", "cuisine_category"]], on="place_id", how="left")
a["state"] = a["cbg"].map(state_of)
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
# panel 1: state distribution (top 8 by their reviews)
sd = (a.state.value_counts(normalize=True) * 100).head(8)[::-1]
ax[0].barh(sd.index, sd.values, color="#756bb1")
for i, v in enumerate(sd.values): ax[0].text(v + 0.3, i, f"{v:.0f}%", va="center", fontsize=9)
ax[0].set_title("Where they review (top states)"); ax[0].set_xlabel("% of their reviews")
# panel 2: rating distribution vs all
rr = (a.rating.round().clip(1, 5).value_counts(normalize=True) * 100).reindex([1, 2, 3, 4, 5]).fillna(0)
allr = (av.rating.round().clip(1, 5).value_counts(normalize=True) * 100).reindex([1, 2, 3, 4, 5]).fillna(0)
x = np.arange(5)
ax[1].bar(x - 0.2, allr.values, 0.4, color="#c2c9d3", label="all users")
ax[1].bar(x + 0.2, rr.values, 0.4, color="#5ab4ac", label="active (≥3 rest.)")
ax[1].set_xticks(x); ax[1].set_xticklabels(["1★", "2★", "3★", "4★", "5★"])
ax[1].set_ylabel("% of ratings"); ax[1].set_title(f"How they rate  (mean {a.rating.mean():.2f} vs {av.rating.mean():.2f})")
ax[1].legend(frameon=False, fontsize=9)
# panel 3: cuisine diversity (distinct cuisines per active diner)
dv = a.groupby("contributor_id").cuisine_category.nunique()
hh = dv.clip(upper=8).value_counts(normalize=True).sort_index() * 100
labels = [(f"{int(k)}" if k < 8 else "8+") for k in hh.index]
ax[2].bar(range(len(hh)), hh.values, color="#5ab4ac", width=0.8)
ax[2].axvline(dv.mean() - 1, ls="--", color="#c75d62", lw=1.4)
ax[2].text(dv.mean() - 1, hh.max(), f" mean {dv.mean():.1f}", color="#c75d62", fontsize=9)
ax[2].set_xticks(range(len(hh))); ax[2].set_xticklabels(labels)
ax[2].set_title("Cuisine variety"); ax[2].set_xlabel("distinct cuisines visited"); ax[2].set_ylabel("% of diners")
share = a.shape[0] / av.shape[0] * 100
lgp = a.drop_duplicates("contributor_id").is_local_guide.mean() * 100
spd = a.groupby("contributor_id").state.nunique().mean()
fig.suptitle(f"The active diners: {len(hv):,} reviewers with 3+ distinct restaurants — {share:.0f}% of all reviews · "
             f"{lgp:.0f}% Local Guides · ~{spd:.0f} states each · {(dv>=5).mean()*100:.0f}% sample 5+ cuisines",
             fontsize=10.5, y=1.02, color="#444")
plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig("figures/eda_active_reviewers.png", bbox_inches="tight"); plt.close()

# ================= FIG I: how people write — review length & photos =================
# text_len / attached_photos are not retained in the reconstructed canonical file, but the
# canonical review_ids are a 100% subset of reviews_flat, so we attach them by review_id.
txt = (pd.read_parquet("data/reviews_flat_unextended.parquet",
            columns=["review_id", "contributor_id", "rating"]).drop_duplicates("review_id")
       .merge(pd.read_parquet("data/reviews_flat.parquet", columns=["review_id", "text_len", "attached_photos"]),
              on="review_id", how="left"))
txt = txt[txt.rating.between(1, 5)]
txt["rb"] = txt.rating.round().astype(int)
RSTARS = [1, 2, 3, 4, 5]
RBAR = ["#c0504d", "#d98558", "#e7c46a", "#7ab07f", "#3f8f78"]   # red (low) → green (high)
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
# panel 1: mean review length by rating
ln = txt.groupby("rb")["text_len"].mean().reindex(RSTARS)
ax[0].bar([f"{i}★" for i in RSTARS], ln.values, color=RBAR)
for i, v in enumerate(ln.values): ax[0].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
ax[0].set_title("Review length by rating"); ax[0].set_ylabel("mean characters"); ax[0].margins(y=0.16)
# panel 2: mean review length by reviewer activity
nrev = txt.groupby("contributor_id").review_id.size()
txt["ubin"] = pd.cut(txt.contributor_id.map(nrev), [0, 1, 3, 5, 10, 1e9], labels=["1", "2–3", "4–5", "6–10", "11+"])
lb = txt.groupby("ubin", observed=True)["text_len"].mean().reindex(["1", "2–3", "4–5", "6–10", "11+"])
ax[1].bar(range(len(lb)), lb.values, color="#5ab4ac", width=0.78)
for i, v in enumerate(lb.values): ax[1].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
ax[1].set_xticks(range(len(lb))); ax[1].set_xticklabels(lb.index)
ax[1].set_title("Review length by reviewer activity"); ax[1].set_xlabel("reviews per user")
ax[1].set_ylabel("mean characters"); ax[1].margins(y=0.16)
# panel 3: mean photos per review by rating
ph = txt.groupby("rb")["attached_photos"].mean().reindex(RSTARS)
ax[2].bar([f"{i}★" for i in RSTARS], ph.values, color=RBAR)
for i, v in enumerate(ph.values): ax[2].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
ax[2].set_title("Photos per review by rating"); ax[2].set_ylabel("mean photos attached"); ax[2].margins(y=0.16)
fig.suptitle("How people write: complaints run long, praise runs short — and photos follow the good experiences",
             y=1.02, fontsize=10.5, color="#444")
plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig("figures/eda_review_text.png", bbox_inches="tight"); plt.close()
print(f"[prose] length by rating: " + " ".join(f"{i}★ {ln[i]:.0f}" for i in RSTARS))
print(f"[prose] length by activity: " + " ".join(f"{k}:{v:.0f}" for k, v in lb.items()))
print(f"[prose] photos by rating (mean): " + " ".join(f"{i}★ {ph[i]:.2f}" for i in RSTARS)
      + f" | overall mean {txt.attached_photos.mean():.2f}, {(txt.attached_photos>0).mean()*100:.0f}% carry ≥1 photo")

print("GENDER:", GTXT.replace("\n", " | ")); print("RACE:", RTXT.replace("\n", " | "))
print("cuisines shown (16):", CORDER)
print(f"[prose] total unique reviewers: {len(u):,} | gender-classified: {len(u[u.gender.isin(['Female','Male'])]):,}"
      f" | race-classified: {len(u[u.race.isin(RORDER)]):,}")
gc_ = u[u.gender.isin(['Female','Male'])].gender.value_counts()
print(f"[prose] gender M {gc_.get('Male',0):,} ({gc_.get('Male',0)/gc_.sum()*100:.1f}%) "
      f"F {gc_.get('Female',0):,} ({gc_.get('Female',0)/gc_.sum()*100:.1f}%) | "
      f"unknown gender {(1-len(u[u.gender.isin(['Female','Male'])])/len(u))*100:.1f}% of all reviewers")
rc_ = u[u.race.isin(RORDER)].race.value_counts().reindex(RORDER)
print("[prose] race counts/%:", {k: f"{int(v):,} ({v/rc_.sum()*100:.1f}%)" for k, v in rc_.items()})
print(f"[prose] distinct reviews (deduped): {rev.shape[0]:,} | single-review users: "
      f"{(rc==1).mean()*100:.1f}% | top10% write {top10:.0f}%")
print(f"[prose] active core (≥3 restaurants): {len(hv):,} = {len(hv)/len(rc)*100:.1f}% of users · "
      f"{share:.1f}% of reviews · {lgp:.1f}% local guides · mean rating {a.rating.mean():.2f} vs {av.rating.mean():.2f} · "
      f"~{spd:.1f} states · mean {dv.mean():.1f} cuisines · {(dv>=5).mean()*100:.0f}% try 5+ · {(dv==1).mean()*100:.1f}% single-cuisine")
print("wrote EDA figures")
