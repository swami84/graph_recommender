"""Conceptual schematic: two feature-engineering tracks enriching the restaurant graph.
Illustrative (not a literal 1:1 of every feature). Regenerate: python figures/make_schematic.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ANALYTIC = "#2c7fb8"   # blue
LLM      = "#e6822e"   # orange
INK      = "#222222"

fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.set_title("Feature engineering: two tracks enriching the restaurant graph",
             fontsize=15, fontweight="bold", pad=12)

def panel(x, y, w, h, color, title, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                 linewidth=2, edgecolor=color, facecolor=color+"18"))
    ax.text(x+w/2, y+h-3.2, title, ha="center", va="top", fontsize=11.5,
            fontweight="bold", color=color)
    for i, ln in enumerate(lines):
        ax.text(x+2.5, y+h-8.5-i*5.2, "•  "+ln, ha="left", va="top", fontsize=9.6, color=INK)

# Left: the two tracks
panel(2, 52, 33, 42, ANALYTIC, "Analytical track\n(computed from data)",
      ["Base metadata: price, rating,\n    cuisine, Food/Service/Atmosphere",
       "Behavioral aggregates: cuisine\n    affinity & entropy, geo range",
       "Spatial density (competition\n    within 500 m / 2 km)",
       "CBG foot-traffic rhythm\n    (lunch / evening / late-night)",
       "Review dynamics: rating trend,\n    velocity, recency"])
panel(2, 6, 33, 40, LLM, "LLM-extracted track\n(read from review text)",
      ["Restaurant attributes: spice,\n    noise, formality, romance …",
       "Dishes as entities +\n    flavor profiles",
       "User preference fingerprint\n    (aligned 1:1 with attributes)",
       "Dietary matching: affinities\n    ↔ vegan / veg / GF flags"])

# Right: the graph
nodes = {
    "User":       (60, 74, 7.2, "#5ab4ac"),
    "Restaurant": (82, 52, 9.0, "#d6604d"),
    "Dish":       (90, 24, 6.0, "#9970ab"),
    "CBG":        (61, 30, 5.2, "#999999"),
    "Cuisine":    (95, 74, 5.0, "#999999"),
}
edges = [("User","Restaurant","reviews"), ("Restaurant","Dish","serves"),
         ("Restaurant","CBG","located in"), ("Restaurant","Cuisine","cuisine")]
for a,b,lbl in edges:
    xa,ya,_,_ = nodes[a]; xb,yb,_,_ = nodes[b]
    ax.plot([xa,xb],[ya,yb], color="#bbbbbb", lw=1.6, zorder=1)
    ax.text((xa+xb)/2,(ya+yb)/2, lbl, fontsize=7.6, color="#777777",
            ha="center", va="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
for name,(x,y,r,c) in nodes.items():
    ax.add_patch(Circle((x,y), r, facecolor=c, edgecolor="white", lw=2, zorder=3))
    ax.text(x,y,name, ha="center", va="center", fontsize=9.2, fontweight="bold",
            color="white", zorder=4)

# Track -> node enrichment arrows
def arrow(x1,y1,x2,y2,color,rad):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), connectionstyle=f"arc3,rad={rad}",
                 arrowstyle="-|>", mutation_scale=15, lw=2, color=color, alpha=0.8, zorder=2))
arrow(35, 80, 53.5, 75, ANALYTIC, -0.18)   # analytic -> user
arrow(35, 66, 73.5, 53, ANALYTIC, -0.10)   # analytic -> restaurant
arrow(35, 38, 54.5, 71, LLM, -0.28)        # llm -> user
arrow(35, 30, 73.5, 50, LLM, -0.12)        # llm -> restaurant
arrow(35, 16, 84.5, 22, LLM, 0.20)         # llm -> dish

# Legend
ax.add_patch(FancyArrowPatch((40,2.5),(46,2.5), arrowstyle="-|>", mutation_scale=13, lw=2, color=ANALYTIC))
ax.text(47,2.5,"analytical enriches", va="center", fontsize=9, color=INK)
ax.add_patch(FancyArrowPatch((70,2.5),(76,2.5), arrowstyle="-|>", mutation_scale=13, lw=2, color=LLM))
ax.text(77,2.5,"LLM enriches", va="center", fontsize=9, color=INK)

plt.tight_layout()
out = "figures/feature_tracks_schematic.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
