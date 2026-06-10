#!/usr/bin/env python3
"""
graphrag_recommend.py — Phase 5: GraphRAG explanation layer.

For a (contributor_id, place_id) pair:
  1. Extract the 2-hop neighbourhood from the knowledge graph
  2. Retrieve the most relevant review snippets via GNN embeddings
  3. Prompt Claude to generate a natural-language recommendation explanation

Requires:
  - data/graph.pkl              (from build_graph.py)
  - data/embeddings/            (from recommendation_gnn.py)
  - models/gnn_encoders.pkl
  - ANTHROPIC_API_KEY in .env

Usage:
    python graphrag_recommend.py CONTRIBUTOR_ID PLACE_ID
    from graphrag_recommend import explain
    print(explain("CONTRIBUTOR_ID", "PLACE_ID"))
"""

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("graphrag")

GRAPH_FILE      = Path("data/graph.pkl")
REVIEWS_FLAT    = Path("data/reviews_flat.parquet")
RESTAURANTS_ENR = Path("data/restaurants_enriched.parquet")
ENCODERS_FILE   = Path("models/gnn_encoders.pkl")
EMB_DIR         = Path("data/embeddings")

# ── Graph loading (cached in module scope) ─────────────────────────────────────
_graph = None
_reviews_df: Optional[pd.DataFrame] = None
_restaurants_df: Optional[pd.DataFrame] = None
_user_emb: Optional[torch.Tensor] = None
_item_emb: Optional[torch.Tensor] = None
_encoders: Optional[dict] = None


def _load_graph():
    global _graph
    if _graph is None:
        if not GRAPH_FILE.exists():
            raise FileNotFoundError(f"Graph not found: {GRAPH_FILE}. Run build_graph.py first.")
        log.info("Loading knowledge graph…")
        with open(GRAPH_FILE, "rb") as f:
            _graph = pickle.load(f)
        log.info(f"Graph loaded: {_graph.number_of_nodes():,} nodes, {_graph.number_of_edges():,} edges")
    return _graph


def _load_embeddings():
    global _user_emb, _item_emb, _encoders
    if _user_emb is None:
        u_path = EMB_DIR / "reviewer_embeddings.pt"
        i_path = EMB_DIR / "restaurant_embeddings.pt"
        enc_path = ENCODERS_FILE
        if u_path.exists() and i_path.exists() and enc_path.exists():
            _user_emb = torch.load(u_path, map_location="cpu")
            _item_emb = torch.load(i_path, map_location="cpu")
            with open(enc_path, "rb") as f:
                _encoders = pickle.load(f)
            log.info("GNN embeddings loaded")
        else:
            log.warning("GNN embeddings not found — similarity ranking unavailable. "
                        "Run recommendation_gnn.py first.")
    return _user_emb, _item_emb, _encoders


def _load_reviews():
    global _reviews_df
    if _reviews_df is None and REVIEWS_FLAT.exists():
        _reviews_df = pd.read_parquet(REVIEWS_FLAT)
    return _reviews_df


def _load_restaurants():
    global _restaurants_df
    if _restaurants_df is None and RESTAURANTS_ENR.exists():
        _restaurants_df = pd.read_parquet(RESTAURANTS_ENR)
    return _restaurants_df


# ── Graph neighbourhood extraction ────────────────────────────────────────────
def _node_id(ntype: str, key: str) -> str:
    return f"{ntype}::{key}"


def get_neighbourhood(contributor_id: str, place_id: str, hops: int = 2) -> dict:
    """
    Extract the 2-hop neighbourhood around (reviewer, restaurant) from the graph.
    Returns structured context for the LLM.
    """
    G = _load_graph()

    reviewer_node    = _node_id("reviewer",    contributor_id)
    restaurant_node  = _node_id("restaurant",  place_id)

    ctx = {
        "reviewer":     {},
        "restaurant":   {},
        "reviewer_history": [],     # restaurants the reviewer has rated
        "similar_restaurants": [],  # restaurants sharing cuisine/CBG with target
        "dishes":       [],         # dishes served at target restaurant
        "top_reviews":  [],         # review snippets for target restaurant
    }

    # Reviewer metadata
    if reviewer_node in G:
        d = G.nodes[reviewer_node]
        ctx["reviewer"] = {
            "name":              d.get("name", ""),
            "predicted_race":    d.get("predicted_race"),
            "predicted_gender":  d.get("predicted_gender"),
            "is_local_guide":    d.get("is_local_guide"),
            "total_reviews":     d.get("total_reviews"),
        }

    # Restaurant metadata
    if restaurant_node in G:
        d = G.nodes[restaurant_node]
        ctx["restaurant"] = {
            "name":        d.get("name", ""),
            "cuisine":     d.get("cuisine", ""),
            "price_level": d.get("price_level"),
            "rating":      d.get("rating"),
            "cbg":         d.get("cbg", ""),
        }

    # Reviewer's past restaurants (outgoing REVIEWED edges)
    if reviewer_node in G:
        for _, dst, edata in G.out_edges(reviewer_node, data=True):
            if edata.get("etype") == "REVIEWED":
                r_data = G.nodes[dst]
                ctx["reviewer_history"].append({
                    "name":    r_data.get("name", ""),
                    "cuisine": r_data.get("cuisine", ""),
                    "rating":  edata.get("rating"),
                })
        # Limit to 10 most recent
        ctx["reviewer_history"] = sorted(
            ctx["reviewer_history"],
            key=lambda x: x.get("rating", 0) or 0, reverse=True
        )[:10]

    # Dishes at target restaurant
    if restaurant_node in G:
        for _, dst, edata in G.out_edges(restaurant_node, data=True):
            if edata.get("etype") == "SERVES":
                dish_data = G.nodes[dst]
                ctx["dishes"].append({
                    "name":    dish_data.get("dish_name", ""),
                    "mentions": edata.get("mention_count", 1),
                })
        ctx["dishes"] = sorted(ctx["dishes"], key=lambda x: x["mentions"], reverse=True)[:10]

    # Similar restaurants (same cuisine in same CBG)
    target_cuisine = ctx["restaurant"].get("cuisine", "")
    target_cbg     = ctx["restaurant"].get("cbg", "")
    if target_cuisine:
        cuisine_node = _node_id("cuisine", target_cuisine)
        if cuisine_node in G:
            for src, _, edata in G.in_edges(cuisine_node, data=True):
                if edata.get("etype") == "HAS_CUISINE" and src != restaurant_node:
                    r_data = G.nodes[src]
                    if r_data.get("cbg") == target_cbg:
                        ctx["similar_restaurants"].append({
                            "name":   r_data.get("name", ""),
                            "rating": r_data.get("rating"),
                        })
        ctx["similar_restaurants"] = ctx["similar_restaurants"][:5]

    return ctx


def get_top_reviews(place_id: str, n: int = 5) -> list[dict]:
    """Return the top-n highest-rated content reviews for a restaurant."""
    reviews = _load_reviews()
    if reviews is None:
        return []
    sub = (
        reviews[
            (reviews["place_id"] == place_id) &
            reviews["has_content"] &
            reviews["rating"].notna()
        ]
        .sort_values("rating", ascending=False)
        .head(n)
    )
    return [
        {
            "text":   row["text"][:400],  # truncate for context window
            "rating": row["rating"],
            "race":   row.get("predicted_race"),
            "gender": row.get("predicted_gender"),
        }
        for _, row in sub.iterrows()
    ]


# ── Embedding-based similarity ─────────────────────────────────────────────────
def similar_restaurants_by_embedding(place_id: str, top_n: int = 5) -> list[dict]:
    """Find restaurants with similar GNN embeddings to the target."""
    _, item_emb, encoders = _load_embeddings()
    if item_emb is None or encoders is None:
        return []

    item_enc = encoders.get("item_enc", {})
    item_dec = encoders.get("item_dec", {})
    if place_id not in item_enc:
        return []

    target_idx  = item_enc[place_id]
    target_vec  = item_emb[target_idx]
    sims        = F.cosine_similarity(target_vec.unsqueeze(0), item_emb, dim=1)
    sims[target_idx] = -1  # exclude self

    top_idx = sims.topk(top_n).indices.tolist()
    restaurants = _load_restaurants()
    rest_map = (
        restaurants.set_index("place_id").to_dict("index")
        if restaurants is not None else {}
    )
    return [
        {
            "place_id":  item_dec.get(idx, ""),
            "name":      rest_map.get(item_dec.get(idx, ""), {}).get("name", ""),
            "cuisine":   rest_map.get(item_dec.get(idx, ""), {}).get("cuisine_category", ""),
            "similarity": float(sims[idx]),
        }
        for idx in top_idx
    ]


try:
    import torch.nn.functional as F
except ImportError:
    pass


# ── LLM backends ──────────────────────────────────────────────────────────────
def _call_claude(prompt: str) -> str:
    try:
        import anthropic
    except ImportError:
        return "[anthropic package not installed — run: pip install anthropic]"

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ANTHROPIC_API_KEY not set in .env]"

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_local(prompt: str, url: str, model: str) -> str:
    """Call a local vLLM/Ollama OpenAI-compatible endpoint."""
    import urllib.request
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def _call_llm(prompt: str, backend: str, local_url: str, local_model: str) -> str:
    if backend == "claude":
        return _call_claude(prompt)
    return _call_local(prompt, local_url, local_model)


def _build_prompt(contributor_id: str, place_id: str, ctx: dict,
                  top_reviews: list[dict], similar_emb: list[dict]) -> str:
    reviewer   = ctx.get("reviewer", {})
    restaurant = ctx.get("restaurant", {})
    history    = ctx.get("reviewer_history", [])
    dishes     = ctx.get("dishes", [])

    history_str = "\n".join(
        f"  - {r['name']} ({r['cuisine']}) — {r['rating']}★"
        for r in history[:6]
    ) or "  (no history available)"

    dishes_str = ", ".join(d["name"] for d in dishes[:6]) or "(none recorded)"

    reviews_str = "\n".join(
        f"  - [{r['rating']}★] \"{r['text'][:200]}…\""
        for r in top_reviews[:3]
    ) or "  (no reviews available)"

    similar_str = "\n".join(
        f"  - {r['name']} ({r['cuisine']}) — similarity {r['similarity']:.2f}"
        for r in similar_emb[:3]
    ) or "  (embedding similarity unavailable)"

    prompt = f"""You are a restaurant recommendation assistant. Based on the context below, write a 2-3 sentence personalised explanation of why this restaurant is a good match for this reviewer. Be specific, warm, and focus on concrete similarities between their past preferences and this restaurant.

REVIEWER:
  Name: {reviewer.get('name', 'Unknown')}
  Local Guide: {reviewer.get('is_local_guide', False)}
  Total reviews written: {reviewer.get('total_reviews', 'unknown')}

RECOMMENDED RESTAURANT:
  Name: {restaurant.get('name', 'Unknown')}
  Cuisine: {restaurant.get('cuisine', 'Unknown')}
  Price level: {restaurant.get('price_level', 'unknown')} / 4
  Rating: {restaurant.get('rating', 'unknown')}★

REVIEWER'S PAST FAVOURITES (highest-rated):
{history_str}

POPULAR DISHES AT THIS RESTAURANT:
  {dishes_str}

WHAT OTHER REVIEWERS SAY:
{reviews_str}

SIMILAR RESTAURANTS THE REVIEWER MAY ALSO LIKE:
{similar_str}

Write the explanation now (2-3 sentences, no bullet points, no preamble):"""

    return prompt


# ── Public API ─────────────────────────────────────────────────────────────────
def explain(
    contributor_id: str,
    place_id: str,
    backend: str = "claude",
    local_url: str = "http://localhost:8082",
    local_model: str = "qwen3.5-9b",
) -> str:
    """
    Generate a natural-language explanation for recommending place_id to contributor_id.

    Args:
        backend:     "claude" (Anthropic API) or "local" (vLLM/Ollama endpoint)
        local_url:   Base URL of local server (used when backend="local")
        local_model: Model name served at local_url

    Returns a 2-3 sentence string.
    """
    ctx         = get_neighbourhood(contributor_id, place_id)
    top_reviews = get_top_reviews(place_id)
    similar_emb = similar_restaurants_by_embedding(place_id)

    prompt      = _build_prompt(contributor_id, place_id, ctx, top_reviews, similar_emb)
    explanation = _call_llm(prompt, backend, local_url, local_model)
    return explanation


def explain_batch(
    pairs: list[tuple[str, str]],
    backend: str = "claude",
    local_url: str = "http://localhost:8082",
    local_model: str = "qwen3.5-9b",
) -> list[dict]:
    """Explain a list of (contributor_id, place_id) pairs. Returns list of dicts."""
    results = []
    for contributor_id, place_id in pairs:
        log.info(f"Explaining {contributor_id} → {place_id}")
        explanation = explain(contributor_id, place_id, backend, local_url, local_model)
        results.append({
            "contributor_id": contributor_id,
            "place_id":       place_id,
            "explanation":    explanation,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="GraphRAG recommendation explanation")
    parser.add_argument("contributor_id", help="Reviewer contributor ID")
    parser.add_argument("place_id",       help="Restaurant place ID")
    parser.add_argument("--backend", choices=["claude", "local"], default="claude",
                        help="LLM backend: 'claude' (Anthropic API) or 'local' (vLLM/Ollama)")
    parser.add_argument("--local-url", default="http://localhost:8082",
                        help="Local server URL (used when --backend=local)")
    parser.add_argument("--local-model", default="qwen3.5-9b",
                        help="Model name at local server (used when --backend=local)")
    parser.add_argument("--context-only", action="store_true",
                        help="Print extracted graph context without calling any LLM")
    args = parser.parse_args()

    if args.context_only:
        ctx = get_neighbourhood(args.contributor_id, args.place_id)
        reviews = get_top_reviews(args.place_id)
        similar = similar_restaurants_by_embedding(args.place_id)
        print(json.dumps({"context": ctx, "top_reviews": reviews,
                          "similar_by_embedding": similar}, indent=2, default=str))
    else:
        result = explain(
            args.contributor_id, args.place_id,
            backend=args.backend,
            local_url=args.local_url,
            local_model=args.local_model,
        )
        print(f"\nRecommendation explanation:\n{result}\n")


if __name__ == "__main__":
    main()
