"""
Flask API server for semantic search engine.
Wraps 05_search.py functionality with REST endpoints.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import json
import time
from pathlib import Path
import numpy as np

# Add parent directory to path to import search modules
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global cache for loaded data
_cache = {}

def load_search_data(index_dir="index_out_v2"):
    """Load all necessary data for search operations."""
    if "loaded" in _cache:
        return _cache

    # Get path relative to project root (parent of api/)
    project_root = Path(__file__).parent.parent
    index_path = project_root / index_dir

    # Load metadata
    meta = []
    with open(index_path / "meta.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))

    # Load vectors
    vectors_title = np.load(index_path / "vectors_title.npy")
    vectors_desc = np.load(index_path / "vectors_desc.npy")
    vectors_specs = np.load(index_path / "vectors_specs.npy")

    # Load numeric data
    numeric_data = np.load(index_path / "numeric_specs.npz")
    numeric_values = numeric_data["values"]
    numeric_mask = numeric_data["mask"]

    with open(index_path / "numeric_schema.json", "r") as f:
        numeric_schema = json.load(f)

    # Load alias map
    with open(index_path / "alias_map.enriched.json", "r") as f:
        alias_map_data = json.load(f)
        alias_map = alias_map_data["aliases"]

    # Load fields with enriched data
    with open(index_path / "fields.enriched.json", "r") as f:
        fields_data = json.load(f)

    _cache.update({
        "loaded": True,
        "meta": meta,
        "vectors_title": vectors_title,
        "vectors_desc": vectors_desc,
        "vectors_specs": vectors_specs,
        "numeric_values": numeric_values,
        "numeric_mask": numeric_mask,
        "numeric_schema": numeric_schema,
        "alias_map": alias_map,
        "fields": fields_data["fields"]
    })

    return _cache


def get_embedding(text, model="text-embedding-3-small"):
    """Get OpenAI embedding for text."""
    from openai import OpenAI
    import os
    from dotenv import load_dotenv

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.embeddings.create(
        model=model,
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def extract_object_query(query_text):
    """Extract noun phrases from query (simplified)."""
    # Simple heuristic: remove numbers, units, and common modifiers
    import re
    tokens = query_text.lower().split()

    # Remove numeric tokens and units
    noun_tokens = []
    for token in tokens:
        if re.match(r'^[\d\./\-]+$', token):
            continue
        if token in ['in', 'inch', 'inches', 'mm', 'cm', 'psi', 'lb', 'lbs']:
            continue
        noun_tokens.append(token)

    return ' '.join(noun_tokens) if noun_tokens else query_text


def detect_aliases(query_text, alias_map):
    """Detect which aliases appear in the query."""
    query_lower = query_text.lower()
    detected = []

    for alias_text, field_id in alias_map.items():
        if alias_text in query_lower:
            detected.append({
                "alias": alias_text,
                "field": field_id,
                "position": query_lower.index(alias_text)
            })

    # Sort by position in query
    detected.sort(key=lambda x: x["position"])
    return detected


def build_numeric_shortlist(constraints, numeric_values, numeric_mask, numeric_schema):
    """Build shortlist of products matching numeric constraints."""
    if not constraints:
        return np.arange(len(numeric_values)), []

    n_products = len(numeric_values)
    shortlist = np.ones(n_products, dtype=bool)

    matched_constraints = []

    for constraint in constraints:
        field_id = constraint["field"]
        target_value = constraint["value"]
        tolerance = constraint.get("tolerance", 0.01)  # Default small tolerance

        # Find field index in schema
        if field_id not in numeric_schema["attrs"]:
            matched_constraints.append({
                **constraint,
                "matched": False,
                "reason": "Field not in schema"
            })
            continue

        field_idx = numeric_schema["attrs"].index(field_id)
        field_values = numeric_values[:, field_idx]
        field_mask = numeric_mask[:, field_idx]

        # Products with this field present
        has_field = field_mask

        # Within tolerance
        lower = target_value - tolerance
        upper = target_value + tolerance
        in_range = (field_values >= lower) & (field_values <= upper)

        # Must have field AND be in range
        matches = has_field & in_range
        shortlist &= matches

        matched_constraints.append({
            **constraint,
            "matched": True,
            "count": int(np.sum(matches))
        })

    return np.where(shortlist)[0], matched_constraints


def compute_semantic_scores(query_vec, product_indices, vectors, weight):
    """Compute cosine similarity scores."""
    if len(product_indices) == 0:
        return np.array([])

    # Normalize query vector
    query_norm = query_vec / np.linalg.norm(query_vec)

    # Get product vectors
    product_vecs = vectors[product_indices]

    # Normalize product vectors
    product_norms = np.linalg.norm(product_vecs, axis=1, keepdims=True)
    product_vecs_norm = product_vecs / (product_norms + 1e-10)

    # Cosine similarity
    similarities = product_vecs_norm @ query_norm

    return similarities * weight


def compute_numeric_boost(constraints, product_indices, numeric_values, numeric_schema):
    """Compute numeric proximity boost for products."""
    if not constraints or len(product_indices) == 0:
        return np.zeros(len(product_indices))

    boosts = np.zeros(len(product_indices))

    for constraint in constraints:
        field_id = constraint["field"]
        target_value = constraint["value"]

        if field_id not in numeric_schema["attrs"]:
            continue

        field_idx = numeric_schema["attrs"].index(field_id)
        field_values = numeric_values[product_indices, field_idx]

        # Exponential decay based on distance
        distances = np.abs(field_values - target_value)
        # Scale factor for decay (adjust as needed)
        scale = target_value * 0.1 if target_value > 0 else 1.0
        decay = np.exp(-distances / (scale + 1e-6))

        boosts += decay

    # Average boost across constraints
    boosts /= len(constraints)

    return boosts


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Get field schema, aliases, and other metadata."""
    try:
        data = load_search_data()

        return jsonify({
            "fields": data["fields"],
            "aliases": data["alias_map"],
            "numeric_schema": {
                "attrs": data["numeric_schema"]["attrs"],
                "families": data["numeric_schema"]["families"]
            },
            "product_count": len(data["meta"])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/search', methods=['POST'])
def search():
    """Execute semantic search with detailed results."""
    try:
        params = request.json
        query_text = params.get("query", "")

        if not query_text:
            return jsonify({"error": "Query text required"}), 400

        # Get parameters with defaults
        w_title = params.get("w_title", 0.35)
        w_desc = params.get("w_desc", 0.50)
        w_specs = params.get("w_specs", 0.15)
        w_full = params.get("w_full", 0.95)
        w_obj = params.get("w_obj", 0.05)
        w_num = params.get("w_num", 0.25)
        k = params.get("k", 20)
        constraints = params.get("constraints", [])

        # Load data
        data = load_search_data()

        # Start timing
        timings = {}
        t0 = time.time()

        # Extract object query
        object_query = extract_object_query(query_text)
        timings["object_extraction"] = time.time() - t0

        # Detect aliases
        t1 = time.time()
        alias_hits = detect_aliases(query_text, data["alias_map"])
        timings["alias_detection"] = time.time() - t1

        # Build numeric shortlist
        t2 = time.time()
        shortlist_indices, matched_constraints = build_numeric_shortlist(
            constraints,
            data["numeric_values"],
            data["numeric_mask"],
            data["numeric_schema"]
        )
        timings["shortlist_build"] = time.time() - t2

        # Get embeddings
        t3 = time.time()
        query_vec_full = get_embedding(query_text)
        query_vec_obj = get_embedding(object_query) if object_query != query_text else query_vec_full
        timings["embedding"] = time.time() - t3

        # Compute semantic scores for shortlist
        t4 = time.time()
        if len(shortlist_indices) == 0:
            # No products match constraints
            return jsonify({
                "results": [],
                "debug": {
                    "query": query_text,
                    "object_query": object_query,
                    "alias_hits": alias_hits,
                    "constraints": matched_constraints,
                    "shortlist_size": 0,
                    "timings": timings
                }
            })

        # Compute scores for each channel
        scores_title_full = compute_semantic_scores(
            query_vec_full, shortlist_indices, data["vectors_title"], w_title
        )
        scores_desc_full = compute_semantic_scores(
            query_vec_full, shortlist_indices, data["vectors_desc"], w_desc
        )
        scores_specs_full = compute_semantic_scores(
            query_vec_full, shortlist_indices, data["vectors_specs"], w_specs
        )

        scores_title_obj = compute_semantic_scores(
            query_vec_obj, shortlist_indices, data["vectors_title"], w_title
        )
        scores_desc_obj = compute_semantic_scores(
            query_vec_obj, shortlist_indices, data["vectors_desc"], w_desc
        )
        scores_specs_obj = compute_semantic_scores(
            query_vec_obj, shortlist_indices, data["vectors_specs"], w_specs
        )

        # Combine full and object queries
        semantic_scores = (
            w_full * (scores_title_full + scores_desc_full + scores_specs_full) +
            w_obj * (scores_title_obj + scores_desc_obj + scores_specs_obj)
        )

        timings["semantic_scoring"] = time.time() - t4

        # Compute numeric boost
        t5 = time.time()
        numeric_boosts = compute_numeric_boost(
            constraints, shortlist_indices, data["numeric_values"], data["numeric_schema"]
        )
        timings["numeric_boost"] = time.time() - t5

        # Final scores
        final_scores = semantic_scores + w_num * numeric_boosts

        # Get top K
        top_k_idx = np.argsort(-final_scores)[:k]
        top_k_product_idx = shortlist_indices[top_k_idx]

        # Build results
        results = []
        for i, prod_idx in enumerate(top_k_product_idx):
            meta = data["meta"][prod_idx]
            shortlist_idx = top_k_idx[i]

            results.append({
                "rank": i + 1,
                "score": float(final_scores[shortlist_idx]),
                "semantic_score": float(semantic_scores[shortlist_idx]),
                "numeric_boost": float(numeric_boosts[shortlist_idx]),
                "channel_scores": {
                    "title_full": float(scores_title_full[shortlist_idx]),
                    "desc_full": float(scores_desc_full[shortlist_idx]),
                    "specs_full": float(scores_specs_full[shortlist_idx]),
                    "title_obj": float(scores_title_obj[shortlist_idx]),
                    "desc_obj": float(scores_desc_obj[shortlist_idx]),
                    "specs_obj": float(scores_specs_obj[shortlist_idx])
                },
                "product": {
                    "sku": meta["sku"],
                    "name": meta["name"],
                    "brand": meta.get("brand", ""),
                    "mpn": meta.get("mpn", ""),
                    "url": meta.get("url", ""),
                    "description": meta.get("raw_desc", "")[:300],
                    "title": meta.get("raw_title", "")[:300],
                    "specs": meta.get("raw_specs", "")[:500]
                }
            })

        timings["total"] = time.time() - t0

        return jsonify({
            "results": results,
            "debug": {
                "query": query_text,
                "object_query": object_query,
                "alias_hits": alias_hits,
                "constraints": matched_constraints,
                "shortlist_size": len(shortlist_indices),
                "candidate_count": len(shortlist_indices),
                "weights": {
                    "title": w_title,
                    "desc": w_desc,
                    "specs": w_specs,
                    "full": w_full,
                    "obj": w_obj,
                    "numeric": w_num
                },
                "timings": timings
            }
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


if __name__ == '__main__':
    # Run on port 5001 (port 5000 is used by macOS AirPlay Receiver)
    app.run(host='0.0.0.0', port=5001, debug=False)
