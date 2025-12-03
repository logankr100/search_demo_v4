"""
Flask API server for semantic search engine.
Wraps 05_search.py functionality with REST endpoints.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import json
import time
import re
import unicodedata
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

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


# ==================== LAYERED EXTRACTION SYSTEM ====================

# Unicode fraction mapping
UNICODE_FRAC = {"½":"1/2","¼":"1/4","¾":"3/4","⅛":"1/8","⅜":"3/8","⅝":"5/8","⅞":"7/8"}

def _to_ascii_fracs(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKD", text)
    for u, r in UNICODE_FRAC.items():
        t = t.replace(u, r)
    return t

# Pattern for mixed fractions: 1-1/2, 1 1/2
MIXED_FRAC = r'\d+\s*[-]\s*\d+/\d+'
# Pattern for simple fractions: 1/2, 3/4, etc.
SIMPLE_FRAC = r'\d+/\d+'
# Pattern for decimal numbers: 1.5, 2.25, etc.
DECIMAL = r'\d+(?:\.\d+)?'
# Combined number pattern
NUMBER_PATTERN = rf'(?:{MIXED_FRAC}|{SIMPLE_FRAC}|{DECIMAL})'

# Unit patterns
LENGTH_UNITS = r'(?:in|inch|inches|"|mm|millimeter|millimeters|cm|centimeter|centimeters|ft|foot|feet|\'|m|meter|meters)'
PRESSURE_UNITS = r'(?:psi|psig|psia|bar|kpa|mpa)'
WEIGHT_UNITS = r'(?:lb|lbs|pound|pounds|kg|kilogram|kilograms|g|gram|grams)'
ALL_UNITS = rf'(?:{LENGTH_UNITS}|{PRESSURE_UNITS}|{WEIGHT_UNITS})'

# Dimension patterns with context
DIMENSION_PATTERN = re.compile(
    rf'({NUMBER_PATTERN})\s*({LENGTH_UNITS})(?:\b|$)',
    re.IGNORECASE
)

# Thread patterns
METRIC_THREAD = re.compile(r'\bM(\d+(?:\.\d+)?)(?:\s*[xX×]\s*(\d+(?:\.\d+)?))?\b', re.I)
IMPERIAL_THREAD = re.compile(r'\b(?:#?)?(\d+(?:/\d+)?)-(\d{2,3})(?:\s*(UNC|UNF|UN))?\b', re.I)
NPT_THREAD = re.compile(rf'\b({NUMBER_PATTERN})\s*["\'"]?\s*NPT\b', re.I)

# Context to field mapping
CONTEXT_TO_FIELD = {
    'wheel': 'wheel_diameter',
    'caster': 'wheel_diameter', 
    'base': 'base_diameter',
    'mount': 'mount_hole_diameter',
    'mounting': 'mounting_hole_diameter',
    'hole': 'hole_diameter',
    'body': 'body_diameter',
    'flange': 'flange_diameter',
    'handle': 'handle_diameter',
    'length': 'length',
    'height': 'height', 
    'width': 'width',
    'depth': 'depth',
    'thick': 'thickness',
    'thickness': 'thickness',
    'diameter': 'diameter',
    'dia': 'diameter',
    'thread': 'thread_size',
    'weight': 'weight',
    'size': 'size'
}

def parse_mixed_fraction(frac_str: str) -> Optional[float]:
    """Parse mixed fractions like '1-1/2' or '1 1/2' into float."""
    frac_str = frac_str.strip()
    try:
        if '-' in frac_str and '/' in frac_str:
            whole_part, frac_part = frac_str.split('-', 1)
            whole = int(whole_part.strip())
            if '/' in frac_part:
                num, den = frac_part.split('/', 1)
                return float(whole + int(num.strip()) / int(den.strip()))
        elif ' ' in frac_str and '/' in frac_str:
            parts = frac_str.split()
            if len(parts) == 2:
                whole = int(parts[0])
                if '/' in parts[1]:
                    num, den = parts[1].split('/', 1)
                    return float(whole + int(num) / int(den))
        elif '/' in frac_str:
            num, den = frac_str.split('/', 1)
            return float(int(num.strip()) / int(den.strip()))
        else:
            return float(frac_str)
    except Exception:
        return None
    return None

def convert_to_canonical_server(family: Optional[str], value: float, unit: Optional[str]) -> Optional[float]:
    """Convert units to canonical form (inches, psi, lb, etc.)."""
    if value is None:
        return None
    u = (unit or "").strip().lower()

    if family == "length":
        if u in {"", None, '"', "in", "inch", "inches"}: return float(value)
        if u in {"mm", "millimeter", "millimeters"}:     return float(value) / 25.4
        if u in {"cm", "centimeter", "centimeters"}:     return float(value) / 2.54
        if u in {"ft", "foot", "feet", "'"}:             return float(value) * 12.0
        return None

    if family == "pressure":
        if u in {"", None, "psi", "psig", "psia"}:       return float(value)
        if u == "bar":                                   return float(value) * 14.5037738
        if u == "kpa":                                   return float(value) * 0.145037738
        if u == "mpa":                                   return float(value) * 145.037738
        return None

    if family in {"mass", "weight"}:
        if u in {"", None, "lb", "lbs", "pound", "pounds"}: return float(value)
        if u in {"kg", "kilogram", "kilograms"}:            return float(value) * 2.20462262
        if u in {"g", "gram", "grams"}:                     return float(value) * 0.00220462262
        return None

    # Unknown family: allow unitless only
    if u in {"", None}: return float(value)
    return None

def find_context_field(query: str, match_pos: Tuple[int, int], available_fields: set) -> Optional[str]:
    """Find the most likely field based on context words around a numeric match."""
    start, end = match_pos
    # Look in a window around the match
    window_start = max(0, start - 50)
    window_end = min(len(query), end + 50)
    context = query[window_start:window_end].lower()
    
    # Score potential fields based on context keywords
    field_scores = {}
    
    for keyword, field in CONTEXT_TO_FIELD.items():
        if field in available_fields and keyword in context:
            # Closer keywords get higher scores
            keyword_pos = context.find(keyword)
            if keyword_pos != -1:
                distance = abs(keyword_pos - (start - window_start))
                field_scores[field] = 1.0 / (1.0 + distance / 10.0)
    
    if field_scores:
        return max(field_scores.items(), key=lambda x: x[1])[0]
    
    return None

def extract_query_constraints_server(query: str, schema_attrs: List[str]) -> List[Dict[str, Any]]:
    """Extract constraints from query using layered approach."""
    constraints = []
    processed_spans = []
    available_fields = set(schema_attrs)
    # Also include known text fields for thread constraints  
    text_fields = {'thread_size', 'thread', 'thread_type', 'threading_type'}
    
    # Layer 1: Thread patterns (highest priority)
    for pattern, thread_type in [(METRIC_THREAD, 'metric'), (IMPERIAL_THREAD, 'imperial'), (NPT_THREAD, 'npt')]:
        for match in pattern.finditer(query):
            span = (match.start(), match.end())
            if any(overlap_spans(span, existing) for existing in processed_spans):
                continue
                
            if thread_type == 'metric':
                diameter = match.group(1)
                pitch = match.group(2) if match.group(2) else None
                if pitch:
                    thread_str = f"M{diameter}x{pitch}"
                else:
                    thread_str = f"M{diameter}"
            elif thread_type == 'imperial':
                size = match.group(1)
                tpi = match.group(2)
                std = match.group(3) if len(match.groups()) > 2 and match.group(3) else None
                if std:
                    thread_str = f"{size}-{tpi} {std}"
                else:
                    thread_str = f"{size}-{tpi}"
            else:  # NPT
                size = match.group(1)
                thread_str = f"{size} NPT"
            
            if 'thread_size' in available_fields or 'thread_size' in text_fields:
                constraints.append({
                    'field': 'thread_size',
                    'value': thread_str,
                    'tolerance': 0.0,  # Exact match for text
                    'match_type': 'regex_thread'
                })
                processed_spans.append(span)
    
    # Layer 2: Dimension patterns
    for match in DIMENSION_PATTERN.finditer(query):
        span = (match.start(), match.end())
        if any(overlap_spans(span, existing) for existing in processed_spans):
            continue
            
        value_str, unit = match.groups()
        value = parse_mixed_fraction(value_str)
        
        if value is not None:
            canonical_value = convert_to_canonical_server('length', value, unit)
            if canonical_value is not None:
                # Look for context to determine field
                context_field = find_context_field(query, span, available_fields)
                
                if context_field and context_field in available_fields:
                    # Use more generous tolerance for better matches
                    tolerance = 0.25 if canonical_value > 1.0 else 0.125  # 1/4" for larger dimensions
                    constraints.append({
                        'field': context_field,
                        'value': canonical_value,
                        'tolerance': tolerance,
                        'match_type': 'regex_context'
                    })
                    processed_spans.append(span)
    
    return constraints

def overlap_spans(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
    """Check if two spans overlap."""
    s1, e1 = span1
    s2, e2 = span2
    return not (e1 <= s2 or s1 >= e2)


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

        # Extract constraints from query using new layered approach
        t1_5 = time.time()
        extracted_constraints = extract_query_constraints_server(query_text, data["numeric_schema"]["attrs"])
        # Merge with manually provided constraints
        all_constraints = constraints + extracted_constraints
        timings["constraint_extraction"] = time.time() - t1_5

        # Build numeric shortlist
        t2 = time.time()
        shortlist_indices, matched_constraints = build_numeric_shortlist(
            all_constraints,
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

        # Compute semantic scores for shortlist (or all products if shortlist is empty)
        t4 = time.time()
        if len(shortlist_indices) == 0:
            # No products match constraints - fall back to semantic search of ALL products
            print("[INFO] No products match constraints, falling back to semantic search of all products")
            shortlist_indices = np.arange(len(data["meta"]))  # Use all products
            # Note: We'll still report constraints in debug, but won't restrict the search

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
            all_constraints, shortlist_indices, data["numeric_values"], data["numeric_schema"]
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
                "extracted_constraints": extracted_constraints,
                "constraints": matched_constraints,
                "shortlist_size": 0 if len(matched_constraints) > 0 and all(c.get('count', 0) == 0 for c in matched_constraints if c.get('matched')) else len(shortlist_indices),
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
