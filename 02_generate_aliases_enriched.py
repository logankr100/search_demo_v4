#!/usr/bin/env python3
"""
02_generate_aliases_enriched.py
--------------------------------
Precise alias enrichment for canonical fields (Step 3).

Design principles:
  - Less is more: accept only exact / industry-standard synonyms.
  - Never merges or alters field IDs; only adds aliases per field.
  - Centroid = mean embedding of [label + existing aliases] (NO examples).
  - Hard gating + semantic threshold + lexical family allowlists.
  - Full audit pack (suggestions, scored, rejected, collisions, coverage).

Inputs:
  - fields.semantic.json (from Step 2)

Outputs:
  - fields.enriched.json
  - alias_map.enriched.json
  - audit_aliases/
      suggestions.jsonl   (raw heuristic + LLM proposals)
      scored.jsonl        (per-candidate scoring breakdown)
      rejected.jsonl      (rejections with reasons)
      collisions.jsonl    (alias proposed for multiple fields)
      coverage.csv        (summary counts per field)
      prompts.jsonl       (if --llm; the exact prompts used)

Backends:
  - Embeddings: TF-IDF (default, local) or OpenAI (set --embed-backend openai)
  - LLM proposals: OpenAI (optional; enable with --llm)

Environment:
  - If using OpenAI, set OPENAI_API_KEY in your environment or .env
"""

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --------------------- Optional libs ---------------------
try:
    from rapidfuzz import fuzz
    HAVE_RAPID = True
except Exception:
    import difflib
    HAVE_RAPID = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAVE_SK = True
except Exception:
    HAVE_SK = False

try:
    from openai import OpenAI
    HAVE_OPENAI = True
except Exception:
    HAVE_OPENAI = False

try:
    from dotenv import load_dotenv
    HAVE_DOTENV = True
except Exception:
    HAVE_DOTENV = False

# --------------------- Utilities ---------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def norm_case_space(s: str) -> str:
    return norm_space(s).lower()

def token_set_ratio(a: str, b: str) -> float:
    if HAVE_RAPID:
        return fuzz.token_set_ratio(a, b) / 100.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def write_json(path: Path, obj: Any):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path: Path, rows: List[Dict[str, Any]], header: List[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# --------------------- Domain Lexicons ---------------------

RESERVED_IDS = {"brand", "mpn", "sku", "manufacturers_part_number"}

GENERIC_BAD = {
    "size", "type", "rating", "model", "info", "data", "spec", "specs"
}

FAMILY_TOKENS = {
    # Used to check lexical compatibility; keep conservative & short.
    "length": {"length", "overall", "total", "long", "l"},
    "pressure": {"pressure", "psig", "psi", "bar", "kpa"},
    "voltage": {"volt", "voltage", "v"},
    "current": {"amp", "amps", "ampere", "current", "a"},
    "frequency": {"hz", "hertz", "frequency"},
    "mass": {"weight", "wt", "lb", "lbs", "kg", "mass"},
    "power": {"watt", "kw", "power", "hp"},
    "flow": {"gpm", "lpm", "cfm", "scfm", "flow"},
    "thread": {"thread", "threaded", "pitch", "tpi", "unc", "unf", "un", "npt", "metric"},
    # Diameter families handled more explicitly below:
}

# Special diameter tokens (outside/inside)
OD_TOKENS = {"od", "o.d.", "outside", "outer", "outside dia", "outer diameter", "outside diameter", "ø outer", "ø outside"}
ID_TOKENS = {"id", "i.d.", "inside", "inner", "inside dia", "inner diameter", "inside diameter", "ø inner", "ø inside"}
DIA_TOKENS = {"diameter", "dia", "ø"}

# Regex helpers
RE_HAS_DIGIT = re.compile(r"\d")
RE_PUNY = re.compile(r"^[^\w]+$")

# --------------------- Embeddings ---------------------

class Embedder:
    def __init__(self, backend: str, openai_model: str):
        self.backend = backend
        self.openai_model = openai_model
        self._tfidf_vec = None  # for tfidf
        self._tfidf_X = None
        self._openai_V = None
        self._openai_client = None

    def fit(self, texts: List[str]):
        if self.backend == "openai":
            if not HAVE_OPENAI:
                raise SystemExit("[error] openai not installed, pip install openai OR use --embed-backend tfidf")
            # Load API key
            if HAVE_DOTENV:
                load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise SystemExit("[error] OPENAI_API_KEY not set (env or .env)")
            self._openai_client = OpenAI(api_key=api_key)
            self._openai_V = self._embed_openai(texts)
        else:
            if not HAVE_SK:
                raise SystemExit("[error] scikit-learn not installed; pip install scikit-learn OR use --embed-backend openai")
            vec = TfidfVectorizer(min_df=1, ngram_range=(1, 3))
            X = vec.fit_transform(texts)
            self._tfidf_vec = vec
            self._tfidf_X = X

    def transform(self, texts: List[str]):
        if self.backend == "openai":
            # Return normalized dense vectors
            return self._embed_openai(texts)
        else:
            return self._tfidf_vec.transform(texts)

    def cosine(self, a, b) -> float:
        if self.backend == "openai":
            # a, b are dense normed vectors
            import numpy as np
            return float((a @ b.T).item()) if a.ndim == 2 else float(a.dot(b))
        else:
            # a, b are sparse row vectors
            sim = cosine_similarity(a, b).ravel()[0]
            return float(sim)

    def mean_vec(self, mats):
        if self.backend == "openai":
            import numpy as np
            M = np.vstack(mats)
            # Vectors are already normalized; mean then renormalize
            mean = M.mean(axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-12)
            return mean
        else:
            # Sparse mean (scikit-learn CSR)
            from scipy.sparse import vstack
            import numpy as np
            if not isinstance(mats, list) or len(mats) == 0:
                return None
            stacked = vstack(mats)
            mean = stacked.mean(axis=0)
            # cosine on sparse is fine as-is
            return mean

    def _embed_openai(self, texts: List[str]):
        # Returns row-normalized vectors (numpy)
        resp = self._openai_client.embeddings.create(model=self.openai_model, input=texts)
        import numpy as np
        V = np.array([d.embedding for d in resp.data], dtype="float32")
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        return V

# --------------------- LLM (optional) ---------------------

LLM_PROMPT_TEMPLATE = """You are helping build a strict synonym list for a structured industrial specification database.

The goal: propose only exact or industry-standard synonyms/shorthand for the SAME specification field. Do NOT include related-but-different fields, values, product names, or marketing terms. Prefer short aliases (<= 3 words). Return a JSON array of strings only.

Field:
- ID: {fid}
- Label: {label}
- Type: {ftype}
- UnitFamily: {ufam}
- ExistingAliases: {aliases}
- ExampleValues (for context only; do not copy): {examples}

Rules:
- Only include aliases that have exactly the same meaning as the label.
- Reject generic terms like "size", "type", "rating", "model".
- Do not include units-only tokens (like "mm", "in"), or numeric-looking aliases.
- Keep to standard industry phrasing and abbreviations.
- 5 or fewer suggestions.

Respond with JSON like: ["Alias 1", "Alias 2"].
"""

def llm_suggest_aliases(openai_client, model: str, field_payload: Dict[str, Any]) -> List[str]:
    """
    Returns a list of alias candidates from the LLM. Empty list on any failure.
    """
    prompt = LLM_PROMPT_TEMPLATE.format(
        fid=field_payload["id"],
        label=field_payload["label"],
        ftype=field_payload["type"],
        ufam=field_payload.get("unit_family") or "None",
        aliases=field_payload.get("existing_aliases") or [],
        examples=field_payload.get("examples") or []
    )
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.choices[0].message.content.strip()
        # Extract JSON array robustly
        start = text.find("["); end = text.rfind("]")
        if start == -1 or end == -1:
            return []
        arr = json.loads(text[start:end+1])
        out = []
        for s in arr:
            if isinstance(s, str) and s.strip():
                out.append(s.strip())
        return out
    except Exception:
        return []

# --------------------- Heuristic Seeds (minimal & conservative) ---------------------

HEURISTIC_SEEDS_BY_ID = {
    # Keep very short and safe. These are exact/standard synonyms only.
    "outside_diameter": ["OD", "Outside Dia", "Outer Diameter"],
    "inside_diameter":  ["ID", "Inside Dia", "Inner Diameter"],
    "thread_size":      ["Thread Size", "Threaded Size", "Thread Pitch"],  # Note: 'Pitch' alone later gated by patterns
    "manufacturers_part_number": ["Manufacturer Part Number", "Mfr Part Number", "MFG P/N"],
    "mpn": ["Mfr Part Number", "Manufacturer Part Number", "MFG P/N"],
    "brand": ["Manufacturer", "Make"],
    "sku": ["Item Number", "Part Number", "Part #"],  # Not MPN
}

# --------------------- Gating / Scoring ---------------------

def is_generic_bad(alias: str) -> bool:
    low = norm_case_space(alias)
    toks = set(re.findall(r"[a-z0-9]+", low))
    return bool(toks & GENERIC_BAD)

def reserved_conflict(field_id: str, alias: str) -> bool:
    low = norm_case_space(alias)
    # forbid cross-bleed among reserved concepts
    if field_id == "brand" and ("supplier" in low or "vendor" in low or "seller" in low):
        return True
    if field_id in {"mpn", "manufacturers_part_number"} and ("sku" in low or "upc" in low or "ean" in low):
        return True
    if field_id == "sku" and ("mpn" in low or "brand" in low or "manufacturer" in low):
        return True
    return False

def looks_like_value(alias: str) -> bool:
    # aliases should be labels, not values: reject numeric-heavy or unit-ish alone
    s = alias.strip()
    if len(s) > 32:
        return True
    if RE_HAS_DIGIT.search(s):
        # Heuristic: if digits present and not typical abbreviations like OD/ID, reject
        low = norm_case_space(s)
        if not (low in {"od", "id"} or low.startswith("ø")):
            return True
    if RE_PUNY.match(s):
        return True
    return False

def lexical_allowed(alias: str, field_id: str, unit_family: Optional[str]) -> bool:
    low = norm_case_space(alias)
    toks = set(re.findall(r"[a-z]+", low))

    # OD/ID special handling
    if field_id.endswith("outside_diameter") or "outside_diameter" == field_id:
        if toks & (OD_TOKENS | DIA_TOKENS):
            return True
        return False
    if field_id.endswith("inside_diameter") or "inside_diameter" == field_id:
        if toks & (ID_TOKENS | DIA_TOKENS):
            return True
        return False
    if "diameter" in field_id:
        if toks & DIA_TOKENS:
            return True
        return False

    # General families
    if unit_family and unit_family in FAMILY_TOKENS:
        if toks & FAMILY_TOKENS[unit_family]:
            return True

    # Thread family nuance
    if field_id.startswith("thread") or (unit_family == "thread"):
        if toks & FAMILY_TOKENS["thread"]:
            return True

    # Brand/mpn/sku are allowed broadly (veto rules handle conflicts)
    if field_id in {"brand", "mpn", "manufacturers_part_number", "sku"}:
        return True

    # Fallback: require at least one overlapping token with canonical id tokens
    base_toks = set(re.findall(r"[a-z]+", field_id))
    return bool(toks & base_toks)

def score_alias(embedder: Embedder, label_vec, centroid_vec, alias_text: str,
                field_id: str, unit_family: Optional[str]) -> Tuple[float, Dict[str, float], List[str]]:
    """
    Returns:
      confidence, subscores dict, reasons (list of strings describing rejections/penalties)
    """
    reasons = []

    # Hard rejects first
    if is_generic_bad(alias_text):
        reasons.append("generic_bad")
        return 0.0, {}, reasons
    if reserved_conflict(field_id, alias_text):
        reasons.append("reserved_conflict")
        return 0.0, {}, reasons
    if looks_like_value(alias_text):
        reasons.append("looks_like_value")
        return 0.0, {}, reasons
    if not lexical_allowed(alias_text, field_id, unit_family):
        reasons.append("lexical_not_allowed")
        return 0.0, {}, reasons

    # Semantic similarity (dual-path: label-only and centroid (label+existing aliases))
    alias_vec = embedder.transform([alias_text])
    sim_label = embedder.cosine(alias_vec, label_vec)
    sim_cent = embedder.cosine(alias_vec, centroid_vec)
    semantic_sim = max(sim_label, sim_cent)

    # Family pattern bonus (small)
    pattern_bonus = 0.0
    low = norm_case_space(alias_text)
    toks = set(re.findall(r"[a-z]+", low))
    if unit_family == "thread" or field_id.startswith("thread"):
        if {"thread", "threaded", "pitch", "tpi", "unc", "unf", "npt", "metric"} & toks:
            pattern_bonus += 0.05
        # Do NOT add 'pitch' alone unless thread tokens present in canonical id
        if "pitch" in toks and "thread" not in field_id:
            reasons.append("pitch_no_thread_context")  # informational only

    if unit_family == "voltage":
        if {"volt", "voltage", "v"} & toks:
            pattern_bonus += 0.05

    if unit_family == "length":
        if {"length", "overall", "total"} & toks:
            pattern_bonus += 0.03

    # Final confidence (precision-first)
    # We heavily weight semantic similarity.
    confidence = 0.8 * semantic_sim + 0.2 * pattern_bonus

    subs = {
        "semantic": round(semantic_sim, 3),
        "pattern_bonus": round(pattern_bonus, 3),
        "sim_label": round(sim_label, 3),
        "sim_centroid": round(sim_cent, 3),
    }

    return float(confidence), subs, reasons

# --------------------- Loading & Centroids ---------------------

def load_fields(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    fields = data.get("fields", [])
    # Normalize structure
    for f in fields:
        f.setdefault("aliases", [])
        f.setdefault("type", "string")
        f.setdefault("unit_family", None)
        f.setdefault("example", "")
        f.setdefault("count", 0)
    return fields

def build_centroid_corpus(fields: List[Dict[str, Any]]) -> List[str]:
    """
    Build the corpus of texts to fit the embedder: all labels + all existing aliases.
    """
    corpus = []
    for f in fields:
        label = f.get("label") or f.get("id")
        corpus.append(norm_space(label))
        for a in f.get("aliases", []):
            t = a.get("text") if isinstance(a, dict) else str(a)
            if t:
                corpus.append(norm_space(t))
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for s in corpus:
        k = norm_case_space(s)
        if k not in seen:
            uniq.append(s)
            seen.add(k)
    return uniq

def field_label_and_centroid(embedder: Embedder, field: Dict[str, Any]):
    """
    Returns (label_vec, centroid_vec) for the field, both in the embedder space.
    Centroid is mean of [label + existing aliases], NO examples.
    """
    # Label-only
    label_text = norm_space(field.get("label") or field.get("id"))
    label_vec = embedder.transform([label_text])

    # Centroid: label + existing aliases
    texts = [label_text]
    for a in field.get("aliases", []):
        t = a.get("text") if isinstance(a, dict) else str(a)
        if t and t.strip():
            texts.append(norm_space(t))
    mats = [embedder.transform([t]) for t in texts]
    centroid_vec = embedder.mean_vec(mats)
    return label_vec, centroid_vec

# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Precise alias enrichment (strict, audit-first).")
    ap.add_argument("--input", default="fields.semantic.json", help="Input fields JSON (from Step 2)")
    ap.add_argument("--out-fields", default="fields.enriched.json", help="Output enriched fields JSON")
    ap.add_argument("--out-alias", default="alias_map.enriched.json", help="Output flat alias map JSON")
    ap.add_argument("--audit-dir", default="audit_aliases", help="Directory for audit artifacts")

    # Backends
    ap.add_argument("--embed-backend", choices=["tfidf", "openai"], default="openai",
                    help="Embedding backend for semantic filtering (default: openai)")
    ap.add_argument("--embed-model", default="text-embedding-3-small", help="OpenAI embedding model")
    ap.add_argument("--llm", action="store_true", help="Enable LLM alias proposals (default off)")
    ap.add_argument("--llm-model", default="gpt-4o-mini", help="OpenAI chat model for alias proposals")

    # Strictness knobs
    ap.add_argument("--min-conf", type=float, default=0.60, help="Minimum confidence to keep a new alias")
    ap.add_argument("--min-sem", type=float, default=0.90, help="Minimum semantic similarity (strict)")
    ap.add_argument("--max-aliases", type=int, default=5, help="Maximum total aliases to keep per field (including existing)")
    ap.add_argument("--dry-run", action="store_true", help="Write audit only; do not change outputs")
    ap.add_argument("--only", default=None, help="Regex to include only matching field ids")
    ap.add_argument("--exclude", default=None, help="Regex to exclude field ids")
    ap.add_argument("--print", dest="print_field", default=None, help="Print debug info for a specific field id")

    args = ap.parse_args()

    # Load fields
    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"[error] Input not found: {in_path}")
    fields = load_fields(in_path)
    print(f"[info] loaded {len(fields)} fields from {args.input}")

    # Optional filters
    import re as _re
    if args.only:
        pat = _re.compile(args.only)
        fields = [f for f in fields if pat.search(f.get("id") or "")]
    if args.exclude:
        patx = _re.compile(args.exclude)
        fields = [f for f in fields if not patx.search(f.get("id") or "")]
    if not fields:
        print("[warn] no fields after filters; exiting")
        return

    # Fit embedder on label+aliases corpus (NOT examples)
    corpus = build_centroid_corpus(fields)
    embedder = Embedder(args.embed_backend, args.embed_model)
    embedder.fit(corpus)

    # Prep OpenAI client if needed
    openai_client = None
    if args.llm:
        if HAVE_DOTENV:
            load_dotenv()
        if not HAVE_OPENAI:
            raise SystemExit("[error] openai not installed; pip install openai or run without --llm")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("[error] OPENAI_API_KEY not set (env or .env)")
        openai_client = OpenAI(api_key=api_key)

    # Audit setup
    audit_dir = Path(args.audit_dir)
    safe_mkdir(audit_dir)
    suggestions_rows = []
    scored_rows = []
    rejected_rows = []
    collisions_rows = []
    prompts_rows = []

    # Build enriched aliases
    out_fields = []
    flat_alias_map: Dict[str, str] = {}
    alias_owner: Dict[str, str] = {}  # alias -> field_id (to detect collisions)

    # Deterministic ordering
    fields_sorted = sorted(fields, key=lambda f: (f.get("id") or ""))

    for f in fields_sorted:
        fid = f.get("id")
        label = f.get("label") or fid
        ftype = f.get("type", "string")
        ufam = f.get("unit_family")
        examples = [f.get("example")] if f.get("example") else []
        existing_aliases = []
        for a in f.get("aliases", []):
            t = a.get("text") if isinstance(a, dict) else str(a)
            if t:
                existing_aliases.append(norm_space(t))

        # Get label & centroid vectors
        label_vec, centroid_vec = field_label_and_centroid(embedder, f)

        # Heuristic seeds (very small & safe)
        seeds = HEURISTIC_SEEDS_BY_ID.get(fid, [])
        # Always include canonical label as an alias candidate (ensures it stays)
        seeds = list(dict.fromkeys(seeds))  # dedupe keep order

        # LLM proposals (optional)
        llm_props = []
        if args.llm:
            payload = {
                "id": fid,
                "label": label,
                "type": ftype,
                "unit_family": ufam,
                "existing_aliases": existing_aliases[:5],
                "examples": examples[:3]
            }
            props = llm_suggest_aliases(openai_client, args.llm_model, payload)
            llm_props = [p for p in props if isinstance(p, str) and p.strip()]
            # Save prompt for audit
            prompts_rows.append({"id": fid, "prompt": LLM_PROMPT_TEMPLATE.format(
                fid=fid, label=label, ftype=ftype, ufam=ufam or "None",
                aliases=existing_aliases[:5], examples=examples[:3]
            ), "llm_model": args.llm_model})

        # Aggregate raw suggestions
        raw_candidates = []
        # Existing aliases always kept with confidence=1 later, but we don't rescore them here.
        # We only score NEW candidates.
        raw_candidates += [{"text": s, "source": "seed"} for s in seeds]
        raw_candidates += [{"text": s, "source": "llm"} for s in llm_props]

        # Audit: suggestions
        suggestions_rows.append({
            "id": fid,
            "label": label,
            "type": ftype,
            "unit_family": ufam,
            "existing_aliases": existing_aliases,
            "seeds": seeds,
            "llm": llm_props
        })

        # Score & gate candidates
        accepted_new = []
        seen_norm = {norm_case_space(a) for a in existing_aliases}  # avoid duping existing
        # Also avoid scoring a candidate that equals label/aliases already present.
        for cand in raw_candidates:
            alias_text = cand["text"]
            alias_norm = norm_case_space(alias_text)
            if not alias_text or alias_norm in seen_norm:
                continue
            conf, subs, reasons = score_alias(embedder, label_vec, centroid_vec, alias_text, fid, ufam)

            row = {
                "id": fid,
                "alias": alias_text,
                "source": cand["source"],
                "confidence": round(conf, 3),
                **subs
            }

            # Strict acceptance
            if subs.get("semantic", 0.0) >= args.min_sem and conf >= args.min_conf and not reasons:
                accepted_new.append({"text": alias_text, "confidence": round(conf, 3)})
                row["decision"] = "accept"
            else:
                row["decision"] = "reject"
                row["reasons"] = reasons
                rejected_rows.append(row)

            scored_rows.append(row)

        # Combine: existing (keep) + accepted_new
        final_aliases = []
        # Keep existing aliases at conf=1.0
        for a in existing_aliases:
            final_aliases.append({"text": a, "confidence": 1.0})
        # Add accepted new ones
        # De-dupe by norm text, keep highest confidence per alias
        best_by_norm: Dict[str, Dict[str, Any]] = {norm_case_space(x["text"]): x for x in final_aliases}
        for x in accepted_new:
            k = norm_case_space(x["text"])
            if k not in best_by_norm or x["confidence"] > best_by_norm[k].get("confidence", 0.0):
                best_by_norm[k] = x
        final_aliases = list(best_by_norm.values())

        # Sort by confidence desc, then text
        final_aliases.sort(key=lambda a: (-float(a.get("confidence", 0.0)), a["text"].lower()))
        # Cap total
        if len(final_aliases) > args.max_aliases:
            final_aliases = final_aliases[:args.max_aliases]

        # Build output field record
        out_f = {
            "id": fid,
            "label": label,
            "type": ftype,
            "unit_family": ufam,
            "example": f.get("example", ""),
            "count": f.get("count", 0),
            "aliases": final_aliases
        }
        out_fields.append(out_f)

        # Populate flat alias map & detect collisions (exact-text keys)
        for a in final_aliases:
            alias_text_exact = a["text"]
            # Skip adding exact canonical label if you prefer (but typically include it for symmetry):
            # we include it, it's harmless and useful for lookup.
            if alias_text_exact in alias_owner and alias_owner[alias_text_exact] != fid:
                # collision
                collisions_rows.append({
                    "alias": alias_text_exact,
                    "first_id": alias_owner[alias_text_exact],
                    "second_id": fid
                })
            else:
                alias_owner[alias_text_exact] = fid

    # Build final flat alias map after collisions audit
    for alias_text, owner_id in alias_owner.items():
        flat_alias_map[alias_text] = owner_id

    # Coverage summary
    coverage_rows = []
    for f in out_fields:
        coverage_rows.append({
            "id": f["id"],
            "label": f["label"],
            "kept_aliases": len(f["aliases"]),
            "avg_conf": round(sum(a.get("confidence", 0.0) for a in f["aliases"]) / max(1, len(f["aliases"])), 3)
        })

    # Write audits
    write_jsonl(Path(args.audit_dir) / "suggestions.jsonl", suggestions_rows)
    write_jsonl(Path(args.audit_dir) / "scored.jsonl", scored_rows)
    write_jsonl(Path(args.audit_dir) / "rejected.jsonl", rejected_rows)
    write_jsonl(Path(args.audit_dir) / "collisions.jsonl", collisions_rows)
    write_csv(Path(args.audit_dir) / "coverage.csv", coverage_rows, ["id", "label", "kept_aliases", "avg_conf"])
    if args.llm:
        write_jsonl(Path(args.audit_dir) / "prompts.jsonl", prompts_rows)

    # Write outputs
    enriched_fields_obj = {
        "version": 1,
        "generated_at": now_iso(),
        "source": {"embeddings": args.embed_backend, "llm": bool(args.llm)},
        "fields": out_fields
    }
    alias_map_obj = {
        "version": 1,
        "generated_at": now_iso(),
        "aliases": flat_alias_map
    }

    if args.dry_run:
        print("[info] --dry-run: wrote audits only; skipping outputs")
        return

    write_json(Path(args.out_fields), enriched_fields_obj)
    write_json(Path(args.out_alias), alias_map_obj)
    print(f"[ok] wrote {args.out_fields} ({len(out_fields)} fields)")
    print(f"[ok] wrote {args.out_alias} ({len(flat_alias_map)} aliases)")

if __name__ == "__main__":
    main()