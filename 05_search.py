#!/usr/bin/env python3
"""
05_search.py — v3 (aliases + τ, numeric shortlist/boost + 3-channel semantic + SKU)

Inputs under --index-dir:
  - meta.jsonl
  - vectors_title.npy / vectors_desc.npy / vectors_specs.npy
  - [optional] index_title.faiss / index_desc.faiss / index_specs.faiss
  - numeric_specs.npz      (values float32 [N,M], mask bool [N,M])
  - numeric_schema.json    {"attrs":[...], "families":{field_id: family}}
  - fields.enriched.json   {"fields":[{id,label,type,unit_family,aliases:[...]}...]}
  - alias_map.enriched.json {"aliases": {"wheel dia":"wheel_diameter", ...}}
  - spec_patterns.py       (classify_value_relaxed/strict, TOLERANCES)

Features:
  - Alias-based hard-spec extraction (longest match, conservative)
  - Strict scalar parsing near alias hits; unit normalization; τ tolerances
  - Numeric shortlist (AND across constraints)
  - 3-channel (title/desc/specs) retrieval + dual-query (full/object)
  - Numeric proximity boost (exp falloff with τ)
  - SKU fuzzy detection for quick lookup

Example:
  python 05_search.py \
    -q "M10 x 1.5 thread 2 in wheel dia" \
    --index-dir index_out_v2 \
    -k 8 \
    --w-title 0.35 --w-desc 0.45 --w-specs 0.20 \
    --w-full 0.90 --w-obj 0.10 \
    --w-num 0.35 \
    --semantic-pool 600 \
    --per-list 200 \
    --debug
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from difflib import get_close_matches
import unicodedata

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Optional FAISS
try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

# --------- spec parsing / τ tolerances (from your local module) ----------
try:
    # classify_value_relaxed is optional; we fallback to strict if absent
    from spec_patterns import classify_value_strict, TOLERANCES
    try:
        from spec_patterns import classify_value_relaxed
        HAVE_RELAXED = True
    except Exception:
        HAVE_RELAXED = False
except Exception as e:
    raise SystemExit(f"[error] failed to import spec_patterns: {e}")

# -------------------- Timing helpers --------------------

def tnow() -> float:
    return time.time()

def normalize_rows(A: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    return A / norms

# -------------------- OpenAI embeddings --------------------

def embed_texts_openai(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    resp = client.embeddings.create(model=model, input=texts)
    arr = np.array([d.embedding for d in resp.data], dtype="float32")
    return arr

# -------------------- SKU detection --------------------

def _to_ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s)

def normalize_sku(s: str) -> str:
    s = _to_ascii(s).strip().upper()
    return re.sub(r"[^A-Z0-9]", "", s)

def looks_like_sku(q: str) -> bool:
    qn = normalize_sku(q)
    return 4 <= len(qn) <= 24 and bool(re.match(r"^[A-Z0-9]+$", qn))

def search_by_sku_fuzzy(query: str, meta: List[dict], top_k: int, cutoff: float = 0.70) -> List[dict]:
    qn = normalize_sku(query)
    sku_map = {}
    normalized_skus = []
    for item in meta:
        raw = item.get("sku", "")
        norm = normalize_sku(raw)
        if norm:
            if norm not in sku_map:
                sku_map[norm] = item
                normalized_skus.append(norm)
    close = get_close_matches(qn, normalized_skus, n=top_k, cutoff=cutoff)
    return [sku_map[n] for n in close]

# -------------------- Object (noun-ish) extraction --------------------

STOP_SPLIT = re.compile(r"\b(that|which|with|for|to|compatible|fits|working|works|designed|supporting|supports)\b", re.I)

def extract_object_query(query: str) -> str:
    q = query.strip()
    head = STOP_SPLIT.split(q, maxsplit=1)[0].strip()
    head = re.sub(
        r"^(i\s+need\s+(a|an|the)\s+|i\s+need\s+|looking\s+for\s+(a|an|the)\s+|looking\s+for\s+|need\s+(a|an|the)\s+|need\s+|find\s+(a|an|the)\s+|find\s+)",
        "", head, flags=re.I
    ).strip()
    words = head.split()
    if len(words) > 3:
        head = " ".join(words[:3])
    return head if head else q

# -------------------- Unit normalization (query-time) --------------------

def convert_to_canonical(family: Optional[str], value: float, unit: Optional[str]) -> Optional[float]:
    """Mirror 04_build_numeric_matrix.py conversions; canonical: in, psi, lb, lb-ft, Hz, W, gpm, °F."""
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

    if family == "torque":
        if u in {"", None, "lb-ft", "lbft", "ft-lb", "ftlb"}: return float(value)
        if u in {"nm", "newton-meter", "newton-meters"}:      return float(value) * 0.737562149
        return None

    if family == "frequency":
        if u in {"", None, "hz", "hertz"}: return float(value)
        return None

    if family == "power":
        if u in {"", None, "w", "watt", "watts"}: return float(value)
        if u in {"kw", "kilowatt", "kilowatts"}:  return float(value) * 1000.0
        if u in {"hp", "horsepower"}:            return float(value) * 745.699872
        return None

    if family == "flow":
        if u in {"", None, "gpm"}: return float(value)
        if u in {"lpm"}:           return float(value) * 0.264172052
        # cfm ambiguous; reject
        return None

    if family == "temperature":
        if u in {"", None, "f", "°f", "degf"}: return float(value)
        if u in {"c", "°c", "degc"}:           return float(value) * 9.0/5.0 + 32.0
        return None

    # Unknown family: allow unitless only
    if u in {"", None}: return float(value)
    return None

def get_tau(field_id: str, family: Optional[str]) -> Tuple[float, bool]:
    """Return (tau, is_relative). If TOLERANCES has 'rel:x', set is_relative True."""
    v = TOLERANCES.get(field_id, TOLERANCES.get(family, TOLERANCES.get("default", 0.25)))
    if isinstance(v, str) and v.lower().startswith("rel:"):
        try:
            r = float(v.split(":", 1)[1])
            return r, True
        except Exception:
            return 0.25, False
    try:
        return float(v), False
    except Exception:
        return 0.25, False

# -------------------- Alias matching (longest, non-overlapping) --------------------

def build_alias_patterns(alias_map: Dict[str, str]) -> List[Tuple[re.Pattern, str, str]]:
    """
    Returns list of (compiled_regex, alias_text, field_id), sorted by alias length desc
    We allow flexible whitespace between tokens; word-ish boundaries when possible.
    """
    items = []
    for alias_text, fid in alias_map.items():
        a = alias_text.strip().lower()
        if not a:
            continue
        # Escape + allow flexible whitespace between tokens
        tokens = [re.escape(t) for t in re.split(r"\s+", a)]
        pat = r"\b" + r"\s+".join(tokens) + r"\b"
        try:
            cre = re.compile(pat, re.I)
            items.append((cre, a, fid))
        except Exception:
            # fallback minimal escape
            try:
                cre = re.compile(re.escape(a), re.I)
                items.append((cre, a, fid))
            except Exception:
                continue
    # Longest aliases first to encourage longest-match greedily
    items.sort(key=lambda x: -len(x[1]))
    return items

def find_alias_hits(query: str, patterns: List[Tuple[re.Pattern, str, str]]) -> List[Tuple[int,int,str]]:
    """
    Returns non-overlapping hits: list of (start, end, field_id)
    Greedy longest-first selection.
    """
    q = query
    hits = []
    for cre, alias, fid in patterns:
        for m in cre.finditer(q):
            s, e = m.start(), m.end()
            hits.append((s, e, fid, e - s))
    if not hits:
        return []
    hits.sort(key=lambda t: (-t[3], t[0]))
    chosen = []
    used = []
    for s, e, fid, _ in hits:
        overlap = False
        for (ss, ee, _) in chosen:
            if not (e <= ss or s >= ee):
                overlap = True
                break
        if not overlap:
            chosen.append((s, e, fid))
    # sort by start
    chosen.sort(key=lambda t: t[0])
    return chosen

# -------------------- Value classification near alias --------------------

def classify_nearby_scalar(query: str, span: Tuple[int,int], family_hint: Optional[str]) -> Optional[Tuple[float, Optional[str]]]:
    """
    Look around alias span (±24 chars) for a scalar value; return normalized (value, unit)
    Unit here is the raw unit token from classifier; conversion to canonical happens after.
    """
    L, R = span
    w = 24
    window = query[max(0, L - w): min(len(query), R + w)]
    # prefer relaxed if available, else strict
    if HAVE_RELAXED:
        cl = classify_value_relaxed(window)
    else:
        cl = classify_value_strict(window)
    if not cl or cl.get("kind") != "scalar":
        return None
    val = cl.get("value")
    unit = cl.get("unit")
    # We don't enforce family here yet; conversion step will validate
    return (float(val), (unit if unit else None))

# -------------------- FAISS / brute helpers --------------------

def load_faiss_or_none(path: Path):
    if HAVE_FAISS and path.exists():
        return faiss.read_index(str(path))
    return None

def faiss_search(index, qvec: np.ndarray, pull: int) -> Tuple[np.ndarray, np.ndarray]:
    D, I = index.search(qvec, pull)
    return D[0], I[0]

def brute_search(V_norm: np.ndarray, qvec_norm: np.ndarray, pull: int) -> Tuple[np.ndarray, np.ndarray]:
    sims = (qvec_norm @ V_norm.T)[0]
    idx = np.argsort(-sims)[:pull]
    return sims[idx], idx

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="Aliases + τ hard-spec shortlist + 3-channel semantic search")
    ap.add_argument("-q", "--query", required=True)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--index-dir", default="index_out_v2")
    ap.add_argument("--model", default="text-embedding-3-small")

    # channel weights
    ap.add_argument("--w-title", type=float, default=0.35)
    ap.add_argument("--w-desc",  type=float, default=0.50)
    ap.add_argument("--w-specs", type=float, default=0.15)

    # dual query weights
    ap.add_argument("--w-full", type=float, default=0.95)
    ap.add_argument("--w-obj",  type=float, default=0.05)

    # numeric boost
    ap.add_argument("--w-num",  type=float, default=0.25)

    # retrieval pulls / pool sizing
    ap.add_argument("--pull", type=int, default=100, help="Base pull per channel before filtering")
    ap.add_argument("--per-list", type=int, default=200, help="When no hard specs, take top-N per list before union")
    ap.add_argument("--semantic-pool", type=int, default=600, help="Max candidates to score when no hard specs (0=disable)")

    # SKU
    ap.add_argument("--sku-cutoff", type=float, default=0.70)

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    t_all0 = tnow()

    # Paths
    idxdir = Path(args.index_dir)
    meta_path = idxdir / "meta.jsonl"
    vt_path = idxdir / "vectors_title.npy"
    vd_path = idxdir / "vectors_desc.npy"
    vs_path = idxdir / "vectors_specs.npy"
    num_npz  = idxdir / "numeric_specs.npz"
    num_schema = idxdir / "numeric_schema.json"
    fields_path = idxdir / "fields.enriched.json"
    alias_path  = idxdir / "alias_map.enriched.json"

    # Required files sanity
    for p in [meta_path, vt_path, vd_path, vs_path, num_npz, num_schema, fields_path, alias_path]:
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    # Load meta
    meta: List[dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                meta.append(json.loads(line))
            except Exception:
                meta.append({})
    N = len(meta)

    # Load vectors
    Vt = np.load(str(vt_path)).astype("float32")
    Vd = np.load(str(vd_path)).astype("float32")
    Vs = np.load(str(vs_path)).astype("float32")
    if not (Vt.shape[0] == Vd.shape[0] == Vs.shape[0] == N):
        raise SystemExit("Vector rows do not match meta rows.")
    Vt_norm = normalize_rows(Vt); Vd_norm = normalize_rows(Vd); Vs_norm = normalize_rows(Vs)

    # Numeric specs & schema
    npz = np.load(str(num_npz))
    values = npz["values"]  # [N, M]
    mask   = npz["mask"]    # [N, M] bool
    schema = json.loads(num_schema.read_text())
    schema_attrs: List[str] = schema["attrs"]
    families_of: Dict[str, Optional[str]] = schema.get("families", {})
    COL_OF: Dict[str, int] = {fid: j for j, fid in enumerate(schema_attrs)}

    # Fields & alias map
    fields_obj = json.loads(fields_path.read_text())
    alias_obj  = json.loads(alias_path.read_text())
    alias_map: Dict[str, str] = alias_obj.get("aliases", {})
    alias_patterns = build_alias_patterns(alias_map)

    # FAISS indices (optional)
    It = load_faiss_or_none(idxdir / "index_title.faiss")
    Id = load_faiss_or_none(idxdir / "index_desc.faiss")
    Is = load_faiss_or_none(idxdir / "index_specs.faiss")

    # OpenAI client
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)

    # Query basics
    full_query = args.query.strip()
    obj_query  = extract_object_query(full_query)

    # SKU-first (if detected)
    sku_first = None
    if looks_like_sku(full_query):
        matches = search_by_sku_fuzzy(full_query, meta, top_k=1, cutoff=args.sku_cutoff)
        if matches:
            sku_first = matches[0]

    # ---------- Hard-spec extraction ----------
    t0 = tnow()
    hits = find_alias_hits(full_query, alias_patterns)  # [(s,e,fid),...]
    constraints: List[Dict[str, Any]] = []
    for (s, e, fid) in hits:
        family = families_of.get(fid)
        cls = classify_nearby_scalar(full_query, (s, e), family)
        if not cls:
            continue
        val_raw, unit_raw = cls
        vcanon = convert_to_canonical(family, val_raw, unit_raw)
        if vcanon is None:
            continue
        constraints.append({
            "field_id": fid,
            "family": family,
            "value": float(vcanon)
        })
    t1 = tnow()

    if args.debug:
        print(f"[debug] object='{obj_query}'")
        if hits:
            print("[debug] alias hits:")
            for (s,e,fid) in hits:
                print(f"  - [{s}:{e}] -> {fid!r} :: '{full_query[s:e]}'")
        if constraints:
            print("[debug] constraints (canonical units):")
            for c in constraints:
                tau, is_rel = get_tau(c["field_id"], c["family"])
                print(f"  - {c['field_id']} = {c['value']} (family={c['family']}, tau={tau}{' rel' if is_rel else ''})")
        else:
            print("[debug] no hard constraints extracted")
        print(f"[timing] extract hard-specs: {t1 - t0:.3f}s")

    # ---------- Shortlist ----------
    t2 = tnow()
    short_ids: Optional[np.ndarray] = None
    if constraints:
        keep = np.ones(values.shape[0], dtype=bool)
        for c in constraints:
            fid, v = c["field_id"], c["value"]
            j = COL_OF.get(fid)
            if j is None:
                continue
            tau, is_rel = get_tau(fid, c["family"])
            if is_rel:
                # relative tolerance (e.g., ±5% of target)
                tau_abs = max(1e-9, abs(v) * float(tau))
            else:
                tau_abs = float(tau)
            has = mask[:, j]
            diff = np.abs(values[:, j] - v)
            cond = has & (diff <= tau_abs)
            keep &= cond
        short_ids = np.nonzero(keep)[0]
    t3 = tnow()
    if args.debug:
        print(f"[timing] shortlist build: {t3 - t2:.3f}s | shortlist size: {len(short_ids) if short_ids is not None else 'semantic-only'}")

    # ---------- Embed queries ----------
    t4 = tnow()
    Q = embed_texts_openai(client, [full_query, obj_query], model=args.model)
    q_full = normalize_rows(Q[0:1, :])
    q_obj  = normalize_rows(Q[1:2, :])
    t5 = tnow()
    if args.debug:
        print(f"[timing] embed queries: {t5 - t4:.3f}s")

    # ---------- Retrieval per channel ----------
    base_pull = max(args.pull, args.k * 50)
    if short_ids is not None:
        base_pull = max(base_pull, len(short_ids) * 3)

    def ch_search(V_norm, idx_faiss):
        if idx_faiss is not None:
            Df, If = faiss_search(idx_faiss, q_full, base_pull)
            Do, Io = faiss_search(idx_faiss, q_obj,  base_pull)
        else:
            Df, If = brute_search(V_norm, q_full, base_pull)
            Do, Io = brute_search(V_norm, q_obj,  base_pull)
        return (Df, If), (Do, Io)

    t6 = tnow()
    (Dt_f, It_f), (Dt_o, It_o) = ch_search(Vt_norm, It)
    (Dd_f, Id_f), (Dd_o, Id_o) = ch_search(Vd_norm, Id)
    (Ds_f, Is_f), (Ds_o, Is_o) = ch_search(Vs_norm, Is)
    t7 = tnow()
    if args.debug:
        print(f"[timing] retrieval per-channel: {t7 - t6:.3f}s")

    # ---------- Candidate set ----------
    cand_ids = set()

    def add_ids(arr, limit=None):
        if limit is None: limit = len(arr)
        cand_ids.update(arr[:limit].tolist())

    if short_ids is None:
        # No hard constraints: deterministic union of top-N per list
        PER_LIST = max(args.per_list, args.k * 50)
        for arr in [It_f, It_o, Id_f, Id_o, Is_f, Is_o]:
            add_ids(arr, limit=PER_LIST)
        cand = list(cand_ids)
        if args.semantic_pool > 0 and len(cand) > args.semantic_pool:
            cand = cand[:args.semantic_pool]
    else:
        # With hard constraints: intersect union with short_ids
        short_set = set(short_ids.tolist())
        for arr in [It_f, It_o, Id_f, Id_o, Is_f, Is_o]:
            cand_ids.update(arr.tolist())
        cand = [i for i in cand_ids if i in short_set]
        if not cand:
            # if intersection empty, fallback to just short_ids
            cand = short_ids.tolist()

    if not cand:
        # last resort: take some from title_full
        cand = It_f[:max(args.k*10, 100)].tolist()

    # Build similarity dicts
    def d_of(ids, sims):
        return {int(i): float(s) for i, s in zip(ids, sims)}

    t8 = tnow()
    t_full = d_of(It_f, Dt_f); t_obj = d_of(It_o, Dt_o)
    d_full = d_of(Id_f, Dd_f); d_obj = d_of(Id_o, Dd_o)
    s_full = d_of(Is_f, Ds_f); s_obj = d_of(Is_o, Ds_o)

    # Semantic score
    def sem_score(i: int) -> float:
        sf = (args.w_title * t_full.get(i, 0.0) +
              args.w_desc  * d_full.get(i, 0.0) +
              args.w_specs * s_full.get(i, 0.0))
        so = (args.w_title * t_obj.get(i, 0.0) +
              args.w_desc  * d_obj.get(i, 0.0) +
              args.w_specs * s_obj.get(i, 0.0))
        return args.w_full * sf + args.w_obj * so

    # Numeric boost
    def num_boost(i: int) -> float:
        if not constraints:
            return 0.0
        scores = []
        for c in constraints:
            fid, v = c["field_id"], c["value"]
            j = COL_OF.get(fid)
            if j is None or not mask[i, j]:
                continue
            tau, is_rel = get_tau(fid, c["family"])
            tau_abs = (abs(v) * tau) if is_rel else tau
            tau_abs = float(tau_abs) if tau_abs else 1e-6
            x = values[i, j]
            s = float(np.exp(-abs(x - v) / tau_abs))
            scores.append(s)
        if not scores:
            return 0.0
        return min(scores)  # conjunctive min

    scored = []
    for i in cand:
        base = sem_score(i)
        nb = num_boost(i)
        final = base + args.w_num * nb
        scored.append((final, i))
    t9 = tnow()
    if args.debug:
        print(f"[timing] combine + numeric: {t9 - t8:.3f}s")

    scored.sort(key=lambda x: -x[0])

    # ---------- Print results ----------
    if sku_first:
        print(f'\n🔍 Detected SKU-style query: "{args.query}"\n')
        print(f"[SKU] {sku_first.get('sku','')} — {sku_first.get('name','')}")
        desc = sku_first.get("description","")
        if desc:
            print(f"     {desc[:200]}{'...' if len(desc)>200 else ''}\n")

    print(f'\n🔍 Top {args.k} matches for: "{args.query}"')
    print(f'    (object="{obj_query}", ch-weights: title={args.w_title}, desc={args.w_desc}, specs={args.w_specs}, '
          f'w_full={args.w_full}, w_obj={args.w_obj}, w_num={args.w_num})\n')

    shown = 0
    sku_norm = normalize_sku(sku_first["sku"]) if sku_first and sku_first.get("sku") else None
    for (score, idx) in scored:
        item = meta[idx] if 0 <= idx < len(meta) else {}
        if sku_norm and normalize_sku(item.get("sku","")) == sku_norm:
            continue
        print(f"[{shown+1}] Score: {score:.4f} | SKU: {item.get('sku','')}  | {item.get('name','')}")
        desc = item.get("description","")
        if desc:
            print(f"     {desc[:220]}{'...' if len(desc)>220 else ''}")
        print()
        shown += 1
        if shown >= args.k:
            break

    if args.debug:
        print(f"[timing] total: {tnow() - t_all0:.3f}s")


if __name__ == "__main__":
    main()