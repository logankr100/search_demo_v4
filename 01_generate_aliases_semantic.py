#!/usr/bin/env python3
"""
01_generate_aliases_semantic.py
--------------------------------
Conservative, auditable alias & schema builder for spec keys.

Pipeline (safe-by-default):
  1) Load raw_keys.json from Part 1 (00_harvest_raw_keys.py).
  2) Build case/space-insensitive seed groups (Brand vs brand).
  3) Create context strings: "{key} :: e.g. {example}".
  4) Embed contexts (default TF-IDF; optional OpenAI embeddings).
  5) Generate candidate merges using nearest neighbors + lexical+semantic+compat checks.
  6) Write AUDIT artifacts (no merges applied by default).
  7) If you provide approvals.jsonl and --apply, apply only approved merges.
  8) Emit:
       - fields.semantic.json  (canonical fields w/ aliases)
       - alias_map.semantic.json (flat alias -> canonical_id)

Usage (dry-run; propose only):
  python 01_generate_aliases_semantic.py \
    --input raw_keys.json \
    --out-fields fields.semantic.json \
    --out-alias alias_map.semantic.json \
    --audit-dir audit \
    --sem-thresh 0.90 \
    --lex-thresh 0.80 \
    --neighbors-k 10

Apply approved merges:
  python 01_generate_aliases_semantic.py \
    --input raw_keys.json \
    --out-fields fields.semantic.json \
    --out-alias alias_map.semantic.json \
    --audit-dir audit \
    --approvals audit/approvals.jsonl \
    --apply

Optional: OpenAI embeddings (set OPENAI_API_KEY) and/or LLM naming:
  --embed-backend openai --embed-model text-embedding-3-small
  --llm --llm-model gpt-4o-mini

Dependencies:
  - pandas
  - rapidfuzz (optional; falls back to difflib)
  - scikit-learn (for TF-IDF; install if using default backend)
  - openai (optional; only if you pass --embed-backend openai or --llm)

"""

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterable

# Optional libs
try:
    from rapidfuzz import fuzz
    HAVE_RAPID = True
except Exception:
    import difflib
    HAVE_RAPID = False

try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    HAVE_PANDAS = False

# TF-IDF fallback embeddings
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAVE_SK = True
except Exception:
    HAVE_SK = False

# Optional OpenAI (embeddings / LLM)
try:
    from openai import OpenAI
    HAVE_OPENAI = True
except Exception:
    HAVE_OPENAI = False


# ---------------------- Helpers & Normalization ----------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def norm_case_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def token_set_ratio(a: str, b: str) -> float:
    if HAVE_RAPID:
        return fuzz.token_set_ratio(a, b) / 100.0
    # difflib fallback
    return difflib.SequenceMatcher(None, a, b).ratio()

def snakeify(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", norm_case_space(s)).strip("_")
    s = re.sub(r"_+", "_", s)
    return s or "field"

def title_label(snake_id: str) -> str:
    return " ".join(w.capitalize() for w in snake_id.split("_"))

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

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


# ---------------------- Unit / Pattern Hints ----------------------

UNIT_LEXICON = {
    "length":  [r'\b(?:in|inch|inches|mm|millimeter|millimeters|cm|centimeter|centimeters|ft|feet|")\b', r"\bm\b(?!.?p[a-z])"],
    "pressure":[r'\b(?:psi|psig|psia|bar|kpa|mpa)\b'],
    "voltage": [r'\b(?:v|volt|volts)\b'],
    "current": [r'\b(?:a|amp|amps|ampere|amperes)\b'],
    "frequency":[r'\b(?:hz|hertz)\b'],
    "temperature":[r'\b(?:°?f|degf|°?c|degc)\b'],
    "mass":    [r'\b(?:lb|lbs|pound|pounds|kg|g|gram|grams|kilogram|kilograms)\b'],
    "power":   [r'\b(?:w|watt|watts|kw|kilowatt|kilowatts|hp|horsepower)\b'],
    "flow":    [r'\b(?:gpm|lpm|cfm|scfm)\b'],
    "thread":  [r'\b(?:npt|bsp|unc|unf|un|tpi|m\d+(?:\.\d+)?)\b', r'\b\d+\s*\/\s*\d+\s*-\s*\d+\b', r'\bm\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?\b'],
}

RE_NUM = re.compile(r"\d")
RE_THREAD_1 = re.compile(r"\b(npt|bsp|unc|unf|un|tpi|m\d+(?:\.\d+)?)\b", re.I)
RE_THREAD_2 = re.compile(r"\b\d+\s*\/\s*\d+\s*-\s*\d+\b", re.I)
RE_THREAD_3 = re.compile(r"\bm\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?\b", re.I)

RESERVED = {"brand", "mpn", "sku", "manufacturers_part_number"}

def infer_numeric_and_family(example: str) -> Tuple[bool, Optional[str]]:
    """
    Very simple inference from example text: numeric-ish and dominant unit family.
    """
    if not example:
        return False, None
    txt = example.lower()
    is_num = bool(RE_NUM.search(txt))
    fam_votes = {}
    for fam, pats in UNIT_LEXICON.items():
        for p in pats:
            if re.search(p, txt, re.I):
                fam_votes[fam] = fam_votes.get(fam, 0) + 1
    family = max(fam_votes.items(), key=lambda kv: kv[1])[0] if fam_votes else None
    return is_num, family

def looks_like_thread(example: str) -> bool:
    if not example:
        return False
    t = example.lower()
    return bool(RE_THREAD_1.search(t) or RE_THREAD_2.search(t) or RE_THREAD_3.search(t))


# ---------------------- Data Structures ----------------------

@dataclass
class RawKey:
    key: str
    count: int
    example: str

@dataclass
class Group:
    # A "group" is a list of RawKeys that are case/space-equal (seed) or merged-by-approval.
    members: List[RawKey]


# ---------------------- Embedding Backends ----------------------

def embed_tfidf(texts: List[str]):
    if not HAVE_SK:
        raise SystemExit("[error] scikit-learn not installed; install it or use --embed-backend openai")
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 3))
    X = vec.fit_transform(texts)
    # We'll compute cosine similarities with sklearn later
    return {"X": X, "vec": vec}

def cosine_topk(X, idx: int, k: int) -> List[Tuple[int, float]]:
    sims = cosine_similarity(X[idx], X).ravel()
    order = sims.argsort()[::-1]
    out = []
    for j in order:
        if j == idx:
            continue
        out.append((int(j), float(sims[j])))
        if len(out) >= k:
            break
    return out

def embed_openai(texts: List[str], model: str):
    if not HAVE_OPENAI:
        raise SystemExit("[error] openai package not installed; pip install openai OR use TF-IDF backend")
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=texts)
    V = [[d.embedding[i] for i in range(len(resp.data[0].embedding))] for d in resp.data]
    # Normalize
    import numpy as np
    V = np.array(V, dtype="float32")
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return V

def cosine_topk_dense(V, idx: int, k: int) -> List[Tuple[int, float]]:
    # V must be row-normalized
    import numpy as np
    sims = (V[idx] @ V.T).astype("float32")
    order = np.argsort(-sims)
    out = []
    for j in order:
        if j == idx:
            continue
        out.append((int(j), float(sims[j])))
        if len(out) >= k:
            break
    return out


# ---------------------- Core Logic ----------------------

def load_raw_keys(path: Path) -> List[RawKey]:
    data = json.loads(path.read_text())
    items = data.get("keys", [])
    out = []
    for it in items:
        out.append(RawKey(key=str(it["key"]), count=int(it.get("count", 0) or 0), example=str(it.get("example", ""))))
    return out

def seed_groups_casefold(raw_keys: List[RawKey]) -> List[Group]:
    buckets: Dict[str, List[RawKey]] = {}
    for rk in raw_keys:
        sig = norm_case_space(rk.key)
        buckets.setdefault(sig, []).append(rk)
    return [Group(members=v) for v in buckets.values()]

def build_contexts(groups: List[Group]) -> Tuple[List[str], List[str]]:
    # centroid text (representative) + per-member pretty string
    centroids = []
    members = []
    for g in groups:
        # choose most frequent member as centroid text
        best = max(g.members, key=lambda r: r.count)
        ct = f"{best.key} :: e.g. {best.example}" if best.example else f"{best.key}"
        centroids.append(ct)
        for r in g.members:
            members.append(f"{r.key} :: e.g. {r.example}" if r.example else r.key)
    return centroids, members

def group_signature_text(g: Group) -> str:
    # clean signature used in lexical comparison
    best = max(g.members, key=lambda r: r.count)
    return norm_case_space(best.key)

def lexical_score(g1: Group, g2: Group) -> float:
    a = group_signature_text(g1)
    b = group_signature_text(g2)
    return token_set_ratio(a, b)

def infer_group_type_family(g: Group) -> Tuple[str, Optional[str], bool]:
    """
    Returns (type_str, unit_family, is_thread_pattern)
      type_str: "numeric" or "string" (we keep it simple)
    """
    num_votes = 0
    fam_votes: Dict[str, int] = {}
    thread_flag = False
    for r in g.members:
        is_num, fam = infer_numeric_and_family(r.example)
        if is_num:
            num_votes += 1
        if fam:
            fam_votes[fam] = fam_votes.get(fam, 0) + 1
        if looks_like_thread(r.example):
            thread_flag = True
    type_str = "numeric" if num_votes > (len(g.members) / 2) else "string"
    family = max(fam_votes.items(), key=lambda kv: kv[1])[0] if fam_votes else None
    return type_str, family, thread_flag

def reserved_kind(g: Group) -> Optional[str]:
    # If group looks like a reserved kind (brand, mpn, sku...), return canonical id hint
    key_text = group_signature_text(g)
    sid = snakeify(key_text)
    if sid in RESERVED:
        return sid
    return None

def build_neighbor_index(centroid_texts: List[str], backend: str, embed_model: str):
    """
    Returns a tuple:
      ("tfidf", {"X": X})      with cosine_topk(...) to query
      or
      ("openai", V)            with cosine_topk_dense(...)
    """
    if backend == "openai":
        V = embed_openai(centroid_texts, embed_model)
        return ("openai", V)
    else:
        emb = embed_tfidf(centroid_texts)
        return ("tfidf", emb)

def topk_neighbors(emb_tuple, idx: int, k: int) -> List[Tuple[int, float]]:
    kind, data = emb_tuple
    if kind == "openai":
        return cosine_topk_dense(data, idx, k)
    else:
        return cosine_topk(data["X"], idx, k)

def candidate_merges(groups: List[Group],
                     emb_tuple,
                     neighbors_k: int,
                     sem_thresh: float,
                     lex_thresh: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    """
    Generate and score candidate merges (no application).
    Returns: (candidates_jsonl_rows, blocked_jsonl_rows, orphan_indices)
    """
    cands = []
    blocked = []
    orphan_idx = set(range(len(groups)))  # everyone is orphan until a valid candidate references them

    # Precompute group-level hints
    g_meta = []
    for gi, g in enumerate(groups):
        g_type, g_fam, g_thread = infer_group_type_family(g)
        g_meta.append({"type": g_type, "family": g_fam, "thread": g_thread, "reserved": reserved_kind(g)})

    for i, g in enumerate(groups):
        nn = topk_neighbors(emb_tuple, i, neighbors_k)
        for j, sim in nn:
            if j <= i:
                continue  # directional
            g1, g2 = groups[i], groups[j]
            meta1, meta2 = g_meta[i], g_meta[j]

            lex = lexical_score(g1, g2)

            # Compatibility checks (veto-first)
            reasons = {
                "sem_ok": sim >= sem_thresh,
                "lex_ok": lex >= lex_thresh,
                "type_compatible": (meta1["type"] == meta2["type"]),
                "family_compatible": (meta1["family"] == meta2["family"]) or (meta1["family"] is None) or (meta2["family"] is None),
                "thread_agree": (meta1["thread"] == meta2["thread"]),
                "reserved_compatible": not ((meta1["reserved"] and meta2["reserved"]) and (meta1["reserved"] != meta2["reserved"]))
            }

            decision = "candidate" if all(reasons.values()) else "blocked"

            record = {
                "left_idx": i,
                "right_idx": j,
                "left_keys": [r.key for r in g1.members],
                "right_keys": [r.key for r in g2.members],
                "left_count": sum(r.count for r in g1.members),
                "right_count": sum(r.count for r in g2.members),
                "lex_score": round(lex, 3),
                "sem_score": round(sim, 3),
                "left_meta": meta1,
                "right_meta": meta2,
                "decision": decision,
            }

            if decision == "candidate":
                cands.append(record)
                orphan_idx.discard(i)
                orphan_idx.discard(j)
            else:
                blocked.append(record)

    return cands, blocked, sorted(list(orphan_idx))

def apply_approvals(groups: List[Group],
                    candidates: List[Dict[str, Any]],
                    approvals_path: Optional[Path]) -> List[Group]:
    """
    Apply only approved merges. If approvals_path is None, return groups as-is.
    approvals.jsonl rows should look like:
      {"left_idx": 1, "right_idx": 7, "approve": true}
    Indexes refer to the *original* groups ordering.
    """
    if approvals_path is None or not approvals_path.exists():
        return groups

    approvals: Dict[Tuple[int, int], bool] = {}
    with approvals_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                a = int(obj["left_idx"]); b = int(obj["right_idx"])
                approvals[(min(a, b), max(a, b))] = bool(obj.get("approve", False))
            except Exception:
                continue

    # Build a union-find to merge approved pairs
    parent = list(range(len(groups)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for cand in candidates:
        i, j = cand["left_idx"], cand["right_idx"]
        key = (min(i, j), max(i, j))
        if approvals.get(key, False):
            union(i, j)

    # build merged groups
    buckets: Dict[int, List[RawKey]] = {}
    for idx, g in enumerate(groups):
        r = find(idx)
        buckets.setdefault(r, []).extend(g.members)

    merged = [Group(members=v) for v in buckets.values()]
    return merged


# ---------------------- Field Building ----------------------

def pick_canonical_id(g: Group) -> Tuple[str, str, int, str]:
    # Canonical text = highest-count key
    counts: Dict[str, int] = {}
    example_for: Dict[str, str] = {}
    for r in g.members:
        counts[r.key] = counts.get(r.key, 0) + r.count
        if r.key not in example_for and r.example:
            example_for[r.key] = r.example
    best_text = max(counts.items(), key=lambda kv: kv[1])[0]
    cid = snakeify(best_text)
    label = title_label(cid)
    total = sum(counts.values())
    example = example_for.get(best_text, next(iter(example_for.values()), ""))
    return cid, label, total, example

def build_fields(groups: List[Group]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    fields = []
    alias_map: Dict[str, str] = {}

    for g in groups:
        cid, label, total_count, example = pick_canonical_id(g)
        typ, fam, thread_flag = infer_group_type_family(g)

        alias_counts: Dict[str, int] = {}
        for r in g.members:
            alias_counts[r.key] = alias_counts.get(r.key, 0) + r.count

        # trivial confidence: relative frequency; we’re not fabricating new aliases here
        best_count = max(alias_counts.values()) if alias_counts else 1
        aliases = []
        for a, cnt in sorted(alias_counts.items(), key=lambda kv: (-kv[1], kv[0].lower())):
            conf = round(min(1.0, cnt / best_count), 3)
            aliases.append({"text": a, "count": int(cnt), "confidence": conf})
            alias_map[a] = cid

        fields.append({
            "id": cid,
            "label": label,
            "type": typ,
            "unit_family": ("thread" if thread_flag else fam),
            "count": int(total_count),
            "example": example,
            "aliases": aliases
        })

    fields.sort(key=lambda f: -f["count"])
    return fields, alias_map


# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser(description="Conservative, auditable alias/schema generator for spec keys.")
    ap.add_argument("--input", default="raw_keys.json", help="Path to raw_keys.json")
    ap.add_argument("--out-fields", default="fields.semantic.json", help="Output canonical fields JSON")
    ap.add_argument("--out-alias", default="alias_map.semantic.json", help="Output alias map JSON")

    # Embeddings
    ap.add_argument("--embed-backend", choices=["tfidf", "openai"], default="tfidf", help="Embedding backend")
    ap.add_argument("--embed-model", default="text-embedding-3-small", help="OpenAI embedding model if using openai backend")

    # Candidate generation & thresholds
    ap.add_argument("--neighbors-k", type=int, default=10, help="Top-k nearest neighbors to consider per group")
    ap.add_argument("--sem-thresh", type=float, default=0.90, help="Semantic cosine similarity threshold (high on purpose)")
    ap.add_argument("--lex-thresh", type=float, default=0.80, help="Lexical token-set similarity threshold (high)")

    # Audit / approvals
    ap.add_argument("--audit-dir", default="audit", help="Directory to write audit artifacts")
    ap.add_argument("--approvals", default=None, help="Path to approvals.jsonl")
    ap.add_argument("--apply", action="store_true", help="Apply approvals and merge groups")
    # (LLM hooks are stubs for later; we keep script runnable without them)
    ap.add_argument("--llm", action="store_true", help="(Optional) Use LLM to suggest nicer ids/labels/aliases")
    ap.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model, if --llm")
    args = ap.parse_args()

    # Load input
    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"[error] Input not found: {in_path}")
    raw_keys = load_raw_keys(in_path)
    if not raw_keys:
        sys.exit("[error] No keys found in input. Run 00_harvest_raw_keys.py first.")

    print(f"[info] loaded {len(raw_keys)} raw keys from {args.input}")

    # Seed groups by case/space only (safe)
    groups = seed_groups_casefold(raw_keys)
    print(f"[info] seed groups (case/space): {len(groups)}")

    # Build centroid contexts and embed
    centroid_texts, _ = build_contexts(groups)
    emb_tuple = build_neighbor_index(centroid_texts, args.embed_backend, args.embed_model)

    # Propose candidates (no merges yet)
    cands, blocked, orphans = candidate_merges(
        groups, emb_tuple, args.neighbors_k, args.sem_thresh, args.lex_thresh
    )

    # Audit outputs
    audit_dir = Path(args.audit_dir)
    safe_mkdir(audit_dir)

    # clusters_seed.jsonl
    seed_rows = []
    for i, g in enumerate(groups):
        seed_rows.append({
            "idx": i,
            "keys": [r.key for r in g.members],
            "total_count": sum(r.count for r in g.members),
            "type_family": infer_group_type_family(g)
        })
    write_jsonl(audit_dir / "clusters_seed.jsonl", seed_rows)

    # neighbors.csv (for quick per-group eyeballing)
    neighbor_rows = []
    for i in range(len(groups)):
        nn = topk_neighbors(emb_tuple, i, args.neighbors_k)
        for j, sim in nn:
            neighbor_rows.append({
                "left_idx": int(i),
                "right_idx": int(j),
                "sem_score": round(sim, 3),
                "left_sig": group_signature_text(groups[i]),
                "right_sig": group_signature_text(groups[j]),
            })
    write_csv(audit_dir / "neighbors.csv", neighbor_rows, ["left_idx", "right_idx", "sem_score", "left_sig", "right_sig"])

    # candidates.jsonl & blocked.jsonl & orphans.jsonl
    write_jsonl(audit_dir / "candidates.jsonl", cands)
    write_jsonl(audit_dir / "blocked.jsonl", blocked)
    write_jsonl(audit_dir / "orphans.jsonl", [{"idx": i, "keys": [r.key for r in groups[i].members]} for i in orphans])

    print(f"[audit] {len(cands)} candidates, {len(blocked)} blocked, {len(orphans)} orphans")
    print(f"[audit] wrote: {audit_dir}/candidates.jsonl, blocked.jsonl, orphans.jsonl, clusters_seed.jsonl, neighbors.csv")

    # If not applying merges, build fields directly from seed groups (super safe)
    final_groups = groups
    if args.apply:
        final_groups = apply_approvals(groups, cands, Path(args.approvals) if args.approvals else None)
        print(f"[apply] final groups after approvals: {len(final_groups)}")
    else:
        print("[info] not applying merges (default). To apply, provide --approvals ... --apply")

    # Build outputs (no LLM by default; aliases are just the raw keys in each final group)
    fields, alias_map = build_fields(final_groups)

    write_json(Path(args.out_fields), {
        "version": 1,
        "generated_at": now_iso(),
        "source": {"embeddings": args.embed_backend, "llm": bool(args.llm)},
        "fields": fields
    })
    write_json(Path(args.out_alias), {
        "version": 1,
        "generated_at": now_iso(),
        "aliases": alias_map
    })

    print(f"[ok] wrote {args.out_fields} ({len(fields)} fields)")
    print(f"[ok] wrote {args.out_alias} ({len(alias_map)} aliases)")

if __name__ == "__main__":
    main()