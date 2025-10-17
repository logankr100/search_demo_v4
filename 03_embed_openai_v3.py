#!/usr/bin/env python3
"""
03_embed_openai_v3.py

Embeds a Global Industrial–style CSV into three channels:
  - title:        product name (+ brand if present)
  - description:  description/meta/og fallbacks
  - specs:        flattened specs JSON + spec_* columns folded into key: value lines

Outputs to --out-dir:
  - meta.jsonl                         (row-aligned metadata)
  - vectors_title.npy                  (float32 [N, D])
  - vectors_desc.npy                   (float32 [N, D])
  - vectors_specs.npy                  (float32 [N, D])
  - (optional) index_title.faiss       (FAISS IP index over normalized vectors)
  - (optional) index_desc.faiss
  - (optional) index_specs.faiss

Example:
  python 03_embed_openai_v3.py \
    --csv globalindustrial_fasteners_full.csv \
    --out-dir index_out_v2 \
    --model text-embedding-3-small \
    --batch-size 128 \
    --build-faiss
"""

import os
import re
import json
import time
import argparse
from pathlib import Path

import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

# Optional progress + FAISS
try:
    from tqdm import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False

try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False


# ----------------------- IO helpers -----------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_json_dumps(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"

def to_str(x) -> str:
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    return str(x)


# ----------------------- Column helpers -----------------------

META_KEYS_TO_SKIP = {
    # marketing / repeated meta
    "price_jsonld","currency_jsonld","meta_description","og_description",
    # identifiers duplicated elsewhere
    "sku","mpn","brand","manufacturers part number","manufacturers_part_number","manufacturers-part-number",
    "spec_sku","spec_mpn","spec_brand"
}

def pick_first(row: dict, *cands: str) -> str:
    """Return the first non-empty string from candidate column names."""
    for c in cands:
        if c and c in row:
            v = to_str(row.get(c, "")).strip()
            if v:
                return v
    return ""

def parse_specs_json(cell: str) -> dict:
    """Parse specs JSON cell into a dict, tolerating minor issues."""
    s = (cell or "").strip()
    if not s or not (s.startswith("{") and s.endswith("}")):
        return {}
    try:
        return json.loads(s)
    except Exception:
        # Try a light single-quote replacement as a last resort
        try:
            fixed = s.replace("'", '"')
            return json.loads(fixed)
        except Exception:
            return {}

def humanize_spec_key(k: str) -> str:
    """Turn CSV header/JSON key to a human-ish label for specs_text."""
    # strip leading spec_ and replace separators with spaces
    k0 = k.strip()
    if k0.lower().startswith("spec_"):
        k0 = k0[5:]
    k0 = k0.replace("_", " ").replace("-", " ").strip()
    # title case except small words
    parts = [w for w in re.split(r"\s+", k0) if w]
    cap = " ".join(w.capitalize() if w.lower() not in {"of","and","or","to","for","with"} else w.lower()
                   for w in parts) or k
    return cap

def flatten_specs(row: dict, specs_col: str) -> str:
    """
    Build the 'specs' channel as newline-joined 'Key: Value' lines.
    Combines dict from specs JSON + any 'spec_*' columns.
    """
    # 1) JSON dict
    merged: dict[str, str] = {}
    raw = pick_first(row, specs_col) if specs_col else ""
    obj = parse_specs_json(raw)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if v is None: 
                continue
            ks = to_str(k).strip()
            vs = to_str(v).strip()
            if not ks or not vs:
                continue
            if ks in META_KEYS_TO_SKIP:
                continue
            merged[ks] = vs

    # 2) spec_* columns
    for col, val in row.items():
        if not col.lower().startswith("spec_"):
            continue
        vs = to_str(val).strip()
        if not vs:
            continue
        # Prefer JSON value if present; otherwise add this.
        if col not in merged and col not in META_KEYS_TO_SKIP:
            merged[col] = vs

    if not merged:
        return "(no specs)"

    # Render sorted for determinism
    items = []
    for k in sorted(merged.keys(), key=lambda s: s.lower()):
        label = humanize_spec_key(k)
        items.append(f"{label}: {merged[k]}")
    return "\n".join(items)


# ----------------------- Channel builders -----------------------

def build_channels_from_row(row: dict, args) -> tuple[str, str, str, dict]:
    """
    Returns (title_text, desc_text, specs_text, meta_dict).
    """
    # identity-ish fields (flexible)
    name = pick_first(row, args.name_col, "name", "title")
    brand = pick_first(row, args.brand_col, "brand", "spec_brand", "Brand")
    desc = pick_first(row, args.desc_col, "description", "spec_description", "spec_metadescription", "spec_ogdescription")
    url = pick_first(row, args.url_col, "url", "product_url")
    mpn = pick_first(row, args.mpn_col, "mpn", "spec_mpn", "spec_manufacturers-part-number", "Manufacturers Part Number")
    sku = pick_first(row, args.sku_col, "sku", "spec_sku")

    # Title channel: name (+ brand)
    title_parts = []
    if name:
        title_parts.append(name)
    if brand and brand.lower() not in name.lower():
        title_parts.append(brand)
    title_text = " — ".join(title_parts) if title_parts else (desc[:80] if desc else "(no title)")

    # Description channel
    desc_text = desc if desc else "(no description)"

    # Specs channel (flatten JSON + spec_* columns)
    specs_text = flatten_specs(row, args.specs_col)

    # Meta for meta.jsonl
    meta = {
        "name": name,
        "brand": brand,
        "sku": sku,
        "mpn": mpn,
        "url": url,
        "specs_col": args.specs_col,
    }
    return title_text, desc_text, specs_text, meta


# ----------------------- OpenAI embedding -----------------------

def embed_texts_openai(client: OpenAI, texts: list[str], model: str, batch_size: int = 128) -> np.ndarray:
    out = []
    rng = range(0, len(texts), batch_size)
    iterator = tqdm(rng, desc=f"Embedding ({model})", unit="batch") if HAVE_TQDM else rng
    for i in iterator:
        chunk = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=chunk)
        out.extend([d.embedding for d in resp.data])
    arr = np.array(out, dtype="float32")
    return arr


# ----------------------- FAISS -----------------------

def build_and_write_faiss(vecs: np.ndarray, out_path: Path):
    if not HAVE_FAISS:
        print(f"[warn] FAISS not installed; skip building {out_path.name}. (pip install faiss-cpu)")
        return
    X = vecs.copy()
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, str(out_path))
    print(f"[faiss] wrote {out_path}")


# ----------------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser(description="Embed CSV into title/description/specs channels (OpenAI)")
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--out-dir", required=True, help="Directory for outputs (meta + vectors + faiss)")
    ap.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model")
    ap.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")
    ap.add_argument("--build-faiss", action="store_true", help="Also build FAISS indices per channel")

    # Column overrides (defaults tuned to your header list)
    ap.add_argument("--name-col", default="name", help="Name/title column")
    ap.add_argument("--desc-col", default="spec_description", help="Description/meta column")
    ap.add_argument("--specs-col", default="specs", help="Specs JSON column to flatten")
    ap.add_argument("--sku-col", default="spec_sku", help="SKU column")
    ap.add_argument("--mpn-col", default="spec_mpn", help="MPN column")
    ap.add_argument("--brand-col", default="brand", help="Brand column")
    ap.add_argument("--url-col", default="url", help="Product URL column")

    args = ap.parse_args()

    t0 = time.time()
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set. Put it in .env or export it.")
    client = OpenAI(api_key=api_key)

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Read CSV as strings (preserve columns)
    import pandas as pd
    print(f"[load] {csv_path}")
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    N = len(df)
    print(f"[load] rows: {N}")

    # Build channels + meta
    title_texts, desc_texts, specs_texts = [], [], []
    meta_lines = []

    iterator = df.iterrows()
    if HAVE_TQDM:
        iterator = tqdm(df.iterrows(), total=len(df), desc="Preparing channels")

    for i, row in iterator:
        rd = {k: to_str(row[k]) for k in df.columns}
        t, d, s, meta = build_channels_from_row(rd, args)
        title_texts.append(t)
        desc_texts.append(d)
        specs_texts.append(s)

        meta_out = {
            "row": int(i),
            **meta,
            # Keep raw snippets (handy for debugging)
            "raw_title": t[:400],
            "raw_desc": d[:400],
            "raw_specs": s[:800],
        }
        meta_lines.append(meta_out)

    # Write meta.jsonl (row-aligned)
    meta_path = out_dir / "meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for m in meta_lines:
            f.write(safe_json_dumps(m) + "\n")
    print(f"[write] {meta_path}")

    # Embed channels
    print(f"[embed] model={args.model} batch={args.batch_size}")
    t1 = time.time()
    vec_title = embed_texts_openai(client, title_texts, args.model, args.batch_size)
    t2 = time.time()
    vec_desc  = embed_texts_openai(client, desc_texts, args.model, args.batch_size)
    t3 = time.time()
    vec_specs = embed_texts_openai(client, specs_texts, args.model, args.batch_size)
    t4 = time.time()

    # Save vectors
    np.save(out_dir / "vectors_title.npy", vec_title.astype("float32"))
    np.save(out_dir / "vectors_desc.npy",  vec_desc.astype("float32"))
    np.save(out_dir / "vectors_specs.npy", vec_specs.astype("float32"))
    print(f"[write] vectors_title.npy  shape={vec_title.shape}")
    print(f"[write] vectors_desc.npy   shape={vec_desc.shape}")
    print(f"[write] vectors_specs.npy  shape={vec_specs.shape}")

    # Optional: per-channel FAISS
    if args.build_faiss:
        build_and_write_faiss(vec_title, out_dir / "index_title.faiss")
        build_and_write_faiss(vec_desc,  out_dir / "index_desc.faiss")
        build_and_write_faiss(vec_specs, out_dir / "index_specs.faiss")

    # Timing
    print(f"[timing] total: {time.time() - t0:.3f}s | title {t2-t1:.3f}s | desc {t3-t2:.3f}s | specs {t4-t3:.3f}s")


if __name__ == "__main__":
    main()