#!/usr/bin/env python3
"""
04_build_numeric_matrix.py
--------------------------

Project a Global Industrial–style CSV into a numeric matrix aligned to your
canonical fields, for use by 03_search.py’s hard-spec shortlist and numeric boost.

Inputs:
  - --csv                 e.g. globalindustrial_fasteners_full.csv
  - --specs-col           JSON column with spec dicts (default: specs)
  - --fields              fields.enriched.json (from 01/02 pipeline)
  - --alias               alias_map.enriched.json (from 02)
  - --out-dir             directory that will receive numeric_specs.npz + numeric_schema.json
  - [optional] --max-rows limit processed rows for testing

What it does:
  1) Loads the CSV as strings (keeps order identical to embedding build).
  2) For each row, builds a flat key→value map by:
       - parsing the JSON from --specs-col (dict)
       - scanning all "spec_*" columns and adding non-empty values
     (JSON keys win if duplicates)
  3) Normalizes keys via alias_map.enriched.json (lowercased text) to get canonical field_ids.
  4) Keeps ONLY fields with type == "numeric" from fields.enriched.json.
  5) For each numeric field, parses its value STRICTLY using spec_patterns.classify_value_strict:
       - accepts only "scalar" (no ranges/duals/threads)
       - converts unit to a canonical unit per family (length→inches, mass→lb, pressure→psi, etc.)
  6) Fills values[N, M] and mask[N, M] where M = number of numeric fields.
  7) Writes numeric_specs.npz and numeric_schema.json.

Usage:
  python 04_build_numeric_matrix.py \
    --csv globalindustrial_fasteners_full.csv \
    --specs-col specs \
    --fields fields.enriched.json \
    --alias alias_map.enriched.json \
    --out-dir index_out_v2
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Local parsing/units helpers
try:
    # You already have this module from earlier steps
    from spec_patterns import classify_value_strict
except Exception as e:
    raise SystemExit(f"[error] spec_patterns import failed: {e}")


# ---------------------- helpers ----------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_str(x) -> str:
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    return str(x)

def parse_specs_json(cell: str) -> Dict[str, str]:
    s = (cell or "").strip()
    if not s or not (s.startswith("{") and s.endswith("}")):
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        try:
            fixed = s.replace("'", '"')
            obj = json.loads(fixed)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

def norm_key(s: str) -> str:
    # normalize for alias matching (lowercase + single space)
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# Canonical unit conversion per family (extend if needed)
def convert_to_canonical(family: Optional[str], value: float, unit: Optional[str]) -> Optional[float]:
    """
    Returns value converted to canonical family units:
      - length: inches
      - pressure: psi
      - mass/weight: lb
      - torque: lb-ft
      - frequency: Hz
      - power: W
      - flow: gpm
      - temperature: °F (assumes input already normalized by classify if needed)
    Unknown/None family → return the raw value (unitless numeric is allowed).
    If unit not understood for the family, returns None to avoid bad data.
    """
    if value is None:
        return None
    u = (unit or "").strip().lower()

    if family == "length":
        # inches canonical
        if u in {"", None}:  # unitless is allowed; treat as already in canonical
            return float(value)
        if u in {'"', "in", "inch", "inches"}:
            return float(value)
        if u in {"mm", "millimeter", "millimeters"}:
            return float(value) / 25.4
        if u in {"cm", "centimeter", "centimeters"}:
            return float(value) / 2.54
        if u in {"ft", "foot", "feet", "'"}:
            return float(value) * 12.0
        # OD/ID synonyms should be handled at field_id level, not here
        return None

    if family == "pressure":
        # psi canonical
        if u in {"", None, "psi", "psig", "psia"}:
            return float(value)
        if u in {"bar"}:
            return float(value) * 14.5037738
        if u in {"kpa"}:
            return float(value) * 0.145037738
        if u in {"mpa"}:
            return float(value) * 145.037738
        return None

    if family in {"mass", "weight"}:
        # pounds canonical
        if u in {"", None, "lb", "lbs", "pound", "pounds"}:
            return float(value)
        if u in {"kg", "kilogram", "kilograms"}:
            return float(value) * 2.20462262
        if u in {"g", "gram", "grams"}:
            return float(value) * 0.00220462262
        return None

    if family == "torque":
        # lb-ft canonical
        if u in {"", None, "lb-ft", "lbft", "ft-lb", "ftlb"}:
            return float(value)
        if u in {"nm", "newton-meter", "newton-meters"}:
            return float(value) * 0.737562149
        return None

    if family == "frequency":
        if u in {"", None, "hz", "hertz"}:
            return float(value)
        return None

    if family == "power":
        # watts canonical
        if u in {"", None, "w", "watt", "watts"}:
            return float(value)
        if u in {"kw", "kilowatt", "kilowatts"}:
            return float(value) * 1000.0
        if u in {"hp", "horsepower"}:
            return float(value) * 745.699872
        return None

    if family == "flow":
        if u in {"", None, "gpm"}:
            return float(value)
        if u in {"lpm"}:
            return float(value) * 0.264172052
        if u in {"cfm"}:
            # ambiguous; keep as None to avoid bad data (change if you want)
            return None
        return None

    if family == "temperature":
        # We won't auto-convert C↔F here (classify_value_strict can be extended to do that).
        # If unit missing or already °F-ish, keep raw.
        if u in {"", None, "f", "°f", "degf"}:
            return float(value)
        # Support a simple C→F if you want:
        if u in {"c", "°c", "degc"}:
            return float(value) * 9.0/5.0 + 32.0
        return None

    # Unknown family: accept unitless only
    if u in {"", None}:
        return float(value)
    return None


# ---------------------- main build ----------------------

def main():
    ap = argparse.ArgumentParser(description="Build numeric_specs matrix from CSV + fields/aliases")
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--specs-col", default="specs", help="Column with specs JSON dict")
    ap.add_argument("--fields", required=True, help="fields.enriched.json")
    ap.add_argument("--alias", required=True, help="alias_map.enriched.json (alias -> field_id)")
    ap.add_argument("--out-dir", required=True, help="Output directory for numeric_specs.*")
    ap.add_argument("--max-rows", type=int, default=-1, help="Limit rows for testing")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Load fields + aliases
    fields_obj = json.loads(Path(args.fields).read_text())
    fields = fields_obj.get("fields", [])
    alias_obj = json.loads(Path(args.alias).read_text())
    alias_map = alias_obj.get("aliases", {})

    # Build list of numeric fields (M columns)
    numeric_fields: List[Tuple[str, str]] = []  # [(field_id, family)]
    family_of: Dict[str, Optional[str]] = {}
    for f in fields:
        fid = f.get("id")
        typ = f.get("type")
        fam = f.get("unit_family")  # may be None
        if fid and typ == "numeric":
            numeric_fields.append((fid, fam))
            family_of[fid] = fam
    if not numeric_fields:
        raise SystemExit("[error] No numeric fields found in fields.enriched.json")

    # alias lookup (normalized)
    # Note: alias_map maps alias TEXT to field_id
    alias_norm_to_field: Dict[str, str] = {}
    for a, fid in alias_map.items():
        alias_norm_to_field[norm_key(a)] = fid

    # Read CSV
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if args.max_rows > 0:
        df = df.iloc[:args.max_rows].copy()
    N = len(df)
    M = len(numeric_fields)
    print(f"[load] rows={N} | numeric fields (cols)={M}")

    # Build row dicts of specs (JSON + spec_* columns)
    def row_specs_map(row: pd.Series) -> Dict[str, str]:
        # 1) start with JSON dict
        specs: Dict[str, str] = {}
        if args.specs_col in row:
            obj = parse_specs_json(to_str(row[args.specs_col]))
            if isinstance(obj, dict):
                for k, v in obj.items():
                    ks = to_str(k).strip()
                    vs = to_str(v).strip()
                    if ks and vs and ks.lower() not in {"price_jsonld", "currency_jsonld"}:
                        specs[ks] = vs
        # 2) add spec_* columns if not present
        for col in row.index:
            if col.lower().startswith("spec_"):
                ks = col
                if ks not in specs:
                    vs = to_str(row[col]).strip()
                    if vs:
                        specs[ks] = vs
        return specs

    # Map a raw key to canonical field_id (via alias map primarily; fallback to id exact match)
    def map_key_to_field_id(raw_key: str) -> Optional[str]:
        nk = norm_key(raw_key)
        fid = alias_norm_to_field.get(nk)
        if fid:
            return fid
        # fallback: exact id match (some CSV headers may already be canonical ids)
        if nk in {norm_key(fid_) for fid_, _ in numeric_fields}:
            # find original casing id
            for fid0, _ in numeric_fields:
                if norm_key(fid0) == nk:
                    return fid0
        return None

    # Build matrices
    values = np.zeros((N, M), dtype="float32")
    mask = np.zeros((N, M), dtype="bool")

    # Precompute column index per field_id
    col_of: Dict[str, int] = {fid: j for j, (fid, _) in enumerate(numeric_fields)}

    # Iterate rows
    for i, row in df.iterrows():
        kv = row_specs_map(row)
        # First pass: group raw keys that map to a known numeric field
        for raw_k, raw_v in kv.items():
            fid = map_key_to_field_id(raw_k)
            if not fid:
                continue
            j = col_of.get(fid)
            if j is None:
                continue
            fam = family_of.get(fid)

            # Parse strictly as scalar
            cl = classify_value_strict(raw_v)
            if cl.get("kind") != "scalar":
                continue
            val = cl.get("value")
            unit = cl.get("unit")
            # Convert per family
            vcanon = convert_to_canonical(fam, val, unit)
            if vcanon is None:
                continue

            values[i, j] = float(vcanon)
            mask[i, j] = True

    # Write outputs
    np.savez_compressed(out_dir / "numeric_specs.npz", values=values, mask=mask)
    schema = {
        "attrs": [fid for fid, _ in numeric_fields],
        "families": {fid: fam for fid, fam in numeric_fields},
    }
    (out_dir / "numeric_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"[write] {out_dir/'numeric_specs.npz'} values.shape={values.shape} mask.sum={mask.sum()}")
    print(f"[write] {out_dir/'numeric_schema.json'} M={M}")
    print("[done] numeric matrix ready.")
    

if __name__ == "__main__":
    main()