#!/usr/bin/env python3
"""
00_harvest_spec_keys.py
----------------------

Lightweight harvester that scans a product catalog CSV and extracts *all*
unique spec keys from a JSON column (e.g., "specs").

Outputs:
  - raw_keys.json : [{"key":..., "count":..., "example":...}, ...]
  - raw_keys.csv  : same data in CSV form

Example usage:
  python 00_harvest_spec_keys.py \
    --input globalindustrial_fasteners_full.csv \
    --specs-col specs \
    --out-json raw_keys.json \
    --out-csv raw_keys.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
import pandas as pd


def parse_specs_cell(cell: str):
    """
    Try to parse a specs cell into a dict.
    Handles single-quoted JSON and empty/null cells.
    """
    if not isinstance(cell, str) or not cell.strip():
        return {}
    s = cell.strip()
    if not s.startswith("{"):
        return {}
    try:
        return json.loads(s)
    except Exception:
        try:
            # replace single quotes → double quotes (safe heuristic)
            fixed = s.replace("'", '"')
            return json.loads(fixed)
        except Exception:
            return {}


def harvest_keys(df: pd.DataFrame, specs_col: str):
    key_counts = {}
    key_examples = {}

    for _, row in df.iterrows():
        specs = parse_specs_cell(row.get(specs_col))
        if not isinstance(specs, dict):
            continue
        for k, v in specs.items():
            key = str(k).strip()
            if not key:
                continue
            key_counts[key] = key_counts.get(key, 0) + 1
            if key not in key_examples and v not in (None, "", "nan", "NaN"):
                key_examples[key] = str(v)

    # build final list
    items = []
    for k in sorted(key_counts.keys(), key=lambda x: -key_counts[x]):
        items.append({
            "key": k,
            "count": int(key_counts[k]),
            "example": key_examples.get(k, "")
        })
    return items


def write_json(data, path: Path):
    path.write_text(json.dumps({"keys": data}, indent=2, ensure_ascii=False))
    print(f"[ok] wrote JSON: {path} ({len(data)} keys)")


def write_csv(data, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "count", "example"])
        for d in data:
            w.writerow([d["key"], d["count"], d["example"]])
    print(f"[ok] wrote CSV: {path} ({len(data)} keys)")


def main():
    ap = argparse.ArgumentParser(description="Harvest all unique spec keys from a CSV column of JSON specs.")
    ap.add_argument("--input", required=True, help="Path to input CSV")
    ap.add_argument("--specs-col", default="specs", help="Column containing JSON specs")
    ap.add_argument("--out-json", default="raw_keys.json", help="Output JSON path")
    ap.add_argument("--out-csv", default="raw_keys.csv", help="Output CSV path")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.specs_col not in df.columns:
        sys.exit(f"[error] column '{args.specs_col}' not found in {args.input}")

    print(f"[info] loaded {len(df):,} rows from {args.input}")
    items = harvest_keys(df, args.specs_col)
    print(f"[summary] unique keys: {len(items)}")

    if not items:
        print("[warn] no keys found — check that the specs column contains valid JSON")
        return

    # preview few
    print("\nTop 20 keys:")
    for d in items[:20]:
        print(f"{d['key']:<40} {d['count']:<6} {d['example']}")

    write_json(items, Path(args.out_json))
    write_csv(items, Path(args.out_csv))


if __name__ == "__main__":
    main()