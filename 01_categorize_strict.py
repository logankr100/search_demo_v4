#!/usr/bin/env python3
"""
01_categorize_strict.py
-----------------------

Conservative, auditable categorization of spec keys:
- Reads raw_keys.json (from 00_harvest_spec_keys.py).
- Groups keys by case/space (safe-only; no fuzzy merges).
- For each original key, classifies example values using strict patterns:
    * scalar with optional known unit (incl. " and ' for in/ft; unicode quotes; 1-1/2)
    * rejects ranges (a–b), duals (208/230 V, 1/4-20), thread-like (M10×1.5, NPT)
    * allows unitless numeric scalars
- Field is numeric only if **all** samples are scalar (unitless or with known unit).
- Family is set only if **all unit-bearing samples** map to the **same** family; else null.
- Identifiers/categorical keys are hard-pinned to string/null.
- Optional meta keys can be dropped entirely.

Outputs:
  - fields.semantic.json  : canonical fields with safe type/family + seed aliases
  - spec_key_report.csv   : flat summary (per canonical field)
  - type_unit_audit.jsonl : deep per-original-key audit (buckets, reasons)
  - warnings.txt          : optional human-readable issues

Usage:
  python 01_categorize_strict.py \
    --input raw_keys.json \
    --out-fields fields.semantic.json \
    --out-report spec_key_report.csv \
    --out-audit type_unit_audit.jsonl \
    --min-support 3 \
    --max-samples 200 \
    --drop-meta true \
    --thread-family-thresh 1.0
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

# Local helper modules (must be in the same directory)
try:
    from spec_lexicon import (
        UNIT_MAP,
        IDENTIFIER_KEYS,
        CATEGORICAL_KEYS,
        META_DROP_KEYS,
        FAMILY_NAME_HINTS,
        unit_to_family,
    )
    from spec_patterns import (
        classify_value_strict,
        family_from_units,
    )
except Exception as e:
    sys.exit(f"[error] Failed to import spec_lexicon/spec_patterns: {e}")


# ---------------------- Small utils ----------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def norm_case_space(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()

def snakeify(s: str) -> str:
    out = []
    prev_us = False
    for ch in norm_case_space(s):
        if ch.isalnum():
            out.append(ch if ch.islower() or ch.isdigit() else ch.lower())
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    ss = "".join(out).strip("_")
    return ss or "field"

def title_label(snake_id: str) -> str:
    return " ".join(w.capitalize() for w in (snake_id or "").split("_")) or "Field"

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


# ---------------------- Data structures ----------------------

@dataclass
class RawKey:
    key: str
    count: int
    example: str

@dataclass
class Group:
    signature: str                      # case/space-folded signature
    members: List[RawKey]               # original keys that fold to signature


# ---------------------- IO ----------------------

def read_raw_keys(path: Path) -> List[RawKey]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        sys.exit(f"[error] cannot read {path}: {e}")
    items = obj.get("keys", [])
    out: List[RawKey] = []
    for it in items:
        try:
            k = str(it["key"])
            c = int(it.get("count", 0) or 0)
            ex = str(it.get("example", "") or "")
            out.append(RawKey(k, c, ex))
        except Exception:
            # skip malformed rows
            continue
    return out

def write_json(path: Path, obj: Any):
    ensure_dir(path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path: Path, rows: List[Dict[str, Any]], header: List[str]):
    ensure_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in header})


# ---------------------- Grouping ----------------------

def group_by_case_space(raw: List[RawKey]) -> List[Group]:
    buckets: Dict[str, List[RawKey]] = {}
    for rk in raw:
        sig = norm_case_space(rk.key)
        if not sig:
            # skip completely blank keys
            continue
        buckets.setdefault(sig, []).append(rk)
    groups = [Group(signature=s, members=v) for s, v in buckets.items()]
    groups.sort(key=lambda g: -sum(m.count for m in g.members))
    return groups


# ---------------------- Classification & Decision ----------------------

@dataclass
class BucketSummary:
    n_samples: int = 0
    scalar_known: List[str] = None
    scalar_unitless: List[str] = None
    range_vals: List[str] = None
    dual_vals: List[str] = None
    thread_vals: List[str] = None
    unknown_vals: List[str] = None
    units_seen: List[str] = None  # canonical unit tokens
    families_seen: List[str] = None

    def __post_init__(self):
        self.scalar_known = self.scalar_known or []
        self.scalar_unitless = self.scalar_unitless or []
        self.range_vals = self.range_vals or []
        self.dual_vals = self.dual_vals or []
        self.thread_vals = self.thread_vals or []
        self.unknown_vals = self.unknown_vals or []
        self.units_seen = self.units_seen or []
        self.families_seen = self.families_seen or []

def collect_samples_for_key(example: str, max_samples: int) -> List[str]:
    """
    raw_keys.json has only one example per key today.
    This function exists to be forward-compatible if you later pass
    more samples (e.g., from scanning the catalog).
    """
    vals: List[str] = []
    if isinstance(example, str) and example.strip():
        vals.append(example.strip())
    return vals[:max_samples]

def summarize_key_samples(example_values: List[str]) -> BucketSummary:
    bs = BucketSummary()
    for v in example_values:
        cl = classify_value_strict(v)
        bs.n_samples += 1
        kind = cl.get("kind")
        unit = cl.get("unit") or ""
        fam = cl.get("family")  # may be None

        if kind == "scalar":
            if unit:
                bs.scalar_known.append(v)
                bs.units_seen.append(unit)
                if fam:
                    bs.families_seen.append(fam)
            else:
                bs.scalar_unitless.append(v)

        elif kind == "range":
            bs.range_vals.append(v)
        elif kind == "dual":
            bs.dual_vals.append(v)
        elif kind == "thread":
            bs.thread_vals.append(v)
        else:
            bs.unknown_vals.append(v)
    return bs

def decide_field_type_family(
    field_id: str,
    field_label: str,
    member_keys: List[RawKey],
    min_support: int,
    thread_family_thresh: float,
) -> Tuple[str, Optional[str], Dict[str, Any], BucketSummary]:
    """
    Returns:
      type_str: "numeric" or "string"
      unit_family: family string or None
      decision_meta: dict with reasons and counts
      agg_bs: aggregated BucketSummary across member keys
    """
    # Aggregate all member examples
    agg = BucketSummary()
    per_key_buckets: List[Tuple[str, BucketSummary]] = []

    for rk in member_keys:
        samples = collect_samples_for_key(rk.example, max_samples=200)  # single example today
        bs = summarize_key_samples(samples)
        per_key_buckets.append((rk.key, bs))
        # merge into agg
        agg.n_samples += bs.n_samples
        agg.scalar_known.extend(bs.scalar_known)
        agg.scalar_unitless.extend(bs.scalar_unitless)
        agg.range_vals.extend(bs.range_vals)
        agg.dual_vals.extend(bs.dual_vals)
        agg.thread_vals.extend(bs.thread_vals)
        agg.unknown_vals.extend(bs.unknown_vals)
        agg.units_seen.extend(bs.units_seen)
        agg.families_seen.extend(bs.families_seen)

    # Name pins (identifiers / categorical)
    if field_id in IDENTIFIER_KEYS or field_id in CATEGORICAL_KEYS:
        return (
            "string",
            None,
            {
                "path": ["name_pin"],
                "id_pin": field_id in IDENTIFIER_KEYS,
                "cat_pin": field_id in CATEGORICAL_KEYS,
            },
            agg,
        )

    # Support guard (only warn — do not force string if clean)
    low_support = agg.n_samples < min_support

    # Numeric rule: ALL samples must be scalar (known unit or unitless)
    violations = len(agg.range_vals) + len(agg.dual_vals) + len(agg.thread_vals) + len(agg.unknown_vals)
    all_scalar = (violations == 0) and (agg.n_samples > 0)

    if all_scalar:
        # Family: only if all unit-bearing samples are the SAME family
        fam = family_from_units(agg.units_seen)  # ignores unitless
        meta = {
            "path": ["numeric_all_scalar", "family_single" if fam else "family_null_or_unitless"],
            "low_support": low_support,
        }
        return "numeric", fam, meta, agg

    # Thread rule (non-numeric)
    n_thread = len(agg.thread_vals)
    p_thread = (n_thread / agg.n_samples) if agg.n_samples else 0.0
    if n_thread > 0 and p_thread >= float(thread_family_thresh) and (len(agg.range_vals) + len(agg.dual_vals) + len(agg.unknown_vals) == 0):
        return "string", "thread", {
            "path": ["thread_only"],
            "p_thread": round(p_thread, 3),
            "low_support": low_support,
        }, agg

    # Otherwise: string/null with violations recorded
    return "string", None, {
        "path": ["string_with_violations"],
        "low_support": low_support,
        "counts": {
            "n_range": len(agg.range_vals),
            "n_dual": len(agg.dual_vals),
            "n_thread": len(agg.thread_vals),
            "n_unknown": len(agg.unknown_vals),
        }
    }, agg


# ---------------------- Field building ----------------------

def pick_canonical_member(members: List[RawKey]) -> RawKey:
    # highest count wins; tie-breaker by key text
    return sorted(members, key=lambda r: (-r.count, r.key.lower()))[0]

def build_aliases(members: List[RawKey]) -> List[Dict[str, Any]]:
    # Seed aliases are only the **observed** surface keys (no invented aliases).
    total = sum(m.count for m in members) or 1
    agg: Dict[str, int] = {}
    for m in members:
        agg[m.key] = agg.get(m.key, 0) + m.count
    best = max(agg.values()) if agg else 1
    out = []
    for k, cnt in sorted(agg.items(), key=lambda kv: (-kv[1], kv[0].lower())):
        conf = min(1.0, cnt / best)
        out.append({"text": k, "count": int(cnt), "confidence": round(conf, 3)})
    return out

def build_fields_from_groups(
    groups: List[Group],
    min_support: int,
    thread_family_thresh: float,
    drop_meta: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    fields: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for g in groups:
        # early drop of pure meta keys (optional)
        sig_id = snakeify(g.signature)
        if drop_meta and sig_id in META_DROP_KEYS:
            continue

        canon = pick_canonical_member(g.members)
        field_id = snakeify(g.signature)           # canonical id from signature (not necessarily from canon key)
        label = title_label(field_id)
        example = canon.example
        total_count = sum(m.count for m in g.members)

        type_str, family, decision_meta, agg = decide_field_type_family(
            field_id, label, g.members, min_support=min_support, thread_family_thresh=thread_family_thresh
        )

        # warnings
        if decision_meta.get("low_support"):
            warnings.append(f"[low-support] id={field_id} n_samples={agg.n_samples}")

        # family mixed warning: if numeric and family None while we saw units
        if type_str == "numeric" and family is None and len(agg.units_seen) > 0:
            warnings.append(f"[mixed-family] id={field_id} units={sorted(set(agg.units_seen))}")

        # violation warning
        if decision_meta.get("path") == ["string_with_violations"]:
            c = decision_meta.get("counts", {})
            if any(c.get(k, 0) > 0 for k in ("n_range", "n_dual", "n_thread", "n_unknown")):
                warnings.append(f"[violations] id={field_id} range={c.get('n_range',0)} dual={c.get('n_dual',0)} thread={c.get('n_thread',0)} unknown={c.get('n_unknown',0)}")

        # seed aliases
        aliases = build_aliases(g.members)

        # main field row
        fields.append({
            "id": field_id,
            "label": label,
            "type": type_str,                # "numeric" or "string"
            "unit_family": family,           # e.g., "length" or None
            "count": int(total_count),
            "example": example,
            "aliases": aliases,
        })

        # CSV report row (per canonical field)
        report_rows.append({
            "id": field_id,
            "label": label,
            "type": type_str,
            "unit_family": (family or ""),
            "count": int(total_count),
            "n_samples": agg.n_samples,
            "n_scalar_unit": len(agg.scalar_known),
            "n_scalar_unitless": len(agg.scalar_unitless),
            "n_range": len(agg.range_vals),
            "n_dual": len(agg.dual_vals),
            "n_thread": len(agg.thread_vals),
            "n_unknown": len(agg.unknown_vals),
            "units_seen": ";".join(sorted(set(agg.units_seen))) if agg.units_seen else "",
            "families_seen": ";".join(sorted(set(agg.families_seen))) if agg.families_seen else "",
            "example": example,
            "decision": "/".join(decision_meta.get("path", [])),
        })

        # Deep audit rows (per original key)
        for rk in g.members:
            bs = summarize_key_samples(collect_samples_for_key(rk.example, 200))
            audit_rows.append({
                "key": rk.key,
                "group_id": field_id,
                "group_label": label,
                "count": rk.count,
                "samples": {
                    "scalar_known": bs.scalar_known[:5],
                    "scalar_unitless": bs.scalar_unitless[:5],
                    "range": bs.range_vals[:5],
                    "dual": bs.dual_vals[:5],
                    "thread": bs.thread_vals[:5],
                    "unknown": bs.unknown_vals[:5],
                },
                "units_seen": sorted(set(bs.units_seen)) if bs.units_seen else [],
                "families_seen": sorted(set(bs.families_seen)) if bs.families_seen else [],
                "n_samples": bs.n_samples,
                "decision_inherited": decision_meta,  # reference for this group
            })

    # sort fields by descending count
    fields.sort(key=lambda f: -f["count"])
    return fields, report_rows, audit_rows, warnings


# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser(description="Conservative categorization of spec keys (ready for alias generation).")
    ap.add_argument("--input", default="raw_keys.json", help="Input JSON from 00_harvest_spec_keys.py")
    ap.add_argument("--out-fields", default="fields.semantic.json", help="Output canonical fields JSON")
    ap.add_argument("--out-report", default="spec_key_report.csv", help="Output CSV report")
    ap.add_argument("--out-audit", default="type_unit_audit.jsonl", help="Output JSONL audit")
    ap.add_argument("--warnings", default="warnings.txt", help="Optional warnings text file")
    ap.add_argument("--min-support", type=int, default=3, help="Minimum number of samples to feel confident")
    ap.add_argument("--max-samples", type=int, default=200, help="Max distinct samples per key (future-proof; today 1)")
    ap.add_argument("--drop-meta", type=str, default="true", help="Drop meta/marketing keys (true/false)")
    ap.add_argument("--thread-family-thresh", type=float, default=1.0, help="Fraction of thread-like samples to set family=thread")
    args = ap.parse_args()

    drop_meta = str(args.drop_meta).lower() in {"1", "true", "yes", "y"}

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"[error] input not found: {in_path}")

    raw = read_raw_keys(in_path)
    if not raw:
        sys.exit("[error] no keys in input")

    print(f"[info] loaded {len(raw)} raw keys from {in_path.name}")

    # Group by case/space only (safe)
    groups = group_by_case_space(raw)
    print(f"[info] groups (case/space-folded): {len(groups)}")

    # Build fields + reports
    fields, report_rows, audit_rows, warnings = build_fields_from_groups(
        groups,
        min_support=args.min_support,
        thread_family_thresh=args.thread_family_thresh,
        drop_meta=drop_meta,
    )

    # Write outputs
    write_json(Path(args.out_fields), {
        "version": 1,
        "generated_at": now_iso(),
        "policy": {
            "numeric_requires_all_scalar": True,
            "family_requires_unanimity": True,
            "unitless_scalar_allowed": True,
        },
        "fields": fields
    })
    print(f"[ok] wrote {args.out_fields} ({len(fields)} fields)")

    csv_header = [
        "id","label","type","unit_family","count",
        "n_samples","n_scalar_unit","n_scalar_unitless",
        "n_range","n_dual","n_thread","n_unknown",
        "units_seen","families_seen","example","decision"
    ]
    write_csv(Path(args.out_report), report_rows, csv_header)
    print(f"[ok] wrote {args.out_report} ({len(report_rows)} rows)")

    write_jsonl(Path(args.out_audit), audit_rows)
    print(f"[ok] wrote {args.out_audit} ({len(audit_rows)} rows)")

    if warnings:
        wp = Path(args.warnings)
        ensure_dir(wp)
        wp.write_text("\n".join(warnings) + "\n", encoding="utf-8")
        print(f"[warn] wrote {wp} ({len(warnings)} warnings)")

    print("[summary] done.")

if __name__ == "__main__":
    main()