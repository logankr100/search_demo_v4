#!/usr/bin/env python3
"""
02_generate_aliases_deterministic.py
------------------------------------

Deterministic alias generator for *hard-spec* fields (no LLMs, no embeddings).

Changes in this version:
  - Tight seeds: bare Dia/Diam/Ø only for generic 'diameter' (not scoped fields)
  - OD/ID keep explicit phrases + acronyms, not Dia/Diam/Ø
  - Compound aliases from labels (e.g., 'Wheel Dia', 'Spindle OD', 'Motor Volts')
  - Safe alias map with collision logging (no silent overwrite)
  - Optional guard: bare OD/ID blocked for scoped fields

Outputs:
  - fields.enriched.json
  - alias_map.enriched.json
  - audit_aliases/accepted.jsonl
  - audit_aliases/blocked.jsonl
  - audit_aliases/collisions.jsonl
  - audit_aliases/orphans.jsonl
  - audit_aliases/summary.csv

Usage:
  python 02_generate_aliases_deterministic.py \
    --input fields.semantic.json \
    --out-fields fields.enriched.json \
    --out-alias alias_map.enriched.json \
    --audit-dir audit_aliases_full \
    --max-aliases 5
"""

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple


# ---------------- utils ----------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path: Path, rows: List[Dict[str, Any]], header: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in header})

def norm_case_space(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()


# ---------------- policy: families, seeds, whitelists, safety ----------------

DEFAULT_HARD_FAMILIES = {
    "length","thread","pressure","voltage","current","power","flow","mass","temperature"
}

# Tightened deterministic seeds:
# - generic diameter keeps Dia/Diam/Ø
# - outside/inside diameter keep explicit phrases + acronyms, not Dia/Diam/Ø
SEED_ENRICH: Dict[str, List[str]] = {
    "diameter":         ["Dia","Diam","Ø"],

    "outside_diameter": ["Outside Diameter","OD","O.D."],
    "inside_diameter":  ["Inside Diameter","ID","I.D."],

    "thread_size":      ["Thread Size","Thread","Thread Spec","Threading","Thread Standard"],
    "pipe_size":        ["Pipe Size","NPS"],

    "voltage":          ["Voltage","Volts","V"],
    "current":          ["Current","Amperage","Amps","A"],
    "pressure":         ["Pressure"],   # units like PSI appear in compounds (e.g., 'Line PSI')
    "flow":             ["Flow","CFM","SCFM","GPM","LPM"],
    "weight":           ["Weight","Wt"],
    "mass":             ["Mass","Weight","Wt"],
    "temperature":      ["Temp","Temperature","°F","°C"],
}

# Substring helpers (order matters)
ID_HELPERS: List[Dict[str, Any]] = [
    {"match": re.compile(r"\boutside_?diameter\b"),           "use": "outside_diameter"},
    {"match": re.compile(r"\bouter_?diameter\b"),             "use": "outside_diameter"},
    {"match": re.compile(r"\bexternal_?diameter\b"),          "use": "outside_diameter"},
    {"match": re.compile(r"\binside_?diameter\b"),            "use": "inside_diameter"},
    {"match": re.compile(r"\binternal_?diameter\b"),          "use": "inside_diameter"},
    {"match": re.compile(r"(^|_)od($|_)"),                    "use": "outside_diameter"},
    {"match": re.compile(r"(^|_)id($|_)"),                    "use": "inside_diameter"},
    {"match": re.compile(r"\bthread_?size\b"),                "use": "thread_size"},
    {"match": re.compile(r"\bthread(?:ing)?\b"),              "use": "thread_size"},
    {"match": re.compile(r"\bthread_?pitch\b"),               "use": "thread_size"},
    {"match": re.compile(r"\btpi\b"),                         "use": "thread_size"},
    {"match": re.compile(r"\bpipe_?size\b"),                  "use": "pipe_size"},
    {"match": re.compile(r"\bvoltage\b"),                     "use": "voltage"},
    {"match": re.compile(r"\bcurrent\b"),                     "use": "current"},
    {"match": re.compile(r"\bamperage\b"),                    "use": "current"},
    {"match": re.compile(r"\bpressure\b"),                    "use": "pressure"},
    {"match": re.compile(r"\bflow\b"),                        "use": "flow"},
    {"match": re.compile(r"\bweight\b"),                      "use": "weight"},
    {"match": re.compile(r"\bmass\b"),                        "use": "mass"},
    {"match": re.compile(r"\btemperature\b"),                 "use": "temperature"},
]

WHITELIST_TOKENS: Dict[str, List[str]] = {
    "outside_diameter": ["od","o.d.","outside diameter"],
    "inside_diameter":  ["id","i.d.","inside diameter","bore"],
    "diameter":         ["dia","diam","ø"],
    "thread_size":      ["thread size","thread","thread spec","threading","thread standard","tpi","thread pitch"],
    "pipe_size":        ["pipe size","nps"],
    "voltage":          ["voltage","volts","v"],
    "current":          ["current","amperage","amps","a"],
    "pressure":         ["pressure","psi"],
    "flow":             ["flow","cfm","scfm","gpm","lpm"],
    "weight":           ["weight","wt"],
    "mass":             ["mass","weight","wt"],
    "temperature":      ["temp","temperature","°f","°c"],
}

# Compound alias generation
CORE_ALIASES = {
    "outside_diameter": ["OD", "O.D.", "Outside Diameter"],
    "inside_diameter":  ["ID", "I.D.", "Inside Diameter", "Bore"],
    "diameter":         ["Dia", "Diam", "Ø"],
    "voltage":          ["Voltage", "Volts", "V"],
    "current":          ["Current", "Amperage", "Amps", "A"],
    "pressure":         ["Pressure", "PSI"],
    "flow":             ["Flow", "CFM", "SCFM", "GPM", "LPM"],
    "weight":           ["Weight", "Wt"],
    "mass":             ["Mass", "Weight", "Wt"],
    "temperature":      ["Temp", "Temperature", "°F", "°C"],
    "thread_size":      ["Thread Size", "Thread", "Thread Spec", "Threading", "Thread Standard"],
    "pipe_size":        ["Pipe Size", "NPS"],
}

CORE_DETECT = [
    ("outside_diameter", re.compile(r"^(?P<prefix>.+?)\s+outside\s+diameter\b", re.I)),
    ("inside_diameter",  re.compile(r"^(?P<prefix>.+?)\s+inside\s+diameter\b",  re.I)),
    ("diameter",         re.compile(r"^(?P<prefix>.+?)\s+diameter\b",           re.I)),
    ("voltage",          re.compile(r"^(?P<prefix>.+?)\s+voltage\b",            re.I)),
    ("current",          re.compile(r"^(?P<prefix>.+?)\s+(?:current|amperage)\b", re.I)),
    ("pressure",         re.compile(r"^(?P<prefix>.+?)\s+pressure\b",           re.I)),
    ("flow",             re.compile(r"^(?P<prefix>.+?)\s+flow\b",               re.I)),
    ("weight",           re.compile(r"^(?P<prefix>.+?)\s+weight\b",             re.I)),
    ("mass",             re.compile(r"^(?P<prefix>.+?)\s+mass\b",               re.I)),
    ("temperature",      re.compile(r"^(?P<prefix>.+?)\s+temperature\b",        re.I)),
    ("thread_size",      re.compile(r"^(?P<prefix>.+?)\s+thread(?:\s+size)?\b", re.I)),
    ("pipe_size",        re.compile(r"^(?P<prefix>.+?)\s+pipe\s+size\b",        re.I)),
]

PREFIX_STOPWORDS = {"overall","total","nominal","approx","maximum","minimum","max","min"}

BLACKLIST = re.compile(
    r"\b(sku|mpn|brand|manufacturer(?:s)?\s*part\s*number|serial|model|series|collection|kit|set|premium)\b",
    re.I,
)

SAFE_CHARS = re.compile(r"^[a-z0-9][a-z0-9\s\-\._/°]{1,47}$", re.I)


# ---------------- helpers ----------------

def is_hard_field(field: dict, hard_families: Set[str]) -> bool:
    return (field.get("type") == "numeric") and (field.get("unit_family") in hard_families)

def collect_global_seeds(fields: List[dict]) -> Set[str]:
    seeds: Set[str] = set()
    for f in fields:
        seeds.add(norm_case_space(f.get("id","")))
        seeds.add(norm_case_space(f.get("label","")))
        for a in f.get("aliases", []):
            seeds.add(norm_case_space(a.get("text","")))
    return seeds

def is_scoped_diameter(fid: str) -> bool:
    """
    True for fields like wheel_diameter, base_diameter, etc.
    Returns False for canonical diameter, inside_diameter, outside_diameter.
    """
    if fid in {"diameter", "outside_diameter", "inside_diameter"}:
        return False
    return fid.endswith("_diameter")

def enrich_field_aliases(field: dict) -> List[str]:
    """
    Attach deterministic seeds while avoiding duplicate generic diameter aliases.
    """
    fid = field.get("id", "")
    low = fid.lower()
    add: List[str] = []

    # 1. Exact match in SEED_ENRICH
    add += SEED_ENRICH.get(fid, [])

    # 2. Otherwise, match helper patterns (carefully)
    if not add:
        for h in ID_HELPERS:
            if h["match"].search(low):
                # Prevent applying generic 'diameter' seeds to scoped *_diameter fields
                if h.get("use") == "diameter" and is_scoped_diameter(fid):
                    add = []
                    break
                add = SEED_ENRICH.get(h["use"], [])
                break

    if not add:
        return []

    # 3. Apply only new aliases
    field.setdefault("aliases", [])
    existing = {a.get("text", "") for a in field["aliases"]}
    added = []
    for t in add:
        if t not in existing:
            field["aliases"].append({"text": t, "confidence": 1.0})
            added.append(t)
    return added

def passes_whitelist(fid: str, alias: str) -> bool:
    toks = WHITELIST_TOKENS.get(fid, [])
    if not toks:
        # try helper-derived family
        use = None
        low = fid.lower()
        for h in ID_HELPERS:
            if h["match"].search(low):
                use = h["use"]
                break
        if use:
            toks = WHITELIST_TOKENS.get(use, [])
    if not toks:
        return True
    al = norm_case_space(alias)
    return any(tok in al for tok in toks)

def is_blocked_text(alias: str) -> Tuple[bool, str]:
    if not alias or len(alias) < 2:
        return True, "too_short"
    if len(alias) > 48:
        return True, "too_long"
    if not SAFE_CHARS.match(alias):
        return True, "bad_charset"
    if BLACKLIST.search(alias):
        return True, "blacklist_token"
    return False, ""

def extract_prefix_core(label: str, fid: str) -> Tuple[str|None, str|None]:
    lbl = (label or "").strip()
    if lbl:
        for core, regex in CORE_DETECT:
            m = regex.match(lbl)
            if not m:
                continue
            raw_prefix = m.group("prefix").strip()
            words = [w for w in re.split(r"\s+", raw_prefix) if w]
            if not (1 <= len(words) <= 2):
                continue
            if any(w.lower() in PREFIX_STOPWORDS for w in words):
                continue
            if not all(re.match(r"^[A-Za-z\-]+$", w) for w in words):
                continue
            prefix = " ".join(words)
            return prefix, core

    low = (fid or "").lower()
    if re.search(r"outside_?diameter", low): return None, "outside_diameter"
    if re.search(r"inside_?diameter",  low): return None, "inside_diameter"
    if re.search(r"_diameter\b",       low): return None, "diameter"
    if re.search(r"\bvoltage\b",       low): return None, "voltage"
    if re.search(r"\bcurrent\b",       low): return None, "current"
    if re.search(r"\bpressure\b",      low): return None, "pressure"
    if re.search(r"\bflow\b",          low): return None, "flow"
    if re.search(r"\bweight\b",        low): return None, "weight"
    if re.search(r"\bmass\b",          low): return None, "mass"
    if re.search(r"\btemperature\b",   low): return None, "temperature"
    if re.search(r"\bthread_?size\b",  low): return None, "thread_size"
    if re.search(r"\bpipe_?size\b",    low): return None, "pipe_size"
    return None, None

def generate_compound_aliases(field: dict) -> List[str]:
    fid = field.get("id","")
    label = field.get("label","")
    prefix, core = extract_prefix_core(label, fid)
    out: List[str] = []
    if prefix and core and core in CORE_ALIASES:
        for var in CORE_ALIASES[core]:
            out.append(f"{prefix} {var}")
    return out

def is_scoped(fid: str) -> bool:
    """True if the field is a scoped variant (e.g., wheel_diameter)."""
    if fid in {"diameter","outside_diameter","inside_diameter"}:
        return False
    return (
        fid.endswith("_diameter")
        or "_outside_diameter" in fid
        or "_inside_diameter" in fid
    )

def is_bare_od_id(alias: str) -> bool:
    return norm_case_space(alias) in {"od","o.d.","id","i.d."}


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Deterministic alias generator for hard-spec fields")
    ap.add_argument("--input", default="fields.semantic.json")
    ap.add_argument("--out-fields", default="fields.enriched.json")
    ap.add_argument("--out-alias", default="alias_map.enriched.json")
    ap.add_argument("--audit-dir", default="audit_aliases")

    ap.add_argument("--max-aliases", type=int, default=5, help="Max new aliases to add per field")
    ap.add_argument("--only-ids", default="", help="Comma-separated field ids to process (optional)")
    ap.add_argument("--families", default=",".join(sorted(DEFAULT_HARD_FAMILIES)),
                    help="Comma-separated unit families to consider 'hard'")
    ap.add_argument("--dry-run", action="store_true", help="Write audits only; do not mutate outputs")
    ap.add_argument("--block-bare-odid-on-scoped", action="store_true",
                    help="If set, bare OD/ID are blocked on scoped fields (compounds only)")
    args = ap.parse_args()

    data = load_json(Path(args.input))
    fields = data.get("fields", [])
    if not fields:
        raise SystemExit("[error] no fields in input")

    hard_families = {x.strip() for x in args.families.split(",") if x.strip()}
    only_ids = set([x.strip() for x in args.only_ids.split(",") if x.strip()]) if args.only_ids else set()

    # 1) choose working fields
    work_fields: List[dict] = []
    for f in fields:
        if not is_hard_field(f, hard_families):
            continue
        if only_ids and f.get("id", "") not in only_ids:
            continue
        work_fields.append(f)

    # 2) pre-enrich with deterministic seeds (log as accepted)
    accepted_rows, blocked_rows, collisions_rows, orphans_rows = [], [], [], []
    for f in work_fields:
        added = enrich_field_aliases(f)
        for t in added:
            accepted_rows.append({"field_id": f.get("id",""), "candidate": t, "source": "seed_enrich"})

    # 3) build safe alias map (captures collisions for existing seeds too)
    alias_map: Dict[str, str] = {}

    def _safe_add_alias(txt: str, fid: str) -> bool:
        key = txt  # keep case
        if not key:
            return False
        if key in alias_map and alias_map[key] != fid:
            collisions_rows.append({
                "alias": key,
                "existing_field": alias_map[key],
                "new_field": fid,
                "reason": "global_collision_seed_or_candidate"
            })
            return False
        alias_map[key] = fid
        return True

    for f in fields:
        fid = f.get("id","")
        _safe_add_alias(f.get("label",""), fid)
        for a in f.get("aliases", []):
            _safe_add_alias(a.get("text",""), fid)

    # global seed set (normalized) for fast collision check
    global_seed_set = set(norm_case_space(k) for k in alias_map.keys())

    summary_rows: List[Dict[str, Any]] = []

    # 4) generate deterministic candidates (base+compound), filter, and accept
    for f in work_fields:
        fid = f.get("id", "")
        label = f.get("label", "")
        existing_alias_texts = {a.get("text","") for a in f.get("aliases", [])}
        seeds_norm = {norm_case_space(fid), norm_case_space(label)} | {norm_case_space(x) for x in existing_alias_texts}

        candidates: List[str] = []

        # base (generic) additions tied to id/helpers; skip already attached
        base_add: List[str] = []
        base_add += SEED_ENRICH.get(fid, [])
        if not base_add:
            low = fid.lower()
            for h in ID_HELPERS:
                if h["match"].search(low):
                    base_add = SEED_ENRICH.get(h["use"], [])
                    break
        for t in base_add:
            if t not in candidates and t not in existing_alias_texts:
                candidates.append(t)

        # compound from label (e.g., 'Wheel Dia', 'Spindle OD')
        compound = generate_compound_aliases(f)
        for t in compound:
            if t not in candidates and t not in existing_alias_texts:
                candidates.append(t)

        added = 0
        for raw_alias in candidates:
            alias_norm = norm_case_space(raw_alias)

            # skip duplicates of this field's seeds
            if alias_norm in seeds_norm:
                continue

            # global collision (other field already owns it)
            if alias_norm in global_seed_set:
                if alias_norm not in seeds_norm:
                    collisions_rows.append({
                        "field_id": fid,
                        "candidate": raw_alias,
                        "reason": "collision_other_field_seed"
                    })
                    blocked_rows.append({"field_id": fid, "candidate": raw_alias, "reason": "collision_other_field_seed"})
                continue

            # safety filters
            bad, why = is_blocked_text(raw_alias)
            if bad:
                blocked_rows.append({"field_id": fid, "candidate": raw_alias, "reason": why})
                continue

            # optional guard: bare OD/ID only on unscoped fields
            if args.block_bare_odid_on_scoped and is_scoped(fid) and is_bare_od_id(raw_alias):
                blocked_rows.append({"field_id": fid, "candidate": raw_alias, "reason": "bare_acronym_requires_prefix"})
                continue

            # whitelist gate
            if not passes_whitelist(fid, raw_alias):
                blocked_rows.append({"field_id": fid, "candidate": raw_alias, "reason": "not_in_whitelist"})
                continue

            # accept
            if not args.dry_run:
                f.setdefault("aliases", [])
                f["aliases"].append({"text": raw_alias, "confidence": 1.0})
                # add to alias_map safely (log if collision)
                if not _safe_add_alias(raw_alias, fid):
                    # we treat failure to add as block
                    blocked_rows.append({"field_id": fid, "candidate": raw_alias, "reason": "collision_on_insert"})
                    continue

                # update global normalized set so next fields see it
                global_seed_set.add(alias_norm)

            accepted_rows.append({"field_id": fid, "candidate": raw_alias, "source": "compound_or_base"})
            added += 1
            if added >= args.max_aliases:
                break

        # orphan = got no additions at all (neither seeds nor candidates)
        if added == 0 and not any(r["field_id"] == fid and r.get("source") == "seed_enrich" for r in accepted_rows):
            orphans_rows.append({"field_id": fid, "label": label})

        summary_rows.append({
            "field_id": fid,
            "label": label,
            "existing_aliases": len(existing_alias_texts),
            "added": added,
        })

    # 5) audits
    audit_dir = Path(args.audit_dir)
    write_jsonl(audit_dir / "accepted.jsonl", accepted_rows)
    write_jsonl(audit_dir / "blocked.jsonl", blocked_rows)
    write_jsonl(audit_dir / "collisions.jsonl", collisions_rows)
    write_jsonl(audit_dir / "orphans.jsonl", orphans_rows)
    write_csv(audit_dir / "summary.csv", summary_rows, ["field_id","label","existing_aliases","added"])

    # 6) outputs
    if not args.dry_run:
        write_json(Path(args.out_fields), {
            "version": 1,
            "generated_at": now_iso(),
            "source": {"deterministic": True},
            "fields": fields
        })
        write_json(Path(args.out_alias), {
            "version": 1,
            "generated_at": now_iso(),
            "aliases": alias_map
        })

    print(f"[summary] processed={len(work_fields)} hard fields | accepted={len(accepted_rows)} blocked={len(blocked_rows)} orphans={len(orphans_rows)}")
    print(f"[ok] audits -> {audit_dir}")
    if not args.dry_run:
        print(f"[ok] wrote {args.out_fields}")
        print(f"[ok] wrote {args.out_alias}")


if __name__ == "__main__":
    main()