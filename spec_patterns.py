"""
spec_patterns.py
----------------
Strict, conservative parsing & classification helpers for spec values.

Design goals:
- "numeric" ONLY if the value is a clean scalar (unitless OR scalar + KNOWN unit from UNIT_MAP).
- Family only when all unit-bearing samples map to a single family.
- Ranges, duals, and thread-like values are NOT numeric (but thread gets family="thread").
- Supports inch/foot quote tokens (", ', and Unicode primes/smart quotes).
- Accepts mixed fractions with hyphen (e.g., 1-1/2").
- Handles thousands separators and leading +/-.
- Conservative early veto for comma-separated or free-text tokens (e.g., STD, Optional).

Depends on spec_lexicon.py for:
    UNIT_MAP: dict[alias_unit] -> (family, canonical_unit)
    unit_to_family(unit_token) -> (family or None, canonical or None)
"""

from __future__ import annotations

import re
from typing import Optional, Tuple, Dict, Any, List
from collections import defaultdict

from spec_lexicon import UNIT_MAP, unit_to_family


# ---------------------- Normalization helpers ----------------------

UNICODE_FRAC = {
    "½": "1/2", "¼": "1/4", "¾": "3/4",
    "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
    "⅐": "1/7", "⅑": "1/9", "⅒": "1/10",
    "⅓": "1/3", "⅔": "2/3",
    "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5",
    "⅙": "1/6", "⅚": "5/6",
}

QUOTE_NORMALIZE_MAP = {
    "″": '"', "”": '"', "“": '"',   # double quotes → "
    "′": "'", "’": "'", "‘": "'",   # single quotes → '
}

def normalize_quotes(s: str) -> str:
    for bad, good in QUOTE_NORMALIZE_MAP.items():
        s = s.replace(bad, good)
    return s

def normalize_fractions(s: str) -> str:
    for u, r in UNICODE_FRAC.items():
        s = s.replace(u, r)
    return s

def normalize_degrees(s: str) -> str:
    # Normalize combined degree symbols to ascii-degree-prefixed tokens
    return s.replace("℉", "°f").replace("℃", "°c")

def normalize_spaces(s: str) -> str:
    # NBSP and narrow NBSP → space
    return s.replace("\u00A0", " ").replace("\u202F", " ")

def normalize_value(s: str) -> str:
    return normalize_spaces(
        normalize_degrees(
            normalize_quotes(
                normalize_fractions((s or "").strip())
            )
        )
    )


# ---------------------- Trailing unit de-duplication ----------------------
# Collapse repeated trailing unit aliases (e.g., "in in", 'in."', '" in') to a single canonical unit.

ALIAS_GROUPS = defaultdict(list)  # canonical_unit -> [aliases...]
for alias, (fam, canon) in UNIT_MAP.items():
    ALIAS_GROUPS[canon].append(re.escape(alias))

DEDUP_PATTERNS: Dict[str, re.Pattern] = {}
for canon, esc_aliases in ALIAS_GROUPS.items():
    # allow optional trailing period on aliases (e.g., "in.")
    alias_union = r"(?:%s)\.?" % "|".join(sorted(esc_aliases, key=len, reverse=True))
    # Two or more unit aliases at the very END of the string (with optional whitespace between)
    pat = re.compile(rf"\s*(?:{alias_union})(?:\s+(?:{alias_union}))+?$", re.IGNORECASE)
    DEDUP_PATTERNS[canon] = pat

def dedupe_trailing_unit_aliases(s: str) -> str:
    """Collapse repeated trailing unit aliases to a single canonical unit token."""
    t = s
    for canon, pat in DEDUP_PATTERNS.items():
        if pat.search(t):
            # replace the whole repeated suffix with a single space + canonical unit
            t = pat.sub(f" {canon}", t)
    return t


# ---------------------- Core regexes ----------------------

# Number token for fraction-aware comparisons
NUM_FRACTION = r'(?:\d+(?:\.\d+)?|\d+[ -]\d+/\d+|\d+/\d+)'

# Strict scalar:
# - optional +/- sign
# - integer with thousands separators OR plain int/float OR mixed fraction (space or hyphen) OR simple fraction
# - optional unit (letters / degree sign / quotes), optional trailing '.'
SCALAR_RE = re.compile(
    r"""^\s*
        ([+-]?(?:
            \d{1,3}(?:,\d{3})*(?:\.\d+)?   # 1,234 or 1,234.56
            |
            \d+(?:\.\d+)?                  # 12 or 12.34
            |
            \d+[ -]\d+/\d+                 # 1 1/2 or 1-1/2
            |
            \d+/\d+                        # 1/2
        ))
        \s*([A-Za-z°"']+\.?)?\s*           # optional unit; allow trailing period
        $
    """,
    re.VERBOSE,
)

# Range-like (e.g., "0–100", "5 - 10", "1-3/8 - 2-1/4")
RANGE_RE = re.compile(fr"\b{NUM_FRACTION}\s*[-–—~]\s*{NUM_FRACTION}\b")

# Textual/inequality ranges (e.g., "0 to 100", "±0.01", "<= 5", ">=10", "up to 50")
RANGE_WORD_RE = re.compile(
    fr"(?:\b{NUM_FRACTION}\b)\s*(?:to|through|thru|upto|up\s*to|±|\+/-|≤|>=|<=|≥|<|>)\s*(?:\b{NUM_FRACTION}\b)",
    re.I,
)

# Dual numeric:
# Tightened so plain fractions (e.g., 1/2") don't match as duals.
# We require either trailing hyphen number (1/4-20) OR a unit-ish character after the slash.
DUAL_NUM_RE = re.compile(r"\b\d+\s*/\s*\d+\s*(?:-\s*\d+|[A-Za-z°\"'])\b")

# Thread-like patterns (metric, imperial UNC/UNF/UN, NPT, TPI)
# Tight imperial clause: require no spaces around hyphen, restrict TPI to 2–3 digits when present.
THREAD_RE = re.compile(
    r"(?i)\b("
    r"m\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?"
    r"|npt\b"
    r"|bsp\b"
    r"|unc\b|unf\b|un\b"
    r"|tpi\b"
    r"|\d+/\d+-\d{2,3}(?:\s*(?:unc|unf|un))?"
    r")\b"
)


# ---------------------- Matching helpers ----------------------

def match_scalar_with_unit(value: str) -> Optional[Tuple[str, str]]:
    """
    Return (number_str, unit_token) if value is a strict scalar with optional unit.
    unit_token is '' for unitless scalars.
    If a unit is present, it MUST be known in UNIT_MAP (strict).
    """
    m = SCALAR_RE.match(value)
    if not m:
        return None

    num = m.group(1)
    unit_raw = (m.group(2) or "").strip()

    # Strip a trailing period from unit (e.g., "in." -> "in")
    if unit_raw.endswith("."):
        unit_raw = unit_raw[:-1]

    utok = unit_raw.lower().strip()

    # Normalize quote tokens explicitly (we already normalized Unicode → ascii above)
    if utok == '"':
        utok = '"'
    elif utok == "'":
        utok = "'"

    # If unit present, it must be recognized
    if utok and utok not in UNIT_MAP:
        return None

    return num, utok  # utok may be '' (unitless)


# ---------------------- Classifier ----------------------

STOPWORD_FREE_TEXT_RE = re.compile(
    r"\b(std|optional|opt|each|pair|set|approx|approx\.|about|min|max)\b", re.I
)
MULTI_VALUE_LIST_RE = re.compile(r"(?:\b(?:and)\b|&|(?:\s+/\s+))", re.I)
GAUGE_NUMBER_RE = re.compile(r"\b(?:no\.?\s*\d+|#\s*\d+|awg\b)\b", re.I)
LEADING_DIAMETER_RE = re.compile(r"^[\s]*[⌀Ø]\s*")

def classify_value_strict(raw: str) -> Dict[str, Any]:
    """
    Classify a single spec value string.
    Returns a dict with:
      kind:    "scalar" | "range" | "dual" | "thread" | "unknown"
      unit:    canonical unit token or '' if unitless
      family:  family string (e.g., "length") or None
      reason:  brief reason string to aid debugging
    """
    if not isinstance(raw, str) or not raw.strip():
        return {"kind": "unknown", "unit": "", "family": None, "reason": "empty"}

    s = normalize_value(raw)

    # Collapse repeated trailing unit aliases (e.g., "in in", 'in."', '" in')
    s = dedupe_trailing_unit_aliases(s)
    # Run twice in case mixed aliases collapse in two steps
    s = dedupe_trailing_unit_aliases(s)

    # Strip leading diameter symbol (⌀, Ø) conservatively; still require clean scalar after.
    s = LEADING_DIAMETER_RE.sub("", s)

    # Early conservative vetoes for non-numeric catalog-ish text
    if "," in s:
        return {"kind": "unknown", "unit": "", "family": None, "reason": "comma_list"}
    if STOPWORD_FREE_TEXT_RE.search(s):
        return {"kind": "unknown", "unit": "", "family": None, "reason": "free_text_tokens"}
    if MULTI_VALUE_LIST_RE.search(s):
        return {"kind": "unknown", "unit": "", "family": None, "reason": "multi_value_list"}
    if GAUGE_NUMBER_RE.search(s):
        return {"kind": "unknown", "unit": "", "family": None, "reason": "gauge_or_number_size"}

    # 1) Threads (non-numeric)
    if THREAD_RE.search(s):
        return {"kind": "thread", "unit": "", "family": "thread", "reason": "thread_pattern"}

    # 2) Ranges (symbols and words) (non-numeric)
    if RANGE_RE.search(s):
        return {"kind": "range", "unit": "", "family": None, "reason": "range_like"}
    if RANGE_WORD_RE.search(s):
        return {"kind": "range", "unit": "", "family": None, "reason": "range_word"}

    # 3) Scalars (numeric candidates) — must come BEFORE dual check so 1/2" is scalar
    m = match_scalar_with_unit(s)
    if m:
        _, utok = m
        if utok:
            fam, canon = UNIT_MAP[utok]  # map alias to (family, canonical_unit)
            return {"kind": "scalar", "unit": canon, "family": fam, "reason": "scalar_known_unit"}
        else:
            return {"kind": "scalar", "unit": "", "family": None, "reason": "scalar_unitless"}

    # 4) Duals (non-numeric)
    if DUAL_NUM_RE.search(s):
        return {"kind": "dual", "unit": "", "family": None, "reason": "dual_like"}

    # 5) Unknown / free text / unrecognized units
    return {"kind": "unknown", "unit": "", "family": None, "reason": "no_scalar_match"}


# ---------------------- Family aggregation ----------------------

def family_from_units(units: List[str]) -> Optional[str]:
    """
    Given canonical unit tokens (e.g., ['in','mm','in','']), return the single family
    if all unit-bearing tokens belong to the same family; else None.
    Unitless ('') entries are ignored for the unanimity check.
    If any unit is unknown (shouldn't happen after match_scalar_with_unit), return None.
    """
    families = set()
    for u in units:
        if not u:
            continue
        fam, _canon = unit_to_family(u)  # returns (family, canonical) or (None, None)
        if fam:
            families.add(fam)
        else:
            return None
    return next(iter(families)) if len(families) == 1 else None


# ---------------------- TOLERANCES ----------------------

# Absolute or relative tolerance per field or family.
# Values are interpreted as absolute deltas in canonical units
# (inches, psi, pounds, etc.) unless they start with "rel:".

TOLERANCES = {
    # ---- family-level defaults ----
    "length": 0.0625,        # 1/16 in
    "pressure": 10.0,        # ±10 psi
    "mass": 0.5,             # ±0.5 lb
    "weight": 0.5,
    "torque": 0.1,           # ±0.1 lb-ft
    "flow": 0.2,             # ±0.2 gpm
    "temperature": 2.0,      # ±2 °F
    "thread": 0.001,         # very tight tolerance for pitch
    "power": 5.0,            # ±5 W
    "frequency": 1.0,        # ±1 Hz

    # ---- field-level overrides (optional) ----
    "wheel_diameter": 0.125,      # 1/8 in window
    "thread_size": 0.002,         # fine pitch tolerance
    "outside_diameter": 0.0625,
    "inside_diameter": 0.0625,
    "base_diameter": 0.0625,
    "mount_hole_diameter": 0.03125,  # 1/32 in
    "length_overall": 0.25,
    "weight_capacity": 1.0,        # ±1 lb or unitless
    "load_capacity_range": "rel:0.05",  # ±5% relative

    # fallback
    "default": 0.25,  # absolute fallback in canonical units
}