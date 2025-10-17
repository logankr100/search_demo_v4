# spec_lexicon.py
# ----------------
# Conservative, auditable lexicon defining unit families, canonical units,
# token aliases, and name-level priors (identifiers/categories).
#
# You can import this anywhere:
#   from spec_lexicon import UNIT_MAP, FAMILY_UNITS, IDENTIFIER_KEYS, CATEGORICAL_KEYS, META_DROP_KEYS, FAMILY_NAME_HINTS

from typing import Dict, Tuple, Set, List

# -------------------------------
# Unit Families & Canonical Units
# -------------------------------
# FAMILY_UNITS is used for reporting / validation.
# UNIT_MAP is the authoritative token -> (family, canonical_unit) mapping.
# Keep UNIT_MAP tight; add tokens only when you’re sure.


FAMILY_UNITS: Dict[str, Set[str]] = {
    "length": {"in", "mm", "cm", "ft"},
    "pressure": {"psi", "bar", "kpa"},
    "voltage": {"v"},
    "current": {"a"},
    "frequency": {"hz"},
    "temperature": {"f", "c"},
    "mass": {"lb", "kg", "g"},
    "flow": {"cfm", "scfm", "gpm", "lpm"},
    "power": {"w", "kw", "hp"},
}

UNIT_MAP: Dict[str, Tuple[str, str]] = {
    # ----- LENGTH -----
    "in": ("length", "in"),
    "inch": ("length", "in"),
    "inches": ("length", "in"),
    '"': ("length", "in"),
    "″": ("length", "in"),   # double prime
    "”": ("length", "in"),   # smart quote
    "ft": ("length", "ft"),
    "foot": ("length", "ft"),
    "feet": ("length", "ft"),
    "'": ("length", "ft"),
    "′": ("length", "ft"),   # prime
    "’": ("length", "ft"),   # smart quote
    "mm": ("length", "mm"),
    "millimeter": ("length", "mm"),
    "millimeters": ("length", "mm"),
    "cm": ("length", "cm"),
    "centimeter": ("length", "cm"),
    "centimeters": ("length", "cm"),

    # ----- PRESSURE -----
    "psi": ("pressure", "psi"),
    "psig": ("pressure", "psi"),
    "psia": ("pressure", "psi"),
    "bar": ("pressure", "bar"),
    "kpa": ("pressure", "kpa"),

    # ----- ELECTRICAL -----
    "v": ("voltage", "v"),
    "volt": ("voltage", "v"),
    "volts": ("voltage", "v"),
    "a": ("current", "a"),
    "amp": ("current", "a"),
    "amps": ("current", "a"),
    "ampere": ("current", "a"),
    "amperes": ("current", "a"),
    "hz": ("frequency", "hz"),
    "hertz": ("frequency", "hz"),

    # ----- MASS -----
    "lb": ("mass", "lb"),
    "lbs": ("mass", "lb"),
    "pound": ("mass", "lb"),
    "pounds": ("mass", "lb"),
    "kg": ("mass", "kg"),
    "g": ("mass", "g"),
    "gram": ("mass", "g"),
    "grams": ("mass", "g"),

    # ----- FLOW -----
    "cfm": ("flow", "cfm"),
    "scfm": ("flow", "cfm"),
    "gpm": ("flow", "gpm"),
    "lpm": ("flow", "lpm"),

    # ----- POWER -----
    "w": ("power", "w"),
    "watt": ("power", "w"),
    "watts": ("power", "w"),
    "kw": ("power", "kw"),
    "kilowatt": ("power", "kw"),
    "kilowatts": ("power", "kw"),
    "hp": ("power", "hp"),
    "horsepower": ("power", "hp"),

    # ----- TEMPERATURE -----
    "°f": ("temperature", "f"),
    "f": ("temperature", "f"),
    "°c": ("temperature", "c"),
    "c": ("temperature", "c"),
}

# -------------------------------
# Known non-numeric field ids/names (hard pins)
# -------------------------------
# Apply after case/space folding and snake_case.
IDENTIFIER_KEYS: Set[str] = {
    "mpn", "manufacturers_part_number", "manufacturer_part_number", "mfr_part_number",
    "mfg_p_n", "mfg_pn", "part_number", "part_", "part_no", "item_number", "item_no",
    "sku", "upc", "ean", "gtin", "isbn"
}

CATEGORICAL_KEYS: Set[str] = {
    "brand", "manufacturer", "color", "colour", "material", "finish",
    "series", "style", "type", "warranty", "package_quantity", "description"
}

# -------------------------------
# Meta/marketing keys to drop early if desired
# -------------------------------
META_DROP_KEYS: Set[str] = {
    "price_jsonld", "currency_jsonld", "meta_description", "og_description",
    "price", "currency", "msrp", "sale_price"
}

# -------------------------------
# Family name hints (weak priors)
# -------------------------------
# If the field id contains these tokens, you *may* nudge the family (for reporting only).
# Do not use hints to force numeric; they are *advisory*.
FAMILY_NAME_HINTS: Dict[str, str] = {
    # length-ish
    "length": "length", "width": "length", "height": "length", "depth": "length",
    "diameter": "length", "dia": "length", "od": "length", "id": "length",
    "dimension_a": "length", "dimension_b": "length", "dimension_c": "length",

    # pressure-ish
    "pressure": "pressure",

    # electrical-ish
    "voltage": "voltage", "volt": "voltage", "current": "current", "amp": "current",
    "frequency": "frequency", "hz": "frequency",

    # flow-ish
    "cfm": "flow", "scfm": "flow", "gpm": "flow", "lpm": "flow",

    # power-ish
    "watt": "power", "kw": "power", "hp": "power",

    # thread-ish (never numeric)
    "thread": "thread",
}

# -------------------------------
# Thread tokens (for detection/reporting only)
# -------------------------------
THREAD_TOKENS: Set[str] = {"npt", "bsp", "unc", "unf", "un", "tpi"}

# For unit lookup convenience
def unit_to_family(unit_token: str):
    """Return (family, canonical_unit) or (None, None) if unknown."""
    return UNIT_MAP.get(unit_token.lower().strip(), (None, None))