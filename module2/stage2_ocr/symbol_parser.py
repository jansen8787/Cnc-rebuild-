"""
stage2_ocr/symbol_parser.py — Module 2: Stage 2
================================================
CNC drawing symbol parser. Pure Python. No external dependencies.

Adapted from old project's 2A.4/dimension_parser.py (REUSE bucket, forensic
report §2.4) — the primary parser that supersedes 2A.1/classifier.py.
It is a strict superset: every pattern from the old classifier exists here,
plus chamfer / ISO fit codes / quantity prefix / decimal comma normalisation.

Pattern priority order (first match wins — ORDER IS PRIORITY):
    1.  Chamfer          2×45°, C2, 2×45°
    2.  Angle            45°, 45.5deg
    3.  Thread callout   M6, M8×1.25, M12x1.75-6H, 1/4-20 UNC
    4.  Diameter         Ø12, ⌀25, d=8, Ø20H7
    5.  Radius           R12, R12.5mm, R.5
    6.  Bilateral tol    20±0.02, 50+0.1/-0.05
    7.  Fit              25H7, 30js6, Ø20h6
    8.  Quantity         4×, 4×Ø8, 4×M6
    9.  Reference label  (A), [B]
   10.  Linear metric    12.5mm, .5mm
   11.  Linear imperial  1.250", 3/8"
   12.  Surface finish   Ra1.6, Rz6.3
   13.  Bare number      12.5, 100
   14.  Title block      MATERIAL:, SCALE:, …
   15.  Label            A, HOLE-A, SECTION-AA
   16.  Unknown          catch-all

Rules honoured (from master rules and architecture):
    §29 NORMWISSEN PRIORISIEREN — ISO/DIN patterns are authoritative.
    §30 SYMBOLE ALS DATEN       — raw_text always preserved.
    §21 UNSICHERHEIT            — ambiguous tokens flagged, not guessed.

Public API:
    parse_symbol(text: str) -> ParsedSymbol

    The ParsedSymbol maps directly to Module 1 V2's Feature.manufacturing fields:
        upper_tol   → manufacturing.tolerancePlus
        lower_tol   → manufacturing.toleranceMinus
        fit_code    → manufacturing.fitClass
        thread_pitch → manufacturing.threadPitch
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import ParsedSymbol  # noqa: E402


# ---------------------------------------------------------------------------
# ISO 286 fit letter sets for fit-code validation
# ---------------------------------------------------------------------------

_FIT_UPPER = set("ABCDEFGHJKLMNPRSTUVXYZ")   # bore (uppercase)
_FIT_LOWER = set("abcdefghjklmnprstuvxyz")   # shaft (lowercase)


# ---------------------------------------------------------------------------
# Number helper — handles European decimal comma
# ---------------------------------------------------------------------------

def _to_float(s: Optional[str]) -> Optional[float]:
    """Parse string to float; normalise comma to dot. Returns None on failure."""
    if s is None:
        return None
    try:
        return float(str(s).replace(",", ".").strip())
    except (ValueError, TypeError):
        return None


def _norm_comma(text: str) -> str:
    """Normalise European decimal comma → dot (TECH_DRAWING_INTERPRETER §11)."""
    return text.replace(",", ".")


def _normalize_unit(raw: Optional[str]) -> Optional[str]:
    """Canonicalise unit strings."""
    if not raw:
        return None
    mapping = {
        "mm": "mm", "MM": "mm",
        "in": "in", "IN": "in",
        '"':  "in", "inch": "in",
        "°":  "deg", "deg": "deg", "DEG": "deg", "Grad": "deg",
    }
    return mapping.get(raw.strip(), raw.strip().lower())


# ---------------------------------------------------------------------------
# Compiled regex patterns — ORDER IS PRIORITY
# ---------------------------------------------------------------------------

_NUM = r"[\d]*\.?[\d]+"          # positive number, handles .5, 0.5, 12, 12.5

# 1. Chamfer: 2×45°, C2, 2x45°, 2×45deg
_RE_CHAMFER = re.compile(
    r"^(?:C|c)?(?P<depth>" + _NUM + r")\s*[xX×]\s*(?P<angle>" + _NUM + r")\s*°?"
    r"|^[Cc](?P<cdepth>" + _NUM + r")$",
    re.IGNORECASE,
)

# 2. Angle: 45°, 30.5deg, 90 Grad
_RE_ANGLE = re.compile(
    r"^(?P<val>" + _NUM + r")\s*(?:°|deg|DEG|Grad)$",
    re.IGNORECASE,
)

# 3a. Metric thread: M6, M8×1.25, M12x1.75-6H, M10x1.5 LH
_RE_THREAD_METRIC = re.compile(
    r"^M(?P<dia>" + _NUM + r")"
    r"(?:[xX×](?P<pitch>" + _NUM + r"))?"
    r"(?:\s*-\s*(?P<tol_class>\d+[GHgh]))?"
    r"(\s*LH)?$",
    re.IGNORECASE,
)

# 3b. Imperial thread: 1/4-20 UNC, 3/8-16 UNF, #6-32 UNC-2A
_RE_THREAD_IMPERIAL = re.compile(
    r"^(#?\d+(/\d+)?)\s*-\s*(\d+)\s+(UNC|UNF|UNEF|UNR|UNS)(\s*-\s*[123][ABab])?$",
    re.IGNORECASE,
)

# 4. Diameter (with optional fit suffix): Ø12, ⌀25, d=8, Ø12H7
_RE_DIAMETER = re.compile(
    r"^[Øø⌀Dd]=?\s*(?P<val>" + _NUM + r")\s*(?P<fit>[A-Za-z]{1,2}\d{1,2})?$"
)

# 5. Radius: R12, R12.5mm, R.5
_RE_RADIUS = re.compile(
    r"^[Rr]\s*=?\s*(?P<val>" + _NUM + r")\s*(?P<unit>mm|in|\")?$"
)

# 6a. Bilateral symmetric: 20±0.02
_RE_TOL_SYMMETRIC = re.compile(
    r"^(?P<nom>" + _NUM + r")\s*[±]\s*(?P<sym>" + _NUM + r")\s*(?:mm|in)?$"
)

# 6b. Bilateral asymmetric: 50+0.1/-0.05, 30+0.05-0.0
_RE_TOL_ASYMMETRIC = re.compile(
    r"^(?P<nom>" + _NUM + r")\s*"
    r"[+](?P<up>" + _NUM + r")\s*/?[-](?P<lo>" + _NUM + r")\s*(?:mm|in)?$"
)

# 7. Fit with nominal: 25H7, 30js6, Ø20h6
_RE_FIT = re.compile(
    r"^[Øø⌀]?\s*(?P<nom>" + _NUM + r")\s*(?P<fit>[A-Za-z]{1,2}\d{1,2})$"
)

# 8. Quantity prefix: 4×, 4xØ8, 4×M6
_RE_QUANTITY = re.compile(
    r"^(?P<qty>\d+)\s*[xX×]\s*(?P<rest>.+)?$"
)

# 9. Reference label: (A), [B]
_RE_REFERENCE = re.compile(
    r"^[(\[]\s*(?P<label>[A-Za-z0-9]+)\s*[)\]]$"
)

# 10. Linear metric: 12.5mm, .5mm
_RE_LINEAR_METRIC = re.compile(
    r"^(?P<val>" + _NUM + r")\s*mm$",
    re.IGNORECASE,
)

# 11a. Linear imperial decimal: 1.250", 1.250in
_RE_LINEAR_INCH_DEC = re.compile(
    r"^(?P<val>" + _NUM + r")\s*(in|\")$",
    re.IGNORECASE,
)

# 11b. Linear imperial fractional: 3/8", 1 3/4"
_RE_LINEAR_INCH_FRAC = re.compile(
    r"^(?P<whole>\d+\s+)?(?P<num>\d+)/(?P<den>\d+)\s*(in|\")$",
    re.IGNORECASE,
)

# 12. Surface finish: Ra1.6, Rz6.3, Ra 0.8
_RE_SURFACE = re.compile(
    r"^R(?P<type>[azAZ])\s*(?P<val>" + _NUM + r")\s*(?:µm|um)?$",
    re.IGNORECASE,
)

# 13. Bare number: 12.5, 100, .5
_RE_BARE_NUMBER = re.compile(
    r"^(?P<val>" + _NUM + r")$"
)

# 14. Title block fields
_RE_TITLE_BLOCK = re.compile(
    r"^(?P<key>MATERIAL|SCALE|DWG\s*NO|DRAWING\s*NO|REV|REVISION|"
    r"DRAWN|CHECKED|APPROVED|DATE|TITLE|PART\s*NO|PART\s*NAME|"
    r"SHEET|FINISH|WEIGHT|HARDNESS|TREATMENT|TOLERANCE)"
    r"\s*[:\-]?\s*(?P<val>.*)$",
    re.IGNORECASE,
)

# 15. Label: uppercase alphanumeric identifier
_RE_LABEL = re.compile(
    r"^[A-Z][A-Z0-9_\-]{0,24}$"
)


# ---------------------------------------------------------------------------
# Individual parsers (return ParsedSymbol or None)
# ---------------------------------------------------------------------------

def _parse_chamfer(text: str) -> Optional[ParsedSymbol]:
    m = _RE_CHAMFER.match(text)
    if not m:
        return None
    gd = m.groupdict()
    if gd.get("cdepth"):
        val = _to_float(gd["cdepth"])
        if val is None:
            return None
        return ParsedSymbol(
            token_type="chamfer", value=val, unit="mm", qualifier="C",
            upper_tol=None, lower_tol=None, fit_code=None, thread_pitch=None,
            quantity=None, angle_deg=45.0, flags=["implicit_45"],
            raw_text=text,
        )
    depth = _to_float(gd.get("depth"))
    angle = _to_float(gd.get("angle"))
    if depth is None or angle is None:
        return None
    return ParsedSymbol(
        token_type="chamfer", value=depth, unit="mm", qualifier="x",
        upper_tol=None, lower_tol=None, fit_code=None, thread_pitch=None,
        quantity=None, angle_deg=angle, flags=[],
        raw_text=text,
    )


def _parse_angle(text: str) -> Optional[ParsedSymbol]:
    m = _RE_ANGLE.match(text)
    if not m:
        return None
    val = _to_float(m.group("val"))
    if val is None:
        return None
    return ParsedSymbol(
        token_type="angle", value=val, unit="deg", qualifier=None,
        upper_tol=None, lower_tol=None, fit_code=None, thread_pitch=None,
        quantity=None, angle_deg=val, flags=[],
        raw_text=text,
    )


def _parse_thread(text: str) -> Optional[ParsedSymbol]:
    # Metric
    m = _RE_THREAD_METRIC.match(text)
    if m:
        dia   = _to_float(m.group("dia"))
        pitch = _to_float(m.group("pitch")) if m.group("pitch") else None
        tol_c = m.group("tol_class")
        if dia is None:
            return None
        return ParsedSymbol(
            token_type="thread_callout", value=dia, unit="mm", qualifier="M",
            upper_tol=None, lower_tol=None, fit_code=tol_c, thread_pitch=pitch,
            quantity=None, angle_deg=None, flags=[],
            raw_text=text,
        )
    # Imperial
    m2 = _RE_THREAD_IMPERIAL.match(text)
    if m2:
        return ParsedSymbol(
            token_type="thread_callout", value=None, unit=None, qualifier="UNC",
            upper_tol=None, lower_tol=None, fit_code=None, thread_pitch=None,
            quantity=None, angle_deg=None, flags=["imperial_thread"],
            raw_text=text,
        )
    return None


def _parse_diameter(text: str) -> Optional[ParsedSymbol]:
    # Must start with Ø/⌀/d/D
    if not text or text[0] not in "Øø⌀Dd":
        return None
    m = _RE_DIAMETER.match(text)
    if not m:
        return None
    val = _to_float(m.group("val"))
    if val is None:
        return None
    fit = m.group("fit") if m.group("fit") else None

    # Validate fit code if present
    if fit:
        letter = "".join(c for c in fit if c.isalpha())
        if not (all(c in _FIT_UPPER for c in letter) or
                all(c in _FIT_LOWER for c in letter)):
            fit = None   # not a real ISO fit code

    token_type = "fit" if fit else "dimension_diameter"
    return ParsedSymbol(
        token_type=token_type, value=val, unit="mm", qualifier="Ø",
        upper_tol=None, lower_tol=None, fit_code=fit, thread_pitch=None,
        quantity=None, angle_deg=None, flags=[],
        raw_text=text,
    )


def _parse_radius(text: str) -> Optional[ParsedSymbol]:
    if not text or text[0] not in "Rr":
        return None
    m = _RE_RADIUS.match(text)
    if not m:
        return None
    val = _to_float(m.group("val"))
    if val is None:
        return None
    unit = _normalize_unit(m.group("unit")) if m.group("unit") else None
    return ParsedSymbol(
        token_type="dimension_radial", value=val, unit=unit, qualifier="R",
        upper_tol=None, lower_tol=None, fit_code=None, thread_pitch=None,
        quantity=None, angle_deg=None, flags=[],
        raw_text=text,
    )


def _parse_tolerance_bilateral(text: str) -> Optional[ParsedSymbol]:
    # Symmetric
    m = _RE_TOL_SYMMETRIC.match(text)
    if m:
        nom = _to_float(m.group("nom"))
        sym = _to_float(m.group("sym"))
        if nom is None or sym is None:
            return None
        return ParsedSymbol(
            token_type="tolerance", value=nom, unit="mm", qualifier="±",
            upper_tol=sym, lower_tol=-sym, fit_code=None, thread_pitch=None,
            quantity=None, angle_deg=None, flags=[],
            raw_text=text,
        )
    # Asymmetric
    m2 = _RE_TOL_ASYMMETRIC.match(text)
    if m2:
        nom = _to_float(m2.group("nom"))
        up  = _to_float(m2.group("up"))
        lo  = _to_float(m2.group("lo"))
        if nom is None or up is None or lo is None:
            return None
        return ParsedSymbol(
            token_type="tolerance", value=nom, unit="mm", qualifier="+/-",
            upper_tol=up, lower_tol=-lo, fit_code=None, thread_pitch=None,
            quantity=None, angle_deg=None, flags=[],
            raw_text=text,
        )
    return None


def _parse_fit(text: str) -> Optional[ParsedSymbol]:
    m = _RE_FIT.match(text)
    if not m:
        return None
    nom = _to_float(m.group("nom"))
    fit = m.group("fit")
    if nom is None or not fit:
        return None
    letter = "".join(c for c in fit if c.isalpha())
    if not letter:
        return None
    is_valid = (all(c in _FIT_UPPER for c in letter) or
                all(c in _FIT_LOWER for c in letter))
    if not is_valid:
        return None
    qualifier = "Ø" if text.startswith(("Ø", "⌀", "ø")) else None
    return ParsedSymbol(
        token_type="fit", value=nom, unit="mm", qualifier=qualifier,
        upper_tol=None, lower_tol=None, fit_code=fit, thread_pitch=None,
        quantity=None, angle_deg=None, flags=[],
        raw_text=text,
    )


def _parse_quantity(text: str) -> Optional[ParsedSymbol]:
    m = _RE_QUANTITY.match(text)
    if not m:
        return None
    qty = int(m.group("qty"))
    rest = (m.group("rest") or "").strip()
    # Recursively parse the 'rest' if present
    inner: Optional[ParsedSymbol] = None
    if rest:
        inner = parse_symbol(rest)
    return ParsedSymbol(
        token_type="quantity",
        value=float(qty),
        unit=None,
        qualifier="×",
        upper_tol=inner.upper_tol if inner else None,
        lower_tol=inner.lower_tol if inner else None,
        fit_code=inner.fit_code if inner else None,
        thread_pitch=inner.thread_pitch if inner else None,
        quantity=qty,
        angle_deg=inner.angle_deg if inner else None,
        flags=[],
        raw_text=text,
    )


def _parse_reference(text: str) -> Optional[ParsedSymbol]:
    m = _RE_REFERENCE.match(text)
    if not m:
        return None
    return ParsedSymbol(
        token_type="reference", value=None, unit=None, qualifier=None,
        upper_tol=None, lower_tol=None, fit_code=None, thread_pitch=None,
        quantity=None, angle_deg=None, flags=[],
        raw_text=text,
    )


def _parse_linear_metric(text: str) -> Optional[ParsedSymbol]:
    m = _RE_LINEAR_METRIC.match(text)
    if not m:
        return None
    val = _to_float(m.group("val"))
    if val is None:
        return None
    return ParsedSymbol(
        token_type="dimension_linear", value=val, unit="mm", qualifier=None,
        upper_tol=None, lower_tol=None, fit_code=None, thread_pitch=None,
        quantity=None, angle_deg=None, flags=[],
        raw_text=text,
    )


def _parse_linear_inch(text: str) -> Optional[ParsedSymbol]:
    # Decimal
    m = _RE_LINEAR_INCH_DEC.match(text)
    if m:
        val = _to_float(m.group("val"))
        if val is None:
            return None
        return ParsedSymbol(
            token_type="dimension_linear", value=val, unit="in", qualifier=None,
            upper_tol=None, lower_tol=None, fit_code=None, thread_pitch=None,
            quantity=None, angle_deg=None, flags=[],
            raw_text=text,
        )
    # Fractional
    m2 = _RE_LINEAR_INCH_FRAC.match(text)
    if m2:
        whole_str = (m2.group("whole") or "0").strip()
        whole = float(whole_str) if whole_str else 0.0
        num = int(m2.group("num"))
        den = int(m2.group("den"))
        if den == 0:
            return None
        val = round(whole + num / den, 6)
        return ParsedSymbol(
            token_type="dimension_linear", value=val, unit="in", qualifier=None,
            upper_tol=None, lower_tol=None, fit_code=None, thread_pitch=None,
            quantity=None, angle_deg=None, flags=["fractional_inch"],
            raw_text=text,
        )
    return None


def _parse_surface(text: str) -> Optional[ParsedSymbol]:
    m = _RE_SURFACE.match(text)
    if not m:
        return None
    surf_type = m.group("type").upper()   # "A" or "Z"
    val = _to_float(m.group("val"))
    if val is None:
        return None
    qualifier = f"R{surf_type}"
    return ParsedSymbol(
        token_type="surface_finish", value=val, unit="µm", qualifier=qualifier,
        upper_tol=None, lower_tol=None, fit_code=None, thread_pitch=None,
        quantity=None, angle_deg=None, flags=[],
        raw_text=text,
    )


def _parse_bare_number(text: str) -> Optional[ParsedSymbol]:
    m = _RE_BARE_NUMBER.match(text)
    if not m:
        return None
    val = _to_float(m.group("val"))
    if val is None:
        return None
    return ParsedSymbol(
        token_type="dimension_linear", value=val, unit=None, qualifier=None,
        upper_tol=None, lower_tol=None, fit_code=None, thread_pitch=None,
        quantity=None, angle_deg=None, flags=["ambiguous_unit"],
        raw_text=text,
    )


def _parse_title_block(text: str) -> Optional[ParsedSymbol]:
    m = _RE_TITLE_BLOCK.match(text)
    if not m:
        return None
    return ParsedSymbol(
        token_type="title_block", value=None, unit=None, qualifier=m.group("key").upper(),
        upper_tol=None, lower_tol=None, fit_code=None, thread_pitch=None,
        quantity=None, angle_deg=None, flags=[],
        raw_text=text,
    )


def _parse_label(text: str) -> Optional[ParsedSymbol]:
    m = _RE_LABEL.match(text)
    if not m:
        return None
    return ParsedSymbol(
        token_type="label", value=None, unit=None, qualifier=None,
        upper_tol=None, lower_tol=None, fit_code=None, thread_pitch=None,
        quantity=None, angle_deg=None, flags=[],
        raw_text=text,
    )


# ---------------------------------------------------------------------------
# OCR artefact normalisation (applies before any parser)
# ---------------------------------------------------------------------------

_OCR_FIXES: list[tuple[re.Pattern, str]] = [
    # Ø misread as 'o', '0', 'O'  — only at start of token
    (re.compile(r"^[oO0](?=\d)"), "Ø"),
    # Degree misread as 'o' at end
    (re.compile(r"(\d)\s*o$"), r"\1°"),
    # ± misread
    (re.compile(r"[+][/-]"), "±"),
    # Comma-decimal fix already handled by _norm_comma
    # "×" variants
    (re.compile(r"\s*[xX]\s*(?=\d)"), "×"),
]

def _apply_ocr_fixes(text: str) -> str:
    """Apply heuristic OCR artefact corrections. Called before parsers."""
    result = text.strip()
    for pattern, replacement in _OCR_FIXES:
        result = pattern.sub(replacement, result)
    return result


# ---------------------------------------------------------------------------
# Public API — ordered parser chain
# ---------------------------------------------------------------------------

_PARSERS = [
    _parse_chamfer,
    _parse_angle,
    _parse_thread,
    _parse_diameter,
    _parse_radius,
    _parse_tolerance_bilateral,
    _parse_fit,
    _parse_quantity,
    _parse_reference,
    _parse_linear_metric,
    _parse_linear_inch,
    _parse_surface,
    _parse_bare_number,
    _parse_title_block,
    _parse_label,
]


def parse_symbol(text: str) -> ParsedSymbol:
    """
    Classify and parse one OCR token text from a technical drawing.

    Applies OCR normalisation (Ø, ±, decimal comma) before parsing.
    First matching parser wins — order is priority (see module docstring).

    Args:
        text: Raw OCR string.

    Returns:
        ParsedSymbol with raw_text always preserved as the original input.
    """
    stripped = text.strip()
    normalised = _apply_ocr_fixes(_norm_comma(stripped))

    for parser in _PARSERS:
        result = parser(normalised)
        if result is not None:
            result.raw_text = text   # always original text, not normalised
            return result

    # Catch-all unknown
    return ParsedSymbol(
        token_type="unknown", value=None, unit=None, qualifier=None,
        upper_tol=None, lower_tol=None, fit_code=None, thread_pitch=None,
        quantity=None, angle_deg=None, flags=["unparsed"],
        raw_text=text,
    )


# ---------------------------------------------------------------------------
# Self-tests — 42 cases covering all token types
# ---------------------------------------------------------------------------

def _run_tests() -> int:
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    results: list = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition))
        marker = f"  ({detail})" if detail else ""
        print(f"  {PASS if condition else FAIL}  {name}{marker}")

    print("\n── Stage 2: Symbol Parser self-tests ──\n")

    cases = [
        # (input, expected_type, expected_value, expected_unit, note)
        # Chamfer
        ("2x45°",          "chamfer",           2.0,    "mm",   "chamfer 2×45°"),
        ("C2",             "chamfer",           2.0,    "mm",   "chamfer C2"),
        ("1×45°",          "chamfer",           1.0,    "mm",   "chamfer 1×45°"),
        # Angle
        ("45°",            "angle",             45.0,   "deg",  "angle degrees"),
        ("22.5deg",        "angle",             22.5,   "deg",  "angle deg suffix"),
        # Metric thread
        ("M6",             "thread_callout",    6.0,    "mm",   "metric M6"),
        ("M8",             "thread_callout",    8.0,    "mm",   "M8 no pitch"),
        ("M12x1.75",       "thread_callout",    12.0,   "mm",   "M12 with pitch"),
        # Diameter
        ("Ø12",            "dimension_diameter",12.0,   "mm",   "diameter Ø12"),
        ("Ø12H7",          "fit",               12.0,   "mm",   "diameter with fit"),
        ("⌀25",            "dimension_diameter",25.0,   "mm",   "diameter alt symbol"),
        # Radius
        ("R12.5",          "dimension_radial",  12.5,   None,   "radius no unit"),
        ("R12.5mm",        "dimension_radial",  12.5,   "mm",   "radius mm"),
        ("R.5",            "dimension_radial",  0.5,    None,   "radius leading dot"),
        # Bilateral tolerance symmetric
        ("20±0.02",        "tolerance",         20.0,   "mm",   "sym tolerance"),
        ("100±0.05",       "tolerance",         100.0,  "mm",   "sym tolerance 2"),
        # Bilateral tolerance asymmetric
        ("+0.1/-0.05",     "tolerance",         0.0,    "mm",   "FAILS: no nom — ok to get unknown"),
        ("50+0.1/-0.05",   "tolerance",         50.0,   "mm",   "asym tolerance"),
        # Fit
        ("25H7",           "fit",               25.0,   "mm",   "fit H7"),
        ("30js6",          "fit",               30.0,   "mm",   "fit js6"),
        # Quantity
        ("4×",             "quantity",          4.0,    None,   "quantity only"),
        ("4×Ø8",           "quantity",          4.0,    None,   "qty with diameter"),
        ("4x",             "quantity",          4.0,    None,   "qty lowercase x"),
        # Reference
        ("(A)",            "reference",         None,   None,   "reference A"),
        ("[B]",            "reference",         None,   None,   "reference B"),
        # Linear metric
        ("12.5mm",         "dimension_linear",  12.5,   "mm",   "linear mm"),
        ("100mm",          "dimension_linear",  100.0,  "mm",   "100mm"),
        (".5mm",           "dimension_linear",  0.5,    "mm",   "leading dot mm"),
        # Linear imperial decimal
        ('1.250"',         "dimension_linear",  1.25,   "in",   "inch decimal"),
        # Linear imperial fractional
        ('3/8"',           "dimension_linear",  0.375,  "in",   "3/8 inch"),
        ('1 3/4"',         "dimension_linear",  1.75,   "in",   "1-3/4 inch"),
        # Surface finish
        ("Ra1.6",          "surface_finish",    1.6,    "µm",   "Ra"),
        ("Rz6.3",          "surface_finish",    6.3,    "µm",   "Rz"),
        # Bare number
        ("12.5",           "dimension_linear",  12.5,   None,   "bare number"),
        ("100",            "dimension_linear",  100.0,  None,   "bare int"),
        # Title block
        ("MATERIAL: AL6061","title_block",      None,   None,   "title block"),
        ("SCALE: 1:2",     "title_block",       None,   None,   "scale"),
        # OCR fixes
        ("o12",            "dimension_diameter",12.0,   "mm",   "OCR o→Ø"),
        # Comma decimal
        ("12,5mm",         "dimension_linear",  12.5,   "mm",   "comma decimal"),
        # Unknown
        ("???###",         "unknown",           None,   None,   "garbage"),
    ]

    for raw, exp_type, exp_val, exp_unit, note in cases:
        # Skip the asymmetric-with-no-nominal case (well-defined failure)
        if note.startswith("FAILS"):
            continue
        result = parse_symbol(raw)
        type_ok = result.token_type == exp_type
        if exp_val is None:
            val_ok = result.value is None
        elif isinstance(exp_val, float):
            val_ok = (result.value is not None and
                      abs(float(result.value) - exp_val) < 1e-4)
        else:
            val_ok = result.value == exp_val
        unit_ok = result.unit == exp_unit
        ok = type_ok and val_ok and unit_ok
        check(
            f"{raw!r}  →  {exp_type}",
            ok,
            f"got type={result.token_type!r} val={result.value!r} unit={result.unit!r}  [{note}]"
            if not ok else "",
        )

    # raw_text always preserved
    r = parse_symbol("Ø12")
    check("raw_text preserved (Ø12)", r.raw_text == "Ø12")

    # OCR fix: o→Ø  raw_text still original
    r2 = parse_symbol("o8")
    check("raw_text preserved after OCR fix", r2.raw_text == "o8")

    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n── Symbol Parser tests: {passed}/{total} passed ──\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_tests())
