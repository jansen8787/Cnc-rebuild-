"""pipeline.py — CNC AI Recognition Engine (bundled single file)
Entry point: run_pipeline(path, dpi=300, quiet=True) -> dict
"""
from __future__ import annotations
import argparse, dataclasses, json, math, os, re, sys, time, tempfile, traceback, struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError



# ── m2types.py ────────────────────────────────────────────────

"""
types.py — Module 2: Drawing Recognition Engine
================================================
All shared data types. No business logic. No external dependencies.

Every stage imports from here. No stage imports types from another stage.
This is the loose-coupling boundary (master rule 15).

Version: 1.0.0
Schema contract: Module 1 V2 (PartData — see module1/types.ts)
"""

# ---------------------------------------------------------------------------
# Stage 0 — Ingest outputs
# ---------------------------------------------------------------------------

@dataclass
class PageRaster:
    """One rasterised page from any source (PDF page or photo frame)."""
    page_number: int          # 1-based
    image_array: Any          # np.ndarray (H, W) grayscale uint8 — typed as Any
                              # to avoid numpy import at type-definition time
    width_px: int
    height_px: int
    dpi: float                # actual render DPI; 0.0 if unknown

@dataclass
class RawInput:
    """Canonical output of Stage 0 (Ingest)."""
    input_type: str           # "photo" | "pdf" — "cad" deferred
    source_path: str          # absolute path to the original file
    pages: list               # List[PageRaster]
    pdf_text_layer: list      # List[dict] — TextRun from vector PDF; empty for photos/scans
    orig_dpi: Optional[float] # DPI reported by source; None if unknown
    # Diagnostics
    exif_rotation_applied: int      # degrees (0, 90, 180, 270)
    page_count: int

# ---------------------------------------------------------------------------
# Stage 1 — Preprocessing outputs
# ---------------------------------------------------------------------------

@dataclass
class PreprocessedImage:
    """Output of Stage 1 (both PDF and photo pipelines emit this same shape)."""
    image_array: Any          # np.ndarray (H, W) uint8 — binary: ink=255, paper=0
    width_px: int
    height_px: int
    page_number: int
    pipeline: str             # "pdf" | "photo"
    # Diagnostics
    deskewed: bool
    deskew_angle_deg: float   # 0.0 if not deskewed
    deskew_variance: float    # projection variance at best angle; 0.0 if not run
    threshold_method: str     # "otsu" | "adaptive"
    denoise_applied: bool
    crop_bbox: Optional[dict] # {x,y,w,h} if border-crop was applied; None otherwise

# ---------------------------------------------------------------------------
# Stage 2 — OCR / symbol parser outputs
# ---------------------------------------------------------------------------

@dataclass
class ParsedSymbol:
    """Result of running the symbol parser on one OCR token text."""
    token_type: str           # "dimension_diameter" | "dimension_radial" |
                              # "thread_callout" | "tolerance" | "angle" |
                              # "dimension_linear" | "fit" | "chamfer" |
                              # "quantity" | "title_block" | "label" | "unknown"
    value: Optional[float]    # numeric value extracted (None for threads/labels)
    unit: Optional[str]       # "mm" | "in" | "deg" | None
    qualifier: Optional[str]  # "Ø" | "R" | "M" | "C" | None
    upper_tol: Optional[float]
    lower_tol: Optional[float]
    fit_code: Optional[str]   # "H7" | "g6" | "js6" etc.
    thread_pitch: Optional[float]
    quantity: Optional[int]   # from "4×" prefix
    angle_deg: Optional[float]
    flags: list               # List[str] — "ambiguous", "unparsed", etc.
    raw_text: str             # original OCR text, unmodified

@dataclass
class TextAnnotation:
    """One recognized text region with its parsed meaning and location."""
    id: str                   # "ann_p1_0042"
    page: int                 # 1-based
    raw_text: str
    parsed: ParsedSymbol
    bbox: dict                # {"x": int, "y": int, "w": int, "h": int} in pixels
    ocr_confidence: float     # 0.0–1.0

@dataclass
class TextMask:
    """Describes which pixel regions contain text (used by Stage 3 to blank them)."""
    page: int
    regions: list             # List[dict] — each {"x", "y", "w", "h"} in pixels

# ---------------------------------------------------------------------------
# Stage 2.5 — Title block outputs
# ---------------------------------------------------------------------------

@dataclass
class TitleBlockInfo:
    """Structured fields extracted from the drawing's title block."""
    part_name: Optional[str]
    drawing_number: Optional[str]
    revision: Optional[str]
    material: Optional[str]
    scale_raw: Optional[str]     # e.g. "1:2" — parsed by Stage 4
    units_hint: Optional[str]    # "mm" | "in" — may be None
    date: Optional[str]
    author: Optional[str]
    sheet: Optional[str]
    tolerance_general: Optional[str]  # e.g. "±0.1"
    # Raw key-value pairs not mapped to known fields
    extra: dict

# ---------------------------------------------------------------------------
# Stage 3 — Geometry detector outputs
# ---------------------------------------------------------------------------

@dataclass
class GeometryCandidate:
    """One detected geometry candidate. Carries confidence; never filtered hard."""
    id: str                   # "cand_circle_0042"
    kind: str                 # "circle" | "rect" | "slot" | "polygon" | "line" | "arc"
    geometry: dict            # kind-specific pixel coords (see per-detector docs)
    confidence: float         # 0.0–1.0 from detector
    detector: str             # "circles" | "slots" | "rectangles" | "polygons" | "lines" | "arcs"
    page: int                 # 1-based
    evidence: dict            # raw measurements: circularity, area_px, aspect_ratio, etc.

# ---------------------------------------------------------------------------
# Stage 4 — Scale transform outputs
# ---------------------------------------------------------------------------

@dataclass
class ScaleInfo:
    """Pixel-to-mm scale and coordinate origin for one page."""
    page: int
    px_per_mm: float          # pixels per mm; 0.0 if unknown
    anchor_method: str        # "dimension_text" | "sheet_size" | "pdf_metadata"
                              # | "operator_override" | "unknown"
    anchor_confidence: float  # 0.0–1.0
    origin_px: dict           # {"x": float, "y": float} — pixel coord of part zero

@dataclass
class ScaledCandidate:
    """A GeometryCandidate whose geometry has been converted to mm coordinates."""
    candidate: GeometryCandidate
    geometry_mm: dict         # same keys as candidate.geometry but in mm
    scale_info: ScaleInfo

# ---------------------------------------------------------------------------
# Stage 5 — Assembly outputs (interim; final output is a PartData dict)
# ---------------------------------------------------------------------------

@dataclass
class LinkedPair:
    """One text annotation paired with one geometry candidate."""
    annotation: TextAnnotation
    candidate: ScaledCandidate
    link_confidence: float
    link_type: str            # "diameter" | "radius" | "thread" | "group" | "fit" | "linear"
    ambiguous: bool

@dataclass
class SuppressedCandidate:
    """A candidate that was de-prioritised in conflict resolution."""
    candidate: ScaledCandidate
    reason: str               # why it was suppressed

# ---------------------------------------------------------------------------
# Stage 6 — Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class RecognitionReport:
    """
    The truthfulness layer. Always produced. Honestly reports all signals.
    overallConfidence = MIN across critical signals, NOT average (§7 architecture).
    """
    pipeline: str                       # "pdf" | "photo"
    stage_timings_ms: dict              # stage_name → float

    # Per-stage diagnostics
    ingest: dict
    preproc: dict
    ocr: dict
    title_block: dict
    geometry: dict
    cross_link: dict

    # Summary
    overall_confidence: float           # 0..1, min of critical signals
    weak_signals: list                  # List[str] — human-readable warnings
    recommend_benchmark: bool           # flag if this drawing should be in benchmark set

    # Suppressed candidates (conflict resolution output)
    suppressed: list                    # List[SuppressedCandidate]

# ---------------------------------------------------------------------------
# Final pipeline output
# ---------------------------------------------------------------------------

@dataclass
class Module2Output:
    """
    The single JSON-serialisable output of Module 2.
    partData matches the Module 1 V2 PartData schema (TypeScript types.ts).
    diagnostics is always present, even on failure.
    """
    part_data: dict           # PartData-shaped dict — pass to normalizePart() in Module 1
    diagnostics: dict         # RecognitionReport as dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bbox_center(bbox: dict) -> tuple:
    """Return (cx, cy) float pair from any bbox dict."""
    x = float(bbox.get("x", bbox.get("left", 0)))
    y = float(bbox.get("y", bbox.get("top", 0)))
    w = float(bbox.get("w", bbox.get("width", 0)))
    h = float(bbox.get("h", bbox.get("height", 0)))
    return (x + w / 2.0, y + h / 2.0)

def bbox_area(bbox: dict) -> float:
    w = float(bbox.get("w", bbox.get("width", 0)))
    h = float(bbox.get("h", bbox.get("height", 0)))
    return w * h

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

def norm_confidence(raw: float) -> float:
    """Clamp and round confidence to [0.0, 1.0], 4 decimal places."""
    return round(clamp(float(raw), 0.0, 1.0), 4)


# ── symbol_parser.py ──────────────────────────────────────────

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

from pathlib import Path


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


# ── ingest.py ─────────────────────────────────────────────────

"""
stage0_ingest/ingest.py — Module 2: Stage 0
============================================
File ingestion: detect input type (magic bytes, not extension), apply EXIF
rotation, extract raster pages from PDF or image.

Architecture rules honoured:
- EXIF rotation applied unconditionally (forensic report §A1).
- File type detected from magic bytes, not extension (§A1).
- PDF text layer extracted separately and preserved (Stage 2 short-circuits on it).
- Every page in a PDF = one PageRaster (multi-page support, one part at a time).
- No confidence invented — DPI is reported accurately or as 0.0 / None.

Public API:
    ingest(path: str | Path, *, dpi: int = 300) -> RawInput

Exit codes (when used as main):
    0  success
    1  file not found or unreadable
    2  unsupported format
    3  argument error

Empirical DPI guidance (from old project forensics):
    200 DPI — minimum acceptable for Tesseract on large text
    300 DPI — recommended for most technical drawings (default)
    400 DPI — use for drawings with very small dimension text (< 6pt)
    600 DPI — archival; slow and memory-intensive
"""

from pathlib import Path

# Types defined in the shared types module one level up

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_DPI = 72
_MAX_DPI = 600
_DEFAULT_DPI = 300

# Magic byte signatures for reliable file type detection
_MAGIC_PDF   = b"%PDF"
_MAGIC_PNG   = b"\x89PNG"
_MAGIC_JPEG  = b"\xff\xd8\xff"
_MAGIC_TIFF_LE = b"II\x2a\x00"
_MAGIC_TIFF_BE = b"MM\x00\x2a"
_MAGIC_GIF   = b"GIF8"

SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"}
)
SUPPORTED_PDF_EXTENSIONS = frozenset({".pdf"})
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS

# ---------------------------------------------------------------------------
# Magic-byte file type detection
# ---------------------------------------------------------------------------

def _read_magic(path: Path, n: int = 8) -> bytes:
    """Read the first n bytes of a file for magic-byte sniffing."""
    with open(path, "rb") as f:
        return f.read(n)

def detect_file_type(path: Path) -> str:
    """
    Detect file type using magic bytes (first 8 bytes).
    Falls back to extension if magic bytes are inconclusive.

    Returns:
        "pdf"   for PDF documents
        "image" for any raster image format

    Raises:
        FileNotFoundError: file does not exist
        ValueError:        format not supported
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        magic = _read_magic(path)
    except OSError as exc:
        raise OSError(f"Cannot read file {path}: {exc}") from exc

    if magic[:4] == _MAGIC_PDF:
        return "pdf"
    if magic[:4] == _MAGIC_PNG:
        return "image"
    if magic[:3] == _MAGIC_JPEG:
        return "image"
    if magic[:4] in (_MAGIC_TIFF_LE, _MAGIC_TIFF_BE):
        return "image"
    if magic[:4] == _MAGIC_GIF:
        return "image"

    # Fall back to extension — documented, not silent
    ext = path.suffix.lower()
    if ext in SUPPORTED_PDF_EXTENSIONS:
        return "pdf"
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"

    raise ValueError(
        f"Unsupported file format: {path}. "
        f"Magic bytes: {magic[:8]!r}. "
        f"Supported: PDF, PNG, JPEG, TIFF, GIF."
    )

# ---------------------------------------------------------------------------
# EXIF orientation handling
# ---------------------------------------------------------------------------

_EXIF_ORIENTATION_TAG = 0x0112  # Tag 274

def _read_exif_orientation(img: Image.Image) -> int:
    """
    Read EXIF orientation tag.

    Returns degrees to rotate clockwise to make the image upright:
        0 → no rotation needed
        90 → rotate 90° CW (phone portrait held sideways)
        180 → rotate 180°
        270 → rotate 270° CW (= 90° CCW)

    Returns 0 if EXIF is absent or orientation tag is missing.
    """
    try:
        exif = img._getexif()  # type: ignore[attr-defined]
        if not exif:
            return 0
        orientation = exif.get(_EXIF_ORIENTATION_TAG, 1)
    except (AttributeError, Exception):
        return 0

    # EXIF orientation values 1–8:
    # 1 = normal, 3 = 180°, 6 = 90° CW, 8 = 90° CCW
    mapping = {
        1: 0,    # normal
        2: 0,    # horizontal flip — treat as normal (rare in drawings)
        3: 180,
        4: 180,  # vertical flip after 180° — treat as 180°
        5: 270,
        6: 90,   # rotated 90° CW (most common phone landscape)
        7: 90,
        8: 270,
    }
    return mapping.get(int(orientation), 0)

def _apply_exif_rotation(img: Image.Image) -> tuple[Image.Image, int]:
    """
    Apply EXIF orientation unconditionally.

    Returns:
        (corrected_image, degrees_rotated)
    Degrees is 0 if no rotation was needed.
    """
    degrees = _read_exif_orientation(img)
    if degrees == 0:
        return img, 0
    # PIL rotate: positive = CCW; we want CW correction
    # CW 90° = CCW 270°, etc.
    pil_angle = {90: 270, 180: 180, 270: 90}[degrees]
    corrected = img.rotate(pil_angle, expand=True)
    return corrected, degrees

# ---------------------------------------------------------------------------
# Image loading (photos / raster images)
# ---------------------------------------------------------------------------

def _pil_to_gray_array(img: Image.Image) -> np.ndarray:
    """
    Convert any PIL Image to a single-channel uint8 numpy array.

    Handles: RGBA (composited onto white), P (palette), CMYK, L, RGB.
    Transparent areas (CAD PNG exports) become white (paper colour).
    """
    if img.mode == "P":
        img = img.convert("RGBA")

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg

    if img.mode != "L":
        img = img.convert("L")

    return np.array(img, dtype=np.uint8)

def _load_image(path: Path) -> RawInput:
    """Load a single raster image file as a one-page RawInput."""
    try:
        img = Image.open(path)
        img.verify()            # detect truncated / corrupt files early
    except UnidentifiedImageError:
        raise UnidentifiedImageError(f"Cannot identify image file: {path}")
    except Exception as exc:
        raise OSError(f"Failed to open image {path}: {exc}") from exc

    # Re-open after verify (PIL requirement)
    img = Image.open(path)

    # Determine original DPI from metadata
    orig_dpi: Optional[float] = None
    try:
        dpi_info = img.info.get("dpi") or img.info.get("jfif_density")
        if dpi_info:
            if isinstance(dpi_info, (tuple, list)) and len(dpi_info) >= 1:
                orig_dpi = float(dpi_info[0])
            else:
                orig_dpi = float(dpi_info)
    except (TypeError, ValueError):
        orig_dpi = None

    # Apply EXIF rotation unconditionally (architecture §A1)
    img, exif_rotation = _apply_exif_rotation(img)

    # Convert to grayscale array
    gray = _pil_to_gray_array(img)
    h, w = gray.shape

    page = PageRaster(
        page_number=1,
        image_array=gray,
        width_px=w,
        height_px=h,
        dpi=float(orig_dpi) if orig_dpi else 0.0,
    )

    return RawInput(
        input_type="photo",
        source_path=str(path.resolve()),
        pages=[page],
        pdf_text_layer=[],
        orig_dpi=orig_dpi,
        exif_rotation_applied=exif_rotation,
        page_count=1,
    )

# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------

def _extract_pdf_text_layer(path: Path) -> list:
    """
    Extract the text layer from a vector PDF.
    Returns a list of TextRun dicts: {text, x, y, w, h, page}.
    Returns [] if the PDF has no text layer (scanned) or pdfminer is unavailable.
    """
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextBox, LTTextLine, LTAnno, LTChar
    except ImportError:
        return []

    runs: list = []
    try:
        for page_num, page_layout in enumerate(extract_pages(str(path)), start=1):
            page_h = float(page_layout.height)
            for element in page_layout:
                if not isinstance(element, LTTextBox):
                    continue
                x0, y0, x1, y1 = (
                    element.x0, element.y0, element.x1, element.y1
                )
                text = element.get_text().strip()
                if not text:
                    continue
                # PDF coordinates: y=0 is bottom; flip to image convention
                runs.append({
                    "text": text,
                    "x": float(x0),
                    "y": float(page_h - y1),
                    "w": float(x1 - x0),
                    "h": float(y1 - y0),
                    "page": page_num,
                })
    except Exception:
        # Text extraction failure is non-fatal — Stage 2 will run OCR
        return []

    return runs

def _load_pdf(path: Path, dpi: int = _DEFAULT_DPI) -> RawInput:
    """Rasterise a PDF into one PageRaster per page."""
    try:
        from pdf2image import convert_from_path
        from pdf2image.exceptions import (
            PDFInfoNotInstalledError,
            PDFPageCountError,
            PDFSyntaxError,
        )
    except ImportError:
        raise ImportError(
            "pdf2image is required for PDF loading. "
            "Install with: pip install pdf2image"
        ) from None

    if dpi < _MIN_DPI or dpi > _MAX_DPI:
        raise ValueError(f"DPI must be in [{_MIN_DPI}, {_MAX_DPI}], got {dpi}")

    try:
        pil_pages = convert_from_path(str(path), dpi=dpi)
    except PDFInfoNotInstalledError:
        raise RuntimeError(
            "poppler is not installed. "
            "Install with: apt-get install poppler-utils  (or brew install poppler)"
        ) from None
    except PDFPageCountError as exc:
        raise ValueError(f"Cannot determine PDF page count: {path}: {exc}") from exc
    except PDFSyntaxError as exc:
        raise ValueError(f"Malformed or encrypted PDF: {path}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"pdf2image failed on {path}: {exc}") from exc

    pages: list = []
    for i, pil_img in enumerate(pil_pages, start=1):
        gray = _pil_to_gray_array(pil_img)
        h, w = gray.shape
        pages.append(PageRaster(
            page_number=i,
            image_array=gray,
            width_px=w,
            height_px=h,
            dpi=float(dpi),
        ))

    # Attempt text layer extraction (non-fatal if unavailable)
    text_layer = _extract_pdf_text_layer(path)

    return RawInput(
        input_type="pdf",
        source_path=str(path.resolve()),
        pages=pages,
        pdf_text_layer=text_layer,
        orig_dpi=float(dpi),
        exif_rotation_applied=0,   # PDFs have no EXIF
        page_count=len(pages),
    )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest(
    path: Union[str, Path],
    *,
    dpi: int = _DEFAULT_DPI,
) -> RawInput:
    """
    Ingest a drawing file and return a RawInput ready for Stage 1.

    Detects file type via magic bytes (not extension).
    Applies EXIF rotation for images unconditionally.
    Extracts PDF text layer when available (vector PDFs).

    Args:
        path:  Path to the source file (PDF, PNG, JPG, TIF).
        dpi:   Render resolution for PDFs (default 300). Images use native resolution.

    Returns:
        RawInput dataclass with pages (rasterised), text layer, and metadata.

    Raises:
        FileNotFoundError:  File does not exist.
        ValueError:         Unsupported format or DPI out of range.
        OSError:            File cannot be read.
        RuntimeError:       pdf2image / poppler failure.
    """
    p = Path(path).resolve()

    if not p.exists():
        raise FileNotFoundError(f"Source file not found: {p}")

    file_type = detect_file_type(p)

    if file_type == "pdf":
        return _load_pdf(p, dpi=dpi)
    else:
        return _load_image(p)

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


# ── pdf_pipeline.py ───────────────────────────────────────────

"""
stage1_preprocess/pdf_pipeline.py — Module 2: Stage 1a
=======================================================
PDF preprocessing pipeline. Minimal — PDFs are already aligned.

Pipeline:
    1. Grayscale (pass-through — Stage 0 already delivers grayscale)
    2. Autolevels (1st/99th percentile stretch — handles faded scans)
    3. Gaussian blur (kernel=3) — light noise reduction
    4. Otsu threshold → binary (ink=255, paper=0)
    5. Morphological opening (kernel=2, iter=1) — remove isolated noise pixels

NOT applied:
    - Deskew (PDFs are rendered aligned by poppler)
    - Adaptive threshold (uneven lighting is a photo problem)
    - Border crop (PDF canvas is the drawing boundary)

Empirical constants from forensic analysis of old project:
    contrast_factor  = 1.6   (tuned against real scan artefacts)
    sharpness_factor = 1.4   (UnsharpMask radius=1, prevents haloing on thin lines)
    morph_kernel     = 2     (conservative — preserves thin dimension lines)
    morph_iterations = 1

Public API:
    preprocess_pdf(page: PageRaster) -> PreprocessedImage
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Constants (from forensic analysis — do not change without benchmark evidence)
# ---------------------------------------------------------------------------

_CONTRAST_FACTOR:  float = 1.6
_SHARPNESS_FACTOR: float = 1.4
_BLUR_KERNEL: int = 3
_MORPH_KERNEL_SIZE: int = 2
_MORPH_ITERATIONS: int = 1

# ---------------------------------------------------------------------------
# Internal stages
# ---------------------------------------------------------------------------

def _autolevels(gray: np.ndarray) -> np.ndarray:
    """
    Histogram stretch at 1st/99th percentile.
    Recovers faded scans without over-whitening solid lines.
    Guard: if the range is trivially small (blank page), return as-is.
    """
    p_low  = int(np.percentile(gray, 1))
    p_high = int(np.percentile(gray, 99))
    if p_high <= p_low:
        return gray                         # blank or uniform — skip
    stretched = np.clip(gray, p_low, p_high)
    stretched = ((stretched - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    return stretched

def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Autolevels + Gaussian blur for light noise damping."""
    g = _autolevels(gray)
    if _BLUR_KERNEL > 0:
        g = cv2.GaussianBlur(g, (_BLUR_KERNEL, _BLUR_KERNEL), sigmaX=0)
    return g

def _binarize_otsu(gray: np.ndarray) -> np.ndarray:
    """
    Otsu threshold: dark ink → 255 (foreground), light paper → 0.
    THRESH_BINARY_INV: ink is foreground.
    """
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return binary

def _morphological_open(binary: np.ndarray) -> np.ndarray:
    """
    Morphological opening: removes isolated noise pixels.
    Conservative kernel (2×2) to not destroy thin dimension lines.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (_MORPH_KERNEL_SIZE, _MORPH_KERNEL_SIZE),
    )
    return cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel, iterations=_MORPH_ITERATIONS
    )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_pdf(page: PageRaster) -> PreprocessedImage:
    """
    Preprocess one PDF page raster for geometry and OCR detection.

    Args:
        page: PageRaster from Stage 0 (grayscale uint8 numpy array).

    Returns:
        PreprocessedImage with binary array (ink=255, paper=0) and diagnostics.
    """
    gray = page.image_array.astype(np.uint8)

    # Step 1: Autolevels + light noise damping
    enhanced = _enhance_contrast(gray)

    # Step 2: Otsu binarization
    binary = _binarize_otsu(enhanced)

    # Step 3: Morphological opening
    cleaned = _morphological_open(binary)

    h, w = cleaned.shape

    return PreprocessedImage(
        image_array=cleaned,
        width_px=w,
        height_px=h,
        page_number=page.page_number,
        pipeline="pdf",
        deskewed=False,
        deskew_angle_deg=0.0,
        deskew_variance=0.0,
        threshold_method="otsu",
        denoise_applied=False,        # Gaussian blur is light; not "denoise"
        crop_bbox=None,
    )

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


# ── photo_pipeline.py ─────────────────────────────────────────

"""
stage1_preprocess/photo_pipeline.py — Module 2: Stage 1b
=========================================================
Photo preprocessing pipeline. More aggressive than PDF pipeline because
phone photos have uneven lighting, perspective, rotation, and noise.

Pipeline:
    1. Grayscale (pass-through — Stage 0 already grayscale + EXIF-corrected)
    2. Border crop — find largest rectangular contour (drawing frame)
    3. Autolevels (1st/99th percentile)
    4. Adaptive threshold (not Otsu — uneven lighting)
    5. Light median denoise (kernel=3) — phone camera noise
    6. Conservative deskew — projection-profile variance, max ±15°, 0.4° threshold

Architecture rules from forensic report §6.3:
    - Projection-profile variance method (not Hough lines — matches old project's
      empirically tuned approach)
    - 15° maximum rotation cap (larger = likely intentional orientation or
      severe distortion not fixable by simple rotation)
    - 0.4° minimum threshold (skip rotation when near-straight)
    - Variance gate: if best_variance < 1.0, rotation signal too weak → skip
    - Record deskewed=True/False + angle in diagnostics unconditionally

Empirical constants:
    max_angle        = 15.0° (from old project preprocessor.py, tuned)
    angle_threshold  = 0.4°  (from old project preprocessor.py, tuned)
    n_projections    = 181   (grid of candidate angles to test)
    variance_floor   = 1.0   (blank-page guard)
    morph_kernel     = 3
    morph_iterations = 1

Public API:
    preprocess_photo(page: PageRaster) -> PreprocessedImage
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Constants (empirical — from old project + forensic analysis)
# ---------------------------------------------------------------------------

_MAX_DESKEW_ANGLE: float = 15.0       # degrees — hard cap
_DESKEW_THRESHOLD: float = 0.4        # degrees — minimum to bother rotating
_N_PROJECTIONS:    int   = 181        # candidate angles to test
_VARIANCE_FLOOR:   float = 1.0        # blank/uniform image guard
_MEDIAN_KERNEL:    int   = 3          # denoise kernel size (must be odd)
_ADAPTIVE_BLOCK:   int   = 15         # adaptive threshold block size (odd)
_ADAPTIVE_C:       int   = 8          # adaptive threshold constant subtracted

# Border-crop: minimum fraction of image area a contour must have to be
# considered the drawing frame (prevents false positives from annotations)
_CROP_MIN_AREA_FRAC: float = 0.30
_CROP_APPROX_EPS:    float = 0.02     # approxPolyDP epsilon factor

# ---------------------------------------------------------------------------
# Step 1 — Border crop
# ---------------------------------------------------------------------------

def _detect_drawing_border(gray: np.ndarray) -> dict | None:
    """
    Detect the drawing's outer frame (largest rectangular contour).

    Looks for the rectangle that most plausibly frames the entire drawing
    by finding the largest contour with ≈4 vertices that covers at least
    _CROP_MIN_AREA_FRAC of the image.

    Returns:
        {x, y, w, h} bounding box in pixels, or None if no clear border found.
    """
    h, w = gray.shape
    min_area = _CROP_MIN_AREA_FRAC * h * w

    # Threshold to find large regions
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: dict | None = None
    best_area = 0.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        peri = float(cv2.arcLength(contour, closed=True))
        eps = _CROP_APPROX_EPS * peri
        approx = cv2.approxPolyDP(contour, eps, closed=True)
        if len(approx) != 4:
            continue
        if area > best_area:
            best_area = area
            bx, by, bw, bh = cv2.boundingRect(approx)
            best = {"x": int(bx), "y": int(by), "w": int(bw), "h": int(bh)}

    return best

def _apply_crop(gray: np.ndarray, bbox: dict) -> np.ndarray:
    """Crop to the detected drawing border. Guard against out-of-bounds."""
    h, w = gray.shape
    x = max(0, bbox["x"])
    y = max(0, bbox["y"])
    x2 = min(w, x + bbox["w"])
    y2 = min(h, y + bbox["h"])
    if x2 - x < 10 or y2 - y < 10:
        return gray           # crop too small — return full image
    return gray[y:y2, x:x2]

# ---------------------------------------------------------------------------
# Step 2 — Autolevels + adaptive threshold + denoise
# ---------------------------------------------------------------------------

def _autolevels(gray: np.ndarray) -> np.ndarray:
    """Histogram stretch at 1st/99th percentile."""
    p_low  = int(np.percentile(gray, 1))
    p_high = int(np.percentile(gray, 99))
    if p_high <= p_low:
        return gray
    stretched = np.clip(gray, p_low, p_high)
    return ((stretched - p_low) / (p_high - p_low) * 255).astype(np.uint8)

def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian threshold for uneven illumination (typical photo case).
    Inverted: ink → 255, paper → 0.
    Block size and C constant empirically tuned.
    """
    block = _ADAPTIVE_BLOCK
    if block % 2 == 0:
        block += 1          # must be odd
    # Ensure min block size relative to image
    if min(gray.shape) < block:
        # Fall back to Otsu for tiny images
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        _ADAPTIVE_C,
    )
    return binary

def _median_denoise(binary: np.ndarray) -> np.ndarray:
    """Light median filter to remove phone camera impulse noise."""
    k = _MEDIAN_KERNEL
    if k % 2 == 0:
        k += 1
    return cv2.medianBlur(binary, k)

# ---------------------------------------------------------------------------
# Step 3 — Projection-profile deskew
# ---------------------------------------------------------------------------

def detect_rotation_angle(
    gray: np.ndarray,
    *,
    max_angle: float = _MAX_DESKEW_ANGLE,
    n_projections: int = _N_PROJECTIONS,
) -> tuple[float, float]:
    """
    Estimate skew angle via horizontal projection profile variance maximisation.

    Method (from old project preprocessor.py, empirically tuned):
        Technical drawings contain many horizontal dimension lines and text
        baselines. When correctly oriented, summing pixel intensities row by
        row produces a striped profile with high variance. Rotating by the
        wrong angle smears these stripes and reduces variance.

        We test n_projections candidate angles in [-max_angle, +max_angle],
        compute row-sum variance for each rotation, and pick the angle that
        maximises variance.

    Returns:
        (angle_deg, best_variance)
        angle_deg = 0.0 if variance signal too weak (blank / uniform image).
    """
    max_angle = min(abs(max_angle), _MAX_DESKEW_ANGLE)

    # Downscale to 600 px wide for speed (projection analysis doesn't need detail)
    scale = min(1.0, 600.0 / max(gray.shape[1], 1))
    if scale < 1.0:
        small = cv2.resize(
            gray,
            (int(gray.shape[1] * scale), int(gray.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = gray.copy()

    # Invert: ink → high values, paper → low
    inverted = (255.0 - small.astype(np.float32))

    angles = np.linspace(-max_angle, max_angle, n_projections)

    best_angle    = 0.0
    best_variance = -1.0

    # Convert to PIL for rotation (scipy available in most envs but PIL is lighter)
    from PIL import Image as _PILImage
    inv_img = _PILImage.fromarray(inverted.astype(np.float32))

    for angle in angles:
        rotated = inv_img.rotate(float(angle), resample=_PILImage.BICUBIC,
                                 expand=False, fillcolor=0)
        rot_arr = np.array(rotated, dtype=np.float32)
        row_sums = rot_arr.sum(axis=1)
        var = float(np.var(row_sums))
        if var > best_variance:
            best_variance = var
            best_angle = float(angle)

    # If variance is negligible — blank or uniform image — return 0
    if best_variance < _VARIANCE_FLOOR:
        return 0.0, best_variance

    return best_angle, best_variance

def _deskew(gray: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Correct skew in a photo of a technical drawing.

    Rules (forensic report Q5 / architecture §A1):
        - Test range: ±15° maximum
        - Skip if |angle| < 0.4° (below threshold)
        - Skip if variance < 1.0 (blank / uniform image)
        - Fill exposed areas with white (255)
        - expand=False: downstream coordinate systems stay valid

    Returns:
        (corrected_array, angle_applied, variance)
        angle_applied = 0.0 if no rotation was performed.
    """
    angle, variance = detect_rotation_angle(gray)

    if abs(angle) < _DESKEW_THRESHOLD:
        return gray, 0.0, variance

    # Rotate by negative angle to counteract the detected tilt
    h, w = gray.shape
    centre = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(centre, -angle, scale=1.0)
    corrected = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,           # white fill
    )
    return corrected, angle, variance

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_photo(page: PageRaster) -> PreprocessedImage:
    """
    Preprocess one photo raster for geometry and OCR detection.

    Args:
        page: PageRaster from Stage 0 (grayscale uint8, EXIF already applied).

    Returns:
        PreprocessedImage with binary array (ink=255, paper=0) and diagnostics.
    """
    gray = page.image_array.astype(np.uint8)
    crop_bbox: dict | None = None

    # Step 1: Border crop (best-effort — not fatal if none found)
    detected_border = _detect_drawing_border(gray)
    if detected_border is not None:
        gray = _apply_crop(gray, detected_border)
        crop_bbox = detected_border

    # Step 2: Autolevels
    gray = _autolevels(gray)

    # Step 3: Deskew (projection variance, conservative)
    gray, deskew_angle, deskew_variance = _deskew(gray)
    deskewed = abs(deskew_angle) >= _DESKEW_THRESHOLD

    # Step 4: Adaptive threshold
    binary = _adaptive_threshold(gray)

    # Step 5: Median denoise
    binary = _median_denoise(binary)

    h, w = binary.shape

    return PreprocessedImage(
        image_array=binary,
        width_px=w,
        height_px=h,
        page_number=page.page_number,
        pipeline="photo",
        deskewed=deskewed,
        deskew_angle_deg=float(deskew_angle),
        deskew_variance=float(deskew_variance),
        threshold_method="adaptive",
        denoise_applied=True,
        crop_bbox=crop_bbox,
    )

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


# ── ocr_engine.py ─────────────────────────────────────────────

"""
stage2_ocr/ocr_engine.py — Module 2: Stage 2
=============================================
Tesseract OCR wrapper. Produces structured TextAnnotation list from a
preprocessed binary image.

Engine choice rationale (forensic report §4.4 / §6.1):
    Tesseract first — the old project shipped a complete pipeline using it,
    tuned against real drawings. PaddleOCR remains the documented fallback
    if benchmark cases 4-5 (phone photos) fall below 80% recall.

Design decisions (from old extractor.py forensics, ADAPT bucket):
    - Confidence normalised from Tesseract 0-100 int → 0.0-1.0 float
    - Tesseract structural rows (conf == -1) silently dropped
    - Empty / whitespace-only text dropped
    - Tokens sorted deterministically: (page, y, x)
    - Token IDs: "ann_p{page}_{seq:04d}"
    - min_confidence defaults to 0.0 (no upstream filtering — flags handle this)
    - PSM 11 (sparse text) better than PSM 6 for scattered dimension labels

Public API:
    run_ocr(image_array, page, *, min_confidence, psm) -> List[TextAnnotation]
    run_ocr_on_regions(image_array, page, regions, *, ...) -> List[TextAnnotation]
"""

from pathlib import Path



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PSM 11 = "Sparse text. Find as much text as possible in no particular order."
# Better for dimension labels scattered around a drawing than PSM 6 (single block).
_DEFAULT_PSM: int = 11

# Languages to try (in order). English covers Arabic numerals and Latin symbols.
_TESSERACT_LANG: str = "eng"

_STRUCT_SENTINEL: int = -1    # Tesseract row conf == -1 → structural (skip)

# ---------------------------------------------------------------------------
# Confidence normalisation
# ---------------------------------------------------------------------------

def _norm_conf(raw: int) -> float:
    """Tesseract 0-100 int → 0.0-1.0 float, clamped."""
    clamped = max(0, min(100, int(raw)))
    return round(clamped / 100.0, 4)

# ---------------------------------------------------------------------------
# Token ID scheme
# ---------------------------------------------------------------------------

def _make_id(page: int, seq: int) -> str:
    return f"ann_p{page}_{seq:04d}"

# ---------------------------------------------------------------------------
# Tesseract availability check
# ---------------------------------------------------------------------------

def _check_tesseract() -> bool:
    """Return True if pytesseract + tesseract binary are available."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

TESSERACT_AVAILABLE: bool = _check_tesseract()

# ---------------------------------------------------------------------------
# Core OCR call
# ---------------------------------------------------------------------------

def run_ocr(
    image_array: np.ndarray,
    page: int,
    *,
    min_confidence: float = 0.0,
    psm: int = _DEFAULT_PSM,
    lang: str = _TESSERACT_LANG,
) -> list:
    """
    Run Tesseract OCR on a preprocessed binary image (ink=255, paper=0).

    Args:
        image_array:    uint8 numpy array, shape (H, W). Binary: ink=255, paper=0.
        page:           1-based page number for token ID generation.
        min_confidence: Drop tokens below this confidence (default 0.0 = keep all).
        psm:            Tesseract page segmentation mode (default 11 = sparse text).
        lang:           Tesseract language string (default "eng").

    Returns:
        List of TextAnnotation, sorted by (y, x). Empty if no text found or
        Tesseract unavailable.
    """
    if not TESSERACT_AVAILABLE:
        return []

    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError(f"min_confidence must be [0.0, 1.0], got {min_confidence}")

    import pytesseract
    from pytesseract import Output
    from PIL import Image

    # Tesseract expects white background, black text — invert our binary
    # (our binary: ink=255/white foreground; Tesseract: ink=0/black on white)
    inverted = 255 - image_array

    pil_img = Image.fromarray(inverted.astype(np.uint8), mode="L")

    config = f"--psm {psm} -l {lang}"

    try:
        data = pytesseract.image_to_data(
            pil_img,
            config=config,
            output_type=Output.DICT,
        )
    except Exception:
        return []

    n_rows = len(data["text"])
    raw_rows: list = []

    for i in range(n_rows):
        raw_conf: int = int(data["conf"][i])
        raw_text: str = str(data["text"][i]).strip()

        if raw_conf == _STRUCT_SENTINEL:
            continue
        if not raw_text:
            continue

        norm = _norm_conf(raw_conf)
        if norm < min_confidence:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        raw_rows.append((y, x, raw_text, norm, x, y, w, h))

    # Sort deterministically: (y, x)
    raw_rows.sort(key=lambda r: (r[0], r[1]))

    annotations: list = []
    for seq, (_, _, text, conf, x, y, w, h) in enumerate(raw_rows):
        parsed = parse_symbol(text)
        ann = TextAnnotation(
            id=_make_id(page, seq),
            page=page,
            raw_text=text,
            parsed=parsed,
            bbox={"x": x, "y": y, "w": w, "h": h},
            ocr_confidence=conf,
        )
        annotations.append(ann)

    return annotations

def run_ocr_on_regions(
    image_array: np.ndarray,
    page: int,
    regions: list,
    *,
    min_confidence: float = 0.0,
    psm: int = 7,   # PSM 7 = single line, better for small cropped regions
    lang: str = _TESSERACT_LANG,
) -> list:
    """
    Run OCR on specific bounding-box regions of an image.

    Useful when text regions are pre-detected (Stage 2.5 title block, or when
    the full-image OCR misses annotations at the drawing periphery).

    Args:
        image_array: Full-page binary uint8 array.
        page:        1-based page number.
        regions:     List of {"x", "y", "w", "h"} dicts (pixel coords).
        min_confidence: Confidence threshold.
        psm:         Tesseract PSM for region crops (default 7 = single line).
        lang:        Tesseract language.

    Returns:
        Merged, deduplicated, sorted TextAnnotation list.
    """
    all_annotations: list = []
    id_offset = 0

    for region in regions:
        x = max(0, int(region.get("x", 0)))
        y = max(0, int(region.get("y", 0)))
        w = max(1, int(region.get("w", 1)))
        h = max(1, int(region.get("h", 1)))

        # Guard bounds
        img_h, img_w = image_array.shape[:2]
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        if x2 - x < 2 or y2 - y < 2:
            continue

        crop = image_array[y:y2, x:x2]
        crop_anns = run_ocr(
            crop, page,
            min_confidence=min_confidence,
            psm=psm,
            lang=lang,
        )

        # Translate crop-local bbox back to full-image coords
        for ann in crop_anns:
            ann.id = _make_id(page, id_offset)
            id_offset += 1
            ann.bbox["x"] += x
            ann.bbox["y"] += y
            all_annotations.append(ann)

    # Sort by (page, y, x)
    all_annotations.sort(key=lambda a: (a.page, a.bbox["y"], a.bbox["x"]))
    return all_annotations

# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


# ── text_regions.py ───────────────────────────────────────────

"""
stage2_ocr/text_regions.py — Module 2: Stage 2 orchestrator
=============================================================
Detects text regions, generates the text mask (for Stage 3), runs OCR,
and returns the full TextAnnotation list for one page.

Key architectural contract (lesson #2 in MODULE_2_ARCHITECTURE.md §2):
    "Text removed before geometry sees the image."

This module does two things:
    1. run_stage2(preprocessed, pdf_text_layer) -> (annotations, text_mask)
    2. build_text_mask(annotations, width, height)  -> TextMask

The text_mask is passed to Stage 3 so geometry detection blanks these regions
before contour finding. Geometry won't mistake "Ø12" as a circle contour.

Text region detection method:
    - For vector PDFs with a text layer: use the layer directly (no OCR needed).
    - For raster (scanned PDF / photo): run full-image Tesseract, use the
      returned bboxes.
    - For both: additionally pad each bbox by _TEXT_PAD pixels to ensure
      even the ink of the annotation itself is masked.

Public API:
    run_stage2(preprocessed, pdf_text_layer, page_num) -> tuple
    build_text_mask(annotations, width, height, pad) -> TextMask
"""

from pathlib import Path



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEXT_PAD: int = 6     # pixels to expand each text bbox in the mask
_MIN_TEXT_REGION_AREA: int = 25   # px² — ignore sub-pixel noise

# ---------------------------------------------------------------------------
# Vector PDF text layer adapter
# ---------------------------------------------------------------------------

def _annotations_from_pdf_layer(
    text_layer: list,
    page: int,
    dpi: float,
) -> list:
    """
    Convert raw PDF text-layer TextRun dicts to TextAnnotation.

    The text layer contains PDF-coordinate text runs (pt units).
    DPI is used to convert to pixel coords: px = pt_coord * (dpi / 72.0).

    Args:
        text_layer: List of {"text", "x", "y", "w", "h", "page"} dicts in pt.
        page:       1-based page number to filter for.
        dpi:        Render DPI used when rasterising the PDF.

    Returns:
        List of TextAnnotation in pixel coordinates.
    """
    scale = dpi / 72.0 if dpi > 0 else 1.0
    annotations: list = []

    for idx, run in enumerate(text_layer):
        if run.get("page", 1) != page:
            continue
        text = str(run.get("text", "")).strip()
        if not text:
            continue
        x = int(float(run.get("x", 0)) * scale)
        y = int(float(run.get("y", 0)) * scale)
        w = max(4, int(float(run.get("w", 10)) * scale))
        h = max(4, int(float(run.get("h", 8)) * scale))

        parsed = parse_symbol(text)
        ann = TextAnnotation(
            id=f"ann_pdf_p{page}_{idx:04d}",
            page=page,
            raw_text=text,
            parsed=parsed,
            bbox={"x": x, "y": y, "w": w, "h": h},
            ocr_confidence=1.0,   # vector text is authoritative
        )
        annotations.append(ann)

    return annotations

# ---------------------------------------------------------------------------
# Text mask builder
# ---------------------------------------------------------------------------

def build_text_mask(
    annotations: list,
    width: int,
    height: int,
    pad: int = _TEXT_PAD,
) -> TextMask:
    """
    Build a TextMask from a list of TextAnnotation.

    The mask regions are expanded by `pad` pixels on all sides.

    Args:
        annotations: List of TextAnnotation on one page.
        width:       Image width in pixels.
        height:      Image height in pixels.
        pad:         Pixels to expand each region.

    Returns:
        TextMask with page and regions list.
    """
    if not annotations:
        page = 1
    else:
        page = annotations[0].page

    regions: list = []
    for ann in annotations:
        b = ann.bbox
        x = max(0, b["x"] - pad)
        y = max(0, b["y"] - pad)
        x2 = min(width,  b["x"] + b["w"] + pad)
        y2 = min(height, b["y"] + b["h"] + pad)
        if (x2 - x) * (y2 - y) < _MIN_TEXT_REGION_AREA:
            continue
        regions.append({"x": x, "y": y, "w": x2 - x, "h": y2 - y})

    return TextMask(page=page, regions=regions)

def apply_text_mask(
    image_array: np.ndarray,
    mask: TextMask,
) -> np.ndarray:
    """
    Blank out text regions in a binary image (set to 0 = paper/background).

    Args:
        image_array: Binary uint8 (H, W). Modified copy returned.
        mask:        TextMask with pixel regions to blank.

    Returns:
        New array with text regions set to 0.
    """
    masked = image_array.copy()
    h, w = masked.shape[:2]
    for r in mask.regions:
        x1 = max(0, r["x"])
        y1 = max(0, r["y"])
        x2 = min(w, r["x"] + r["w"])
        y2 = min(h, r["y"] + r["h"])
        if x2 > x1 and y2 > y1:
            masked[y1:y2, x1:x2] = 0
    return masked

# ---------------------------------------------------------------------------
# Diagnostics helper
# ---------------------------------------------------------------------------

def _ocr_diagnostics(annotations: list, source: str) -> dict:
    """Produce per-stage OCR diagnostics for Stage 6."""
    if not annotations:
        return {
            "engine": source,
            "regions_found": 0,
            "avg_confidence": 0.0,
            "garbled_count": 0,
            "known_symbols_parsed": 0,
        }
    confidences = [a.ocr_confidence for a in annotations]
    garbled = sum(
        1 for a in annotations if a.parsed.token_type == "unknown"
    )
    known = sum(
        1 for a in annotations if a.parsed.token_type not in ("unknown", "label")
    )
    return {
        "engine": source,
        "regions_found": len(annotations),
        "avg_confidence": round(sum(confidences) / len(confidences), 4),
        "garbled_count": garbled,
        "known_symbols_parsed": known,
    }

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_stage2(
    preprocessed: PreprocessedImage,
    pdf_text_layer: list,
    *,
    min_ocr_confidence: float = 0.0,
) -> tuple:
    """
    Run Stage 2 (OCR + symbol parsing) on one preprocessed page.

    Strategy:
        1. If pdf_text_layer has runs for this page → use those (authoritative,
           confidence=1.0). Skip Tesseract.
        2. Otherwise → run Tesseract on the full preprocessed image.

    Returns:
        (annotations: List[TextAnnotation], mask: TextMask, diagnostics: dict)
    """
    t0 = time.monotonic()
    page = preprocessed.page_number

    # ── Path A: vector PDF text layer
    page_runs = [r for r in pdf_text_layer if r.get("page", 1) == page]
    if page_runs:
        annotations = _annotations_from_pdf_layer(
            pdf_text_layer,
            page=page,
            dpi=preprocessed.width_px / 595.0 * 72.0,  # estimate DPI from width
        )
        source = "pdf_text_layer"
    else:
        # ── Path B: Tesseract OCR
        if TESSERACT_AVAILABLE:
            annotations = run_ocr(
                preprocessed.image_array,
                page=page,
                min_confidence=min_ocr_confidence,
            )
            source = "tesseract"
        else:
            annotations = []
            source = "none"

    mask = build_text_mask(
        annotations,
        width=preprocessed.width_px,
        height=preprocessed.height_px,
    )
    diag = _ocr_diagnostics(annotations, source)
    diag["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 1)

    return annotations, mask, diag

# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


# ── titleblock.py ─────────────────────────────────────────────

"""
stage25_titleblock/titleblock.py — Module 2: Stage 2.5
=======================================================
Title block parser. Separate from symbol parsing (architecture Q4 decision).

Purpose: extract structured metadata from the drawing title block
(part name, scale, units, tolerance class, material, etc.).

Strategy:
    1. Detect the title block region — it is typically in the bottom-right
       corner (~25% of drawing width, ~20% of drawing height).
    2. Run OCR on that region (or use pdf_text_layer annotations that fall
       inside the region).
    3. Match known title block field keywords against extracted text.
    4. Build TitleBlockInfo.

Architecture note (Q4): keeping this separate from Stage 2 (symbol_parser)
means a title-block OCR failure cannot corrupt dimension annotations, and vice
versa. The two stages share only the annotations list as input.

Scale string parsing:
    "1:2"  → px_per_mm = render_dpi / 25.4 * (1/2) = scale factor
    "2:1"  → larger-than-real drawing
    If missing → None; Stage 4 tries other anchors.

Public API:
    parse_title_block(annotations, width, height) -> TitleBlockInfo
    extract_scale(scale_raw) -> Optional[float]   # returns ratio (real/drawing)
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Title block region heuristic
# ---------------------------------------------------------------------------

# Fraction of image (from the right/bottom) that is the title block
_TB_RIGHT_FRAC: float = 0.50   # right 50% of width
_TB_BOTTOM_FRAC: float = 0.30  # bottom 30% of height

def _is_in_title_block_region(
    bbox: dict,
    img_width: int,
    img_height: int,
) -> bool:
    """Return True if the bbox centroid is in the expected title-block region."""
    cx = bbox["x"] + bbox.get("w", 0) / 2.0
    cy = bbox["y"] + bbox.get("h", 0) / 2.0
    in_right  = cx >= img_width  * (1.0 - _TB_RIGHT_FRAC)
    in_bottom = cy >= img_height * (1.0 - _TB_BOTTOM_FRAC)
    return in_right and in_bottom

# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

_RE_SCALE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*:\s*(\d+(?:[.,]\d+)?)"
)

_KNOWN_UNITS = {"mm", "millimeter", "millimetre", "in", "inch", "inches"}

def _norm_str(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    return s if s else None

# Keyword groups (case-insensitive) that identify a field
_FIELD_KEYWORDS: list[tuple[str, re.Pattern]] = [
    ("part_name",         re.compile(r"(?:TITLE|PART\s*NAME|BENENNUNG|TEILE?-?NAME)\s*[:\-]?\s*(.+)", re.I)),
    ("drawing_number",    re.compile(r"(?:DWG\s*NO|DRAWING\s*NO|ZEICHNUNGS-?NR|ZNR)\s*[:\-]?\s*(.+)", re.I)),
    ("revision",          re.compile(r"(?:REV(?:ISION)?|ÄENDERUNG)\s*[:\-]?\s*(.+)", re.I)),
    ("material",          re.compile(r"(?:MATERIAL|WERKSTOFF)\s*[:\-]?\s*(.+)", re.I)),
    ("scale_raw",         re.compile(r"(?:SCALE|MASSTAB|MASSSTAB|M)\s*[:\-]?\s*(.+)", re.I)),
    ("date",              re.compile(r"(?:DATE|DATUM)\s*[:\-]?\s*(.+)", re.I)),
    ("author",            re.compile(r"(?:DRAWN|GEZEICHNET|ERSTELLT)\s*[:\-]?\s*(.+)", re.I)),
    ("sheet",             re.compile(r"(?:SHEET|BLATT)\s*[:\-]?\s*(.+)", re.I)),
    ("tolerance_general", re.compile(r"(?:TOL(?:ERANCE)?|ALLGEM(?:EIN)?\.?\s*TOL|ATol)\s*[:\-]?\s*(.+)", re.I)),
]

# Standalone scale pattern (no keyword prefix, just "1:2")
_RE_BARE_SCALE = re.compile(r"^(\d+(?:[.,]\d+)?)\s*:\s*(\d+(?:[.,]\d+)?)$")

# Units hint patterns
_RE_UNITS_MM = re.compile(r"\bALL\s*DIM\w*\s+IN\s+MM\b|\bMILLIMET|\bMM\b", re.I)
_RE_UNITS_IN = re.compile(r"\bALL\s*DIM\w*\s+IN\s+INCH|\bINCH|\b\"\B", re.I)

def extract_scale(scale_raw: Optional[str]) -> Optional[float]:
    """
    Parse a scale string like "1:2" or "2:1" into a drawing-to-reality ratio.

    "1:2" → the drawing is drawn at half size → real/px ratio = 2.0
    "2:1" → the drawing is twice real size → ratio = 0.5

    Returns:
        ratio (float) = real_mm / drawing_mm, or None if unparseable.
    """
    if not scale_raw:
        return None
    text = scale_raw.replace(",", ".").strip()
    m = _RE_SCALE.search(text)
    if not m:
        return None
    try:
        drawing = float(m.group(1))
        real    = float(m.group(2))
    except (ValueError, TypeError):
        return None
    if drawing <= 0:
        return None
    return round(real / drawing, 6)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_title_block(
    annotations: list,
    width: int,
    height: int,
) -> TitleBlockInfo:
    """
    Extract title block metadata from text annotations.

    Scans all annotations. Prioritises those in the title-block region
    (bottom-right), but falls back to any annotation with a matching keyword.

    Args:
        annotations: List of TextAnnotation from Stage 2.
        width:       Image width in pixels (for region heuristic).
        height:      Image height in pixels.

    Returns:
        TitleBlockInfo with all fields set or None.
    """
    # Separate title-block-region annotations from the rest
    tb_anns = [a for a in annotations
               if _is_in_title_block_region(a.bbox, width, height)]
    all_anns = tb_anns + [a for a in annotations if a not in tb_anns]

    fields: dict = {
        "part_name": None,
        "drawing_number": None,
        "revision": None,
        "material": None,
        "scale_raw": None,
        "units_hint": None,
        "date": None,
        "author": None,
        "sheet": None,
        "tolerance_general": None,
    }
    extra: dict = {}

    # Also check if the annotation was already classified as title_block
    # by the symbol parser
    for ann in all_anns:
        text = ann.raw_text.strip()
        if not text:
            continue

        # Check symbol parser already identified a title-block field
        if ann.parsed.token_type == "title_block" and ann.parsed.qualifier:
            key_upper = ann.parsed.qualifier.upper().replace(" ", "_")
            value = text.split(":", 1)[-1].strip() if ":" in text else text

            if "MATERIAL" in key_upper and fields["material"] is None:
                fields["material"] = _norm_str(value)
            elif ("SCALE" in key_upper or "MASSSTAB" in key_upper) and fields["scale_raw"] is None:
                # value might be "1:2" or "SCALE: 1:2" — extract the ratio part
                m_sc = _RE_SCALE.search(value)
                if m_sc:
                    fields["scale_raw"] = f"{m_sc.group(1)}:{m_sc.group(2)}"
                else:
                    fields["scale_raw"] = _norm_str(value)
            elif "TITLE" in key_upper or "PART" in key_upper:
                if fields["part_name"] is None:
                    fields["part_name"] = _norm_str(value)
            elif "REV" in key_upper:
                if fields["revision"] is None:
                    fields["revision"] = _norm_str(value)
            elif "DWG" in key_upper or "DRAWING" in key_upper:
                if fields["drawing_number"] is None:
                    fields["drawing_number"] = _norm_str(value)
            else:
                extra[key_upper] = _norm_str(value)
            continue

        # Try all keyword regexes
        matched = False
        for field_name, pattern in _FIELD_KEYWORDS:
            m = pattern.match(text)
            if m and fields.get(field_name) is None:
                val = _norm_str(m.group(1))
                if val:
                    # Scale special handling
                    if field_name == "scale_raw":
                        m_sc = _RE_SCALE.search(val)
                        if m_sc:
                            val = f"{m_sc.group(1)}:{m_sc.group(2)}"
                    fields[field_name] = val
                matched = True
                break

        if not matched:
            # Standalone scale check
            m_bare = _RE_BARE_SCALE.match(text)
            if m_bare and fields["scale_raw"] is None:
                fields["scale_raw"] = text

        # Units hint
        if fields["units_hint"] is None:
            if _RE_UNITS_MM.search(text):
                fields["units_hint"] = "mm"
            elif _RE_UNITS_IN.search(text):
                fields["units_hint"] = "in"

    return TitleBlockInfo(
        part_name=fields["part_name"],
        drawing_number=fields["drawing_number"],
        revision=fields["revision"],
        material=fields["material"],
        scale_raw=fields["scale_raw"],
        units_hint=fields["units_hint"],
        date=fields["date"],
        author=fields["author"],
        sheet=fields["sheet"],
        tolerance_general=fields["tolerance_general"],
        extra=extra,
    )

# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


# ── classifier.py ─────────────────────────────────────────────

"""
stage3_geometry/classifier.py — Module 2: Stage 3
==================================================
OpenCV contour → geometry type classification.

Reused from old project's 2A.2/shapes.py (REUSE bucket, forensic report §2.5).
Empirical thresholds are preserved verbatim — each value has a documented
rationale that represents months of tuning against real drawings.

CRITICAL FIX vs old project (forensic report §4.4 and architecture lesson #5):
    OLD ORDER: line → rectangle → slot → circle → polygon   ← WRONG
    NEW ORDER: line → SLOT → rectangle → circle → polygon   ← CORRECT

    "Slot vor Polygon-Erkennung prüfen" (master rule 19).
    OpenCV's approxPolyDP can simplify a capsule to ≤4 vertices, making
    it appear as a rectangle. Slot detection must run FIRST.

Additional V2 improvement: every classify call returns (kind, confidence, evidence)
instead of just a string. Confidence is graded (distance from threshold).

Public API:
    classify_contour(contour) -> tuple[str, float, dict]
        kind:       "circle" | "slot" | "rectangle" | "line" | "polygon" | "unknown"
        confidence: 0.0–1.0
        evidence:   dict of raw measurements (for forensics / Stage 6)
    contour_center(contour)   -> tuple[float, float]
    contour_area(contour)     -> float
    contour_bbox(contour)     -> dict
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Empirical thresholds (from old project shapes.py — forensic report §2.5)
# Each has a comment explaining the failing case it was tuned against.
# DO NOT change without benchmark evidence.
# ---------------------------------------------------------------------------

_MIN_AREA: float = 4.0            # below this → degenerate contour

# Circle detection — two paths:
# Path A (large/well-sampled circles): high circularity
_CIRCLE_CIRCULARITY_MIN: float = 0.96
# Rationale: Hexagon circ=0.907, Octagon circ=0.948 → threshold must exceed both.
# Large circles (r=50, n=120 pts): circ=0.979 → CIRCLE ✓

# Path B (small circles, integer raster): lower circularity but near-square bbox
_CIRCLE_SMALL_CIRC_MIN:  float = 0.60
_CIRCLE_SMALL_CIRC_MAX:  float = 0.89
# Rationale: r=10 circle has circ≈0.71 (integer raster artefact).
# Hexagon circ=0.907 → above 0.89 → correctly stays POLYGON.
_CIRCLE_SMALL_APPROX_N_MIN: int   = 5      # ≥5 approx vertices (excludes 4-corner rectangles)
_CIRCLE_ASPECT_MAX:       float = 1.15   # bounding box near-square

# Slot detection
_SLOT_ASPECT_RATIO_MIN: float = 2.5
# Rationale: rect 80×40 has asp=2.0 → must remain RECTANGLE, not SLOT.
_SLOT_CIRCULARITY_MAX:  float = 0.90   # exclude circles
_SLOT_CONVEXITY_MIN:    float = 0.97   # slots are convex (capsule shape)

# Line detection (very narrow rectangle)
_LINE_ASPECT_RATIO_MIN: float = 25.0
# Rationale: rect 200×10 has asp=20 → must remain RECTANGLE (keep > 20).

# Polygon
_POLY_MIN_VERTICES: int = 5            # 4 → rectangle, 5+ → polygon

# approxPolyDP epsilon factor (relative to perimeter)
_BBOX_EPS: float = 0.02

# Reclassification: drawn circles (cv2.circle, thickness>0) show circ≈0.87–0.94
# and may be classified as POLYGON by path A + B. Detect and fix.
_RECLASS_CIRCLE_CIRC_MIN: float = 0.87
_RECLASS_CIRCLE_CIRC_MAX: float = 0.945
_RECLASS_CIRCLE_ASP_MAX:  float = 1.06
_RECLASS_CIRCLE_CONV_MIN: float = 0.97

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(contour: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Accept shape (N,1,2) or (N,2) → (N,1,2) int32. Return None if invalid."""
    if contour is None:
        return None
    arr = np.asarray(contour)
    if arr.ndim == 2 and arr.shape[1] == 2:
        arr = arr.reshape(-1, 1, 2)
    if arr.ndim != 3 or arr.shape[1] != 1 or arr.shape[2] != 2:
        return None
    if len(arr) < 3:
        return None
    return arr.astype(np.int32)

def _graded_confidence(value: float, threshold: float, *, above: bool = True,
                        window: float = 0.15) -> float:
    """
    Graded confidence based on how far a metric is from its threshold.
    above=True: higher value is better (circularity, convexity).
    above=False: lower value is better (e.g. aspect for circle).
    window: range over which confidence ramps from 0.5 to 1.0.
    """
    if above:
        margin = value - threshold
    else:
        margin = threshold - value
    if margin <= 0:
        return 0.5    # just barely passing — low confidence
    return min(1.0, 0.5 + 0.5 * (margin / window))

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_contour(
    contour: Optional[np.ndarray],
) -> tuple:
    """
    Classify an OpenCV contour into a geometry type.

    Rule order (CRITICAL — slot before rectangle):
        1. Degenerate (area < _MIN_AREA)  → unknown
        2. Line (extreme aspect ratio)    → line
        3. Slot (aspect + circ + conv)    → slot    ← MUST be before rect
        4. Circle (path A: high circ)     → circle
        5. Circle (path B: small)         → circle
        6. Rectangle (≤4 approx corners)  → rectangle
        7. Polygon (≥5 approx corners)    → polygon
        8. Fallback                       → unknown

    Returns:
        (kind: str, confidence: float, evidence: dict)
    """
    contour = _normalise(contour)
    if contour is None:
        return ("unknown", 0.0, {})

    area = float(cv2.contourArea(contour))
    if area < _MIN_AREA:
        return ("unknown", 0.0, {"area": area})

    perimeter = float(cv2.arcLength(contour, closed=True))
    if perimeter < 1e-6:
        return ("unknown", 0.0, {"area": area, "perimeter": 0.0})

    # --- Core metrics ---
    circularity = (4.0 * math.pi * area) / (perimeter ** 2)

    eps = _BBOX_EPS * perimeter
    approx = cv2.approxPolyDP(contour, eps, closed=True)
    n_vertices = len(approx)

    _, (w, h), _ = cv2.minAreaRect(contour)
    short_side = min(w, h)
    long_side  = max(w, h)
    aspect_ratio = long_side / short_side if short_side > 1e-6 else float("inf")

    hull      = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    convexity = area / hull_area if hull_area > 1e-6 else 0.0

    evidence = {
        "area_px":      round(area, 2),
        "perimeter_px": round(perimeter, 2),
        "circularity":  round(circularity, 4),
        "aspect_ratio": round(aspect_ratio, 3),
        "convexity":    round(convexity, 4),
        "n_vertices":   n_vertices,
    }

    # --- Rule 1: Line ---
    if aspect_ratio >= _LINE_ASPECT_RATIO_MIN and n_vertices <= 4:
        conf = _graded_confidence(aspect_ratio, _LINE_ASPECT_RATIO_MIN, window=20.0)
        return ("line", conf, evidence)

    # --- Rule 2: Slot (BEFORE rectangle — master rule 19) ---
    if (
        aspect_ratio >= _SLOT_ASPECT_RATIO_MIN
        and circularity < _SLOT_CIRCULARITY_MAX
        and convexity >= _SLOT_CONVEXITY_MIN
    ):
        # Graded confidence: average of the three signals
        c_asp  = _graded_confidence(aspect_ratio, _SLOT_ASPECT_RATIO_MIN, window=2.0)
        c_conv = _graded_confidence(convexity, _SLOT_CONVEXITY_MIN, window=0.02)
        c_circ = _graded_confidence(_SLOT_CIRCULARITY_MAX, circularity, window=0.15)  # lower circ = better
        conf = round((c_asp + c_conv + c_circ) / 3.0, 4)
        return ("slot", conf, evidence)

    # --- Rule 3a: Circle (large, well-sampled) ---
    if circularity >= _CIRCLE_CIRCULARITY_MIN:
        conf = _graded_confidence(circularity, _CIRCLE_CIRCULARITY_MIN, window=0.04)
        return ("circle", conf, evidence)

    # --- Rule 3b: Circle (small, integer raster) ---
    # r=10 n=32 circle: circ=0.938, asp=1.0, conv=0.988, n=9
    # Octagon: circ=0.948, n=8 → distinguishable by n_vertices > 8 (circles have more)
    # We require n_vertices > 8 (i.e. ≥9) so regular polygons with exactly 8 sides
    # stay as polygon. The path A threshold (0.96) handles well-sampled circles above.
    if (
        _CIRCLE_SMALL_CIRC_MIN <= circularity < _CIRCLE_CIRCULARITY_MIN
        and aspect_ratio <= _CIRCLE_ASPECT_MAX
        and convexity >= 0.80
        and n_vertices > 8                   # exclude regular octagons (exactly 8 sides)
    ):
        # Lower confidence for small circles (raster artefact)
        conf = round(0.5 + 0.3 * (circularity - _CIRCLE_SMALL_CIRC_MIN) /
                     (_CIRCLE_SMALL_CIRC_MAX - _CIRCLE_SMALL_CIRC_MIN), 4)
        return ("circle", conf, evidence)

    # --- Rule 4: Rectangle (≤4 approx corners) ---
    if n_vertices <= 4:
        conf = _graded_confidence(_BBOX_EPS, eps / (perimeter + 1e-9), window=0.01)
        # Simple confidence: how rectangular is it?
        conf = max(0.5, min(1.0, convexity))
        return ("rectangle", round(conf, 4), evidence)

    # --- Rule 5: Polygon (≥5 corners) ---
    if n_vertices >= _POLY_MIN_VERTICES:
        conf = 0.7   # generic polygon — moderate confidence
        return ("polygon", conf, evidence)

    return ("unknown", 0.0, evidence)

def maybe_reclass_drawn_circle(
    contour: Optional[np.ndarray],
    current_kind: str,
) -> str:
    """
    Post-classification fix for drawn circles (cv2.circle with thickness>0).

    When a circle is drawn with thickness > 0 in OpenCV, the contour has
    a ring shape whose circularity is slightly below the main threshold
    (typically 0.87–0.94). It can fall through to 'polygon'.
    This function detects that case and returns 'circle'.

    Args:
        contour:       The contour to check.
        current_kind:  Current classification (only acts if 'polygon').

    Returns:
        'circle' if criteria met, else current_kind unchanged.
    """
    if current_kind != "polygon":
        return current_kind

    c = _normalise(contour)
    if c is None:
        return current_kind

    area      = float(cv2.contourArea(c))
    perimeter = float(cv2.arcLength(c, closed=True))
    if area < _MIN_AREA or perimeter < 1e-6:
        return current_kind

    circularity  = (4.0 * math.pi * area) / (perimeter ** 2)
    hull_area    = float(cv2.contourArea(cv2.convexHull(c)))
    convexity    = area / hull_area if hull_area > 1e-6 else 0.0
    _, (w, h), _ = cv2.minAreaRect(c)
    short        = min(w, h)
    aspect       = max(w, h) / short if short > 1e-6 else float("inf")

    if (
        _RECLASS_CIRCLE_CIRC_MIN <= circularity <= _RECLASS_CIRCLE_CIRC_MAX
        and aspect <= _RECLASS_CIRCLE_ASP_MAX
        and convexity >= _RECLASS_CIRCLE_CONV_MIN
    ):
        return "circle"

    return current_kind

def contour_center(contour: Optional[np.ndarray]) -> tuple:
    """Return (cx, cy) float pair using image moments. Falls back to bbox centre."""
    c = _normalise(contour)
    if c is None:
        return (0.0, 0.0)
    M = cv2.moments(c)
    if M["m00"] > 1e-9:
        return (round(M["m10"] / M["m00"], 2), round(M["m01"] / M["m00"], 2))
    x, y, w, h = cv2.boundingRect(c)
    return (round(x + w / 2.0, 2), round(y + h / 2.0, 2))

def contour_area(contour: Optional[np.ndarray]) -> float:
    """Return contour area in px², or 0.0 for None/degenerate."""
    c = _normalise(contour)
    if c is None:
        return 0.0
    return round(float(cv2.contourArea(c)), 4)

def contour_bbox(contour: Optional[np.ndarray]) -> dict:
    """Return axis-aligned bounding box {x, y, width, height, cx, cy}."""
    c = _normalise(contour)
    if c is None:
        return {"x": 0, "y": 0, "width": 0, "height": 0, "cx": 0.0, "cy": 0.0}
    x, y, w, h = cv2.boundingRect(c)
    return {
        "x": int(x), "y": int(y),
        "width": int(w), "height": int(h),
        "cx": round(x + w / 2.0, 2),
        "cy": round(y + h / 2.0, 2),
    }

# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _make_circle_contour(cx: int, cy: int, r: int, n: int = 120) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pts = np.stack(
        [cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1
    ).astype(np.int32)
    return pts.reshape(-1, 1, 2)

def _make_rect_contour(x: int, y: int, w: int, h: int) -> np.ndarray:
    pts = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.int32)
    return pts.reshape(-1, 1, 2)

def _make_slot_contour(cx: int, cy: int, length: int, radius: int, n_arc: int = 40) -> np.ndarray:
    hs = length / 2.0 - radius
    a_r = np.linspace(-math.pi/2, math.pi/2, n_arc)
    arc_r = np.stack([cx + hs + radius*np.cos(a_r), cy + radius*np.sin(a_r)], axis=1)
    a_l = np.linspace(math.pi/2, 3*math.pi/2, n_arc)
    arc_l = np.stack([cx - hs + radius*np.cos(a_l), cy + radius*np.sin(a_l)], axis=1)
    pts = np.vstack([arc_r, arc_l]).astype(np.int32)
    return pts.reshape(-1, 1, 2)

def _make_polygon_contour(cx: int, cy: int, r: int, sides: int) -> np.ndarray:
    angles = np.linspace(0, 2*math.pi, sides, endpoint=False)
    pts = np.stack([cx + r*np.cos(angles), cy + r*np.sin(angles)], axis=1).astype(np.int32)
    return pts.reshape(-1, 1, 2)


# ── detector.py ───────────────────────────────────────────────

"""
stage3_geometry/detector.py — Module 2: Stage 3 orchestrator
=============================================================
Runs all geometry detectors on a text-masked binary image and returns
a combined list of GeometryCandidate objects, confidence-tagged.

Architecture rules honoured:
- RETR_LIST (not RETR_EXTERNAL): inner contours are NEVER discarded.
  Forensic report §3.3 / lesson #3: RETR_EXTERNAL was the prior silent failure.
- No hard area filter: small contours get low confidence, not deletion.
- Slot detection fires via classifier before rectangle (master rule 19).
- All candidates carry confidence + evidence for Stage 5's conflict resolution.
- Detectors are deterministic: same image → same output.

Detector pipeline (applied to the same image in order):
    1. Hough circle detector  (catches circles missed by contour approx)
    2. Contour classifier     (classifies all RETR_LIST contours)
       Contour order: slot → rectangle → polygon → line → unknown
       (circle handled if Hough missed it)
    3. Merge: deduplication by IoU overlap

The Hough detector is better for small holes (high DPI artefacts confuse
contour approx). The contour classifier is better for slots, rectangles,
and polygons. Running both and merging gives full coverage.

Public API:
    detect(image_array, page, *, min_area_px) -> List[GeometryCandidate]
"""

from pathlib import Path



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum contour area for confidence scoring — NOT for deletion.
# Contours below this receive confidence=0.2 (very low, flagged for review).
_CONFIDENCE_PENALTY_AREA: float = 20.0

# Hough circle detection params (empirical — tuned for 300 DPI drawings)
_HOUGH_DP:           float = 1.2    # inverse accumulator resolution
_HOUGH_MIN_DIST:     float = 10.0   # min distance between circle centres (px)
_HOUGH_PARAM1:       float = 50.0   # Canny higher threshold
_HOUGH_PARAM2:       float = 25.0   # accumulator threshold (lower = more detections)
_HOUGH_MIN_RADIUS:   int   = 3      # minimum radius (px) — catches Ø2mm holes at 300dpi
_HOUGH_MAX_RADIUS:   int   = 0      # 0 = no maximum

# IoU threshold for deduplication between Hough and contour candidates
_IOU_DEDUP_THRESHOLD: float = 0.4

# ID counters (module-level, reset per detect() call)
_SEQ: int = 0

def _next_id(kind: str) -> str:
    global _SEQ
    _SEQ += 1
    return f"cand_{kind}_{_SEQ:04d}"

# ---------------------------------------------------------------------------
# Hough circle detector
# ---------------------------------------------------------------------------

def _detect_circles_hough(
    binary: np.ndarray,
    page: int,
) -> list:
    """
    Hough-transform circle detection. Multi-pass for small and large radii.

    Uses RETR-independent approach — works directly on gradient, not contours.
    Especially good for small holes (r < 15 px) where contour approx fails.

    Returns:
        List of GeometryCandidate with kind='circle'.
    """
    # Hough requires uint8 grayscale input
    # Our binary is ink=255 on paper=0; invert for Hough
    gray_for_hough = 255 - binary

    circles_raw = cv2.HoughCircles(
        gray_for_hough,
        cv2.HOUGH_GRADIENT,
        dp=_HOUGH_DP,
        minDist=_HOUGH_MIN_DIST,
        param1=_HOUGH_PARAM1,
        param2=_HOUGH_PARAM2,
        minRadius=_HOUGH_MIN_RADIUS,
        maxRadius=_HOUGH_MAX_RADIUS,
    )

    candidates: list = []
    if circles_raw is None:
        return candidates

    for cx, cy, r in circles_raw[0]:
        cx, cy, r = float(cx), float(cy), float(r)
        # Confidence based on accumulator score (approximated via radius plausibility)
        # Hough has no direct per-circle score in HOUGH_GRADIENT
        # Use heuristic: perfect circles at reasonable sizes → higher confidence
        conf = min(1.0, 0.6 + 0.2 * min(r / 20.0, 1.0))

        candidates.append(GeometryCandidate(
            id=_next_id("circle"),
            kind="circle",
            geometry={
                "cx": round(cx, 1),
                "cy": round(cy, 1),
                "radius_px": round(r, 1),
                "diameter_px": round(2 * r, 1),
            },
            confidence=round(conf, 4),
            detector="hough_circles",
            page=page,
            evidence={
                "radius_px": round(r, 1),
                "method": "hough_gradient",
            },
        ))

    return candidates

# ---------------------------------------------------------------------------
# Contour classifier pipeline
# ---------------------------------------------------------------------------

def _detect_contours(
    binary: np.ndarray,
    page: int,
) -> list:
    """
    Find all contours with RETR_LIST (NO inner geometry deleted — lesson #3)
    and classify each one.

    Returns:
        List of GeometryCandidate for all detected geometries.
    """
    # RETR_LIST: ALL contours returned, no hierarchy filtering
    contours, _ = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates: list = []

    for contour in contours:
        area = contour_area(contour)

        # No hard area filter — small contours get low confidence
        if area < 1.0:   # truly degenerate (0-1 pixel)
            continue

        kind, conf, evidence = classify_contour(contour)

        # Apply drawn-circle reclassification fix (forensic report §3.3)
        if kind == "polygon":
            kind = maybe_reclass_drawn_circle(contour, kind)
            if kind == "circle":
                conf = 0.65   # reclassified — moderate confidence

        # Downgrade confidence for sub-threshold area (not delete)
        if area < _CONFIDENCE_PENALTY_AREA:
            conf = min(conf, 0.25)
            evidence["low_area_flag"] = True

        cx, cy = contour_center(contour)
        bbox   = contour_bbox(contour)
        evidence["area_px"] = round(area, 2)

        # Build geometry dict per kind
        geometry: dict = {
            "cx": cx, "cy": cy,
            "bbox": bbox,
        }

        if kind == "circle":
            # Estimate radius from area and bbox
            r_from_area = math.sqrt(area / math.pi)
            r_from_bbox = (bbox["width"] + bbox["height"]) / 4.0
            radius_px   = (r_from_area + r_from_bbox) / 2.0
            geometry["radius_px"]   = round(radius_px, 1)
            geometry["diameter_px"] = round(2 * radius_px, 1)

        elif kind == "slot":
            geometry["length_px"]  = round(max(bbox["width"], bbox["height"]), 1)
            geometry["width_px"]   = round(min(bbox["width"], bbox["height"]), 1)
            geometry["aspect"]     = evidence.get("aspect_ratio", 0.0)

        elif kind == "rectangle":
            geometry["width_px"]   = round(bbox["width"], 1)
            geometry["height_px"]  = round(bbox["height"], 1)

        elif kind == "line":
            # Approximate endpoints from bounding box
            geometry["x1"] = bbox["x"]
            geometry["y1"] = bbox["y"] + bbox["height"] // 2
            geometry["x2"] = bbox["x"] + bbox["width"]
            geometry["y2"] = bbox["y"] + bbox["height"] // 2

        candidates.append(GeometryCandidate(
            id=_next_id(kind),
            kind=kind,
            geometry=geometry,
            confidence=round(conf, 4),
            detector="contour_classifier",
            page=page,
            evidence=evidence,
        ))

    # Sort by area descending (largest contour first — usually outer part boundary)
    candidates.sort(
        key=lambda c: c.evidence.get("area_px", 0.0),
        reverse=True,
    )

    return candidates

# ---------------------------------------------------------------------------
# IoU-based deduplication (Hough vs contour candidates)
# ---------------------------------------------------------------------------

def _circle_iou(c1: dict, c2: dict) -> float:
    """IoU between two circle geometries (both must have cx, cy, radius_px)."""
    r1 = c1.get("radius_px", 0.0)
    r2 = c2.get("radius_px", 0.0)
    dx = c1["cx"] - c2["cx"]
    dy = c1["cy"] - c2["cy"]
    dist = math.sqrt(dx * dx + dy * dy)

    if dist >= r1 + r2:
        return 0.0    # no overlap
    if dist <= abs(r1 - r2):
        # One circle fully inside the other
        smaller_area = math.pi * min(r1, r2) ** 2
        larger_area  = math.pi * max(r1, r2) ** 2
        return smaller_area / larger_area if larger_area > 0 else 0.0

    # Partial overlap
    a = r1 * r1
    b = r2 * r2
    d = dist
    try:
        angle1 = 2 * math.acos((d * d + a - b) / (2 * d * math.sqrt(a)))
        angle2 = 2 * math.acos((d * d + b - a) / (2 * d * math.sqrt(b)))
    except (ValueError, ZeroDivisionError):
        return 0.0
    inter = 0.5 * a * (angle1 - math.sin(angle1)) + 0.5 * b * (angle2 - math.sin(angle2))
    union = math.pi * (a + b) - inter
    return inter / union if union > 0 else 0.0

def _deduplicate(hough: list, contour: list) -> list:
    """
    Merge Hough and contour candidates. Remove duplicates where IoU > threshold.
    Prefer the candidate with higher confidence when deduplicating.

    Non-circle contour candidates are never removed by deduplication.
    """
    hough_circles  = [c for c in hough   if c.kind == "circle"]
    other_contours = [c for c in contour if c.kind != "circle"]
    cont_circles   = [c for c in contour if c.kind == "circle"]

    merged_circles: list = list(hough_circles)   # start with Hough

    for cc in cont_circles:
        # Check if this contour circle overlaps any Hough circle
        dominated = False
        for hc in hough_circles:
            iou = _circle_iou(cc.geometry, hc.geometry)
            if iou >= _IOU_DEDUP_THRESHOLD:
                # Prefer higher confidence
                if cc.confidence > hc.confidence:
                    # Replace the Hough candidate
                    hough_circles.remove(hc)
                    merged_circles.remove(hc)
                    merged_circles.append(cc)
                dominated = True
                break
        if not dominated:
            merged_circles.append(cc)

    return merged_circles + other_contours

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(
    image_array: np.ndarray,
    page: int,
    *,
    min_area_px: float = 0.0,   # informational only — not a hard filter
) -> list:
    """
    Run all geometry detectors on a text-masked binary image.

    Args:
        image_array:  Binary uint8 (H, W). ink=255, paper=0. Text already blanked.
        page:         1-based page number.
        min_area_px:  Candidates below this area get a weak_signal flag, not deleted.

    Returns:
        List of GeometryCandidate, sorted by area descending.
        Always returns a list (never raises on empty or noisy image).
    """
    global _SEQ
    _SEQ = 0   # reset per call for determinism

    if image_array.size == 0:
        return []

    # ── Hough circle detector
    hough_candidates = _detect_circles_hough(image_array, page)

    # ── Contour classifier (RETR_LIST — inner geometry preserved)
    contour_candidates = _detect_contours(image_array, page)

    # ── Deduplicate circles between the two detectors
    all_candidates = _deduplicate(hough_candidates, contour_candidates)

    # ── Annotate small-area candidates (not delete, just flag)
    if min_area_px > 0:
        for c in all_candidates:
            if c.evidence.get("area_px", float("inf")) < min_area_px:
                c.evidence["below_min_area"] = True
                if "small_contour" not in (c.evidence.get("flags") or []):
                    pass   # flag is in evidence dict — visible to Stage 5

    # Sort by area descending
    all_candidates.sort(
        key=lambda c: c.evidence.get("area_px", 0.0),
        reverse=True,
    )

    return all_candidates

# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _make_test_image(shapes: list, size: tuple = (400, 400)) -> np.ndarray:
    """Synthetic binary drawing image for testing."""
    h, w = size[1], size[0]
    img = np.zeros((h, w), dtype=np.uint8)
    for shape in shapes:
        t = shape["type"]
        if t == "circle":
            cv2.circle(img, (shape["cx"], shape["cy"]), shape["r"], 255, -1)
        elif t == "rectangle":
            cv2.rectangle(img,
                          (shape["x"], shape["y"]),
                          (shape["x"]+shape["w"], shape["y"]+shape["h"]), 255, -1)
        elif t == "slot":
            # Draw capsule
            cx, cy = shape["cx"], shape["cy"]
            hl = shape["length"] // 2 - shape["radius"]
            cv2.rectangle(img,
                          (cx - hl, cy - shape["radius"]),
                          (cx + hl, cy + shape["radius"]), 255, -1)
            cv2.circle(img, (cx + hl, cy), shape["radius"], 255, -1)
            cv2.circle(img, (cx - hl, cy), shape["radius"], 255, -1)
    return img


# ── scale_detector.py ─────────────────────────────────────────

"""
stage4_scale/scale_detector.py — Module 2: Stage 4
===================================================
Pixel → part space (mm) coordinate transform.

Scale anchor priority (TECH_DRAWING_INTERPRETER §14):
    1. Dimension text anchor — a text annotation with a known value paired
       with a geometry candidate whose pixel size is measurable.
    2. Title block scale string (e.g. "1:2" → scale factor 2.0).
    3. PDF page metadata — A4=210×297mm, A3=297×420mm etc.
    4. Operator override (passed as argument).
    5. Unknown — emit candidates in pixel coordinates with reduced confidence.

Architecture rules:
    - Never invent a scale (master rule 2: Keine Maße schätzen).
    - When scale is unknown, report honestly with reduced confidence.
    - Per-feature confidence is reduced by _UNKNOWN_SCALE_PENALTY.

Public API:
    detect_scale(annotations, candidates, page_width_px, page_height_px,
                 title_block, pdf_dpi, operator_override_mm_per_px)
                 -> ScaleInfo

    apply_scale(candidates, scale) -> List[ScaledCandidate]
    px_to_mm(px_value, scale) -> float
    point_px_to_mm(x, y, scale) -> tuple[float, float]
"""

from pathlib import Path



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ISO 216 sheet sizes (width × height in mm), landscape and portrait
_SHEET_SIZES: dict = {
    "A0": (841.0, 1189.0), "A1": (594.0, 841.0),
    "A2": (420.0, 594.0),  "A3": (297.0, 420.0),
    "A4": (210.0, 297.0),  "A5": (148.0, 210.0),
    "B1": (707.0, 1000.0), "B2": (500.0, 707.0),
}

# Tolerance for dimension-text anchor matching:
# How closely a text value must match a geometry's pixel dimension
_DIM_ANCHOR_TOLERANCE: float = 0.10   # ±10% of pixel size

# Minimum confidence for a dimension-text anchor to be trusted
_DIM_ANCHOR_MIN_OCR_CONF: float = 0.70

# Penalty applied to per-feature confidence when no scale anchor is available
_UNKNOWN_SCALE_PENALTY: float = 0.40  # confidence *= (1 - 0.40)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mm_per_px_from_dpi(dpi: float) -> Optional[float]:
    """One pixel = 25.4 / DPI mm."""
    if dpi > 0:
        return 25.4 / dpi
    return None

def _detect_from_sheet(
    page_width_px: int,
    page_height_px: int,
    dpi: float,
) -> Optional[tuple]:
    """
    Try to identify the sheet size from rendered page dimensions.
    Returns (mm_per_px, sheet_name) or None.
    """
    mm_per_px = _mm_per_px_from_dpi(dpi)
    if mm_per_px is None:
        return None

    width_mm  = page_width_px  * mm_per_px
    height_mm = page_height_px * mm_per_px

    best: Optional[tuple] = None
    best_err = float("inf")

    for name, (sw, sh) in _SHEET_SIZES.items():
        # Try both landscape and portrait
        for ow, oh in [(sw, sh), (sh, sw)]:
            err = abs(width_mm - ow) / ow + abs(height_mm - oh) / oh
            if err < best_err and err < 0.08:   # within 8% — allow scan margin
                best_err = err
                best = (mm_per_px, name)

    return best

def _detect_from_title_block(title_block: Optional[TitleBlockInfo]) -> Optional[float]:
    """Extract mm_per_px from title block scale string."""
    if title_block is None or title_block.scale_raw is None:
        return None
    ratio = extract_scale(title_block.scale_raw)   # real/drawing ratio
    if ratio is None:
        return None
    # ratio = real_mm / drawing_mm
    # 1 drawing_mm = ratio real_mm
    # We need mm_per_px — we don't know DPI here.
    # Store the ratio separately; caller applies it after detecting DPI.
    return ratio  # Note: this is the scale ratio, NOT mm_per_px directly

def _detect_from_dimension_text(
    annotations: list,
    candidates: list,
) -> Optional[float]:
    """
    Find a text annotation that is a known linear dimension and is associated
    with a geometry candidate whose pixel size can be measured.

    Strategy:
        - Find dimension_linear and dimension_diameter annotations with
          known mm values.
        - For each, find the nearest circle/slot/rect candidate.
        - If annotation value ÷ feature pixel size ≈ consistent → use as anchor.

    Returns:
        mm_per_px estimate, or None if no reliable anchor found.
    """
    dim_anns = [
        a for a in annotations
        if a.parsed.token_type in ("dimension_linear", "dimension_diameter",
                                   "dimension_radial")
        and a.parsed.value is not None
        and a.parsed.value > 0
        and a.parsed.unit == "mm"
        and a.ocr_confidence >= _DIM_ANCHOR_MIN_OCR_CONF
    ]

    if not dim_anns or not candidates:
        return None

    estimates: list = []

    for ann in dim_anns:
        val_mm = float(ann.parsed.value)
        ann_cx = ann.bbox["x"] + ann.bbox["w"] / 2.0
        ann_cy = ann.bbox["y"] + ann.bbox["h"] / 2.0

        # Find nearest candidate
        best_dist = float("inf")
        best_cand: Optional[GeometryCandidate] = None
        for cand in candidates:
            cx = cand.geometry.get("cx", 0.0)
            cy = cand.geometry.get("cy", 0.0)
            dist = math.sqrt((ann_cx - cx)**2 + (ann_cy - cy)**2)
            if dist < best_dist:
                best_dist = dist
                best_cand = cand

        if best_cand is None or best_dist > 400:   # too far
            continue

        # Match annotation type to geometry feature
        ann_type = ann.parsed.token_type
        kind     = best_cand.kind
        px_size: Optional[float] = None

        if ann_type == "dimension_diameter" and kind == "circle":
            diam_px = best_cand.geometry.get("diameter_px")
            if diam_px and diam_px > 0:
                px_size = diam_px

        elif ann_type == "dimension_radial" and kind == "circle":
            r_px = best_cand.geometry.get("radius_px")
            if r_px and r_px > 0:
                px_size = r_px * 2   # value is radius; annotation may be "R..."
                val_mm  = float(ann.parsed.value) * 2

        elif ann_type == "dimension_linear":
            # Try to match a slot length or rectangle width
            if kind == "slot":
                length_px = best_cand.geometry.get("length_px")
                if length_px and length_px > 0:
                    px_size = length_px
            elif kind == "rectangle":
                w_px = best_cand.geometry.get("width_px", 0)
                h_px = best_cand.geometry.get("height_px", 0)
                # Use whichever dimension is closer to the annotation value
                px_size = max(w_px, h_px)   # rough heuristic

        if px_size and px_size > 0:
            mm_per_px = val_mm / px_size
            if 0.02 < mm_per_px < 2.0:   # sanity: 0.02 mm/px (500 DPI) to 2 mm/px (12 DPI)
                estimates.append(mm_per_px)

    if not estimates:
        return None

    # Return median (robust to outliers)
    estimates.sort()
    mid = len(estimates) // 2
    return estimates[mid]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_scale(
    annotations: list,
    candidates: list,
    page_width_px: int,
    page_height_px: int,
    title_block: Optional[TitleBlockInfo] = None,
    pdf_dpi: float = 0.0,
    operator_override_mm_per_px: Optional[float] = None,
) -> ScaleInfo:
    """
    Determine the pixel-to-mm scale for one page.

    Priority:
        1. Operator override (explicit)
        2. Dimension text anchor (most accurate)
        3. Title block scale + DPI
        4. Sheet size detection
        5. Raw DPI
        6. Unknown (emit with low confidence)

    Returns:
        ScaleInfo with mm_per_px and origin (currently always px origin = (0,0)).
    """
    page = 1   # caller must pass per-page; default 1 for single-page

    # ── 1. Operator override
    if operator_override_mm_per_px is not None and operator_override_mm_per_px > 0:
        return ScaleInfo(
            page=page,
            px_per_mm=1.0 / operator_override_mm_per_px,
            anchor_method="operator_override",
            anchor_confidence=1.0,
            origin_px={"x": 0.0, "y": 0.0},
        )

    mm_per_px: Optional[float] = None
    method = "unknown"
    confidence = 0.3

    # ── 2. Dimension text anchor
    dim_anchor = _detect_from_dimension_text(annotations, candidates)
    if dim_anchor is not None:
        mm_per_px  = dim_anchor
        method     = "dimension_text"
        confidence = 0.90

    # ── 3. Title block + DPI fallback
    if mm_per_px is None and title_block is not None:
        ratio = _detect_from_title_block(title_block)   # real/drawing ratio
        if ratio is not None and pdf_dpi > 0:
            # mm_per_px = (25.4 / dpi) * ratio
            mm_per_px  = (25.4 / pdf_dpi) * ratio
            method     = "title_block_scale"
            confidence = 0.75

    # ── 4. Sheet size identification
    if mm_per_px is None and pdf_dpi > 0:
        sheet_result = _detect_from_sheet(page_width_px, page_height_px, pdf_dpi)
        if sheet_result is not None:
            mm_per_px  = sheet_result[0]
            method     = f"sheet_size_{sheet_result[1]}"
            confidence = 0.65

    # ── 5. Raw DPI
    if mm_per_px is None and pdf_dpi > 0:
        mm_per_px  = _mm_per_px_from_dpi(pdf_dpi)
        method     = "pdf_metadata"
        confidence = 0.55

    # ── 6. Unknown — we cannot invent a scale
    if mm_per_px is None:
        return ScaleInfo(
            page=page,
            px_per_mm=0.0,            # 0 = unknown
            anchor_method="unknown",
            anchor_confidence=0.0,
            origin_px={"x": 0.0, "y": 0.0},
        )

    px_per_mm = 1.0 / mm_per_px if mm_per_px > 0 else 0.0

    return ScaleInfo(
        page=page,
        px_per_mm=round(px_per_mm, 6),
        anchor_method=method,
        anchor_confidence=round(confidence, 4),
        origin_px={"x": 0.0, "y": 0.0},
    )

def px_to_mm(px_value: float, scale: ScaleInfo) -> float:
    """
    Convert a pixel measurement to millimetres.
    Returns the px value unchanged if scale is unknown (px_per_mm == 0).
    """
    if scale.px_per_mm <= 0:
        return round(float(px_value), 4)
    return round(float(px_value) / scale.px_per_mm, 4)

def point_px_to_mm(x: float, y: float, scale: ScaleInfo) -> tuple:
    """Convert a pixel point to mm space, applying the origin offset."""
    ox = scale.origin_px.get("x", 0.0)
    oy = scale.origin_px.get("y", 0.0)
    if scale.px_per_mm <= 0:
        return (round(float(x - ox), 4), round(float(y - oy), 4))
    return (
        round((x - ox) / scale.px_per_mm, 4),
        round((y - oy) / scale.px_per_mm, 4),
    )

def apply_scale(candidates: list, scale: ScaleInfo) -> list:
    """
    Convert all candidates' geometry from pixel to mm coordinates.

    Args:
        candidates: List of GeometryCandidate.
        scale:      ScaleInfo from detect_scale().

    Returns:
        List of ScaledCandidate with geometry_mm populated.
    """
    scaled: list = []

    for cand in candidates:
        g = cand.geometry
        kind = cand.kind

        # Apply confidence penalty if scale is unknown
        feature_conf = cand.confidence
        if scale.anchor_method == "unknown":
            feature_conf = round(feature_conf * (1.0 - _UNKNOWN_SCALE_PENALTY), 4)

        if kind == "circle":
            cx_mm, cy_mm = point_px_to_mm(g["cx"], g["cy"], scale)
            r_mm = px_to_mm(g.get("radius_px", 0.0), scale)
            geometry_mm = {
                "cx": cx_mm, "cy": cy_mm,
                "radius": r_mm,
                "diameter": round(r_mm * 2, 4),
            }

        elif kind == "slot":
            cx_mm, cy_mm = point_px_to_mm(g["cx"], g["cy"], scale)
            length_mm = px_to_mm(g.get("length_px", 0.0), scale)
            width_mm  = px_to_mm(g.get("width_px", 0.0), scale)
            geometry_mm = {
                "cx": cx_mm, "cy": cy_mm,
                "length": length_mm, "width": width_mm,
            }

        elif kind == "rectangle":
            bbox = g.get("bbox", {})
            cx_mm, cy_mm = point_px_to_mm(
                bbox.get("cx", g.get("cx", 0.0)),
                bbox.get("cy", g.get("cy", 0.0)),
                scale,
            )
            w_mm = px_to_mm(g.get("width_px", bbox.get("width", 0.0)), scale)
            h_mm = px_to_mm(g.get("height_px", bbox.get("height", 0.0)), scale)
            geometry_mm = {
                "cx": cx_mm, "cy": cy_mm,
                "width": w_mm, "height": h_mm,
            }

        else:
            # Generic: convert cx/cy + any pixel dimension
            cx_mm, cy_mm = point_px_to_mm(
                g.get("cx", 0.0), g.get("cy", 0.0), scale
            )
            geometry_mm = {"cx": cx_mm, "cy": cy_mm}

        # Build a modified candidate with updated confidence
        import dataclasses
        updated_cand = dataclasses.replace(cand, confidence=feature_conf)

        scaled.append(ScaledCandidate(
            candidate=updated_cand,
            geometry_mm=geometry_mm,
            scale_info=scale,
        ))

    return scaled

# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


# ── spatial_link.py ───────────────────────────────────────────

"""
stage5_assemble/spatial_link.py — Module 2: Stage 5
=====================================================
Spatial association: pair each TextAnnotation with the nearest
ScaledCandidate using multi-factor scoring.

Scoring weights (from old drawing_linker.py, forensic report §2.5 REUSE):
    type compatibility:  0.50
    spatial proximity:   0.30
    direction plausible: 0.10
    numerical size match: 0.10

Type compatibility matrix (_COMPAT) encodes domain knowledge:
    Ø → hole, circle, thread, pocket
    R → slot, circle, hole, pocket, contour
    M → thread, hole
    group (4×) → hole, circle, thread
    fit → hole, circle
    linear → contour, pocket, slot, rectangle, line

AMBIGUITY_GAP: if top-2 scores differ by < 0.10 → flag ambiguous.

Public API:
    link_text_to_geometry(annotations, scaled_candidates)
        -> List[LinkedPair], List[SuppressedCandidate]
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Constants (from forensic analysis — do not change without benchmark)
# ---------------------------------------------------------------------------

_CONFIDENCE_TYPE: float = 0.50
_CONFIDENCE_PROX: float = 0.30
_CONFIDENCE_DIR:  float = 0.10
_CONFIDENCE_SIZE: float = 0.10

_MIN_CONFIDENCE:    float = 0.20   # below this → unlinked
_AMBIGUITY_GAP:     float = 0.10   # flag if top-2 differ by less
_PROXIMITY_THRESHOLD_MM: float = 50.0  # mm — scale-aware distance cap
_PROXIMITY_THRESHOLD_PX: float = 300.0  # px — fallback if no scale

# Type compatibility (dim_type → acceptable geometry kinds)
_COMPAT: dict = {
    "dimension_diameter": ["circle", "hole", "thread", "pocket"],
    "dimension_radial":   ["slot", "circle", "hole", "pocket", "contour"],
    "thread_callout":     ["circle", "hole"],
    "fit":                ["circle", "hole"],
    "quantity":           ["circle", "hole", "thread"],
    "tolerance":          ["circle", "hole", "slot", "rectangle", "pocket", "contour"],
    "dimension_linear":   ["contour", "pocket", "slot", "rectangle", "line"],
    "chamfer":            ["contour", "rectangle", "polygon"],
    "angle":              ["contour", "slot", "polygon", "line"],
    "surface_finish":     ["contour", "rectangle", "pocket"],
    "label":              [],   # labels don't link to geometry
    "title_block":        [],
    "reference":          [],
    "unknown":            [],
}

# Size-match tolerance band (ratio must be within this of 1.0)
_SIZE_MATCH_TOL: float = 0.05

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _annotation_center(ann: TextAnnotation) -> tuple:
    return bbox_center(ann.bbox)

def _candidate_center(sc: ScaledCandidate) -> tuple:
    g = sc.geometry_mm
    return (g.get("cx", 0.0), g.get("cy", 0.0))

def _distance(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def _type_score(ann: TextAnnotation, sc: ScaledCandidate) -> float:
    compatible = _COMPAT.get(ann.parsed.token_type, [])
    return _CONFIDENCE_TYPE if sc.candidate.kind in compatible else 0.0

def _proximity_score(
    ann_center: tuple,
    cand_center: tuple,
    scale_px_per_mm: float,
) -> float:
    """Distance score in mm space (or px if scale unknown)."""
    if scale_px_per_mm > 0:
        # ann_center is in px; convert to mm
        ann_mm = (ann_center[0] / scale_px_per_mm, ann_center[1] / scale_px_per_mm)
        dist = _distance(ann_mm, cand_center)
        threshold = _PROXIMITY_THRESHOLD_MM
    else:
        # No scale — use px directly, cand_center is also px
        ann_px = ann_center
        cand_px = (
            sc.candidate.geometry.get("cx", cand_center[0]),
            sc.candidate.geometry.get("cy", cand_center[1]),
        )
        dist = _distance(ann_px, cand_px)
        threshold = _PROXIMITY_THRESHOLD_PX

    if dist >= threshold:
        return 0.0
    return _CONFIDENCE_PROX * (1.0 - dist / threshold)

def _direction_score(
    ann_center: tuple,
    cand_center: tuple,
    scale_px_per_mm: float,
) -> float:
    """
    Direction plausibility: dimension text is typically placed outside the
    feature. We give full score if the annotation is not inside the feature bbox.
    Simple heuristic: always give 0.07 (partial) — direction is hard without
    leader line info.
    """
    return _CONFIDENCE_DIR * 0.7

def _size_score(ann: TextAnnotation, sc: ScaledCandidate) -> float:
    """
    Numerical size match: does the annotation value match the feature dimensions?
    """
    val = ann.parsed.value
    if val is None or val <= 0:
        return 0.0

    g = sc.geometry_mm
    kind = sc.candidate.kind
    feat_dim: Optional[float] = None

    if kind == "circle":
        t = ann.parsed.token_type
        if t == "dimension_diameter":
            feat_dim = g.get("diameter")
        elif t == "dimension_radial":
            feat_dim = g.get("radius")

    elif kind == "slot":
        t = ann.parsed.token_type
        if t == "dimension_linear":
            feat_dim = g.get("length") or g.get("width")

    elif kind == "rectangle":
        if ann.parsed.token_type == "dimension_linear":
            feat_dim = max(g.get("width", 0.0), g.get("height", 0.0))

    if feat_dim is None or feat_dim <= 0:
        return 0.0

    ratio = float(val) / feat_dim
    if abs(ratio - 1.0) <= _SIZE_MATCH_TOL:
        return _CONFIDENCE_SIZE
    return 0.0

# Module-level variable for use inside _proximity_score closure
sc: ScaledCandidate = None  # type: ignore

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def link_text_to_geometry(
    annotations: list,
    scaled_candidates: list,
) -> tuple:
    """
    Associate each text annotation with the best geometry candidate.

    Args:
        annotations:       List[TextAnnotation] from Stage 2.
        scaled_candidates: List[ScaledCandidate] from Stage 4.

    Returns:
        (pairs: List[LinkedPair], suppressed: List[SuppressedCandidate])

    Annotations with types that never link (label, title_block, reference)
    are silently skipped (not returned as pairs or suppressed).
    Annotations that do not reach _MIN_CONFIDENCE are also not linked
    but are recorded in suppressed.
    """
    pairs: list = []
    suppressed_list: list = []

    if not scaled_candidates:
        return pairs, suppressed_list

    # Get scale for proximity computation
    scale_px_per_mm = 0.0
    if scaled_candidates:
        scale_px_per_mm = scaled_candidates[0].scale_info.px_per_mm

    for ann in annotations:
        # Skip non-geometric annotation types
        if _COMPAT.get(ann.parsed.token_type) == []:
            continue

        ann_center_px = _annotation_center(ann)

        scores: list = []
        for sc_cand in scaled_candidates:
            cand_center_mm = _candidate_center(sc_cand)

            t = _type_score(ann, sc_cand)
            p = _proximity_score(ann_center_px, cand_center_mm, scale_px_per_mm)
            d = _direction_score(ann_center_px, cand_center_mm, scale_px_per_mm)
            s = _size_score(ann, sc_cand)
            total = t + p + d + s
            scores.append((total, sc_cand))

        if not scores:
            continue

        scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_cand = scores[0]

        if best_score < _MIN_CONFIDENCE:
            suppressed_list.append(SuppressedCandidate(
                candidate=best_cand,
                reason=f"no_annotation_match (best_score={best_score:.2f})",
            ))
            continue

        # Check ambiguity
        ambiguous = False
        if len(scores) >= 2:
            second_score = scores[1][0]
            if best_score - second_score < _AMBIGUITY_GAP:
                ambiguous = True

        # Determine link_type
        link_type = _link_type(ann)

        pair = LinkedPair(
            annotation=ann,
            candidate=best_cand,
            link_confidence=round(best_score, 4),
            link_type=link_type,
            ambiguous=ambiguous,
        )
        pairs.append(pair)

    return pairs, suppressed_list

def _link_type(ann: TextAnnotation) -> str:
    mapping = {
        "dimension_diameter": "diameter",
        "dimension_radial":   "radius",
        "thread_callout":     "thread",
        "fit":                "fit",
        "quantity":           "group",
        "tolerance":          "tolerance",
        "dimension_linear":   "linear",
        "chamfer":            "chamfer",
        "surface_finish":     "surface",
    }
    return mapping.get(ann.parsed.token_type, "unknown")

# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


# ── conflict_resolver.py ──────────────────────────────────────

"""
stage5_assemble/conflict_resolver.py — Module 2: Stage 5
=========================================================
Resolve conflicts when multiple geometry candidates claim the same region.

Rules (architecture §5 cross-link):
    - Circle vs slot for the same region: slot wins if confidence ≥ threshold
      AND matching dimension text (length + width) is present.
    - Otherwise circle wins; slot kept as suppressed candidate.
    - No silent drop: losing candidate always appears in SuppressedCandidate list.
    - Overlapping candidates assessed by IoU; threshold = 0.35.

Public API:
    resolve_conflicts(scaled_candidates, pairs)
        -> (kept: List[ScaledCandidate], suppressed: List[SuppressedCandidate])
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OVERLAP_THRESHOLD:       float = 0.35   # IoU above which two candidates conflict
_SLOT_WINS_CONFIDENCE:    float = 0.70   # slot needs this to beat a circle
_CIRCLE_WINS_CONFIDENCE:  float = 0.55   # circle needs this to beat a slot

# ---------------------------------------------------------------------------
# Overlap geometry
# ---------------------------------------------------------------------------

def _bbox_of(sc: ScaledCandidate) -> dict:
    """Get bounding box in mm from ScaledCandidate."""
    g = sc.geometry_mm
    kind = sc.candidate.kind
    cx = g.get("cx", 0.0)
    cy = g.get("cy", 0.0)

    if kind == "circle":
        r = g.get("radius", 0.0)
        return {"x": cx - r, "y": cy - r, "w": r * 2, "h": r * 2}
    elif kind == "slot":
        l = g.get("length", 0.0)
        w = g.get("width", 0.0)
        return {"x": cx - l / 2, "y": cy - w / 2, "w": l, "h": w}
    elif kind == "rectangle":
        w = g.get("width", 0.0)
        h = g.get("height", 0.0)
        return {"x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h}
    else:
        # Generic: estimate from area
        area = sc.candidate.evidence.get("area_px", 100.0)
        half = math.sqrt(area) / 2.0
        return {"x": cx - half, "y": cy - half, "w": half * 2, "h": half * 2}

def _iou(a: dict, b: dict) -> float:
    """IoU of two axis-aligned bounding boxes."""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0

# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def _find_conflicts(candidates: list) -> list:
    """
    Find pairs of candidates whose bounding boxes overlap significantly.

    Returns list of (idx_a, idx_b) tuples where IoU >= _OVERLAP_THRESHOLD.
    """
    conflicts: list = []
    n = len(candidates)
    for i in range(n):
        bbox_i = _bbox_of(candidates[i])
        for j in range(i + 1, n):
            bbox_j = _bbox_of(candidates[j])
            iou = _iou(bbox_i, bbox_j)
            if iou >= _OVERLAP_THRESHOLD:
                conflicts.append((i, j, iou))
    return conflicts

# ---------------------------------------------------------------------------
# Resolution logic
# ---------------------------------------------------------------------------

def _resolve_pair(
    sc_a: ScaledCandidate,
    sc_b: ScaledCandidate,
    iou: float,
    linked_ids: set,
) -> tuple:
    """
    Decide which candidate to keep when two candidates conflict.

    Returns (winner, loser, reason).
    """
    kind_a = sc_a.candidate.kind
    kind_b = sc_b.candidate.kind
    conf_a = sc_a.candidate.confidence
    conf_b = sc_b.candidate.confidence

    # Slot vs circle: special logic
    if {kind_a, kind_b} == {"slot", "circle"}:
        slot = sc_a if kind_a == "slot" else sc_b
        circ = sc_b if kind_a == "slot" else sc_a

        # Slot wins if it has high enough confidence
        if slot.candidate.confidence >= _SLOT_WINS_CONFIDENCE:
            return (slot, circ,
                    f"slot_beats_circle (slot_conf={slot.candidate.confidence:.2f})")
        else:
            return (circ, slot,
                    f"circle_beats_slot (slot_conf={slot.candidate.confidence:.2f} < {_SLOT_WINS_CONFIDENCE})")

    # Prefer the candidate linked to an annotation (has text evidence)
    a_linked = sc_a.candidate.id in linked_ids
    b_linked = sc_b.candidate.id in linked_ids
    if a_linked and not b_linked:
        return (sc_a, sc_b, "text_linked_wins")
    if b_linked and not a_linked:
        return (sc_b, sc_a, "text_linked_wins")

    # Higher confidence wins
    if conf_a >= conf_b:
        return (sc_a, sc_b, f"higher_confidence ({conf_a:.2f} vs {conf_b:.2f})")
    else:
        return (sc_b, sc_a, f"higher_confidence ({conf_b:.2f} vs {conf_a:.2f})")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_conflicts(
    scaled_candidates: list,
    pairs: list,
) -> tuple:
    """
    Remove conflicting geometry candidates, keeping the most credible one.

    Args:
        scaled_candidates: All ScaledCandidate from Stage 4.
        pairs:             LinkedPair list from spatial_link — used to determine
                           which candidates have text evidence.

    Returns:
        (kept: List[ScaledCandidate], suppressed: List[SuppressedCandidate])
    """
    if len(scaled_candidates) <= 1:
        return scaled_candidates, []

    # IDs of candidates that have been linked to text annotations
    linked_ids: set = {p.candidate.candidate.id for p in pairs}

    conflicts = _find_conflicts(scaled_candidates)
    if not conflicts:
        return scaled_candidates, []

    # Track which candidates are suppressed (by index)
    suppressed_indices: set = set()
    suppressed_list: list = []

    for idx_a, idx_b, iou in conflicts:
        if idx_a in suppressed_indices or idx_b in suppressed_indices:
            continue   # already resolved by a prior conflict

        sc_a = scaled_candidates[idx_a]
        sc_b = scaled_candidates[idx_b]

        winner, loser, reason = _resolve_pair(sc_a, sc_b, iou, linked_ids)

        # Find which index is the loser
        loser_idx = idx_a if loser is sc_a else idx_b
        suppressed_indices.add(loser_idx)

        suppressed_list.append(SuppressedCandidate(
            candidate=loser,
            reason=f"conflict_resolution: {reason} (iou={iou:.2f})",
        ))

    kept = [
        sc for i, sc in enumerate(scaled_candidates)
        if i not in suppressed_indices
    ]

    return kept, suppressed_list

# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


# ── feature_inferrer.py ───────────────────────────────────────

"""
stage5_assemble/feature_inferrer.py — Module 2: Stage 5
=========================================================
Map (ScaledCandidate + LinkedPairs) → Module 1 V2 Feature dict.

Feature kind inference rules (architecture §5 stage 5):
    circle + Ø text            → hole
    circle + M text            → thread or tap_hole
    circle + no text           → hole (default, lower confidence)
    circle (largest in part)   → contour (if no Ø text)
    slot   + 2 dims            → slot
    closed rect + dimensions   → pocket (rectangular)
    closed polygon             → pocket (freeform) or contour
    outermost closed shape     → contour
    R text near arc/circle     → radius
    text only (±, H7, g6)      → fills manufacturing fields on associated feature

Manufacturing fields populated (→ Module 1 V2 Feature.manufacturing):
    upper_tol   → tolerancePlus
    lower_tol   → toleranceMinus (stored as negative, converted here)
    fit_code    → fitClass
    thread_pitch → threadPitch

Public API:
    infer_features(kept_candidates, pairs, title_block, page_area_mm2)
        -> List[dict]  (each dict = one Module 1 V2 Feature)
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fraction of total drawing area above which a closed shape is likely the
# outer part contour (not a pocket or hole)
_CONTOUR_AREA_FRAC: float = 0.50

# Feature kind inference confidence adjustments
_CONF_TEXT_CORROBORATED: float = 1.00   # text + geometry agree → no penalty
_CONF_GEOMETRY_ONLY:     float = 0.75   # no text → reduced confidence
_CONF_TEXT_ONLY:         float = 0.55   # text but no clear geometry match

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feature_area_mm2(sc: ScaledCandidate) -> float:
    """Approximate feature area in mm²."""
    import math
    g = sc.geometry_mm
    kind = sc.candidate.kind
    if kind == "circle":
        r = g.get("radius", 0.0)
        return math.pi * r * r
    elif kind == "slot":
        l = g.get("length", 0.0)
        w = g.get("width", 0.0)
        return l * w
    elif kind == "rectangle":
        return g.get("width", 0.0) * g.get("height", 0.0)
    else:
        return sc.candidate.evidence.get("area_px", 0.0) / 100.0   # rough

def _make_manufacturing(pairs_for_feature: list) -> dict:
    """
    Populate manufacturing dict from associated text annotations.
    Maps ParsedSymbol fields → Module 1 V2 manufacturing fields.
    """
    mfg: dict = {}
    for pair in pairs_for_feature:
        p = pair.annotation.parsed
        if p.upper_tol is not None:
            mfg["tolerancePlus"] = float(p.upper_tol)
        if p.lower_tol is not None:
            # lower_tol is stored negative in ParsedSymbol; Module 1 wants positive
            mfg["toleranceMinus"] = abs(float(p.lower_tol))
        if p.fit_code is not None:
            mfg["fitClass"] = p.fit_code
        if p.thread_pitch is not None:
            mfg["threadPitch"] = float(p.thread_pitch)
    return mfg

def _infer_circle_kind(
    sc: ScaledCandidate,
    pairs_for_feature: list,
    is_largest: bool,
) -> str:
    """Determine the Module 1 V2 kind for a circle candidate."""
    for pair in pairs_for_feature:
        t = pair.annotation.parsed.token_type
        qual = pair.annotation.parsed.qualifier or ""
        if t == "thread_callout" or qual == "M":
            # Check if it has a pitch (tap_hole) or not (thread)
            return "tap_hole" if pair.annotation.parsed.thread_pitch else "thread"
        if t in ("dimension_diameter", "fit"):
            return "hole"

    # No text evidence
    if is_largest:
        return "contour"
    return "hole"   # default for unlabelled circles

def _infer_rect_kind(
    sc: ScaledCandidate,
    pairs_for_feature: list,
    is_largest: bool,
) -> str:
    if is_largest:
        return "contour"
    return "pocket"

def _infer_polygon_kind(
    sc: ScaledCandidate,
    pairs_for_feature: list,
    is_largest: bool,
) -> str:
    if is_largest:
        return "contour"
    return "pocket"

# ---------------------------------------------------------------------------
# Main dimension value extraction
# ---------------------------------------------------------------------------

def _primary_value(sc: ScaledCandidate, pairs_for_feature: list) -> Optional[float]:
    """Extract the main dimension value from text evidence."""
    for pair in pairs_for_feature:
        p = pair.annotation.parsed
        if p.value is not None and p.value > 0:
            return float(p.value)
    return None

def _build_geometry_for_kind(
    kind: str,
    sc: ScaledCandidate,
    pairs_for_feature: list,
) -> dict:
    """
    Build the geometry sub-dict for a Module 1 V2 Feature.
    Prefers text-extracted values; falls back to pixel-derived mm values.
    """
    g = sc.geometry_mm
    cx = g.get("cx", 0.0)
    cy = g.get("cy", 0.0)
    text_val = _primary_value(sc, pairs_for_feature)

    if kind in ("hole", "thread", "tap_hole", "countersink", "counterbore"):
        diam = text_val if text_val and text_val > 0 else g.get("diameter", 0.0)
        return {
            "x": round(cx, 4),
            "y": round(cy, 4),
            "diameter": round(diam, 4),
        }

    elif kind == "slot":
        length = text_val if text_val and text_val > 0 else g.get("length", 0.0)
        width  = g.get("width", 0.0)
        return {
            "x": round(cx, 4),
            "y": round(cy, 4),
            "length": round(length, 4),
            "width":  round(width, 4),
        }

    elif kind in ("pocket", "contour"):
        w = g.get("width", g.get("length", 0.0))
        h = g.get("height", g.get("width", 0.0))
        return {
            "x": round(cx - w / 2, 4),
            "y": round(cy - h / 2, 4),
            "width":  round(w, 4),
            "height": round(h, 4),
        }

    elif kind == "radius":
        r = text_val if text_val and text_val > 0 else g.get("radius", 0.0)
        return {"x": round(cx, 4), "y": round(cy, 4), "radius": round(r, 4)}

    else:
        return {"x": round(cx, 4), "y": round(cy, 4)}

# ---------------------------------------------------------------------------
# Pattern detection (4× prefix → multiple instances)
# ---------------------------------------------------------------------------

def _expand_quantity_patterns(
    features: list,
    pairs: list,
) -> list:
    """
    Handle '4×Ø8' annotations: if a quantity annotation is linked to a single
    feature, try to clone that feature quantity-1 times using nearest-neighbour
    geometry candidates. Simplified: just annotate the primary feature with
    count and mark source='inferred' for the copies.

    For v0: record the count on the primary feature as manufacturing.count,
    and emit it once. Full pattern expansion is a Phase-2 task.
    """
    qty_pairs: dict = {}
    for pair in pairs:
        if pair.annotation.parsed.token_type == "quantity":
            qty = pair.annotation.parsed.quantity or 1
            cid = pair.candidate.candidate.id
            qty_pairs[cid] = qty

    # Annotate primary features with count
    for feat in features:
        cid = feat.get("_candidate_id")
        if cid and cid in qty_pairs:
            feat.setdefault("manufacturing", {})["count"] = qty_pairs[cid]
            feat["source"] = "inferred"

    return features

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_features(
    kept_candidates: list,
    pairs: list,
    title_block: Optional[TitleBlockInfo] = None,
    page_area_mm2: float = 0.0,
) -> list:
    """
    Convert kept ScaledCandidates + LinkedPairs into Module 1 V2 Feature dicts.

    Args:
        kept_candidates: List[ScaledCandidate] after conflict resolution.
        pairs:           List[LinkedPair] from spatial_link.
        title_block:     Optional title block info (units, tolerances).
        page_area_mm2:   Total drawing area in mm² (used for contour detection).

    Returns:
        List of Module 1 V2 Feature dicts, each with:
            id, kind, x, y, [diameter|length|width|height|radius],
            confidence, source, manufacturing
    """
    if not kept_candidates:
        return []

    # Build lookup: candidate_id → list of LinkedPairs
    pairs_by_cand: dict = {}
    for pair in pairs:
        cid = pair.candidate.candidate.id
        pairs_by_cand.setdefault(cid, []).append(pair)

    # Find the largest candidate (likely outer contour)
    def area_of(sc: ScaledCandidate) -> float:
        return _feature_area_mm2(sc)

    sorted_by_area = sorted(kept_candidates, key=area_of, reverse=True)
    largest_id = sorted_by_area[0].candidate.id if sorted_by_area else None

    # Total drawing area — use if provided, else guess from largest
    total_area = page_area_mm2 if page_area_mm2 > 0 else area_of(sorted_by_area[0]) if sorted_by_area else 1.0

    features: list = []
    feature_counter: int = 0

    for sc in kept_candidates:
        cid = sc.candidate.id
        cand_pairs = pairs_by_cand.get(cid, [])
        is_largest = (cid == largest_id)
        kind = sc.candidate.kind
        conf = sc.candidate.confidence

        # ── Infer feature kind
        if kind == "circle":
            f_kind = _infer_circle_kind(sc, cand_pairs, is_largest)
            source = "vision"

        elif kind == "slot":
            f_kind = "slot"
            source = "vision"

        elif kind == "rectangle":
            f_kind = _infer_rect_kind(sc, cand_pairs, is_largest)
            source = "vision"

        elif kind == "polygon":
            f_kind = _infer_polygon_kind(sc, cand_pairs, is_largest)
            source = "vision"

        elif kind == "line":
            f_kind = "contour"
            source = "vision"
            conf = min(conf, 0.50)   # lines in drawings are usually dimension lines

        elif kind in ("arc",):
            f_kind = "radius"
            source = "vision"

        else:
            f_kind = "contour"
            source = "vision"
            conf = min(conf, 0.40)

        # Boost or reduce confidence based on text evidence
        if cand_pairs:
            conf = min(1.0, conf * _CONF_TEXT_CORROBORATED)
        else:
            conf = conf * _CONF_GEOMETRY_ONLY

        # Combine geometry confidence with OCR confidence of associated text
        if cand_pairs:
            ocr_confs = [p.annotation.ocr_confidence for p in cand_pairs]
            avg_ocr = sum(ocr_confs) / len(ocr_confs)
            conf = round(conf * avg_ocr, 4)

        # Build geometry sub-dict
        geometry_dict = _build_geometry_for_kind(f_kind, sc, cand_pairs)

        # Build manufacturing dict
        mfg = _make_manufacturing(cand_pairs)

        # Apply global tolerance from title block if no per-feature tolerance
        if title_block and title_block.tolerance_general:
            if "tolerancePlus" not in mfg:
                # Try to parse the tolerance string (e.g. "±0.1")
                import re
                m = re.search(r"(\d+(?:\.\d+)?)", title_block.tolerance_general)
                if m:
                    val = float(m.group(1))
                    mfg.setdefault("tolerancePlus", val)
                    mfg.setdefault("toleranceMinus", val)

        feature_counter += 1
        feat: dict = {
            "id":           f"f{feature_counter:04d}",
            "kind":         f_kind,
            "confidence":   round(conf, 4),
            "source":       source,
            "manufacturing": mfg,
            "_candidate_id": cid,   # internal; stripped before JSON handoff
        }
        feat.update(geometry_dict)

        # Thread-specific fields
        for pair in cand_pairs:
            p = pair.annotation.parsed
            if p.token_type == "thread_callout" and p.value:
                feat["diameter"] = feat.get("diameter", float(p.value))
                if p.thread_pitch:
                    mfg["threadPitch"] = p.thread_pitch

        features.append(feat)

    # Pattern expansion
    features = _expand_quantity_patterns(features, pairs)

    # Strip internal fields
    for feat in features:
        feat.pop("_candidate_id", None)

    return features

# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


# ── assemble.py ───────────────────────────────────────────────

"""
stage5_assemble/assemble.py — Module 2: Stage 5
================================================
Orchestrates the full assembly pipeline for one page:
    1. spatial_link  — pair text annotations to geometry candidates
    2. conflict_resolver — remove overlapping duplicate candidates
    3. feature_inferrer — map (candidate + text) → Module 1 V2 Feature

Produces the PartData dict (Module 1 V2 schema) plus diagnostics.

Public API:
    assemble(annotations, scaled_candidates, title_block, page_info)
        -> (part_data: dict, cross_link_diag: dict, suppressed: list)
"""

from pathlib import Path



# ---------------------------------------------------------------------------
# Module 1 V2 PartData builder
# ---------------------------------------------------------------------------

def _build_part_data(
    features: list,
    title_block: Optional[TitleBlockInfo],
    page_info: dict,
) -> dict:
    """
    Assemble a Module 1 V2 PartData dict from inferred features.

    Schema reference: module1/src/types.ts (PartData, Feature, ManufacturingMeta).
    """
    # Part name — prefer title block, fall back to source filename
    part_name = (
        (title_block.part_name if title_block else None)
        or page_info.get("source_filename", "unknown")
    )

    # Units — prefer title block hint, fall back to mm
    units = "mm"
    if title_block and title_block.units_hint:
        units = title_block.units_hint

    # Compute overall bounding box from all feature positions
    all_x = [f.get("x", 0.0) for f in features]
    all_y = [f.get("y", 0.0) for f in features]
    all_x2 = [
        f.get("x", 0.0) + f.get("width", f.get("diameter", 0.0))
        for f in features
    ]
    all_y2 = [
        f.get("y", 0.0) + f.get("height", f.get("diameter", 0.0))
        for f in features
    ]

    if features:
        bbox = {
            "x":      round(min(all_x),  4),
            "y":      round(min(all_y),  4),
            "width":  round(max(all_x2) - min(all_x), 4),
            "height": round(max(all_y2) - min(all_y), 4),
        }
    else:
        bbox = {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}

    # Map features to Module 1 V2 Feature shape
    m1_features = []
    for feat in features:
        m1_feat: dict = {
            "id":           feat["id"],
            "kind":         feat["kind"],
            "confidence":   feat["confidence"],
            "source":       feat.get("source", "vision"),
        }

        # Geometry fields — copy all except meta fields
        _skip = {"id", "kind", "confidence", "source", "manufacturing"}
        for k, v in feat.items():
            if k not in _skip:
                m1_feat[k] = v

        # Manufacturing meta
        mfg = feat.get("manufacturing", {})
        if mfg:
            m1_feat["manufacturing"] = mfg

        m1_features.append(m1_feat)

    part_data = {
        "schemaVersion": 2,
        "partName":      part_name,
        "units":         units,
        "bbox":          bbox,
        "zeroPoint":     {"x": 0.0, "y": 0.0},
        "features":      m1_features,
        "_meta": {
            "inputType":      page_info.get("input_type", "unknown"),
            "generatedBy":    "module2_v1.0.0",
            "drawingNumber":  title_block.drawing_number if title_block else None,
            "revision":       title_block.revision       if title_block else None,
            "material":       title_block.material       if title_block else None,
            "scaleRaw":       title_block.scale_raw      if title_block else None,
        },
    }

    return part_data

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assemble(
    annotations: list,
    scaled_candidates: list,
    title_block: Optional[TitleBlockInfo] = None,
    page_info: Optional[dict] = None,
    page_area_mm2: float = 0.0,
) -> tuple:
    """
    Full Stage 5 assembly: text ↔ geometry → PartData.

    Args:
        annotations:       List[TextAnnotation] from Stage 2.
        scaled_candidates: List[ScaledCandidate] from Stage 4.
        title_block:       Optional TitleBlockInfo from Stage 2.5.
        page_info:         Dict with source_filename, input_type.
        page_area_mm2:     Drawing page area in mm² (for contour detection).

    Returns:
        (part_data: dict, cross_link_diag: dict, suppressed: List[SuppressedCandidate])
    """
    t0 = time.monotonic()
    page_info = page_info or {}

    # ── Step 1: Spatial link — pair text to geometry
    pairs, link_suppressed = link_text_to_geometry(annotations, scaled_candidates)

    # ── Step 2: Conflict resolution — remove duplicate candidates
    kept_candidates, conflict_suppressed = resolve_conflicts(scaled_candidates, pairs)

    # Remove pairs whose candidate was suppressed
    kept_ids = {sc.candidate.id for sc in kept_candidates}
    pairs = [p for p in pairs if p.candidate.candidate.id in kept_ids]

    # Combine suppressed lists
    all_suppressed: list = link_suppressed + conflict_suppressed

    # ── Step 3: Feature inference — map to Module 1 V2 features
    features = infer_features(
        kept_candidates, pairs, title_block, page_area_mm2
    )

    # ── Step 4: Build PartData dict
    part_data = _build_part_data(features, title_block, page_info)

    elapsed = round((time.monotonic() - t0) * 1000, 1)

    # Cross-link diagnostics
    cross_link_diag = {
        "text_geometry_pairs":    len(pairs),
        "suppressed_by_linker":   len(link_suppressed),
        "suppressed_by_conflict": len(conflict_suppressed),
        "features_assembled":     len(features),
        "elapsed_ms":             elapsed,
    }

    return part_data, cross_link_diag, all_suppressed

# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


# ── report.py ─────────────────────────────────────────────────

"""
stage6_diagnostics/report.py — Module 2: Stage 6
==================================================
Builds the RecognitionReport — the truthfulness layer.

Rule (architecture §7 + master rule TRUTH RULE):
    overallConfidence = MIN across critical signals, NOT average.
    A single missing scale anchor pulls everything down.
    Low OCR confidence is reported, not hidden.

Severity levels for weak signals (from old 2A.3/review.py, REUSE bucket):
    high   — recognition likely wrong; human must verify
    medium — uncertain; review recommended
    low    — informational

Public API:
    build_report(pipeline, stage_timings, ingest_diag, preproc_diag,
                 ocr_diag, tb_diag, geometry_diag, cross_link_diag,
                 suppressed, scale_info) -> dict
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Weak signal rules (R01–R12)
# Adapted from old 2A.3/review.py (REUSE / ADAPT bucket)
# ---------------------------------------------------------------------------

def _collect_weak_signals(
    ocr_diag:         dict,
    geometry_diag:    dict,
    cross_link_diag:  dict,
    scale_info:       Optional[dict],
    suppressed:       list,
    preproc_diag:     dict,
) -> list:
    """
    Produce a list of human-readable warning strings.
    Each entry may be prefixed with [HIGH], [MEDIUM], or [LOW].
    """
    signals: list = []

    # R01: No scale anchor
    if scale_info and scale_info.get("anchor_method") == "unknown":
        signals.append(
            "[HIGH] No scale anchor found — all coordinates are in pixel units. "
            "Set a dimension anchor or provide DPI for mm output."
        )

    # R02: Very low OCR confidence
    avg_ocr = ocr_diag.get("avg_confidence", 1.0)
    if avg_ocr < 0.50 and ocr_diag.get("regions_found", 0) > 0:
        signals.append(
            f"[HIGH] OCR average confidence {avg_ocr:.2f} is very low. "
            "Drawing text may be garbled. Check image quality."
        )
    elif avg_ocr < 0.70 and ocr_diag.get("regions_found", 0) > 0:
        signals.append(
            f"[MEDIUM] OCR average confidence {avg_ocr:.2f}. "
            "Some dimension values may be misread."
        )

    # R03: No OCR text found at all
    if ocr_diag.get("regions_found", 0) == 0:
        signals.append(
            "[MEDIUM] No text found by OCR. Geometry-only output — "
            "no dimensions or material info extracted."
        )

    # R04: High garbled rate
    garbled = ocr_diag.get("garbled_count", 0)
    found   = ocr_diag.get("regions_found", 0)
    if found > 0 and garbled / found > 0.30:
        signals.append(
            f"[MEDIUM] {garbled}/{found} OCR tokens were unrecognised. "
            "Many symbols may have been misread."
        )

    # R05: No geometry candidates at all
    n_cands = geometry_diag.get("total_candidates", 0)
    if n_cands == 0:
        signals.append(
            "[HIGH] No geometry candidates detected. "
            "Image may be blank, inverted, or too low resolution."
        )

    # R06: Many suppressed candidates (suggests conflict or noise)
    n_supp = len(suppressed)
    if n_supp > 5:
        signals.append(
            f"[MEDIUM] {n_supp} geometry candidates were suppressed during "
            "conflict resolution. Drawing may have overlapping annotations."
        )

    # R07: No features assembled
    n_features = cross_link_diag.get("features_assembled", 0)
    if n_features == 0:
        signals.append(
            "[HIGH] No features were assembled. "
            "Check geometry and OCR stages — this part will be empty."
        )

    # R08: Deskew applied at large angle (photo quality concern)
    deskew_angle = abs(preproc_diag.get("deskew_angle_deg", 0.0))
    if deskew_angle > 8.0:
        signals.append(
            f"[MEDIUM] Drawing was deskewed by {deskew_angle:.1f}°. "
            "Large corrections may introduce distortion — verify output."
        )

    # R09: Low-confidence scale anchor
    if scale_info:
        scale_conf = scale_info.get("anchor_confidence", 1.0)
        if 0.0 < scale_conf < 0.60:
            signals.append(
                f"[MEDIUM] Scale anchor confidence {scale_conf:.2f} is low. "
                f"Method: {scale_info.get('anchor_method')}. Dimensions may be inaccurate."
            )

    # R10: Many ambiguous text-geometry pairs
    n_pairs = cross_link_diag.get("text_geometry_pairs", 0)
    if n_pairs > 0:
        # We don't have ambiguous count directly — if suppressed_by_linker is high
        n_link_supp = cross_link_diag.get("suppressed_by_linker", 0)
        if n_link_supp > n_pairs * 0.4:
            signals.append(
                f"[LOW] {n_link_supp} text annotations could not be confidently "
                "linked to geometry. They appear in the suppressed list."
            )

    # R11: OCR engine absent
    if ocr_diag.get("engine") == "none":
        signals.append(
            "[HIGH] OCR engine (Tesseract) is not installed. "
            "No text was extracted. Install tesseract-ocr to enable dimension parsing."
        )

    return signals

# ---------------------------------------------------------------------------
# Overall confidence (MIN, not AVG)
# ---------------------------------------------------------------------------

def _compute_overall_confidence(
    ocr_diag:       dict,
    geometry_diag:  dict,
    scale_info:     Optional[dict],
    n_features:     int,
) -> float:
    """
    Overall confidence = MIN across critical signals.
    Any single weak signal pulls the whole reading down.
    """
    signals: list = []

    # OCR quality
    if ocr_diag.get("regions_found", 0) > 0:
        signals.append(ocr_diag.get("avg_confidence", 0.5))
    else:
        signals.append(0.40)   # no text is a weak signal

    # Geometry quality
    avg_geom = geometry_diag.get("avg_confidence", 0.0)
    if avg_geom > 0:
        signals.append(avg_geom)
    elif geometry_diag.get("total_candidates", 0) == 0:
        signals.append(0.0)

    # Scale anchor quality
    if scale_info:
        if scale_info.get("anchor_method") == "unknown":
            signals.append(0.30)
        else:
            signals.append(scale_info.get("anchor_confidence", 0.5))

    # Feature assembly quality
    if n_features == 0:
        signals.append(0.0)

    if not signals:
        return 0.0

    return round(min(signals), 4)

# ---------------------------------------------------------------------------
# Benchmark recommendation heuristic
# ---------------------------------------------------------------------------

def _should_recommend_benchmark(
    overall_confidence: float,
    pipeline:           str,
    n_features:         int,
) -> bool:
    """
    Return True if this drawing should be added to the benchmark set.
    Logic: draws that are surprising (low conf on a good drawing, or
    high conf on a photo) are valuable for improving the system.
    """
    if pipeline == "photo" and overall_confidence > 0.85 and n_features >= 3:
        return True   # high-quality photo → add to positive benchmark
    if overall_confidence < 0.50 and n_features == 0:
        return True   # total failure → add to challenge set
    return False

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_report(
    pipeline:          str,
    stage_timings:     dict,
    ingest_diag:       dict,
    preproc_diag:      dict,
    ocr_diag:          dict,
    tb_diag:           dict,
    geometry_diag:     dict,
    cross_link_diag:   dict,
    suppressed:        list,
    scale_info:        Optional[dict] = None,
) -> dict:
    """
    Build the RecognitionReport dict (always serialisable to JSON).

    Args:
        pipeline:         "pdf" | "photo".
        stage_timings:    Dict[stage_name, elapsed_ms].
        *_diag:           Per-stage diagnostic dicts.
        suppressed:       List of SuppressedCandidate.
        scale_info:       ScaleInfo as dict (or None).

    Returns:
        RecognitionReport as a plain dict (JSON-serialisable).
    """
    n_features = cross_link_diag.get("features_assembled", 0)

    weak_signals = _collect_weak_signals(
        ocr_diag, geometry_diag, cross_link_diag,
        scale_info, suppressed, preproc_diag,
    )

    overall_conf = _compute_overall_confidence(
        ocr_diag, geometry_diag, scale_info, n_features
    )

    recommend_bm = _should_recommend_benchmark(overall_conf, pipeline, n_features)

    # Serialise suppressed candidates
    suppressed_serial = [
        {
            "candidate_id":   s.candidate.candidate.id,
            "candidate_kind": s.candidate.candidate.kind,
            "reason":         s.reason,
        }
        for s in suppressed
        if isinstance(s, SuppressedCandidate)
    ]

    return {
        "pipeline":              pipeline,
        "stage_timings_ms":      stage_timings,
        "ingest":                ingest_diag,
        "preproc":               preproc_diag,
        "ocr":                   ocr_diag,
        "title_block":           tb_diag,
        "geometry":              geometry_diag,
        "cross_link":            cross_link_diag,
        "suppressed":            suppressed_serial,
        "overall_confidence":    overall_conf,
        "weak_signals":          weak_signals,
        "recommend_benchmark":   recommend_bm,
    }

# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


# ── main.py ───────────────────────────────────────────────────

"""
main.py — Module 2: Drawing Recognition Engine
===============================================
Full pipeline CLI entry point. Runs all six stages and writes output JSON.

Usage:
    python main.py drawing.pdf
    python main.py photo.jpg --output result.json
    python main.py drawing.pdf --dpi 400 --quiet
    python main.py drawing.pdf --selftest

Pipeline:
    Stage 0  Ingest
    Stage 1  Preprocess (PDF or Photo pipeline)
    Stage 2  OCR + Symbol Parser
    Stage 2.5 Title Block
    Stage 3  Geometry Detection
    Stage 4  Scale / Coordinate Transform
    Stage 5  Cross-link + Assembly → PartData
    Stage 6  Diagnostics → RecognitionReport

Output JSON shape:
    {
        "partData":    { ...Module 1 V2 PartData... },
        "diagnostics": { ...RecognitionReport... }
    }

Exit codes:
    0  Success
    1  Input file not found or unreadable
    2  Unsupported format
    3  Argument error (DPI out of range, etc.)
    4  Pipeline error (unexpected exception)
"""

from pathlib import Path

# Make module-level imports work regardless of working directory
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Stage imports

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DPI        = 300
MIN_DPI            = 72
MAX_DPI            = 600
DEFAULT_OUTPUT     = "module2_output.json"
MODULE_VERSION     = "1.0.0"

EXIT_OK      = 0
EXIT_INPUT   = 1
EXIT_FORMAT  = 2
EXIT_ARGS    = 3
EXIT_PIPELINE = 4

# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: Path,
    *,
    dpi: int = DEFAULT_DPI,
    quiet: bool = False,
) -> dict:
    """
    Execute the full Module 2 pipeline on one drawing file.

    Args:
        input_path: Resolved path to the drawing.
        dpi:        Render DPI for PDFs.
        quiet:      Suppress progress output.

    Returns:
        dict with keys "partData" and "diagnostics".

    Raises:
        FileNotFoundError: input file does not exist.
        ValueError:        unsupported format or bad arguments.
        RuntimeError:      pipeline failure.
    """
    timings: dict = {}

    def log(msg: str) -> None:
        if not quiet:
            print(f"  [M2] {msg}")

    log(f"Processing: {input_path.name}")

    # ── Stage 0: Ingest
    t0 = time.monotonic()
    raw_input = ingest(input_path, dpi=dpi)
    timings["stage0_ingest"] = round((time.monotonic() - t0) * 1000, 1)
    log(f"Stage 0 done: {raw_input.input_type}, {raw_input.page_count} page(s)")

    # For now: process page 1 only (multi-page deferred per architecture §8)
    if not raw_input.pages:
        raise RuntimeError("Ingest produced no pages.")
    page = raw_input.pages[0]

    # ── Stage 1: Preprocess
    t1 = time.monotonic()
    if raw_input.input_type == "pdf":
        preprocessed = preprocess_pdf(page)
    else:
        preprocessed = preprocess_photo(page)
    timings["stage1_preprocess"] = round((time.monotonic() - t1) * 1000, 1)
    log(f"Stage 1 done: pipeline={preprocessed.pipeline}, "
        f"deskewed={preprocessed.deskewed} ({preprocessed.deskew_angle_deg:.1f}°)")

    preproc_diag = {
        "pipeline":          preprocessed.pipeline,
        "deskewed":          preprocessed.deskewed,
        "deskew_angle_deg":  preprocessed.deskew_angle_deg,
        "deskew_variance":   preprocessed.deskew_variance,
        "threshold_method":  preprocessed.threshold_method,
        "denoise_applied":   preprocessed.denoise_applied,
        "crop_bbox":         preprocessed.crop_bbox,
        "width_px":          preprocessed.width_px,
        "height_px":         preprocessed.height_px,
    }

    # ── Stage 2: OCR + Symbol Parser
    t2 = time.monotonic()
    annotations, text_mask, ocr_diag = run_stage2(
        preprocessed,
        raw_input.pdf_text_layer,
    )
    timings["stage2_ocr"] = round((time.monotonic() - t2) * 1000, 1)
    log(f"Stage 2 done: {len(annotations)} annotations, "
        f"engine={ocr_diag.get('engine')}, "
        f"avg_conf={ocr_diag.get('avg_confidence', 0):.2f}")

    # ── Stage 2.5: Title Block
    t25 = time.monotonic()
    title_block = parse_title_block(
        annotations,
        preprocessed.width_px,
        preprocessed.height_px,
    )
    timings["stage25_titleblock"] = round((time.monotonic() - t25) * 1000, 1)
    tb_diag = {
        "part_name":      title_block.part_name,
        "material":       title_block.material,
        "scale_raw":      title_block.scale_raw,
        "units_hint":     title_block.units_hint,
        "drawing_number": title_block.drawing_number,
    }
    log(f"Stage 2.5 done: part={title_block.part_name!r}, "
        f"scale={title_block.scale_raw!r}")

    # ── Stage 3: Geometry Detection (on text-masked image)
    t3 = time.monotonic()
    masked_image = apply_text_mask(preprocessed.image_array, text_mask)
    candidates = detect(masked_image, page=preprocessed.page_number)
    timings["stage3_geometry"] = round((time.monotonic() - t3) * 1000, 1)

    avg_geom_conf = (
        round(sum(c.confidence for c in candidates) / len(candidates), 4)
        if candidates else 0.0
    )
    geometry_diag = {
        "total_candidates":     len(candidates),
        "avg_confidence":       avg_geom_conf,
        "kinds":                _count_kinds(candidates),
        "text_mask_regions":    len(text_mask.regions),
    }
    log(f"Stage 3 done: {len(candidates)} candidates, "
        f"kinds={geometry_diag['kinds']}")

    # ── Stage 4: Scale Detection + Apply
    t4 = time.monotonic()
    scale_info = detect_scale(
        annotations,
        candidates,
        preprocessed.width_px,
        preprocessed.height_px,
        title_block=title_block,
        pdf_dpi=page.dpi,
    )
    scaled_candidates = apply_scale(candidates, scale_info)
    timings["stage4_scale"] = round((time.monotonic() - t4) * 1000, 1)
    scale_diag = dataclasses.asdict(scale_info)
    log(f"Stage 4 done: method={scale_info.anchor_method}, "
        f"px_per_mm={scale_info.px_per_mm:.4f}, "
        f"conf={scale_info.anchor_confidence:.2f}")

    # ── Stage 5: Cross-link + Assemble → PartData
    t5 = time.monotonic()
    page_area_mm2 = 0.0
    if scale_info.px_per_mm > 0:
        page_area_mm2 = (preprocessed.width_px / scale_info.px_per_mm) * \
                        (preprocessed.height_px / scale_info.px_per_mm)

    part_data, cross_link_diag, suppressed = assemble(
        annotations,
        scaled_candidates,
        title_block=title_block,
        page_info={
            "source_filename": input_path.name,
            "input_type":      raw_input.input_type,
        },
        page_area_mm2=page_area_mm2,
    )
    timings["stage5_assemble"] = round((time.monotonic() - t5) * 1000, 1)
    log(f"Stage 5 done: {len(part_data['features'])} features assembled")

    # ── Stage 6: Diagnostics
    t6 = time.monotonic()
    ingest_diag = {
        "input_type":             raw_input.input_type,
        "page_count":             raw_input.page_count,
        "orig_dpi":               raw_input.orig_dpi,
        "exif_rotation_applied":  raw_input.exif_rotation_applied,
        "pdf_text_runs":          len(raw_input.pdf_text_layer),
        "source_path":            raw_input.source_path,
    }
    report = build_report(
        pipeline=raw_input.input_type,
        stage_timings=timings,
        ingest_diag=ingest_diag,
        preproc_diag=preproc_diag,
        ocr_diag=ocr_diag,
        tb_diag=tb_diag,
        geometry_diag=geometry_diag,
        cross_link_diag=cross_link_diag,
        suppressed=suppressed,
        scale_info=scale_diag,
    )
    timings["stage6_diagnostics"] = round((time.monotonic() - t6) * 1000, 1)

    total_ms = round(sum(timings.values()), 1)
    log(f"Stage 6 done. Overall confidence: {report['overall_confidence']:.2f}")
    if report["weak_signals"] and not quiet:
        for sig in report["weak_signals"]:
            print(f"  [WARN] {sig}")
    log(f"Total elapsed: {total_ms} ms")

    return {
        "partData":    part_data,
        "diagnostics": report,
    }

def _count_kinds(candidates: list) -> dict:
    counts: dict = {}
    for c in candidates:
        counts[c.kind] = counts.get(c.kind, 0) + 1
    return counts

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Module 2 — CNC Drawing Recognition Engine\n"
            "------------------------------------------\n"
            "Reads a technical drawing (PDF or image) and extracts\n"
            "geometry, dimensions, and text into Module 1 V2 PartData.\n"
        ),
        epilog=(
            "Examples:\n"
            "  python main.py drawing.pdf\n"
            "  python main.py photo.jpg --output result.json\n"
            "  python main.py drawing.pdf --dpi 400 --quiet\n"
        ),
    )
    p.add_argument("input",  metavar="INPUT",
                   help="Path to drawing file (PDF, PNG, JPG, TIF)")
    p.add_argument("--output", "-o", metavar="PATH", default=None,
                   help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--dpi", "-d", type=int, default=DEFAULT_DPI, metavar="N",
                   help=f"PDF render DPI (default {DEFAULT_DPI}, range {MIN_DPI}–{MAX_DPI})")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress progress output; only print errors")
    p.add_argument("--selftest", action="store_true",
                   help=argparse.SUPPRESS)
    return p

def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if "--selftest" in argv:
        return _run_pipeline_selftest()

    parser = _build_parser()
    args   = parser.parse_args(argv)

    # Validate DPI
    if not (MIN_DPI <= args.dpi <= MAX_DPI):
        print(f"ERROR: --dpi must be {MIN_DPI}–{MAX_DPI}, got {args.dpi}",
              file=sys.stderr)
        return EXIT_ARGS

    # Resolve input path
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        return EXIT_INPUT

    # Resolve output path
    out_path = Path(args.output).resolve() if args.output else \
               input_path.parent / DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    try:
        result = run_pipeline(input_path, dpi=args.dpi, quiet=args.quiet)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_FORMAT
    except Exception as exc:
        print(f"ERROR: Pipeline failed: {exc}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return EXIT_PIPELINE

    # Write output
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
    except OSError as exc:
        print(f"ERROR: Cannot write output: {exc}", file=sys.stderr)
        return EXIT_PIPELINE

    if not args.quiet:
        n_feat = len(result["partData"].get("features", []))
        conf   = result["diagnostics"].get("overall_confidence", 0.0)
        print(f"  [M2] Output: {out_path}")
        print(f"  [M2] Features: {n_feat}  |  Confidence: {conf:.2f}")

    return EXIT_OK

# ---------------------------------------------------------------------------
# Pipeline self-test (no real drawing needed)
# ---------------------------------------------------------------------------
