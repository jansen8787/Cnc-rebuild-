"""
types.py — Module 2: Drawing Recognition Engine
================================================
All shared data types. No business logic. No external dependencies.

Every stage imports from here. No stage imports types from another stage.
This is the loose-coupling boundary (master rule 15).

Version: 1.0.0
Schema contract: Module 1 V2 (PartData — see module1/types.ts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


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
