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

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import (  # noqa: E402
    TextAnnotation,
    GeometryCandidate,
    ScaleInfo,
    ScaledCandidate,
    TitleBlockInfo,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage25_titleblock"))
from titleblock import extract_scale  # noqa: E402


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

def _run_tests() -> int:
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    results: list = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition))
        marker = f"  ({detail})" if detail else ""
        print(f"  {PASS if condition else FAIL}  {name}{marker}")

    print("\n── Stage 4: Scale Detector self-tests ──\n")

    from m2types import ParsedSymbol

    # ── 1. Operator override
    scale = detect_scale([], [], 2480, 3508,
                         operator_override_mm_per_px=0.1)
    check("operator override → method",  scale.anchor_method == "operator_override")
    check("operator override → conf",    scale.anchor_confidence == 1.0)
    check("operator override → px/mm",   abs(scale.px_per_mm - 10.0) < 0.01,
          f"px_per_mm={scale.px_per_mm}")

    # ── 2. px_to_mm with known scale
    scale10 = ScaleInfo(1, 10.0, "test", 0.9, {"x": 0.0, "y": 0.0})
    check("px_to_mm: 100px at 10px/mm → 10mm",
          abs(px_to_mm(100.0, scale10) - 10.0) < 0.01,
          f"{px_to_mm(100.0, scale10)}")

    # ── 3. px_to_mm with unknown scale → returns px unchanged
    unknown_scale = ScaleInfo(1, 0.0, "unknown", 0.0, {"x": 0.0, "y": 0.0})
    check("unknown scale → px_to_mm unchanged",
          abs(px_to_mm(50.0, unknown_scale) - 50.0) < 0.01)

    # ── 4. point_px_to_mm with origin offset
    scale5 = ScaleInfo(1, 5.0, "test", 0.9, {"x": 10.0, "y": 20.0})
    xmm, ymm = point_px_to_mm(60.0, 70.0, scale5)
    # (60-10)/5 = 10, (70-20)/5 = 10
    check("point_px_to_mm with offset",
          abs(xmm - 10.0) < 0.01 and abs(ymm - 10.0) < 0.01,
          f"({xmm}, {ymm})")

    # ── 5. PDF scale from DPI (A4 at 300 DPI)
    # A4: 210×297mm → at 300dpi: 2480×3508px (approx)
    scale_a4 = detect_scale([], [], 2480, 3508, pdf_dpi=300.0)
    check("A4 at 300dpi detected",
          scale_a4.anchor_method.startswith("sheet_size") or
          scale_a4.anchor_method == "pdf_metadata",
          f"method={scale_a4.anchor_method}")
    check("A4 mm/px ≈ 0.0847",
          abs(1.0/scale_a4.px_per_mm - 25.4/300.0) < 0.002
          if scale_a4.px_per_mm > 0 else True,
          f"px_per_mm={scale_a4.px_per_mm:.4f}")

    # ── 6. Unknown scale → px_per_mm == 0
    scale_unk = detect_scale([], [], 1000, 800)
    check("no anchor → unknown method",  scale_unk.anchor_method == "unknown")
    check("no anchor → px_per_mm == 0", scale_unk.px_per_mm == 0.0)
    check("no anchor → conf == 0.0",    scale_unk.anchor_confidence == 0.0)

    # ── 7. apply_scale: circle candidate
    cand = GeometryCandidate(
        id="cand_circle_0001", kind="circle",
        geometry={"cx": 100.0, "cy": 200.0, "radius_px": 30.0, "diameter_px": 60.0},
        confidence=0.9, detector="test", page=1,
        evidence={"area_px": 2827.0},
    )
    scaled = apply_scale([cand], scale10)
    check("apply_scale returns list",    len(scaled) == 1)
    sc = scaled[0]
    check("circle cx_mm",   abs(sc.geometry_mm["cx"] - 10.0) < 0.01,
          f"cx_mm={sc.geometry_mm['cx']}")
    check("circle radius",  abs(sc.geometry_mm["radius"] - 3.0) < 0.01,
          f"r_mm={sc.geometry_mm['radius']}")

    # ── 8. apply_scale: confidence penalised when scale unknown
    scaled_unk = apply_scale([cand], unknown_scale)
    check("unknown scale penalises confidence",
          scaled_unk[0].candidate.confidence < 0.9,
          f"conf={scaled_unk[0].candidate.confidence}")

    # ── 9. _mm_per_px_from_dpi
    result = _mm_per_px_from_dpi(300.0)
    check("300 dpi → mm/px ≈ 0.0847",
          abs(result - 25.4/300.0) < 0.001, f"{result:.5f}")
    check("0 dpi → None", _mm_per_px_from_dpi(0.0) is None)

    # ── 10. _detect_from_sheet A3
    # A3 landscape at 300dpi: 420mm wide × 297mm tall → 4960 × 3508 px
    r_a3 = _detect_from_sheet(4960, 3508, 300.0)
    check("A3 landscape detected",
          r_a3 is not None and "A3" in r_a3[1] if r_a3 else False,
          f"got {r_a3}")

    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n── Scale Detector tests: {passed}/{total} passed ──\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_tests())
