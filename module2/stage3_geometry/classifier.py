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

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


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


def _run_tests() -> int:
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    results: list = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition))
        marker = f"  ({detail})" if detail else ""
        print(f"  {PASS if condition else FAIL}  {name}{marker}")

    print("\n── Stage 3: Classifier self-tests ──\n")

    # Large circle
    c = _make_circle_contour(100, 100, 50)
    kind, conf, ev = classify_contour(c)
    check("circle r=50 → circle",      kind == "circle",  f"got {kind}")
    check("circle r=50 → conf > 0.7",  conf > 0.7,        f"conf={conf:.3f}")

    # Small circle (integer raster)
    c2 = _make_circle_contour(200, 200, 10, n=32)
    kind2, conf2, _ = classify_contour(c2)
    check("circle r=10 small → circle", kind2 == "circle", f"got {kind2}")

    # Large outer circle
    c3 = _make_circle_contour(0, 0, 200)
    kind3, _, _ = classify_contour(c3)
    check("circle r=200 → circle",     kind3 == "circle")

    # Slot (normal)
    s = _make_slot_contour(150, 150, 120, 20)
    kind_s, conf_s, _ = classify_contour(s)
    check("slot 120×40 → slot",        kind_s == "slot",  f"got {kind_s}")
    check("slot conf > 0.5",           conf_s > 0.5,      f"conf={conf_s:.3f}")

    # Slot (narrow)
    s2 = _make_slot_contour(100, 100, 200, 15)
    kind_s2, _, _ = classify_contour(s2)
    check("slot 200×30 narrow → slot", kind_s2 == "slot", f"got {kind_s2}")

    # Rectangle (square)
    r = _make_rect_contour(0, 0, 50, 50)
    kind_r, _, _ = classify_contour(r)
    check("rect 50×50 → rectangle",    kind_r == "rectangle", f"got {kind_r}")

    # Rectangle (wide but not a slot)
    r2 = _make_rect_contour(0, 0, 80, 40)
    kind_r2, _, _ = classify_contour(r2)
    check("rect 80×40 (asp=2<2.5) → rectangle", kind_r2 == "rectangle", f"got {kind_r2}")

    # Hexagon → polygon (NOT circle — fixed threshold)
    hex_c = _make_polygon_contour(100, 100, 60, 6)
    kind_h, _, ev_h = classify_contour(hex_c)
    check("hexagon → polygon (not circle)", kind_h == "polygon", f"got {kind_h}")

    # Octagon → polygon
    oct_c = _make_polygon_contour(100, 100, 60, 8)
    kind_o, _, _ = classify_contour(oct_c)
    check("octagon → polygon",          kind_o == "polygon", f"got {kind_o}")

    # None input → unknown
    kind_n, _, _ = classify_contour(None)
    check("None → unknown",             kind_n == "unknown")

    # Degenerate tiny area → unknown
    tiny = np.array([[[0,0]],[[1,0]],[[0,1]]], dtype=np.int32)
    kind_t, _, _ = classify_contour(tiny)
    check("tiny area → unknown",        kind_t == "unknown")

    # contour_center accuracy
    cc = _make_circle_contour(100, 100, 50)
    cx, cy = contour_center(cc)
    check("center x≈100",               abs(cx - 100) < 2, f"cx={cx:.1f}")
    check("center y≈100",               abs(cy - 100) < 2, f"cy={cy:.1f}")

    # contour_area
    rect = _make_rect_contour(0, 0, 100, 50)
    area = contour_area(rect)
    check("rect 100×50 area≈5000",      abs(area - 5000) < 10, f"area={area:.0f}")

    # contour_bbox
    bb = contour_bbox(_make_rect_contour(10, 20, 80, 60))
    check("bbox x=10",   bb["x"] == 10)
    check("bbox y=20",   bb["y"] == 20)
    check("bbox w≈80",   abs(bb["width"] - 80) <= 1, f"w={bb['width']}")
    check("bbox h≈60",   abs(bb["height"] - 60) <= 1, f"h={bb['height']}")
    check("bbox cx≈50",  abs(bb["cx"] - 50) < 2, f"cx={bb['cx']:.1f}")

    # Slot detected BEFORE rectangle (master rule 19 verification)
    # A wide rectangle with asp~3 must be classified as slot if convex
    slot_narrow = _make_slot_contour(200, 200, 120, 15)
    kn, _, evn = classify_contour(slot_narrow)
    check("narrow slot detected (not rect)", kn == "slot",
          f"got {kn} asp={evn.get('aspect_ratio',0):.2f}")

    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n── Classifier tests: {passed}/{total} passed ──\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_tests())
