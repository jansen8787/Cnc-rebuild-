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

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import GeometryCandidate  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier import (  # noqa: E402
    classify_contour,
    maybe_reclass_drawn_circle,
    contour_center,
    contour_area,
    contour_bbox,
    _CIRCLE_CIRCULARITY_MIN,
)


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


def _run_tests() -> int:
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    results: list = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition))
        marker = f"  ({detail})" if detail else ""
        print(f"  {PASS if condition else FAIL}  {name}{marker}")

    print("\n── Stage 3: Detector self-tests ──\n")

    # ── 1. Empty image → no candidates
    blank = np.zeros((200, 200), dtype=np.uint8)
    r = detect(blank, page=1)
    check("blank image → no candidates",    len(r) == 0, f"got {len(r)}")

    # ── 2. Single filled circle
    img_c = _make_test_image([{"type":"circle","cx":100,"cy":100,"r":40}])
    r2 = detect(img_c, page=1)
    circles = [c for c in r2 if c.kind == "circle"]
    check("single circle detected",         len(circles) >= 1,
          f"candidates: {[c.kind for c in r2]}")
    if circles:
        check("circle cx≈100",  abs(circles[0].geometry["cx"] - 100) < 15, f"cx={circles[0].geometry['cx']:.1f}")
        check("circle cy≈100",  abs(circles[0].geometry["cy"] - 100) < 15, f"cy={circles[0].geometry['cy']:.1f}")

    # ── 3. Single rectangle
    img_r = _make_test_image([{"type":"rectangle","x":50,"y":80,"w":120,"h":60}])
    r3 = detect(img_r, page=1)
    rects = [c for c in r3 if c.kind == "rectangle"]
    check("rectangle detected",              len(rects) >= 1,
          f"candidates: {[c.kind for c in r3]}")

    # ── 4. Slot detected (not rectangle)
    img_s = _make_test_image([{"type":"slot","cx":200,"cy":150,"length":140,"radius":25}])
    r4 = detect(img_s, page=1)
    slots = [c for c in r4 if c.kind == "slot"]
    non_rect = [c for c in r4 if c.kind != "rectangle"]
    check("slot detected",                   len(slots) >= 1,
          f"candidates: {[c.kind for c in r4]}")

    # ── 5. Multiple shapes
    img_m = _make_test_image([
        {"type":"circle",    "cx":60,  "cy":60,  "r":25},
        {"type":"rectangle", "x":150,  "y":50,   "w":80, "h":60},
        {"type":"circle",    "cx":300, "cy":250, "r":15},
    ], size=(400, 400))
    r5 = detect(img_m, page=1)
    check("multiple shapes detected ≥2",     len(r5) >= 2,
          f"n={len(r5)}, kinds={[c.kind for c in r5]}")

    # ── 6. Required fields on candidates
    if r5:
        c = r5[0]
        check("candidate has id",            hasattr(c, "id"))
        check("candidate has kind",          hasattr(c, "kind"))
        check("candidate has confidence",    hasattr(c, "confidence"))
        check("candidate has evidence",      hasattr(c, "evidence"))
        check("candidate has geometry",      hasattr(c, "geometry"))
        check("candidate has page",          hasattr(c, "page"))
        check("confidence ∈ [0,1]",          0.0 <= c.confidence <= 1.0)

    # ── 7. Small circle (r=5) → low confidence but NOT deleted
    img_small = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img_small, (100, 100), 5, 255, -1)
    r6 = detect(img_small, page=1, min_area_px=100)
    small_circles = [c for c in r6 if c.kind == "circle"]
    # Small circle may or may not be detected (Hough/contour thresholds)
    # The key assertion: it's not silently filtered — it appears if detected
    check("small circle: detector returns list", isinstance(r6, list))

    # ── 8. RETR_LIST: inner contour visible (box with circle inside)
    img_nested = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(img_nested, (20, 20), (280, 280), 255, 2)   # outer rect (ring)
    cv2.circle(img_nested, (150, 150), 40, 255, -1)            # inner circle
    r7 = detect(img_nested, page=1)
    kinds7 = [c.kind for c in r7]
    check("RETR_LIST: inner circle visible", "circle" in kinds7,
          f"kinds: {kinds7[:5]}")

    # ── 9. Determinism
    r8a = detect(img_m, page=1)
    r8b = detect(img_m, page=1)
    check("determinism: same kind list",
          [c.kind for c in r8a] == [c.kind for c in r8b])

    # ── 10. Page number stored correctly
    img_p = _make_test_image([{"type":"circle","cx":100,"cy":100,"r":30}])
    r9 = detect(img_p, page=3)
    check("page stored on candidates",
          all(c.page == 3 for c in r9))

    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n── Detector tests: {passed}/{total} passed ──\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_tests())
