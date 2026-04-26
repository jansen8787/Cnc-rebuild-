"""
stage4_scale/scale_detector.py — Module 2: Stage 4 (FIXED)
===========================================================
Robust pixel -> mm scale detection.

Fixes:
- safer dimension-anchor logic
- uses multiple anchors + median
- no fake mm when scale unknown
- cleaner confidence handling
- stable fallback order

Public API unchanged:
    detect_scale(...)
    apply_scale(...)
    px_to_mm(...)
    point_px_to_mm(...)
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import ScaleInfo, ScaledCandidate

UNKNOWN_SCALE_PENALTY = 0.40


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _mm_per_px_from_dpi(dpi):
    if dpi and dpi > 0:
        return 25.4 / float(dpi)
    return None


def _candidate_px_size(c):
    g = c.geometry

    if c.kind == "circle":
        return g.get("diameter_px", 0.0)

    if c.kind == "slot":
        return max(
            g.get("length_px", 0.0),
            g.get("width_px", 0.0),
        )

    if c.kind == "rectangle":
        return max(
            g.get("width_px", 0.0),
            g.get("height_px", 0.0),
        )

    return 0.0


def _ann_center(a):
    return (
        a.bbox["x"] + a.bbox["w"] / 2.0,
        a.bbox["y"] + a.bbox["h"] / 2.0,
    )


def _cand_center(c):
    return (
        c.geometry.get("cx", 0.0),
        c.geometry.get("cy", 0.0),
    )


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------------------
# Detect by dimensions
# ---------------------------------------------------------------------------

def _dimension_anchor(annotations, candidates):
    vals = []

    for ann in annotations:
        if ann.parsed.value is None:
            continue

        if ann.parsed.value <= 0:
            continue

        if ann.ocr_confidence < 0.70:
            continue

        token = ann.parsed.token_type
        if token not in (
            "dimension_linear",
            "dimension_diameter",
            "dimension_radial",
        ):
            continue

        ac = _ann_center(ann)

        ranked = []

        for cand in candidates:
            px = _candidate_px_size(cand)
            if px <= 0:
                continue

            d = _dist(ac, _cand_center(cand))
            ranked.append((d, cand))

        ranked.sort(key=lambda x: x[0])

        for d, cand in ranked[:3]:
            px = _candidate_px_size(cand)
            if px <= 0:
                continue

            mm_per_px = float(ann.parsed.value) / float(px)

            if 0.01 <= mm_per_px <= 5.0:
                vals.append(mm_per_px)

    if len(vals) >= 1:
        return statistics.median(vals)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_scale(
    annotations,
    candidates,
    page_width_px,
    page_height_px,
    title_block=None,
    pdf_dpi=0.0,
    operator_override_mm_per_px=None,
):
    page = 1

    # 1 manual override
    if operator_override_mm_per_px and operator_override_mm_per_px > 0:
        return ScaleInfo(
            page=page,
            px_per_mm=1.0 / operator_override_mm_per_px,
            anchor_method="operator_override",
            anchor_confidence=1.0,
            origin_px={"x": 0.0, "y": 0.0},
        )

    # 2 dimensions
    mm_per_px = _dimension_anchor(annotations, candidates)
    if mm_per_px:
        return ScaleInfo(
            page=page,
            px_per_mm=1.0 / mm_per_px,
            anchor_method="dimension_text",
            anchor_confidence=0.90,
            origin_px={"x": 0.0, "y": 0.0},
        )

    # 3 dpi metadata
    mm_per_px = _mm_per_px_from_dpi(pdf_dpi)
    if mm_per_px:
        return ScaleInfo(
            page=page,
            px_per_mm=1.0 / mm_per_px,
            anchor_method="pdf_metadata",
            anchor_confidence=0.55,
            origin_px={"x": 0.0, "y": 0.0},
        )

    # 4 unknown
    return ScaleInfo(
        page=page,
        px_per_mm=0.0,
        anchor_method="unknown",
        anchor_confidence=0.0,
        origin_px={"x": 0.0, "y": 0.0},
    )


def px_to_mm(px_value, scale):
    if scale.px_per_mm <= 0:
        return None
    return round(float(px_value) / scale.px_per_mm, 4)


def point_px_to_mm(x, y, scale):
    ox = scale.origin_px.get("x", 0.0)
    oy = scale.origin_px.get("y", 0.0)

    if scale.px_per_mm <= 0:
        return (None, None)

    return (
        round((x - ox) / scale.px_per_mm, 4),
        round((y - oy) / scale.px_per_mm, 4),
    )


def apply_scale(candidates, scale):
    scaled = []

    import dataclasses

    for cand in candidates:
        conf = cand.confidence

        if scale.anchor_method == "unknown":
            conf = round(conf * (1.0 - UNKNOWN_SCALE_PENALTY), 4)

        g = cand.geometry
        kind = cand.kind

        geom = {}

        if kind == "circle":
            cx, cy = point_px_to_mm(g["cx"], g["cy"], scale)
            r = px_to_mm(g.get("radius_px", 0.0), scale)

            geom = {
                "cx": cx,
                "cy": cy,
                "radius": r,
                "diameter": None if r is None else round(r * 2, 4),
            }

        elif kind == "slot":
            cx, cy = point_px_to_mm(g["cx"], g["cy"], scale)

            geom = {
                "cx": cx,
                "cy": cy,
                "length": px_to_mm(g.get("length_px", 0.0), scale),
                "width": px_to_mm(g.get("width_px", 0.0), scale),
            }

        elif kind == "rectangle":
            bbox = g.get("bbox", {})

            cx, cy = point_px_to_mm(
                bbox.get("cx", 0.0),
                bbox.get("cy", 0.0),
                scale,
            )

            geom = {
                "cx": cx,
                "cy": cy,
                "width": px_to_mm(
                    g.get("width_px", bbox.get("width", 0.0)),
                    scale,
                ),
                "height": px_to_mm(
                    g.get("height_px", bbox.get("height", 0.0)),
                    scale,
                ),
            }

        updated = dataclasses.replace(cand, confidence=conf)

        scaled.append(
            ScaledCandidate(
                candidate=updated,
                geometry_mm=geom,
                scale_info=scale,
            )
        )

    return scaled

   
    
