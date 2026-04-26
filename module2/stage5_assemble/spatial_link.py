"""
stage5_assemble/spatial_link.py — Module 2: Stage 5 (FIXED)
===========================================================
Improved spatial association: pair TextAnnotation with the best
ScaledCandidate using robust multi-factor scoring.

Fixes vs old version:
- removed hidden global-variable bug
- better distance scoring
- stronger type compatibility
- size plausibility scoring improved
- ambiguity handling improved
- deterministic stable sort
- cleaner suppression logic

Public API unchanged:
    link_text_to_geometry(annotations, scaled_candidates)
        -> (pairs, suppressed)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import (  # noqa: E402
    LinkedPair,
    SuppressedCandidate,
    bbox_center,
)

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

W_TYPE = 0.45
W_DIST = 0.35
W_SIZE = 0.15
W_DIR  = 0.05

MIN_SCORE = 0.28
AMBIGUITY_GAP = 0.08


# ---------------------------------------------------------------------------
# Compatibility map
# ---------------------------------------------------------------------------

COMPAT = {
    "dimension_diameter": {"circle", "hole", "thread"},
    "dimension_radial": {"circle", "hole", "slot"},
    "thread_callout": {"circle", "hole"},
    "fit": {"circle", "hole"},
    "quantity": {"circle", "hole", "thread"},
    "tolerance": {"circle", "hole", "slot", "rectangle", "polygon"},
    "dimension_linear": {"slot", "rectangle", "polygon", "line"},
    "chamfer": {"polygon", "rectangle", "line"},
    "angle": {"line", "polygon"},
    "surface_finish": {"polygon", "rectangle"},
    "label": set(),
    "title_block": set(),
    "reference": set(),
    "unknown": set(),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ann_center(ann):
    return bbox_center(ann.bbox)


def _cand_center(sc):
    g = sc.geometry_mm
    return (
        float(g.get("cx", 0.0)),
        float(g.get("cy", 0.0)),
    )


def _px_to_mm(pt_px, px_per_mm):
    if px_per_mm <= 0:
        return pt_px
    return (pt_px[0] / px_per_mm, pt_px[1] / px_per_mm)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _type_score(ann, sc):
    allowed = COMPAT.get(ann.parsed.token_type, set())
    if sc.candidate.kind in allowed:
        return W_TYPE
    return 0.0


def _distance_score(ann, sc, px_per_mm):
    ac = _ann_center(ann)
    cc = _cand_center(sc)

    ac_mm = _px_to_mm(ac, px_per_mm)
    d = _dist(ac_mm, cc)

    # soft falloff
    if d >= 80:
        return 0.0

    return W_DIST * (1.0 - d / 80.0)


def _direction_score(ann, sc):
    # lightweight bonus only
    return W_DIR * 0.7


def _feature_nominal_size(sc):
    g = sc.geometry_mm
    k = sc.candidate.kind

    if k == "circle":
        return g.get("diameter", 0.0)

    if k == "slot":
        return max(g.get("length", 0.0), g.get("width", 0.0))

    if k == "rectangle":
        return max(g.get("width", 0.0), g.get("height", 0.0))

    return 0.0


def _size_score(ann, sc):
    val = ann.parsed.value
    if val is None or val <= 0:
        return 0.0

    feat = _feature_nominal_size(sc)
    if feat <= 0:
        return 0.0

    ratio = float(val) / float(feat)

    # smooth plausibility curve
    err = abs(ratio - 1.0)

    if err <= 0.05:
        return W_SIZE
    if err <= 0.15:
        return W_SIZE * 0.7
    if err <= 0.30:
        return W_SIZE * 0.4
    return 0.0


def _link_type(ann):
    mapping = {
        "dimension_diameter": "diameter",
        "dimension_radial": "radius",
        "thread_callout": "thread",
        "fit": "fit",
        "quantity": "group",
        "tolerance": "tolerance",
        "dimension_linear": "linear",
        "chamfer": "chamfer",
        "surface_finish": "surface",
    }
    return mapping.get(ann.parsed.token_type, "unknown")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def link_text_to_geometry(annotations, scaled_candidates):
    """
    Returns:
        (pairs, suppressed)
    """
    pairs = []
    suppressed = []

    if not annotations or not scaled_candidates:
        return pairs, suppressed

    px_per_mm = scaled_candidates[0].scale_info.px_per_mm

    for ann in annotations:
        token = ann.parsed.token_type

        if token not in COMPAT:
            continue

        if not COMPAT[token]:
            continue

        scored = []

        for sc in scaled_candidates:
            s = (
                _type_score(ann, sc)
                + _distance_score(ann, sc, px_per_mm)
                + _size_score(ann, sc)
                + _direction_score(ann, sc)
            )

            scored.append((round(s, 6), sc))

        scored.sort(
            key=lambda x: (
                -x[0],
                x[1].candidate.id
            )
        )

        best_score, best = scored[0]

        if best_score < MIN_SCORE:
            suppressed.append(
                SuppressedCandidate(
                    candidate=best,
                    reason=f"no_annotation_match ({best_score:.2f})",
                )
            )
            continue

        ambiguous = False
        if len(scored) > 1:
            second_score = scored[1][0]
            if (best_score - second_score) < AMBIGUITY_GAP:
                ambiguous = True

        pair = LinkedPair(
            annotation=ann,
            candidate=best,
            link_confidence=float(best_score),
            link_type=_link_type(ann),
            ambiguous=ambiguous,
        )

        pairs.append(pair)

    return pairs, suppressed