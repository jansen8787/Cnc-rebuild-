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

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import (  # noqa: E402
    TextAnnotation,
    ScaledCandidate,
    LinkedPair,
    SuppressedCandidate,
    bbox_center,
)


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

def _run_tests() -> int:
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    results: list = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        results.append((name, cond))
        marker = f"  ({detail})" if detail else ""
        print(f"  {PASS if cond else FAIL}  {name}{marker}")

    print("\n── Stage 5: Spatial Link self-tests ──\n")

    from m2types import (
        ParsedSymbol, GeometryCandidate, ScaleInfo, ScaledCandidate, TextAnnotation
    )
    from stage2_ocr.symbol_parser import parse_symbol

    def make_ann(text: str, x: int, y: int) -> TextAnnotation:
        return TextAnnotation(
            id=f"ann_{x}_{y}", page=1, raw_text=text,
            parsed=parse_symbol(text),
            bbox={"x": x, "y": y, "w": 30, "h": 12},
            ocr_confidence=0.9,
        )

    def make_circle(cx_mm: float, cy_mm: float, r_mm: float,
                    cx_px: float = None, cy_px: float = None) -> ScaledCandidate:
        from dataclasses import replace
        cand = GeometryCandidate(
            id="cand_c_0001", kind="circle",
            geometry={"cx": cx_px or cx_mm*10, "cy": cy_px or cy_mm*10,
                      "radius_px": r_mm*10, "diameter_px": r_mm*20},
            confidence=0.9, detector="test", page=1,
            evidence={"area_px": 3.14*(r_mm*10)**2},
        )
        scale = ScaleInfo(1, 10.0, "test", 0.9, {"x": 0.0, "y": 0.0})
        return ScaledCandidate(
            candidate=cand,
            geometry_mm={"cx": cx_mm, "cy": cy_mm, "radius": r_mm, "diameter": r_mm*2},
            scale_info=scale,
        )

    # ── 1. Ø12 annotation near Ø12 circle → linked
    ann_diam = make_ann("Ø12", x=120, y=50)   # px near circle at (100,100)px=(10,10)mm
    sc1 = make_circle(10.0, 10.0, 6.0)        # cx_px=100, cy_px=100

    pairs, suppressed = link_text_to_geometry([ann_diam], [sc1])
    check("Ø12 near circle → linked",      len(pairs) == 1,
          f"pairs={len(pairs)} suppressed={len(suppressed)}")
    if pairs:
        check("link_type == diameter",     pairs[0].link_type == "diameter")
        check("link_confidence > 0",       pairs[0].link_confidence > 0)

    # ── 2. Label annotation → not linked (no compat types)
    ann_label = make_ann("SECTION-A", x=50, y=50)
    pairs2, _ = link_text_to_geometry([ann_label], [sc1])
    check("label not linked",              len(pairs2) == 0)

    # ── 3. Title block → not linked
    ann_tb = make_ann("MATERIAL: AL6061", x=400, y=500)
    pairs3, _ = link_text_to_geometry([ann_tb], [sc1])
    check("title_block not linked",        len(pairs3) == 0)

    # ── 4. No candidates → no pairs
    pairs4, _ = link_text_to_geometry([ann_diam], [])
    check("no candidates → no pairs",     len(pairs4) == 0)

    # ── 5. No annotations → no pairs
    pairs5, _ = link_text_to_geometry([], [sc1])
    check("no annotations → no pairs",    len(pairs5) == 0)

    # ── 6. Two candidates for one annotation → choose nearest
    sc_near = make_circle(10.0, 10.0, 3.0, cx_px=100, cy_px=100)
    sc_far  = make_circle(50.0, 50.0, 3.0, cx_px=500, cy_px=500)
    pairs6, _ = link_text_to_geometry([ann_diam], [sc_near, sc_far])
    if pairs6:
        check("nearest circle chosen",    pairs6[0].candidate.candidate.id == sc_near.candidate.id,
              f"got {pairs6[0].candidate.candidate.id}")

    # ── 7. _type_score: Ø → circle gets type bonus
    ts = _type_score(ann_diam, sc1)
    check("Ø→circle type_score == 0.5",   abs(ts - 0.5) < 0.01, f"ts={ts}")

    # ── 8. _size_score: Ø12 → r=6mm circle (diameter=12mm)
    ss = _size_score(ann_diam, sc1)
    check("Ø12 → 12mm circle size match", ss > 0, f"ss={ss}")

    # ── 9. Ambiguous when two candidates very close in score
    sc_a = make_circle(10.0, 10.0, 6.0, cx_px=100, cy_px=100)
    sc_b = make_circle(10.5, 10.5, 6.0, cx_px=105, cy_px=105)
    pairs9, _ = link_text_to_geometry([ann_diam], [sc_a, sc_b])
    if pairs9:
        check("close candidates → ambiguous",  pairs9[0].ambiguous)

    # ── 10. link_type mapping
    ann_r  = make_ann("R12.5", x=0, y=0)
    ann_m  = make_ann("M8",    x=0, y=0)
    ann_lin = make_ann("25mm", x=0, y=0)
    check("R → link_type radius",  _link_type(ann_r)   == "radius")
    check("M → link_type thread",  _link_type(ann_m)   == "thread")
    check("25mm → link_type linear", _link_type(ann_lin) == "linear")

    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n── Spatial Link tests: {passed}/{total} passed ──\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_tests())
