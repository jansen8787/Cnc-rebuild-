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

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import (  # noqa: E402
    ScaledCandidate,
    LinkedPair,
    TitleBlockInfo,
)


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

def _run_tests() -> int:
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    results: list = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        results.append((name, cond))
        marker = f"  ({detail})" if detail else ""
        print(f"  {PASS if cond else FAIL}  {name}{marker}")

    print("\n── Stage 5: Feature Inferrer self-tests ──\n")

    from m2types import GeometryCandidate, ScaleInfo, ScaledCandidate, TextAnnotation, LinkedPair
    from stage2_ocr.symbol_parser import parse_symbol

    scale = ScaleInfo(1, 10.0, "test", 0.9, {"x": 0.0, "y": 0.0})

    def make_circle_sc(cid: str, cx: float, cy: float, r: float,
                       conf: float = 0.85) -> ScaledCandidate:
        cand = GeometryCandidate(
            id=cid, kind="circle",
            geometry={"cx": cx*10, "cy": cy*10, "radius_px": r*10, "diameter_px": r*20},
            confidence=conf, detector="test", page=1,
            evidence={"area_px": 3.14*(r*10)**2},
        )
        return ScaledCandidate(
            candidate=cand,
            geometry_mm={"cx": cx, "cy": cy, "radius": r, "diameter": r*2},
            scale_info=scale,
        )

    def make_pair(ann_text: str, sc: ScaledCandidate) -> LinkedPair:
        ann = TextAnnotation(
            id="ann_test", page=1, raw_text=ann_text,
            parsed=parse_symbol(ann_text),
            bbox={"x": 50, "y": 50, "w": 30, "h": 12},
            ocr_confidence=0.9,
        )
        return LinkedPair(
            annotation=ann,
            candidate=sc,
            link_confidence=0.8,
            link_type="diameter",
            ambiguous=False,
        )

    # ── 1. Circle + Ø text → hole
    sc1 = make_circle_sc("c1", 10.0, 10.0, 3.0)
    p1  = make_pair("Ø6", sc1)
    feats = infer_features([sc1], [p1])
    check("circle + Ø → hole",             feats[0]["kind"] == "hole",
          f"got {feats[0]['kind']}")
    check("hole has diameter",             "diameter" in feats[0])
    check("hole has x/y",                  "x" in feats[0] and "y" in feats[0])

    # ── 2. Circle + M text → thread/tap_hole
    sc2 = make_circle_sc("c2", 20.0, 20.0, 4.0)
    p2  = make_pair("M8", sc2)
    feats2 = infer_features([sc2], [p2])
    check("circle + M → thread/tap_hole",
          feats2[0]["kind"] in ("thread", "tap_hole"),
          f"got {feats2[0]['kind']}")

    # ── 3. Slot candidate → slot kind
    from m2types import GeometryCandidate
    slot_cand = GeometryCandidate(
        id="slot1", kind="slot",
        geometry={"cx": 50.0, "cy": 50.0, "length_px": 120.0, "width_px": 30.0},
        confidence=0.85, detector="test", page=1,
        evidence={"area_px": 3600.0},
    )
    slot_sc = ScaledCandidate(
        candidate=slot_cand,
        geometry_mm={"cx": 50.0, "cy": 50.0, "length": 12.0, "width": 3.0},
        scale_info=scale,
    )
    feats3 = infer_features([slot_sc], [])
    check("slot candidate → slot kind",   feats3[0]["kind"] == "slot",
          f"got {feats3[0]['kind']}")
    check("slot has length",              "length" in feats3[0])
    check("slot has width",               "width"  in feats3[0])

    # ── 4. Largest circle → contour (no text)
    large_sc = make_circle_sc("big", 50.0, 50.0, 40.0)
    small_sc = make_circle_sc("sm",  10.0, 10.0,  3.0)
    feats4 = infer_features([large_sc, small_sc], [])
    kinds4 = {f["id"]: f["kind"] for f in feats4}
    check("largest unlabelled circle → contour",
          feats4[0]["kind"] == "contour",  # sorted by area, first is largest
          f"kinds: {list(kinds4.values())}")

    # ── 5. Manufacturing fields from tolerance annotation
    sc5 = make_circle_sc("c5", 10.0, 10.0, 5.0)
    ann_tol = TextAnnotation(
        id="ann_tol", page=1, raw_text="25±0.05",
        parsed=parse_symbol("25±0.05"),
        bbox={"x": 0, "y": 0, "w": 50, "h": 12},
        ocr_confidence=0.92,
    )
    p5 = LinkedPair(ann_tol, sc5, 0.8, "tolerance", False)
    feats5 = infer_features([sc5], [p5])
    mfg5 = feats5[0].get("manufacturing", {})
    check("tolerance → tolerancePlus",    "tolerancePlus"  in mfg5,
          f"mfg={mfg5}")
    check("tolerance → toleranceMinus",   "toleranceMinus" in mfg5)

    # ── 6. Fit annotation → fitClass in manufacturing
    sc6 = make_circle_sc("c6", 10.0, 10.0, 5.0)
    ann_fit = TextAnnotation(
        id="ann_fit", page=1, raw_text="Ø20H7",
        parsed=parse_symbol("Ø20H7"),
        bbox={"x": 0, "y": 0, "w": 50, "h": 12},
        ocr_confidence=0.95,
    )
    p6 = LinkedPair(ann_fit, sc6, 0.8, "fit", False)
    feats6 = infer_features([sc6], [p6])
    mfg6 = feats6[0].get("manufacturing", {})
    check("fit annotation → fitClass",    mfg6.get("fitClass") == "H7",
          f"mfg={mfg6}")

    # ── 7. Feature has id, confidence, source
    f0 = feats[0]
    check("feature has id",             "id" in f0)
    check("feature has confidence",     "confidence" in f0)
    check("feature has source",         "source" in f0)
    check("source == vision",           f0["source"] == "vision")
    check("confidence ∈ [0,1]",         0.0 <= f0["confidence"] <= 1.0)

    # ── 8. Empty → empty
    feats_empty = infer_features([], [])
    check("empty input → empty output", feats_empty == [])

    # ── 9. No text → geometry-only confidence
    feats9 = infer_features([sc1], [])
    check("no text → confidence < text-corroborated",
          feats9[0]["confidence"] < feats[0]["confidence"])

    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n── Feature Inferrer tests: {passed}/{total} passed ──\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_tests())
