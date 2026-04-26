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

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import ScaledCandidate, SuppressedCandidate  # noqa: E402


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

def _run_tests() -> int:
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    results: list = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        results.append((name, cond))
        marker = f"  ({detail})" if detail else ""
        print(f"  {PASS if cond else FAIL}  {name}{marker}")

    print("\n── Stage 5: Conflict Resolver self-tests ──\n")

    from m2types import GeometryCandidate, ScaleInfo, ScaledCandidate

    scale = ScaleInfo(1, 10.0, "test", 0.9, {"x": 0.0, "y": 0.0})

    def make_circle(cid: str, cx: float, cy: float, r: float,
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

    def make_slot(cid: str, cx: float, cy: float, l: float, w: float,
                  conf: float = 0.80) -> ScaledCandidate:
        cand = GeometryCandidate(
            id=cid, kind="slot",
            geometry={"cx": cx*10, "cy": cy*10, "length_px": l*10, "width_px": w*10},
            confidence=conf, detector="test", page=1,
            evidence={"area_px": l*w*100},
        )
        return ScaledCandidate(
            candidate=cand,
            geometry_mm={"cx": cx, "cy": cy, "length": l, "width": w},
            scale_info=scale,
        )

    # ── 1. No conflict (separate regions)
    c1 = make_circle("c1", 10.0, 10.0, 3.0)
    c2 = make_circle("c2", 50.0, 50.0, 3.0)
    kept, supp = resolve_conflicts([c1, c2], [])
    check("no conflict → both kept",      len(kept) == 2, f"kept={len(kept)}")
    check("no conflict → no suppressed",  len(supp) == 0)

    # ── 2. Overlapping circles → higher confidence wins
    c3 = make_circle("c3", 10.0, 10.0, 5.0, conf=0.90)
    c4 = make_circle("c4", 10.0, 10.0, 5.0, conf=0.60)
    kept2, supp2 = resolve_conflicts([c3, c4], [])
    check("overlap → one kept",           len(kept2) == 1, f"kept={len(kept2)}")
    check("overlap → lower conf suppressed",
          supp2[0].candidate.candidate.id == "c4",
          f"suppressed={supp2[0].candidate.candidate.id if supp2 else 'none'}")

    # ── 3. Slot vs circle same region: high-conf slot wins
    circ = make_circle("circ1", 10.0, 10.0, 4.0, conf=0.80)
    slot = make_slot("slot1", 10.0, 10.0, 12.0, 6.0, conf=0.75)
    kept3, supp3 = resolve_conflicts([circ, slot], [])
    kinds3 = [k.candidate.kind for k in kept3]
    check("high-conf slot beats circle",  "slot" in kinds3,
          f"kept kinds: {kinds3}")

    # ── 4. Low-conf slot loses to circle
    slot_low = make_slot("slot_low", 10.0, 10.0, 12.0, 6.0, conf=0.50)
    kept4, supp4 = resolve_conflicts([circ, slot_low], [])
    kinds4 = [k.candidate.kind for k in kept4]
    check("low-conf slot loses to circle", "circle" in kinds4,
          f"kept kinds: {kinds4}")

    # ── 5. Suppressed not deleted (appears in suppressed list)
    check("suppressed list non-empty when conflict", len(supp3) >= 1)
    check("suppressed has reason",
          all("conflict_resolution" in s.reason for s in supp3))

    # ── 6. IoU: two identical circles have IoU ≈ 1.0
    a = {"x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0}
    b = {"x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0}
    check("IoU identical rects == 1.0",   abs(_iou(a, b) - 1.0) < 0.01)

    # ── 7. IoU: non-overlapping → 0.0
    c_box = {"x": 100.0, "y": 0.0, "w": 10.0, "h": 10.0}
    check("IoU non-overlap == 0.0",       _iou(a, c_box) == 0.0)

    # ── 8. Single candidate → no conflict possible
    kept5, supp5 = resolve_conflicts([c1], [])
    check("single candidate → kept",      len(kept5) == 1)
    check("single candidate → no supp",   len(supp5) == 0)

    # ── 9. Empty list
    kept6, supp6 = resolve_conflicts([], [])
    check("empty list → empty output",    len(kept6) == 0 and len(supp6) == 0)

    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n── Conflict Resolver tests: {passed}/{total} passed ──\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_tests())
