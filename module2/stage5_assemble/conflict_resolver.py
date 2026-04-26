"""
stage5_assemble/conflict_resolver.py — FINAL STABLE VERSION
===========================================================
Resolve overlapping geometry candidates.

FINAL RULES:
- No silent deletion.
- Text-linked candidates preferred.
- Circle vs Slot:
    Slot only wins if:
        1. slot confidence >= threshold
        2. slot has linked annotation evidence
    otherwise circle wins.
- Generic overlaps:
    linked > unlinked
    else higher confidence wins
- If no conflicts: unchanged return
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import ScaledCandidate, SuppressedCandidate


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

_OVERLAP_THRESHOLD = 0.35
_SLOT_WINS_CONFIDENCE = 0.70


# ---------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------

def _bbox_of(sc: ScaledCandidate) -> dict:
    g = sc.geometry_mm
    kind = sc.candidate.kind

    cx = g.get("cx", 0.0)
    cy = g.get("cy", 0.0)

    if kind == "circle":
        r = g.get("radius", 0.0)
        return {
            "x": cx - r,
            "y": cy - r,
            "w": r * 2,
            "h": r * 2,
        }

    if kind == "slot":
        l = g.get("length", 0.0)
        w = g.get("width", 0.0)
        return {
            "x": cx - l / 2,
            "y": cy - w / 2,
            "w": l,
            "h": w,
        }

    if kind == "rectangle":
        w = g.get("width", 0.0)
        h = g.get("height", 0.0)
        return {
            "x": cx - w / 2,
            "y": cy - h / 2,
            "w": w,
            "h": h,
        }

    area = sc.candidate.evidence.get("area_px", 100.0)
    side = math.sqrt(area)

    return {
        "x": cx - side / 2,
        "y": cy - side / 2,
        "w": side,
        "h": side,
    }


def _iou(a: dict, b: dict) -> float:
    ax2 = a["x"] + a["w"]
    ay2 = a["y"] + a["h"]

    bx2 = b["x"] + b["w"]
    by2 = b["y"] + b["h"]

    ix1 = max(a["x"], b["x"])
    iy1 = max(a["y"], b["y"])
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter

    if union <= 0:
        return 0.0

    return inter / union


# ---------------------------------------------------------
# Decision logic
# ---------------------------------------------------------

def _resolve_pair(a, b, linked_ids):
    kind_a = a.candidate.kind
    kind_b = b.candidate.kind

    conf_a = a.candidate.confidence
    conf_b = b.candidate.confidence

    a_linked = a.candidate.id in linked_ids
    b_linked = b.candidate.id in linked_ids

    # ---------------------------------------------
    # Circle vs Slot
    # ---------------------------------------------
    if {kind_a, kind_b} == {"circle", "slot"}:
        slot = a if kind_a == "slot" else b
        circ = a if kind_a == "circle" else b

        slot_linked = slot.candidate.id in linked_ids

        if (
            slot.candidate.confidence >= _SLOT_WINS_CONFIDENCE
            and slot_linked
        ):
            return (
                slot,
                circ,
                "slot_beats_circle(linked_high_conf)"
            )

        return (
            circ,
            slot,
            "circle_beats_slot"
        )

    # ---------------------------------------------
    # Generic text-linked priority
    # ---------------------------------------------
    if a_linked and not b_linked:
        return a, b, "text_linked_wins"

    if b_linked and not a_linked:
        return b, a, "text_linked_wins"

    # ---------------------------------------------
    # Confidence
    # ---------------------------------------------
    if conf_a >= conf_b:
        return a, b, "higher_confidence"

    return b, a, "higher_confidence"


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def resolve_conflicts(
    scaled_candidates: list,
    pairs: list,
):
    if len(scaled_candidates) <= 1:
        return scaled_candidates, []

    linked_ids = {
        p.candidate.candidate.id
        for p in pairs
    }

    conflicts = []

    for i in range(len(scaled_candidates)):
        box_i = _bbox_of(scaled_candidates[i])

        for j in range(i + 1, len(scaled_candidates)):
            box_j = _bbox_of(scaled_candidates[j])

            overlap = _iou(box_i, box_j)

            if overlap >= _OVERLAP_THRESHOLD:
                conflicts.append((i, j, overlap))

    if not conflicts:
        return scaled_candidates, []

    suppressed_idx = set()
    suppressed = []

    for i, j, overlap in conflicts:
        if i in suppressed_idx or j in suppressed_idx:
            continue

        cand_a = scaled_candidates[i]
        cand_b = scaled_candidates[j]

        winner, loser, reason = _resolve_pair(
            cand_a,
            cand_b,
            linked_ids
        )

        loser_idx = i if loser is cand_a else j
        suppressed_idx.add(loser_idx)

        suppressed.append(
            SuppressedCandidate(
                candidate=loser,
                reason=f"conflict_resolution: {reason} (iou={overlap:.2f})"
            )
        )

    kept = [
        sc
        for idx, sc in enumerate(scaled_candidates)
        if idx not in suppressed_idx
    ]

    return kept, suppressed