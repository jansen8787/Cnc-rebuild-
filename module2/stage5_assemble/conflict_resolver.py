"""
stage5_assemble/conflict_resolver.py — Module 2: Stage 5 (FIXED)
===============================================================
Resolve overlapping geometry candidates.

Fixes:
- better multi-conflict handling
- slot beats circle only with text evidence + confidence
- deterministic winner logic
- no silent suppression

Public API unchanged:
    resolve_conflicts(scaled_candidates, pairs)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import SuppressedCandidate


OVERLAP_THRESHOLD = 0.35
SLOT_WIN_CONF = 0.70


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _bbox(sc):
    g = sc.geometry_mm
    k = sc.candidate.kind

    cx = g.get("cx", 0.0) or 0.0
    cy = g.get("cy", 0.0) or 0.0

    if k == "circle":
        r = g.get("radius", 0.0) or 0.0
        return (cx - r, cy - r, cx + r, cy + r)

    if k == "slot":
        l = g.get("length", 0.0) or 0.0
        w = g.get("width", 0.0) or 0.0
        return (cx - l / 2, cy - w / 2, cx + l / 2, cy + w / 2)

    if k == "rectangle":
        w = g.get("width", 0.0) or 0.0
        h = g.get("height", 0.0) or 0.0
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    return (cx - 1, cy - 1, cx + 1, cy + 1)


def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)

    inter = iw * ih

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


# ---------------------------------------------------------------------------
# Winner logic
# ---------------------------------------------------------------------------

def _linked_ids(pairs):
    return {p.candidate.candidate.id for p in pairs}


def _winner(a, b, linked):
    ka = a.candidate.kind
    kb = b.candidate.kind

    ca = a.candidate.confidence
    cb = b.candidate.confidence

    ida = a.candidate.id
    idb = b.candidate.id

    a_link = ida in linked
    b_link = idb in linked

    # slot vs circle special rule
    if {ka, kb} == {"slot", "circle"}:
        slot = a if ka == "slot" else b
        circ = b if ka == "slot" else a

        slot_link = slot.candidate.id in linked

        if slot.candidate.confidence >= SLOT_WIN_CONF and slot_link:
            return slot, circ, "slot_beats_circle"

        return circ, slot, "circle_beats_slot"

    # linked wins
    if a_link and not b_link:
        return a, b, "text_linked"

    if b_link and not a_link:
        return b, a, "text_linked"

    # confidence wins
    if ca > cb:
        return a, b, "higher_conf"

    if cb > ca:
        return b, a, "higher_conf"

    # stable fallback by id
    if ida <= idb:
        return a, b, "stable_id"

    return b, a, "stable_id"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_conflicts(scaled_candidates, pairs):
    if len(scaled_candidates) <= 1:
        return scaled_candidates, []

    linked = _linked_ids(pairs)

    active = list(scaled_candidates)
    suppressed = []

    changed = True

    while changed:
        changed = False

        for i in range(len(active)):
            if changed:
                break

            for j in range(i + 1, len(active)):
                a = active[i]
                b = active[j]

                ov = _iou(_bbox(a), _bbox(b))

                if ov < OVERLAP_THRESHOLD:
                    continue

                keep, lose, reason = _winner(a, b, linked)

                active.remove(lose)

                suppressed.append(
                    SuppressedCandidate(
                        candidate=lose,
                        reason=f"conflict_resolution: {reason} (iou={ov:.2f})",
                    )
                )

                changed = True
                break

    return active, suppressed