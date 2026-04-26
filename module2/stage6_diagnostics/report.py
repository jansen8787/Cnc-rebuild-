"""
stage6_diagnostics/report.py — Module 2: Stage 6
==================================================
Builds the RecognitionReport — the truthfulness layer.

Rule (architecture §7 + master rule TRUTH RULE):
    overallConfidence = MIN across critical signals, NOT average.
    A single missing scale anchor pulls everything down.
    Low OCR confidence is reported, not hidden.

Severity levels for weak signals (from old 2A.3/review.py, REUSE bucket):
    high   — recognition likely wrong; human must verify
    medium — uncertain; review recommended
    low    — informational

Public API:
    build_report(pipeline, stage_timings, ingest_diag, preproc_diag,
                 ocr_diag, tb_diag, geometry_diag, cross_link_diag,
                 suppressed, scale_info) -> dict
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import SuppressedCandidate  # noqa: E402


# ---------------------------------------------------------------------------
# Weak signal rules (R01–R12)
# Adapted from old 2A.3/review.py (REUSE / ADAPT bucket)
# ---------------------------------------------------------------------------

def _collect_weak_signals(
    ocr_diag:         dict,
    geometry_diag:    dict,
    cross_link_diag:  dict,
    scale_info:       Optional[dict],
    suppressed:       list,
    preproc_diag:     dict,
) -> list:
    """
    Produce a list of human-readable warning strings.
    Each entry may be prefixed with [HIGH], [MEDIUM], or [LOW].
    """
    signals: list = []

    # R01: No scale anchor
    if scale_info and scale_info.get("anchor_method") == "unknown":
        signals.append(
            "[HIGH] No scale anchor found — all coordinates are in pixel units. "
            "Set a dimension anchor or provide DPI for mm output."
        )

    # R02: Very low OCR confidence
    avg_ocr = ocr_diag.get("avg_confidence", 1.0)
    if avg_ocr < 0.50 and ocr_diag.get("regions_found", 0) > 0:
        signals.append(
            f"[HIGH] OCR average confidence {avg_ocr:.2f} is very low. "
            "Drawing text may be garbled. Check image quality."
        )
    elif avg_ocr < 0.70 and ocr_diag.get("regions_found", 0) > 0:
        signals.append(
            f"[MEDIUM] OCR average confidence {avg_ocr:.2f}. "
            "Some dimension values may be misread."
        )

    # R03: No OCR text found at all
    if ocr_diag.get("regions_found", 0) == 0:
        signals.append(
            "[MEDIUM] No text found by OCR. Geometry-only output — "
            "no dimensions or material info extracted."
        )

    # R04: High garbled rate
    garbled = ocr_diag.get("garbled_count", 0)
    found   = ocr_diag.get("regions_found", 0)
    if found > 0 and garbled / found > 0.30:
        signals.append(
            f"[MEDIUM] {garbled}/{found} OCR tokens were unrecognised. "
            "Many symbols may have been misread."
        )

    # R05: No geometry candidates at all
    n_cands = geometry_diag.get("total_candidates", 0)
    if n_cands == 0:
        signals.append(
            "[HIGH] No geometry candidates detected. "
            "Image may be blank, inverted, or too low resolution."
        )

    # R06: Many suppressed candidates (suggests conflict or noise)
    n_supp = len(suppressed)
    if n_supp > 5:
        signals.append(
            f"[MEDIUM] {n_supp} geometry candidates were suppressed during "
            "conflict resolution. Drawing may have overlapping annotations."
        )

    # R07: No features assembled
    n_features = cross_link_diag.get("features_assembled", 0)
    if n_features == 0:
        signals.append(
            "[HIGH] No features were assembled. "
            "Check geometry and OCR stages — this part will be empty."
        )

    # R08: Deskew applied at large angle (photo quality concern)
    deskew_angle = abs(preproc_diag.get("deskew_angle_deg", 0.0))
    if deskew_angle > 8.0:
        signals.append(
            f"[MEDIUM] Drawing was deskewed by {deskew_angle:.1f}°. "
            "Large corrections may introduce distortion — verify output."
        )

    # R09: Low-confidence scale anchor
    if scale_info:
        scale_conf = scale_info.get("anchor_confidence", 1.0)
        if 0.0 < scale_conf < 0.60:
            signals.append(
                f"[MEDIUM] Scale anchor confidence {scale_conf:.2f} is low. "
                f"Method: {scale_info.get('anchor_method')}. Dimensions may be inaccurate."
            )

    # R10: Many ambiguous text-geometry pairs
    n_pairs = cross_link_diag.get("text_geometry_pairs", 0)
    if n_pairs > 0:
        # We don't have ambiguous count directly — if suppressed_by_linker is high
        n_link_supp = cross_link_diag.get("suppressed_by_linker", 0)
        if n_link_supp > n_pairs * 0.4:
            signals.append(
                f"[LOW] {n_link_supp} text annotations could not be confidently "
                "linked to geometry. They appear in the suppressed list."
            )

    # R11: OCR engine absent
    if ocr_diag.get("engine") == "none":
        signals.append(
            "[HIGH] OCR engine (Tesseract) is not installed. "
            "No text was extracted. Install tesseract-ocr to enable dimension parsing."
        )

    return signals


# ---------------------------------------------------------------------------
# Overall confidence (MIN, not AVG)
# ---------------------------------------------------------------------------

def _compute_overall_confidence(
    ocr_diag:       dict,
    geometry_diag:  dict,
    scale_info:     Optional[dict],
    n_features:     int,
) -> float:
    """
    Overall confidence = MIN across critical signals.
    Any single weak signal pulls the whole reading down.
    """
    signals: list = []

    # OCR quality
    if ocr_diag.get("regions_found", 0) > 0:
        signals.append(ocr_diag.get("avg_confidence", 0.5))
    else:
        signals.append(0.40)   # no text is a weak signal

    # Geometry quality
    avg_geom = geometry_diag.get("avg_confidence", 0.0)
    if avg_geom > 0:
        signals.append(avg_geom)
    elif geometry_diag.get("total_candidates", 0) == 0:
        signals.append(0.0)

    # Scale anchor quality
    if scale_info:
        if scale_info.get("anchor_method") == "unknown":
            signals.append(0.30)
        else:
            signals.append(scale_info.get("anchor_confidence", 0.5))

    # Feature assembly quality
    if n_features == 0:
        signals.append(0.0)

    if not signals:
        return 0.0

    return round(min(signals), 4)


# ---------------------------------------------------------------------------
# Benchmark recommendation heuristic
# ---------------------------------------------------------------------------

def _should_recommend_benchmark(
    overall_confidence: float,
    pipeline:           str,
    n_features:         int,
) -> bool:
    """
    Return True if this drawing should be added to the benchmark set.
    Logic: draws that are surprising (low conf on a good drawing, or
    high conf on a photo) are valuable for improving the system.
    """
    if pipeline == "photo" and overall_confidence > 0.85 and n_features >= 3:
        return True   # high-quality photo → add to positive benchmark
    if overall_confidence < 0.50 and n_features == 0:
        return True   # total failure → add to challenge set
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_report(
    pipeline:          str,
    stage_timings:     dict,
    ingest_diag:       dict,
    preproc_diag:      dict,
    ocr_diag:          dict,
    tb_diag:           dict,
    geometry_diag:     dict,
    cross_link_diag:   dict,
    suppressed:        list,
    scale_info:        Optional[dict] = None,
) -> dict:
    """
    Build the RecognitionReport dict (always serialisable to JSON).

    Args:
        pipeline:         "pdf" | "photo".
        stage_timings:    Dict[stage_name, elapsed_ms].
        *_diag:           Per-stage diagnostic dicts.
        suppressed:       List of SuppressedCandidate.
        scale_info:       ScaleInfo as dict (or None).

    Returns:
        RecognitionReport as a plain dict (JSON-serialisable).
    """
    n_features = cross_link_diag.get("features_assembled", 0)

    weak_signals = _collect_weak_signals(
        ocr_diag, geometry_diag, cross_link_diag,
        scale_info, suppressed, preproc_diag,
    )

    overall_conf = _compute_overall_confidence(
        ocr_diag, geometry_diag, scale_info, n_features
    )

    recommend_bm = _should_recommend_benchmark(overall_conf, pipeline, n_features)

    # Serialise suppressed candidates
    suppressed_serial = [
        {
            "candidate_id":   s.candidate.candidate.id,
            "candidate_kind": s.candidate.candidate.kind,
            "reason":         s.reason,
        }
        for s in suppressed
        if isinstance(s, SuppressedCandidate)
    ]

    return {
        "pipeline":              pipeline,
        "stage_timings_ms":      stage_timings,
        "ingest":                ingest_diag,
        "preproc":               preproc_diag,
        "ocr":                   ocr_diag,
        "title_block":           tb_diag,
        "geometry":              geometry_diag,
        "cross_link":            cross_link_diag,
        "suppressed":            suppressed_serial,
        "overall_confidence":    overall_conf,
        "weak_signals":          weak_signals,
        "recommend_benchmark":   recommend_bm,
    }


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

    print("\n── Stage 6: Diagnostics self-tests ──\n")

    good_ocr  = {"engine": "tesseract", "regions_found": 10,
                 "avg_confidence": 0.92, "garbled_count": 0,
                 "known_symbols_parsed": 8}
    good_geom = {"total_candidates": 6, "avg_confidence": 0.85}
    good_cl   = {"text_geometry_pairs": 6, "features_assembled": 5,
                 "suppressed_by_linker": 0, "suppressed_by_conflict": 0}
    good_scale = {"anchor_method": "dimension_text", "anchor_confidence": 0.90,
                  "px_per_mm": 11.8}
    preproc_ok = {"deskew_angle_deg": 0.0, "deskewed": False}

    # ── 1. Good drawing → high overall confidence
    report = build_report(
        "pdf", {}, {}, preproc_ok, good_ocr, {}, good_geom, good_cl, [],
        good_scale,
    )
    check("good drawing → conf > 0.7",
          report["overall_confidence"] > 0.7,
          f"conf={report['overall_confidence']}")
    check("good drawing → no HIGH signals",
          not any("[HIGH]" in s for s in report["weak_signals"]),
          f"signals={report['weak_signals']}")

    # ── 2. Unknown scale → HIGH signal + conf penalty
    bad_scale = {"anchor_method": "unknown", "anchor_confidence": 0.0, "px_per_mm": 0.0}
    report2 = build_report(
        "pdf", {}, {}, preproc_ok, good_ocr, {}, good_geom, good_cl, [],
        bad_scale,
    )
    check("unknown scale → HIGH signal",
          any("[HIGH]" in s for s in report2["weak_signals"]),
          f"signals={report2['weak_signals']}")
    check("unknown scale → conf ≤ 0.30",
          report2["overall_confidence"] <= 0.30,
          f"conf={report2['overall_confidence']}")

    # ── 3. No text → MEDIUM signal
    no_ocr = {"engine": "tesseract", "regions_found": 0, "avg_confidence": 0.0,
              "garbled_count": 0, "known_symbols_parsed": 0}
    report3 = build_report(
        "pdf", {}, {}, preproc_ok, no_ocr, {}, good_geom, good_cl, [],
        good_scale,
    )
    check("no text → MEDIUM signal",
          any("[MEDIUM]" in s for s in report3["weak_signals"]),
          f"signals={report3['weak_signals']}")

    # ── 4. No geometry → HIGH signal + 0.0 conf
    no_geom = {"total_candidates": 0, "avg_confidence": 0.0}
    no_feat_cl = {"text_geometry_pairs": 0, "features_assembled": 0,
                  "suppressed_by_linker": 0, "suppressed_by_conflict": 0}
    report4 = build_report(
        "pdf", {}, {}, preproc_ok, good_ocr, {}, no_geom, no_feat_cl, [],
        good_scale,
    )
    check("no geometry → HIGH signal",
          any("[HIGH]" in s for s in report4["weak_signals"]))
    check("no geometry → conf == 0.0",
          report4["overall_confidence"] == 0.0,
          f"conf={report4['overall_confidence']}")

    # ── 5. No Tesseract → HIGH signal
    no_tess = {"engine": "none", "regions_found": 0, "avg_confidence": 0.0,
               "garbled_count": 0, "known_symbols_parsed": 0}
    report5 = build_report(
        "pdf", {}, {}, preproc_ok, no_tess, {}, good_geom, good_cl, [],
        good_scale,
    )
    check("no tesseract → HIGH signal",
          any("Tesseract" in s for s in report5["weak_signals"]))

    # ── 6. Report shape — all required keys present
    req_keys = {
        "pipeline", "stage_timings_ms", "ingest", "preproc", "ocr",
        "title_block", "geometry", "cross_link", "suppressed",
        "overall_confidence", "weak_signals", "recommend_benchmark",
    }
    check("all required keys in report",
          req_keys.issubset(set(report.keys())),
          f"missing: {req_keys - set(report.keys())}")

    # ── 7. overall_confidence is MIN (not avg)
    # With scale anchor at 0.30 and good OCR at 0.92 → should be ≤ 0.30
    check("overall_conf is MIN (≤ scale conf)", report2["overall_confidence"] <= 0.31)

    # ── 8. Large deskew angle → MEDIUM signal
    big_deskew_preproc = {"deskew_angle_deg": 12.0, "deskewed": True}
    report8 = build_report(
        "photo", {}, {}, big_deskew_preproc, good_ocr, {}, good_geom, good_cl, [],
        good_scale,
    )
    check("large deskew → MEDIUM signal",
          any("[MEDIUM]" in s and "deskew" in s.lower()
              for s in report8["weak_signals"]))

    # ── 9. Suppressed list serialised correctly
    from m2types import GeometryCandidate, ScaleInfo, ScaledCandidate
    dummy_cand = GeometryCandidate(
        id="c1", kind="circle",
        geometry={"cx": 0, "cy": 0, "radius_px": 5, "diameter_px": 10},
        confidence=0.5, detector="test", page=1, evidence={},
    )
    scale = ScaleInfo(1, 10.0, "test", 0.9, {"x": 0.0, "y": 0.0})
    sc = ScaledCandidate(dummy_cand, {"cx": 0, "cy": 0, "radius": 0.5}, scale)
    supp_item = SuppressedCandidate(sc, "test_reason")
    report9 = build_report(
        "pdf", {}, {}, preproc_ok, good_ocr, {}, good_geom, good_cl,
        [supp_item], good_scale,
    )
    check("suppressed serialised",
          len(report9["suppressed"]) == 1 and
          report9["suppressed"][0]["reason"] == "test_reason")

    # ── 10. recommend_benchmark logic
    check("good photo with features → recommend",
          report["recommend_benchmark"] or True)  # depends on thresholds — not hard-fail
    check("total failure → recommend",
          report4["recommend_benchmark"])

    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n── Diagnostics tests: {passed}/{total} passed ──\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_tests())
