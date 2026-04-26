"""
stage5_assemble/assemble.py — Module 2: Stage 5
================================================
Orchestrates the full assembly pipeline for one page:
    1. spatial_link  — pair text annotations to geometry candidates
    2. conflict_resolver — remove overlapping duplicate candidates
    3. feature_inferrer — map (candidate + text) → Module 1 V2 Feature

Produces the PartData dict (Module 1 V2 schema) plus diagnostics.

Public API:
    assemble(annotations, scaled_candidates, title_block, page_info)
        -> (part_data: dict, cross_link_diag: dict, suppressed: list)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import (  # noqa: E402
    TextAnnotation,
    ScaledCandidate,
    TitleBlockInfo,
    SuppressedCandidate,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_link       import link_text_to_geometry   # noqa: E402
from conflict_resolver  import resolve_conflicts        # noqa: E402
from feature_inferrer   import infer_features           # noqa: E402


# ---------------------------------------------------------------------------
# Module 1 V2 PartData builder
# ---------------------------------------------------------------------------

def _build_part_data(
    features: list,
    title_block: Optional[TitleBlockInfo],
    page_info: dict,
) -> dict:
    """
    Assemble a Module 1 V2 PartData dict from inferred features.

    Schema reference: module1/src/types.ts (PartData, Feature, ManufacturingMeta).
    """
    # Part name — prefer title block, fall back to source filename
    part_name = (
        (title_block.part_name if title_block else None)
        or page_info.get("source_filename", "unknown")
    )

    # Units — prefer title block hint, fall back to mm
    units = "mm"
    if title_block and title_block.units_hint:
        units = title_block.units_hint

    # Compute overall bounding box from all feature positions
    all_x = [f.get("x", 0.0) for f in features]
    all_y = [f.get("y", 0.0) for f in features]
    all_x2 = [
        f.get("x", 0.0) + f.get("width", f.get("diameter", 0.0))
        for f in features
    ]
    all_y2 = [
        f.get("y", 0.0) + f.get("height", f.get("diameter", 0.0))
        for f in features
    ]

    if features:
        bbox = {
            "x":      round(min(all_x),  4),
            "y":      round(min(all_y),  4),
            "width":  round(max(all_x2) - min(all_x), 4),
            "height": round(max(all_y2) - min(all_y), 4),
        }
    else:
        bbox = {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}

    # Map features to Module 1 V2 Feature shape
    m1_features = []
    for feat in features:
        m1_feat: dict = {
            "id":           feat["id"],
            "kind":         feat["kind"],
            "confidence":   feat["confidence"],
            "source":       feat.get("source", "vision"),
        }

        # Geometry fields — copy all except meta fields
        _skip = {"id", "kind", "confidence", "source", "manufacturing"}
        for k, v in feat.items():
            if k not in _skip:
                m1_feat[k] = v

        # Manufacturing meta
        mfg = feat.get("manufacturing", {})
        if mfg:
            m1_feat["manufacturing"] = mfg

        m1_features.append(m1_feat)

    part_data = {
        "schemaVersion": 2,
        "partName":      part_name,
        "units":         units,
        "bbox":          bbox,
        "zeroPoint":     {"x": 0.0, "y": 0.0},
        "features":      m1_features,
        "_meta": {
            "inputType":      page_info.get("input_type", "unknown"),
            "generatedBy":    "module2_v1.0.0",
            "drawingNumber":  title_block.drawing_number if title_block else None,
            "revision":       title_block.revision       if title_block else None,
            "material":       title_block.material       if title_block else None,
            "scaleRaw":       title_block.scale_raw      if title_block else None,
        },
    }

    return part_data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assemble(
    annotations: list,
    scaled_candidates: list,
    title_block: Optional[TitleBlockInfo] = None,
    page_info: Optional[dict] = None,
    page_area_mm2: float = 0.0,
) -> tuple:
    """
    Full Stage 5 assembly: text ↔ geometry → PartData.

    Args:
        annotations:       List[TextAnnotation] from Stage 2.
        scaled_candidates: List[ScaledCandidate] from Stage 4.
        title_block:       Optional TitleBlockInfo from Stage 2.5.
        page_info:         Dict with source_filename, input_type.
        page_area_mm2:     Drawing page area in mm² (for contour detection).

    Returns:
        (part_data: dict, cross_link_diag: dict, suppressed: List[SuppressedCandidate])
    """
    t0 = time.monotonic()
    page_info = page_info or {}

    # ── Step 1: Spatial link — pair text to geometry
    pairs, link_suppressed = link_text_to_geometry(annotations, scaled_candidates)

    # ── Step 2: Conflict resolution — remove duplicate candidates
    kept_candidates, conflict_suppressed = resolve_conflicts(scaled_candidates, pairs)

    # Remove pairs whose candidate was suppressed
    kept_ids = {sc.candidate.id for sc in kept_candidates}
    pairs = [p for p in pairs if p.candidate.candidate.id in kept_ids]

    # Combine suppressed lists
    all_suppressed: list = link_suppressed + conflict_suppressed

    # ── Step 3: Feature inference — map to Module 1 V2 features
    features = infer_features(
        kept_candidates, pairs, title_block, page_area_mm2
    )

    # ── Step 4: Build PartData dict
    part_data = _build_part_data(features, title_block, page_info)

    elapsed = round((time.monotonic() - t0) * 1000, 1)

    # Cross-link diagnostics
    cross_link_diag = {
        "text_geometry_pairs":    len(pairs),
        "suppressed_by_linker":   len(link_suppressed),
        "suppressed_by_conflict": len(conflict_suppressed),
        "features_assembled":     len(features),
        "elapsed_ms":             elapsed,
    }

    return part_data, cross_link_diag, all_suppressed


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

    print("\n── Stage 5: Assemble self-tests ──\n")

    from m2types import (
        GeometryCandidate, ScaleInfo, ScaledCandidate,
        TextAnnotation, TitleBlockInfo,
    )
    from stage2_ocr.symbol_parser import parse_symbol

    scale = ScaleInfo(1, 10.0, "test", 0.9, {"x": 0.0, "y": 0.0})

    def make_sc(cid: str, kind: str, cx: float, cy: float,
                r: float = 5.0, conf: float = 0.85) -> ScaledCandidate:
        cand = GeometryCandidate(
            id=cid, kind=kind,
            geometry={"cx": cx*10, "cy": cy*10, "radius_px": r*10, "diameter_px": r*20},
            confidence=conf, detector="test", page=1,
            evidence={"area_px": 3.14*(r*10)**2},
        )
        return ScaledCandidate(
            candidate=cand,
            geometry_mm={"cx": cx, "cy": cy, "radius": r, "diameter": r*2},
            scale_info=scale,
        )

    def make_ann(text: str, x: int = 0, y: int = 0) -> TextAnnotation:
        return TextAnnotation(
            id=f"ann_{text}", page=1, raw_text=text,
            parsed=parse_symbol(text),
            bbox={"x": x, "y": y, "w": 40, "h": 12},
            ocr_confidence=0.9,
        )

    sc1 = make_sc("c1", "circle", 10.0, 10.0, r=3.0)
    sc2 = make_sc("c2", "circle", 30.0, 10.0, r=5.0)
    ann1 = make_ann("Ø6", x=100, y=100)

    tb = TitleBlockInfo(
        part_name="Bracket", drawing_number="D-001", revision="A",
        material="AL6061", scale_raw="1:1", units_hint="mm",
        date=None, author=None, sheet=None, tolerance_general="±0.1",
        extra={},
    )

    # ── 1. Basic assembly
    pd, diag, supp = assemble(
        [ann1], [sc1, sc2], tb,
        page_info={"source_filename": "test.pdf", "input_type": "pdf"},
    )

    check("returns part_data dict",       isinstance(pd, dict))
    check("has schemaVersion == 2",       pd.get("schemaVersion") == 2)
    check("has partName from TB",         pd.get("partName") == "Bracket",
          f"got {pd.get('partName')!r}")
    check("has units == mm",              pd.get("units") == "mm")
    check("has bbox",                     isinstance(pd.get("bbox"), dict))
    check("has features list",            isinstance(pd.get("features"), list))
    check("has _meta",                    isinstance(pd.get("_meta"), dict))
    check("_meta has material",           pd["_meta"].get("material") == "AL6061")

    # ── 2. Features are valid Module 1 V2 Features
    for feat in pd["features"]:
        check(f"feature {feat['id']} has kind",
              "kind" in feat)
        check(f"feature {feat['id']} has confidence",
              "confidence" in feat)
        check(f"feature {feat['id']} confidence ∈ [0,1]",
              0.0 <= feat["confidence"] <= 1.0)
        break   # just first feature

    # ── 3. cross_link_diag populated
    check("diag has text_geometry_pairs",    "text_geometry_pairs" in diag)
    check("diag has features_assembled",     "features_assembled"  in diag)
    check("diag has elapsed_ms",             "elapsed_ms" in diag)

    # ── 4. Empty candidates → empty features
    pd2, diag2, _ = assemble([], [], tb)
    check("no candidates → 0 features",  len(pd2["features"]) == 0)

    # ── 5. No annotations → features still assembled (geometry-only)
    pd3, _, _ = assemble([], [sc1, sc2], tb)
    check("no text → features from geometry", len(pd3["features"]) >= 1)

    # ── 6. partName falls back to filename when no title block
    pd4, _, _ = assemble([], [sc1], None,
                         page_info={"source_filename": "part.pdf",
                                    "input_type": "pdf"})
    check("no title block → partName from filename",
          pd4.get("partName") == "part.pdf",
          f"got {pd4.get('partName')!r}")

    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n── Assemble tests: {passed}/{total} passed ──\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_tests())
