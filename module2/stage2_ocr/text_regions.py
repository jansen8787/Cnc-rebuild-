"""
stage2_ocr/text_regions.py — Module 2: Stage 2 orchestrator
=============================================================
Detects text regions, generates the text mask (for Stage 3), runs OCR,
and returns the full TextAnnotation list for one page.

Key architectural contract (lesson #2 in MODULE_2_ARCHITECTURE.md §2):
    "Text removed before geometry sees the image."

This module does two things:
    1. run_stage2(preprocessed, pdf_text_layer) -> (annotations, text_mask)
    2. build_text_mask(annotations, width, height)  -> TextMask

The text_mask is passed to Stage 3 so geometry detection blanks these regions
before contour finding. Geometry won't mistake "Ø12" as a circle contour.

Text region detection method:
    - For vector PDFs with a text layer: use the layer directly (no OCR needed).
    - For raster (scanned PDF / photo): run full-image Tesseract, use the
      returned bboxes.
    - For both: additionally pad each bbox by _TEXT_PAD pixels to ensure
      even the ink of the annotation itself is masked.

Public API:
    run_stage2(preprocessed, pdf_text_layer, page_num) -> tuple
    build_text_mask(annotations, width, height, pad) -> TextMask
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import (  # noqa: E402
    PreprocessedImage,
    TextAnnotation,
    TextMask,
    ParsedSymbol,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from symbol_parser import parse_symbol  # noqa: E402
from ocr_engine import run_ocr, TESSERACT_AVAILABLE  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEXT_PAD: int = 6     # pixels to expand each text bbox in the mask
_MIN_TEXT_REGION_AREA: int = 25   # px² — ignore sub-pixel noise


# ---------------------------------------------------------------------------
# Vector PDF text layer adapter
# ---------------------------------------------------------------------------

def _annotations_from_pdf_layer(
    text_layer: list,
    page: int,
    dpi: float,
) -> list:
    """
    Convert raw PDF text-layer TextRun dicts to TextAnnotation.

    The text layer contains PDF-coordinate text runs (pt units).
    DPI is used to convert to pixel coords: px = pt_coord * (dpi / 72.0).

    Args:
        text_layer: List of {"text", "x", "y", "w", "h", "page"} dicts in pt.
        page:       1-based page number to filter for.
        dpi:        Render DPI used when rasterising the PDF.

    Returns:
        List of TextAnnotation in pixel coordinates.
    """
    scale = dpi / 72.0 if dpi > 0 else 1.0
    annotations: list = []

    for idx, run in enumerate(text_layer):
        if run.get("page", 1) != page:
            continue
        text = str(run.get("text", "")).strip()
        if not text:
            continue
        x = int(float(run.get("x", 0)) * scale)
        y = int(float(run.get("y", 0)) * scale)
        w = max(4, int(float(run.get("w", 10)) * scale))
        h = max(4, int(float(run.get("h", 8)) * scale))

        parsed = parse_symbol(text)
        ann = TextAnnotation(
            id=f"ann_pdf_p{page}_{idx:04d}",
            page=page,
            raw_text=text,
            parsed=parsed,
            bbox={"x": x, "y": y, "w": w, "h": h},
            ocr_confidence=1.0,   # vector text is authoritative
        )
        annotations.append(ann)

    return annotations


# ---------------------------------------------------------------------------
# Text mask builder
# ---------------------------------------------------------------------------

def build_text_mask(
    annotations: list,
    width: int,
    height: int,
    pad: int = _TEXT_PAD,
) -> TextMask:
    """
    Build a TextMask from a list of TextAnnotation.

    The mask regions are expanded by `pad` pixels on all sides.

    Args:
        annotations: List of TextAnnotation on one page.
        width:       Image width in pixels.
        height:      Image height in pixels.
        pad:         Pixels to expand each region.

    Returns:
        TextMask with page and regions list.
    """
    if not annotations:
        page = 1
    else:
        page = annotations[0].page

    regions: list = []
    for ann in annotations:
        b = ann.bbox
        x = max(0, b["x"] - pad)
        y = max(0, b["y"] - pad)
        x2 = min(width,  b["x"] + b["w"] + pad)
        y2 = min(height, b["y"] + b["h"] + pad)
        if (x2 - x) * (y2 - y) < _MIN_TEXT_REGION_AREA:
            continue
        regions.append({"x": x, "y": y, "w": x2 - x, "h": y2 - y})

    return TextMask(page=page, regions=regions)


def apply_text_mask(
    image_array: np.ndarray,
    mask: TextMask,
) -> np.ndarray:
    """
    Blank out text regions in a binary image (set to 0 = paper/background).

    Args:
        image_array: Binary uint8 (H, W). Modified copy returned.
        mask:        TextMask with pixel regions to blank.

    Returns:
        New array with text regions set to 0.
    """
    masked = image_array.copy()
    h, w = masked.shape[:2]
    for r in mask.regions:
        x1 = max(0, r["x"])
        y1 = max(0, r["y"])
        x2 = min(w, r["x"] + r["w"])
        y2 = min(h, r["y"] + r["h"])
        if x2 > x1 and y2 > y1:
            masked[y1:y2, x1:x2] = 0
    return masked


# ---------------------------------------------------------------------------
# Diagnostics helper
# ---------------------------------------------------------------------------

def _ocr_diagnostics(annotations: list, source: str) -> dict:
    """Produce per-stage OCR diagnostics for Stage 6."""
    if not annotations:
        return {
            "engine": source,
            "regions_found": 0,
            "avg_confidence": 0.0,
            "garbled_count": 0,
            "known_symbols_parsed": 0,
        }
    confidences = [a.ocr_confidence for a in annotations]
    garbled = sum(
        1 for a in annotations if a.parsed.token_type == "unknown"
    )
    known = sum(
        1 for a in annotations if a.parsed.token_type not in ("unknown", "label")
    )
    return {
        "engine": source,
        "regions_found": len(annotations),
        "avg_confidence": round(sum(confidences) / len(confidences), 4),
        "garbled_count": garbled,
        "known_symbols_parsed": known,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_stage2(
    preprocessed: PreprocessedImage,
    pdf_text_layer: list,
    *,
    min_ocr_confidence: float = 0.0,
) -> tuple:
    """
    Run Stage 2 (OCR + symbol parsing) on one preprocessed page.

    Strategy:
        1. If pdf_text_layer has runs for this page → use those (authoritative,
           confidence=1.0). Skip Tesseract.
        2. Otherwise → run Tesseract on the full preprocessed image.

    Returns:
        (annotations: List[TextAnnotation], mask: TextMask, diagnostics: dict)
    """
    t0 = time.monotonic()
    page = preprocessed.page_number

    # ── Path A: vector PDF text layer
    page_runs = [r for r in pdf_text_layer if r.get("page", 1) == page]
    if page_runs:
        annotations = _annotations_from_pdf_layer(
            pdf_text_layer,
            page=page,
            dpi=preprocessed.width_px / 595.0 * 72.0,  # estimate DPI from width
        )
        source = "pdf_text_layer"
    else:
        # ── Path B: Tesseract OCR
        if TESSERACT_AVAILABLE:
            annotations = run_ocr(
                preprocessed.image_array,
                page=page,
                min_confidence=min_ocr_confidence,
            )
            source = "tesseract"
        else:
            annotations = []
            source = "none"

    mask = build_text_mask(
        annotations,
        width=preprocessed.width_px,
        height=preprocessed.height_px,
    )
    diag = _ocr_diagnostics(annotations, source)
    diag["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 1)

    return annotations, mask, diag


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_tests() -> int:
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    results: list = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition))
        marker = f"  ({detail})" if detail else ""
        print(f"  {PASS if condition else FAIL}  {name}{marker}")

    print("\n── Stage 2: Text Regions self-tests ──\n")

    def make_preprocessed(w: int = 400, h: int = 300,
                           page_num: int = 1) -> PreprocessedImage:
        arr = np.zeros((h, w), dtype=np.uint8)
        # Simulate text at (50,50,100,20)
        arr[50:70, 50:150] = 255
        return PreprocessedImage(
            image_array=arr, width_px=w, height_px=h,
            page_number=page_num, pipeline="pdf",
            deskewed=False, deskew_angle_deg=0.0, deskew_variance=0.0,
            threshold_method="otsu", denoise_applied=False, crop_bbox=None,
        )

    # ── 1. build_text_mask — correct regions from annotations
    ann = TextAnnotation(
        id="ann_p1_0001", page=1, raw_text="Ø12",
        parsed=parse_symbol("Ø12"),
        bbox={"x": 50, "y": 30, "w": 40, "h": 15},
        ocr_confidence=0.9,
    )
    mask = build_text_mask([ann], width=400, height=300, pad=5)
    check("mask has 1 region",     len(mask.regions) == 1)
    check("mask region padded",    mask.regions[0]["x"] <= 45)
    check("mask page == 1",        mask.page == 1)

    # ── 2. apply_text_mask — blanks the region
    img = np.ones((300, 400), dtype=np.uint8) * 255
    masked = apply_text_mask(img, mask)
    rx = mask.regions[0]
    region_vals = masked[rx["y"]:rx["y"]+rx["h"], rx["x"]:rx["x"]+rx["w"]]
    check("text region blanked to 0", int(region_vals.max()) == 0)
    # Area outside the region should be unchanged
    outside = masked[0:5, 0:5]
    check("outside region unchanged", int(outside.mean()) == 255)

    # ── 3. PDF text layer adapter
    text_layer = [
        {"text": "Ø12", "x": 100.0, "y": 200.0, "w": 30.0, "h": 12.0, "page": 1},
        {"text": "R5",  "x": 200.0, "y": 150.0, "w": 20.0, "h": 12.0, "page": 1},
        {"text": "M6",  "x": 300.0, "y": 100.0, "w": 20.0, "h": 12.0, "page": 2},
    ]
    pdf_anns = _annotations_from_pdf_layer(text_layer, page=1, dpi=300.0)
    check("pdf layer: 2 annotations for page 1", len(pdf_anns) == 2)
    check("pdf layer: confidence == 1.0",
          all(a.ocr_confidence == 1.0 for a in pdf_anns))
    check("pdf layer: Ø12 parsed as diameter",
          any(a.parsed.token_type == "dimension_diameter" for a in pdf_anns))

    # ── 4. run_stage2 with PDF text layer (no OCR needed)
    pp = make_preprocessed(page_num=1)
    anns, mask2, diag = run_stage2(pp, text_layer, min_ocr_confidence=0.0)
    check("run_stage2 pdf: returns 2 annotations", len(anns) == 2)
    check("run_stage2 pdf: source == pdf_text_layer",
          diag["engine"] == "pdf_text_layer")
    check("run_stage2 pdf: mask has regions", len(mask2.regions) >= 1)

    # ── 5. run_stage2 with no PDF layer → OCR path (may return empty if no Tesseract)
    pp2 = make_preprocessed(page_num=2)
    anns2, mask3, diag2 = run_stage2(pp2, [], min_ocr_confidence=0.0)
    check("run_stage2 raster: returns list", isinstance(anns2, list))
    check("run_stage2 raster: engine set",
          diag2["engine"] in ("tesseract", "none"))

    # ── 6. _ocr_diagnostics — counts
    test_anns = [
        TextAnnotation("a0", 1, "Ø12", parse_symbol("Ø12"),
                       {"x":0,"y":0,"w":10,"h":10}, 0.9),
        TextAnnotation("a1", 1, "???", parse_symbol("???"),
                       {"x":20,"y":0,"w":10,"h":10}, 0.3),
    ]
    diag3 = _ocr_diagnostics(test_anns, "tesseract")
    check("diagnostics: regions_found == 2",      diag3["regions_found"] == 2)
    check("diagnostics: garbled_count == 1",      diag3["garbled_count"] == 1)
    check("diagnostics: known_symbols == 1",      diag3["known_symbols_parsed"] == 1)
    check("diagnostics: avg_confidence",
          0.0 < diag3["avg_confidence"] <= 1.0)

    # ── 7. Empty annotation list → mask with no regions
    empty_mask = build_text_mask([], 400, 300)
    check("empty anns → 0 mask regions", len(empty_mask.regions) == 0)

    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n── Text Regions tests: {passed}/{total} passed ──\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_tests())
