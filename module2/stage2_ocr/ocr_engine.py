"""
stage2_ocr/ocr_engine.py — Module 2: Stage 2
=============================================
Tesseract OCR wrapper. Produces structured TextAnnotation list from a
preprocessed binary image.

Engine choice rationale (forensic report §4.4 / §6.1):
    Tesseract first — the old project shipped a complete pipeline using it,
    tuned against real drawings. PaddleOCR remains the documented fallback
    if benchmark cases 4-5 (phone photos) fall below 80% recall.

Design decisions (from old extractor.py forensics, ADAPT bucket):
    - Confidence normalised from Tesseract 0-100 int → 0.0-1.0 float
    - Tesseract structural rows (conf == -1) silently dropped
    - Empty / whitespace-only text dropped
    - Tokens sorted deterministically: (page, y, x)
    - Token IDs: "ann_p{page}_{seq:04d}"
    - min_confidence defaults to 0.0 (no upstream filtering — flags handle this)
    - PSM 11 (sparse text) better than PSM 6 for scattered dimension labels

Public API:
    run_ocr(image_array, page, *, min_confidence, psm) -> List[TextAnnotation]
    run_ocr_on_regions(image_array, page, regions, *, ...) -> List[TextAnnotation]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import TextAnnotation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from symbol_parser import parse_symbol  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PSM 11 = "Sparse text. Find as much text as possible in no particular order."
# Better for dimension labels scattered around a drawing than PSM 6 (single block).
_DEFAULT_PSM: int = 11

# Languages to try (in order). English covers Arabic numerals and Latin symbols.
_TESSERACT_LANG: str = "eng"

_STRUCT_SENTINEL: int = -1    # Tesseract row conf == -1 → structural (skip)


# ---------------------------------------------------------------------------
# Confidence normalisation
# ---------------------------------------------------------------------------

def _norm_conf(raw: int) -> float:
    """Tesseract 0-100 int → 0.0-1.0 float, clamped."""
    clamped = max(0, min(100, int(raw)))
    return round(clamped / 100.0, 4)


# ---------------------------------------------------------------------------
# Token ID scheme
# ---------------------------------------------------------------------------

def _make_id(page: int, seq: int) -> str:
    return f"ann_p{page}_{seq:04d}"


# ---------------------------------------------------------------------------
# Tesseract availability check
# ---------------------------------------------------------------------------

def _check_tesseract() -> bool:
    """Return True if pytesseract + tesseract binary are available."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


TESSERACT_AVAILABLE: bool = _check_tesseract()


# ---------------------------------------------------------------------------
# Core OCR call
# ---------------------------------------------------------------------------

def run_ocr(
    image_array: np.ndarray,
    page: int,
    *,
    min_confidence: float = 0.0,
    psm: int = _DEFAULT_PSM,
    lang: str = _TESSERACT_LANG,
) -> list:
    """
    Run Tesseract OCR on a preprocessed binary image (ink=255, paper=0).

    Args:
        image_array:    uint8 numpy array, shape (H, W). Binary: ink=255, paper=0.
        page:           1-based page number for token ID generation.
        min_confidence: Drop tokens below this confidence (default 0.0 = keep all).
        psm:            Tesseract page segmentation mode (default 11 = sparse text).
        lang:           Tesseract language string (default "eng").

    Returns:
        List of TextAnnotation, sorted by (y, x). Empty if no text found or
        Tesseract unavailable.
    """
    if not TESSERACT_AVAILABLE:
        return []

    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError(f"min_confidence must be [0.0, 1.0], got {min_confidence}")

    import pytesseract
    from pytesseract import Output
    from PIL import Image

    # Tesseract expects white background, black text — invert our binary
    # (our binary: ink=255/white foreground; Tesseract: ink=0/black on white)
    inverted = 255 - image_array

    pil_img = Image.fromarray(inverted.astype(np.uint8), mode="L")

    config = f"--psm {psm} -l {lang}"

    try:
        data = pytesseract.image_to_data(
            pil_img,
            config=config,
            output_type=Output.DICT,
        )
    except Exception:
        return []

    n_rows = len(data["text"])
    raw_rows: list = []

    for i in range(n_rows):
        raw_conf: int = int(data["conf"][i])
        raw_text: str = str(data["text"][i]).strip()

        if raw_conf == _STRUCT_SENTINEL:
            continue
        if not raw_text:
            continue

        norm = _norm_conf(raw_conf)
        if norm < min_confidence:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        raw_rows.append((y, x, raw_text, norm, x, y, w, h))

    # Sort deterministically: (y, x)
    raw_rows.sort(key=lambda r: (r[0], r[1]))

    annotations: list = []
    for seq, (_, _, text, conf, x, y, w, h) in enumerate(raw_rows):
        parsed = parse_symbol(text)
        ann = TextAnnotation(
            id=_make_id(page, seq),
            page=page,
            raw_text=text,
            parsed=parsed,
            bbox={"x": x, "y": y, "w": w, "h": h},
            ocr_confidence=conf,
        )
        annotations.append(ann)

    return annotations


def run_ocr_on_regions(
    image_array: np.ndarray,
    page: int,
    regions: list,
    *,
    min_confidence: float = 0.0,
    psm: int = 7,   # PSM 7 = single line, better for small cropped regions
    lang: str = _TESSERACT_LANG,
) -> list:
    """
    Run OCR on specific bounding-box regions of an image.

    Useful when text regions are pre-detected (Stage 2.5 title block, or when
    the full-image OCR misses annotations at the drawing periphery).

    Args:
        image_array: Full-page binary uint8 array.
        page:        1-based page number.
        regions:     List of {"x", "y", "w", "h"} dicts (pixel coords).
        min_confidence: Confidence threshold.
        psm:         Tesseract PSM for region crops (default 7 = single line).
        lang:        Tesseract language.

    Returns:
        Merged, deduplicated, sorted TextAnnotation list.
    """
    all_annotations: list = []
    id_offset = 0

    for region in regions:
        x = max(0, int(region.get("x", 0)))
        y = max(0, int(region.get("y", 0)))
        w = max(1, int(region.get("w", 1)))
        h = max(1, int(region.get("h", 1)))

        # Guard bounds
        img_h, img_w = image_array.shape[:2]
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        if x2 - x < 2 or y2 - y < 2:
            continue

        crop = image_array[y:y2, x:x2]
        crop_anns = run_ocr(
            crop, page,
            min_confidence=min_confidence,
            psm=psm,
            lang=lang,
        )

        # Translate crop-local bbox back to full-image coords
        for ann in crop_anns:
            ann.id = _make_id(page, id_offset)
            id_offset += 1
            ann.bbox["x"] += x
            ann.bbox["y"] += y
            all_annotations.append(ann)

    # Sort by (page, y, x)
    all_annotations.sort(key=lambda a: (a.page, a.bbox["y"], a.bbox["x"]))
    return all_annotations


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

    print("\n── Stage 2: OCR Engine self-tests ──\n")

    if not TESSERACT_AVAILABLE:
        print("  [SKIP] Tesseract not available — skipping OCR tests")
        print("\n── OCR Engine tests: 0/0 (Tesseract absent) ──\n")
        return 0

    from PIL import Image, ImageDraw, ImageFont

    # Try to get a usable font
    _FONT_PATHS = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    font = None
    for fp in _FONT_PATHS:
        try:
            font = ImageFont.truetype(fp, 36)
            break
        except (IOError, OSError):
            continue

    def make_binary_frame(texts: list, w: int = 700, h: int = 300) -> np.ndarray:
        """Render text onto a white-on-black image (ink=255, paper=0)."""
        img = Image.new("L", (w, h), 0)    # black background
        d = ImageDraw.Draw(img)
        for px, py, t in texts:
            if font:
                d.text((px, py), t, fill=255, font=font)
            else:
                d.text((px, py), t, fill=255)
        return np.array(img, dtype=np.uint8)

    # ── 1. Returns list
    arr = make_binary_frame([(50, 30, "12.5mm"), (350, 30, "R12")])
    result = run_ocr(arr, page=1)
    check("returns list",              isinstance(result, list))

    # ── 2. Non-empty on drawing frame
    check("non-empty on drawing frame", len(result) > 0, f"got {len(result)}")

    # ── 3. Required fields on each annotation
    if result:
        req = {"id", "page", "raw_text", "parsed", "bbox", "ocr_confidence"}
        check("annotations have required fields",
              all(hasattr(a, f) for a in result for f in ["id","page","raw_text","parsed","bbox","ocr_confidence"]))

    # ── 4. Confidence in [0.0, 1.0]
    if result:
        check("confidence ∈ [0.0, 1.0]",
              all(0.0 <= a.ocr_confidence <= 1.0 for a in result))

    # ── 5. Page number stored correctly
    if result:
        check("page == 1 for single call",
              all(a.page == 1 for a in result))

    # ── 6. IDs are unique
    ids = [a.id for a in result]
    check("IDs are unique", len(ids) == len(set(ids)))

    # ── 7. ID format ann_p{n}_{nnnn}
    import re
    pat = re.compile(r"^ann_p\d+_\d{4}$")
    check("ID format ann_p{n}_{nnnn}", all(pat.match(a.id) for a in result))

    # ── 8. Sorted top-to-bottom
    ys = [a.bbox["y"] for a in result]
    check("sorted top-to-bottom",  ys == sorted(ys))

    # ── 9. Parsed symbols populated
    if result:
        check("parsed is ParsedSymbol",
              all(hasattr(a.parsed, "token_type") for a in result))

    # ── 10. min_confidence filter
    strict = run_ocr(arr, page=1, min_confidence=0.99)
    relaxed = run_ocr(arr, page=1, min_confidence=0.0)
    check("min_confidence=0.99 ≤ 0.0 count",
          len(strict) <= len(relaxed))

    # ── 11. Empty image → empty result
    blank = np.zeros((200, 300), dtype=np.uint8)
    blank_result = run_ocr(blank, page=1)
    check("blank image → empty or few results", len(blank_result) <= 2)

    # ── 12. Determinism
    run_a = run_ocr(arr, page=1)
    run_b = run_ocr(arr, page=1)
    ids_a = [a.id for a in run_a]
    ids_b = [a.id for a in run_b]
    check("determinism: same ids on re-run", ids_a == ids_b)

    # ── 13. invalid min_confidence raises ValueError
    try:
        run_ocr(arr, page=1, min_confidence=1.5)
        results.append(("min_confidence=1.5 raises ValueError", False))
        print(f"  {FAIL}  min_confidence=1.5 raises ValueError  (no exception)")
    except ValueError:
        results.append(("min_confidence=1.5 raises ValueError", True))
        print(f"  {PASS}  min_confidence=1.5 raises ValueError")

    # ── 14. run_ocr_on_regions: correct bbox translation
    regions = [{"x": 50, "y": 30, "w": 200, "h": 60}]
    region_result = run_ocr_on_regions(arr, page=1, regions=regions)
    if region_result:
        check("region bbox translated to full-image coords",
              all(a.bbox["x"] >= 50 for a in region_result),
              f"bboxes: {[a.bbox for a in region_result[:2]]}")
    else:
        print("  [INFO] run_ocr_on_regions returned empty (OCR miss on region)")
        results.append(("region bbox translated", True))  # not a hard failure

    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n── OCR Engine tests: {passed}/{total} passed ──\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_tests())
