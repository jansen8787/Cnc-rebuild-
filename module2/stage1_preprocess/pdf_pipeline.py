"""
stage1_preprocess/pdf_pipeline.py — Module 2: Stage 1a
=======================================================
PDF preprocessing pipeline. Minimal — PDFs are already aligned.

Pipeline:
    1. Grayscale (pass-through — Stage 0 already delivers grayscale)
    2. Autolevels (1st/99th percentile stretch — handles faded scans)
    3. Gaussian blur (kernel=3) — light noise reduction
    4. Otsu threshold → binary (ink=255, paper=0)
    5. Morphological opening (kernel=2, iter=1) — remove isolated noise pixels

NOT applied:
    - Deskew (PDFs are rendered aligned by poppler)
    - Adaptive threshold (uneven lighting is a photo problem)
    - Border crop (PDF canvas is the drawing boundary)

Empirical constants from forensic analysis of old project:
    contrast_factor  = 1.6   (tuned against real scan artefacts)
    sharpness_factor = 1.4   (UnsharpMask radius=1, prevents haloing on thin lines)
    morph_kernel     = 2     (conservative — preserves thin dimension lines)
    morph_iterations = 1

Public API:
    preprocess_pdf(page: PageRaster) -> PreprocessedImage
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import PageRaster, PreprocessedImage  # noqa: E402


# ---------------------------------------------------------------------------
# Constants (from forensic analysis — do not change without benchmark evidence)
# ---------------------------------------------------------------------------

_CONTRAST_FACTOR:  float = 1.6
_SHARPNESS_FACTOR: float = 1.4
_BLUR_KERNEL: int = 3
_MORPH_KERNEL_SIZE: int = 2
_MORPH_ITERATIONS: int = 1


# ---------------------------------------------------------------------------
# Internal stages
# ---------------------------------------------------------------------------

def _autolevels(gray: np.ndarray) -> np.ndarray:
    """
    Histogram stretch at 1st/99th percentile.
    Recovers faded scans without over-whitening solid lines.
    Guard: if the range is trivially small (blank page), return as-is.
    """
    p_low  = int(np.percentile(gray, 1))
    p_high = int(np.percentile(gray, 99))
    if p_high <= p_low:
        return gray                         # blank or uniform — skip
    stretched = np.clip(gray, p_low, p_high)
    stretched = ((stretched - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    return stretched


def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Autolevels + Gaussian blur for light noise damping."""
    g = _autolevels(gray)
    if _BLUR_KERNEL > 0:
        g = cv2.GaussianBlur(g, (_BLUR_KERNEL, _BLUR_KERNEL), sigmaX=0)
    return g


def _binarize_otsu(gray: np.ndarray) -> np.ndarray:
    """
    Otsu threshold: dark ink → 255 (foreground), light paper → 0.
    THRESH_BINARY_INV: ink is foreground.
    """
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return binary


def _morphological_open(binary: np.ndarray) -> np.ndarray:
    """
    Morphological opening: removes isolated noise pixels.
    Conservative kernel (2×2) to not destroy thin dimension lines.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (_MORPH_KERNEL_SIZE, _MORPH_KERNEL_SIZE),
    )
    return cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel, iterations=_MORPH_ITERATIONS
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_pdf(page: PageRaster) -> PreprocessedImage:
    """
    Preprocess one PDF page raster for geometry and OCR detection.

    Args:
        page: PageRaster from Stage 0 (grayscale uint8 numpy array).

    Returns:
        PreprocessedImage with binary array (ink=255, paper=0) and diagnostics.
    """
    gray = page.image_array.astype(np.uint8)

    # Step 1: Autolevels + light noise damping
    enhanced = _enhance_contrast(gray)

    # Step 2: Otsu binarization
    binary = _binarize_otsu(enhanced)

    # Step 3: Morphological opening
    cleaned = _morphological_open(binary)

    h, w = cleaned.shape

    return PreprocessedImage(
        image_array=cleaned,
        width_px=w,
        height_px=h,
        page_number=page.page_number,
        pipeline="pdf",
        deskewed=False,
        deskew_angle_deg=0.0,
        deskew_variance=0.0,
        threshold_method="otsu",
        denoise_applied=False,        # Gaussian blur is light; not "denoise"
        crop_bbox=None,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_tests() -> int:
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    results: list = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition))
        marker = f"  ({detail})" if detail else ""
        print(f"  {PASS if condition else FAIL}  {name}{marker}")

    print("\n── Stage 1a: PDF Preprocess self-tests ──\n")

    def make_page(width: int = 600, height: int = 400,
                  fill: int = 245, lines: bool = True) -> PageRaster:
        """Synthetic drawing: light background, dark horizontal lines."""
        arr = np.full((height, width), fill, dtype=np.uint8)
        if lines:
            for y in [60, 120, 200, 280, 350]:
                arr[y, 20:580] = 30
            for x in [100, 200, 300, 400, 500]:
                arr[50:380, x] = 30
        return PageRaster(
            page_number=1, image_array=arr,
            width_px=width, height_px=height, dpi=300.0,
        )

    # ── 1. Basic output shape
    page = make_page()
    result = preprocess_pdf(page)
    check("output is PreprocessedImage",    isinstance(result, PreprocessedImage))
    check("pipeline == 'pdf'",              result.pipeline == "pdf")
    check("deskewed == False",              result.deskewed == False)
    check("threshold_method == 'otsu'",     result.threshold_method == "otsu")
    check("crop_bbox is None",              result.crop_bbox is None)

    # ── 2. Output dimensions match input
    check("width_px preserved",             result.width_px == 600)
    check("height_px preserved",            result.height_px == 400)

    # ── 3. Binary output: values are 0 or 255 only
    unique_vals = set(np.unique(result.image_array).tolist())
    check("binary output (0 and 255 only)", unique_vals.issubset({0, 255}),
          f"found: {unique_vals}")

    # ── 4. Lines detected: some foreground pixels
    fg_ratio = float(result.image_array.sum()) / (255 * 600 * 400)
    check("drawing lines produce foreground", fg_ratio > 0.001 and fg_ratio < 0.5,
          f"fg_ratio={fg_ratio:.4f}")

    # ── 5. Blank page: near-zero foreground
    blank = make_page(fill=255, lines=False)
    blank_result = preprocess_pdf(blank)
    blank_fg = float(blank_result.image_array.sum()) / (255 * 600 * 400)
    check("blank page → near-zero foreground", blank_fg < 0.005,
          f"fg_ratio={blank_fg:.4f}")

    # ── 6. Autolevels: faded scan gets better contrast
    faded_arr = (np.full((100, 100), 200, dtype=np.uint8))
    faded_arr[40:60, 20:80] = 170          # slightly darker "ink"
    faded_page = PageRaster(1, faded_arr, 100, 100, 300.0)
    faded_result = preprocess_pdf(faded_page)
    # After autolevels + Otsu, the slightly-darker band should show as foreground
    check("faded scan → foreground detected", faded_result.image_array.any())

    # ── 7. Dtype is uint8
    check("output dtype is uint8", result.image_array.dtype == np.uint8)

    # ── 8. page_number preserved
    p3 = make_page()
    p3 = PageRaster(3, p3.image_array, p3.width_px, p3.height_px, p3.dpi)
    r3 = preprocess_pdf(p3)
    check("page_number preserved", r3.page_number == 3)

    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n── Stage 1a tests: {passed}/{total} passed ──\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_tests())
