"""
stage1_preprocess/photo_pipeline.py — Module 2: Stage 1b
=========================================================
Photo preprocessing pipeline. More aggressive than PDF pipeline because
phone photos have uneven lighting, perspective, rotation, and noise.

Pipeline:
    1. Grayscale (pass-through — Stage 0 already grayscale + EXIF-corrected)
    2. Border crop — find largest rectangular contour (drawing frame)
    3. Autolevels (1st/99th percentile)
    4. Adaptive threshold (not Otsu — uneven lighting)
    5. Light median denoise (kernel=3) — phone camera noise
    6. Conservative deskew — projection-profile variance, max ±15°, 0.4° threshold

Architecture rules from forensic report §6.3:
    - Projection-profile variance method (not Hough lines — matches old project's
      empirically tuned approach)
    - 15° maximum rotation cap (larger = likely intentional orientation or
      severe distortion not fixable by simple rotation)
    - 0.4° minimum threshold (skip rotation when near-straight)
    - Variance gate: if best_variance < 1.0, rotation signal too weak → skip
    - Record deskewed=True/False + angle in diagnostics unconditionally

Empirical constants:
    max_angle        = 15.0° (from old project preprocessor.py, tuned)
    angle_threshold  = 0.4°  (from old project preprocessor.py, tuned)
    n_projections    = 181   (grid of candidate angles to test)
    variance_floor   = 1.0   (blank-page guard)
    morph_kernel     = 3
    morph_iterations = 1

Public API:
    preprocess_photo(page: PageRaster) -> PreprocessedImage
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import PageRaster, PreprocessedImage  # noqa: E402


# ---------------------------------------------------------------------------
# Constants (empirical — from old project + forensic analysis)
# ---------------------------------------------------------------------------

_MAX_DESKEW_ANGLE: float = 15.0       # degrees — hard cap
_DESKEW_THRESHOLD: float = 0.4        # degrees — minimum to bother rotating
_N_PROJECTIONS:    int   = 181        # candidate angles to test
_VARIANCE_FLOOR:   float = 1.0        # blank/uniform image guard
_MEDIAN_KERNEL:    int   = 3          # denoise kernel size (must be odd)
_ADAPTIVE_BLOCK:   int   = 15         # adaptive threshold block size (odd)
_ADAPTIVE_C:       int   = 8          # adaptive threshold constant subtracted

# Border-crop: minimum fraction of image area a contour must have to be
# considered the drawing frame (prevents false positives from annotations)
_CROP_MIN_AREA_FRAC: float = 0.30
_CROP_APPROX_EPS:    float = 0.02     # approxPolyDP epsilon factor


# ---------------------------------------------------------------------------
# Step 1 — Border crop
# ---------------------------------------------------------------------------

def _detect_drawing_border(gray: np.ndarray) -> dict | None:
    """
    Detect the drawing's outer frame (largest rectangular contour).

    Looks for the rectangle that most plausibly frames the entire drawing
    by finding the largest contour with ≈4 vertices that covers at least
    _CROP_MIN_AREA_FRAC of the image.

    Returns:
        {x, y, w, h} bounding box in pixels, or None if no clear border found.
    """
    h, w = gray.shape
    min_area = _CROP_MIN_AREA_FRAC * h * w

    # Threshold to find large regions
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: dict | None = None
    best_area = 0.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        peri = float(cv2.arcLength(contour, closed=True))
        eps = _CROP_APPROX_EPS * peri
        approx = cv2.approxPolyDP(contour, eps, closed=True)
        if len(approx) != 4:
            continue
        if area > best_area:
            best_area = area
            bx, by, bw, bh = cv2.boundingRect(approx)
            best = {"x": int(bx), "y": int(by), "w": int(bw), "h": int(bh)}

    return best


def _apply_crop(gray: np.ndarray, bbox: dict) -> np.ndarray:
    """Crop to the detected drawing border. Guard against out-of-bounds."""
    h, w = gray.shape
    x = max(0, bbox["x"])
    y = max(0, bbox["y"])
    x2 = min(w, x + bbox["w"])
    y2 = min(h, y + bbox["h"])
    if x2 - x < 10 or y2 - y < 10:
        return gray           # crop too small — return full image
    return gray[y:y2, x:x2]


# ---------------------------------------------------------------------------
# Step 2 — Autolevels + adaptive threshold + denoise
# ---------------------------------------------------------------------------

def _autolevels(gray: np.ndarray) -> np.ndarray:
    """Histogram stretch at 1st/99th percentile."""
    p_low  = int(np.percentile(gray, 1))
    p_high = int(np.percentile(gray, 99))
    if p_high <= p_low:
        return gray
    stretched = np.clip(gray, p_low, p_high)
    return ((stretched - p_low) / (p_high - p_low) * 255).astype(np.uint8)


def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian threshold for uneven illumination (typical photo case).
    Inverted: ink → 255, paper → 0.
    Block size and C constant empirically tuned.
    """
    block = _ADAPTIVE_BLOCK
    if block % 2 == 0:
        block += 1          # must be odd
    # Ensure min block size relative to image
    if min(gray.shape) < block:
        # Fall back to Otsu for tiny images
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        _ADAPTIVE_C,
    )
    return binary


def _median_denoise(binary: np.ndarray) -> np.ndarray:
    """Light median filter to remove phone camera impulse noise."""
    k = _MEDIAN_KERNEL
    if k % 2 == 0:
        k += 1
    return cv2.medianBlur(binary, k)


# ---------------------------------------------------------------------------
# Step 3 — Projection-profile deskew
# ---------------------------------------------------------------------------

def detect_rotation_angle(
    gray: np.ndarray,
    *,
    max_angle: float = _MAX_DESKEW_ANGLE,
    n_projections: int = _N_PROJECTIONS,
) -> tuple[float, float]:
    """
    Estimate skew angle via horizontal projection profile variance maximisation.

    Method (from old project preprocessor.py, empirically tuned):
        Technical drawings contain many horizontal dimension lines and text
        baselines. When correctly oriented, summing pixel intensities row by
        row produces a striped profile with high variance. Rotating by the
        wrong angle smears these stripes and reduces variance.

        We test n_projections candidate angles in [-max_angle, +max_angle],
        compute row-sum variance for each rotation, and pick the angle that
        maximises variance.

    Returns:
        (angle_deg, best_variance)
        angle_deg = 0.0 if variance signal too weak (blank / uniform image).
    """
    max_angle = min(abs(max_angle), _MAX_DESKEW_ANGLE)

    # Downscale to 600 px wide for speed (projection analysis doesn't need detail)
    scale = min(1.0, 600.0 / max(gray.shape[1], 1))
    if scale < 1.0:
        small = cv2.resize(
            gray,
            (int(gray.shape[1] * scale), int(gray.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = gray.copy()

    # Invert: ink → high values, paper → low
    inverted = (255.0 - small.astype(np.float32))

    angles = np.linspace(-max_angle, max_angle, n_projections)

    best_angle    = 0.0
    best_variance = -1.0

    # Convert to PIL for rotation (scipy available in most envs but PIL is lighter)
    from PIL import Image as _PILImage
    inv_img = _PILImage.fromarray(inverted.astype(np.float32))

    for angle in angles:
        rotated = inv_img.rotate(float(angle), resample=_PILImage.BICUBIC,
                                 expand=False, fillcolor=0)
        rot_arr = np.array(rotated, dtype=np.float32)
        row_sums = rot_arr.sum(axis=1)
        var = float(np.var(row_sums))
        if var > best_variance:
            best_variance = var
            best_angle = float(angle)

    # If variance is negligible — blank or uniform image — return 0
    if best_variance < _VARIANCE_FLOOR:
        return 0.0, best_variance

    return best_angle, best_variance


def _deskew(gray: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Correct skew in a photo of a technical drawing.

    Rules (forensic report Q5 / architecture §A1):
        - Test range: ±15° maximum
        - Skip if |angle| < 0.4° (below threshold)
        - Skip if variance < 1.0 (blank / uniform image)
        - Fill exposed areas with white (255)
        - expand=False: downstream coordinate systems stay valid

    Returns:
        (corrected_array, angle_applied, variance)
        angle_applied = 0.0 if no rotation was performed.
    """
    angle, variance = detect_rotation_angle(gray)

    if abs(angle) < _DESKEW_THRESHOLD:
        return gray, 0.0, variance

    # Rotate by negative angle to counteract the detected tilt
    h, w = gray.shape
    centre = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(centre, -angle, scale=1.0)
    corrected = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,           # white fill
    )
    return corrected, angle, variance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_photo(page: PageRaster) -> PreprocessedImage:
    """
    Preprocess one photo raster for geometry and OCR detection.

    Args:
        page: PageRaster from Stage 0 (grayscale uint8, EXIF already applied).

    Returns:
        PreprocessedImage with binary array (ink=255, paper=0) and diagnostics.
    """
    gray = page.image_array.astype(np.uint8)
    crop_bbox: dict | None = None

    # Step 1: Border crop (best-effort — not fatal if none found)
    detected_border = _detect_drawing_border(gray)
    if detected_border is not None:
        gray = _apply_crop(gray, detected_border)
        crop_bbox = detected_border

    # Step 2: Autolevels
    gray = _autolevels(gray)

    # Step 3: Deskew (projection variance, conservative)
    gray, deskew_angle, deskew_variance = _deskew(gray)
    deskewed = abs(deskew_angle) >= _DESKEW_THRESHOLD

    # Step 4: Adaptive threshold
    binary = _adaptive_threshold(gray)

    # Step 5: Median denoise
    binary = _median_denoise(binary)

    h, w = binary.shape

    return PreprocessedImage(
        image_array=binary,
        width_px=w,
        height_px=h,
        page_number=page.page_number,
        pipeline="photo",
        deskewed=deskewed,
        deskew_angle_deg=float(deskew_angle),
        deskew_variance=float(deskew_variance),
        threshold_method="adaptive",
        denoise_applied=True,
        crop_bbox=crop_bbox,
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

    print("\n── Stage 1b: Photo Preprocess self-tests ──\n")

    def make_drawing(h: int = 400, w: int = 600,
                     bg: int = 245, tilted_deg: float = 0.0) -> np.ndarray:
        """Synthetic photo: light background + dark horizontal lines."""
        arr = np.full((h, w), bg, dtype=np.uint8)
        for y in [60, 120, 200, 280, 350]:
            arr[max(0, y - 1): min(h, y + 2), 20: w - 20] = 30
        for x in [100, 200, 300, 400, 500]:
            arr[50: h - 20, max(0, x - 1): min(w, x + 2)] = 30
        if abs(tilted_deg) > 0.01:
            centre = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(centre, tilted_deg, 1.0)
            arr = cv2.warpAffine(arr, M, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=245)
        return arr

    def make_page(arr: np.ndarray, page_num: int = 1) -> PageRaster:
        h, w = arr.shape
        return PageRaster(page_num, arr, w, h, 0.0)

    # ── 1. Output type and pipeline tag
    page = make_page(make_drawing())
    result = preprocess_photo(page)
    check("output is PreprocessedImage",    isinstance(result, PreprocessedImage))
    check("pipeline == 'photo'",            result.pipeline == "photo")
    check("threshold_method == 'adaptive'", result.threshold_method == "adaptive")
    check("denoise_applied == True",        result.denoise_applied == True)

    # ── 2. Binary output
    unique_vals = set(np.unique(result.image_array).tolist())
    check("binary output (0 and 255 only)", unique_vals.issubset({0, 255}),
          f"found: {unique_vals}")

    # ── 3. Lines produce foreground
    fg_ratio = float(result.image_array.sum()) / (255 * result.width_px * result.height_px)
    check("drawing lines produce foreground", 0.001 < fg_ratio < 0.5,
          f"fg_ratio={fg_ratio:.4f}")

    # ── 4. Straight drawing → near-zero deskew angle
    straight = make_page(make_drawing())
    r_straight = preprocess_photo(straight)
    check("straight drawing → deskew angle < 1°",
          abs(r_straight.deskew_angle_deg) < 1.0,
          f"angle={r_straight.deskew_angle_deg:.2f}°")

    # ── 5. Tilted 4° drawing → deskew detects rotation
    tilted = make_page(make_drawing(tilted_deg=4.0))
    r_tilted = preprocess_photo(tilted)
    check("4° tilt → non-zero angle detected",
          abs(r_tilted.deskew_angle_deg) > 0.3,
          f"angle={r_tilted.deskew_angle_deg:.2f}°")

    # ── 6. Blank page → deskew skipped (variance too low)
    blank_arr = np.full((400, 600), 255, dtype=np.uint8)
    r_blank = preprocess_photo(make_page(blank_arr))
    check("blank page → deskewed == False",  r_blank.deskewed == False)
    check("blank page → angle == 0.0",       r_blank.deskew_angle_deg == 0.0)

    # ── 7. detect_rotation_angle — angle in [-15, 15]
    arr3 = make_drawing(tilted_deg=3.0)
    angle_detected, _ = detect_rotation_angle(arr3)
    check("rotation angle within ±15°",
          -15.0 <= angle_detected <= 15.0,
          f"angle={angle_detected:.2f}°")

    # ── 8. page_number preserved
    p7 = make_page(make_drawing(), page_num=7)
    r7 = preprocess_photo(p7)
    check("page_number preserved",           r7.page_number == 7)

    # ── 9. uint8 output dtype
    check("output dtype is uint8",           result.image_array.dtype == np.uint8)

    # ── 10. Autolevels: faded scan expands range
    faded = np.full((200, 300), 200, dtype=np.uint8)
    faded[80:120, 50:250] = 175
    r_faded = preprocess_photo(make_page(faded))
    check("faded scan produces some foreground", r_faded.image_array.any())

    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n── Stage 1b tests: {passed}/{total} passed ──\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_tests())
