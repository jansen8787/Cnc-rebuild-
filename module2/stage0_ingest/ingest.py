"""
stage0_ingest/ingest.py — Module 2: Stage 0
============================================
File ingestion: detect input type (magic bytes, not extension), apply EXIF
rotation, extract raster pages from PDF or image.

Architecture rules honoured:
- EXIF rotation applied unconditionally (forensic report §A1).
- File type detected from magic bytes, not extension (§A1).
- PDF text layer extracted separately and preserved (Stage 2 short-circuits on it).
- Every page in a PDF = one PageRaster (multi-page support, one part at a time).
- No confidence invented — DPI is reported accurately or as 0.0 / None.

Public API:
    ingest(path: str | Path, *, dpi: int = 300) -> RawInput

Exit codes (when used as main):
    0  success
    1  file not found or unreadable
    2  unsupported format
    3  argument error

Empirical DPI guidance (from old project forensics):
    200 DPI — minimum acceptable for Tesseract on large text
    300 DPI — recommended for most technical drawings (default)
    400 DPI — use for drawings with very small dimension text (< 6pt)
    600 DPI — archival; slow and memory-intensive
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Types defined in the shared types module one level up
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import PageRaster, RawInput  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_DPI = 72
_MAX_DPI = 600
_DEFAULT_DPI = 300

# Magic byte signatures for reliable file type detection
_MAGIC_PDF   = b"%PDF"
_MAGIC_PNG   = b"\x89PNG"
_MAGIC_JPEG  = b"\xff\xd8\xff"
_MAGIC_TIFF_LE = b"II\x2a\x00"
_MAGIC_TIFF_BE = b"MM\x00\x2a"
_MAGIC_GIF   = b"GIF8"

SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"}
)
SUPPORTED_PDF_EXTENSIONS = frozenset({".pdf"})
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS


# ---------------------------------------------------------------------------
# Magic-byte file type detection
# ---------------------------------------------------------------------------

def _read_magic(path: Path, n: int = 8) -> bytes:
    """Read the first n bytes of a file for magic-byte sniffing."""
    with open(path, "rb") as f:
        return f.read(n)


def detect_file_type(path: Path) -> str:
    """
    Detect file type using magic bytes (first 8 bytes).
    Falls back to extension if magic bytes are inconclusive.

    Returns:
        "pdf"   for PDF documents
        "image" for any raster image format

    Raises:
        FileNotFoundError: file does not exist
        ValueError:        format not supported
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        magic = _read_magic(path)
    except OSError as exc:
        raise OSError(f"Cannot read file {path}: {exc}") from exc

    if magic[:4] == _MAGIC_PDF:
        return "pdf"
    if magic[:4] == _MAGIC_PNG:
        return "image"
    if magic[:3] == _MAGIC_JPEG:
        return "image"
    if magic[:4] in (_MAGIC_TIFF_LE, _MAGIC_TIFF_BE):
        return "image"
    if magic[:4] == _MAGIC_GIF:
        return "image"

    # Fall back to extension — documented, not silent
    ext = path.suffix.lower()
    if ext in SUPPORTED_PDF_EXTENSIONS:
        return "pdf"
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"

    raise ValueError(
        f"Unsupported file format: {path}. "
        f"Magic bytes: {magic[:8]!r}. "
        f"Supported: PDF, PNG, JPEG, TIFF, GIF."
    )


# ---------------------------------------------------------------------------
# EXIF orientation handling
# ---------------------------------------------------------------------------

_EXIF_ORIENTATION_TAG = 0x0112  # Tag 274

def _read_exif_orientation(img: Image.Image) -> int:
    """
    Read EXIF orientation tag.

    Returns degrees to rotate clockwise to make the image upright:
        0 → no rotation needed
        90 → rotate 90° CW (phone portrait held sideways)
        180 → rotate 180°
        270 → rotate 270° CW (= 90° CCW)

    Returns 0 if EXIF is absent or orientation tag is missing.
    """
    try:
        exif = img._getexif()  # type: ignore[attr-defined]
        if not exif:
            return 0
        orientation = exif.get(_EXIF_ORIENTATION_TAG, 1)
    except (AttributeError, Exception):
        return 0

    # EXIF orientation values 1–8:
    # 1 = normal, 3 = 180°, 6 = 90° CW, 8 = 90° CCW
    mapping = {
        1: 0,    # normal
        2: 0,    # horizontal flip — treat as normal (rare in drawings)
        3: 180,
        4: 180,  # vertical flip after 180° — treat as 180°
        5: 270,
        6: 90,   # rotated 90° CW (most common phone landscape)
        7: 90,
        8: 270,
    }
    return mapping.get(int(orientation), 0)


def _apply_exif_rotation(img: Image.Image) -> tuple[Image.Image, int]:
    """
    Apply EXIF orientation unconditionally.

    Returns:
        (corrected_image, degrees_rotated)
    Degrees is 0 if no rotation was needed.
    """
    degrees = _read_exif_orientation(img)
    if degrees == 0:
        return img, 0
    # PIL rotate: positive = CCW; we want CW correction
    # CW 90° = CCW 270°, etc.
    pil_angle = {90: 270, 180: 180, 270: 90}[degrees]
    corrected = img.rotate(pil_angle, expand=True)
    return corrected, degrees


# ---------------------------------------------------------------------------
# Image loading (photos / raster images)
# ---------------------------------------------------------------------------

def _pil_to_gray_array(img: Image.Image) -> np.ndarray:
    """
    Convert any PIL Image to a single-channel uint8 numpy array.

    Handles: RGBA (composited onto white), P (palette), CMYK, L, RGB.
    Transparent areas (CAD PNG exports) become white (paper colour).
    """
    if img.mode == "P":
        img = img.convert("RGBA")

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg

    if img.mode != "L":
        img = img.convert("L")

    return np.array(img, dtype=np.uint8)


def _load_image(path: Path) -> RawInput:
    """Load a single raster image file as a one-page RawInput."""
    try:
        img = Image.open(path)
        img.verify()            # detect truncated / corrupt files early
    except UnidentifiedImageError:
        raise UnidentifiedImageError(f"Cannot identify image file: {path}")
    except Exception as exc:
        raise OSError(f"Failed to open image {path}: {exc}") from exc

    # Re-open after verify (PIL requirement)
    img = Image.open(path)

    # Determine original DPI from metadata
    orig_dpi: Optional[float] = None
    try:
        dpi_info = img.info.get("dpi") or img.info.get("jfif_density")
        if dpi_info:
            if isinstance(dpi_info, (tuple, list)) and len(dpi_info) >= 1:
                orig_dpi = float(dpi_info[0])
            else:
                orig_dpi = float(dpi_info)
    except (TypeError, ValueError):
        orig_dpi = None

    # Apply EXIF rotation unconditionally (architecture §A1)
    img, exif_rotation = _apply_exif_rotation(img)

    # Convert to grayscale array
    gray = _pil_to_gray_array(img)
    h, w = gray.shape

    page = PageRaster(
        page_number=1,
        image_array=gray,
        width_px=w,
        height_px=h,
        dpi=float(orig_dpi) if orig_dpi else 0.0,
    )

    return RawInput(
        input_type="photo",
        source_path=str(path.resolve()),
        pages=[page],
        pdf_text_layer=[],
        orig_dpi=orig_dpi,
        exif_rotation_applied=exif_rotation,
        page_count=1,
    )


# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------

def _extract_pdf_text_layer(path: Path) -> list:
    """
    Extract the text layer from a vector PDF.
    Returns a list of TextRun dicts: {text, x, y, w, h, page}.
    Returns [] if the PDF has no text layer (scanned) or pdfminer is unavailable.
    """
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextBox, LTTextLine, LTAnno, LTChar
    except ImportError:
        return []

    runs: list = []
    try:
        for page_num, page_layout in enumerate(extract_pages(str(path)), start=1):
            page_h = float(page_layout.height)
            for element in page_layout:
                if not isinstance(element, LTTextBox):
                    continue
                x0, y0, x1, y1 = (
                    element.x0, element.y0, element.x1, element.y1
                )
                text = element.get_text().strip()
                if not text:
                    continue
                # PDF coordinates: y=0 is bottom; flip to image convention
                runs.append({
                    "text": text,
                    "x": float(x0),
                    "y": float(page_h - y1),
                    "w": float(x1 - x0),
                    "h": float(y1 - y0),
                    "page": page_num,
                })
    except Exception:
        # Text extraction failure is non-fatal — Stage 2 will run OCR
        return []

    return runs


def _load_pdf(path: Path, dpi: int = _DEFAULT_DPI) -> RawInput:
    """Rasterise a PDF into one PageRaster per page."""
    try:
        from pdf2image import convert_from_path
        from pdf2image.exceptions import (
            PDFInfoNotInstalledError,
            PDFPageCountError,
            PDFSyntaxError,
        )
    except ImportError:
        raise ImportError(
            "pdf2image is required for PDF loading. "
            "Install with: pip install pdf2image"
        ) from None

    if dpi < _MIN_DPI or dpi > _MAX_DPI:
        raise ValueError(f"DPI must be in [{_MIN_DPI}, {_MAX_DPI}], got {dpi}")

    try:
        pil_pages = convert_from_path(str(path), dpi=dpi)
    except PDFInfoNotInstalledError:
        raise RuntimeError(
            "poppler is not installed. "
            "Install with: apt-get install poppler-utils  (or brew install poppler)"
        ) from None
    except PDFPageCountError as exc:
        raise ValueError(f"Cannot determine PDF page count: {path}: {exc}") from exc
    except PDFSyntaxError as exc:
        raise ValueError(f"Malformed or encrypted PDF: {path}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"pdf2image failed on {path}: {exc}") from exc

    pages: list = []
    for i, pil_img in enumerate(pil_pages, start=1):
        gray = _pil_to_gray_array(pil_img)
        h, w = gray.shape
        pages.append(PageRaster(
            page_number=i,
            image_array=gray,
            width_px=w,
            height_px=h,
            dpi=float(dpi),
        ))

    # Attempt text layer extraction (non-fatal if unavailable)
    text_layer = _extract_pdf_text_layer(path)

    return RawInput(
        input_type="pdf",
        source_path=str(path.resolve()),
        pages=pages,
        pdf_text_layer=text_layer,
        orig_dpi=float(dpi),
        exif_rotation_applied=0,   # PDFs have no EXIF
        page_count=len(pages),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest(
    path: Union[str, Path],
    *,
    dpi: int = _DEFAULT_DPI,
) -> RawInput:
    """
    Ingest a drawing file and return a RawInput ready for Stage 1.

    Detects file type via magic bytes (not extension).
    Applies EXIF rotation for images unconditionally.
    Extracts PDF text layer when available (vector PDFs).

    Args:
        path:  Path to the source file (PDF, PNG, JPG, TIF).
        dpi:   Render resolution for PDFs (default 300). Images use native resolution.

    Returns:
        RawInput dataclass with pages (rasterised), text layer, and metadata.

    Raises:
        FileNotFoundError:  File does not exist.
        ValueError:         Unsupported format or DPI out of range.
        OSError:            File cannot be read.
        RuntimeError:       pdf2image / poppler failure.
    """
    p = Path(path).resolve()

    if not p.exists():
        raise FileNotFoundError(f"Source file not found: {p}")

    file_type = detect_file_type(p)

    if file_type == "pdf":
        return _load_pdf(p, dpi=dpi)
    else:
        return _load_image(p)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_tests() -> int:
    """Standalone self-tests. No pytest required. Returns exit code."""
    import tempfile, shutil

    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    results: list = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition))
        marker = f"  ({detail})" if detail else ""
        print(f"  {PASS if condition else FAIL}  {name}{marker}")

    def throws(fn, exc_type, name: str) -> None:
        try:
            fn()
            results.append((name, False))
            print(f"  {FAIL}  {name}  (no exception raised)")
        except exc_type:
            results.append((name, True))
            print(f"  {PASS}  {name}")
        except Exception as e:
            results.append((name, False))
            print(f"  {FAIL}  {name}  (wrong exception: {type(e).__name__}: {e})")

    print("\n── Stage 0: Ingest self-tests ──\n")

    tmp = tempfile.mkdtemp(prefix="mod2_ingest_test_")

    try:
        # ── 1. PNG round-trip
        png_path = os.path.join(tmp, "drawing.png")
        img = Image.new("RGB", (200, 150), (240, 240, 240))
        img.save(png_path)

        result = ingest(png_path)
        check("PNG → input_type == photo",   result.input_type == "photo")
        check("PNG → one page",              result.page_count == 1)
        check("PNG → PageRaster shape",      result.pages[0].width_px == 200)
        check("PNG → image_array is ndarray",
              isinstance(result.pages[0].image_array, np.ndarray))
        check("PNG → no exif rotation",      result.exif_rotation_applied == 0)
        check("PNG → source_path absolute",  os.path.isabs(result.source_path))

        # ── 2. JPEG
        jpg_path = os.path.join(tmp, "photo.jpg")
        Image.new("RGB", (300, 200), (200, 180, 160)).save(jpg_path, quality=85)
        r2 = ingest(jpg_path)
        check("JPEG → input_type == photo",  r2.input_type == "photo")
        check("JPEG → page_count == 1",      r2.page_count == 1)

        # ── 3. RGBA PNG — transparent areas become white
        rgba_path = os.path.join(tmp, "cad_export.png")
        rgba_img = Image.new("RGBA", (100, 80), (0, 0, 0, 0))
        rgba_img.save(rgba_path)
        r3 = ingest(rgba_path)
        arr = r3.pages[0].image_array
        check("RGBA → grayscale output",     arr.ndim == 2)
        check("RGBA → transparent → white",  int(arr[0, 0]) >= 200)

        # ── 4. Grayscale PNG
        gray_path = os.path.join(tmp, "gray.png")
        Image.new("L", (120, 90), 180).save(gray_path)
        r4 = ingest(gray_path)
        check("Grayscale PNG → ndarray uint8",
              r4.pages[0].image_array.dtype == np.uint8)

        # ── 5. detect_file_type — extension-mismatch (magic bytes win)
        mis_path = os.path.join(tmp, "drawing.txt")
        with open(mis_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
        dt = detect_file_type(Path(mis_path))
        check("Magic-byte detection (PNG magic in .txt)", dt == "image")

        # ── 6. Missing file → FileNotFoundError
        throws(lambda: ingest(os.path.join(tmp, "ghost.png")),
               FileNotFoundError, "Missing file → FileNotFoundError")

        # ── 7. Unsupported format → ValueError
        svg_path = os.path.join(tmp, "drawing.svg")
        with open(svg_path, "w") as f:
            f.write("<svg/>")
        throws(lambda: detect_file_type(Path(svg_path)),
               ValueError, "SVG → ValueError")

        # ── 8. _read_exif_orientation with no EXIF
        plain = Image.new("RGB", (50, 50), (128, 128, 128))
        rot = _read_exif_orientation(plain)
        check("No EXIF → rotation == 0",     rot == 0)

        # ── 9. _pil_to_gray_array correctness
        white_rgb = Image.new("RGB", (10, 10), (255, 255, 255))
        arr_white = _pil_to_gray_array(white_rgb)
        check("White RGB → gray 255",        int(arr_white[0, 0]) == 255)

        black_rgb = Image.new("RGB", (10, 10), (0, 0, 0))
        arr_black = _pil_to_gray_array(black_rgb)
        check("Black RGB → gray 0",          int(arr_black[0, 0]) == 0)

        # ── 10. PDF test (only if pdf2image + poppler available)
        pdf_ok = False
        try:
            from pdf2image import convert_from_path  # noqa: F401
            pdf_path = os.path.join(tmp, "minimal.pdf")
            # Minimal valid single-page white PDF (no text content)
            pdf_bytes = (
                b"%PDF-1.4\n"
                b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
                b"3 0 obj<</Type/Page/MediaBox[0 0 595 842]/Parent 2 0 R>>endobj\n"
                b"xref\n0 4\n"
                b"0000000000 65535 f\r\n"
                b"0000000009 00000 n\r\n"
                b"0000000058 00000 n\r\n"
                b"0000000115 00000 n\r\n"
                b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
            )
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            rp = ingest(pdf_path, dpi=72)
            check("PDF → input_type == pdf",     rp.input_type == "pdf")
            check("PDF → page_count == 1",       rp.page_count == 1)
            check("PDF → image_array exists",    rp.pages[0].image_array is not None)
            check("PDF → dpi stored",            rp.pages[0].dpi == 72.0)
            pdf_ok = True
        except (ImportError, RuntimeError) as e:
            print(f"  [SKIP] PDF tests (pdf2image/poppler not available: {e})")

        _ = pdf_ok  # suppress unused

        # ── 11. DPI range validation
        png_p = Path(png_path)
        throws(lambda: _load_pdf(png_p, dpi=10), ValueError, "DPI < 72 → ValueError")
        throws(lambda: _load_pdf(png_p, dpi=700), ValueError, "DPI > 600 → ValueError")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n── Stage 0 tests: {passed}/{total} passed ──\n")
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json, time

    if "--selftest" in sys.argv:
        sys.exit(_run_tests())

    parser = argparse.ArgumentParser(
        prog="ingest.py",
        description="Module 2 Stage 0 — Drawing Ingest"
    )
    parser.add_argument("input",  help="Path to PDF, PNG, JPG, or TIF drawing")
    parser.add_argument("--dpi",  type=int, default=_DEFAULT_DPI,
                        help=f"Render DPI for PDFs (default {_DEFAULT_DPI}, range {_MIN_DPI}–{_MAX_DPI})")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not (_MIN_DPI <= args.dpi <= _MAX_DPI):
        print(f"ERROR: --dpi must be {_MIN_DPI}–{_MAX_DPI}", file=sys.stderr)
        sys.exit(3)

    t0 = time.monotonic()
    try:
        result = ingest(args.input, dpi=args.dpi)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed = (time.monotonic() - t0) * 1000

    summary = {
        "input_type":           result.input_type,
        "source_path":          result.source_path,
        "page_count":           result.page_count,
        "orig_dpi":             result.orig_dpi,
        "exif_rotation_applied": result.exif_rotation_applied,
        "pdf_text_runs":        len(result.pdf_text_layer),
        "pages": [
            {"page": p.page_number, "width_px": p.width_px,
             "height_px": p.height_px, "dpi": p.dpi}
            for p in result.pages
        ],
        "elapsed_ms": round(elapsed, 1),
    }

    if not args.quiet:
        print(json.dumps(summary, indent=2))
