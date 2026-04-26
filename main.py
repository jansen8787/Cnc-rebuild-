"""
main.py — Module 2: Drawing Recognition Engine
===============================================
Full pipeline CLI entry point. Runs all six stages and writes output JSON.

Usage:
    python main.py drawing.pdf
    python main.py photo.jpg --output result.json
    python main.py drawing.pdf --dpi 400 --quiet
    python main.py drawing.pdf --selftest

Pipeline:
    Stage 0  Ingest
    Stage 1  Preprocess (PDF or Photo pipeline)
    Stage 2  OCR + Symbol Parser
    Stage 2.5 Title Block
    Stage 3  Geometry Detection
    Stage 4  Scale / Coordinate Transform
    Stage 5  Cross-link + Assembly → PartData
    Stage 6  Diagnostics → RecognitionReport

Output JSON shape:
    {
        "partData":    { ...Module 1 V2 PartData... },
        "diagnostics": { ...RecognitionReport... }
    }

Exit codes:
    0  Success
    1  Input file not found or unreadable
    2  Unsupported format
    3  Argument error (DPI out of range, etc.)
    4  Pipeline error (unexpected exception)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

# Make module-level imports work regardless of working directory
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Stage imports
from stage0_ingest.ingest         import ingest
from stage1_preprocess.pdf_pipeline   import preprocess_pdf
from stage1_preprocess.photo_pipeline import preprocess_photo
from stage2_ocr.text_regions          import run_stage2
from stage25_titleblock.titleblock    import parse_title_block
from stage3_geometry.detector         import detect
from stage2_ocr.text_regions          import apply_text_mask
from stage4_scale.scale_detector      import detect_scale, apply_scale
from stage5_assemble.assemble         import assemble
from stage6_diagnostics.report        import build_report


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DPI        = 300
MIN_DPI            = 72
MAX_DPI            = 600
DEFAULT_OUTPUT     = "module2_output.json"
MODULE_VERSION     = "1.0.0"

EXIT_OK      = 0
EXIT_INPUT   = 1
EXIT_FORMAT  = 2
EXIT_ARGS    = 3
EXIT_PIPELINE = 4


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: Path,
    *,
    dpi: int = DEFAULT_DPI,
    quiet: bool = False,
) -> dict:
    """
    Execute the full Module 2 pipeline on one drawing file.

    Args:
        input_path: Resolved path to the drawing.
        dpi:        Render DPI for PDFs.
        quiet:      Suppress progress output.

    Returns:
        dict with keys "partData" and "diagnostics".

    Raises:
        FileNotFoundError: input file does not exist.
        ValueError:        unsupported format or bad arguments.
        RuntimeError:      pipeline failure.
    """
    timings: dict = {}

    def log(msg: str) -> None:
        if not quiet:
            print(f"  [M2] {msg}")

    log(f"Processing: {input_path.name}")

    # ── Stage 0: Ingest
    t0 = time.monotonic()
    raw_input = ingest(input_path, dpi=dpi)
    timings["stage0_ingest"] = round((time.monotonic() - t0) * 1000, 1)
    log(f"Stage 0 done: {raw_input.input_type}, {raw_input.page_count} page(s)")

    # For now: process page 1 only (multi-page deferred per architecture §8)
    if not raw_input.pages:
        raise RuntimeError("Ingest produced no pages.")
    page = raw_input.pages[0]

    # ── Stage 1: Preprocess
    t1 = time.monotonic()
    if raw_input.input_type == "pdf":
        preprocessed = preprocess_pdf(page)
    else:
        preprocessed = preprocess_photo(page)
    timings["stage1_preprocess"] = round((time.monotonic() - t1) * 1000, 1)
    log(f"Stage 1 done: pipeline={preprocessed.pipeline}, "
        f"deskewed={preprocessed.deskewed} ({preprocessed.deskew_angle_deg:.1f}°)")

    preproc_diag = {
        "pipeline":          preprocessed.pipeline,
        "deskewed":          preprocessed.deskewed,
        "deskew_angle_deg":  preprocessed.deskew_angle_deg,
        "deskew_variance":   preprocessed.deskew_variance,
        "threshold_method":  preprocessed.threshold_method,
        "denoise_applied":   preprocessed.denoise_applied,
        "crop_bbox":         preprocessed.crop_bbox,
        "width_px":          preprocessed.width_px,
        "height_px":         preprocessed.height_px,
    }

    # ── Stage 2: OCR + Symbol Parser
    t2 = time.monotonic()
    annotations, text_mask, ocr_diag = run_stage2(
        preprocessed,
        raw_input.pdf_text_layer,
    )
    timings["stage2_ocr"] = round((time.monotonic() - t2) * 1000, 1)
    log(f"Stage 2 done: {len(annotations)} annotations, "
        f"engine={ocr_diag.get('engine')}, "
        f"avg_conf={ocr_diag.get('avg_confidence', 0):.2f}")

    # ── Stage 2.5: Title Block
    t25 = time.monotonic()
    title_block = parse_title_block(
        annotations,
        preprocessed.width_px,
        preprocessed.height_px,
    )
    timings["stage25_titleblock"] = round((time.monotonic() - t25) * 1000, 1)
    tb_diag = {
        "part_name":      title_block.part_name,
        "material":       title_block.material,
        "scale_raw":      title_block.scale_raw,
        "units_hint":     title_block.units_hint,
        "drawing_number": title_block.drawing_number,
    }
    log(f"Stage 2.5 done: part={title_block.part_name!r}, "
        f"scale={title_block.scale_raw!r}")

    # ── Stage 3: Geometry Detection (on text-masked image)
    t3 = time.monotonic()
    masked_image = apply_text_mask(preprocessed.image_array, text_mask)
    candidates = detect(masked_image, page=preprocessed.page_number)
    timings["stage3_geometry"] = round((time.monotonic() - t3) * 1000, 1)

    avg_geom_conf = (
        round(sum(c.confidence for c in candidates) / len(candidates), 4)
        if candidates else 0.0
    )
    geometry_diag = {
        "total_candidates":     len(candidates),
        "avg_confidence":       avg_geom_conf,
        "kinds":                _count_kinds(candidates),
        "text_mask_regions":    len(text_mask.regions),
    }
    log(f"Stage 3 done: {len(candidates)} candidates, "
        f"kinds={geometry_diag['kinds']}")

    # ── Stage 4: Scale Detection + Apply
    t4 = time.monotonic()
    scale_info = detect_scale(
        annotations,
        candidates,
        preprocessed.width_px,
        preprocessed.height_px,
        title_block=title_block,
        pdf_dpi=page.dpi,
    )
    scaled_candidates = apply_scale(candidates, scale_info)
    timings["stage4_scale"] = round((time.monotonic() - t4) * 1000, 1)
    scale_diag = dataclasses.asdict(scale_info)
    log(f"Stage 4 done: method={scale_info.anchor_method}, "
        f"px_per_mm={scale_info.px_per_mm:.4f}, "
        f"conf={scale_info.anchor_confidence:.2f}")

    # ── Stage 5: Cross-link + Assemble → PartData
    t5 = time.monotonic()
    page_area_mm2 = 0.0
    if scale_info.px_per_mm > 0:
        page_area_mm2 = (preprocessed.width_px / scale_info.px_per_mm) * \
                        (preprocessed.height_px / scale_info.px_per_mm)

    part_data, cross_link_diag, suppressed = assemble(
        annotations,
        scaled_candidates,
        title_block=title_block,
        page_info={
            "source_filename": input_path.name,
            "input_type":      raw_input.input_type,
        },
        page_area_mm2=page_area_mm2,
    )
    timings["stage5_assemble"] = round((time.monotonic() - t5) * 1000, 1)
    log(f"Stage 5 done: {len(part_data['features'])} features assembled")

    # ── Stage 6: Diagnostics
    t6 = time.monotonic()
    ingest_diag = {
        "input_type":             raw_input.input_type,
        "page_count":             raw_input.page_count,
        "orig_dpi":               raw_input.orig_dpi,
        "exif_rotation_applied":  raw_input.exif_rotation_applied,
        "pdf_text_runs":          len(raw_input.pdf_text_layer),
        "source_path":            raw_input.source_path,
    }
    report = build_report(
        pipeline=raw_input.input_type,
        stage_timings=timings,
        ingest_diag=ingest_diag,
        preproc_diag=preproc_diag,
        ocr_diag=ocr_diag,
        tb_diag=tb_diag,
        geometry_diag=geometry_diag,
        cross_link_diag=cross_link_diag,
        suppressed=suppressed,
        scale_info=scale_diag,
    )
    timings["stage6_diagnostics"] = round((time.monotonic() - t6) * 1000, 1)

    total_ms = round(sum(timings.values()), 1)
    log(f"Stage 6 done. Overall confidence: {report['overall_confidence']:.2f}")
    if report["weak_signals"] and not quiet:
        for sig in report["weak_signals"]:
            print(f"  [WARN] {sig}")
    log(f"Total elapsed: {total_ms} ms")

    return {
        "partData":    part_data,
        "diagnostics": report,
    }


def _count_kinds(candidates: list) -> dict:
    counts: dict = {}
    for c in candidates:
        counts[c.kind] = counts.get(c.kind, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Module 2 — CNC Drawing Recognition Engine\n"
            "------------------------------------------\n"
            "Reads a technical drawing (PDF or image) and extracts\n"
            "geometry, dimensions, and text into Module 1 V2 PartData.\n"
        ),
        epilog=(
            "Examples:\n"
            "  python main.py drawing.pdf\n"
            "  python main.py photo.jpg --output result.json\n"
            "  python main.py drawing.pdf --dpi 400 --quiet\n"
        ),
    )
    p.add_argument("input",  metavar="INPUT",
                   help="Path to drawing file (PDF, PNG, JPG, TIF)")
    p.add_argument("--output", "-o", metavar="PATH", default=None,
                   help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--dpi", "-d", type=int, default=DEFAULT_DPI, metavar="N",
                   help=f"PDF render DPI (default {DEFAULT_DPI}, range {MIN_DPI}–{MAX_DPI})")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress progress output; only print errors")
    p.add_argument("--selftest", action="store_true",
                   help=argparse.SUPPRESS)
    return p


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if "--selftest" in argv:
        return _run_pipeline_selftest()

    parser = _build_parser()
    args   = parser.parse_args(argv)

    # Validate DPI
    if not (MIN_DPI <= args.dpi <= MAX_DPI):
        print(f"ERROR: --dpi must be {MIN_DPI}–{MAX_DPI}, got {args.dpi}",
              file=sys.stderr)
        return EXIT_ARGS

    # Resolve input path
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        return EXIT_INPUT

    # Resolve output path
    out_path = Path(args.output).resolve() if args.output else \
               input_path.parent / DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    try:
        result = run_pipeline(input_path, dpi=args.dpi, quiet=args.quiet)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_FORMAT
    except Exception as exc:
        print(f"ERROR: Pipeline failed: {exc}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return EXIT_PIPELINE

    # Write output
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
    except OSError as exc:
        print(f"ERROR: Cannot write output: {exc}", file=sys.stderr)
        return EXIT_PIPELINE

    if not args.quiet:
        n_feat = len(result["partData"].get("features", []))
        conf   = result["diagnostics"].get("overall_confidence", 0.0)
        print(f"  [M2] Output: {out_path}")
        print(f"  [M2] Features: {n_feat}  |  Confidence: {conf:.2f}")

    return EXIT_OK


# ---------------------------------------------------------------------------
# Pipeline self-test (no real drawing needed)
# ---------------------------------------------------------------------------

def _run_pipeline_selftest() -> int:
    """Smoke test using a synthetic minimal PNG image."""
    import tempfile
    import shutil

    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    results: list = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        results.append((name, cond))
        marker = f"  ({detail})" if detail else ""
        print(f"  {PASS if cond else FAIL}  {name}{marker}")

    print("\n── Module 2 Pipeline self-test ──\n")

    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        print(f"  [SKIP] Required library missing: {exc}")
        return 0

    tmp = tempfile.mkdtemp(prefix="mod2_selftest_")

    try:
        # Create a synthetic drawing: white background, circles + rectangles
        img = np.ones((600, 800), dtype=np.uint8) * 240
        cv2.circle(img,    (200, 300), 40, 30, -1)   # filled circle (hole)
        cv2.circle(img,    (400, 300), 20, 30, -1)   # smaller hole
        cv2.rectangle(img, (500, 200), (700, 400), 30, -1)   # pocket
        # Add some text-like noise (not real text — Tesseract will struggle)
        cv2.putText(img, "Ø80", (140, 260), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, 30, 2, cv2.LINE_AA)

        png_path = Path(tmp) / "synthetic_drawing.png"
        pil_img  = Image.fromarray(img)
        pil_img.save(str(png_path))

        # Run the pipeline
        out_path = Path(tmp) / "output.json"
        result_code = main([str(png_path), "--output", str(out_path),
                            "--quiet"])

        check("pipeline exits 0",            result_code == EXIT_OK,
              f"exit_code={result_code}")
        check("output JSON written",          out_path.exists())

        if out_path.exists():
            with open(out_path) as f:
                result = json.load(f)

            check("output has partData",      "partData"    in result)
            check("output has diagnostics",   "diagnostics" in result)
            check("partData schemaVersion 2",
                  result["partData"].get("schemaVersion") == 2)
            check("partData has features",
                  isinstance(result["partData"].get("features"), list))
            check("partData has units",
                  result["partData"].get("units") in ("mm", "in"))
            check("diagnostics has overall_confidence",
                  "overall_confidence" in result["diagnostics"])
            check("diagnostics has pipeline",
                  result["diagnostics"].get("pipeline") == "photo")
            check("diagnostics confidence ∈ [0,1]",
                  0.0 <= result["diagnostics"]["overall_confidence"] <= 1.0)
            check("partData has bbox",
                  isinstance(result["partData"].get("bbox"), dict))
            check("partData _meta present",
                  "_meta" in result["partData"])

        # Test exit code for missing file
        code_missing = main(["ghost_drawing.png", "--quiet"])
        check("missing file → EXIT_INPUT",    code_missing == EXIT_INPUT)

        # Test exit code for bad DPI
        code_dpi = main([str(png_path), "--dpi", "9999", "--quiet"])
        check("bad DPI → EXIT_ARGS",          code_dpi == EXIT_ARGS)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n── Pipeline tests: {passed}/{total} passed ──\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
