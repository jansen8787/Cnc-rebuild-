"""
server.py — CNC AI Backend
===========================
Flask server exposing POST /api/recognize.

Wires directly to /home/claude/module2/main.py::run_pipeline().
No API keys. No external services. Fully local.

Usage:
    python server.py                  # development, port 5000
    python server.py --port 8080
    python server.py --debug          # verbose errors in response

Routes:
    POST /api/recognize
        multipart/form-data:
            file  — drawing file (PDF, PNG, JPG, TIF)
            dpi   — render DPI for PDFs (optional, default 300)
        Response 200:
            { "partData": {...}, "diagnostics": {...} }
        Response 400:
            { "error": "human-readable reason" }
        Response 500:
            { "error": "...", "traceback": "..." }   (debug mode only)

    GET /api/health
        { "status": "ok", "module2": "available" }
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

# ── Module 2 import ──────────────────────────────────────────────────────────
MODULE2_DIR = Path(__file__).resolve().parent / "module2"
if str(MODULE2_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE2_DIR))

try:
    from main import run_pipeline, EXIT_OK  # noqa: E402
    MODULE2_AVAILABLE = True
except Exception as _m2_err:
    MODULE2_AVAILABLE = False
    _M2_IMPORT_ERROR  = str(_m2_err)

# ── Flask ────────────────────────────────────────────────────────────────────
from flask import Flask, jsonify, request  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB upload limit

# Whether to include Python tracebacks in 500 responses (set via --debug flag)
_VERBOSE_ERRORS = False

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _err(msg: str, status: int = 400, tb: str | None = None) -> tuple:
    body = {"error": msg}
    if tb and _VERBOSE_ERRORS:
        body["traceback"] = tb
    return jsonify(body), status


def _validate_file(file_storage) -> str | None:
    """Return error string or None if valid."""
    if file_storage is None or file_storage.filename == "":
        return "No file uploaded. Send multipart field 'file'."
    ext = Path(file_storage.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return (
            f"Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return None


def _safe_dpi(raw: str | None) -> tuple[int, str | None]:
    """Parse DPI string. Returns (dpi, error_or_None)."""
    if raw is None:
        return 300, None
    try:
        dpi = int(raw)
    except (ValueError, TypeError):
        return 300, f"dpi must be an integer, got '{raw}'"
    if not (72 <= dpi <= 600):
        return 300, f"dpi must be 72–600, got {dpi}"
    return dpi, None


def _to_json_safe(obj):
    """Recursively make an object JSON-serialisable (handles dataclasses, etc.)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _to_json_safe(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    if not MODULE2_AVAILABLE:
        return jsonify({
            "status": "degraded",
            "module2": "unavailable",
            "error":   _M2_IMPORT_ERROR,
        }), 503
    return jsonify({"status": "ok", "module2": "available"})


@app.route("/api/recognize", methods=["POST"])
def recognize():
    # ── Guard: module2 importable ────────────────────────────────────────────
    if not MODULE2_AVAILABLE:
        return _err(
            f"Module 2 failed to load: {_M2_IMPORT_ERROR}",
            status=503,
        )

    # ── Validate inputs ──────────────────────────────────────────────────────
    file_storage = request.files.get("file")
    file_err = _validate_file(file_storage)
    if file_err:
        return _err(file_err, status=400)

    dpi, dpi_err = _safe_dpi(request.form.get("dpi"))
    if dpi_err:
        return _err(dpi_err, status=400)

    # ── Save upload to temp file (Module 2 needs a real path) ────────────────
    suffix = Path(file_storage.filename).suffix.lower()
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, prefix="cnc_upload_"
        ) as tmp:
            file_storage.save(tmp)
            tmp_path = Path(tmp.name)

        # ── Run Module 2 pipeline ────────────────────────────────────────────
        result = run_pipeline(tmp_path, dpi=dpi, quiet=True)

        # result = { "partData": {...}, "diagnostics": {...} }
        if not isinstance(result, dict):
            return _err(
                "Pipeline returned an unexpected type "
                f"({type(result).__name__}). Expected dict.",
                status=500,
            )

        if "partData" not in result or "diagnostics" not in result:
            return _err(
                f"Pipeline result missing keys. Got: {list(result.keys())}",
                status=500,
            )

        return jsonify(_to_json_safe(result)), 200

    except FileNotFoundError as exc:
        tb = traceback.format_exc()
        return _err(f"File not found during processing: {exc}", status=500, tb=tb)

    except ValueError as exc:
        tb = traceback.format_exc()
        return _err(f"Invalid input: {exc}", status=400, tb=tb)

    except Exception as exc:
        tb = traceback.format_exc()
        # Always log full traceback server-side
        print(f"\n[ERROR] /api/recognize — {type(exc).__name__}: {exc}", file=sys.stderr)
        print(tb, file=sys.stderr)
        return _err(
            f"{type(exc).__name__}: {exc}",
            status=500,
            tb=tb,
        )

    finally:
        # Always clean up the temp file
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# ── CORS (dev convenience — remove in production) ────────────────────────────

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/api/recognize", methods=["OPTIONS"])
@app.route("/api/health",    methods=["OPTIONS"])
def options_handler():
    return "", 204


# ── Startup ──────────────────────────────────────────────────────────────────

def main():
    global _VERBOSE_ERRORS

    parser = argparse.ArgumentParser(description="CNC AI Backend Server")
    parser.add_argument("--port",  type=int, default=5000)
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--debug", action="store_true",
                        help="Include Python tracebacks in error responses")
    args = parser.parse_args()

    _VERBOSE_ERRORS = args.debug

    print(f"\n CNC AI Backend")
    print(f" Module 2: {'✓ available' if MODULE2_AVAILABLE else '✗ UNAVAILABLE'}")
    if not MODULE2_AVAILABLE:
        print(f"   Error: {_M2_IMPORT_ERROR}", file=sys.stderr)
    print(f" Listening on http://{args.host}:{args.port}")
    print(f" Debug errors: {'on' if _VERBOSE_ERRORS else 'off'}")
    print(f" Max upload:   50 MB\n")

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
