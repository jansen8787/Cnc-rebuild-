"""
server.py — CNC AI Backend
===========================
Flask server exposing POST /api/recognize.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import tempfile
import traceback
from pathlib import Path

# ── Module 2 import ──────────────────────────────────────────────────────────
MODULE2_DIR = Path(__file__).resolve().parent / "module2"
if str(MODULE2_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE2_DIR))

try:
    from main import run_pipeline
    MODULE2_AVAILABLE = True
except Exception as _m2_err:
    MODULE2_AVAILABLE = False
    _M2_IMPORT_ERROR = str(_m2_err)

# ── Flask ────────────────────────────────────────────────────────────────────
from flask import Flask, jsonify, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

_VERBOSE_ERRORS = False

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ── Root Route (NEU) ─────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "message": "CNC AI Backend is running",
        "health": "/api/health",
        "recognize": "/api/recognize"
    })


# ── Helpers ──────────────────────────────────────────────────────────────────
def _err(msg: str, status: int = 400, tb: str | None = None):
    body = {"error": msg}
    if tb and _VERBOSE_ERRORS:
        body["traceback"] = tb
    return jsonify(body), status


def _validate_file(file_storage):
    if file_storage is None or file_storage.filename == "":
        return "No file uploaded. Send multipart field 'file'."

    ext = Path(file_storage.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return f"Unsupported file type '{ext}'"
    return None


def _safe_dpi(raw):
    if raw is None:
        return 300, None
    try:
        dpi = int(raw)
    except Exception:
        return 300, "dpi must be integer"

    if not (72 <= dpi <= 600):
        return 300, "dpi must be 72-600"

    return dpi, None


def _to_json_safe(obj):
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
            "error": _M2_IMPORT_ERROR
        }), 503

    return jsonify({
        "status": "ok",
        "module2": "available"
    })


@app.route("/api/recognize", methods=["POST"])
def recognize():
    if not MODULE2_AVAILABLE:
        return _err(f"Module 2 failed to load: {_M2_IMPORT_ERROR}", 503)

    file_storage = request.files.get("file")
    file_err = _validate_file(file_storage)
    if file_err:
        return _err(file_err, 400)

    dpi, dpi_err = _safe_dpi(request.form.get("dpi"))
    if dpi_err:
        return _err(dpi_err, 400)

    suffix = Path(file_storage.filename).suffix.lower()
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
            prefix="cnc_upload_"
        ) as tmp:
            file_storage.save(tmp)
            tmp_path = Path(tmp.name)

        result = run_pipeline(tmp_path, dpi=dpi, quiet=True)

        if not isinstance(result, dict):
            return _err("Pipeline returned invalid result", 500)

        return jsonify(_to_json_safe(result)), 200

    except Exception as exc:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return _err(f"{type(exc).__name__}: {exc}", 500, tb)

    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# ── CORS ─────────────────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/recognize", methods=["OPTIONS"])
@app.route("/api/health", methods=["OPTIONS"])
def options_handler():
    return "", 204


# ── Startup ──────────────────────────────────────────────────────────────────
def main():
    global _VERBOSE_ERRORS

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    _VERBOSE_ERRORS = args.debug

    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        use_reloader=False
    )


if __name__ == "__main__":
    main()