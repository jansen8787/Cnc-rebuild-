"""
server.py — CNC AI (flat deployment version)
Serves React frontend (frontend/dist/) + POST /api/recognize

Production: gunicorn server:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
"""
from __future__ import annotations
import dataclasses, json, os, sys, tempfile, traceback
from pathlib import Path

# ── Load pipeline from bundled single file ────────────────────────────────────
_here = Path(__file__).resolve().parent
_pipeline_path = _here / "pipeline.py"

_pipeline_ns: dict = {"__file__": str(_pipeline_path)}
try:
    exec(_pipeline_path.read_text(), _pipeline_ns)
    run_pipeline = _pipeline_ns["run_pipeline"]
    MODULE2_OK = True
    MODULE2_ERR = ""
except Exception as _e:
    run_pipeline = None
    MODULE2_OK = False
    MODULE2_ERR = str(_e)
    import traceback as _tb; _tb.print_exc()

# ── Flask ─────────────────────────────────────────────────────────────────────
from flask import Flask, jsonify, request, send_from_directory, send_file

DIST = _here / "dist"
app  = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

VERBOSE = os.environ.get("DEBUG_ERRORS", "0") == "1"

def err(msg, status=400, tb=None):
    body = {"error": msg}
    if tb and VERBOSE:
        body["traceback"] = tb
    return jsonify(body), status

def safe(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return safe(dataclasses.asdict(obj))
    if isinstance(obj, dict):            return {k: safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):   return [safe(v) for v in obj]
    if isinstance(obj, Path):            return str(obj)
    return obj

@app.route("/api/health")
def health():
    if not MODULE2_OK:
        return jsonify({"status": "degraded", "error": MODULE2_ERR}), 503
    return jsonify({"status": "ok", "module2": "available"})

@app.route("/api/recognize", methods=["POST", "OPTIONS"])
def recognize():
    if request.method == "OPTIONS": return "", 204
    if not MODULE2_OK:              return err(f"Pipeline unavailable: {MODULE2_ERR}", 503)

    f = request.files.get("file")
    if not f or not f.filename:
        return err("No file. Send multipart field 'file'.")
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return err(f"Unsupported type '{ext}'. Accepted: {', '.join(sorted(ALLOWED))}")

    try:
        dpi = int(request.form.get("dpi", 300))
        assert 72 <= dpi <= 600
    except Exception:
        return err("dpi must be 72-600")

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False, prefix="cnc_") as t:
            f.save(t); tmp = Path(t.name)
        result = run_pipeline(tmp, dpi=dpi, quiet=True)
        if not isinstance(result, dict) or "partData" not in result:
            return err("Pipeline returned unexpected shape", 500)
        return jsonify(safe(result))
    except ValueError as e:
        return err(str(e), 400, traceback.format_exc())
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] {type(e).__name__}: {e}\n{tb}", file=sys.stderr)
        return err(f"{type(e).__name__}: {e}", 500, tb)
    finally:
        if tmp and tmp.exists():
            try: tmp.unlink()
            except OSError: pass

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def frontend(path):
    if path.startswith("api/"): return err("Not found", 404)
    asset = DIST / path
    if path and asset.exists() and asset.is_file():
        return send_from_directory(str(DIST), path)
    idx = DIST / "index.html"
    if idx.exists(): return send_file(str(idx))
    return err("Frontend not built. Run: cd frontend && npm install && npm run build", 503)

@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return r

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n CNC AI  |  module2: {'ok' if MODULE2_OK else 'FAIL'}  |  :{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
