from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import sys
import traceback
import uuid
import os

app = Flask(__name__, static_folder=".", static_url_path="")

MODULE2_DIR = Path(__file__).resolve().parent / "module2"
sys.path.insert(0, str(MODULE2_DIR))

try:
    from main import run_pipeline
except Exception as e:
    run_pipeline = None
    IMPORT_ERROR = str(e)


def clean(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
    except:
        pass
    return obj


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "online"})


@app.route("/api/recognize", methods=["POST"])
def recognize():
    if run_pipeline is None:
        return jsonify({"error": IMPORT_ERROR}), 500

    if "file" not in request.files:
        return jsonify({"error": "Keine Datei hochgeladen"}), 400

    file = request.files["file"]

    ext = Path(file.filename).suffix
    temp = Path("/tmp") / f"{uuid.uuid4()}{ext}"

    try:
        file.save(str(temp))

        result = run_pipeline(temp)

        return jsonify(clean(result))

    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__,
            "trace": traceback.format_exc()
        }), 500

    finally:
        if temp.exists():
            os.remove(temp)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)