from flask import Flask, request, jsonify, send_from_directory
import os
import sys
from pathlib import Path

app = Flask(__name__, static_folder=".", static_url_path="")

MODULE2_DIR = Path(__file__).resolve().parent / "module2"
sys.path.insert(0, str(MODULE2_DIR))

try:
    from main import run_pipeline
except Exception as e:
    run_pipeline = None
    IMPORT_ERROR = str(e)

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/api/health")
def health():
    return jsonify({"status": "online"})

@app.route("/api/recognize", methods=["POST"])
def recognize():
    if run_pipeline is None:
        return jsonify({"error": IMPORT_ERROR})

    if "file" not in request.files:
        return jsonify({"error": "Keine Datei hochgeladen"})

    file = request.files["file"]
    temp = "/tmp/" + file.filename
    file.save(temp)

    result = run_pipeline(temp)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)