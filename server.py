from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import tempfile
import traceback
import os

app = Flask(**name**, static_folder=”.”, static_url_path=””)

@app.route(”/”)
def home():
return send_from_directory(”.”, “index.html”)

@app.route(”/api/recognize”, methods=[“POST”])
def recognize():
if “file” not in request.files:
return jsonify({“error”: “Keine Datei empfangen”}), 400

```
uploaded = request.files["file"]
if uploaded.filename == "":
    return jsonify({"error": "Keine Datei gewählt"}), 400

temp_path = None
try:
    suffix = Path(uploaded.filename).suffix or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        uploaded.save(tmp.name)
        temp_path = Path(tmp.name)

    try:
        from module2.main import run_pipeline
    except Exception as e:
        return jsonify({
            "error": "Importfehler module2.main",
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

    try:
        result = run_pipeline(temp_path)
    except Exception as e:
        return jsonify({
            "error": "Pipeline Crash",
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

    try:
        return jsonify(result)
    except Exception:
        return jsonify({
            "success": True,
            "raw_result": str(result)
        })

except Exception as e:
    return jsonify({
        "error": "Serverfehler",
        "type": type(e).__name__,
        "message": str(e),
        "traceback": traceback.format_exc()
    }), 500
finally:
    try:
        if temp_path and temp_path.exists():
            os.remove(temp_path)
    except Exception:
        pass
```

if **name** == “**main**”:
app.run(host=“0.0.0.0”, port=10000)