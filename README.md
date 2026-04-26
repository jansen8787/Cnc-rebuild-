# CNC AI — Drawing Recognition App

## Quick start (one command)

```bash
cd cnc-ai-app
chmod +x start.sh
./start.sh
```

Open **http://localhost:3000**

---

## Manual start (two terminals)

### Terminal 1 — Backend

```bash
cd cnc-ai-app
pip install flask pillow opencv-python-headless numpy
python server.py --debug
```

Backend runs on **http://localhost:5000**
Check: `curl http://localhost:5000/api/health`

### Terminal 2 — Frontend

```bash
cd cnc-ai-app/frontend
npm install
npm run dev
```

Frontend runs on **http://localhost:3000**
Vite proxies `/api/*` → `http://localhost:5000`

---

## Requirements

| Tool    | Version  |
|---------|----------|
| Python  | 3.10+    |
| Node.js | 18+      |
| pip     | any      |
| npm     | any      |

Optional for PDF rendering:
```bash
# macOS
brew install poppler

# Ubuntu/Debian
apt-get install poppler-utils
```

---

## API

```
POST /api/recognize
  Content-Type: multipart/form-data
  file: <drawing file>   PDF · PNG · JPG · TIF
  dpi:  300              72–600, PDF render resolution

Response 200:
{
  "partData":    { ...Module 1 V2 PartData... },
  "diagnostics": { ...RecognitionReport...   }
}

Response 400:  { "error": "reason" }
Response 500:  { "error": "reason", "traceback": "..." }

GET /api/health
{ "status": "ok", "module2": "available" }
```

---

## Project layout

```
cnc-ai-app/
├── server.py          ← Flask backend (wraps module2)
├── start.sh           ← One-command startup
├── module2/           ← Recognition engine (Python)
│   ├── main.py
│   ├── m2types.py
│   ├── stage0_ingest/
│   ├── stage1_preprocess/
│   ├── stage2_ocr/
│   ├── stage25_titleblock/
│   ├── stage3_geometry/
│   ├── stage4_scale/
│   ├── stage5_assemble/
│   └── stage6_diagnostics/
└── frontend/          ← React + Vite
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        └── App.jsx
```
