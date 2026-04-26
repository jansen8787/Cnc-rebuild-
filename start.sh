#!/usr/bin/env bash
# start.sh — CNC AI: start backend + frontend in one command
# Run from the project root (same folder as this script)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE2_DIR="$SCRIPT_DIR/module2"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
BACKEND_PORT=5000
FRONTEND_PORT=3000

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; AMBER='\033[0;33m'; RESET='\033[0m'
ok()  { echo -e "${GREEN}  ✓  $*${RESET}"; }
err() { echo -e "${RED}  ✗  $*${RESET}"; }
inf() { echo -e "${AMBER}  →  $*${RESET}"; }

echo ""
echo "  CNC AI — Drawing Recognition"
echo "  ════════════════════════════"
echo ""

# ── Check Python ─────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  err "python3 not found. Install Python 3.10+"; exit 1
fi
ok "Python: $(python3 --version)"

# ── Check Module 2 importable ────────────────────────────────────────────────
if ! python3 -c "import sys; sys.path.insert(0,'$MODULE2_DIR'); from main import run_pipeline" 2>/dev/null; then
  err "Module 2 failed to import. Check $MODULE2_DIR"
  python3 -c "import sys; sys.path.insert(0,'$MODULE2_DIR'); from main import run_pipeline" 2>&1 | sed 's/^/    /'
  exit 1
fi
ok "Module 2: available"

# ── Install Python deps if needed ────────────────────────────────────────────
inf "Checking Python dependencies…"
python3 -c "import flask" 2>/dev/null || pip3 install flask --break-system-packages -q
python3 -c "import cv2"   2>/dev/null || pip3 install opencv-python-headless --break-system-packages -q
python3 -c "import PIL"   2>/dev/null || pip3 install pillow --break-system-packages -q
python3 -c "import numpy" 2>/dev/null || pip3 install numpy --break-system-packages -q
ok "Python dependencies ready"

# ── Check Node / npm ─────────────────────────────────────────────────────────
if ! command -v node &>/dev/null; then
  err "node not found. Install Node.js 18+"; exit 1
fi
ok "Node: $(node --version)"

# ── Install frontend deps ─────────────────────────────────────────────────────
cd "$FRONTEND_DIR"
if [ ! -d node_modules ]; then
  inf "Installing frontend dependencies (first run only)…"
  npm install
  ok "npm install done"
else
  ok "node_modules: present"
fi

# ── Start Flask backend ───────────────────────────────────────────────────────
cd "$SCRIPT_DIR"
inf "Starting backend on http://127.0.0.1:$BACKEND_PORT …"
python3 server.py --port "$BACKEND_PORT" --debug &
BACKEND_PID=$!

# Wait for backend to be ready
for i in {1..20}; do
  if curl -sf "http://127.0.0.1:$BACKEND_PORT/api/health" >/dev/null 2>&1; then
    ok "Backend ready (pid $BACKEND_PID)"
    break
  fi
  sleep 0.3
  if [ $i -eq 20 ]; then
    err "Backend did not start within 6s"
    kill $BACKEND_PID 2>/dev/null
    exit 1
  fi
done

# ── Start Vite frontend ───────────────────────────────────────────────────────
cd "$FRONTEND_DIR"
inf "Starting frontend on http://localhost:$FRONTEND_PORT …"
npm run dev &
FRONTEND_PID=$!

sleep 1.5
ok "Frontend started (pid $FRONTEND_PID)"

echo ""
echo "  ┌──────────────────────────────────────────┐"
echo "  │                                          │"
echo "  │   Open:  http://localhost:$FRONTEND_PORT          │"
echo "  │   API:   http://localhost:$BACKEND_PORT/api    │"
echo "  │                                          │"
echo "  │   Ctrl+C to stop both servers            │"
echo "  │                                          │"
echo "  └──────────────────────────────────────────┘"
echo ""

# ── Trap Ctrl+C — kill both ───────────────────────────────────────────────────
cleanup() {
  echo ""
  inf "Stopping servers…"
  kill $BACKEND_PID  2>/dev/null
  kill $FRONTEND_PID 2>/dev/null
  wait $BACKEND_PID  2>/dev/null
  wait $FRONTEND_PID 2>/dev/null
  ok "Done."
}
trap cleanup INT TERM

wait $BACKEND_PID $FRONTEND_PID
