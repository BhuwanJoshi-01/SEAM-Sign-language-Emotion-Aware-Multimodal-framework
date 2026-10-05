#!/usr/bin/env bash
# Start the SEAM website on this computer and open it in the browser.
#   ./run_demo.sh            http://127.0.0.1:8000/
#   ./run_demo.sh 9000       another port
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8000}"

if [ ! -f .venv/bin/activate ]; then
  echo "Run ./setup_linux.sh first."
  exit 1
fi
# shellcheck disable=SC1091
. .venv/bin/activate
export PYTHONPATH="$PWD/src"
export SEAM_MEDIAPIPE_MODELS="$PWD/models/mediapipe"
# Keep everything inside this folder instead of the paths of the machine it was built on.
export SEAM_DATA_ROOT="${SEAM_DATA_ROOT:-$PWD/data}"

URL="http://127.0.0.1:$PORT/"
echo "SEAM is starting at $URL  (Ctrl+C here stops it)"
# SEAM_NO_BROWSER=1 skips opening a browser, for a machine with no desktop.
if [ -z "${SEAM_NO_BROWSER:-}" ]; then
  ( sleep 2
    if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 || true
    elif command -v open >/dev/null 2>&1; then open "$URL" || true
    fi ) &
fi
exec python -m seam.cli serve --port "$PORT"
