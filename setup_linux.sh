#!/usr/bin/env bash
# Set up SEAM in a virtual environment (venv). Linux and macOS.
#
#   ./setup_linux.sh          everything: website, analyse your own videos, tests
#   ./setup_linux.sh --lite   only what the website needs (about 60 MB instead of 1.5 GB)
#
# Needs Python 3.11 or 3.12 (3.12 is what it was built on); --lite also runs on 3.13. Nothing is installed outside the .venv folder it creates.
set -euo pipefail
cd "$(dirname "$0")"

MODE=full
[ "${1:-}" = "--lite" ] && MODE=lite

# The full install needs Python 3.11 or 3.12: the pinned MediaPipe has no build for 3.13.
# The website alone (--lite) also runs on 3.13.
if [ "$MODE" = "lite" ]; then NEWEST=13; else NEWEST=12; fi
PY=""
for candidate in python3.12 python3.11 python3.13 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, $NEWEST) else 1)" 2>/dev/null; then
    PY="$candidate"
    break
  fi
done
if [ -z "$PY" ]; then
  if [ "$MODE" = "full" ]; then
    echo "The full install needs Python 3.11 or 3.12, and neither was found."
    echo "  Ubuntu/Debian:  sudo apt install python3.12 python3.12-venv"
    echo "  macOS:          brew install python@3.12"
    echo "  Any system:     https://www.python.org/downloads/release/python-3120/"
    echo "Or run  ./setup_linux.sh --lite  for the website only, which also works on 3.13."
  else
    echo "Python 3.11, 3.12 or 3.13 was not found. Install one from https://www.python.org/downloads/"
  fi
  exit 1
fi
echo "Using $($PY --version) at $(command -v "$PY")"

if ! "$PY" -m venv .venv 2>/dev/null; then
  echo "Could not create a virtual environment. On Ubuntu/Debian install the venv module:"
  echo "  sudo apt install python3-venv     (or python3.12-venv)"
  exit 1
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --quiet --upgrade pip

if [ "$MODE" = "lite" ]; then
  echo "Installing the website's packages ..."
  python -m pip install -r requirements-lite.txt
else
  echo "Installing everything (this downloads about 1.5 GB the first time) ..."
  # The CPU build of PyTorch: no graphics card is needed to run anything here.
  python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev,avatar]"
fi

export PYTHONPATH="$PWD/src"
export SEAM_MEDIAPIPE_MODELS="$PWD/models/mediapipe"
echo
python scripts/check_install.py
