#!/usr/bin/env bash
# Run a Python command against the *perception* interpreter.
#
# Why this exists
# ---------------
# This project pins `mediapipe==0.10.14` (pyproject.toml). That version has no
# Windows build for Python 3.13, but it does have one for Python 3.12. The
# project's own virtualenv (.venv) is 3.13, so perception can never run there on
# Windows - not a bug, just a missing wheel.
#
# So the perception path needs a second interpreter. This script finds it and
# exports the model directory, so callers do not have to remember either.
#
# Usage:
#   scripts/run_perception.sh -c "import mediapipe; print(mediapipe.__version__)"
#   scripts/run_perception.sh scripts/your_script.py --flag
#   SEAM_PERCEPTION_PY=/path/to/python scripts/run_perception.sh -c "print(1)"
#
# Override the interpreter with SEAM_PERCEPTION_PY and the model directory with
# SEAM_MEDIAPIPE_MODELS. Both are resolved below with sensible defaults.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- locate the perception interpreter -------------------------------------
# Preference order: explicit override, then the known-good 3.12 env, then any
# python3.12 on PATH. 3.13 is deliberately never chosen: the pinned mediapipe
# cannot exist there on Windows.
find_python() {
  if [[ -n "${SEAM_PERCEPTION_PY:-}" ]]; then
    echo "$SEAM_PERCEPTION_PY"
    return 0
  fi
  local candidates=(
    "$HOME/.workbuddy-ai/binaries/python/envs/landmarks312/Scripts/python.exe"
    "$HOME/.workbuddy-ai/binaries/python/envs/landmarks312/bin/python"
  )
  local c
  for c in "${candidates[@]}"; do
    [[ -x "$c" ]] && { echo "$c"; return 0; }
  done
  if command -v python3.12 >/dev/null 2>&1; then
    command -v python3.12
    return 0
  fi
  echo ""
}

PY="$(find_python)"
if [[ -z "$PY" ]]; then
  cat >&2 <<'MSG'
ERROR: no Python 3.12 interpreter found for the perception path.

mediapipe==0.10.14 (the version this project pins) has no Windows/Python 3.13
build, so perception needs a 3.12 interpreter. Create one with:

  python3.12 -m venv ~/.workbuddy-ai/binaries/python/envs/landmarks312
  ~/.workbuddy-ai/binaries/python/envs/landmarks312/Scripts/pip install \
      "mediapipe==0.10.14" opencv-python numpy

or point SEAM_PERCEPTION_PY at an existing one.
MSG
  exit 2
fi

# --- locate the MediaPipe model bundles ------------------------------------
# The code searches SEAM_MEDIAPIPE_MODELS before its built-in directories.
if [[ -z "${SEAM_MEDIAPIPE_MODELS:-}" ]]; then
  for d in "$HOME/Desktop/.mp_models" "$REPO/artifacts/mediapipe_tasks"; do
    if [[ -f "$d/face_landmarker.task" ]]; then
      export SEAM_MEDIAPIPE_MODELS="$d"
      break
    fi
  done
fi

# --- verify the interpreter can actually import mediapipe ------------------
if ! "$PY" -c "import mediapipe" >/dev/null 2>&1; then
  echo "ERROR: $PY cannot import mediapipe." >&2
  echo "       Do not use the 3.13 'landmarks' env: its native bindings are broken." >&2
  exit 2
fi

# Git Bash hands out POSIX-style paths (/c/Users/...) but Windows Python needs
# native ones (C:\Users\...). `cygpath -w` converts; where it is unavailable we
# are on a POSIX host and the path is already correct. Getting this wrong is
# silent: Python simply does not add the directory and the import fails as if
# the package were missing.
#
# The separator matters just as much. Windows Python splits PYTHONPATH on ';',
# POSIX Python on ':'. Using ':' on Windows makes the entire string one
# unresolvable path, and the failure looks identical to a missing package.
if command -v cygpath >/dev/null 2>&1; then
  REPO_SRC="$(cygpath -w "$REPO/src")"
  SEP=';'
else
  REPO_SRC="$REPO/src"
  SEP=':'
fi
export PYTHONPATH="$REPO_SRC${PYTHONPATH:+$SEP$PYTHONPATH}"
exec "$PY" "$@"
