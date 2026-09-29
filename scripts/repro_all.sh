#!/usr/bin/env bash
# Regenerate every artifact the paper cites, in dependency order.
#
# This exists because `make repro` pointed at a script that was never written, so the
# target failed silently in the sense that mattered: nothing had ever run end to end.
#
# Design notes:
#
# - Ordered by dependency. Landmarks before markers, markers before the affect model,
#   the factorizer before its probes. Running these out of order produces artifacts
#   that are internally consistent and wrong.
# - Each stage is independently skippable, because a full pass re-runs the ONNX export
#   and the benchmark, and the benchmark takes a real idle machine to mean anything.
#   Set SEAM_REPRO_STAGES to a subset, e.g. SEAM_REPRO_STAGES="verify,alignment".
# - Stages that need the GPU say so and are skipped with a notice rather than being
#   allowed to produce a benchmark on a busy machine, which is how the K6 number
#   nearly got recorded as noise in the first place.
#
# Not destructive: writes under artifacts/, never touches data/ or src/.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-/home/bhuwan/miniconda3/envs/slr/bin/python}"
STAGES="${SEAM_REPRO_STAGES:-verify,alignment,landmarks,markers,fer,audit,grounding,m4,parity,bench,provenance}"

ALL_STAGES=(verify alignment landmarks markers fer audit grounding m4 parity bench provenance)
declare -A DESC=(
  [verify]="verify data resources and print the readiness table"
  [alignment]="ASLLRP token-to-crop-frame alignment report"
  [landmarks]="EmoSign landmark extraction"
  [markers]="marker labelling with provenance"
  [fer]="FER models: train, export, parity"
  [audit]="confound audit and instrument diagnosis"
  [grounding]="cue-grounding report"
  [m4]="affect factorizer, LOSO folds, probes, ablation"
  [parity]="ONNX FP32/INT8 parity"
  [bench]="latency / VRAM benchmark"
  [provenance]="fail on any number in the paper with no artifact behind it"
)

log()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

want() { [[ ",$STAGES," == *",$1,"* ]]; }
run()  { want "$1" || return 0; shift; note "\$ $*"; "$@" || die "$*"; }

log "SEAM repro — stages: $STAGES"
note "repo: $REPO"
note "python: $PY"
"$PY" -c 'import sys; assert sys.version_info >= (3, 12), sys.version' \
  || die "need Python >= 3.12"

# ---------------------------------------------------------------- cheap, always safe
if want verify; then
  log "[1/11] $(printf '%s' "${DESC[verify]}")"
  run verify "$PY" -m seam.cli data verify --all --table
fi

if want alignment; then
  log "[2/11] $(printf '%s' "${DESC[alignment]}")"
  run alignment "$PY" scripts/check_asllrp_alignment.py
fi

# ---------------------------------------------------------------- perception
if want landmarks; then
  log "[3/11] $(printf '%s' "${DESC[landmarks]}")"
  run landmarks "$PY" -m seam.cli landmarks extract --dataset emosign
  run facegate "$PY" -m seam.cli landmarks face-gate --sample 24
fi

if want markers; then
  log "[4/11] $(printf '%s' "${DESC[markers]}")"
  run markers "$PY" scripts/label_markers.py
fi

# ---------------------------------------------------------------- affect, no GPU needed
if want fer; then
  log "[5/11] $(printf '%s' "${DESC[fer]}")"
  run fer "$PY" scripts/train_fer.py
fi

if want audit; then
  log "[6/11] $(printf '%s' "${DESC[audit]}")"
  run audit "$PY" scripts/run_confound_audit.py
  run ferdiag "$PY" scripts/diagnose_fer_affect.py
  run fersens "$PY" scripts/diagnose_fer_sensitivity.py
fi

if want grounding; then
  log "[7/11] $(printf '%s' "${DESC[grounding]}")"
  run grounding "$PY" scripts/cue_grounding.py
fi

if want m4; then
  log "[8/11] $(printf '%s' "${DESC[m4]}")"
  run m4 "$PY" scripts/train_factorizer.py
  run m4ablate "$PY" scripts/train_factorizer.py --ablate
fi

# ---------------------------------------------------------------- GPU
if want parity; then
  log "[9/11] $(printf '%s' "${DESC[parity]}")"
  run parity "$PY" -m seam.cli export
fi

if want bench; then
  log "[10/11] $(printf '%s' "${DESC[bench]}")"
  # A benchmark on a loaded machine is worse than no benchmark: it produces a number
  # that looks like a measurement. Check load and refuse rather than record noise.
  load="$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)"
  note "1-minute load average: $load"
  if [[ "$load" != "0" ]] && awk -v l="$load" 'BEGIN{exit !(l>1.5)}'; then
    die "load $load > 1.5; the K6 number would measure contention, not the model. Idle the machine, or skip with SEAM_REPRO_STAGES without 'bench'."
  fi
  run bench "$PY" -m seam.cli bench
  run benchseq "$PY" -m seam.cli bench --sequential
fi

# ---------------------------------------------------------------- the guard
if want provenance; then
  log "[11/11] $(printf '%s' "${DESC[provenance]}")"
  if ! run prov "$PY" -m pytest tests/test_provenance.py -q; then
    die "the paper cites a number no artifact produced — see tests/test_provenance.py"
  fi
fi

log "done"
note "artifacts: $(find artifacts -name '*.json' | wc -l) JSON files"
note "next: make paper"
