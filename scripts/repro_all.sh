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
STAGES="${SEAM_REPRO_STAGES:-verify,alignment,signstream,landmarks,markers,labels,fer,audit,grounding,m4,m5a,avatar,parity,bench,provenance}"

ALL_STAGES=(verify alignment signstream landmarks markers labels fer audit grounding m4 m5a avatar parity bench provenance)
declare -A DESC=(
  [verify]="verify data resources and print the readiness table"
  [alignment]="ASLLRP token-to-crop-frame alignment report"
  [signstream]="parse the SignStream XML: utterances, signs, human non-manual events"
  [landmarks]="EmoSign landmark extraction"
  [markers]="marker labelling with provenance"
  [labels]="heuristic labels against human annotation: syntactic (kappa) and visual (frame-level AUC)"
  [fer]="FER models: train, export, parity"
  [audit]="confound audit on isolated signs and on continuous signing, and instrument diagnosis"
  [grounding]="cue-grounding report"
  [m4]="affect factorizer under LOSO: heuristic labels, human labels, 3-seed ablation"
  [m5a]="gloss recogniser against its most-frequent and shuffled-label baselines"
  [avatar]="skinned glTF export, checked by an independent reader"
  [parity]="ONNX FP32/INT8 parity"
  [bench]="latency / VRAM benchmark"
  [provenance]="fail on an untraced number, or on a result written by code that has since changed"
)

log()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

want() { [[ ",$STAGES," == *",$1,"* ]]; }

# Echo a command and run it; a non-zero exit stops the whole run.
#
# This used to take a label as its first argument and return 0 without running anything
# when that label was not a requested stage. Every call sits inside an `if want <stage>`
# block already, so the check did nothing for the first command of a stage and silently
# skipped every other one that had been given its own label: the face gate, both FER
# diagnostics, the sequential benchmark, the ablation, and the provenance guard itself.
# The script printed each stage header, ran the first command, and reported "done".
run()  { note "\$ $*"; "$@" || die "$*"; }

# A stage name that is not a stage is a typo, and a typo used to mean "skip it quietly".
IFS=',' read -r -a _requested <<< "$STAGES"
for _s in "${_requested[@]}"; do
  [[ " ${ALL_STAGES[*]} " == *" $_s "* ]] || die "unknown stage '$_s'; stages are: ${ALL_STAGES[*]}"
done

log "SEAM repro — stages: $STAGES"
note "repo: $REPO"
note "python: $PY"
"$PY" -c 'import sys; assert sys.version_info >= (3, 12), sys.version' \
  || die "need Python >= 3.12"

# ---------------------------------------------------------------- cheap, always safe
if want verify; then
  log "[verify] ${DESC[verify]}"
  run "$PY" -m seam.cli data verify --all --table
fi

if want alignment; then
  log "[alignment] ${DESC[alignment]}"
  run "$PY" scripts/check_asllrp_alignment.py
fi

# The SignStream XML is licence-gated and lives outside the repository. Without it the
# human-label stages cannot run, and saying so beats failing three stages later.
XML_DIR="${SEAM_SIGNSTREAM_XML:-/mnt/DevProd/seam_data/asllrp_signstream_xml/raw}"
if want signstream; then
  log "[signstream] ${DESC[signstream]}"
  [[ -d "$XML_DIR" ]] || die "SignStream XML not found at $XML_DIR (set SEAM_SIGNSTREAM_XML); see paper/provenance/STEP3_HANDOFF.md"
  run "$PY" scripts/parse_signstream.py --path "$XML_DIR" --min-utterances 100
  # In range is not aligned: validate the session-to-clip frame mapping against an
  # independent signal (annotated blinks vs the eyeBlink blendshape) before any stage
  # below reads a frame-level label.
  run "$PY" scripts/check_signstream_alignment.py --xml "$XML_DIR"
fi

# ---------------------------------------------------------------- perception
if want landmarks; then
  log "[landmarks] ${DESC[landmarks]}"
  run "$PY" -m seam.cli landmarks extract --dataset emosign
  run "$PY" -m seam.cli landmarks face-gate --sample 24
  # The two hand slots must hold two hands. Until 2026-10-05 they held one hand twice and
  # nothing checked; this exits non-zero if that ever recurs.
  run "$PY" scripts/check_hand_slots.py
fi

if want markers; then
  log "[markers] ${DESC[markers]}"
  run "$PY" scripts/label_markers.py
  # The negation / head-shake association has been wrong here once, so it is re-checked
  # against the annotators, a placebo axis and each signer every time it is re-measured.
  run "$PY" scripts/check_negation_head_shake.py --xml "$XML_DIR"
fi

if want labels; then
  log "[labels] ${DESC[labels]}"
  run "$PY" scripts/check_label_quality.py --xml "$XML_DIR"
  # The visual markers against human frame labels, leave-one-signer-out.
  run "$PY" scripts/validate_markers.py --xml "$XML_DIR"
fi

# ---------------------------------------------------------------- affect, no GPU needed
if want fer; then
  log "[fer] ${DESC[fer]}"
  run "$PY" scripts/train_fer.py
fi

if want audit; then
  log "[audit] ${DESC[audit]}"
  run "$PY" scripts/run_confound_audit.py
  run "$PY" scripts/diagnose_fer_affect.py
  run "$PY" scripts/diagnose_fer_sensitivity.py
  # C1 where M1 said it had to be tested: continuous signing, human frame-level markers.
  # Its design and decision rule are fixed in the script; re-running reproduces the
  # registered analysis, it does not re-open it.
  run "$PY" scripts/run_confound_audit_continuous.py --xml "$XML_DIR"
fi

if want grounding; then
  log "[grounding] ${DESC[grounding]}"
  run "$PY" scripts/cue_grounding.py
fi

# train_factorizer.py exits 1 when the M4 gate is not met. The gate is a measured result,
# not a build failure - M4 is reported as refuted - so only a crash (exit >= 2) stops the
# run here. Each invocation writes its own file; the ablation used to overwrite the
# single-seed run because both defaulted to one name.
m4run() {
  note "\$ $*"
  local rc=0
  "$@" || rc=$?
  [[ $rc -le 1 ]] || die "$* (exit $rc)"
  [[ $rc -eq 0 ]] || note "gate not met (exit 1) - recorded as a result, not an error"
}
if want m4; then
  log "[m4] ${DESC[m4]}"
  m4run "$PY" scripts/train_factorizer.py
  m4run "$PY" scripts/train_factorizer.py --labels human --xml-dir "$XML_DIR"
  m4run "$PY" scripts/train_factorizer.py --ablate --seeds 0 1 2 --tag ablation
fi

if want m5a; then
  log "[m5a] ${DESC[m5a]}"
  run "$PY" scripts/train_recogniser.py
  # The same test with both hands' shape added. Until 2026-10-05 the two hand slots held
  # one hand twice, so this variant had never been run on real two-hand data.
  run "$PY" scripts/train_recogniser.py --part upper+hands
fi

# The glTF tests skip loudly without the licence-gated SMPL-X model or a Chrome binary;
# a skip is printed, never counted as a pass.
if want avatar; then
  log "[avatar] ${DESC[avatar]}"
  run "$PY" -m pytest tests/test_gltf_export.py tests/test_gltf_conformance.py -q -rs
fi

# ---------------------------------------------------------------- GPU
if want parity; then
  log "[parity] ${DESC[parity]}"
  run "$PY" -m seam.cli export
fi

if want bench; then
  log "[bench] ${DESC[bench]}"
  # A benchmark on a loaded machine is worse than no benchmark: it produces a number
  # that looks like a measurement. Check load and refuse rather than record noise.
  load="$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)"
  note "1-minute load average: $load"
  if [[ "$load" != "0" ]] && awk -v l="$load" 'BEGIN{exit !(l>1.5)}'; then
    die "load $load > 1.5; the K6 number would measure contention, not the model. Idle the machine, or skip with SEAM_REPRO_STAGES without 'bench'."
  fi
  run "$PY" -m seam.cli bench
  run "$PY" -m seam.cli bench --sequential
fi

# ---------------------------------------------------------------- the guard
if want provenance; then
  log "[provenance] ${DESC[provenance]}"
  run "$PY" -m pytest tests/test_provenance.py tests/test_artifact_staleness.py -q
fi

log "done"
note "artifacts: $(find artifacts -name '*.json' | wc -l) JSON files"
note "next: make paper"
