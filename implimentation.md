
# SEAM — Implementation Breakdown

**This file is the execution expansion of `plan.md`.** `plan.md` is canonical for scope, evidence
gates and KPIs. This file answers *how*: which files, which commands, which tests, in what order.

Read both. If they ever disagree, `plan.md` wins and this file is the bug.

- Status: **M0 in progress** (see §M0 for live state)
- Executor: 1 AI implementer + 1 human reviewer
- Machine: RTX 3050 4 GB · 16 vCPU · 15 GB RAM (~3 GB free) · env `slr` (Python 3.12.13)

---

## 0. Repository map

```
Sign_Language_EmotionAware/
├── plan.md                     # canonical: scope, gates, KPIs
├── implimentation.md           # this file: task-level execution
├── project_breakdown.md        # technical bible
├── team.md                     # superseded by plan.md (see note below)
├── pyproject.toml              # exact pins, no ranges
├── Makefile                    # setup lint test train bench repro paper
├── .pre-commit-config.yaml
├── configs/
│   ├── base.yaml               # paths, data, perception, train, deploy
│   ├── data/{emosign,asllrp,wlasl,how2sign,asl_citizen}.yaml
│   ├── model/{gru,stgcn,transformer}.yaml
│   └── train/{default,fast}.yaml
├── src/seam/
│   ├── cli.py                  # seam doctor | data | landmarks | bench | serve
│   ├── config.py               # YAML load, deep merge, dot access
│   ├── paths.py                # project-root + data-root resolution
│   ├── logging.py
│   ├── seed.py
│   ├── data/
│   │   ├── sources.py          # resource registry (the §1 ground truth, in code)
│   │   ├── manifest.py         # SHA256 manifest, verify/truncated detection
│   │   ├── fetch.py            # download / hardlink-reuse / verify
│   │   ├── emosign.py          # CSV load, signer parse, ASLLRP utterance-ID join
│   │   ├── asllrp.py           # gloss-token table, frame alignment, signer-disjoint splits
│   │   ├── splits.py           # signer-disjoint + LOSO fold construction
│   │   └── readiness.py        # the readiness table + blocker registry
│   ├── perception/
│   │   ├── tasks_api.py        # Face/Hand/Pose landmarker wrappers (Tasks API, not legacy)
│   │   ├── blendshapes.py      # 52 ARKit coefficients + AU proxies
│   │   ├── headpose.py         # solvePnP rotation + gaze proxy
│   │   ├── extract.py          # video → landmark record
│   │   └── face_gate.py        # M0 face-visibility gate
│   ├── preprocess/             # normalize, interpolate, one-euro, window, augment  (M0 stub)
│   ├── features/               # manual, non-manual, prosody                        (M0 stub)
│   ├── models/                 # gru, stgcn, transformer                             (M0 stub)
│   ├── affect/                 # factorized encoder, GRL, vCLUB, probes              (M4)
│   ├── translate/              # gloss2text, pose2text, emotion conditioning         (M5/M6)
│   ├── avatar/                 # retargeting, blendshape map, modulation, export     (M7)
│   ├── serve/                  # fastapi, websocket, backpressure, auth guard       (M7)
│   ├── export/                 # onnx, quantization, vram guard                      (M2)
│   └── eval/                   # metrics, loso, benchmarks, tables                   (M1+)
├── tests/                      # unit, property, integration, benchmark, budget
├── scripts/                    # repro_all.sh, bench.sh, download_*.sh
├── artifacts/                  # gitignored: data cache, checkpoints, onnx, reports
└── paper/                      # main.tex, refs.bib, CLAIMS_LEDGER.md, EXPERIMENT_LOG.md
```

> `team.md` describes a 4-person squad with a dated 10-week calendar. Per `plan.md` §6 it is
> **superseded** for planning purposes; it is retained for role history and review-gate language.

---

## 1. Command map

| Command            | Purpose                                                 | Gate              |
| ------------------ | ------------------------------------------------------- | ----------------- |
| `make setup`     | editable install + pre-commit install                   | M0                |
| `make lint`      | `ruff check` + `ruff format --check` + `mypy src` | every milestone   |
| `make test`      | `pytest -q`                                           | every milestone   |
| `make doctor`    | env report: GPU, VRAM, mediapipe, ffmpeg, disk, network | M0                |
| `make data`      | `seam data fetch --all && seam data verify --all`     | M0                |
| `make readiness` | print the readiness table + blockers with owners        | **M0 gate** |
| `make landmarks` | extract landmarks+blendshapes for the EmoSign 200       | M0                |
| `make facegate`  | M0 face-visibility gate over ≥20 sampled clips         | **M0 gate** |
| `make bench`     | 3050 latency / VRAM harness                             | M2                |
| `make train`     | train from`configs/`                                  | M1+               |
| `make repro`     | regenerate every table and figure                       | M9                |
| `make paper`     | `latexmk -pdf paper/main.tex`                         | M9                |

---

## M0 — Data spine & engineering foundation · **GATE CLOSED 2026-09-26**

**Evidence gate met with margin.** EmoSign labels verified 1/1 · EmoSign video 200/200 downloaded
(87 MB), 200/200 decodable · MediaPipe task bundles 4/4 reused from local cache · face-visibility
gate **OPEN at 24/24** sampled, and the post-extraction census confirmed **200/200 (100%)** across
the whole benchmark · 75 tests green · ruff + mypy clean.

Measured and recorded: 24,733 frames · face rate median 1.000 (min 0.871) · median 31 of 52
blendshapes varying · brow SD median 0.262 · brow coefficients dominate the variance ·
9 coefficients are constant and droppable · 3-graph concurrency gives 1.63× · 14.9 FPS, short of
the 20 FPS target, deferred to M2 with the load confound named.

**Three defects the M0 tests caught**, all of the plausible-but-wrong kind:

1. `manifest_for` scanned single-file resources as directories, so the EmoSign labels reported
   "missing" immediately after a successful fetch.
2. The blendshape basis was written from the ARKit documentation; the real `face_landmarker.task`
   emits `_neutral` + 51 coefficients including `eyeLookUp*` and excluding `tongueOut`. A
   permuted basis inverts every brow-raise rule while every shape check still passes.
3. `reset_clock()` and `seed_worker()` referenced names not in scope — latent `NameError`s on
   paths that only a fork or a restart would reach.

### M0.1 Documentation · DONE

- [X] `plan.md` v2 written (supersedes v1; v1 backed up outside the repo)
- [X] `implimentation.md` written (this file)
- [X] `project_breakdown.md` §4/§5/§6/§8 updated: add Funakoshi & Zhu 2026 + Silva et al. 2020 to
  Related Work; replace the dataset table with the measured §1 table; retire the
  "MediaPipe Tasks HolisticLandmarker" and "ASL Citizen video" claims
- [X] `README.md` document table + four-numbers block updated to v2 KPIs
- [X] `CLAIMS_LEDGER.md`: add K1 (confound bias) as a headline claim; retarget C3's bar
  20.76 → 21.09; add the positive-control claim
- [X] `EXPERIMENT_LOG.md`: append the M0 resource-availability findings

### M0.2 Engineering scaffold

- [X] `pyproject.toml`, exact pins taken from the working `slr` env
- [X] `Makefile` with the targets in §1
- [X] `.gitignore`: `artifacts/`, `*.npz`, `*.pt`, `*.onnx`, `*.parquet`, `.env`, `node_modules/`
- [X] `.pre-commit-config.yaml`: ruff, ruff-format, trailing-whitespace, end-of-file
- [X] `configs/base.yaml` + `configs/data/*.yaml`
- [X] `src/seam/{__init__,paths,config,logging,seed}.py`
- [X] `src/seam/cli.py` with subcommands stubbed, `seam` console script

- **Test:** trivial import test; `make lint` clean; `make test` green.
- **DoD:** `pip install -e .` works in the `slr` env; `seam --help` runs.
- **Demo:** `seam doctor` prints GPU/VRAM/disk/env report.

### M0.3 Resource registry + manifest + fetcher

- [X] `data/sources.py` — the §1 table as typed records: id, kind, url(s), license status,
  expected bytes/sha where known, local reuse path, role, blocking milestone
- [X] `data/manifest.py` — SHA256 per file, resumable, **truncation detection** (a zip/mp4 whose
  hash or probe fails is flagged `truncated`, never silently used)
- [X] `data/fetch.py` — three strategies per resource: `reuse` (hardlink an existing local path),
  `download`, `stream` (per-file for HF folder trees). Never re-download what is local.

- **Test:** manifest round-trip; a deliberately truncated fixture is flagged; the registry has no
  entry with `license_status != "granted"` that a loader will actually read.
- **DoD:** `seam data fetch --all` completes with every resource in `ok` or an explicit
  `blocked` + owner.
- **Demo:** `seam data fetch --all` prints per-resource actions taken (reused vs downloaded).

### M0.4 EmoSign loader + ASLLRP join

- [X] `data/emosign.py` — load the 200-row CSV; parse `video_name` into
  `(signer, year, session, utterance_id)`; the join key is the **trailing numeric token**
- [X] assert 200/200 join; raise loudly on any miss, never silently drop a row
- [X] expose the 3 reasoning columns as a first-class `annotator_cues` field (M3 depends on it)

- **Test:** the join is exact for all 200 rows; signer counts are
  Cory 87 / Jonathan 54 / Rachel 52 / Ben 7; `Rachel_2011` and `Rachel_2012` map to one LOSO
  fold; a mangled `video_name` raises.
- **DoD:** an EmoSignFrame table with `signer, utterance_id, sentiment7, 10 emotion intensities, 3 cue strings`.
- **Demo:** `seam data emosign` prints the class distribution + signer table.

### M0.5 Fetch the 200 clips

- [X] Stream `crop_original_video.mp4` per utterance ID into `artifacts/data/emosign/video/<id>.mp4`
- [X] Verify every file is a decodable video (ffprobe), not just a non-zero byte count
- [X] Record native fps / frame count / WxH per clip

- **Test:** all 200 present and decodable; a zero-byte or non-video fixture is rejected.
- **DoD:** 200/200 clips on disk, manifest recorded.
- **Demo:** the count, the total bytes, and the fps/frame-count summary table.

### M0.6 Blendshape-capable perception + face-visibility gate

- [X] `perception/tasks_api.py` — compose **FaceLandmarker** (with
  `output_face_blendshapes=True`, 52 ARKit coefficients) + 2× **HandLandmarker** +
  **PoseLandmarker**. v1's "HolisticLandmarker" does not exist; do not look for it.
- [X] `perception/blendshapes.py` — **folded into `tasks_api.py`**; the basis, index map and ordering assertion live with the extractor that produces them, which is where a permuted basis would be caught — 52-coefficient record + brow/mouth groupings + AU proxies
- [ ] `perception/headpose.py` — **deferred to M1**; the 4×4 `facial_transformation_matrixes` are already persisted per frame, so the solvePnP wrapper is needed only when a consumer appears — `solvePnP` head rotation + gaze proxy from iris
- [X] `perception/extract.py` — video → landmark record (`.npz` shards + JSON sidecar), resumable
- [X] `perception/face_gate.py` — over ≥20 sampled clips: face-detected rate, blendshape
  activation rate, landmark counts, native fps spread

- **Test:** golden-file shape/count test on the sampled clips; the landmark ordering contract is
  asserted, not assumed; a clip with no face is handled by the documented missing policy, not
  by a crash.
- **DoD:** face gate report produced and its verdict recorded.
- **Demo:** per-stage millisecond readout + a landmark/blendshape overlay frame dump.

### M0.7 Quality gate

- [X] `make lint` clean, `make test` green (**75 tests**), `mypy src` clean
- [ ] `paper/EXPERIMENT_LOG.md` appended with the M0 findings and the mirror decision

### M0.8 Readiness gate — **the M0 evidence gate**

- [X] `seam data verify --all` → readiness table:
  `resource · license · bytes · integrity · role · status · blocker · owner`
- [X] every `blocked` row has a named owner and a next action
- [X] `artifacts/reports/readiness.md` and `.csv` written

- **DoD:** table generated, checked in, every blocker owned.
- **Demo:** the table, printed.

---

## M1 — The confound audit ★ · **GATE CLOSED 2026-09-27 — C1 refuted on WLASL**

- [X] `preprocess/normalize.py` — interpolate (part-slice aware), shoulder-center + scale
  normalize, One Euro, 12 fps resample, windowing. Translation/scale invariance and One-Euro
  causality asserted as properties.
- [X] `features/prosody.py` — speed, peak speed, amplitude, signing-space volume, repetition
  (autocorrelation peaks), pause fraction/mean, jerk, sign count, active fraction
- [X] `features/markers.py` — brow-raise / brow-furrow / mouth-morpheme / head-shake / head-nod,
  plus a `mouth_positive` **control** marker. Clip-relative `median + k·MAD` thresholds,
  every constant stated in the module docstring
- [X] `data/wlasl.py` — annotation join, decodability probe, failure classification, repair
- [X] `eval/fer.py` + `eval/fer_audit.py` — non-signer FER baselines, face cropping, matched-pair
  design, cluster bootstrap, uniform null model, MDE
- [X] `scripts/extract_wlasl.py`, `fetch_rafdb*.py`, `train_fer.py`, `run_confound_audit.py`,
  `diagnose_fer_sensitivity.py`, `diagnose_fer_affect.py`
- [X] Run the audit on ≥500 clips — **2,565 clips, 189,352 frames, 4,168 scorable windows**
- [X] **MDE added to every result** and asserted by test
- [X] FER front end fixed — grayscale + histogram equalisation + per-image z-score, **one
  function on both the fitting and inference paths**
- [X] Positive control for the instrument (`diagnose_fer_affect.py`): does the FER read-out track
  true affect on sign video at all
- [X] Full suite **140 passing**; ruff + mypy clean

**Result.** No marker-induced negative bias is detectable. The two effects that reach
significance run the *wrong way* and do not replicate across models. MDE 0.0030–0.0083 = 1.0–2.8%
of the 0.297 baseline, over 202–329 clips. `mouth_positive` control null in all three models;
uniform null model null in all six rows; a planted +0.20 bias is recovered by the same code path.

**The plan change M1 forces:** the confound is a **discourse** phenomenon, and WLASL is isolated
dictionary signing with no discourse context. →

- [ ] **Move the marker labeller and M4's LOSO evaluation to continuous signing** (ASLLRP 200
  utterances, local; or How2Sign keypoints). This is the testable version of C1.

### Deviations from the original M1 sketch, and why

1. **WLASL is 2,657 clips, not 3,863.** The on-disk filename is `<gloss>/<instance_id>.mp4`, and
   1,206 of the 3,863 files are `_yt.mp4.part.mp4` *duplicates* of instances that also have a
   good copy. `plan.md` §1 is corrected.
2. **The 92 undecodable clips are not truncated video — they are HTML error pages** saved with
   an `.mp4` extension (first bytes `<!DOCTYPE html>`). No re-cut can recover them; only a
   re-download can. Failures are now *classified* (`html_placeholder` / `truncated` /
   `no_index_other`) because the three have different remedies. **0 of 2,657 are genuinely
   truncated**, so the prior system's "1,130 repairable" population does not exist in this copy.
3. **The FER models are trained here, not downloaded.** "Non-signer FER model" must mean a model
   whose training distribution is *known* to be non-signers and whose accuracy on that
   distribution can be quoted. RAF-DB, no pretrained backbone — ImageNet weights are neither
   signer nor non-signer data, and importing one would make "trained only on non-signers" quietly
   untrue. RAF-DB accuracy is reported next to every audit result.
4. **The read-out is negative probability *mass*, not argmax** — the claim is about a direction of
   misreading, so the statistic has to be continuous. `neutral` is in neither side, or the read-out
   answers a different question.
5. **The bootstrap resamples clips, not windows.** Windows within a clip are correlated; a
   window-level bootstrap would shrink the interval by ~√(windows per clip) and manufacture
   significance. A test asserts the clustering directly, by showing that two designs with equal
   mean effect but different concentration get different interval widths.
6. **Two null models.** A uniform-random classifier on the same matched pairs shows the
   *machinery* is sound. A positive control on EmoSign's own sentiment shows the *instrument* is
   live. Without both, a null is uninterpretable.
7. **T=24, not T=64.** T=64 came from the continuous-SLT design where a window is a phrase. WLASL
   is isolated signs: median 73 native frames, **33 at 12 fps**, and T=64 produced **zero windows
   on every clip in the corpus** while the audit cheerfully reported an empty table.
8. **Facial negation is not detectable here.** The canonical nose-wrinkle + tongue-out display
   needs `noseSneer*` / `tongueOut`, and `noseSneer*` carries **zero variance** on the EmoSign
   corpus. Negation is carried by the head shake alone, and the limitation is stated.

---

## M2 — Efficiency harness

- [X] `export/onnx_export.py` — FP32 export, INT8 quantisation, parity with a calibrated
  tolerance and an independent argmax gate
- [X] `export/vram_guard.py` — hard ceiling, fails loudly; enforces the larger of torch's
  and the driver's figure
- [X] `export/runtime.py` — preloads the bundled CUDA 13 libraries so the CUDA EP actually
  activates, and provides the nvidia-smi VRAM instrument torch cannot supply
- [X] `eval/bench.py` — p50/p95/p99 over per-call timings, peak VRAM, sustained FPS
- [X] `eval/bench.py::resident_stack_bench` — 6 models resident simultaneously
- [X] `onnxruntime-gpu` CUDA EP active (TensorRT absent, `libnvinfer.so.10` not installed)
- [X] **K6 re-measured on an idle, performance-governor machine: 20.4 FPS**
  (repeats 19.7–20.4; p50 41.8 / p95 73.2 / p99 78.1 ms over 900 timed calls). Reported
  as straddling the ≥20 target rather than as a pass.
- [X] Sequential baseline re-measured with the same instrument: 73.0 ms p50, 12.8 FPS,
  so concurrency is worth 1.75x
- [X] K5: 186 MB for 3 MediaPipe graphs + 3 ONNX graphs live at once, 7% of the 2500 MB
  ceiling
- [X] Benchmark now records governor, during-run core clock, load and memory, and refuses
  to call a run reportable without them. Six instrument defects fixed along the way
  (percentiles over run-means, `noise_ratio` blindness to sustained load, post-run clock
  sampling, an 80%-of-single-core-turbo gate, a conditionally-checked governor, and
  in-place report overwriting)
- [X] M1 instrument lesson resolved in code: one decoder, `seam.data.rafdb`, shared by the
  trainer and the parity harness, with the 768-space box mapping pinned by test
- [ ] Resolve the M1 instrument lesson in code: one front end, asserted equal by test (rule 16)

---

## M3 — Linguistic-marker supervision

- [X] Track A: `L` labeller with a provenance flag on every label — `features/syntactic.py`
  (syntactic half, from real ASLLRP gloss annotation) + `features/markers.py` (visual half)
- [X] fetched the ASLLRP gloss resources; **200/200 EmoSign utterances join** to the
  human-authored gloss map, so the syntactic track has a real input
- [X] agreement statistics for the heuristic labels, with base rates and a degeneracy
  check — `artifacts/audit/marker_labels.json`, 200 clips, gate met
- [X] established that `fs-` is compound-sign notation, not facial signal
- [X] fixed two head-path bugs in `markers._oscillation` (`head_angle/2` vs documented
  `head_angle`; clip-global `min_reversals`); head_shake 0.98 → 0.785, head_nod 0.65 → 0.170
- [ ] **re-point at continuous signing** (M1's finding): ASLLRP 200 utterances, where the
  interrogative / negation / topicalisation syntax of the English caption gives an
  independent handle on the marker that isolated signs do not
- [X] **clip-level marker calibration** — `markers.ClipCriteria` (run duration + peak +
  coverage) and `markers.clip_magnitude` (per-frame mean, duration-free)
- [X] fixed two scale bugs: MAD is the wrong scale for heavy-tailed blendshapes
  (normalise by `p95 - p50` instead), and `clip_presence` compared clip-range evidence
  against the MAD constant `k=1.5` so every blendshape marker read zero coverage
- [X] **caught a duration confound that would have been a false positive**: every
  syntactic label tracks clip length (interrogative r_pb=+0.68), and an integral
  statistic inherits it — interrogative/brow_raise went +0.414 (p=0.015) -> **-0.045**
  on a per-frame mean
- [X] **negation <-> head_shake: r=0.554, perm p=0.011, Bonferroni 0.033, MDE 0.411** —
  survives duration control; interrogative pairings are powered nulls
- [X] **cue-grounding report** — `scripts/cue_grounding.py`, `features/cues.py`;
  artefact `artifacts/audit/cue_grounding.json`
- [X] separated **motor cues** from **affective interpretations** in the annotator
  text (482/600 motor, 118 affective-only); the affective ones are FER ground truth,
  not marker-feature evidence, and counting them would have manufactured agreement
- [X] 15 cue/feature tests, duration-controlled, permutation + MDE + Bonferroni:
  **nothing survives** — the assumed feature set is *not* validated
- [X] the head channel is the one with independent corroboration: `head_shake` r=+0.275
  vs annotator text, and r=+0.554 (p=0.011) vs lexical negation in the previous run
- [X] `head_nod` reported BLIND (zero on 83% of clips) rather than as a null
- [X] **6 annotator-named cues have no feature at all** — `head_tilt`, `eye_widen`,
  `blink_close`, `gaze_shift`, `fingerspelling`, `body_posture` — a data-derived spec
  for M4's feature set
- [X] permutation null calibration asserted in the suite (median p near 0.5 on noise)
- [ ] Track B (async): BU access request for the real SignStream non-manual XML
- [ ] cue-grounding report: our prosody features vs the 600 Deaf-annotator cue strings

---

## M4 — Factorized encoder ★ core contribution

- [ ] `affect/grl.py`, `affect/vclub.py`, `affect/orthogonality.py`
- [ ] `affect/encoder.py` — `z_L`, `z_A`, heads, ≤1M params each, asserted by a test
- [ ] `eval/loso.py` — 4 folds, signer-disjoint, zero-overlap assertion
- [ ] `eval/probes.py` — frozen cross-probes → cross-prediction AUC
- [ ] **positive control:** signer probe, asserted ≥ 0.80
- [ ] per-λ ablation harness
- [ ] qualitative neutral-affect wh-question/negation set

---

## M5 — Recognition + translation

- [ ] M5a: continuous recognition on the 17,522 frame-aligned ASLLRP gloss tokens
- [ ] M5b: 255-dim pose → linear → T5-small on How2Sign keypoints, 24 fps and 12 fps
- [ ] FLOPs table; 3 seeds; signer-disjoint

---

## M6 — Emotion-conditioned generation

- [ ] paraphrase corpus build + semantic-equivalence filter + disclosure
- [ ] emotion control tokens in T5-small
- [ ] style accuracy / BERTScore-F1 / degeneracy guard / controllability test

---

## M7 — Avatar + live demo

- [ ] retargeting with joint clamping + quaternion continuity (gimbal-flip test)
- [ ] 52 blendshapes → VRM/ARKit expression map
- [ ] emotion modulation of gain, amplitude, timing
- [ ] client-side MediaPipe Tasks browser demo
- [ ] pre-registered preference study, ≥5 raters, inter-rater agreement

---

## M8 — Cross-lingual ablation

- [ ] NSL/INCLUDE integrity re-audit (v1's `.part` finding is stale)
- [ ] zero-shot then few-shot transfer; Top-1 vs chance

---

## M9 — Paper, repro, release

- [ ] `scripts/repro_all.sh`; CI test that fails on any unlogged number
- [ ] model cards; ethics, limitations, Deaf-community statement
- [ ] close the §6 provenance item
- [ ] arXiv + workshop submission
