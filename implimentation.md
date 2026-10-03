
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
- [ ] `perception/headpose.py` — **ACTIONABLE, cheap.** The 4x4s are persisted (`output_facial_transformation_matrixes=True` in `perception/tasks_api.py:307`) but no module turns them into head yaw/pitch/roll. The M7 demo already needs this; it is unconsumed data sitting in the landmarks. — **deferred to M1**; the 4×4 `facial_transformation_matrixes` are already persisted per frame, so the solvePnP wrapper is needed only when a consumer appears — `solvePnP` head rotation + gaze proxy from iris
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
- [x] `paper/EXPERIMENT_LOG.md` appended with the M0 findings and the mirror decision — **done**, log item 3 records the mirror's gloss tokens and the absent SignStream non-manuals

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

- [x] **Move the marker labeller and M4's LOSO evaluation to continuous signing** — **done**: M4 runs on the ASLLRP 200 utterances (`artifacts/m4/factorizer_multilabel.json`) (ASLLRP 200
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
- [ ] Resolve the M1 instrument lesson in code: one front end, asserted equal by test — **ACTIONABLE, cheap.** `seam.data.rafdb` *is* the single front end now (M1's fix), but `tests/test_rafdb_bbox.py` has 4 tests and none asserts the two FER paths decode identically. The lesson is half-landed. (rule 16)

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
- [x] **re-point at continuous signing** (M1's finding) — **done**, same run: ASLLRP 200 utterances, where the
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
- [X] `serve/app.py` + `web/index.html` + `seam serve` — live demo, client-side MediaPipe,
  no video leaves the browser; withholds affect and recognition with their measured reasons
- [X] `make serve-check` HTTP smoke gate (caught a 44-byte 404 page that started "successfully")
- [ ] M7 human preference study (≥5 raters, inter-rater agreement, signer participation stated) — **BLOCKED twice over, and the first blocker is not the people.**
  - **Verified 2026-09-30: there are no stimuli to show.** `src/seam/web/index.html` has zero canvas/three.js/model-viewer/GLB references — `render()` draws a table of six marker magnitudes — and `src/seam/serve/app.py` does not import `seam.avatar` at all. The avatar is a tested library (22 tests) that **nothing renders and no pipeline drives.**
  - So the missing prerequisite is software, not raters: landmarks → SMPL-X parameters → real mesh → video, per condition. Plus a real mesh exporter, since `export_glb()` writes JSON to a `.glb` path and `trimesh` is not installed.
  - **Dependency order:** step 1 (SMPL-X) blocks this hard; the M4 decision must be settled before stimuli are generated, because it decides whether affect appears; the DWPose corpus does **not** block it (different track — the avatar is driven by perception landmarks, not by a recogniser); the BU request does not block it (external outage, and running on heuristic markers with the limitation stated is honest).
  - Once stimuli exist, the raters are needed — and raters are calendar time, so line them up early.
- [x] VRM/GLB avatar export: joint-limit clamping + quaternion-continuity tests — **done 2026-09-29** (`src/seam/avatar/synthesis.py`, 22 tests)
- [ ] Real mesh export + stimulus pipeline — **built 2026-09-30; real geometry still gated.** `src/seam/avatar/mesh.py` (15 tests) writes real binary glTF and verifies it by re-loading; `scripts/make_stimuli.py` runs landmarks → SMPL-X params → joint positions → mesh → `.glb` on real extracted landmarks. `trimesh==4.7.1` pinned as the `avatar` extra.
  - **The old `export_glb()` that wrote a JSON parameter dump to a `.glb` path is removed.** That was the defect: a file that passes `ls` and opens as nothing in a viewer. `mesh.export_glb` writes `glTF`-magic bytes, re-loads them, and checks the geometry count against the frame count; the parameter dump moved to `mesh.export_parameters`, which writes `.json` deliberately.
  - Added `synthesis.forward_kinematics()`, which was missing entirely — nothing could be *placed* in the body before this. It uses `canonical_rest_pose()` rather than the licence-gated model, so the chain works before approval arrives. Verified by the invariant that zero rotation reproduces the rest pose exactly.
  - **Output today is a joint-capsule proxy, not a human body**, stamped `is_proxy: true` in the GLB metadata, `is_human_mesh: false` in the per-clip sidecar, and repeated in `manifest.json`. **Do not collect preference-study ratings on proxy output** — raters would be judging the stand-in, not the retargeting.
  - `mesh.smplx_mesh()` is the one remaining function, and it raises `NotImplementedError` on purpose: returning a proxy from a function named `smplx_mesh` would be indistinguishable from success at the call site, and the study would silently measure the stand-in.
  - After the weights land, two gaps remain: implement `smplx_mesh`, and render the `.glb` to mp4 with blinded filenames for the raters.
- [X] M5a data layer: `data/asllrp.py` — tolerant CSV parser (bare inner quotes),
  both source-collection signer forms, 17,522 tokens / 0 malformed rows
- [!] **M5a BLOCKED on frame alignment.** ASLLRP token indices are absolute positions in a
  long session recording (median utterance span 5,123 frames ≈ 2.8 min); the EmoSign clips
  are 4.6 s excerpts (median 109 frames). 1,725 of 1,738 overlapping tokens fall outside
  the extracted video, and the ratio spans 1.00–222.33 so it is not a rescale. Encoded as
  `asllrp.check_alignment(...).aligned` with tests. **Route: the `Sign video filename`
  column — 17,522 isolated sign clips, each one gloss, alignable by construction.**
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
- [ ] Track B (async): real SignStream non-manual XML — **researched 2026-09-30; the obstacle is misdiagnosed and mostly self-service.**
  - **The non-manuals were never in the mirror.** The `17,522 tokens / 4 signers` download is the Sign Bank *sign-clip* set, whose columns (ASLLRP Report 24) are gloss, frames, handshape and sign type — no non-manual columns. The Hugging Face mirror was built from it, so it is not missing anything; it is the wrong artifact. The non-manuals are a *separate* download from `dai.cs.rutgers.edu/dai/s/dai`, whose XML export contains a `<NON_MANUALS>` block (ASLLRP Report 18 §8.3).
  - **Access is a free self-service account, not an approval** (Report 18 §8.1: the account exists "to help keep track of prior downloads"). No PI letter or IRB documented.
  - Contact confirmed current: **carol@bu.edu** (Carol Neidle, Director ASLLRP). DAI support: augustine.opoku@gmail.com.
  - Drafted, ready to review and send: `paper/provenance/bu_access_request.md`, with the research, the confirmed/unconfirmed split, the field IDs, and the no-redistribution terms.
  - **Also unblocked:** the real labels will make it possible to check the project's marker definitions (`eye brows` → brow raise/furrow, `head mvmt: shake` → head shake, `mouth` → mouth morpheme). Those definitions were written from blendshape heuristics and have **never** been validated against the real annotations.
  - **RESOLVED 2026-10-01.** Portal recovered; Professor Neidle replied and confirmed the Download Cart route at `/dai/s/cart`. **51 XML collections downloaded** (Ben 10, Cory 10, Jonathan 8, Rachel 19, RIT 4).
  - **Parsed and joined: 2,407 utterances, 21,902 signs, 43,038 non-manual events, all frame-aligned, 100% mapped to markers.**
  - **The join that unblocks M3: 200/200 of our EmoSign utterance IDs appear directly in the XML**, carrying **4,443 human non-manual annotations** across all 24 markers — brow_raise 282, brow_furrow 202, head_shake 125, head_nod 111, negation 41, question_wh 9, question_yn 25, topic 102, plus eye_aperture 1,315 and head position/posture channels.
  - **M3's "blocked on the visual instrument" limitation is lifted.** The marker labels are no longer heuristic pseudo-labels from blendshapes.
- [x] cue-grounding report: our prosody features vs the 600 Deaf-annotator cue strings — **done**, `artifacts/audit/cue_grounding.json`

---

## M4 — Factorized encoder ★ core contribution

- [x] `affect/grl.py`, `affect/vclub.py`, `affect/orthogonality.py` — **all exist**
- [x] `affect/encoder.py` — `z_L`, `z_A`, heads, ≤1M params each, asserted by a test — **done**
- [x] `eval/loso.py` — 4 folds, signer-disjoint, zero-overlap assertion — **done**
- [x] `eval/probes.py` — frozen cross-probes → cross-prediction AUC — **done**
- [x] **positive control:** signer probe, asserted ≥ 0.80 — **done**, measured 0.973
- [x] per-λ ablation harness — **done**, `scripts/train_factorizer.py --ablate` runs the full grid
- [ ] qualitative neutral-affect wh-question/negation set — **not started.** Doable now that M3 has the interrogative/negation labels; needs a curated frame set.

---

## M5 — Recognition + translation

- [x] M4 gate decision — **decided 2026-10-01: option (c) attempted and refuted, so option (a).** The "more data" branch was tried as the decisive test, and the decisive move turned out to be *better labels* rather than more of them, available immediately after step 3. Re-running the whole experiment with human `y_L` in the same four slots moved worst-fold cross-AUC 0.7276 → **0.7031**, gate still failing, with cross A→L getting worse (0.505 → 0.617). The label-noise explanation is refuted. M4 is reported as a refuted hypothesis with a validated instrument; gate not moved to fit.
  - Underpinning it: three of four heuristic linguistic labels are at or near chance against human annotation (kappa 0.028 / 0.038 / 0.141; negation 0.639) — `artifacts/m3/label_agreement.json`, measured with Cohen's kappa rather than majority-dominated accuracy.
- [x] M5a: continuous recognition on the 17,522 frame-aligned ASLLRP gloss tokens — **pipeline built and run 2026-09-29; result is NEGATIVE, and that is the finding.**
  - `src/seam/features/signpose.py` + `scripts/train_recogniser.py`; artifact `artifacts/m5a/recogniser.json`.
  - 1,563 alignable tokens over 499 glosses (175 dropped as overshooting their crop, counted not clamped). 284 of 499 types are hapax.
  - **OOV floor 24-48% by fold** (30.6% on Cory): a third of one signer's tokens use a gloss the other three never use, so the open-vocabulary WER is bounded below by the data, not the model.
  - **WER 0.916 against a most-frequent baseline of 0.916 and a shuffled-label control of 0.911** - the model is indistinguishable from always predicting the most common gloss. Closed-vocabulary WER 0.875. Removing duration changes nothing (0.916).
  - So signing-space pose over these 200 utterances carries **no usable lexical signal** at this scale. The honest reading is that 200 utterances is too little to learn 499 classes from 3.1 tokens each; this is a data-scale verdict, not proof that pose is uninformative. Scaling to the 49k-frame DWPose corpus is the next test.
  - Four real bugs found and fixed en route, all of which had made the result look *better* than it was: an inverted `wer()`, a `predict` that argmaxed features instead of logits, a units bug reporting the 29.5% OOV floor as 0.1%, and a 180-degree-ambiguous levelling rotation. Each has a regression test.
- [ ] M5b: 255-dim pose → linear → T5-small on How2Sign keypoints, 24 fps and 12 fps
- [x] Step 4 (DWPose corpus) — **run 2026-10-03; the task as written was wrong.**
  - The 1.17 GB tar extracts to 1,355 utterance directories with 149,852 crop JPEGs and 1,354 `crop_original_video.mp4` files — **and zero DWPose `.npz`**, despite the repo listing advertising ~49,000 of them. Verified by `find` for `results_dwpose` in the extracted tree: nothing.
  - Size verified exactly: `1169520640` bytes, sha256 `1bdbf59d81fd4839ccc12f0695c6935c341346c28c6c4236b3e3727187abcc66`. The API reports no `lfs.oid` for it, so the checksum has nothing to be checked against — recorded rather than claimed as verified.
  - Fetching the pose separately wants **115,457 files at 3–5/s ≈ 8 hours.** Stopped deliberately; per-file HTTP latency, not bandwidth, is the cost.
  - **What we have instead:** 1,354 videos at 256×256 whose ids **all 1,355 intersect the gloss token table** — about 6.8× our current corpus. Only 138 of our 200 EmoSign clips are among them.
  - **Next:** run our own gated perception pipeline over the crop videos rather than downloading someone else's pose. — **not started**, needs the How2Sign keypoints fetched.
- [ ] FLOPs table; 3 seeds; signer-disjoint — **partial.** `--seed` exists and folds are signer-disjoint; the FLOPs table is not measured.

---

## M6 — Emotion-conditioned generation

- [ ] paraphrase corpus build + semantic-equivalence filter + disclosure — **not started**, whole-corpus build.
- [ ] emotion control tokens in T5-small — **not started**, depends on the line above.
- [ ] style accuracy / BERTScore-F1 / degeneracy guard / controllability test — **not started**, depends on the line above.

---

## M7 — Avatar + live demo

- [x] retargeting with joint clamping + quaternion continuity (gimbal-flip test) — **done**, SMPL-X body + MANO + FLAME
- [ ] 52 blendshapes → VRM/ARKit expression map — **partial.** `retarget_expression` projects to SMPL-X's 10 FLAME coefficients, which is the avatar path; the VRM/ARKit `expression`/blendshape-name map is not built.
- [ ] emotion modulation of gain, amplitude, timing — **not started**, depends on M4 clearing its gate.
- [x] client-side MediaPipe Tasks browser demo — **done**, `seam serve` + `make serve-check`
- [ ] pre-registered preference study, ≥5 raters, inter-rater agreement

---

## M8 — Cross-lingual ablation

- [ ] NSL/INCLUDE integrity re-audit (v1's `.part` finding is stale) — **partial.** NSL is in `artifacts/reports/readiness.csv`; INCLUDE is not.
- [ ] zero-shot then few-shot transfer; Top-1 vs chance — **not started.** Depends on M5a/M5b existing first.

---

## M9 — Paper, repro, release

- [x] `scripts/repro_all.sh`; CI test that fails on any unlogged number — **done 2026-09-29.** `scripts/repro_all.sh` runs 11 stages in dependency order, refuses to benchmark on a loaded machine, and `make repro` now has a target that works. `tests/test_provenance.py` fails on any number in `EXPERIMENT_LOG.md`/`CLAIMS_LEDGER.md` that no artifact under `artifacts/` produced, with `paper/provenance_exemptions.json` as a ratchet: 44 audited historical lines are waived by stated reason, and both the waiver list and the value list self-clean when a line stops needing an exemption. `.github/workflows/ci.yml` runs lint/typecheck/tests plus the provenance guard.
- [ ] model cards; ethics, limitations, Deaf-community statement — **partial.** `paper/main.tex` carries ethics/limitations/Deaf text; there is no model card per component.
- [ ] close the §6 provenance item — **not started.** Depends on the NSL/INCLUDE re-audit above.
- [ ] arXiv + workshop submission — **not started**, and premature until M5a/M6 exist.
