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

| Command | Purpose | Gate |
|---|---|---|
| `make setup` | editable install + pre-commit install | M0 |
| `make lint` | `ruff check` + `ruff format --check` + `mypy src` | every milestone |
| `make test` | `pytest -q` | every milestone |
| `make doctor` | env report: GPU, VRAM, mediapipe, ffmpeg, disk, network | M0 |
| `make data` | `seam data fetch --all && seam data verify --all` | M0 |
| `make readiness` | print the readiness table + blockers with owners | **M0 gate** |
| `make landmarks` | extract landmarks+blendshapes for the EmoSign 200 | M0 |
| `make facegate` | M0 face-visibility gate over ≥20 sampled clips | **M0 gate** |
| `make bench` | 3050 latency / VRAM harness | M2 |
| `make train` | train from `configs/` | M1+ |
| `make repro` | regenerate every table and figure | M9 |
| `make paper` | `latexmk -pdf paper/main.tex` | M9 |

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
- [x] `plan.md` v2 written (supersedes v1; v1 backed up outside the repo)
- [x] `implimentation.md` written (this file)
- [x] `project_breakdown.md` §4/§5/§6/§8 updated: add Funakoshi & Zhu 2026 + Silva et al. 2020 to
      Related Work; replace the dataset table with the measured §1 table; retire the
      "MediaPipe Tasks HolisticLandmarker" and "ASL Citizen video" claims
- [x] `README.md` document table + four-numbers block updated to v2 KPIs
- [x] `CLAIMS_LEDGER.md`: add K1 (confound bias) as a headline claim; retarget C3's bar
      20.76 → 21.09; add the positive-control claim
- [x] `EXPERIMENT_LOG.md`: append the M0 resource-availability findings

### M0.2 Engineering scaffold
- [x] `pyproject.toml`, exact pins taken from the working `slr` env
- [x] `Makefile` with the targets in §1
- [x] `.gitignore`: `artifacts/`, `*.npz`, `*.pt`, `*.onnx`, `*.parquet`, `.env`, `node_modules/`
- [x] `.pre-commit-config.yaml`: ruff, ruff-format, trailing-whitespace, end-of-file
- [x] `configs/base.yaml` + `configs/data/*.yaml`
- [x] `src/seam/{__init__,paths,config,logging,seed}.py`
- [x] `src/seam/cli.py` with subcommands stubbed, `seam` console script
- **Test:** trivial import test; `make lint` clean; `make test` green.
- **DoD:** `pip install -e .` works in the `slr` env; `seam --help` runs.
- **Demo:** `seam doctor` prints GPU/VRAM/disk/env report.

### M0.3 Resource registry + manifest + fetcher
- [x] `data/sources.py` — the §1 table as typed records: id, kind, url(s), license status,
      expected bytes/sha where known, local reuse path, role, blocking milestone
- [x] `data/manifest.py` — SHA256 per file, resumable, **truncation detection** (a zip/mp4 whose
      hash or probe fails is flagged `truncated`, never silently used)
- [x] `data/fetch.py` — three strategies per resource: `reuse` (hardlink an existing local path),
      `download`, `stream` (per-file for HF folder trees). Never re-download what is local.
- **Test:** manifest round-trip; a deliberately truncated fixture is flagged; the registry has no
      entry with `license_status != "granted"` that a loader will actually read.
- **DoD:** `seam data fetch --all` completes with every resource in `ok` or an explicit
      `blocked` + owner.
- **Demo:** `seam data fetch --all` prints per-resource actions taken (reused vs downloaded).

### M0.4 EmoSign loader + ASLLRP join
- [x] `data/emosign.py` — load the 200-row CSV; parse `video_name` into
      `(signer, year, session, utterance_id)`; the join key is the **trailing numeric token**
- [x] assert 200/200 join; raise loudly on any miss, never silently drop a row
- [x] expose the 3 reasoning columns as a first-class `annotator_cues` field (M3 depends on it)
- **Test:** the join is exact for all 200 rows; signer counts are
      Cory 87 / Jonathan 54 / Rachel 52 / Ben 7; `Rachel_2011` and `Rachel_2012` map to one LOSO
      fold; a mangled `video_name` raises.
- **DoD:** an EmoSignFrame table with `signer, utterance_id, sentiment7, 10 emotion intensities,
      3 cue strings`.
- **Demo:** `seam data emosign` prints the class distribution + signer table.

### M0.5 Fetch the 200 clips
- [x] Stream `crop_original_video.mp4` per utterance ID into `artifacts/data/emosign/video/<id>.mp4`
- [x] Verify every file is a decodable video (ffprobe), not just a non-zero byte count
- [x] Record native fps / frame count / WxH per clip
- **Test:** all 200 present and decodable; a zero-byte or non-video fixture is rejected.
- **DoD:** 200/200 clips on disk, manifest recorded.
- **Demo:** the count, the total bytes, and the fps/frame-count summary table.

### M0.6 Blendshape-capable perception + face-visibility gate
- [x] `perception/tasks_api.py` — compose **FaceLandmarker** (with
      `output_face_blendshapes=True`, 52 ARKit coefficients) + 2× **HandLandmarker** +
      **PoseLandmarker**. v1's "HolisticLandmarker" does not exist; do not look for it.
- [x] `perception/blendshapes.py` — **folded into `tasks_api.py`**; the basis, index map and ordering assertion live with the extractor that produces them, which is where a permuted basis would be caught — 52-coefficient record + brow/mouth groupings + AU proxies
- [ ] `perception/headpose.py` — **deferred to M1**; the 4×4 `facial_transformation_matrixes` are already persisted per frame, so the solvePnP wrapper is needed only when a consumer appears — `solvePnP` head rotation + gaze proxy from iris
- [x] `perception/extract.py` — video → landmark record (`.npz` shards + JSON sidecar), resumable
- [x] `perception/face_gate.py` — over ≥20 sampled clips: face-detected rate, blendshape
      activation rate, landmark counts, native fps spread
- **Test:** golden-file shape/count test on the sampled clips; the landmark ordering contract is
      asserted, not assumed; a clip with no face is handled by the documented missing policy, not
      by a crash.
- **DoD:** face gate report produced and its verdict recorded.
- **Demo:** per-stage millisecond readout + a landmark/blendshape overlay frame dump.

### M0.7 Quality gate
- [x] `make lint` clean, `make test` green (**75 tests**), `mypy src` clean
- [ ] `paper/EXPERIMENT_LOG.md` appended with the M0 findings and the mirror decision

### M0.8 Readiness gate — **the M0 evidence gate**
- [x] `seam data verify --all` → readiness table:
      `resource · license · bytes · integrity · role · status · blocker · owner`
- [x] every `blocked` row has a named owner and a next action
- [x] `artifacts/reports/readiness.md` and `.csv` written
- **DoD:** table generated, checked in, every blocker owned.
- **Demo:** the table, printed.

---

## M1 — The confound audit ★
- [ ] `preprocess/` normalize + interpolate + One-Euro + 12 fps resample + T=64 window
- [ ] `features/prosody.py` — speed, amplitude (signing-space volume), repetition (autocorrelation
      peaks), pause duration, jerk, sign-duration stats
- [ ] `features/markers.py` — brow-raise / brow-furrow / head-shake / mouth-morpheme rules on
      blendshapes, each with a documented threshold and a synthetic-signal unit test
- [ ] `eval/fer_audit.py` — run ≥3 pretrained non-signer FER models over the non-manual channel
- [ ] matched-segment design: marker-bearing vs marker-free, matched on duration, signer, and
      signing-space volume
- [ ] stats: per-model bias with bootstrap CIs, effect sizes, marker→emotion confusion matrix
- **Test:** a scripted fast/large motion scores higher speed/amplitude than a slow/small one;
      a synthetic brow-raise is detected; marker rate is stable across input fps and resolution.
- **Demo:** the bias table + the marker→emotion confusion matrix.

## M2 — Efficiency harness
- [ ] `export/onnx_export.py` + `export/parity.py` (FP32↔INT8 tolerance test)
- [ ] `export/vram_guard.py` — hard ceiling, fails loudly
- [ ] `eval/bench.py` — p50/p95/p99, peak VRAM, FPS on the 3050
- [ ] add `onnxruntime-gpu` for the CUDA EP
- **Test:** CI test that fails if peak VRAM > 2500 MB or p95 > 400 ms.
- **Demo:** the Pareto table.

## M3 — Linguistic-marker supervision
- [ ] Track A: heuristic `L` labeller with a provenance flag on every label
- [ ] Track B (async): BU access request
- [ ] cue-grounding report: our prosody features vs the 600 Deaf-annotator cue strings
- [ ] agreement statistics for the heuristic labels

## M4 — Factorized encoder ★ core contribution
- [ ] `affect/grl.py`, `affect/vclub.py`, `affect/orthogonality.py`
- [ ] `affect/encoder.py` — `z_L`, `z_A`, heads, ≤1M params each, asserted by a test
- [ ] `eval/loso.py` — 4 folds, signer-disjoint, zero-overlap assertion
- [ ] `eval/probes.py` — frozen cross-probes → cross-prediction AUC
- [ ] **positive control:** signer probe, asserted ≥ 0.80
- [ ] per-λ ablation harness
- [ ] qualitative neutral-affect wh-question/negation set

## M5 — Recognition + translation
- [ ] M5a: continuous recognition on the 17,522 frame-aligned ASLLRP gloss tokens
- [ ] M5b: 255-dim pose → linear → T5-small on How2Sign keypoints, 24 fps and 12 fps
- [ ] FLOPs table; 3 seeds; signer-disjoint

## M6 — Emotion-conditioned generation
- [ ] paraphrase corpus build + semantic-equivalence filter + disclosure
- [ ] emotion control tokens in T5-small
- [ ] style accuracy / BERTScore-F1 / degeneracy guard / controllability test

## M7 — Avatar + live demo
- [ ] retargeting with joint clamping + quaternion continuity (gimbal-flip test)
- [ ] 52 blendshapes → VRM/ARKit expression map
- [ ] emotion modulation of gain, amplitude, timing
- [ ] client-side MediaPipe Tasks browser demo
- [ ] pre-registered preference study, ≥5 raters, inter-rater agreement

## M8 — Cross-lingual ablation
- [ ] NSL/INCLUDE integrity re-audit (v1's `.part` finding is stale)
- [ ] zero-shot then few-shot transfer; Top-1 vs chance

## M9 — Paper, repro, release
- [ ] `scripts/repro_all.sh`; CI test that fails on any unlogged number
- [ ] model cards; ethics, limitations, Deaf-community statement
- [ ] close the §6 provenance item
- [ ] arXiv + workshop submission
