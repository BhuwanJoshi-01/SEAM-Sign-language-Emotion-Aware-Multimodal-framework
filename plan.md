# SEAM — Execution Plan

**Audience: the AI agent implementing this project.** This file is the single executable checklist.
Read `project_breakdown.md` for *why*; read this for *what to do next*. `team.md` maps tasks to
humans. Update the checkboxes and the KPI table as you go — this file is state, not prose.

- **Timeline:** 10 weeks, Mon 2026-08-10 → Sun 2026-10-18
- **Team:** 4 humans (R1–R4, see `team.md`) + AI implementer
- **Train on:** RTX 4060 16 GB (college) + Colab/Kaggle
- **Deploy on:** RTX 3050 4 GB — hard ceiling, peak inference VRAM < 2500 MB
- **Paper target:** arXiv preprint + workshop (CVPR/ICCV MSLR or LREC SignLang); journal extension after

---

## 0. Agent Operating Rules

These are binding. Violating them is how this project fails.

1. **Test first.** Every task lists tests. Write them before or with the implementation. `pytest`
   must be green before a task is marked done.
2. **Every task ends in a demo.** If it cannot be demonstrated, it is not done.
3. **No number without a run ID.** Any figure that could appear in the paper gets logged to W&B
   and appended to `paper/EXPERIMENT_LOG.md` with its run ID, commit SHA, config, and date.
4. **LOSO or nothing on EmoSign.** 200 clips, 4 signers. Never use a random split. Always report
   mean ± std over the 4 signer-held-out folds.
5. **Measure latency and VRAM on the 3050, never the 4060, never estimated.**
6. **Never redistribute dataset video.** ASL Citizen forbids it; ASLLRP is access-controlled;
   EmoSign ships IDs+labels only. Release code, weights, and labels.
7. **The serving layer is localhost-only with no auth by default.** It streams webcam biometrics.
   Any change to binding or exposure requires an explicit auth+TLS story and human sign-off.
8. **Do not overclaim.** EmoSign's average Krippendorff α is 0.593 (surprise− 0.119, disgust 0.166).
   High per-class accuracy on low-α classes is a bug signal, not a result. Say so in the log.
9. **Pin dependency versions exactly.** No open ranges.
10. **When blocked, escalate in writing** to the task owner in `team.md` rather than silently
    substituting an approach. Fallbacks that are already documented here may be taken freely,
    but must be recorded as taken.

---

## 1. Phase Map

| Phase | Sprint | Weeks | Dates | Tasks | Exit gate |
|---|---|---|---|---|---|
| **P1 Foundations** | S0 | 1–2 | Aug 10 – Aug 23 | T1–T4 | Perception ≥20 FPS on 3050; dataset readiness table produced; ASLLRP + EmoSign access requests submitted |
| **P2 Recognition** | S1 | 3–4 | Aug 24 – Sep 6 | T5–T6 | First full vertical slice: video → gloss → English |
| **P3 Affect** | S2 | 5–6 | Sep 7 – Sep 20 | T7–T9 | EmoSign affect baseline beats GPT-4o video-only (wF1 > 20.76) under LOSO |
| **P4 Contribution** | S3 | 7–8 | Sep 21 – Oct 4 | T10–T12 | Disentanglement gain demonstrated (cross-pred AUC ≤0.60 with no affect-F1 loss); Phase-B BLEU logged |
| **P5 Ship** | S4 | 9–10 | Oct 5 – Oct 18 | T13–T16 | Live demo inside 2500 MB VRAM; paper submitted; every table reproducible by one command |

**Declared cut-lines if the schedule slips** (cut in this order, do not cut anything else):
1. T16's INCLUDE cross-lingual ablation
2. T12 Phase-B continuous translation
3. T15's GLB/VRMA export (keep the live viewer)

Never cut: T10 (the contribution), T8 (the benchmark comparison), T13 (the efficiency result),
LOSO methodology, or the ethics/limitations statement.

---

## PHASE 1 — FOUNDATIONS (Weeks 1–2)

### T1 — Documentation spine and repo scaffold
**Owner:** R4 · **Depends on:** — · **Est:** 2 days

- [ ] `project_breakdown.md` created and reviewed by all 4 members
- [ ] `plan.md` (this file) created
- [ ] `team.md` created
- [ ] `paper/main.tex` (IEEEtran), `refs.bib`, `PAPER_TEMPLATE.md`, `WRITING_GUIDE.md`,
      `CLAIMS_LEDGER.md`, `EXPERIMENT_LOG.md` created
- [ ] Python package scaffold under `src/seam/` per breakdown §11
- [ ] `pyproject.toml` with **exactly pinned** versions; `Makefile` targets:
      `setup lint test train bench repro paper`
- [ ] `pytest` + `ruff` + `mypy` + `pre-commit` configured and passing
- [ ] W&B project initialized; `configs/` skeleton committed
- [ ] `.gitignore` excludes `artifacts/`, video, checkpoints, `.env`

**Tests:** trivial import test; `make lint` clean; `latexmk` builds `main.tex` to PDF.
**DoD:** `pip install -e .` works, `make test` green, paper PDF builds with empty result tables.
**Demo:** four documents render; paper PDF opens; `make test` output shown.

---

### T2 — Dataset acquisition, licensing, and integrity gate
**Owner:** R1 · **Depends on:** T1 · **Est:** 4 days (access requests are async — fire them on day 1)

- [ ] **DAY 1, BEFORE ANYTHING ELSE:** submit ASLLRP data-access request (Boston University)
- [ ] **DAY 1:** accept HuggingFace gated terms for `catfang/emosign`
- [ ] **DAY 1:** submit DFEW and MAFW license requests
- [ ] Confirm free disk space on `/mnt/Volume2` and record it; abort bulk downloads if < 250 GB
- [ ] Implement `src/seam/data/acquire/` fetchers with SHA256 manifests per dataset
- [ ] Implement `seam data verify --all` → readiness table
      (dataset · license status · bytes · integrity · blocker)
- [ ] Run verify against `/home/bhuwan/Videos/data` and confirm the 12 truncated INCLUDE `.part`
      archives (incl. `Train_Test_Split`); record the list in the readiness report
- [ ] Download ASL Citizen (Microsoft Download Center)
- [ ] Download How2Sign **keypoints only** (never the 80 h of video)
- [ ] Fetch WLASL-100/300 splits; record the link-rot rate as a documented number
- [ ] Decide and record: re-download the 12 INCLUDE parts, or restrict the ISL ablation to intact
      categories (and say which in the paper)

**Tests:** manifest verification unit tests; fixture test proving a truncated zip is flagged;
license-status registry test (no dataset can be used before its status is `granted`).
**DoD:** readiness table generated, checked in, and every blocker has a named owner and a date.
**Demo:** `seam data verify --all` prints the readiness table — the artifact that tells the team
what is actually usable.

---

### T3 — Perception pipeline with measured throughput
**Owner:** R1 · **Depends on:** T1 · **Est:** 4 days

- [ ] MediaPipe **Tasks** `HolisticLandmarker` wrapper (legacy Holistic is superseded — do not use it)
- [ ] Emit per frame: 543 landmarks (33 pose + 21 + 21 hands + 468 face), **52 blendshapes**,
      head pose (rotation matrix / solvePnP), gaze proxy
- [ ] Configurable input fps (default 12) and CPU/GPU delegate switching
- [ ] Compressed `.npz` shard cache + manifest; resumable extraction
- [ ] Explicit NaN/missing-hand policy (documented, not implicit)
- [ ] YOLO11n person-detection path implemented but **off by default** (ablation only)

**Tests:** golden-file test on 3 sample videos (landmark counts, array shapes, NaN policy);
benchmark test asserting **≥20 FPS on the RTX 3050**; delegate-switch parity test.
**DoD:** 100 ASL Citizen clips extracted end-to-end with a throughput report.
**Demo:** webcam + sample-video run with a landmark overlay and per-stage millisecond readout.

---

### T4 — Preprocessing and augmentation, validated
**Owner:** R1 · **Depends on:** T3 · **Est:** 3 days

- [ ] Shoulder-centered normalization; shoulder-width scale normalization; optional rotation alignment
- [ ] Missing-hand interpolation; confidence-weighted gap filling
- [ ] **One Euro Filter** smoothing (chosen over moving-average for low interactive lag)
- [ ] 12 fps resampling; T=64 windowing; 256-frame cap for the translation path
- [ ] Augmentations: frame-skip, speed warp, random crop, landmark jitter, Gaussian noise,
      horizontal mirror **with correct left/right hand swap**
- [ ] Dataset statistics report: length histograms, missing-landmark rate per split, class balance

**Tests:** property-based tests — normalization is translation/scale invariant; mirroring swaps
hands correctly; no NaN survives the pipeline; augmented sequences remain in-distribution
(distribution-shift assertion).
**DoD:** stats report committed; all property tests green.
**Demo:** side-by-side animation of raw vs processed sequences + the statistics report.

**PHASE 1 GATE (Fri Aug 21):** ≥20 FPS perception on the 3050, readiness table green or blockers
escalated with owners, all access requests submitted and tracked.

---

## PHASE 2 — RECOGNITION (Weeks 3–4)

### T5 — Isolated recognition baselines: a ladder of three
**Owner:** R2 · **Depends on:** T4 · **Est:** 6 days

- [ ] Manual-channel feature builder: 85-keypoint subset → 255-dim/frame (mirrors compact-SLT spec
      so our numbers are comparable to published ones)
- [ ] **Model A:** GRU baseline
- [ ] **Model B:** ST-GCN (decoupled spatial-temporal), **≤2M params**
- [ ] **Model C:** SPOTER-style pose transformer
- [ ] Train on WLASL-100 first (fast iteration loop), then scale to ASL Citizen
- [ ] Signer-independent splits using ASL Citizen signer metadata
- [ ] W&B sweeps on the 4060; record best configs in `configs/`

**Tests:** overfit-single-batch test per model; deterministic-seed reproducibility test;
**parameter-count assertion (<2M)**; signer-overlap assertion on splits.
**DoD:** leaderboard table (Top-1 / Top-5 / params / latency) for all three models on both datasets.
**Demo:** `seam recognize --video X` prints top-5 glosses with confidences.
**KPI checkpoint:** ASL Citizen Top-1 within 3 pts of published pose baselines; WLASL-100 Top-1 ≥80%.

---

### T6 — Gloss → text translation
**Owner:** R3 · **Depends on:** T5, T2 (ASLLRP access) · **Est:** 4 days

- [ ] Build gloss→English pairs from ASLLRP (+ How2Sign-derived pairs)
- [ ] Fine-tune `google/t5-v1_1-small`; AdaFactor, lr 1e-3 constant, 10-epoch linear warmup,
      grad-clip 1.0, label smoothing 0.1, dropout 0.1, effective batch 128, beam 5, 128-token cap
      (hyperparameters taken from the validated compact-SLT recipe — start here, then tune)
- [ ] Metrics: SacreBLEU BLEU-1..4 (default tokenization), ROUGE-L, METEOR, BERTScore

**Tests:** tokenizer round-trip; metric implementations validated against known reference values;
10-step smoke-training run in CI.
**DoD:** BLEU/ROUGE/METEOR/BERTScore table logged with a run ID.
**Demo:** **first full vertical slice** — `video → landmarks → glosses → fluent English` for a
handful of clips.

**PHASE 2 GATE (Fri Sep 4):** the vertical slice runs end to end. If it does not, stop and fix
before touching affect.

---

## PHASE 3 — AFFECT (Weeks 5–6)

### T7 — Affect feature engineering and the prosody channel
**Owner:** R3 · **Depends on:** T4 · **Est:** 3 days

- [ ] Non-manual features: 52 blendshapes, AU proxies derived from them, brow/mouth groupings,
      head tilt/rotation/nod frequency, gaze shifts, ~30-point semantic face-landmark subset
- [ ] **Prosody features from the manual channel:** signing speed, movement amplitude (signing-space
      volume), repetition count (autocorrelation peaks), pause duration, jerk, sign-duration stats
      — this is exactly the cue set EmoSign's Deaf annotators named, and eJSL showed hand motion
      improves signer-emotion recognition, so prosody belongs to affect, not grammar
- [ ] Feature-importance and correlation report against EmoSign sentiment labels

**Tests:** synthetic-signal tests (a scripted fast/large motion must score higher speed/amplitude
than a slow/small one); feature stability across input resolutions and fps.
**DoD:** correlation report shows the features carry signal before any model is trained.
**Demo:** feature-importance plot + correlation table vs EmoSign labels.

---

### T8 — Affect baseline on EmoSign under LOSO
**Owner:** R3 · **Depends on:** T7, T2 (ASLLRP video access) · **Est:** 4 days

- [ ] Single-branch classifier (no disentanglement yet) on the affect features
- [ ] **4-fold leave-one-signer-out CV** (200 clips, 4 signers)
- [ ] Three task heads: 7-point sentiment, 3-class sentiment, 10-emotion + neutral
- [ ] Class-imbalance handling (weighted loss / resampling), documented
- [ ] Metrics: weighted accuracy, weighted F1, per-class accuracy, confusion matrices, mean ± std
- [ ] Head-to-head table against the published EmoSign baselines (no re-running needed — the
      numbers are in the paper: GPT-4o video-only emotion wF1 20.76, sentiment-7 wAF 26.35;
      AffectGPT 11.03; Qwen2.5-VL 18.53; MiniGPT4 22.02; hearing non-signer sentiment-7 wAF 21.39)

**Tests:** split-integrity test asserting **zero signer overlap** across folds; imbalance-handling
test; a guard test that fails if any single-split (non-LOSO) number is written to the results dir.
**DoD:** LOSO results table with mean ± std and CIs, logged with run IDs.
**Demo:** the first head-to-head comparison table vs frontier MLLMs — the moment the core claim
becomes measurable.
**KPI checkpoint:** emotion wF1 ≥ 35, sentiment-7 wAF ≥ 35, both video-only.

---

### T9 — Dynamic-FER pretraining and transfer
**Owner:** R3 + R1 · **Depends on:** T8, T2 (DFEW/MAFW licenses) · **Est:** 4 days

- [ ] Pretrain the non-manual encoder on DFEW (~16k clips) + MAFW (10,045 clips)
- [ ] **Fallback if licenses stall:** RAF-DB / AffectNet static pretraining, or off-the-shelf AU
      extraction (OpenFace 2.0 / py-feat) as fixed features — record which path was taken
- [ ] Fine-tune on EmoSign under the same LOSO protocol
- [ ] Ablation: from-scratch vs pretrained vs frozen linear-probe

**Tests:** checkpoint-load and shape-compatibility tests; frozen-encoder linear-probe test proving
transfer actually happened (probe accuracy > from-scratch).
**DoD:** transfer ablation table logged.
**Demo:** ablation table showing the pretraining gain.

**PHASE 3 GATE (Fri Sep 18):** the affect model beats GPT-4o video-only under LOSO. This is the
paper's minimum viable result — if it fails, escalate immediately and consider re-scoping to a
sentiment-only claim rather than pushing on to T10.

---

## PHASE 4 — CONTRIBUTION (Weeks 7–8)

### T10 — The factorized non-manual encoder ★ CORE CONTRIBUTION
**Owner:** R3 · **Depends on:** T9 · **Est:** 7 days

- [ ] Extract `L` labels (linguistic non-manual markers: yes/no question brow-raise, wh-furrow,
      negation head-shake, topic marking, mouth morphemes) from **ASLLRP non-manual annotations**
- [ ] **Documented fallback if ASLLRP is unavailable:** rule-based blendshape heuristics
      (brow-raise magnitude, brow-lowerer, head-shake frequency, mouth-shape clusters) +
      syntactic analysis of the English caption (interrogative / negation / topicalization).
      Mark clearly as pseudo-labels in the paper.
- [ ] Two-branch encoder: `z_L = Enc_L(NM)`, `z_A = Enc_A(NM, P)`, each ≤1M params
- [ ] Losses: `CE(head_L)`, `CE(head_A)`, `MSE(head_VA)`, **GRL** both directions,
      **feature orthogonality** `||z_L^T z_A||_F^2`, **vCLUB** MI upper-bound minimization
- [ ] Frozen cross-probes: predict emotion from `z_L`, markers from `z_A` → cross-prediction AUC
- [ ] Per-loss-term ablation harness (toggle each λ independently)
- [ ] **Qualitative set:** curate wh-question and negation clips with neutral affect; show the
      entangled baseline calls them anger/frustration (the documented hearing-non-signer error)
      and the factorized model does not

**Tests:** disentanglement metric test — cross-prediction AUC must move toward 0.5 vs the
entangled baseline; ablation harness test; parameter-budget assertion.
**DoD:** headline table (entangled vs factorized × cross-pred AUC × affect wF1) + qualitative figure.
**Demo:** the paper's key figure and the misclassification-case walkthrough.
**KPI checkpoint:** cross-prediction AUC ≤ 0.60 **with no drop** in affect wF1 (ideally a gain).

---

### T11 — Emotion-conditioned generation
**Owner:** R3 + R4 · **Depends on:** T10, T6 · **Est:** 4 days

- [ ] Build the emotion-styled paraphrase corpus with an LLM from neutral English references;
      filter for semantic equivalence; **disclose the procedure in the paper**
- [ ] Inject `A` into T5-small as control tokens / prefix embedding
- [ ] Evaluate on three independent axes (BLEU alone is meaningless for style transfer):
      style accuracy (held-out classifier), semantic preservation (BERTScore-F1 ≥ 0.90),
      human preference (pre-registered rubric, blinded pairwise vs neutral)
- [ ] Degeneracy guard (repetition, truncation, empty output)

**Tests:** controllability test — flipping the emotion token must change the style classifier's
output while BERTScore stays ≥0.90; degeneracy guard test.
**DoD:** conditioned-generation results table + example set.
**Demo:** one utterance rendered neutral / happy / sad / angry side by side with metrics.
**KPI checkpoint:** style accuracy ≥ 80%, BERTScore-F1 ≥ 0.90.

---

### T12 — Phase B: gloss-free continuous translation *(cut-line 2)*
**Owner:** R2 · **Depends on:** T4, T6 · **Est:** 5 days

- [ ] Linear projection: 255-dim pose → 512-dim T5-small embedding space (single FC layer)
- [ ] Train on How2Sign keypoints; 256-frame input cap, 128-token output cap, beam 5
- [ ] Run at **both 24 fps and 12 fps** to replicate the frame-rate trade-off on our own stack
      (published: BLEU-4 10.06 @24fps vs 9.53 @12fps, with ~75% less encoder attention compute)
- [ ] Report params vs BLEU vs FLOPs against compact-SLT (77M) and T5-base (248M, BLEU-4 11.89)

**Tests:** sequence-length/truncation tests; BLEU regression test against a stored reference run.
**DoD:** fps/BLEU/FLOPs trade-off table.
**Demo:** continuous ASL clip → sentence, with the trade-off table.
**KPI checkpoint:** BLEU-4 ≥ 8.0 at ≤80M params.

**PHASE 4 GATE (Fri Oct 2):** disentanglement result in hand, conditioned generation working,
paper Method section written and its claims entered in `CLAIMS_LEDGER.md`.

---

## PHASE 5 — SHIP (Weeks 9–10)

### T13 — Export, quantize, benchmark on the 3050 ★ HEADLINE EFFICIENCY RESULT
**Owner:** R2 · **Depends on:** T5, T10, T11 · **Est:** 4 days

- [ ] ONNX export for every model (recognition, affect, translation)
- [ ] INT8 dynamic quantization for T5; ONNX Runtime CUDA EP; TensorRT optional
- [ ] VRAM ceiling enforcer in code (fail loudly, do not degrade silently)
- [ ] Benchmark harness on the **actual RTX 3050**: FPS, p50/p95/p99 end-to-end latency,
      peak VRAM, params, FLOPs — across configs (fps, model, precision)
- [ ] Produce the accuracy × latency × VRAM Pareto table

**Tests:** numerical-parity test (FP32 vs INT8 within tolerance); **hard CI test that fails if
peak VRAM > 2500 MB or p95 latency > 400 ms**.
**DoD:** Pareto table generated from real measurements on the 3050.
**Demo:** the efficiency table — a headline result, not an appendix.
**KPI checkpoint:** p95 < 400 ms, peak VRAM < 2500 MB, ≥20 FPS sustained.

---

### T14 — Serving layer
**Owner:** R4 · **Depends on:** T13 · **Est:** 3 days

- [ ] FastAPI + WebSocket: frames in → glosses / text / emotion / avatar-control stream out
- [ ] Bounded queues, backpressure, graceful degradation under GPU saturation
- [ ] **Localhost-only bind, no auth, by default** — assert it in a test; document that exposing
      it beyond loopback requires a token/API key + TLS because it streams webcam biometrics
- [ ] Latency HUD in the response stream

**Tests:** WebSocket integration tests; 30 fps load test with latency-under-load assertions;
**a security test asserting the default bind is loopback and that non-loopback binding without
auth configured raises**.
**DoD:** server runs stably for 10 minutes at 30 fps input without leak or drift.
**Demo:** live webcam → browser text + emotion readout with a latency HUD.

---

### T15 — Expressive avatar
**Owner:** R4 · **Depends on:** T14 · **Est:** 5 days

- [ ] React Three Fiber + three-vrm viewer
- [ ] Pose landmarks → VRM humanoid bone rotations, with joint-limit clamping and quaternion
      continuity (community reports document finger distortion and rotation instability from naive
      MediaPipe retargeting — this is a known trap, test for it)
- [ ] 52 MediaPipe blendshapes → VRM/ARKit expression targets (near-1:1, no training needed)
- [ ] **Emotion modulation from `A`:** blendshape gain, motion amplitude, playback timing —
      angry = faster/sharper/shorter path, sad = slower/smaller, matching documented ASL
      affective prosody
- [ ] GLB / VRMA clip export *(cut-line 3)*

**Tests:** retargeting unit tests (known pose → expected bone quaternions within tolerance);
no-NaN / no-gimbal-flip guard; browser smoke test.
**DoD:** avatar reproduces hands, posture, face and emotion from live input.
**Demo:** the full pipeline — sign to the webcam, watch the avatar mirror it, read the
emotion-aware sentence underneath.
**KPI checkpoint:** human preference for emotion-modulated vs neutral ≥ 60%.

---

### T16 — Cross-lingual ablation, reproducibility, paper assembly
**Owner:** all · **Depends on:** T13–T15 · **Est:** 5 days

- [ ] *(cut-line 1)* INCLUDE (ISL) cross-lingual generalization: zero-shot then few-shot with the
      ASL-trained recognition and affect encoders. Re-fetch the 12 truncated archives first, or
      restrict to intact categories **and say which in the paper**
- [ ] `scripts/repro_all.sh` — one command regenerates every table and figure
- [ ] Model cards for every released checkpoint (training data, signer demographics, failure modes,
      α-bounded reliability of low-agreement emotion classes)
- [ ] **Deaf-community involvement and limitations statement** (mandatory for this domain)
- [ ] Fill `paper/CLAIMS_LEDGER.md` — every claim → experiment → run ID → verdict
- [ ] Assemble `paper/main.tex`; verify every number traces to a logged run
- [ ] 3-minute demo video
- [ ] arXiv submission + workshop submission

**Tests:** CI runs `repro_all.sh` on a small subset; a test that fails if any results table
contains a number absent from `EXPERIMENT_LOG.md`.
**DoD:** paper PDF with all tables/figures populated from real runs; submission confirmed.
**Demo:** compiled paper + demo video.

**PHASE 5 GATE (Fri Oct 16):** live demo inside 2500 MB VRAM, paper submitted, repro script green.

---

## 2. KPI Dashboard

Update `Current` after every relevant run and cite the run ID. `—` means not yet measured.
Never fill a cell from an estimate.

| # | KPI | Baseline / reference | Target | Stretch | Current | Run ID | Task |
|---|---|---|---|---|---|---|---|
| K1 | ASL Citizen Top-1 (2,731-class) | ASL Citizen paper pose baselines | within 3 pts | beat | — | — | T5 |
| K2 | ASL Citizen Top-5 | — | ≥85% | ≥90% | — | — | T5 |
| K3 | WLASL-100 Top-1 | SPOTER-class | ≥80% | ≥85% | — | — | T5 |
| K4 | Recognition params | — | <2M | <1M | — | — | T5 |
| K5 | Gloss→text BLEU-4 | — | ≥15 | ≥20 | — | — | T6 |
| K6 | **EmoSign emotion wF1 (video-only, LOSO)** | **GPT-4o 20.76** | **≥35** | ≥45 | — | — | T8 |
| K7 | **EmoSign sentiment-7 wAF (video-only)** | **GPT-4o 26.35** | **≥35** | ≥45 | — | — | T8 |
| K8 | EmoSign sentiment-3 wF1 | GPT-4o 76.72 | ≥60 | ≥77 | — | — | T8 |
| K9 | FER pretraining gain (Δ wF1) | 0 | ≥+5 | ≥+10 | — | — | T9 |
| K10 | **Disentanglement cross-pred AUC** | 0.5 = perfect | **≤0.60** | ≤0.55 | — | — | T10 |
| K11 | Affect wF1 change from factorization | 0 | ≥0 (no loss) | ≥+3 (gain) | — | — | T10 |
| K12 | Conditioned-gen style accuracy | — | ≥80% | ≥90% | — | — | T11 |
| K13 | Semantic preservation BERTScore-F1 | — | ≥0.90 | ≥0.93 | — | — | T11 |
| K14 | How2Sign BLEU-4 @ ≤80M params | 10.06 published | ≥8.0 | ≥10 | — | — | T12 |
| K15 | **End-to-end p95 latency (RTX 3050)** | — | **<400 ms** | <250 ms | — | — | T13 |
| K16 | **Peak inference VRAM** | 4096 MB hard limit | **<2500 MB** | <1800 MB | — | — | T13 |
| K17 | Sustained capture FPS | — | ≥20 | ≥30 | — | — | T3/T13 |
| K18 | Human preference (emotion vs neutral avatar) | 50% = tie | ≥60% | ≥70% | — | — | T15 |
| K19 | INCLUDE cross-lingual Top-1 (zero-shot) | chance | report honestly | — | — | — | T16 |
| K20 | Reproducibility | — | 1 command, all tables | + released weights | — | — | T16 |

**Bold KPIs are the paper's headline claims.** If K6, K10, K15 or K16 fail, the paper's story
changes and the team must be told immediately — do not paper over it.

---

## 3. Weekly Rhythm

- **Monday, 30 min** — plan against this file; move checkboxes; confirm the week's demo target.
- **Wednesday, async** — written blocker sweep; anything red gets an owner and a date.
- **Friday, 45 min** — *demo or it didn't happen*. Update the KPI dashboard. Append to
  `paper/EXPERIMENT_LOG.md`. Write one paragraph of the paper.
- **Daily** — written async standup: done / next / blocked.

**Paper is written continuously, not at the end.** Each phase gate requires its section drafted:
P1 → Datasets; P2 → Recognition; P3 → Affect experiments; P4 → Method + Contribution;
P5 → Efficiency, Demo, Ethics, Conclusion.

---

## 4. Critical Path

```
T2 ASLLRP access request  ─────────────────────────────► T6, T8, T10
      (async, submit DAY 1 — everything downstream waits on it)

T1 ─► T3 ─► T4 ─► T5 ─► T6 ──────────────► T11 ─► T13 ─► T14 ─► T15 ─► T16
              └─► T7 ─► T8 ─► T9 ─► T10 ─────┘        │
              └────────────► T12 ────────────────────┘
```

The single highest-leverage action in the entire project is submitting the ASLLRP access request
on day 1. It gates the `L` labels, which gate T10, which is the contribution.

---

## 5. Definition of Done (project level)

- [ ] All 16 tasks complete with tests green
- [ ] Every bold KPI met or its miss explained in the paper's limitations
- [ ] Live demo runs on the RTX 3050 under 2500 MB VRAM
- [ ] `scripts/repro_all.sh` regenerates every table and figure from scratch
- [ ] Every paper number traces to a W&B run ID in `EXPERIMENT_LOG.md`
- [ ] `CLAIMS_LEDGER.md` has no unverified claims
- [ ] Ethics, limitations, and Deaf-community statements written
- [ ] No dataset video redistributed; model cards published
- [ ] arXiv preprint + workshop submission filed
