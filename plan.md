# SEAM — Execution Plan v2

**Audience: the AI agent implementing this project.** Read `project_breakdown.md` for *why*; this
file for *what to do next*. Milestone-driven: **no dates.** Milestones are ordered by
dependency × evidence-value ÷ cost, and each one ends in a runnable artifact and a logged
number. A milestone is done when its **evidence gate** closes — not when its code compiles.

`implimentation.md` is the execution expansion of this file: the same plan plus the per-task
breakdown, file map, and commands. **`plan.md` is canonical for gates, KPIs and scope.**

- **Executor:** 1 implementer (AI) + 1 reviewer (human). `team.md`'s 4-person RACI is superseded.
- **Hardware:** RTX 3050 4 GB. **Train and measure on the same GPU.** ~3 GB usable RAM, swap
  saturated. Disk: `/mnt/DevProd` (110 GB) for bulk, repo on `/mnt/Volume2` (36 GB).
- **Environment:** conda env `slr` (Python 3.12.13, torch 2.13.0+cu130, mediapipe 0.10.14).
  Reused rather than duplicated — a second torch install costs ~3 GB on a disk- and RAM-starved
  machine for no gain.
- **Paper target:** arXiv preprint + workshop (CVPR/ICCV MSLR or LREC SignLang).
- **Supersedes:** plan.md v1 (10-week, 16-task, RTX 4060, dated Mon 2026-08-10 → Sun 2026-10-18).
  v1's assumptions were invalidated — see §1. Do not execute v1.

---

## §0 Operating rules

Inherited from v1: test-first · every task ends in a demo · no number without a run ID · no
overclaiming · pin dependency versions exactly · localhost-only serving with no auth by default ·
never redistribute dataset video · measure latency and VRAM on the actual 3050.

Added or replaced in v2:

11. **Train and measure on the 3050.** Any result needing >4 GB VRAM is a stretch result and is
    labeled as such.
12. **Confound-audit rule.** An affect number computed without linguistic-marker control is
    reported as *entangled*, never as a result.
13. **Positive-control rule.** The disentanglement metric must be shown to *detect* entanglement
    when it exists (M4.4). A metric reading 0.5 on everything is not evidence.
14. **Stream, never materialize.** `num_workers: 0`, lazy `.npz` reads, bulk data on DevProd.
    This machine has ~3 GB free RAM.
15. **When blocked, escalate in writing.** Fallbacks already documented here may be taken freely
    but must be recorded as taken, with the reason, in `EXPERIMENT_LOG.md`.
16. **One front end, asserted by test.** The fit and inference preprocessing paths must be the
    same function, and a test must assert it. Two independent instances of the same defect in
    this project - a `mask=None` normalization call in the prior ISLR system, worth 19 accuracy
    points, and the FER front-end skew in M1, which invalidated a whole audit - make this a rule
    rather than an anecdote.
17. **Every null carries its power.** No result is reported as "no effect" without the minimum
    detectable effect beside it, in the same units. A null without its MDE is an absence of
    measurement, and the two are indistinguishable to a reader who is not told which it is.

---

## §1 Ground truth — measured 2026-09-26

v1 assumed several resources that are not what it said. This table is the ground truth; every
milestone plans against it, not against v1.

| Resource | Status | Size | Unlocks |
|---|---|---|---|
| EmoSign labels `catfang/emosign` | **ungated**, fetched | 43 KB, 200 rows | sentiment-7, 10 emotions, 600 free-text Deaf-annotator cue strings |
| EmoSign video via `FangSen9000/ASLLRP_utterances_results` | **200/200 IDs join** to `crop_original_video.mp4` | ~113 MB | **the affect benchmark — v1's critical path is resolved** |
| ASLLRP gloss tokens `asllrp_sentence_signs_2025_06_28.csv` | ungated | 17,522 tokens, frame-aligned, signer-tagged | continuous recognition + gloss sequences |
| ASLLRP SignStream non-manual XML | **not in mirror** | — | `L` labels → heuristics + BU request (async) |
| ASLLRP English sentences | **not in mirror** | — | gloss→English → ladder in M5 |
| How2Sign `Kavitha/how2sign_user3_mediapipe_pose` | ungated, published keypoints | — | gloss-free SLT + published BLEU-4 10.06 to hit |
| WLASL local `/home/bhuwan/Videos/wlasl` | 3,863 mp4 / 668 glosses | 7 GB | recognition backbone + label-free confound audit |
| ASL Citizen `SorensenAI/asl-citizen-poses` | ungated MediaPipe `.pose`, **no blendshapes** | 81 GB, 1000/batch | K1/K2 without the 403 |
| `Pelmeshek/raf-db-7emotions-mediapipe-768` | ungated | 2.7 GB | non-signer FER baseline to audit |
| NSL / INCLUDE local | present | 8.6 GB | cross-lingual ablation |
| `face_landmarker.task` | **already cached locally** | 3.7 MB | blendshape extraction, zero download |

**How the EmoSign join works.** EmoSign's `video_name` is not a filename — its trailing numeric
token *is* the ASLLRP utterance ID. `Jonathan_2012-11-27_sc93_5572615` → `5572615` →
`crop_original_video.mp4`. Measured: 200/200 match, all four signers, no exceptions. EmoSign's
signers are **Cory 87, Jonathan 54, Rachel 52, Ben 7**; Rachel spans `Rachel_2011` + `Rachel_2012`
and must be **one** LOSO fold. Ben's 7-clip fold is reported separately, flagged low-power, never
pooled away.

**EmoSign subset caveat:** the "Single Expression Set of 140 clips" cited by prior work is **not
reconstructible from the released CSV** (5 subset rules tested; best match 106). We define, name,
disclose and sensitivity-test our own rule, and report the full 200 alongside.

**Stale v1 findings to discard:** v1's `/home/bhuwan/Videos/data` (INCLUDE, 12 truncated `.part`)
no longer exists — re-audit the NSL data before M8. v1 also names a MediaPipe **Tasks
`HolisticLandmarker`**, which does not exist; the Tasks API has separate Face/Hand/Pose
landmarkers and they must be composed.

---

## §2 Corrections to the research position

Two prior papers must enter Related Work; v1 cites neither.

- **Funakoshi & Zhu, "Emotion Recognition in Signers"** (arXiv 2512.15376v2, Jul 2026) published
  the adjacent result: hand motion helps, segment selection helps, and it beats GPT-4o on EmoSign
  (macro F1 **21.09** vs GPT-4o **11.15** on the 140-clip set). Its conclusion states *"EAN and
  EANwH do not understand signed linguistic content in utterances… integration with sign language
  understanding also must be explored."* **That sentence is our contribution, with a named
  competitor whose stated future work we occupy.** K4's bar moves from 20.76 to **21.09**.
- **Silva et al. 2020** (BSL, AU-annotated grammatical vs affective facial expressions) attacks the
  same disentanglement problem. Cite it and state plainly how our factorization differs.

Positioning line: **all published sign-emotion models are hearing non-signers.** M1 measures
exactly what that costs, M4 fixes it, M2/M7 ship it on 4 GB. The efficiency claim is unshared —
every published baseline ran on an 80 GB A100 or a 300M-parameter model.

---

## §3 Milestone map

### M0 — Data spine & engineering foundation
Repo `src/seam/` (`data/ perception/ preprocess/ features/ models/ affect/ translate/ avatar/
serve/ export/ eval/`); fetcher with SHA256 manifests, **hard-linking local data rather than
re-downloading**; download the 200 EmoSign clips; `pyproject.toml` exact pins; `Makefile:
setup lint test train bench repro paper`; pytest + ruff + mypy + pre-commit green; W&B project.

**Face-visibility gate:** the mirror's `crop_*` videos may be signer-centred or face-occluded.
Validate blendshape extractability on ≥20 sampled clips *before* committing the pipeline. If the
face is not reliably visible, M1/M4 pivot to clips that pass and the loss is reported.

**Rebuild-from-scratch discipline.** We are rebuilding, but not re-learning. Carry these five
*measured* findings from the prior ISLR system forward as design constraints, each with a test:
(a) one shared normalization module + a train/serve parity test — a `mask=None` call bug there
cost 19 accuracy points; (b) extract with the **Tasks** API everywhere and *measure* extractor
mismatch rather than assuming parity (~7 points there); (c) assert zero signer overlap in every
split test (WLASL's official split leaks signers in 99% of test clips); (d) make no architecture
claims on small data (ST-GCN was data-bound at 745 clips, not broken); (e) calibrate confidence
from measurement (median top-1 confidence was 0.29), never from intuition.

**Gate:** readiness table published · 200/200 clips local · blendshape extraction validated ·
`make test` green.
**Kill switch:** none.
**Survives:** everything.

### M1 — The confound audit ★ **GATE CLOSED 2026-09-27 — C1 REFUTED on WLASL**

Extract the non-manual channel (52 ARKit blendshapes + head pose + gaze) from WLASL. Run ≥3
non-signer FER models over it. Detect grammatical markers by rule on the *same* blendshapes:
brow-raise (yes/no Q), brow-furrow (wh-Q), head-shake (negation), mouth-morpheme clusters. Test
whether FER output shifts toward negative affect on marker-bearing vs matched marker-free
segments, within clip.

**What happened.** 2,565 clips, 4,168 scorable windows, three non-signer FER models
(RAF-DB, 61.7 / 63.7 / 66.3% on their own test split). **No marker-induced negative bias is
detectable.** The two effects that reach significance run the *wrong way* and do not replicate
across models. MDE is 0.0030–0.0083, i.e. 1.0–2.8% of the 0.297 baseline, over 202–329 clips —
a powered null, not an absence of measurement.

**Gate:** met. Measured, 3 models, 2,565 clips (≥500), CIs + effect sizes + MDE, marker→emotion
table, `mouth_positive` control null in all three, uniform-random null model null in all six rows.
**Kill switch, taken:** the null is the result. It is published as a localisation, not a failure.
**Survives:** the feature stack for every later milestone, plus a real negative result.

**The finding, precisely.** The confound is a **discourse** phenomenon — a signer raises their
brows *because the utterance is a question*. WLASL is isolated dictionary signing, one gloss per
clip, neutral and posed, with no discourse context for a marker to be syntactically determined by.
So this audit bounds the effect on *isolated* signing and says nothing about continuous signing.

**Consequence for M3/M4, which is the point.** The testable version of C1 needs continuous
signing where markers are determined by syntax: **ASLLRP utterances (we have 200) or How2Sign
(media-pipe keypoints published, plus official English).** Point M3's marker labeller and M4's
LOSO evaluation there rather than at isolated signs.

**An instrument bug worth keeping in the record.** The first run of this audit was **invalid**:
all three FER models predicted one class for all 200 EmoSign clips (sadness 200/200, fear
200/200, anger 200/200), with per-frame spread *within* a clip of 0.0002 against 0.283 *across*
clips. They were reading lighting, background and resolution, not faces. The cause was the
absence of a FER front end - and of a shared one: the fitting path took tight uint8 boxes from
PIL while the inference path took [0,1] float crops at a 0.35 margin. Fixed with grayscale +
histogram equalisation + per-image z-score in **one function on both paths**; the negative-mass
baseline fell from 0.78 to 0.297, which is where most of the apparent bias in the broken run
came from. The broken models are kept at `artifacts/fer_v1_broken_preproc/`. This is the second
occurrence in this project of the same failure family, so `plan.md` §0 rule 16 now requires the
fit and inference front ends to be asserted equal by test.

### M2 — Efficiency harness & the 4 GB budget, proven early

**Status: GATE MET.** Every element of the gate is now measured on the 3050 with an
instrument that records the conditions it was measured under.

**Gate:** measured p50/p95/p99 + peak VRAM on the 3050 for the perception stage; CI test that
fails above the ceiling.

| element | result |
|---|---|
| p50 / p95 / p99, perception, on the 3050 | **41.8 / 73.2 / 78.1 ms** over 900 individually-timed calls |
| sustained rate | **20.4 FPS** (repeats: 19.7, 20.0, 20.3) |
| sequential baseline, same instrument | 73.0 ms p50, 12.8 FPS — **concurrency worth 1.75x** |
| peak VRAM, perception | 90 MB |
| peak VRAM, **6 models live at once** | **186 MB against the 2500 MB ceiling (7%)**, 20.3 FPS |
| CI gate that fails above the ceiling | `make bench` exits non-zero; 185 tests, 24 on the guards |
| FP32↔INT8 parity harness | met, and it rejected `fer_cnn_a` INT8 |
| `onnxruntime-gpu` CUDA EP | met, after preloading the bundled CUDA 13 libraries |

- **K6 is met marginally, and is reported that way.** Four reportable runs give
  19.7–20.4 FPS against a ≥20 target: it meets the target at the median and does not
  clear it at the low end. A single 20.4 FPS run would have been the dishonest way to
  report it.
- **K5 is comfortably met**: 186 MB for the whole live stack (3 MediaPipe graphs + 3
  ONNX FER graphs resident simultaneously) against a 2500 MB ceiling.
- **VRAM is not the binding constraint; the CPU is.** MediaPipe runs on the CPU
  delegate, so perception is bound by the core clock. This was not visible until
  recently: the host came up in `powersave` at 1.14 GHz of 4.5 GHz and the same
  benchmark read 118 ms against 41.8 ms. Two frame-mix and hand-detection
  explanations were measured and rejected first, so the attribution is not a guess.
- **The instruments had to be fixed before the numbers could mean anything.** Each of
  these produced a plausible, wrong reading: quantiles taken over 5 run-means rather
  than over frames; `noise_ratio`, which compares runs within one session and so is
  blind to a uniformly busy machine; a post-run clock sample that reports 16% of peak
  for a run at 87% and rejected a *passing* measurement; an 80%-of-single-core-turbo
  gate that flagged a healthy full-turbo run; and a governor check that only ran in
  one branch, so a `powersave` run that happened to catch a full clock was accepted.
  The benchmark now records governor, during-run clock, load and memory, and refuses
  to call a run reportable without them.
- **Reports are non-destructive and run-stamped.** `bench.save`/`save_report` write a
  run-scoped file plus a `latest.json` and never overwrite — an earlier in-place save
  let a CPU run destroy the only stored GPU perception measurement, while the
  experiment log cited figures with no artefact left to check them against. Run IDs
  hash the measured numbers, so a cited ID cannot drift from its artefact.
- **K6's honest caveat:** the FER graphs in the live-stack figure are fed zero tensors
  of the correct shape, not perception output, because the crop stage that would feed
  them does not exist yet. That measurement is about residency and invocation, not
  end-to-end inference.

### M3 — Linguistic-marker supervision (`L` labels)

**Status: gate met on labels, and the visual instrument ARRIVED 2026-10-01 — this line is
stale and superseded by the entry below.**

### M3 instrument status — resolved 2026-10-01

The SignStream non-manual XML was downloaded (51 collections) and every one of the 200
EmoSign utterances joins it directly by ID, carrying 4,443 human non-manual annotations.
The pseudo-label limitation is lifted for these clips.

**But measuring the old pseudo-labels against the human ones found most of them
unusable** (`artifacts/m3/label_agreement.json`, Cohen's kappa):

| category | kappa | verdict |
|---|---|---|
| interrogative ~ rhetorical question | +0.734 | usable |
| negation | +0.639 | usable |
| topicalization ~ topic/focus | +0.141 | weak |
| interrogative (wh + yes/no) | +0.028 | **chance** |
| reference-establishment ~ conditional/when | +0.038 | **chance** (`reference_establishment` over-fires: 93 heuristic positives vs 37 human) |

So M3 is now on human labels rather than guesses, and separately has a published result
about how unreliable the pseudo-labels were. The earlier duration-controlled finding
(r = 0.554 for negation ↔ head shake) is consistent with this: negation was the one
heuristic label with genuine signal to detect.

- **The syntactic track now has a real input.** `asllrp_utterance_map` and
  `asllrp_gloss_tokens` were declared in `sources.py` but never fetched and marked
  blocking-M5; they are what M3 actually needs, so they were pulled (3.8 MB).
  **200/200 EmoSign utterances join to the human-authored ASLLRP gloss map**, so
  every clip has real linguistic context. `seam.features.syntactic` labels
  interrogative / negation / reference-establishment from that annotation by lexical
  rule, and topicalization as an explicitly weak inference (confidence capped at
  0.5, `pseudo`), because the gloss is a flat token sequence with no constituent
  structure. Every label carries a provenance string.
- **`fs-` is compound signs, not facial signal.** Checked rather than assumed; it
  would have added 98 token types to the non-manual vocabulary for no reason.
- **Blocking finding: 4 of 6 visual markers are degenerate at clip level.**
  `mouth_morpheme` 0.95, `brow_raise` 0.86, `mouth_positive` 0.91, `head_shake`
  0.79 of clips. `mouth_positive` is the *control*, and a control on 91% of clips
  is not a control. This is a candidate explanation for M1's null that has nothing
  to do with marker effects being absent, since M1's audit compared near-constant
  signals.
- **The threshold rule is not at fault.** On iid noise the `median + 1.5*MAD` rule
  fires on 0.2% of frames. The real signals are heavy-tailed; raising k to 4.5 does
  not fix prevalence, and tuning it until agreement looked good would be p-hacking.
- **Two real head-path bugs fixed:** `_oscillation` thresholded at `head_angle/2`
  while the field documents `head_angle` (10° effective vs 20° documented, inside
  the jitter of a 256×256 face), and `min_reversals` counted sign changes anywhere
  in a clip so one wobble licensed every noisy turning point. `head_shake` 0.98 →
  0.785, `head_nod` 0.65 → 0.170.
- **The harness now refuses to print a p-value for a degenerate marker.** A lift of
  ~1.0 is what a broken instrument and a real null both look like, so the two are not
  reported as the same thing. Of three expected pairings, two are not interpretable
  and one — interrogative ↔ brow_furrow, the marker inside the usable band — is a
  **resolvable null**: lift 1.00, p=0.96, from two independent sources.
- **Calibration resolved, and the first strong result was mostly clip length.** A
  pre-registered criterion (control ≤25% prevalence, all markers in [5%, 60%])
  could not be met by any of 64 settings, because `mouth_positive` is genuinely
  active in ~every clip and is not a control. Two scale bugs followed: MAD is the
  wrong scale for heavy-tailed blendshape signals (27:0 spread in peak evidence), and
  `clip_presence` compared the new clip-range evidence against the MAD-scale constant
  `k=1.5`, so all four blendshape markers silently read zero coverage.
- **The first association result was a duration artefact.** On the *integral* of the
  excursion, all three pairings looked strong and correctly directed (r up to +0.71).
  Every syntactic label tracks clip length (interrogative 5.77s vs 4.48s, r_pb=+0.68)
  because questions and negated statements are longer utterances, and an integral
  inherits that mechanically. On a per-frame mean, interrogative/brow_raise went
  **+0.414 → −0.045** — the whole association was clip length, and reporting it would
  have been a false positive.
- **Result (partial correlation on log-duration, permutation, Bonferroni ×3):**
  **negation ↔ head_shake r=0.554, p=0.011, p×3=0.033, MDE 0.411** — the canonical ASL
  negation marker, and the one effect that survives. Interrogative ↔ brow_raise
  −0.084 (p=0.62) and ↔ brow_furrow 0.199 (p=0.24) are **powered nulls** against an
  MDE of ~0.34, not "no effect".
- **Does not reopen M1**: different substrate (continuous utterances, marker–syntax
  co-occurrence, duration-controlled) versus M1 (isolated signs, marker effect on a FER
  prediction). The two are compatible.
- **Grounding move done, and it returns a negative result.** The feature set was tested
  against the 600 free-text Deaf-annotator cue strings. The annotators write two kinds
  of thing — motor cues ("bared teeth", "head shake") and affective *interpretations*
  ("conveys surprise", "signifies worry") — and only the first is evidence about a
  feature, since affect is FER's target. 482/600 strings carry a motor cue; 118 are
  affective-only and are excluded by construction rather than counted as misses.
  **None of 15 cue/feature tests survives Bonferroni** (duration-controlled,
  permutation, MDE reported). The assumed feature set is not validated, which is the
  point of running the check.
- **One channel is corroborated by two independent ground truths.** `head_shake` gives
  r=+0.275 (p=0.077) against annotator text and r=+0.554 (p=0.011) against lexical
  negation from the ASLLRP gloss. The brow and mouth channels return nulls *and* are
  the channels measured as firing on 79-95% of clips, so those nulls are statements
  about saturated instruments.
- **`head_nod` is reported BLIND, not null** — zero on 83% of clips, so it cannot
  discriminate. Reporting its non-separation as evidence would be indistinguishable from
  saying the annotators were wrong.
- **M4's feature set is now specified by the corpus rather than by taste**: six cue
  categories the annotators actually used have no feature behind them —
  `head_tilt`, `eye_widen`, `blink_close`, `gaze_shift`, `fingerspelling`,
  `body_posture`. The first two of the eye cues are recoverable from blendshape
  coefficients already being computed and simply not read.
- **Track B (BU access for real SignStream non-manual XML) remains outstanding** and is
  the only M3 item not closed; it is external.
- Artefact `artifacts/audit/marker_labels.json`; 177 tests pass.

### M4 — Factorized non-manual encoder + EmoSign LOSO ★ CORE CONTRIBUTION

**Status: GATE NOT MET, AND THE NEGATIVE RESULT IS CONFIRMED — decision taken 2026-10-01: report as a refuted hypothesis (option (a)), not a deferred gate.**

**Why this is now a finding rather than a failure.** Two things had to be excluded before
"the factorisation does not hold" was a claim anyone could act on, and both are now
excluded:

1. **The labels were not the cause.** Three of M4's four linguistic labels come from
   `seam.features.syntactic` heuristic pseudo-labels. Those were measured against the
   human ASLLRP annotations downloaded 2026-10-01, on the same 200 clips
   (`artifacts/m3/label_agreement.json`, Cohen's kappa):

   | category | human + | heuristic + | kappa | verdict |
   |---|---|---|---|---|
   | negation | 39 | 27 | **+0.639** | usable |
   | interrogative ~ rhetorical question | 40 | 46 | **+0.734** | usable |
   | topicalization ~ topic/focus | 86 | 84 | +0.141 | weak |
   | interrogative (wh + yes/no) | 31 | 46 | **+0.028** | chance |
   | reference-establishment ~ conditional/when | 37 | 93 | **+0.038** | chance |

   Three of the four are at or near chance, and `reference_establishment` over-fires on
   93 clips against 37 human positives. That is a plausible cause of leakage on its own:
   z_L fitted to noise, where the noise happens to correlate with a human affect rating of
   the same video.

   **It is not the cause.** Re-running the whole experiment with the *human* labels in the
   same four slots, same folds, same features (`artifacts/m4/factorizer_human_labels.json`):

   | | heuristic `y_L` | human `y_L` |
   |---|---|---|
   | worst-fold cross-AUC | 0.7276 | **0.7031** |
   | cross L→A | 0.6911 | 0.7004 |
   | cross A→L | 0.5048 | **0.6174** |
   | GRL head accuracy | 0.500 | 0.507 |
   | signer control | 0.9729 | 0.9731 |
   | gate | FAIL | FAIL |

   Marginal improvement in the worst fold, and the *other* direction got worse. The gate
   is not missed because the target was noise.

2. **The instrument works.** The signer control sits at 0.973 against a ≥0.80 floor in
   both arms, so the pipeline detects signer identity far better than chance. A negative
   result from an instrument that cannot see anything would be worthless.

**What is therefore claimed.** On EmoSign, with a validated instrument, linguistic and
affect information do not separate under an adversarially-separated representation: the
worst fold admits affect into the linguistic factor at AUC 0.70 against a ≤0.60 target,
with leakage asymmetric (L→A strong, A→L at or above chance). The confound that would
have made this uninterpretable — heuristic target labels — is measured and ruled out.

**Cost of the decision.** The core contribution as originally framed is withdrawn. What
replaces it: two refuted hypotheses with sound instruments (M1's C1, and this), a measured
result about pseudo-label reliability in sign-language affect work, and a reusable
adversarial/probe harness that passes its own positive control.

**Status detail follows.** All the instruments exist and are falsifiable; the model does
not separate the factors and does not learn either task.

**Gate:** cross-prediction AUC ≤ 0.60 **with no drop** in affect wF1, under LOSO, positive control passing.
**Kill switch:** if AUC improves but wF1 drops, rewrite the claim to "separation without loss" and publish the trade-off curve.

**Rerun with multi-label affect** (run `m4-factorizer-002`) — the framing was the bug:

| | single-expr | **multi-label** | target |
|---|---|---|---|
| trainable windows | 314 | **1,765** | — |
| cross A→L (linguistic from z_A) | 0.695 | **0.505** (chance) | — |
| cross L→A (affect from z_L), weighted | 0.879 | **0.691** | ≤ 0.60 |
| worst fold | — | **0.884** (Ben) | ≤ 0.60 |
| affect micro-F1 | n/a | **0.359** | no drop |
| balanced acc, linguistic | 0.547 | 0.529 | — |
| balanced acc, affect | 0.173 | 0.497 (ref 0.5) | — |
| signer positive control | 0.971 | **0.973** (passes) | ≥ 0.80 |

- **The gate still fails**, on the worst fold: 0.884 (Ben) against ≤ 0.60. The control
  passes, so the failure is the model's, not the instrument's.
- **What the fix bought.** EmoSign's affect annotation is multi-label by construction, so
  requiring one committed emotion per clip discarded 1,451 of 1,765 windows. Multi-hot
  targets with BCE over eight independent binaries keeps all of them, and **cross A→L
  reached chance (0.505)** — the affect factor no longer carries linguistic information,
  which is a real result and the affect-side GRL is demonstrably doing its job.
- **The full model is best on all three target metrics at once** — worst cross AUC *and*
  micro-F1 — so separation no longer costs task performance. That clears the plan's
  kill-switch condition: removing any separation term lowers micro-F1 by 0.024–0.060.
- **The remaining obstacles are not separation terms.** (a) The Ben fold has 7 clips and
  only 3 of 8 affect labels reach support: it needs an explicit "insufficient support"
  verdict instead of a number, or merged small-signer folds. (b) Residual affect in
  `z_L` at 0.691 may be genuine — M3 measured a real head-shake/negation association, and
  the brow and mouth channels remain saturated.

- **The control passes, so the failure is real and not a blind instrument.** That was the
  point of rule 13: a metric that cannot detect entanglement when it exists would pass
  a completely entangled model.
- **The separation terms work, monotonically, and are not enough.** Removing the
  orthogonality penalty moves cross L→A from 0.879 to 0.903; removing all separation
  pressure gives 0.898 with the adversarial head rising 0.143 → 0.162. Correctly signed,
  far short of the target.
- **The binding constraint is the data, not the architecture.** Requiring one committed
  emotion per clip leaves **314 trainable windows from 200 clips** — 1,451 of 1,765 are
  dropped because EmoSign's affect labels are multi-label by construction. M4's
  single-expression 8-class framing discards 82% of the windows and the remainder does
  not learn. **The next step is a multi-label affect objective over the discarded
  windows, not more separation terms.**
- **Plain accuracy would have hidden this.** The majority-class predictor scores 0.863
  (linguistic) and 0.334 (affect) and beats both the factorized model and the entangled
  baseline. A majority reference is now printed with every accuracy, and the gate uses
  weighted F1 as the plan specifies.
- **Five instruments were wrong before the model was, each caught by refusing to believe
  a number:** the plan's vCLUB returns an *inverted* ordering and was replaced by a
  JSD bound; a dependence detector scores **0.94 on training pairs and 0.4996 held out**,
  so all probes are cross-fitted; `auc()` silently returned **3.996** on 8-class labels
  and now raises; the probe's hand-rolled optimiser diverged; and **neither adversarial
  term was in the loss at all**, so the GRL heads were untrained — which dissolved an
  apparent "adversarial defeat" finding that was really an unfitted head. Two advertised
  ablation levers were config fields the loss never read, so one variant reproduced
  `full` byte for byte.
- **What survives this milestone:** the loss primitives (GRL both directions,
  scale-normalised orthogonality, a calibrated JSD MI bound), LOSO with the
  signer-disjointness assertion as a constructor invariant, and a cross-probe metric with
  a positive control that gates its own interpretation. That is the reusable part, and
  it is what the M4 rerun needs.

### M5 — Recognition + translation
**M5a continuous recognition:** train on the 17,522 frame-aligned ASLLRP gloss tokens with
signer-disjoint splits — real gloss supervision and real temporal alignment, no download risk.
**M5b gloss-free SLT:** 255-dim pose → single linear layer → T5-small on How2Sign published
keypoints, at **24 fps and 12 fps** to reproduce the frame-rate trade-off against the published
BLEU-4 10.06 / 9.53. M5b is the translation claim with a citable published reference; M5a feeds
the demo.
**Known gap:** ASLLRP English is not in the mirror. Ladder: BU English → How2Sign English
(gloss-free path) → LLM gloss back-translation, **disclosed as synthetic**.

**Gate:** BLEU-4 at ≤80M params on How2Sign, 3 seeds, signer-disjoint, with the 24-vs-12 fps FLOPs
table.
**Kill switch:** M5a alone still yields a working video→gloss→English demo.
**Survives:** the translation section and the demo backbone.

### M6 — Emotion-conditioned generation
LLM-built emotion-styled paraphrase corpus from neutral references, semantic-equivalence filtered,
**procedure disclosed**. Inject `A` into T5-small as control tokens / prefix. Three independent
axes: style accuracy (held-out classifier), semantic preservation (BERTScore-F1 ≥ 0.90), degeneracy
guard. **BLEU is reported as meaningless here and labeled so.**

**Gate:** style accuracy ≥ 80% at BERTScore-F1 ≥ 0.90; controllability test (flipping the token
must change style output while semantics hold).
**Kill switch:** demote to a controlled generation demo with no metric claim.
**Survives:** the demo.

### M7 — Expressive avatar + live demo
Pose → VRM bone rotations with joint-limit clamping and quaternion continuity — **gimbal flips
and finger distortion are a documented MediaPipe-retargeting trap; test explicitly.** 52
blendshapes → VRM/ARKit expression targets, near-1:1, no training. **Emotion modulation from `A`:**
blendshape gain, motion amplitude, playback timing (angry = faster/sharper/shorter path,
sad = slower/smaller, matching documented ASL affective prosody). **Live browser demo with
client-side MediaPipe Tasks in WASM**, so the server ships no video and the demo runs anywhere.
Blinded pairwise human preference, pre-registered rubric, ≥5 raters, inter-rater agreement
reported; **if no signer participates, that is stated in Limitations, not glossed over.**

**Status: browser demo built and gated; no human preference study yet.**

- `seam serve` / `make serve` runs it; `make serve-check` boots the server and asserts the
  HTTP contract, which is how a 44-byte 404 page that started "successfully" was caught.
- **Video never leaves the browser** — MediaPipe Tasks runs in WASM and the page posts
  only landmark numbers, so there is nothing to store or leak. The cost is stated in the
  response: the server cannot verify what the client measured, so every reply carries its
  input provenance.
- **It refuses to display what this build cannot support**, with the measured reason
  inline: affect is withheld because M4 measured 0.497 balanced accuracy against a 0.5
  reference, and gloss recognition is withheld because M5a's labels are misaligned.
- **A blind feature is rendered as blind** (zero fraction in the payload), so "no marker"
  and "this instrument cannot tell" stay distinct on screen.
- **Outstanding for the M7 gate:** the blinded pairwise human-preference study (≥5 raters,
  inter-rater agreement, and the explicit statement if no signer participates), and VRM/GLB
  avatar export with joint-limit and quaternion-continuity tests. 253 tests pass.

**Kill switch:** drop GLB/VRMA export first, keep the live viewer.
**Survives:** the demo video and system section.

### M8 — Cross-lingual ISL ablation
ASL-trained encoders → INCLUDE/NSL zero-shot then few-shot, using the 8.6 GB already local.
Re-audit integrity first (v1's `.part` finding is stale — that path no longer exists). If the
official split cannot be reproduced, restrict to intact categories **and say which in the paper**.

**Gate:** zero-shot Top-1 vs chance, reported honestly either way.
**Kill switch:** first thing cut; it is an ablation.
**Survives:** one ablation table.

### M9 — Paper, repro, release
`scripts/repro_all.sh` regenerates every table and figure; **CI test that fails if any results
table contains a number absent from `EXPERIMENT_LOG.md`.** Model cards for every checkpoint
(training data, signer demographics, failure modes, **α-bounded reliability of low-agreement
classes** — surprise− α=0.119, disgust α=0.166). **Deaf-community involvement and limitations
statement** — non-optional in this domain. Fill `CLAIMS_LEDGER.md`; verify every number traces to
a run ID + commit SHA. arXiv + workshop submission. 3-minute demo video. Public repo.

**Gate:** `repro_all.sh` green on a small subset; paper PDF with every table from a real run.
**Kill switch:** none.
**Survives:** the submission.

---

## §4 KPI dashboard

Never fill a cell from an estimate. Every cell cites a run ID.

| # | KPI | Reference | Target | Current | Milestone |
|---|---|---|---|---|---|
| **K1** | Non-signer FER bias from grammatical markers | 0 = no bias | ≠ 0, 2+ models, CIs | **refuted on WLASL: −0.008…+0.002, MDE 0.003–0.008** | M1 |
| **K2** | Disentanglement cross-prediction AUC | 0.5 = perfect | **≤ 0.60**, no affect loss | — | M4 |
| **K3** | Positive control: signer probe AUC | — | ≥ 0.80 | — | M4 |
| **K4** | EmoSign emotion macro-F1, video-only, LOSO | **eJSL EANwH 21.09**; GPT-4o 20.76 | **> 21.09** | — | M4 |
| **K5** | Peak inference VRAM / p95 latency on the 3050 | 4096 MB hard limit | **< 2500 MB / < 400 ms** | **186 MB, 6 models live / 73.2 ms p95 perception** | M2, M7 |
| K6 | Sustained capture FPS | — | ≥ 20 | **20.4 (repeats 19.7-20.4)** | M2 |
| K7 | How2Sign BLEU-4 at ≤80M params | 10.06 published | ≥ 8.0 | — | M5 |
| K8 | Conditioned-gen style acc. @ BERTScore ≥0.90 | — | ≥ 80% | — | M6 |
| K9 | Avatar preference vs neutral | 50% = tie | ≥ 60% | — | M7 |
| K10 | ISL zero-shot Top-1 | chance | report honestly | — | M8 |
| K11 | Reproducibility | — | 1 command, all tables | — | M9 |

**Bold KPIs are the paper.** If K1, K2, K4 or K5 misses, that is the only agenda item at the next
review — do not paper over it.

---

## §5 Parallel async tracks

These are long-lead and never block code. Each needs a named owner and a weekly written check.

1. **BU ASLLRP access request** — filed M0 day 1. Unlocks real `L` labels and English. Fallback
   already documented in M3 Track A / M5 ladder.
2. **Human raters for M7** — 5+, ideally ≥1 signer. Recruit during M4; the study cannot be
   retrofitted.
3. **Paper sections, written continuously** — M1 → Intro + Confound Audit; M2 → Efficiency;
   M3/M4 → Method + Contribution; M5 → Recognition/Translation; M7 → System + Demo;
   M9 → Ethics, Limitations, Conclusion.
4. **Weekly written blocker sweep** — anything red gets an owner and a date.
5. **Friday demo-or-it-didn't-happen** — a running artifact, not a slide. KPI dashboard updated,
   `EXPERIMENT_LOG.md` appended, one paragraph of paper written.

---

## §6 Known open items

- **ASLLRP mirror provenance.** The 200 EmoSign clips come from an ungated re-upload of a
  Boston-University-controlled corpus. Decision taken 2026-09-26: use for research now, formalize
  later. **Open item, owner: reviewer, due before M9 submission.** Before the paper is submitted
  the Ethics section must state the provenance, and we confirm the BU terms or obtain author
  permission. Never redistributed — IDs, labels, weights only.
- **EmoSign video framing quality** — the `crop_*` framing is unvalidated until M0's face-visibility
  gate runs.
- **ASLLRP English absent** — M5 ladder, decide at M5.
- **Single-person execution** — `team.md` RACI is superseded; reviewer sign-off is the only gate.

---

## §7 Definition of done

- [ ] M0–M9 evidence gates closed
- [ ] Every bold KPI met, or its miss explained in Limitations
- [ ] Live demo on the RTX 3050 under 2500 MB peak VRAM, p95 < 400 ms
- [ ] `scripts/repro_all.sh` regenerates every table and figure
- [ ] Every paper number traces to a W&B run ID + commit SHA in `EXPERIMENT_LOG.md`
- [ ] `CLAIMS_LEDGER.md` has no unverified claim
- [ ] Ethics, limitations, Deaf-community statement written; §6 provenance item closed
- [ ] Model cards published; no dataset video redistributed
- [ ] arXiv preprint + workshop submission filed
