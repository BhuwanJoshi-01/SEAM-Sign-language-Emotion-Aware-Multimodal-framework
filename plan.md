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
18. **A result does not outlive the code that produced it.** Every result artifact carries a
    stamp of the modules it depends on (`seam.provenance.stamp`), and
    `tests/test_artifact_staleness.py` fails when one no longer matches the tree. A result
    that is overtaken is moved to `artifacts/superseded/` and registered in
    `paper/artifact_status.json`, never overwritten in place and never cited again. This rule
    exists because the yaw fix of 2026-10-04 left three cited results standing on the broken
    code for a day, one of them a headline verdict.
19. **A comparison arm is trained by the same function as the thing it is compared with.**
    Rule 16 for models rather than front ends: M4's entangled baseline had its own
    unweighted losses, and the five-fold gap that produced read as a finding.

---

## §1 Ground truth — measured 2026-09-26

v1 assumed several resources that are not what it said. This table is the ground truth; every
milestone plans against it, not against v1.

| Resource                                                    | Status                                                    | Size                                        | Unlocks                                                            |
| ----------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------ |
| EmoSign labels`catfang/emosign`                           | **ungated**, fetched                                | 43 KB, 200 rows                             | sentiment-7, 10 emotions, 600 free-text Deaf-annotator cue strings |
| EmoSign video via`FangSen9000/ASLLRP_utterances_results`  | **200/200 IDs join** to `crop_original_video.mp4` | ~113 MB                                     | **the affect benchmark — v1's critical path is resolved**   |
| ASLLRP gloss tokens `asllrp_sentence_signs_2025_06_28.csv` | ungated | 17,522 tokens, signer-tagged, frame bounds on a **30 fps session timeline** | continuous recognition + gloss sequences; map to clip frames with `asllrp.crop_frame_range(clip_fps=...)` |
| ASLLRP SignStream non-manual XML | **obtained 2026-10-01** via the DAI Download Cart (not in the mirror; licence-gated, never committed) | 51 collections, 2,407 utterances, 43,038 non-manual events | human `L` labels, frame-level, for all 200 EmoSign utterances — on a **30 fps timeline**, so they must be rescaled for the 138 clips that are 24 fps (`signstream.frame_mask`) |
| ASLLRP English sentences | **in the SignStream XML** (found 2026-10-05; not in the mirror) | a translation on 2,403 of 2,407 utterances, 200/200 EmoSign | gloss→English with real references — the first rung of the M5 ladder is available |
| How2Sign `martinctl/how2sign-asl-landmarks` | ungated, **on disk** (corrected 2026-10-04: the `Kavitha/how2sign_user3_mediapipe_pose` repo first named here holds JPEGs and a constant caption — no keypoints, no English) | 4.8 GB, 991 shards, 35,176 sentences | gloss-free SLT; MediaPipe features, so the published BLEU-4 10.06 (MMPose) is a quoted reference, not like-for-like |
| WLASL local`/home/bhuwan/Videos/wlasl`                    | 3,863 mp4 / 668 glosses                                   | 7 GB                                        | recognition backbone + label-free confound audit                   |
| ASL Citizen`SorensenAI/asl-citizen-poses`                 | ungated MediaPipe`.pose`, **no blendshapes**      | 81 GB, 1000/batch                           | K1/K2 without the 403                                              |
| `Pelmeshek/raf-db-7emotions-mediapipe-768`                | ungated                                                   | 2.7 GB                                      | non-signer FER baseline to audit                                   |
| NSL / INCLUDE local                                         | present                                                   | 8.6 GB                                      | cross-lingual ablation                                             |
| `face_landmarker.task`                                    | **already cached locally**                          | 3.7 MB                                      | blendshape extraction, zero download                               |

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

Repo `src/seam/` (`data/ perception/ preprocess/ features/ models/ affect/ translate/ avatar/ serve/ export/ eval/`); fetcher with SHA256 manifests, **hard-linking local data rather than
re-downloading**; download the 200 EmoSign clips; `pyproject.toml` exact pins; `Makefile: setup lint test train bench repro paper`; pytest + ruff + mypy + pre-commit green; W&B project.

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
**Status 2026-10-05: OPEN at 7/10 resources, on two reviewer-owned decisions** — ASL Citizen
(81 GB, not fetched) and NSL provenance. Face-visibility gate passed 24/24. WLASL is 3,771 of
3,775 indexed clips usable (88 substituted, 4 lost), and `make readiness` now reports that
from the index over every file rather than demanding a repair pass from a 300-file sample.
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
**Correction 2026-10-05.** The audit ran before two changes to the head path (the oscillation
threshold on 09-28 and the yaw decomposition on 10-04). Re-run from the cached FER scores
(`artifacts/audit/confound_audit.json`): the four blendshape markers are unchanged to
the last digit, which is the control, and **`head_shake` and `head_nod` each have one matched
pair**, down from 251 and 13. So the refutation covers brow raise, brow furrow and mouth
morphemes on isolated signs. It says nothing about head shake, for which there is no working
instrument, and nothing about continuous signing.
**Tested on continuous signing 2026-10-05, and not supported there either.** The test this
section asks for was run on the 200 EmoSign utterances with *human* frame-level markers
(`scripts/run_confound_audit_continuous.py`), design and decision rule fixed before the first
run: 0.5 s windows inside the signing span, within-clip matched pairs, placebo and null
controls, support requiring a corrected positive shift in two of three models. **No marker
meets the rule.** Brow furrow — the wh-question example the hypothesis is usually stated
with — leans the *other* way in all three models. Brow raise and rhetorical questions lean
positive in two models and do not survive correction; they do reach significance if the
resting frames at the clip edges are included, which is a signing-versus-resting contrast
and not the claim. Yes/no and wh-questions have too few clips (8 and 4) to estimate.
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

| element                                   | result                                                            |
| ----------------------------------------- | ----------------------------------------------------------------- |
| p50 / p95 / p99, perception, on the 3050  | **41.8 / 73.2 / 78.1 ms** over 900 individually-timed calls |
| sustained rate                            | **20.4 FPS** (repeats: 19.7, 20.0, 20.3)                    |
| sequential baseline, same instrument      | 73.0 ms p50, 12.8 FPS —**concurrency worth 1.75x**         |
| peak VRAM, perception                     | 90 MB                                                             |
| peak VRAM,**6 models live at once** | **186 MB against the 2500 MB ceiling (7%)**, 20.3 FPS       |
| CI gate that fails above the ceiling      | `make bench` exits non-zero; 185 tests, 24 on the guards        |
| FP32↔INT8 parity harness                 | met, and it rejected`fer_cnn_a` INT8                            |
| `onnxruntime-gpu` CUDA EP               | met, after preloading the bundled CUDA 13 libraries               |

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

| category                                   | kappa  | verdict                                                                                       |
| ------------------------------------------ | ------ | --------------------------------------------------------------------------------------------- |
| interrogative ~ rhetorical question        | +0.734 | usable                                                                                        |
| negation                                   | +0.639 | usable                                                                                        |
| topicalization ~ topic/focus               | +0.141 | weak                                                                                          |
| interrogative (wh + yes/no)                | +0.028 | **chance**                                                                              |
| reference-establishment ~ conditional/when | +0.038 | **chance** (`reference_establishment` over-fires: 93 heuristic positives vs 37 human) |

So M3 is now on human labels rather than guesses, and separately has a published result
about how unreliable the pseudo-labels were.

**Withdrawn 2026-10-04, confirmed by re-run 2026-10-05:** the duration-controlled finding
r = 0.554 for negation ↔ head shake, quoted in three bullets below, was produced by a yaw
decomposition that read the wrong matrix entries. On fixed code `head_shake` is zero on 90%
of clips and the pairing is reported as *not interpretable*
(`artifacts/audit/marker_labels.json`). The bullets are kept as the record of what
was claimed; none of them may be cited. The human annotations mark a head shake in 89 of the
200 clips, so the visual marker is missing most of what is there. The current marker set is
**1 usable (`brow_furrow`), 3 degenerate, 2 blind** — a clip-prevalence classification.

**Validated against human frame labels 2026-10-05, and the picture inverts**
(`artifacts/m3/marker_validation.json`, leave-one-signer-out, within-clip AUC, gate 0.80 on
the three large folds). **`brow_raise` is validated**: 0.838 / 0.822 / 0.878 — the marker
M3 called degenerate. `brow_furrow`, the one M3 called usable, is **not** (0.599 on the
largest fold). `head_shake` and `head_nod` are unreadable even by a supervised detector on
head pose and its dynamics, so that is a limit of the signal on 256-pixel crops, not of the
thresholds. One trustworthy visual marker exists; negation has none.

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
- **[WITHDRAWN — yaw bug] Result (partial correlation on log-duration, permutation, Bonferroni ×3):**
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
- **[WITHDRAWN — yaw bug] One channel is corroborated by two independent ground truths.** `head_shake` gives
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
- **Track B is closed** (2026-10-01): 51 SignStream collections, 43,038 human non-manual
  events, 200/200 EmoSign utterances joined.
- Current artefact `artifacts/audit/marker_labels.json`; the pre-fix file is kept under
  `artifacts/superseded/pre_yawfix/` (`paper/artifact_status.json`).

### M4 — Factorized non-manual encoder + EmoSign LOSO ★ CORE CONTRIBUTION

**Status: GATE NOT MET, AND THE NEGATIVE RESULT IS CONFIRMED — decision taken 2026-10-01: report as a refuted hypothesis (option (a)), not a deferred gate.**

**Re-run 2026-10-05, and these are the numbers to cite.** Every figure further down this
section was measured before the yaw fix and with two defects in M4's own instrument: the
entangled baseline was trained without the class and positive-label weights the factorized
model had, and checkpoints were selected on a softmax over a multi-label head. All three
are corrected (`paper/EXPERIMENT_LOG.md`, 2026-10-05).

| | as logged 2026-10-01 | **current** | target |
| --- | --- | --- | --- |
| worst-fold cross-AUC, heuristic `y_L` | 0.7276 | **0.6944** | ≤ 0.60 |
| worst-fold cross-AUC, human `y_L` | 0.7031 | **0.7264** | ≤ 0.60 |
| worst-fold cross-AUC, 3 seeds | — | **0.710 ± 0.014** | ≤ 0.60 |
| signer positive control | 0.9729 | **0.9812** | ≥ 0.80 |
| cross A→L | 0.5048 | 0.601 ± 0.017 | — |
| affect micro-F1, factorized / entangled baseline | 0.359 / 0.059 | 0.324 ± 0.019 / 0.308 | no drop |
| affect balanced accuracy | 0.497 | 0.481–0.507 | reference 0.5 |

- **The verdict is unchanged and is more robust than before:** the gate fails for every
  ablation variant in every one of three seeds, with the control passing.
- **Three claims made below are withdrawn.** "Cross A→L reached chance" was a mean over
  folds on both sides of 0.5. "The full model is best on all three target metrics" and
  "the separation terms work, monotonically" do not survive three seeds and a like-for-like
  baseline: no lever moves cross L→A, and the factorized model's affect micro-F1 is inside
  the seed spread of the baseline's. Claim C2 is not supported.
- **The linguistic task was chosen on a withdrawn result.** `LINGUISTIC_TASK = "negation"`
  was picked "because it is the one M3 found a real effect for". That effect was the yaw
  bug. The label itself is sound (kappa 0.639 against human annotation); whether any
  visual signal for it reaches the encoder is not established.
- **A limit of the gate metric, left as registered:** it takes the larger cross-AUC per
  fold, so a fold at 0.30 counts as separated, and one fold's value swings from 0.30 to
  0.65 across seeds.

**Why this is now a finding rather than a failure.** Two things had to be excluded before
"the factorisation does not hold" was a claim anyone could act on, and both are now
excluded:

1. **The labels were not the cause.** Three of M4's four linguistic labels come from
   `seam.features.syntactic` heuristic pseudo-labels. Those were measured against the
   human ASLLRP annotations downloaded 2026-10-01, on the same 200 clips
   (`artifacts/m3/label_agreement.json`, Cohen's kappa):

   | category                                   | human + | heuristic + | kappa            | verdict |
   | ------------------------------------------ | ------- | ----------- | ---------------- | ------- |
   | negation                                   | 39      | 27          | **+0.639** | usable  |
   | interrogative ~ rhetorical question        | 40      | 46          | **+0.734** | usable  |
   | topicalization ~ topic/focus               | 86      | 84          | +0.141           | weak    |
   | interrogative (wh + yes/no)                | 31      | 46          | **+0.028** | chance  |
   | reference-establishment ~ conditional/when | 37      | 93          | **+0.038** | chance  |

   Three of the four are at or near chance, and `reference_establishment` over-fires on
   93 clips against 37 human positives. That is a plausible cause of leakage on its own:
   z_L fitted to noise, where the noise happens to correlate with a human affect rating of
   the same video.

   **It is not the cause.** Re-running the whole experiment with the *human* labels in the
   same four slots, same folds, same features (`artifacts/m4/factorizer_human_labels.json`):

   |                      | heuristic`y_L` | human`y_L`     |
   | -------------------- | ---------------- | ---------------- |
   | worst-fold cross-AUC | 0.7276           | **0.7031** |
   | cross L→A           | 0.6911           | 0.7004           |
   | cross A→L           | 0.5048           | **0.6174** |
   | GRL head accuracy    | 0.500            | 0.507            |
   | signer control       | 0.9729           | 0.9731           |
   | gate                 | FAIL             | FAIL             |

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

|                                        | single-expr | **multi-label**    | target  |
| -------------------------------------- | ----------- | ------------------------ | ------- |
| trainable windows                      | 314         | **1,765**          | —      |
| cross A→L (linguistic from z_A)       | 0.695       | **0.505** (chance) | —      |
| cross L→A (affect from z_L), weighted | 0.879       | **0.691**          | ≤ 0.60 |
| worst fold                             | —          | **0.884** (Ben)    | ≤ 0.60 |
| affect micro-F1                        | n/a         | **0.359**          | no drop |
| balanced acc, linguistic               | 0.547       | 0.529                    | —      |
| balanced acc, affect                   | 0.173       | 0.497 (ref 0.5)          | —      |
| signer positive control                | 0.971       | **0.973** (passes) | ≥ 0.80 |

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
**M5b gloss-free SLT:** pose → single linear layer → T5-small on the How2Sign landmark cache
now on disk (§1), at **24 fps and 12 fps** to reproduce the frame-rate trade-off. The published
BLEU-4 10.06 / 9.53 used 255-dim MMPose keypoints; ours are MediaPipe-derived, so those figures
are a quoted reference and the paper must say so. M5b is the translation claim with a citable published reference; M5a feeds
the demo.
**Known gap, closed 2026-10-05:** ASLLRP English is not in the mirror, but the SignStream XML
downloaded for M3 carries a translation for 2,403 of its 2,407 utterances, including all 200
EmoSign clips. The ladder's first rung (BU English) is therefore available; How2Sign English
remains the gloss-free path, and no synthetic back-translation is needed.

**M5a status 2026-10-05: gate not met, on a corrected instrument.** The first run (WER 0.916
= most-frequent baseline 0.916) is superseded: its model predicted a single gloss in every
fold, and its token frames were misaligned on the 138 clips that are 24 fps, because ASLLRP
frame indices are on a 30 fps timeline. Both are fixed and tested (a positive control for the
recogniser; an independent blink-based check for the mapping). Re-run: 1,736 tokens over 546
glosses, WER 0.920 against a most-frequent baseline of 0.920. Still no better than the
baseline, and now a measurement. It is a data-scale verdict — 3.18 tokens per gloss — and the
1,354 crop videos on disk are the fair test.
**M5b status:** not started; `src/seam/translate/` is empty; data on disk.

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
  reference (0.481–0.507 over three seeds as re-measured 2026-10-05), and gloss recognition
  is withheld because M5a does not beat its most-frequent baseline.
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

| #            | KPI                                           | Reference                                | Target                            | Current                                                       | Milestone |
| ------------ | --------------------------------------------- | ---------------------------------------- | --------------------------------- | ------------------------------------------------------------- | --------- |
| **K1** | Non-signer FER bias from grammatical markers | 0 = no bias | ≠ 0, 2+ models, CIs | **no support, measured twice.** Isolated signs (WLASL, heuristic markers): −0.008…+0.002, MDE 0.003–0.008. Continuous signing (EmoSign, human frame-level markers, 2026-10-05): no marker meets the pre-registered rule; brow-marker MDE 0.012–0.037 against baselines of 0.40–0.46 | M1 |
| **K2** | Disentanglement cross-prediction AUC | 0.5 = perfect | **≤ 0.60**, no affect loss | **not met: 0.6944 heuristic `y_L`, 0.7264 human `y_L`; 0.710 ± 0.014 over 3 seeds** (`artifacts/m4/`, 2026-10-05) | M4 |
| **K3** | Positive control: signer probe AUC | — | ≥ 0.80 | **met: 0.9812** | M4 |
| **K4** | EmoSign emotion macro-F1, video-only, LOSO | **eJSL EANwH 21.09**; GPT-4o 20.76 | **> 21.09** | **never measured on a comparable protocol.** Affect balanced accuracy 0.481–0.507 against 0.5, so no affect signal is learned yet | M4 |
| **K5** | Peak inference VRAM / p95 latency on the 3050 | 4096 MB hard limit                       | **< 2500 MB / < 400 ms**    | **186 MB, 6 models live / 73.2 ms p95 perception**      | M2, M7    |
| K6           | Sustained capture FPS                         | —                                       | ≥ 20                             | **20.4 (repeats 19.7-20.4)**                            | M2        |
| K7 | How2Sign BLEU-4 at ≤80M params | 10.06 published | ≥ 8.0 | not started; data on disk | M5 |
| K8 | Conditioned-gen style acc. @ BERTScore ≥0.90 | — | ≥ 80% | not started | M6 |
| K9 | Avatar preference vs neutral | 50% = tie | ≥ 60% | stimuli ready, **0 raters** | M7 |
| K10 | ISL zero-shot Top-1 | chance | report honestly | blocked on NSL terms | M8 |
| K11 | Reproducibility | — | 1 command, all tables | partial: `scripts/repro_all.sh` (15 stages) + provenance and staleness guards; paper tables not yet generated from it | M9 |

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
- ~~**EmoSign video framing quality**~~ — closed: the face-visibility gate passed 24/24 sampled
  clips and the full census found a usable face in 200/200.
- ~~**ASLLRP English absent**~~ — closed 2026-10-05: translations are in the SignStream XML
  for 2,403 of 2,407 utterances.
- **Frame-level use of ASLLRP annotations requires the clip frame rate.** Indices are on a
  30 fps timeline; 138 of 200 EmoSign clips and about two-thirds of the wider mirror are
  24 fps. Any new consumer must go through `asllrp.crop_frame_position`.
- **One open experiment has never been run** and has all its inputs: K4 on a protocol
  comparable to the published 20.76 / 21.09. (Run on 2026-10-05: C1 on continuous signing —
  not supported; marker validation against human frames — `brow_raise` validated, the other
  three not.)
- **C1 for question marking is untested for want of clips.** The 1,354 crop videos on disk
  would supply them; they need perception run over them and the frame-rate-aware mapping.
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
