# Task 5 questions for the author — ANSWERED

**Status: answered 2026-10-04. Decisions taken, implemented and measured.**

These were written because `make_study_stimuli.py` and `build_study_spec.py` both cite this
file for the baseline condition and it did not exist. A script pointing at a missing
document is a decision that will be made by accident.

The author instructed that the open calls be made rather than escalated. They are recorded
here with their reasoning, because a decision nobody can audit is worth as little as one
nobody made. **None of this creates a preference result.** Everything below is a design
choice or a measurement; no human rating exists and none may be reported.

---

## Q1. What is condition B? → **Arm B is this project's own previous front end**

The study is a pairwise forced choice. Rather than invent a stand-in, the comparison is
between two arms that both already exist in this repository:

| Arm | Pose source | Why |
|---|---|---|
| **A** | **SMPLer-X** — learned whole-body SMPL-X regression (`smpler_x_b32`, ViT-B) | The new front end. A learned 3D prior, so it needs no per-joint rotation solve. |
| **B** | **MediaPipe landmarks → per-joint rotation solve** | What this project did before. Real, not a stub. |

**Why this and not B1–B4 from the original list.** A proxy mesh or scripted-motion baseline
would answer "does a real body beat a stand-in", which is a question about the renderer
rather than about this project's contribution. Arm B is the honest control because it *is*
the project's own output: any difference is attributable to the perception swap alone.
Neither arm was written for the experiment.

Implemented in `scripts/make_avatar_demo.py`. Both arms share one camera, one light and one
mesh model, so the only variable is where the pose came from.

**Fairness details that were got wrong first and are now enforced in code:**

- One camera for both arms. Fitting each separately gave 5.92 m vs 4.28 m on clip 1372, so
  the arms appeared at different scales — and a viewer asked which looks better answers
  "the bigger one".
- The camera distance comes from the **99th percentile** of vertex radius over both arms
  combined. Using `max` let one stray vertex in the landmark arm push the camera far enough
  back to shrink both arms to specks.
- Identical frame rate, and the side-by-side truncates to the shorter arm. Resampling only
  one arm would silently double the other's playback speed.
- No arm label is drawn on any frame. The same files feed the blinded study.

## Q2. How many trials? → **Deferred; the pipeline, not the count, is the blocker**

The corpus offers 1,354 utterance videos and 200 EmoSign clips with landmarks. Trial count
is a function of how many raters can be seated, which is not mine to decide. The pipeline
now runs end to end on 4 clips; `--limit N` scales it.

## Q3. Raters, and is any a Deaf signer? → **Unresolved and not mine to resolve**

The floor is 5 independent raters and inter-rater agreement must be reported. The ethical
point stands regardless: a study of ASL avatar quality judged only by hearing non-signers
does not establish that the work is good, only that some raters preferred one video. This
is now the **single remaining blocker** for M7 and it is a human resource, not code.

## Q4. Seated signer rendered as standing? → **Solved by measurement, not by cropping**

The original finding was that all signers are seated while the renderer drew a standing
figure, with unusable legs (hip→knee 0.06–1.42 torso-lengths).

That was a property of *landmark retargeting*, not of the corpus. With SMPLer-X the regressed
skeletons are genuinely seated — posed mesh heights of 1.49–1.92 m against a standing
1.72 m, with hips below the head in **every** frame of every clip after correction. So the
answer is neither "crop to the upper body" nor "state the limitation": the legs are now
real and the full body is kept.

MediaPipe leg tracking remains broken and that is why the baseline arm still looks wrong.
That contrast is the finding.

## Q5. What if raters cannot tell the arms apart? → **Report the null**

A blinded null is a result and gets reported as one. Two hypotheses in this project are
already refuted and recorded that way, so the standard is set.

---

## What was measured (2026-10-04)

Four EmoSign clips, 541 frames, real geometry, `is_human_mesh: true` on every mesh:

| Clip | Detection coverage | Collapsed frames repaired | Arm B hip→knee min/mean/max |
|---|---|---|---|
| 1372 | 1.00 | 0 | 0.41 / 0.43 / 0.79 |
| 1470684 | 1.00 | 3 | 0.34 / 0.40 / 0.46 |
| 1539625 | 0.99 | 1 | 0.27 / 0.39 / 0.54 |
| 1540021 | 1.00 | 1 | 0.17 / 0.36 / 0.46 |

Arm A after correction: head above pelvis in **100%** of frames (min 0.675 m on clip 1372,
up-axis tilt ≤ 13°).

**These are pipeline measurements, not quality measurements.** They show the perception
front end runs and produces anatomically coherent poses. They do **not** show that Arm A
looks better to a viewer — that needs the raters in Q3. Anyone tempted to read the table as
a result should note that the same table would have looked similar for a confidently wrong
pipeline.

## Honesty constraints carried into the implementation

- **No ratings exist and none were simulated.** `make_avatar_demo.py` emits stimuli and
  measurements only.
- **SMPLer-X is third-party** and knows nothing about sign language. It is a body prior: it
  will render plausible signing whatever the hands say. Its fingers in particular should be
  treated as unverified without a dedicated hand model (HaMeR), and **nothing derived from
  it may be presented as evidence about linguistic content**.
- **Weights are not vendored.** They live in the read-only pipeline tree, licence-gated like
  the SMPL-X parameters. `SMPLERX_ROOT` degrades to a clear error, never a broken half-state.