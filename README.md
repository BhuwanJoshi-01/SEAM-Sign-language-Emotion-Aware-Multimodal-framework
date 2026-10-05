# SEAM — Sign-language Emotion-Aware Multimodal framework

**A real-time computer-vision pipeline that reads the face, head, hands and body of a person
signing, and tests whether facial *grammar* can be told apart from facial *emotion*.**

In American Sign Language the face is part of the grammar: raised brows mark a yes/no
question, furrowed brows mark a wh-question, and a head shake marks negation. The same face
also shows feeling. Emotion-recognition models are trained on hearing non-signers, so the
concern is that they read grammar as emotion. SEAM builds the vision pipeline needed to study
that, measures every part of it against human annotation, and reports what held and what did
not.

| | |
|---|---|
| **Live demo** | [`docs/index.html`](docs/index.html) — runs fully in the browser. Hosted on GitHub Pages once enabled: `https://bhuwanjoshi-01.github.io/SEAM-Sign-language-Emotion-Aware-Multimodal-framework/` |
| **Run locally** | `make setup && make serve`, then open <http://127.0.0.1:8000/> |
| **Team reading guide** | [`docs/TEAM_GUIDE.md`](docs/TEAM_GUIDE.md) — what the project is, what happened, viva preparation |
| **Full experiment record** | [`paper/EXPERIMENT_LOG.md`](paper/EXPERIMENT_LOG.md), [`paper/CLAIMS_LEDGER.md`](paper/CLAIMS_LEDGER.md) |

![The live demo tracking a face, a hand and the upper body, with the linguistic and affective channels read out beside the video](docs/figures/demo_live.png)

*The live page: 478 face landmarks, two hands and the body tracked in the browser with
MediaPipe Tasks, and the two non-manual channels read out in real time. The image is
MediaPipe's public sample photo; no dataset video is shown or shipped.*

---

## Contents

1. [Results at a glance](#1-results-at-a-glance)
2. [Background](#2-background)
3. [Methodology](#3-methodology)
4. [Results](#4-results)
5. [The live demo](#5-the-live-demo)
6. [Limitations and ethics](#6-limitations-and-ethics)
7. [Future work](#7-future-work)
8. [Reproducing everything](#8-reproducing-everything)
9. [Repository map](#9-repository-map)

---

## 1. Results at a glance

| What we asked | What we measured | Verdict |
|---|---|---|
| Can the full perception stack run in real time on a 4 GB laptop GPU? | 41.8 ms median, **73.2 ms p95**, 19.7–20.4 FPS; **186 MB** peak VRAM with six models live against a 2,500 MB budget | **Yes** |
| Can a grammatical marker be read off the face on signers never seen? | Brow raise vs human frame-level annotation: within-clip AUC **0.838 / 0.822 / 0.878** on three held-out signers | **Yes, for brow raise** |
| Can the other markers be read too? | Brow furrow 0.60–0.78. Head shake **0.70–0.76** and head nod 0.55–0.77 once the correct head axis is read; both had scored 0.50 while our offline code measured the wrong axis | **Partly: above chance, below the 0.80 gate** |
| Does a head shake mark negation in our data? | A head shake is detected in **25 of 27** negated clips and 80 of 173 others (permutation p = 0.00015); the linguists' own marks show the same pattern, and the nod axis shows nothing | **Yes, on these four signers** |
| Do emotion models read grammar as negative emotion? | Isolated signs: head-shake windows are read as more negative by 2 of 3 models (+0.017 in negative-emotion probability, 12 clips); no brow effect. Continuous signing, human markers, pre-registered rule: **no marker met the rule** | **One marker, small sample; not confirmed on continuous signing** |
| Can a factorized encoder separate grammar from affect? | Worst-fold cross-prediction AUC **0.705–0.758** against a target of ≤ 0.60, in all 3 seeds and 5 ablation variants | **Refuted** |
| Can pose alone recognise glosses on this corpus? | WER 0.920 from the body, 0.922 with both hands added; always predicting the most frequent gloss gives 0.920 | **Not at this data scale** |
| Can a video become an animated 3D avatar? | SMPL-X body and fingers regressed per frame and exported as a skinned, animated glTF (55 joints, 6.4 mm error); frame-to-frame jump cut from 0.32–0.69 m to 0.004–0.035 m; fingers and wrists solved from MediaPipe hand points | **Yes** |

Two of our original hypotheses were refuted. We report them as results, because each was
measured with an instrument that passes a positive control. The record also contains the
bugs we found in our own instruments, which changed several numbers; they are listed in
[section 4.7](#47-what-we-got-wrong-and-how-we-found-out).

---

## 2. Background

### 2.1 The problem

Sign languages are full natural languages with five parameters: handshape, location,
movement, orientation, and **non-manual signals** (face, brows, mouth, gaze, head, torso).
Most sign-language recognition systems model the hands and discard the face. That loses two
things at once, because the face does two jobs:

| Function | Example in ASL | What a naive model sees |
|---|---|---|
| **Linguistic** (grammar) | Raised brows on a yes/no question; furrowed brows on a wh-question; head shake for negation | "Surprise", "anger", "disagreement" |
| **Affective** (emotion) | A smile, a frown, widened eyes | The same blendshapes as above |

The EmoSign benchmark (2025) reported that a hearing non-signer perceived neutral ASL clips
as negative, and that large multimodal models do poorly at emotion from sign video (GPT-4o
reaches a weighted F1 of 20.76). That is the motivation: a system for signers needs to know
which facial movements are grammar.

### 2.2 What we set out to test

| ID | Hypothesis |
|---|---|
| C1 | Non-signer facial-emotion models systematically read grammatical markers as negative emotion |
| C1b | The grammatical and affective signals can be separated in a learned representation, and the separation can be measured |
| C4 | The whole perception pipeline fits a 4 GB consumer GPU at interactive latency |

### 2.3 Data

| Dataset | What it is | How we used it |
|---|---|---|
| **EmoSign** | 200 ASL utterances, 4 signers, emotion labels from 3 Deaf annotators | The affect benchmark; leave-one-signer-out evaluation |
| **ASLLRP SignStream** | Linguistic annotation of the same utterances by ASL linguists: 2,407 utterances, **43,038 frame-aligned non-manual events** | Human ground truth for brow raise, furrow, head shake, nod, negation, questions |
| **WLASL** | 3,771 usable isolated-sign clips | First confound audit (isolated signs) |
| **RAF-DB** | Face images of non-signers with 7 emotion labels | Training (5,164 images) and testing (2,652) the non-signer emotion models we audit |
| **How2Sign landmarks** | 35,176 sentences with English | Prepared for translation; not yet trained |

EmoSign is built on ASLLRP, so the *same 200 utterances* carry emotion labels from Deaf
annotators and grammar labels from linguists. That overlap is what makes the grammar-versus-
emotion question testable at all. All 200 utterance IDs join exactly between the two.

Signer video is never redistributed. This repository contains code, tests, documentation and
figures only.

---

## 3. Methodology

### 3.1 System overview

```mermaid
flowchart LR
    V[Video frame] --> F[Face Landmarker<br/>478 points · 52 blendshapes · head transform]
    V --> H[Hand Landmarker ×2<br/>21 points each]
    V --> P[Pose Landmarker<br/>33 points]
    F --> N[Normalise · interpolate · One-Euro filter · resample to 12 fps]
    H --> N
    P --> N
    N --> NM[Non-manual channel<br/>blendshapes + head pose]
    N --> M[Manual channel<br/>hands and arms]
    N --> PR[Prosody channel<br/>speed · size · pauses]
    NM --> MK[Marker read-out<br/>brow raise · furrow · head shake · nod]
    NM --> ENC[Factorized encoder<br/>z_L linguistic · z_A affect]
    PR --> ENC
    M --> REC[Gloss recogniser]
    MK --> VAL[Validation against<br/>human frame-level annotation]
    V --> SX[SMPLer-X body regressor] --> AV[Skinned glTF avatar]
```

Everything downstream of perception works on landmarks, not pixels. That is why the pipeline
is light (keypoints are a few hundred numbers per frame) and why the live demo can keep the
video on the user's device.

### 3.2 Perception: three MediaPipe graphs per frame

We use the **MediaPipe Tasks** API, not the older Holistic solution, because only Tasks'
Face Landmarker emits the 52 ARKit-style **blendshape coefficients** and the facial
transformation matrix.

| Graph | Output | Why we need it |
|---|---|---|
| Face Landmarker | 478 3D points, 52 blendshape scores, a 4×4 head transform | Brows, eyes, mouth and head pose: the non-manual channel |
| Hand Landmarker (one graph, two hands) | 21 3D points per hand | The manual channel and signing prosody |
| Pose Landmarker | 33 body points | Shoulders for normalisation; arms |

Together that is **553 landmarks plus 52 blendshapes per frame**. Three implementation
details mattered:

- **The three graphs run concurrently.** TensorFlow Lite releases the Python GIL, so running
  them in threads cuts the median frame time from 73.0 ms to 41.8 ms, a **1.75× speed-up**.
  A test asserts the concurrent and sequential outputs are identical.
- **Timestamps must increase per graph instance.** Decoding two videos back to back without a
  process-wide monotonic clock silently degrades tracking.
- **The blendshape order is asserted, not assumed.** The real model emits `_neutral` plus 51
  coefficients in an order that differs from the ARKit documentation. A permuted basis would
  invert every brow rule while passing every shape check.

Where time goes (measured per module): hands 41.2 ms, pose 29.5 ms, face 15.9 ms. Input
resolution made no difference between 160 and 480 pixels, so the cost is per-graph overhead,
which is why concurrency helps and downscaling does not.

### 3.3 Preprocessing

1. **Gap interpolation** for frames where a hand or the face is not detected, with a maximum
   gap so a long absence is not invented.
2. **Normalisation**: translate to the shoulder midpoint and divide by shoulder width, making
   features invariant to where the signer stands and how far away they are.
3. **One-Euro filter**: an adaptive low-pass filter that removes jitter at low speed without
   adding lag at high speed, which matters for fast signing.
4. **Resampling to 12 fps** and windowing. Attention cost grows with the square of sequence
   length, so halving the frame rate cuts it by about 75%.

Each step has property tests: translation and scale invariance, filter causality, and so on.

### 3.4 Three feature channels

| Channel | Built from | Captures |
|---|---|---|
| **Manual (M)** | Hand and arm keypoints | What is being signed |
| **Non-manual (NM)** | 52 blendshapes, head rotation | Grammar and emotion on the face |
| **Prosody (P)** | Speed, movement size, pauses, jerk, repetition of the hands | *How* it is signed; Deaf annotators name these as emotion cues |

### 3.5 Reading grammatical markers from blendshapes

Each marker is a small, inspectable function of the face signals:

| Marker | Signal | ASL function |
|---|---|---|
| Brow raise | mean of `browInnerUp`, `browOuterUpLeft`, `browOuterUpRight` | Yes/no question, topic |
| Brow furrow | mean of `browDownLeft`, `browDownRight` | Wh-question |
| Head shake | back-and-forth energy of the head's left-right turn, in degrees RMS | Negation |
| Head nod | back-and-forth energy of the head's up-down angle, in degrees RMS | Affirmation |

Brow thresholds are **relative to the person's own baseline**, not absolute, because a signer
who habitually holds a slight furrow would otherwise be "furrowing" on every frame.

Head angles are read from MediaPipe's facial transformation matrix as the direction the face
points (its third column) and its right-hand direction (its first). A shake or a nod is then
scored by three running averages: a fast one removes tracking noise, a slow one follows
posture, and the energy of their difference is the part of the angle that goes back and
forth. The page reports it in degrees and additionally requires two real reversals before it
calls an event, so that one quick turn of the head is not a "shake".

> **A naming bug worth knowing about.** Our first offline implementation took the angles
> from a ZYX Euler decomposition and named them roll, pitch and yaw. Those names assume the
> forward axis is x, as on an aircraft; a face's forward axis is z. The mathematics was
> right and the names were not: "head shake" was reading head *tilt*, and "head nod" a head
> *turn*. Section 4.2 shows what that cost.

### 3.6 Aligning human annotation to video frames

The linguists' annotations give start and end frames for every event. We found that those
frame numbers are on a **30 fps session timeline**, while 138 of our 200 clips are 24 fps
re-encodes. A frame-for-frame mapping therefore ran 25% fast on most of the data.

We did not discover this by reading documentation. We tested alignment against an
independent signal: annotated **blinks** against the eye-blink blendshape. A blink lasts a
few frames, so the two only agree when the mapping is right.

![Line chart of alignment AUC against the scale applied to annotated frames. For 24 fps clips the curve peaks near 0.82; for 30 fps clips it peaks at 1.00.](docs/figures/frame_alignment.png)

On the 24 fps clips the frame-for-frame mapping scores AUC 0.554 (near chance) and the
frame-rate-aware mapping 0.710. Each group peaks exactly where its frame rate predicts.
The conversion now lives in one function that requires the clip's frame rate and has no
default.

### 3.7 Validating markers against human annotation

For each marker we score the detector's per-frame signal against the human track:

- **Leave one signer out**: the three large signers (87, 54 and 52 clips) each take a turn
  as the unseen test signer.
- **Within-clip AUC**: the mean of per-clip AUCs, so a detector cannot score well merely by
  telling clips or signers apart.
- **A gate fixed before the run**: validated at 0.80 on every large fold.
- **Two detectors per marker**: the hand-written signal above, and a logistic regression on
  all 52 blendshapes plus head pose and its short-range dynamics.

### 3.8 The confound audit: do emotion models read grammar as negative emotion?

**The emotion models.** We trained three compact CNNs from scratch on RAF-DB (non-signer
faces) so that their training distribution is known. They reach 61.7%, 63.7% and 66.3% test
accuracy. Faces are cropped from the landmarks and passed through one shared front end:
grayscale, histogram equalisation, per-image standardisation.

**The design.** We compare the model's *negative probability mass* (anger + disgust + fear +
sadness) on windows that carry a marker against matched windows of the same clip that do
not:

| Element | Choice | Reason |
|---|---|---|
| Contrast | Within clip | A clip's emotion label is constant, so emotion cannot cause the difference |
| Matching | Nearest window on hand speed and movement size | So a busy window is not compared with a still one |
| Region | Only inside the signing span | So a signing face is not compared with a resting face |
| Uncertainty | Bootstrap over clips, 4,000 resamples | Windows in a clip are correlated |
| Power | Minimum detectable effect beside every null | A null without its power is not a finding |
| Placebo | The same marker track, shifted in time | A real effect must vanish here |
| Decision rule | Positive shift, Bonferroni-corrected, in 2 of 3 models | Fixed before the first run |

### 3.9 The factorized encoder

The model has two small branches over the non-manual and prosody features. One produces a
linguistic code `z_L`, the other an affective code `z_A`, and the loss pushes them apart:

```
L = CE(head_L(z_L), marker)                    linguistic supervision
  + BCE(head_A(z_A), emotions)                 affective supervision (multi-label)
  + BCE(head_A'(GRL(z_L)), emotions)           z_L must not predict emotion
  + CE(head_L'(GRL(z_A)), marker)              z_A must not predict the marker
  + λ · ‖ z_Lᵀ z_A ‖²                          orthogonality
  + λ · JSD-MI(z_L ; z_A)                      mutual-information bound
```

`GRL` is a gradient-reversal layer: the head learns to predict, and the encoder beneath it
learns to prevent that.

**How separation is measured.** After training, frozen probes try to predict emotion from
`z_L` and the marker from `z_A`. The **cross-prediction AUC** is 0.5 when the codes are
perfectly separated. The target, fixed in advance, was ≤ 0.60 on the worst fold under
leave-one-signer-out.

**The positive control.** A metric that reads 0.5 on everything proves nothing, so the same
probe must be able to detect something that is certainly there: which signer is signing. It
must score at least 0.80.

### 3.10 Gloss recognition baseline

A linear model on per-token pose summaries (mean, spread and frame-to-frame change of the
upper-body and hand landmarks), evaluated signer-disjoint, with three references reported
beside it: the most-frequent-gloss baseline, a shuffled-label control, and the
out-of-vocabulary floor.

### 3.11 Video to 3D avatar

A learned whole-body regressor (**SMPLer-X**, ViT-B) predicts SMPL-X body parameters per
frame: pose, facial expression and body shape. We convert its output to a
standard orientation, repair frames where the regressor loses the person, and export one
**skinned mesh with a real animation clip** in glTF format (55 joints, inverse bind matrices,
110 animation channels). The export is verified by loading it in a completely independent
reader, three.js in headless Chrome, which found two bugs our own verifier had agreed with.

A per-frame regressor shakes, so the sequence is **stabilised** before export. The
regressor's estimate of the body's distance from the camera is close to noise from one frame
to the next, which made the first avatar jump back and forth; we hold that axis at its
median over the clip. Every joint rotation, including the 30 finger joints, is then smoothed
with a Gaussian window computed on the rotation sphere (on quaternions, after aligning their
signs), not on the raw axis-angle numbers, which wrap around at 180 degrees.

**The hands do not come from the body regressor.** A whole-body model sees a hand as a few
dozen pixels, and its 30 finger joints barely move: over a clip they open and close through
only 12 to 24 degrees while the signer forms handshapes. We already run a dedicated
hand model, so the fingers are solved from its 21 points per hand instead:

1. The palm is rigid, so three of its points (wrist, index knuckle, little-finger knuckle)
   define a frame. We build that frame on the tracked hand and on the body model's rest hand.
2. Expressed in its own palm frame, each finger bone of the tracked hand is a direction the
   model's bone has to reach. Each joint's rotation is the smallest rotation that takes its
   rest bone there, solved down the chain from knuckle to fingertip.
3. The wrist is solved the same way in world space, from where the palm frame points, and
   kept only when the result is anatomically possible.

Nothing in this depends on the camera, the hand's size in the image, or the arm. A hand lost
for up to eight frames is bridged on the rotation sphere; where there is no hand to track,
the regressor's value stays. The solver is tested on a synthetic hand, where it has to
reproduce every bone it is given for any orientation and scale, for left and right hands.

### 3.12 Efficiency

Models are exported to ONNX and checked for numerical parity against PyTorch on real faces.
Latency is reported as percentiles over individually timed calls, on the target GPU, with
the CPU governor and core clock recorded, because the same benchmark read 118 ms instead of
42 ms when the machine was in power-save mode.

### 3.13 Making the numbers trustworthy

This project found many of its own bugs, and built guards so they cannot return silently:

- **Provenance test**: every number in the experiment log must exist in a result file.
- **Staleness test**: every result file records a fingerprint of the code that produced it,
  and the build fails when the code changes underneath a cited result.
- **One front end**: training and inference paths must call the same preprocessing function,
  asserted by test.
- **Positive controls**: every metric is shown to detect an effect that is planted.

The suite has over 500 automated tests.

---

## 4. Results

### 4.1 Real-time perception on a 4 GB GPU

| Measure | Value |
|---|---|
| Median / p95 / p99 frame time | 41.8 / **73.2** / 78.1 ms over 900 timed calls |
| Sustained rate | 20.4 FPS (repeat runs: 19.7, 20.0, 20.3) |
| Sequential baseline | 73.0 ms, 12.8 FPS, so concurrency is worth 1.75× |
| Peak VRAM, perception | 90 MB |
| Peak VRAM, six models live | **186 MB** of a 2,500 MB budget (7%) |
| Emotion CNNs on ONNX Runtime | 6.9–11.6 ms median per call, 188–578 KB on disk |

The target of 20 FPS is met at the median and missed at the low end, and we report it that
way. The limit is the CPU, not VRAM.

### 4.2 Which markers can actually be read

![Dot plot of within-clip AUC for four markers on three held-out signers. Brow raise sits between 0.82 and 0.88, above the 0.80 gate. Brow furrow lies between 0.60 and 0.78. Head shake lies between 0.70 and 0.76 and head nod between 0.55 and 0.77, with hollow markers at 0.50 showing where both stood while the wrong head axis was read.](docs/figures/marker_validation.png)

| Marker | Detector | Cory | Jonathan | Rachel | Verdict |
|---|---|---|---|---|---|
| **Brow raise** | mean of three blendshapes | 0.838 | 0.822 | 0.878 | **Validated** |
| Brow furrow | mean of two blendshapes | 0.599 | 0.717 | 0.780 | Not validated |
| Head shake | first version, wrong axis | 0.500 | 0.502 | 0.500 | (a bug, not a result) |
| Head shake | **correct axis, as on the live page** | 0.757 | 0.702 | 0.696 | Above chance, not validated |
| Head nod | first version, wrong axis | 0.498 | 0.500 | 0.499 | (a bug, not a result) |
| Head nod | **correct axis, as on the live page** | 0.592 | 0.767 | 0.549 | Weak |

**Brow raise works**, on signers the detector was never tuned on, using the plain mean of
three blendshapes. A fitted 61-feature detector did *worse* (0.70–0.87): with four signers,
more capacity bought less generalisation.

**Head movements looked unreadable, and that was our bug.** In the first run both head
markers scored exactly 0.50, and so did a fitted detector, and we wrote that the input does
not carry head movement. Then the live page, written separately, plainly responded to a head
shake. Comparing the two showed the offline code was reading head tilt for "shake" and head
turn for "nod" (section 3.5). We confirmed it without trusting either implementation, by
correlating each angle with the face landmarks themselves: the angle the offline code called
"yaw" follows the tilt of the eye line (|r| = 0.95) and the one it called "pitch" follows
the nose moving sideways (|r| = 0.90).

On the correct axis, **head shake reads 0.70 to 0.76** on the three held-out signers. That is
well above chance and still short of our gate, so we report it as visible but not validated.
**Head nod stays weak**: one signer at 0.77, two under 0.60. The detector's three constants
were chosen on the training signers only in every fold.

A ranking score is not the whole story for a live demo, which has to say yes or no. At the
threshold the page uses, the head-shake event fires on 40%, 10% and 52% of the frames the
annotators marked for Cory, Jonathan and Rachel, with false alarms on 12%, 0% and 16% of the
frames they did not. The event the page fired before this change managed 33%, 9% and 45%.
A threshold in degrees does not transfer well between people: fluent signers shake the head
by a few degrees, some by much less than others. The page responds far more reliably to a
deliberate movement, and we describe it that way.

**Brow furrow could not be improved.** We tried six alternative signals, including brow
height and inner-brow gap measured directly on the landmarks, and a small fitted model. None
lifted the weakest signer above 0.64. For two of our signers the `browDown` blendshape sits
near 0.42 even when the annotators mark nothing, so there is little room left for it to
rise.

### 4.3 Do emotion models read grammar as negative emotion?

We ran the audit on two corpora, and after the corrections of 5 October they no longer say
the same thing.

**Continuous signing (200 utterances, human frame-level markers, pre-registered rule).**

![Forest plot of the shift in negative-emotion probability for eight markers across three emotion models, with 95% intervals. Most intervals cross zero.](docs/figures/c1_continuous.png)

On 2,646 scored half-second windows, **no marker met the pre-registered rule**:

- **Brow furrow leans the other way** in all three models. This is the marker the usual
  example rests on (a wh-question furrow read as anger).
- **Brow raise and rhetorical questions lean positive in two models** and do not survive
  correction for multiple tests.
- **Head shake leans positive in one model** and does not survive correction either.
- **Yes/no and wh-questions could not be tested**: only 8 and 4 clips contribute.

The same analysis *without* restricting to the signing span reports support for brow raise.
That version compares signing faces with resting faces, which is a different question. The
restriction was decided before either analysis ran.

**Isolated signs (2,565 WLASL clips, markers detected by our own code).** For the brows and
the mouth the result is null, with shifts between -0.008 and +0.002. Head shake is
different, and it is new: until the head-axis fix this row had one marked window and was
not a test at all.

| Emotion model | Matched pairs | Clips | Shift in negative-emotion probability | 95% interval | p |
|---|---|---|---|---|---|
| A | 14 | 12 | +0.0172 | +0.0054 to +0.0286 | 0.0050 |
| B | 14 | 12 | +0.0171 | +0.0062 to +0.0317 | 0.0005 |
| C | 14 | 12 | +0.0063 | -0.0210 to +0.0406 | 0.6925 |

Two of three models read head-shake windows as more negative than matched windows of the
same clip without one, with intervals that exclude zero. That meets the criterion we set at
the start (a non-zero shift in two or more models), and it is the first support for the
hypothesis anywhere in the project.

We do not report it as a confirmation, for three reasons we can state:

1. It is **12 clips**. The shift is smaller than the effect this design could detect with
   80% power.
2. **It did not replicate** on continuous signing, where the head shakes were marked by
   linguists and there are many more of them.
3. **There is a simpler explanation.** The classes that move are sadness and surprise, at the
   expense of neutral. A face turned away from the camera could do that to a classifier
   trained on frontal faces, whether or not anything grammatical is happening.

**What this section does not show:** that the confound does or does not exist. These are
small CNNs, and question marking could not be tested on this corpus.

### 4.3b Negation shows up as a head shake

In ASL a head shake marks negation. Once our head-shake marker was reading the right axis,
that textbook fact appeared in the data. Each clip has a negation label read from its
glosses and a head-shake magnitude read from its video.

| | Negated clips | Other clips |
|---|---|---|
| Head shake detected by our marker | 25 of 27 | 80 of 173 |
| Head shake marked by the linguists | 20 of 27 | 69 of 173 |

The difference in head-shake magnitude between negated and other clips is 0.97 pooled
standard deviations after controlling for clip length (permutation test, p = 0.00015;
p = 0.00045 after correcting for the three pairings we had registered).

This claim has been wrong here before: an earlier version of it was produced by a bug and
withdrawn. So we checked it three more ways before writing it down:

- **The annotators see it too** (second row of the table), so it is in the signing and not
  only in our detector.
- **It is specific to the shake axis.** The same statistic for head *nod* is 0.07.
- **It holds inside each signer** who has negated clips to compare: 0.79 for Cory,
  0.80 for Rachel, 0.88 for Ben.

Limits: 27 negated clips, 15 of them one signer's; negation is read from the glosses by
rule; and the marker's threshold was set on these clips (against head-shake marks, not
against negation), so this is not an out-of-sample estimate.

### 4.4 The factorized encoder does not separate grammar from affect

![Dot plot of worst-fold cross-prediction AUC for five model variants, three seeds each. All lie between 0.68 and 0.82, above the 0.60 target.](docs/figures/m4_gate.png)

| Quantity | Result | Target |
|---|---|---|
| Worst-fold cross-AUC, heuristic labels | 0.705 | ≤ 0.60 |
| Worst-fold cross-AUC, human labels | 0.758 | ≤ 0.60 |
| Over three seeds | 0.709 ± 0.021 | ≤ 0.60 |
| Signer positive control | 0.980 | ≥ 0.80 |
| Emotion balanced accuracy | 0.493–0.506 | reference 0.5 |

The gate fails in every seed and every ablation variant, while the positive control passes,
so the instrument can see and the model does not separate. Swapping the heuristic grammar
labels for human ones did not rescue it, which rules out label noise as the cause. The
emotion head does not beat chance, so the live demo does not show an emotion label.

A related finding that others can reuse: our heuristic grammar labels agreed with human
annotation well for negation (Cohen's κ = 0.639) and rhetorical questions (0.734), and at
**chance** for wh/yes-no questions (0.028) and conditionals (0.038).

### 4.5 Gloss recognition

| Features | Dimensions | WER | Most-frequent-gloss baseline | Shuffled-label control |
|---|---|---|---|---|
| Upper body | 82 | 0.920 | 0.920 | 0.910 |
| Upper body and both hands | 460 | 0.922 | 0.920 | 0.910 |

On 1,736 tokens over 546 glosses (3.18 examples per gloss), word error rate equals that of
always predicting the most frequent gloss.

The second row is new and was the one with a reason to work. Most of what distinguishes one
sign from another is the handshape, and until 5 October our pipeline had never really
tracked two hands (section 4.7), so the hands variant had never had real input. With both
hands properly tracked it still does not beat the baseline. That points at the data, not
the features: 25 to 32% of each test signer's tokens are glosses the model never
saw in training, and the rest have about three examples each. This is a data-scale result,
not evidence that pose is uninformative.

### 4.6 Avatar

| | |
|---|---|
| Structure | 1 skinned mesh, 55 joints, 110 animation channels |
| Error against the mesh pipeline | 6.4 mm (glTF allows 4 skinning influences; SMPL-X uses up to 10) |
| Detection coverage on 8 clips, 1,232 frames | 0.99–1.00 |
| Root movement between frames, before and after stabilising | 0.32–0.69 m, then 0.004–0.035 m (at least 17 times less on every clip) |
| Each hand tracked by MediaPipe | 80–99% of frames |
| How far the fingers open and close over a clip, body regressor | 12–24° |
| The same, solved from MediaPipe's hand points | 44–76° |
| Finger-joint movement between frames, before and after smoothing | 0.24–0.43 rad, then 0.09–0.16 rad |
| Verified by | three.js `GLTFLoader` in headless Chrome, which loads and plays every clip (`scripts/avatar_smoke.py`) |

We checked the solved hands by eye against the source frames (an open palm facing the
camera, a hand at the chin, two flat hands edge-on all come out as in the video). That is
an inspection, not a measurement: no human preference study has been run, and we make no
claim about how well the avatar signs.

### 4.7 What we got wrong, and how we found out

| Bug | Effect | How it was caught |
|---|---|---|
| Emotion models trained and applied with different face crops | All three predicted one class for all 200 clips | Per-frame spread within a clip was 0.0002 |
| Head yaw read from the wrong matrix entries | Head shake was zero for a real head shake; one "positive result" was produced by the bug | The live demo showed a blank row |
| Head angles named by an aircraft convention | "Head shake" measured head tilt and "head nod" a head turn; both scored 0.50 and were written up as unreadable | The browser page, written separately, worked; the landmarks settled which was right |
| The body regressor's depth used as the avatar's position | The avatar jumped 0.32 to 0.69 m between frames | Watching it, then measuring the step size per axis |
| The software renderer flipped left and right | Every offline avatar video was a mirror image | A test that renders one raised hand and asks which side it is on |
| Two identical hand trackers, each asked for one hand | Both found the same hand, so the "left" and "right" slots held one hand twice in every frame of every clip. The project tracked one hand and called it two | The avatar's hands would not sign; comparing the two slots showed them identical on 80 of 80 frames |
| A hard-coded census of the markers kept inside the marker module | Recording a new measurement changed the code every result is fingerprinted against, so measuring something made everything look stale | The re-run itself; the census is now a data file |
| Annotation frames read at 30 fps on 24 fps clips | Labels pointed at the wrong frames on 138 of 200 clips | The blink alignment test in section 3.6 |
| Gloss classifier could only predict one gloss | Its "negative result" was the baseline by construction | A planted-signal positive control |
| Baseline model trained without the main model's class weights | A five-fold "improvement" that was an artefact | Reading the re-run instead of comparing headlines |
| Reproduction script skipped six of its own commands | The provenance check had never run from the pipeline | Reading what was executed, not the stage headers |

Each is now covered by a test.

---

## 5. The live demo

[`docs/index.html`](docs/index.html) is one static file. It loads MediaPipe Tasks (WASM, GPU
when available) and runs all three landmarkers on your webcam or on a video file you choose.
**No video or landmark leaves the device.**

**What the site is for.** It does four jobs, in the order the page presents them:

1. **It shows the vision system working.** Anyone with a browser and a camera can see the
   face mesh, both hands and the body tracked in real time, with the frame rate and each
   model's latency on screen. That is the part of the project a reader cannot take on trust
   from a table.
2. **It is an instrument for the grammar markers.** Raise your brows, furrow them, shake or
   nod your head, and the read-out for that marker moves, next to the number that says how
   far that read-out agreed with linguists' annotation. It is how the head-axis bug in
   section 4.2 was found.
3. **It is the report's front page.** The results, the figures, the method and the limits
   are on the same page as the demo, each claim beside its evidence, so it serves as the
   hosted link for the submission.
4. **It shows the output side.** The same page plays the 3D avatar built from a signing
   video.

What it is not: a sign-language translator or an emotion detector. It translates nothing,
and it deliberately shows no emotion label, because our emotion model did not beat chance.

| Panel | Shows |
|---|---|
| Stage | Face mesh, hand skeletons and upper body drawn over the video, with FPS and per-model latency |
| Linguistic channel | Brow raise, brow furrow, head shake, head nod, each with the validation result we measured. Shake and nod read in degrees of back-and-forth motion |
| 3D avatar | A section of the same page, reached from the nav bar. On the local server it is an interactive viewer: orbit, zoom, scrub, half speed, skeleton overlay, one button per utterance, with that clip's measurements beside it. On the hosted page it shows rendered frames and says why the viewer cannot be hosted |
| Affective cues | Smile, frown, eye widening, jaw opening as raw signals, with no emotion label |
| Head pose and prosody | Yaw, pitch, roll; hand speed and signing-space size in shoulder widths |
| Timeline | The last twelve seconds of four signals |
| Marker events | An entry each time a marker stays active, with its ASL meaning |

"Skeleton only" hides the camera image and leaves just the landmarks:

![The live page in skeleton-only mode: a wireframe face, hand and body on a black stage, with the affective channel reading a smile](docs/figures/demo_skeleton.png)

The page is checked in a real browser by `scripts/site_smoke.py`, which runs headless Chrome
with a video file standing in for the camera and reads back what the page measured.

```bash
make serve                                   # http://127.0.0.1:8000/
python scripts/site_smoke.py --video my_clip.mp4 --shot out.png
```

The page's head detector is JavaScript and the validation is Python, so a test runs the
page's own classes in Node and checks them number for number against the Python functions
that were scored; the thresholds in the page are read from the validation artifact by
another.

The 3D viewer is part of that page (run `make demo-avatar` once to build its clips). The
server also offers `/server` (the earlier demo, with server-side marker analysis) and
`/api/coverage` (what is and is not claimed).

---

## 6. Limitations and ethics

- **Four signers, 200 utterances.** Every result is leave-one-signer-out, and none should be
  read as general across signers, dialects or recording conditions.
- **Lab-recorded, scripted signing.** Nothing here is evaluated on conversation.
- **Latency was measured before the hand-tracking fix.** The 73 ms figure in section 4.1 was
  taken with two one-hand graphs running; the pipeline now runs one graph that finds two
  hands. We have not re-measured it, because the benchmark needs the CPU governor set by an
  administrator. We expect it to be no slower and do not claim so.
- **Face tracking is not perfectly repeatable.** Extracting the same 200 clips twice gave
  identical blendshapes on 87% of frames and differences above 0.02 on 4.5%, in 21 clips.
  Every result here was computed from one extraction, so they are consistent with each
  other, but a re-extraction moves the third decimal.
- **Emotion labels are annotators' perceptions**, not the signer's internal state, and
  agreement on some classes is low.
- **We are hearing researchers.** The grammar labels come from ASL linguists and the emotion
  labels from Deaf annotators; no Deaf signer has evaluated this system.
- **No clinical, legal or emergency use.**
- **Data stewardship.** ASLLRP and EmoSign video are used under research terms and never
  redistributed. The SMPL-X body model is licence-gated and not included.
- **Privacy.** The demo processes video on the device. The local server binds to localhost.

---

## 7. Future work

These parts of the original plan are **not done** and are marked as future work.

| Item | State | What it needs |
|---|---|---|
| Sign-to-English translation (pose → T5-small on How2Sign) | Data prepared: 35,176 sentences. No model code | About 2–3 days of GPU time. Published results near 10 BLEU rely on pretraining on about 1,000 hours of extra video; from How2Sign alone, low single digits are expected |
| Emotion recognition compared with published baselines | Never run on a comparable protocol | A decision on which published protocol to match |
| Emotion-conditioned text generation | Not started | A working emotion signal, which we do not have |
| Emotion-modulated avatar | Avatar works; modulation not built | Same dependency |
| Avatar preference study | Stimuli and blinding ready; zero raters | Five or more raters, ideally including a Deaf signer |
| Question marking in the confound audit | Too few clips in this corpus | Running perception over 1,354 further utterance videos already on disk |
| A validated head-movement instrument | Head shake is visible on the correct axis (0.70–0.76) but short of the 0.80 gate; head nod is weak | A threshold that adapts to each signer, and more than four signers to tune on |
| Cross-lingual test on Indian/Nepali sign data | Blocked | Establishing the terms of use of the local data |

---

## 8. Reproducing everything

```bash
make setup                 # editable install + pre-commit hooks
make lint typecheck test   # ruff, mypy, the whole test suite
make serve                 # live demo at http://127.0.0.1:8000/
make serve-check           # boots the server and asserts every page
make repro                 # regenerates every cited result, in dependency order
make provenance            # fails on an untraced number or a stale result
python scripts/make_report_figures.py   # redraws the figures in this README
```

Environment: Python 3.12, PyTorch 2.13, MediaPipe 0.10.14, an RTX 3050 (4 GB). The datasets
are not included; `make readiness` reports what is present and what each missing item blocks.
The SignStream annotations require a free account with the ASLLRP data portal.

### Running it on another computer, without training anything

For a teammate on Windows or Linux who only wants to run it:

```bash
python scripts/make_share_bundle.py     # writes dist/SEAM_share.zip, about 23 MB
```

They extract the zip and follow `START_HERE.md`:

| | Windows | Linux / macOS |
|---|---|---|
| Set up once (creates a `.venv`) | `setup_windows.bat` | `./setup_linux.sh` |
| Start the website | `run_demo.bat` | `./run_demo.sh` |
| Analyse their own video, offline | `python scripts\analyse_video.py clip.mp4` | `python scripts/analyse_video.py clip.mp4` |

The zip holds the code, the website, MediaPipe's model files, our three trained emotion
models with their ONNX exports, and every result file, so nothing has to be trained or
downloaded apart from the Python packages. It never contains dataset video, the SignStream
annotation files or the SMPL-X body model. The animated avatar clips contain that model's
mesh, so they go in only with `--with-avatar`, for someone who holds the SMPL-X licence too;
without them the avatar section shows rendered frames.

---

## 9. Repository map

| Path | What is there |
|---|---|
| `src/seam/perception/` | MediaPipe wrappers, landmark extraction, SMPLer-X adapter |
| `src/seam/preprocess/`, `src/seam/features/` | Normalisation, filtering, markers, prosody, pose features |
| `src/seam/data/` | Dataset loaders, the SignStream parser, frame alignment |
| `src/seam/eval/` | Emotion models, the confound audit, probes, leave-one-signer-out |
| `src/seam/affect/` | The factorized encoder and its loss terms |
| `src/seam/avatar/` | Skinning, glTF export, rendering |
| `src/seam/serve/`, `src/seam/web/` | The local server and its pages |
| `docs/` | The standalone live demo, figures, and the team guide |
| `scripts/` | One script per experiment, plus the reproduction and smoke tests |
| `tests/` | Unit, property, integration and provenance tests |
| `paper/` | Experiment log, claims ledger, paper scaffold |
| `plan.md` | Milestones, gates and the KPI dashboard |

### Acknowledgements

EmoSign (Chua et al., 2025); the American Sign Language Linguistic Research Project (Neidle
and Metaxas, Boston and Rutgers Universities, <http://www.bu.edu/asllrp/>,
<http://dai.cs.rutgers.edu/>); WLASL; RAF-DB; How2Sign; MediaPipe; SMPL-X and SMPLer-X.
