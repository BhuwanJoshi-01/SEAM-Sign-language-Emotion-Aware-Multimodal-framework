# SEAM — Team Reading Guide

For everyone on the team. Read it once in order; before the viva, reread sections 3, 6 and 7.
The submission report is [`README.md`](../README.md). This guide is the plain-language
version, with the story behind the numbers and the viva preparation.

---

## 1. What SEAM is

SEAM reads the face and head of a person signing in American Sign Language, in real time,
and asks one question: **is this facial movement grammar or emotion?**

In ASL the face is part of the grammar. Raised brows mark a yes/no question, furrowed brows
mark a wh-question, and a head shake marks negation. The same face also shows emotion.
Ordinary emotion-recognition models are trained on hearing non-signers, so the worry is that
they read grammar as feeling, for example a question as anger.

SEAM stands for Sign-language Emotion-Aware Multimodal framework. We built a computer-vision
pipeline around MediaPipe, measured each part against human annotation, and reported what
held and what did not.

**The one-sentence version for the viva:** we built a real-time pipeline that fits a 4 GB
GPU and extracts 52 facial blendshapes, head pose, hands and body from sign video, and we
tested with human-annotated data whether grammar and emotion in the face can be told apart.
Most of our original hypotheses were refuted, and we can show exactly why with numbers.

---

## 2. How the computer-vision system works

1. **Perception.** Three MediaPipe Tasks models run on every frame: Face Landmarker (478
   points, 52 blendshape scores, a head transform), two Hand Landmarkers (21 points each) and
   Pose Landmarker (33 points). That is 553 landmarks per frame.
2. **Speed.** The three models run in parallel threads: 41.8 ms per frame instead of 73.0 ms,
   a 1.75× speed-up, about 20 frames per second on an RTX 3050.
3. **Preprocessing.** Fill short gaps, centre on the shoulders, divide by shoulder width,
   smooth with a One-Euro filter, resample to 12 frames per second.
4. **Three channels.** Manual (hands), non-manual (face and head), prosody (how fast and how
   large the signing is).
5. **Marker read-out.** Brow raise is the mean of three brow blendshapes above the signer's
   own baseline. Brow furrow uses two. Head shake and nod are oscillations of head angles.
6. **Validation.** Linguists annotated the same 200 clips frame by frame. We score every
   marker against them on signers the detector has never seen.
7. **Two experiments on top.** An audit of whether emotion models misread grammar, and a
   two-branch encoder that tries to separate grammar from emotion.
8. **Avatar.** A separate model (SMPLer-X) turns the video into a 3D body, exported as an
   animated glTF file.

**Blendshapes** are the key idea. MediaPipe does not only give face points; it gives 52 named
scores between 0 and 1 such as `browInnerUp`, `mouthSmileLeft` and `eyeBlinkRight`. They are
compact, they have meanings, and they do not depend on who the person is. Everything about
the face in this project is built on them.

---

## 3. What we measured

| Question | Number | What it means |
|---|---|---|
| Is it real-time on a 4 GB GPU? | 73.2 ms p95, 19.7–20.4 FPS, 186 MB VRAM | Yes. The limit is the CPU, not the GPU |
| Can we read brow raise? | AUC 0.838 / 0.822 / 0.878 on three unseen signers | Yes. Our one validated instrument |
| Can we read brow furrow? | AUC 0.60–0.78 | Not reliably |
| Can we read head shake and nod? | AUC 0.50 | No, even with a trained detector |
| Do emotion models read grammar as negative emotion? | 0 of 8 markers met the rule | Not supported, on isolated or continuous signing |
| Does the encoder separate grammar from emotion? | 0.694–0.726; target ≤ 0.60 | No. Refuted in every seed |
| Does the encoder recognise emotion? | Balanced accuracy 0.48–0.51 | No better than chance |
| Does pose recognise glosses? | WER 0.920 = baseline 0.920 | Not with 3 examples per gloss |
| Does the avatar export work? | 55 joints, 6.4 mm error | Yes, verified in a real 3D viewer |
| How much human ground truth? | 43,038 annotated events, 200 of 200 clips joined | Enough to validate against |
| How well tested? | Over 500 automated tests | Including tests that catch stale results |

**AUC** is the probability that the detector scores a "marker present" frame higher than a
"marker absent" frame. 0.5 is a coin flip, 1.0 is perfect. Our gate was 0.80.

---

## 4. What happened

Newest first.

| Date | What happened | In one line |
|---|---|---|
| 5 Oct | Final review | Re-ran every result on corrected code; verdicts held, several finer claims withdrawn |
| 5 Oct | Marker validation | Brow raise validated against human labels; head shake and nod unreadable |
| 5 Oct | Confound audit on continuous signing | The test we said was needed: not supported |
| 5 Oct | Frame-rate bug | Annotations are on a 30 fps timeline, 138 clips are 24 fps; found with a blink test |
| 5 Oct | Gloss model bug | The model could only predict one gloss; its "result" was the baseline by construction |
| 4 Oct | Yaw bug | Head yaw was read from the wrong matrix entries; our one positive result was the bug |
| 4 Oct | Avatar working | SMPLer-X replaced landmark guessing; animated glTF verified in three.js |
| 1 Oct | Human annotations arrived | 51 SignStream files from Boston University: 43,038 events |
| 1 Oct | Encoder decision | Tried human labels to rescue the encoder; it did not help; reported as refuted |
| 29 Sep | Gloss recognition | First run: no better than baseline |
| 28 Sep | Efficiency measured | 20 FPS and 186 MB; found the benchmark was measuring the CPU power mode |
| 27 Sep | First confound audit | On isolated signs: no effect. First version invalid because of a preprocessing mismatch |
| 26 Sep | Data foundation | 200 EmoSign clips joined to ASLLRP by utterance ID; face visible in all 200 |
| 7 Aug | Project start | Plan, literature, datasets chosen |

### The pattern worth remembering

Almost every bug had the same shape: **a check that could not fail.**

- A test asserted the wrong thing and passed.
- A verifier shared the assumptions of the code it verified.
- A "validation" counted whether frames were in range, not whether they were the right frames.
- A model that predicts one class always equals its baseline.

Each was found the same way: by building a check that *could* fail, such as comparing blink
annotations with the blink signal, or planting a known effect and seeing whether the method
recovers it. If an examiner asks what you learned, this is the answer.

---

## 5. The live demo

**Hosted:** the GitHub Pages link in the README (works once Pages is enabled on the repo).

**Locally:**

```bash
make serve        # then open http://127.0.0.1:8000/live
```

Or simply open `docs/index.html` through any local web server. Click **Start camera**. The
first load downloads about 15 MB of models.

### A 60-second demo to perform

1. Sit still. Point at the HUD: frames per second and per-model milliseconds.
2. **Raise your eyebrows** and hold. The brow-raise bar fills, the brows glow white, the
   banner says "LINGUISTIC · brow raise → question / topic marker", and an event is logged.
   Say: "this marker is validated at AUC 0.82 to 0.88 against human annotation."
3. **Furrow your brows.** Say: "this one is labelled not validated, because it scored 0.60
   on our largest signer."
4. **Smile.** The affective channel lights up instead. Say: "same face, different channel."
5. **Move your hands.** Hand speed and signing-space bars respond.
6. Tick **Skeleton only**. Say: "the video never leaves the device; everything is computed
   from these landmarks."

If the camera is refused, use **Use a video file**.

**Do not** claim the demo recognises emotion or translates signs. It says so itself on the
page, and an examiner will respect that more than an overclaim.

---

## 6. Viva: the 5-minute presentation

The brief says: focus on the computer-vision system, and present facts and numbers.

**0:00 – 0:40 · The problem.** "In ASL the face is grammar. Raised brows mean a yes/no
question. Emotion models are trained on non-signers, so they may read grammar as emotion. We
built a vision pipeline to measure that."

**0:40 – 1:50 · The system.** "Three MediaPipe Tasks models per frame: 478 face points, 52
blendshapes, a head transform, two hands, and the body: 553 landmarks. We run them
concurrently: 41.8 ms instead of 73, a 1.75× speed-up. On a 4 GB RTX 3050 that is 73 ms at
the 95th percentile, 20 frames per second, and 186 MB of VRAM. Then shoulder-centred
normalisation, a One-Euro filter, and three channels: manual, non-manual, prosody."

**1:50 – 2:50 · Why we trust the measurements.** "We obtained 43,038 frame-level annotations
by ASL linguists for the same 200 clips. We validated every marker against them,
leave-one-signer-out. Brow raise scores 0.84, 0.82 and 0.88 on three unseen signers. Brow
furrow does not pass. Head shake and nod sit at 0.50, even with a trained detector."
*(Show the marker validation figure.)*

**2:50 – 3:50 · The two hypotheses.** "First: do emotion models read grammar as negative
emotion? We tested three models on 200 utterances with a rule fixed in advance. No marker met
it; brow furrow leaned the other way. Second: can an encoder with gradient reversal separate
grammar from emotion? Cross-prediction AUC was 0.71 against a target of 0.60, in every seed,
while our positive control passed at 0.98. Both hypotheses are refuted."
*(Show the forest plot and the encoder plot.)*

**3:50 – 4:30 · What we learned.** "We found six bugs in our own instruments, including
annotations read at the wrong frame rate. We caught it by checking blink annotations against
the blink signal: AUC went from 0.55 to 0.71 after the fix. Results now carry a fingerprint
of the code that produced them, and the build fails if that code changes."

**4:30 – 5:00 · Demo and close.** Raise your eyebrows on the live page. "One validated
marker, a real-time pipeline, and two honest negative results. Translation and emotion-styled
output are future work."

---

## 7. Viva: likely questions

**Why MediaPipe and not YOLO or OpenPose?**
We need the face in detail. Only MediaPipe's Face Landmarker gives 52 blendshape scores and a
head transform. A person detector is redundant because MediaPipe detects internally, and it
would add latency.

**Why keypoints and not raw video?**
Keypoints are a few hundred numbers per frame, so the models are tiny and fit 4 GB. They also
remove identity, which matters for privacy and for generalising across signers.

**What is a blendshape?**
A named facial-movement score between 0 and 1, such as `browInnerUp`. MediaPipe predicts 52 of
them from the face mesh. Brow raise is the mean of three.

**What is the One-Euro filter and why use it?**
A low-pass filter whose cutoff rises with speed. It smooths jitter when the hand is still and
does not lag when it moves fast. A moving average would blur fast signing.

**What is leave-one-signer-out?**
Train on three signers, test on the fourth, and rotate. A random split would let the model
recognise the person instead of the sign.

**What is a gradient-reversal layer?**
A layer that passes values forward unchanged and flips the gradient's sign backward. A head
on top learns to predict emotion, and the encoder below learns to make that impossible.

**What is cross-prediction AUC?**
After training, we freeze the encoder and train a probe to predict emotion from the grammar
code. 0.5 means it cannot, so the codes are separated. We got 0.71.

**Why did the encoder fail?**
Two reasons we can support. Four signers and 200 clips is very little data. And the emotion
head never learned (balanced accuracy 0.5), so there was no emotion code to separate. We
ruled out label noise by swapping in human labels: no improvement.

**Your main hypotheses failed. Is the project a failure?**
No. A refuted hypothesis measured with a validated instrument is a result. Our positive
controls pass, so the nulls are informative. We also produced one validated detector, a
real-time pipeline, and a reusable finding about heuristic label quality.

**How do you know your null result is not just a weak method?**
Three ways. A planted effect of 0.05 is recovered by the same code. Every null is reported
with the smallest effect it could have detected. And a placebo, the same marker track shifted
in time, behaves as it should.

**Why does head shake not work?**
MediaPipe's head transform on 256-pixel face crops does not carry half-second head movements
well enough. Even a trained detector on head angles and their dynamics stays at chance. The
events exist: annotators mark a head shake in 89 of 200 clips.

**What was the frame-rate bug?**
Annotation frame numbers are on a 30 fps timeline, and 138 of our 200 clips are 24 fps. Our
mapping ran 25% fast. The old check only counted whether frames landed inside the clip, which
passed. The blink test showed the truth.

**How is it real-time? What limits it?**
The CPU. MediaPipe runs on the CPU delegate, so the GPU is nearly idle at 186 MB. In
power-save mode the same benchmark was 118 ms instead of 42 ms.

**Does the demo detect emotion?**
No, deliberately. It shows raw expression signals. Our emotion model did not beat chance, so
showing a label would be showing noise.

**What would you do with more time?**
Run perception over the 1,354 further annotated videos we have, to test question marking;
train translation on How2Sign; get a head-pose estimator that works; and run the avatar
study with Deaf raters.

**Is the data ethical to use?**
The datasets are research corpora used under their terms. We never redistribute signer video.
The demo runs on the device. We state that no Deaf signer has evaluated the system.

---

## 8. Future work

| Item | Why it is not done |
|---|---|
| Sign-to-English translation | Needs 2–3 days of GPU training; published scores also rely on about 1,000 hours of extra pretraining data |
| Emotion recognition against published baselines | Never run on a comparable protocol |
| Emotion-styled text and emotion-modulated avatar | Both need a working emotion signal |
| Avatar preference study | Needs five or more human raters |
| Question marking in the confound audit | Only 8 and 4 clips in this corpus |
| Head-movement detection | Needs a better head-pose source |
| Cross-lingual test | Data terms not established |

Say "future work" plainly. Do not present any of these as done.

---

## 9. Glossary and where things live

| Term | Meaning |
|---|---|
| ASL | American Sign Language, a full natural language |
| Non-manual marker | A grammatical signal made with the face or head |
| Gloss | A written label for a sign, such as `BOOK` |
| Blendshape | One of 52 named face-movement scores from MediaPipe |
| LOSO | Leave-one-signer-out cross-validation |
| AUC | Probability a positive example outranks a negative one; 0.5 is chance |
| GRL | Gradient-reversal layer |
| MDE | Minimum detectable effect: the smallest effect a test could have found |
| FER | Facial emotion recognition |
| SMPL-X | A parametric 3D human body model |
| glTF | A standard 3D file format |
| WER | Word error rate |

| You want | Look in |
|---|---|
| The report | `README.md` |
| The live page | `docs/index.html` |
| The figures | `docs/figures/`, drawn by `scripts/make_report_figures.py` |
| MediaPipe code | `src/seam/perception/tasks_api.py` |
| Marker definitions | `src/seam/features/markers.py` |
| The encoder | `src/seam/affect/encoder.py` |
| The confound audit | `scripts/run_confound_audit_continuous.py`, `src/seam/eval/fer_audit.py` |
| Marker validation | `scripts/validate_markers.py` |
| Every experiment, dated | `paper/EXPERIMENT_LOG.md` |
| The plan and KPIs | `plan.md` |
