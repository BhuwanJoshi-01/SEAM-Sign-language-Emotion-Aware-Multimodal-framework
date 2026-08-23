# SEAM — Project Breakdown

**SEAM: Sign-language Emotion-Aware Multimodal framework**
*Factorized Non-Manual Modeling for Emotion-Aware American Sign Language Translation and Expressive Avatar Generation*

This document is the technical bible for the project. It explains what we are building, why,
what the literature already established, what data we use, how every component works, how we
evaluate it, and what can go wrong. `plan.md` is the executable checklist; `team.md` is the
ownership map; `paper/` is the write-up. This file is the source of truth for *understanding*.

Status: v1.0 — created 2026-08-07. Update the changelog at the bottom on every substantive edit.

---

## 1. Executive Summary

Sign language translation systems overwhelmingly model the **manual channel** (hands, arms) and
discard the **non-manual channel** (face, brows, mouth, gaze, head, torso). This is not a cosmetic
omission: in ASL the non-manual channel is *grammatically obligatory* — raised brows mark
yes/no questions, a furrowed brow marks wh-questions, a head-shake marks negation, mouth
morphemes modify predicates. The same channel *simultaneously* carries affect. Existing systems
therefore fail twice: they drop grammar, and they drop emotion.

SEAM addresses this with a **factorized non-manual encoder** that splits the non-manual signal
into two explicitly-supervised factors:

- `L` — the **linguistic factor** (question marking, negation, topic marking, mouthing)
- `A` — the **affective factor** (valence/arousal and discrete emotion)

plus a **manual prosody channel** `P` (signing speed, movement amplitude, repetition, pause
duration, jerk) that feeds `A`, because Deaf annotators consistently name these as emotion cues.
The factors are separated by training objectives (gradient reversal, feature orthogonality,
mutual-information minimization) and the separation is **measured**, not asserted.

The recovered affect then conditions two downstream generators: an **emotion-conditioned text
decoder** (so "I missed you" becomes "I'm so happy to see you again!" when the signer is joyful)
and an **expressive avatar controller** (so the avatar's blendshape intensity, motion amplitude
and timing carry the emotion).

Everything is engineered to run inference on a **4 GB RTX 3050** at interactive latency. Efficiency
is a headline result, not an afterthought: keypoint inputs, 12 fps sampling, sub-2M-parameter
recognition models, T5-small, ONNX Runtime with INT8.

**One-line contribution claim:** *a ~2M-parameter specialist model that separates grammatical from
affective non-manual signals, beats frontier multimodal LLMs on the only existing sign-emotion
benchmark, and runs end-to-end in under 2.5 GB of VRAM.*

---

## 2. Motivation

### 2.1 The communication gap
Sign language is the primary language of a large Deaf and hard-of-hearing population. Automatic
translation systems that produce flat, affect-free text create a second-order barrier: the message
survives, the person does not. The consequences are documented and concrete — misreading a
signer's emotional state has been reported as a source of bias in legal settings and emergency
departments.

### 2.2 Why non-manual signals cannot be ignored
Sign languages are composed of five parameters: handshape, place of articulation, movement,
orientation, and **non-manual behaviours**. The fifth is not decoration. Some published facts we
build directly on:

- Facial expressions serve **linguistic and affective functions simultaneously**; "puffed cheeks"
  marks intensity, "raised eyebrows" marks a yes/no question — neither is an emotion.
- Affective prosody in ASL is carried partly by the **hands**: when expressing anger, sentence
  duration shortens, movement paths shorten, and movement becomes more angular.
- **Hearing non-signers systematically misread linguistic facial markers as negative emotion.**
  In the EmoSign benchmark a hearing annotator perceived neutral clips as negative and positive
  clips as neutral.

That last point is the crux of the whole project. A model trained on ordinary facial-emotion data
is, in effect, a hearing non-signer: it will read a wh-question furrow as anger. Fixing that
requires modelling the two functions separately. That is SEAM's reason to exist.

### 2.3 Why now
Two things changed in 2025–2026 that make this project feasible and timely:
1. **A benchmark exists.** EmoSign (2025) is the first sign-video dataset with sentiment and
   emotion labels, and its baselines are weak enough that a small specialist model can win.
2. **Lightweight SLT is a solved recipe.** A 77M-parameter pose→T5-small pipeline was shown to
   land within ~1.8 BLEU-4 of a 3× larger model, and skeleton models under 1M parameters reach
   96% on isolated benchmarks. Low-VRAM deployment no longer requires a research breakthrough.

---

## 3. Research Objectives and Hypotheses

| ID | Objective | Testable hypothesis |
|---|---|---|
| O1 | Recognize isolated ASL signs from keypoints under a strict parameter budget | A ≤2M-param ST-GCN on 85 keypoints comes within 3 points of published pose baselines on ASL Citizen |
| O2 | Translate glosses / continuous pose sequences to fluent English at small scale | T5-small with a linear pose projection reaches BLEU-4 ≥ 8 on How2Sign |
| O3 | Recognize signer affect from non-manual + manual-prosody cues | A specialist model exceeds GPT-4o video-only wF1 (20.76) by ≥ 14 points on EmoSign under LOSO CV |
| O4 | **Separate grammatical from affective non-manual signals** | Adding GRL + orthogonality + MI-minimization drives cross-prediction AUC from >0.75 toward ≤0.60 while *improving* affect wF1 |
| O5 | Preserve emotion through generation | Emotion-conditioned decoding achieves ≥80% style accuracy at ≥0.90 BERTScore semantic preservation |
| O6 | Render emotion in an expressive avatar | Human raters prefer emotion-modulated avatar output over neutral ≥60% of the time |
| O7 | Deploy on 4 GB VRAM at interactive latency | End-to-end p95 < 400 ms, ≥20 FPS capture, peak VRAM < 2500 MB on RTX 3050 |

O4 is the scientific core. O1, O2, O7 are engineering enablers. O5, O6 are the application payoff.

---

## 4. Related Work — synthesis and positioning

### 4.1 Isolated sign language recognition (ISLR)
- **Skeleton/keypoint methods.** SAM-SLR (SL-GCN + multi-modal fusion) won the 2021 ChaLearn
  signer-independent challenge with 98.42% (RGB) / 98.53% (RGB-D) on AUTSL. ST-GCN is the
  structural ancestor: the body is naturally a graph.
- **Pose transformers.** SPOTER (Bohacek & Hruz, WACV 2022) established the pose-sequence
  transformer as the compact ISLR baseline; it is the model EmoSign's authors cite for word-level
  recognition from 2D pose.
- **Extremely small models.** SignBart reports 96.04% on LSA-64 with **749,888 parameters**,
  demonstrating that skeleton-sequence models do not need to be large.
- **RGB video upper bounds.** Self-supervised video transformers (MaskFeat) reach 79.02% top-1 on
  WLASL2000, outperforming pose-based and supervised video models. We cite this as the ceiling we
  deliberately do not deploy: it does not fit our latency/VRAM envelope.
- **Training strategy matters as much as architecture.** "Training Strategies for ISLR" (2024)
  improved WLASL SOTA by 1.63% and Slovo by 14.12% through augmentation/schedule choices alone.

**Position:** we do not claim ISLR novelty. We adopt the strongest *compact* recipes (ST-GCN,
SPOTER) as a solid, honest backbone and spend our novelty budget on the non-manual factorization.

### 4.2 Sign language translation (SLT)
- **Gloss-based.** Neural Sign Language Translation (Camgoz et al., CVPR 2018) with
  RWTH-PHOENIX-Weather 2014T defined the task.
- **Gloss-free.** GFSLT-VLP, GloFE, Sign2GPT and LLaVA-SLT remove gloss supervision by borrowing
  priors from pretrained language models. A 2026 "unbiased evaluation" paper warns that reported
  gains often come from data and pretraining rather than architecture — a caution we adopt in our
  own claims.
- **Compact SLT.** The key reference for us: MMPose 85 keypoints (255-dim/frame) → **one linear
  layer** → **T5-small**, 77M params total, BLEU-4 **10.06** at 24 fps vs **9.53** at 12 fps on
  How2Sign, against 11.89 for a ~248M T5-base. Halving fps cuts encoder self-attention cost ~75%
  because attention is O(n²).

**Position:** we replicate the compact recipe as our Phase-B backbone and extend it with emotion
conditioning — which nobody in this line of work does.

### 4.3 Emotion in sign language
This is a nearly empty field, which is exactly why we are here.
- **EmoSign (2025)** — first sign-emotion dataset. 200 ASL utterances from ASLLRP, ~16 min total,
  4 signers, annotated by 3 Deaf native signers with professional interpreting experience:
  7-point sentiment, 10 emotion intensities (joy, excited, surprise+, surprise−, worry, sadness,
  fear, disgust, frustration, anger), and free-text emotion-cue descriptions.
  Krippendorff α: sentiment 0.738, joy 0.699, worry 0.555, average 0.593 — negative emotions agree
  worse than positive ones (surprise− 0.119, disgust 0.166), which bounds achievable accuracy.
- **eJSL (2026)** — Japanese Sign Language, 2 signers × 78 utterances × 7 emotions = 1,092 clips.
  Findings we reuse: temporal segment selection matters a lot, and **adding hand motion improves
  emotion recognition in signers**.
- **German Sign Language + Facial Action Units** — supports an AU-based feature design.
- Older psycholinguistics: affective prosody in ASL (Reilly et al. 1992), emotion in Finnish SL
  hand-movement quality (Hietanen et al. 2004), facial expressions/emotions/sign languages
  (Elliott & Jacobs 2013).

**Published EmoSign baselines — our targets.** Transcribed from the EmoSign paper's Tables 3–4.
`wAF`/`wF1` denote weighted F1. **Every cell must be re-verified against the published PDF before
it enters our paper** — the Qwen2.5-VL sentiment-7 row in particular was ambiguous in transcription
and is marked `verify` in `paper/main.tex`.

| Model | Modality | Sentiment-3 wAcc | Sentiment-3 wF1 | Sentiment-7 wAcc | Sentiment-7 wF1 | Emotion wAcc | Emotion wF1 |
|---|---|---|---|---|---|---|---|
| MiniGPT4-video | video | 34.68 | 40.00 | 14.46 | 13.03 | 13.01 | 22.02 |
| Qwen2.5-VL-7B | video | 27.34 | 16.47 | 10.26 | *verify* | 14.39 | 18.53 |
| AffectGPT | video | — | — | — | — | 12.62 | 11.03 |
| GPT-4o | video | 52.13 | 76.72 | 22.89 | 26.35 | 11.50 | 20.76 |
| Hearing non-signer | video | 55.64 | 57.64 | 25.48 | 21.39 | — | — |
| MiniGPT4-video | video+caption | — | — | — | — | 23.56 | 35.89 |
| Qwen2.5-VL-7B | video+caption | — | — | — | — | 34.96 | 44.67 |
| AffectGPT | video+caption | — | — | — | — | 30.17 | 47.77 |
| GPT-4o | video+caption | — | — | — | — | 35.97 | 55.09 |

Qualitative failure modes reported (and which we must not reproduce): AffectGPT collapses to
"neutral"; GPT-4o and Qwen skew positive; models construct post-hoc explanations consistent with
the *text* sentiment rather than genuinely reading the video. All inference was run on an 80 GB
A100 — our entire system targets 4 GB, which sharpens the efficiency comparison.

**Position:** we are the first to (a) treat grammatical vs affective non-manual function as an
explicit factorization objective on this benchmark, and (b) propagate the recovered affect into
both text generation and avatar animation.

### 4.4 Sign language production and avatars
SLP typically goes text → gloss → pose → render. SignLLM tokenizes sign video into language-like
tokens; SignAvatar uses a transformer CVAE for word-level sign motion reconstruction/generation.
On the engineering side, MediaPipe `FaceLandmarker` emits **52 ARKit-compatible blendshape
coefficients** that map near-1:1 onto VRM/ARKit avatar expressions, and community work maps those
52 coefficients onto FLAME expression space.

**Position:** v1 uses deterministic retargeting (reliable, zero training) with emotion as a
*parametric modulation* of blendshape gain and motion prosody. Generative motion synthesis is
explicitly future work — it cannot be de-risked inside 10 weeks.

### 4.5 Techniques we borrow for the factorization
- **Gradient reversal / DANN** (Ganin & Lempitsky, 2015) — adversarially remove information about
  one factor from the other branch's features.
- **CLUB / vCLUB** (Cheng et al., ICML 2020) — a contrastive log-ratio *upper bound* on mutual
  information, minimizable as a loss; the right tool when you want to *reduce* MI between factors.
- **Feature orthogonality penalties** — cheap, stable, complementary to the above.
- **One Euro Filter** (Casiez et al., CHI 2012) — low-lag real-time smoothing; preferable to
  moving-average for interactive landmark streams.

---

## 5. Datasets

### 5.1 Primary datasets

| Dataset | Language | Content | Scale | Role in SEAM | Access |
|---|---|---|---|---|---|
| **ASL Citizen** | ASL | Isolated signs, crowdsourced, everyday environments | 83,399 videos / 2,731 signs / 52 signers | **Primary ISLR training set** | Microsoft Download Center, first-party, IRB-consented; no redistribution |
| **EmoSign** | ASL | Sentiment + 10 emotions + cue text on ASLLRP utterances | 200 utterances / 4 signers / ~16 min | **Primary affect benchmark** | HuggingFace `catfang/emosign` (labels + IDs only) |
| **ASLLRP** | ASL | Glosses, English text, **linguistic non-manual annotations** | 2,651 utterances / 19 signers | **Source video + `L` supervision** | Boston University data-access interface — request required |
| **How2Sign** | ASL | Continuous ASL, multiview, **keypoints published separately** | ~80 h / ~35k clips | **Phase B continuous SLT** | Public; download keypoints only, not video |
| **WLASL-100 / 300** | ASL | Isolated signs (YouTube-sourced) | subset of ~12k videos / 2k glosses | Literature comparability only | Public; expect link rot |
| **DFEW** | — | Dynamic FER in the wild (movies) | ~16,000 clips, 7 classes | Non-manual encoder pretraining | License request |
| **MAFW** | — | Multi-modal compound affective, in the wild | 10,045 clips | Non-manual encoder pretraining | License request |
| **INCLUDE** | ISL | Isolated Indian SL, 15 word categories | 4,287 videos / 263 signs / 0.27M frames | **Cross-lingual generalization ablation** | Already local (partially) |

### 5.2 Local data status — verified 2026-08-07
Path `/home/bhuwan/Videos/data/` contains **INCLUDE**, not WLASL. Category zips present:
Adjectives (8 parts), Animals (2), Clothes (2), Colours (2), Days_and_Time (3), Electronics (2),
Greetings (2), Home (4), Jobs (2), Means_of_Transportation (2), People (5), Places (4),
Pronouns (2), Seasons (1), Society (3), plus `Train_Test_Split`.

**12 archives are truncated `.part` downloads** and must be re-fetched before use:
`Electronics_1of2`, `Home_2of4`, `Home_3of4`, `Jobs_1of2`, `Colours_2of2`, `People_1of5`,
`People_2of5`, `People_5of5`, `Places_2of4`, `Greetings_2of2`, `Means_of_Transportation_2of2`,
and critically **`Train_Test_Split`** — without which the official INCLUDE split cannot be
reproduced. There is an official baseline repo (`AI4Bharat/INCLUDE`) with pretrained models and
an INCLUDE-50 subset for comparison.

### 5.3 Datasets deliberately rejected
- **RWTH-PHOENIX-Weather 2014T** — excellent for translation, useless for affect: weather-news
  signing is near affectively flat. EmoSign's authors rejected several corpora for this exact
  reason after VADER analysis showed captions clustered at neutral.
- **YouTube-ASL (984 h)** — signing/caption quality uncertain; EmoSign excluded it too. Too large
  for our compute and adds unverifiable noise.
- **Raw How2Sign video (80 h)** — unnecessary; keypoints are published separately.
- **CSL-Daily, ISLTranslate, iSign, CISLR** — out of scope once the ASL spine was chosen. iSign
  (ISL, ACL Findings 2024, 228 GB full release, pose-only archive separable, CC-BY-NC-SA-4.0,
  gated) remains the natural vehicle for a future ISL/NSL follow-up paper.

### 5.4 Storage strategy
Video is a transient artifact. For every dataset: download → extract keypoints once → persist
compressed `.npz` shards + a manifest → delete or archive the video. Keypoint sequences for
83k clips are on the order of a few GB; the source video is orders of magnitude larger. Disk
headroom on `/mnt/Volume2` must be confirmed in Phase 1 before any bulk download starts.

### 5.5 The EmoSign / ASLLRP coincidence — why it matters
EmoSign is built **on top of ASLLRP**, and ASLLRP independently publishes *linguistic non-manual
annotations* alongside glosses and English text. Therefore the same 200 utterances carry:

- `L` labels — grammatical non-manual markers (from ASLLRP)
- `A` labels — sentiment and 10 emotion intensities (from EmoSign)
- text — English translation and gloss (from ASLLRP)

No other corpus in existence gives both label types on the same frames. This is the entire
empirical foundation of objective O4, and it is why **ASLLRP access is week-1 critical path**.

---

## 6. System Architecture

### 6.1 Pipeline

```
Webcam / video (720p)
  → temporal sampling @ 12 fps                       [O(n²) attention cost ÷ 4]
  → MediaPipe Tasks HolisticLandmarker
        · 33 pose · 21 left hand · 21 right hand · 468 face   = 543 landmarks
        · 52 ARKit-compatible blendshape coefficients
        · head pose (rotation matrix / solvePnP) · gaze proxy
  → preprocessing: interpolate → normalize → One-Euro smooth → T=64 window
  → three feature channels
        M  manual        85-keypoint subset (hands + upper body), 255-dim/frame
        NM non-manual    52 blendshapes + AU proxies + head dynamics + gaze
        P  prosody       speed, amplitude, repetition, pause, jerk (derived from M)
  → M  → ST-GCN / SPOTER            → gloss sequence
  → M  → linear projection → T5-small encoder   (Phase B, gloss-free)
  → NM + P → factorized non-manual encoder
                 ├── head_L → linguistic markers (question / negation / topic / mouthing)
                 └── head_A → valence-arousal + 10 emotions + neutral
                 (GRL + orthogonality + vCLUB between the two branches)
  → gloss + L  → T5-small translator             → literal English
  → literal English + A → emotion-conditioned decoder → emotion-aware English
  → landmarks + A → avatar controller → VRM bone rotations + 52 blendshapes
                                        + emotion-modulated gain / amplitude / timing
  → React Three Fiber viewer, GLB / VRMA export
```

### 6.2 Why person detection was removed
The source concept placed YOLO11/RT-DETR before landmark extraction. MediaPipe already runs its
own person/pose detector internally, so a separate detection stage is redundant work on the
critical path — pure latency and VRAM cost for no accuracy gain on single-signer framing (which
is what ASL Citizen, WLASL, How2Sign and EmoSign all are). YOLO11n is retained **only** as an
optional multi-person / in-the-wild robustness ablation, off by default.

### 6.3 Feature specification

**Manual channel (M).** 85 keypoints: 21 + 21 hands, ~43 upper-body/arm/shoulder points, x/y/z
normalized by frame dimensions → 255-dim per frame. This mirrors the compact-SLT feature spec so
our numbers are comparable to theirs.

**Non-manual channel (NM).** The 52 blendshape coefficients are the backbone (they are already a
compact, semantically-named, FACS-adjacent basis — using them avoids learning a face encoder from
468 raw landmarks and slashes parameters). Augmented with: AU proxies derived from blendshapes,
head tilt/rotation/nod frequency, gaze shifts, brow-specific and mouth-specific groupings, and a
~30-point semantic face-landmark subset for geometry the blendshapes omit.

**Prosody channel (P).** Computed from M: per-frame signing speed, movement amplitude (bounding
volume of the signing space), repetition count (autocorrelation peaks), pause duration, jerk
(third derivative), and sign-duration statistics. Justification is empirical: EmoSign's Deaf
annotators explicitly named sign size, speed, repetition and emphatic finger-spelling as affective
cues, and eJSL found hand motion improves signer-emotion recognition.

### 6.4 The factorized non-manual encoder (core contribution)

Two branches over the NM+P stream (small temporal transformers or GRUs, ≤1M params each):

```
z_L = Enc_L(NM)        z_A = Enc_A(NM, P)

L_total =  λ1 · CE(head_L(z_L), y_marker)          linguistic supervision (ASLLRP)
         + λ2 · CE(head_A(z_A), y_emotion)         affective supervision (EmoSign)
         + λ3 · MSE(head_VA(z_A), y_valence_arousal)
         + λ4 · CE(head_A'(GRL(z_L)), y_emotion)   gradient reversal: z_L must NOT predict emotion
         + λ5 · CE(head_L'(GRL(z_A)), y_marker)    gradient reversal: z_A must NOT predict markers
         + λ6 · || z_L^T z_A ||_F^2                feature orthogonality
         + λ7 · vCLUB(z_L ; z_A)                   mutual-information upper-bound minimization
```

**Measuring separation.** Train frozen probes to predict emotion from `z_L` and markers from `z_A`.
Report cross-prediction AUC: 0.5 means perfect separation, high values mean entanglement. The
headline ablation is entangled single-branch vs factorized, on both cross-prediction AUC *and*
affect wF1 — the claim is that separation **improves** affect accuracy rather than trading against
it, because the model stops attributing grammar to emotion.

**Qualitative evidence.** Curate a set of wh-question and negation clips with neutral affect. The
entangled baseline should label them anger/frustration (the documented hearing-non-signer error);
the factorized model should not. This is the paper's most persuasive figure.

**Fallback if ASLLRP access is delayed.** Derive weak `L` labels from (i) rule-based blendshape
heuristics (brow-raise magnitude, brow-lowerer, head-shake frequency, mouth-shape clusters) and
(ii) syntactic analysis of the English caption (interrogative form, negation, topicalization).
This is a pseudo-label fallback and must be labelled as such in the paper — it weakens but does
not destroy O4.

### 6.5 Emotion-conditioned generation
No paired emotional-paraphrase corpus for ASL exists. We construct one: for each utterance's
neutral English reference, generate emotion-styled paraphrases with an LLM across the target
emotion set, filter for semantic equivalence, and disclose the procedure in the paper. The `A`
factor is then injected into T5-small as control tokens / a prefix embedding.

Three-axis evaluation, because BLEU alone is meaningless for style transfer:
1. **Style accuracy** — a held-out classifier recovers the intended emotion from the output text.
2. **Semantic preservation** — BERTScore-F1 against the neutral reference ≥ 0.90.
3. **Human preference** — pre-registered rubric, blinded pairwise comparison vs neutral output.

### 6.6 Avatar controller
- **Body:** pose landmarks → VRM humanoid bone rotations (three-vrm), with joint-limit clamping and
  quaternion continuity to prevent gimbal flips.
- **Face:** the 52 MediaPipe blendshapes map near-1:1 onto VRM/ARKit expression targets. No trained
  model required.
- **Emotion modulation:** `A` scales blendshape gain, motion amplitude, and playback timing —
  angry signing becomes faster/sharper/shorter-path, sad becomes slower/smaller, matching the
  documented affective prosody of ASL.
- **Output:** live WebSocket stream to a React Three Fiber viewer, plus GLB / VRMA clip export.

### 6.7 Deployment
FastAPI + WebSocket; frames in, glosses/text/emotion/avatar-control out. Bounded queues with
backpressure and graceful degradation when the GPU saturates. All models exported to ONNX; T5
dynamically quantized to INT8; ONNX Runtime CUDA EP, TensorRT optional.

**Security note (do not silently ignore):** the server binds to **localhost only, with no
authentication**, by default. It streams webcam-derived biometric data. Exposing it beyond
loopback requires a token/API key and TLS. This constraint is asserted in code and in tests.

### 6.8 Latency and VRAM budget

| Stage | Latency target | VRAM target |
|---|---|---|
| MediaPipe holistic | ≤ 25 ms/frame | ≤ 200 MB |
| ST-GCN recognition (T=64) | ≤ 5 ms | ≤ 200 MB |
| Factorized affect encoder | ≤ 3 ms | ≤ 100 MB |
| T5-small INT8, beam 2 | ≤ 120 ms/utterance | ≤ 600 MB |
| Avatar rendering | browser-side | 0 MB server |
| **End-to-end p95** | **< 400 ms** | **< 2500 MB peak** |
| Sustained capture | ≥ 20 FPS | — |

The 2500 MB ceiling is deliberate: a 4 GB laptop 3050 driving a desktop session leaves roughly
3.0–3.2 GB usable. CI asserts the ceiling from Phase 5 onward.

---

## 7. Evaluation Protocol

### 7.1 Non-negotiable methodology
1. **Leave-one-signer-out CV on EmoSign.** 200 clips, 4 signers → 4 folds. Any random split leaks
   signer identity and the result is worthless. Report mean ± std across folds, always.
2. **Signer-independent splits everywhere else too** (ASL Citizen ships signer metadata; use it).
3. **Report confidence intervals**, not point estimates, on anything derived from EmoSign.
4. **Every number traces to a logged run ID** (W&B) recorded in `paper/EXPERIMENT_LOG.md`.
5. **Latency and VRAM measured on the actual RTX 3050**, never estimated, never on the 4060.
6. **Inter-annotator ceiling acknowledged:** EmoSign's average Krippendorff α is 0.593
   (surprise− 0.119, disgust 0.166). Per-class accuracy on low-α classes is inherently bounded;
   claiming high accuracy there would be a red flag, not a win.

### 7.2 Metrics by stage

| Stage | Metrics |
|---|---|
| Isolated recognition | Top-1, Top-5, precision, recall, macro-F1, params, latency |
| Translation | BLEU-1..4 (SacreBLEU, default tokenization), ROUGE-L, METEOR, BERTScore |
| Affect | weighted accuracy, weighted F1, per-class accuracy, confusion matrix, MAE on valence/arousal |
| Disentanglement | cross-prediction AUC (target ≤0.60), probe accuracy, per-loss-term ablation |
| Conditioned generation | style accuracy, BERTScore-F1, human preference, degeneracy rate |
| Efficiency | FPS, p50/p95/p99 end-to-end latency, peak VRAM, params, FLOPs |
| Cross-lingual | Top-1/Top-5 on INCLUDE without ISL fine-tuning, then few-shot |

### 7.3 Baselines we must beat or honestly report against
- Affect: GPT-4o, Qwen2.5-VL-7B, AffectGPT, MiniGPT4-video (video-only and video+caption), plus
  the reported hearing-non-signer score — all from the EmoSign paper, so no re-running required.
- Recognition: ASL Citizen paper's own baselines; SPOTER-class pose results on WLASL-100.
- Translation: compact-SLT 77M (BLEU-4 10.06 @24fps / 9.53 @12fps) and T5-base 248M (11.89).
- Internal ablations: entangled vs factorized; with/without prosody channel; with/without DFEW
  pretraining; 24 vs 12 fps; FP32 vs INT8.

---

## 8. Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11 | ecosystem |
| DL framework | PyTorch 2.x | ecosystem, ONNX export |
| Landmarks | MediaPipe **Tasks** API (`HolisticLandmarker`, `FaceLandmarker`) | legacy Holistic is superseded; Tasks gives 52 ARKit blendshapes natively |
| Graph models | ST-GCN family (decoupled spatial-temporal) | body is a graph; ≤2M params |
| Sequence models | GRU baseline, SPOTER-style pose transformer | compact, published ISLR baselines |
| Translation | T5-small (`google/t5-v1_1-small`) | 77M-param recipe is literature-validated |
| FER pretraining | DFEW, MAFW (fallback RAF-DB/AffectNet, OpenFace AUs) | dynamic in-the-wild affect |
| Smoothing | One Euro Filter | low lag for interactive use |
| Tracking | Weights & Biases | sweeps + run IDs for the experiment log |
| Serving | FastAPI + WebSocket | streaming, async |
| Export/runtime | ONNX, ONNX Runtime (CUDA EP), INT8 dynamic quant, TensorRT optional | the 4 GB target |
| Frontend | React + Three.js / React Three Fiber + three-vrm | VRM/ARKit blendshape compatibility |
| Avatar assets | VRM / Ready Player Me / Mixamo, Blender for rigging | 52-blendshape ARKit convention |
| Metrics | SacreBLEU, BERTScore, scikit-learn | standard, comparable |
| Quality | pytest, ruff, pre-commit, mypy | test-first discipline |
| Paper | LaTeX (IEEEtran), BibTeX | workshop/conference target |

**Dropped from the original concept, with reasons:** YOLO11/RT-DETR/YOLO-NAS as a mandatory stage
(redundant with MediaPipe); Video Swin Transformer for deployment (violates the VRAM/latency
budget — retained only as a cited upper bound); mT5/BART/MarianMT (T5-small is the validated
compact choice; more variants is breadth without depth); PHOENIX14T (affectively flat).

---

## 9. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| RK1 | ASLLRP access delayed/denied → no `L` labels | Medium | **Critical** (kills O4) | Apply day 1; heuristic + caption-syntax pseudo-label fallback, disclosed | R1 |
| RK2 | 200 EmoSign clips, 4 signers → fragile results | High | High | LOSO CV, mean±std, CIs, no single-split claims | R3 |
| RK3 | DFEW/MAFW license lag | Medium | Medium | RAF-DB/AffectNet + OpenFace AU fallback | R1 |
| RK4 | ASL Citizen download size / disk exhaustion | Medium | High | Verify free space first; keypoints-then-delete strategy | R1 |
| RK5 | 4 GB VRAM ceiling breached late in the project | Medium | High | CI assertion from Phase 5; budget tracked from Phase 2 | R2 |
| RK6 | WLASL YouTube link rot | High | Low | ASL Citizen is primary; WLASL is comparability-only |R1 |
| RK7 | 10-week overrun | Medium | High | Declared cut-lines: Phase-B SLT and the INCLUDE ablation | R4 |
| RK8 | Overclaiming affect accuracy | Medium | **Critical** (rejection) | Claims ledger; α-bounded per-class reporting | R3 |
| RK9 | Emotion-conditioned generation degenerates / paraphrase corpus is low quality | Medium | Medium | Semantic-preservation floor 0.90 BERTScore; human spot-check; disclose LLM generation | R3 |
| RK10 | Avatar retargeting instability (gimbal flips, finger distortion) — a documented pain point in community reports | High | Medium | Quaternion continuity, joint clamping, unit tests on known poses | R4 |
| RK11 | Ethical/consent misstep with signer biometric data | Low | **Critical** | Respect ASLLRP/ASL Citizen no-redistribution terms; release IDs+labels only; localhost-only server; Deaf-community statement in the paper | R4 |

---

## 10. Ethics and Community

This project processes biometric data of a linguistic minority. Non-optional commitments:

- **Respect dataset terms.** ASL Citizen prohibits redistribution of data or modifications;
  ASLLRP videos are access-controlled; EmoSign itself releases only video IDs and labels for this
  reason. We follow the same norm: release code, weights, and labels — never redistributed video.
- **No unauthenticated network exposure** of a service that streams webcam data.
- **Deaf-community involvement and limitations statement** in the paper. We are hearing
  researchers building on annotations produced by Deaf native signers; we say so, we cite the
  documented finding that hearing non-signers misread linguistic markers as negative affect, and
  we frame that as motivation rather than as a solved problem.
- **No claims of clinical, legal, or emergency-setting readiness.** The cited harms (bias in legal
  settings, emergency departments) are motivation for careful research, not a deployment pitch.
- **Model cards** for every released checkpoint, documenting training data, signer demographics,
  known failure modes, and the α-bounded reliability of low-agreement emotion classes.

---

## 11. Repository Layout

```
Sign_Language_EmotionAware/
├── plan.md                     # executable phase/task checklist + KPIs (agent-facing)
├── project_breakdown.md        # this file — the technical bible
├── team.md                     # squad, sprints, RACI, gates
├── signemotionaware.md         # original concept note (kept for provenance)
├── pyproject.toml              # pinned deps
├── Makefile                    # setup / lint / test / train / bench / repro
├── configs/                    # hydra-style yaml: data, model, train, deploy
├── src/seam/
│   ├── data/                   # acquire, verify, manifests, splits, datasets
│   ├── perception/             # mediapipe wrappers, blendshapes, head pose
│   ├── preprocess/             # normalize, interpolate, one-euro, window, augment
│   ├── features/               # manual, non-manual, prosody, AU proxies
│   ├── models/                 # gru, stgcn, spoter, projections
│   ├── affect/                 # factorized encoder, GRL, vCLUB, orthogonality, probes
│   ├── translate/              # t5 gloss2text, pose2text, emotion conditioning
│   ├── avatar/                 # retargeting, blendshape mapping, modulation, export
│   ├── serve/                  # fastapi, websocket, backpressure, auth guard
│   ├── export/                 # onnx, quantization, tensorrt, vram guard
│   └── eval/                   # metrics, loso, benchmarks, tables, figures
├── frontend/                   # react + r3f + three-vrm viewer
├── tests/                      # unit, property, integration, benchmark, budget
├── scripts/                    # repro_all.sh, download_*.sh, bench.sh
├── notebooks/                  # exploration only, never the source of a paper number
├── artifacts/                  # npz shards, checkpoints, onnx (gitignored)
└── paper/
    ├── main.tex                # IEEEtran skeleton with all result tables stubbed
    ├── refs.bib                # seeded with the literature in section 4
    ├── PAPER_TEMPLATE.md       # section-by-section template + word budgets
    ├── WRITING_GUIDE.md        # house style, rules for honest claims
    ├── CLAIMS_LEDGER.md        # claim → experiment → run ID → status
    ├── EXPERIMENT_LOG.md       # append-only log of every run that touches the paper
    └── figures/
```

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Gloss** | A written label for a sign (e.g. `THANK-YOU`); an intermediate representation, not a translation |
| **Manual features** | Handshape, orientation, location, movement — the hands and arms |
| **Non-manual features** | Face, brows, mouth, gaze, head, torso |
| **Non-manual marker (NMM)** | A non-manual signal with *grammatical* function (brow raise = yes/no question) |
| **Affective prosody** | Emotion carried by *how* signs are produced: speed, amplitude, path shape, repetition |
| **ISLR** | Isolated Sign Language Recognition — one sign per clip |
| **CSLR / SLT** | Continuous recognition / translation — full utterances |
| **Gloss-free SLT** | Video/pose → text without gloss supervision |
| **Blendshape** | A named facial deformation coefficient; MediaPipe emits 52 ARKit-compatible ones |
| **AU** | Action Unit — a FACS facial muscle-movement code |
| **LOSO** | Leave-One-Signer-Out cross-validation |
| **GRL** | Gradient Reversal Layer — adversarially removes information from a representation |
| **vCLUB** | Variational Contrastive Log-ratio Upper Bound on mutual information |
| **VRM** | An open 3D humanoid avatar format with standardized humanoid bones and expressions |
| **Krippendorff α** | Inter-annotator agreement; EmoSign averages 0.593 |

---

## 13. Changelog

| Date | Version | Change |
|---|---|---|
| 2026-08-07 | 1.0 | Initial breakdown. Dataset audit corrected local data from "WLASL" to INCLUDE (12 truncated archives found). ASL spine selected; EmoSign adopted as affect benchmark; factorized non-manual encoder adopted as core contribution; 4 GB inference budget specified. |
