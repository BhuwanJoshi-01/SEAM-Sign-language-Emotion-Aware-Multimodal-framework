# Research Paper Template — SEAM

Use this as the writing scaffold. `main.tex` is the LaTeX skeleton; this file explains **what goes
in each section, in what order, at what length, and with which evidence**. Fill sections as the
phases complete — the paper is written continuously, not in week 10.

- **Target:** arXiv preprint + workshop paper (CVPR/ICCV MSLR, LREC SignLang). 8 pages + refs.
- **Journal extension** afterwards (IEEE Access / Scientific Reports): +continuous SLT, +ISL
  cross-lingual results, +larger human study.
- **Lead author / coordinator:** R3. Section owners listed per section.

---

## Working title

> **Grammar or Feeling? Factorized Non-Manual Modeling for Emotion-Aware Sign Language
> Translation on Commodity Hardware**

Alternatives:
- *Separating Grammar from Affect: Factorized Non-Manual Representations for Emotion-Aware ASL Translation*
- *SEAM: Emotion-Aware Sign Language Translation and Expressive Avatar Generation in 4 GB*

Pick the one that matches the venue's taste. Workshop reviewers reward a clear, specific claim over
a grand one.

---

## Structure and budgets

| § | Section | Words | Owner | Written at |
|---|---|---|---|---|
| — | Abstract | 180–220 | R3 | P4 gate, revised P5 |
| 1 | Introduction | 700–850 | R3 | P3 gate |
| 2 | Related Work | 600–750 | R3 | P2 gate |
| 3 | Datasets | 450–550 | R1 | P1 gate |
| 4 | Method | 1100–1300 | R3 | P4 gate |
| 5 | Experimental Setup | 400–500 | R3 | P3 gate |
| 6 | Results | 1100–1300 | R2, R3 | P4–P5 |
| 7 | System & Demo | 350–450 | R4 | P5 |
| 8 | Limitations & Ethics | 350–450 | R4 | P5 |
| 9 | Conclusion | 150–200 | R3 | P5 |
| — | **Total body** | **~5500–6500** | | |

---

## Abstract — the 5-sentence recipe

1. **Gap.** Sign language systems model the hands and discard the face — yet in ASL the face is
   grammatically obligatory *and* carries affect, and the two functions are confounded.
2. **Insight.** Grammatical and affective non-manual signals can be explicitly factorized, and the
   manual channel carries affective prosody (speed, amplitude, repetition) that belongs to the
   affect factor, not the grammar factor.
3. **Method.** A two-branch non-manual encoder with gradient reversal, orthogonality, and
   mutual-information minimization; the recovered affect conditions both text generation and an
   expressive 3D avatar.
4. **Result.** On EmoSign, a ~XM-parameter model reaches wF1 XX.X versus 20.76 for GPT-4o from
   video, under leave-one-signer-out CV; cross-prediction AUC falls from X.XX to X.XX.
5. **Efficiency.** The full pipeline runs end-to-end in <XXXX MB of VRAM at XXX ms p95 on an
   RTX 3050 — versus the 80 GB A100 used to evaluate the MLLM baselines.

Numbers stay as `X` until they come from a logged run. Never write a placeholder that looks real.

---

## 1. Introduction

Five paragraphs, in this order:

1. **The stakes.** Sign language is a full language; flat, affect-free translation strips the person
   from the message. Cite the documented consequences (bias in legal settings, emergency
   departments) — as motivation for careful research, *not* as a deployment pitch.
2. **The technical gap.** Non-manual signals are grammatically obligatory (brow raise = yes/no
   question, wh-furrow, negation head-shake, mouth morphemes) and simultaneously affective. Cite
   the linguistics (Valli & Lucas; Pfau et al.; Elliott & Jacobs) and the psycholinguistics of
   affective prosody in ASL (Reilly et al. 1992; Hietanen et al. 2004).
3. **The killer fact.** Hearing non-signers systematically misread linguistic markers as negative
   emotion — and so do frontier models. In EmoSign, GPT-4o reaches wF1 20.76 on emotion from video,
   AffectGPT collapses to "neutral" (11.03), and a hearing annotator scores wAF 21.39 on 7-class
   sentiment. Models are behaving like hearing non-signers. **This is the paper's hook.**
4. **Our approach in one paragraph.** Factorize; supervise both factors from the ASLLRP/EmoSign
   label overlap; measure separation via cross-prediction; propagate affect into text and avatar;
   do it all inside 4 GB.
5. **Contributions**, as a tight numbered list (4 items max):
   - C1: A factorized non-manual encoder that separates grammatical from affective function, with
     an explicit disentanglement metric rather than an assertion.
   - C2: A prosody channel showing manual dynamics carry affect, quantified by ablation.
   - C3: State-of-the-art on EmoSign under signer-independent LOSO, beating frontier MLLMs by
     ~Nx fewer parameters and ~Nx less memory.
   - C4: An end-to-end emotion-aware pipeline — recognition, translation, conditioned generation,
     expressive avatar — deployable on a 4 GB consumer GPU, with a full accuracy/latency/VRAM
     Pareto analysis.

---

## 2. Related Work

Four short subsections. Each must end with a sentence positioning *us*, not just summarizing them.

- **2.1 Isolated sign language recognition.** ST-GCN lineage; SAM-SLR/SL-GCN (98.42% AUTSL);
  SPOTER pose transformer; SignBart (749K params, 96.04% LSA-64); RGB upper bounds (MaskFeat
  79.02% WLASL2000). *Position:* we adopt compact recipes and claim no ISLR novelty.
- **2.2 Sign language translation.** Camgoz et al. / PHOENIX14T; gloss-free line
  (GFSLT-VLP, GloFE, Sign2GPT, LLaVA-SLT); compact SLT (77M, T5-small, BLEU-4 10.06 @24fps vs
  9.53 @12fps vs 11.89 for 248M T5-base); the 2026 unbiased-evaluation caution that gains often
  come from data, not architecture. *Position:* we replicate the compact recipe and add emotion
  conditioning, which this line does not do.
- **2.3 Emotion in sign language.** EmoSign (200 ASL utterances, 4 signers, 3 Deaf annotators,
  ᾱ=0.593); eJSL (JSL, 1,092 clips; hand motion helps); DGS + facial action units. Note that all
  of these are 2025–2026 — the field is nascent. *Position:* nobody treats the
  grammatical/affective confound as an explicit training objective; we do.
- **2.4 Sign language production and avatars.** SignLLM; SignAvatar (CVAE); MediaPipe's 52
  ARKit-compatible blendshapes; VRM. *Position:* we use deterministic retargeting with
  emotion-parametric modulation; generative motion is future work, stated plainly.

---

## 3. Datasets

A table first (name, language, scale, signers, labels, role, licence), then prose on:
- **ASL Citizen** (83,399 videos / 2,731 signs / 52 signers) as the ISLR training set — chosen over
  WLASL for first-party hosting and no link rot; WLASL-100/300 retained only for comparability.
- **EmoSign** (200 utterances, ~16 min, 4 signers) — with the **α table reproduced or cited**
  (sentiment 0.738, joy 0.699, worry 0.555; surprise− 0.119, disgust 0.166; mean 0.593). State
  outright that low-α classes bound achievable accuracy. This paragraph is what makes reviewers
  trust the rest of the paper.
- **ASLLRP** as the source of both the video and the **linguistic non-manual annotations** — and
  the key methodological point: EmoSign is built on ASLLRP, so the *same utterances* carry both
  grammatical-marker and affect labels. No other corpus offers this.
- **How2Sign** keypoints for continuous translation; **DFEW/MAFW** for non-manual pretraining.
- **Rejected datasets, with reasons** — PHOENIX14T (affectively flat weather news; EmoSign's
  authors rejected corpora on exactly this basis after VADER analysis), YouTube-ASL (unverified
  quality). Saying what you rejected and why is a credibility signal.
- **Splits.** Signer-independent everywhere; LOSO 4-fold on EmoSign. State it here and repeat it
  in §5. Reviewers look for this.

---

## 4. Method

- **4.1 Overview** + the pipeline figure (Fig. 1).
- **4.2 Perception and preprocessing.** MediaPipe Tasks; 543 landmarks + 52 blendshapes + head
  pose; One Euro Filter; 12 fps; T=64. Justify 12 fps with the O(n²) attention argument and the
  published 0.5-BLEU-4 cost.
- **4.3 Three channels.** Manual M (85 keypoints, 255-dim), non-manual NM (52 blendshapes + AU
  proxies + head/gaze), prosody P (speed, amplitude, repetition, pause, jerk). Justify P from the
  literature — EmoSign's annotators named these cues; eJSL showed hand motion helps.
- **4.4 The factorized encoder.** Give the loss in full:
  supervised `L` + supervised `A` + valence/arousal regression + GRL both directions +
  orthogonality + vCLUB. Explain *why each term exists* in one sentence each; a reviewer who cannot
  tell which term does the work will assume none of them do.
- **4.5 Disentanglement measurement.** Frozen cross-probes; cross-prediction AUC with 0.5 as the
  ideal. Define this *before* showing results.
- **4.6 Emotion-conditioned generation.** Control tokens into T5-small. **Disclose the
  LLM-generated paraphrase corpus here, in the method, not buried in a footnote.**
- **4.7 Avatar control.** Bone retargeting; 52→VRM blendshape mapping; emotion modulating
  blendshape gain, amplitude and timing, motivated by ASL affective prosody.
- **4.8 Efficiency design.** State that lightweight is a design choice, not post-hoc compression:
  keypoints not pixels, 12 fps not 24, ≤2M-param recognition, T5-small, INT8.

---

## 5. Experimental Setup

Training hardware (RTX 4060 16 GB + Colab/Kaggle) vs **inference hardware (RTX 3050 4 GB)** — and
the contrast with the 80 GB A100 used for the published MLLM baselines. Optimizer and schedule
(AdaFactor, lr 1e-3 constant, 10-epoch warmup, grad clip 1.0, label smoothing 0.1, batch 128, beam
5, 256-frame / 128-token caps). Seeds and repetitions. Metrics with tooling (SacreBLEU default
tokenization, BERTScore, weighted acc/F1). The LOSO protocol restated. Statement that every number
comes from a logged run.

---

## 6. Results

Order matters — build the argument, don't dump tables.

- **6.1 Isolated recognition** (Table I): three models × Top-1/Top-5/params/latency. Establishes
  the backbone is sound.
- **6.2 Affect recognition vs frontier MLLMs** (Table II): ours vs GPT-4o / Qwen2.5-VL-7B /
  AffectGPT / MiniGPT4-video, video-only and video+caption, plus the hearing-non-signer reference.
  mean ± std over 4 LOSO folds. **The headline table.**
- **6.3 Disentanglement** (Table III + Fig. 2): entangled vs factorized × cross-pred AUC × affect
  wF1; per-loss-term ablation. **The contribution table.**
- **6.4 Qualitative** (Fig. 3): wh-question and negation clips with neutral affect — baseline says
  anger/frustration, factorized model does not. **The most persuasive figure in the paper.**
- **6.5 Prosody and pretraining ablations** (Table IV): ±prosody channel, ±DFEW/MAFW pretraining.
- **6.6 Emotion-conditioned generation** (Table V): style accuracy, BERTScore, human preference,
  with examples.
- **6.7 Continuous translation** (Table VI): 24 vs 12 fps BLEU/FLOPs, vs compact-SLT and T5-base.
- **6.8 Efficiency Pareto** (Table VII): accuracy × p50/p95 latency × peak VRAM × params,
  measured on the 3050. **The second headline.**

Every table caption must name the split protocol. Every comparison to a published number must cite
it and note whether it was re-run or quoted.

---

## 7. System & Demo

The runtime path (FastAPI + WebSocket → React Three Fiber + three-vrm), the measured latency HUD, a
screenshot strip of the avatar under different emotions, and a link to the demo video. Note the
localhost-only, unauthenticated-by-default posture and why that is deliberate for biometric data.

---

## 8. Limitations & Ethics

Do not soften this section; it is where domain reviewers decide whether to trust you.

- **200 clips, 4 signers.** Even under LOSO, variance is high; report CIs and refuse to claim
  generality across signers, dialects, or recording conditions.
- **Annotation ceiling.** ᾱ=0.593; surprise− 0.119, disgust 0.166. Per-class results on those
  classes are near-unreliable by construction, and we say so.
- **Grammar labels.** If the ASLLRP fallback was used, say that `L` supervision was partly
  pseudo-labelled and that this weakens the disentanglement claim.
- **LLM-generated paraphrase corpus** for emotion conditioning — disclosed, with its failure modes.
- **Domain shift.** ASLLRP is lab-recorded, single-signer, scripted-utterance data. No claim about
  in-the-wild multi-signer conversation.
- **Deaf-community involvement.** We build on annotations produced by Deaf native signers and cite
  the finding that hearing non-signers misread grammatical markers. State the level of Deaf
  involvement in *our* work honestly, including if it was limited.
- **No clinical, legal, or emergency readiness claim.**
- **Data stewardship.** No redistribution of dataset video (ASL Citizen forbids it; ASLLRP is
  access-controlled; EmoSign releases IDs+labels only). We release code, weights, and labels.
- **Cross-lingual caution.** If the INCLUDE (ISL) ablation ran on a partial download, say which
  categories were used.

---

## 9. Conclusion

Restate the factorization insight, the two headline numbers (affect wF1 and VRAM/latency), and name
the future work concretely: continuous emotion-aware translation, generative expressive motion
(CVAE/diffusion), ISL/NSL extension via iSign, and larger Deaf-annotated affect corpora.

---

## Figures & Tables checklist

| ID | Content | Source task | Owner | Status |
|---|---|---|---|---|
| Fig. 1 | Pipeline architecture | design | R4 | ☐ |
| Fig. 2 | Cross-prediction AUC: entangled vs factorized | T10 | R3 | ☐ |
| Fig. 3 | Qualitative: wh-question misread as anger | T10 | R3 | ☐ |
| Fig. 4 | Avatar under 4 emotions (screenshot strip) | T15 | R4 | ☐ |
| Fig. 5 | Latency/VRAM Pareto plot | T13 | R2 | ☐ |
| Tab. I | Recognition ladder | T5 | R2 | ☐ |
| Tab. II | EmoSign vs MLLM baselines (LOSO) | T8/T9 | R3 | ☐ |
| Tab. III | Disentanglement + loss ablation | T10 | R3 | ☐ |
| Tab. IV | Prosody & pretraining ablations | T7/T9 | R3 | ☐ |
| Tab. V | Emotion-conditioned generation | T11 | R3 | ☐ |
| Tab. VI | Continuous SLT, 24 vs 12 fps | T12 | R2 | ☐ |
| Tab. VII | Efficiency Pareto on RTX 3050 | T13 | R2 | ☐ |

---

## Submission checklist

- [ ] Every number traces to a run ID in `EXPERIMENT_LOG.md`
- [ ] `CLAIMS_LEDGER.md` has zero unverified claims
- [ ] Every table caption states the split protocol
- [ ] Every quoted baseline number is cited and marked quoted-vs-rerun
- [ ] LOSO stated in Datasets, Setup, and every relevant caption
- [ ] Limitations section names the α ceiling and the 4-signer constraint
- [ ] Ethics: no video redistribution, model cards, Deaf-community statement
- [ ] Anonymized for review if required; arXiv version de-anonymized
- [ ] Page limit and format verified against the venue's current CFP
- [ ] Code + weights release links live (or "upon acceptance" stated)
- [ ] Demo video uploaded and linked
