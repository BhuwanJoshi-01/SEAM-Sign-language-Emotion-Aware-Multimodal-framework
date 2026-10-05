# Claims Ledger

Every claim the paper makes must appear here **before** it appears in `main.tex`, with the
experiment that proves it and the run ID that produced the number. A claim with no run ID is a
hypothesis, not a result.

**Status values:** `hypothesis` → `in progress` → `verified` → `refuted` / `weakened`

A `refuted` claim is not a failure of the project — it is a finding. Report it. A `weakened` claim
must be rewritten in the paper to match what the evidence actually supports.

---

## Headline claims

| ID | Claim | Evidence required | Milestone | Run ID | Status |
|---|---|---|---|---|---|
| **C1** | Off-the-shelf non-signer facial-emotion models systematically misread grammatical non-manual markers in sign language as negative affect | Marker-bearing vs matched marker-free segments, >=2 pretrained FER models, >=500 clips, CIs + effect sizes + marker->emotion confusion matrix | M1 | `artifacts/audit/confound_audit.json`, `artifacts/audit/confound_audit_continuous.json` | **refuted as measured** — no support on isolated signs (heuristic markers) or on continuous signing (human frame-level markers); wh- and yes/no questions untestable on this corpus |
| **C1b** | Grammatical and affective non-manual signals can be explicitly factorized, and separation is measurable | Cross-prediction AUC drops from entangled baseline toward 0.5; per-loss-term ablation | M4 | `artifacts/m4/factorizer_multilabel.json`, `factorizer_human_labels.json` | **refuted** |
| **C1c** | The disentanglement metric detects entanglement when entanglement exists (positive control) | Signer-embedding probe AUC >= 0.80 on the same features; a metric that reads 0.5 on everything is not evidence | M4 | `artifacts/m4/factorizer_multilabel.json` | **verified** |
| **C2** | Factorization improves affect recognition rather than trading against it | Affect wF1 with factorization ≥ entangled baseline under LOSO | T10 | `artifacts/m4/factorizer_multilabel_ablation.json` | **not supported** — unevaluable while neither model learns affect; see the 2026-10-05 update |
| **C3** | A compact specialist model beats frontier MLLMs on EmoSign from video alone | wF1 > 20.76 (GPT-4o), > 18.53 (Qwen2.5-VL), > 11.03 (AffectGPT), > 22.02 (MiniGPT4) under 4-fold LOSO | T8, T9 | — | hypothesis — **never run** on a protocol comparable to the quoted baselines |
| **C4** | The full emotion-aware pipeline runs on a 4 GB consumer GPU at interactive latency | Measured on RTX 3050: peak VRAM < 2500 MB, p95 < 400 ms, ≥20 FPS | T13 | `artifacts/bench/latest.json` | in progress — perception stage **verified**; no end-to-end pipeline exists to measure |

## Supporting claims

| ID | Claim | Evidence required | Milestone | Run ID | Status |
|---|---|---|---|---|---|
| C5 | Manual prosody (speed, amplitude, repetition, pause, jerk) carries affective information | Ablation: ±prosody channel changes affect wF1 by a margin exceeding LOSO std | M3, M4 | — | hypothesis — not run |
| C6 | Dynamic-FER pretraining transfers to sign-video affect | Ablation: pretrained > from-scratch; linear probe > chance | M4 | — | hypothesis — not run; no dynamic-FER pretraining was done |
| C7 | A ≤2M-parameter keypoint model is competitive for isolated ASL recognition | Within 3 pts of published ASL Citizen pose baselines; WLASL-100 Top-1 ≥80% | M5 | — | hypothesis — not run; ASL Citizen not fetched |
| C8 | 12 fps input preserves translation quality while cutting attention cost ~75% | BLEU-4 at 12 vs 24 fps; measured FLOPs | M5 | — | hypothesis — M5b not started |
| C9 | Emotion conditioning changes style while preserving meaning | Style accuracy ≥80% at BERTScore-F1 ≥0.90 | M6 | — | hypothesis — M6 not started |
| C10 | Emotion-modulated avatar output is preferred over neutral | Blinded pairwise human preference ≥60% with reported inter-rater agreement | M7 | — | hypothesis — stimuli ready, **zero raters**; and no learned affect signal exists to modulate with |
| C11 | The entangled baseline reproduces the documented hearing-non-signer error (grammatical markers read as negative affect) | Qualitative set of neutral-affect wh-question / negation clips, with per-clip predictions | T10 | — | hypothesis — not run |
| C12 | The system generalizes partially across sign languages | INCLUDE zero-shot and few-shot Top-1 vs chance | M8, M9 | — | hypothesis — M8 blocked on NSL terms |

## Claims we explicitly do NOT make

Recording these protects the paper from reviewer over-reading and protects users from us.

| Non-claim | Why |
|---|---|
| Reliable per-class accuracy on `surprise (negative)` and `disgust` | Krippendorff α of 0.119 and 0.166 in EmoSign bound what is knowable |
| Generalization across signers, dialects, or in-the-wild conversation | 4 signers, lab-recorded scripted utterances |
| Readiness for clinical, legal, or emergency-setting deployment | Not evaluated for safety-critical use; motivation ≠ product claim |
| Novelty in isolated sign recognition architecture | We adopt published compact recipes deliberately |
| That emotion labels are ground truth about the signer's internal state | They are third-party annotator judgments, with documented disagreement |
| That the LLM-generated paraphrase corpus reflects natural Deaf English usage | It is synthetic and disclosed as such |

---

## How to use this file

1. When you propose a claim, add the row with status `hypothesis` and the evidence you would accept.
2. When you run the experiment, paste the run ID and set `in progress` → `verified`/`weakened`/`refuted`.
3. Before submission, `main.tex` must contain **no claim absent from this table with status
   `verified`**, and no `verified` row whose run ID is missing from `EXPERIMENT_LOG.md`.
4. If a headline claim ends `refuted`, escalate at the next Friday review — the paper's narrative
   changes and that is a team decision, not an author's.


---

## Ledger update — 2026-10-01, M4 decision taken

**Decision: M4 is reported as a refuted hypothesis, not a passed contribution.**

| Claim | Before | After |
|---|---|---|
| Linguistic/affect factorisation separates on EmoSign | claimed, gate **NOT MET** (worst-fold cross-AUC 0.7276 vs ≤0.60) | **refuted**, and the obvious confound ruled out |

The gate was not moved to fit the result. Option (c) — collect more data and re-run — was
attempted on 2026-10-01 and **refuted**: the experiment that could have explained the
failure away was run, and it did not explain it.

**What the negative result rests on.** Two things are excluded rather than assumed:

1. **Label quality.** Three of M4's four linguistic labels are heuristic pseudo-labels
   measured at or near chance against the human ASLLRP annotations on the same 200 clips
   (kappa 0.028, 0.038, 0.141; the fourth, negation, is 0.639). Re-running the entire
   experiment with human labels in the same slots moved worst-fold cross-AUC from 0.7276
   to 0.7031 and left the gate failing. Artifact: `artifacts/m4/factorizer_human_labels.json`.
2. **Instrument validity.** The signer positive control reads 0.973 against a ≥0.80 floor.
   A negative result from an instrument that cannot see anything is not evidence.

**What may not be claimed.** That the factors are separable with a better model, a larger
corpus, or a different architecture — none of those was tested, and EmoSign is exhausted at
200 clips so "more data" of that kind is not currently available. What may be claimed is
narrower and is stated in `plan.md` and `EXPERIMENT_LOG.md`.

**New supporting claim, from the same work.** The project's heuristic pseudo-labels for
interrogative, topicalization and reference-establishment are unusable as linguistic
targets: kappa 0.028, 0.141 and 0.038 against human annotation. Artifact:
`artifacts/m3/label_agreement.json`. This is a reusable result for anyone building
pseudo-labelled sign-language data, and it is the one finding here that other people will
need.


---

## Ledger update — 2026-10-05, the headline table brought up to date

Until today every row of both tables read `hypothesis`, including C1 and C1b, which were
refuted on 2026-09-27 and 2026-10-01. The status columns above now say what was measured.
What changed in substance:

**C1 is narrower than it was recorded.** It is refuted for brow raise, brow furrow and mouth
morphemes on isolated WLASL signs. The two head-shake rows that were reported as nulls were
measured on a broken yaw decomposition; re-run, `head_shake` and `head_nod` each have one
matched pair and no estimate. And the claim has still never been tested where the literature
places the effect - continuous signing - although three FER models and human frame-aligned
markers on the same 200 utterances are all in hand.

**C1b stays refuted, on corrected numbers.** M4 was re-run on yaw-fixed code after two
defects in its own instrument were removed. Worst-fold cross-AUC is 0.6944 with heuristic
linguistic labels and 0.7264 with human ones, against ≤0.60, and the gate fails for every
ablation variant in each of three seeds. Artifacts: `artifacts/m4/factorizer_multilabel.json`,
`factorizer_human_labels.json`, `factorizer_multilabel_ablation.json`.

**C1c is verified.** The signer probe reads 0.9812 against a ≥0.80 floor.

**C2 is not supported, and was nearly recorded the other way.** The entangled baseline was
trained without the class and positive-label weights the factorized model had, which put its
affect micro-F1 near 0.06 against 0.33 - a gap that reads as "factorization improves affect
recognition". Trained like-for-like the baseline scores 0.308 and the factorized model
0.324 ± 0.019 over three seeds, so there is no measurable difference; and affect balanced
accuracy is 0.481 to 0.507 against a reference of 0.5, so neither model has learned the task
there would be to improve.

**C3 remains a hypothesis because it has never been run.** No affect model has been scored on
a protocol comparable to the quoted 20.76 or 21.09. This is the largest gap between what the
project set out to claim and what it has measured.

**One supporting claim is withdrawn.** The 2026-10-01 update's companion finding in
`EXPERIMENT_LOG.md` - that the negation heuristic was corroborated by a visual head-shake
association - rested on the same yaw bug. The label-agreement result itself (kappa for the
five heuristic categories) is unaffected: it was re-run today and is identical.

**What may not be claimed.** That separation "works but is not enough", that the affect
factor is free of linguistic information, or that any separation loss is responsible for an
improvement. Three seeds show no lever moving the direction the gate fails on.


### Same day, second update — a result that was recorded as negative and was not a result

**M5a (gloss recognition) is superseded, with the same verdict.** It is not a ledger claim,
but its conclusion - "signing-space pose carries no usable lexical signal at this scale" -
was being carried in the plan and the README, and it rested on two defects:

1. ASLLRP frame indices are on a 30 fps timeline and 138 of the 200 clips are 24 fps. The
   token frames were mapped frame-for-frame, so on most of the corpus the labels pointed at
   the wrong frames. Validated against an independent signal - annotated blinks against the
   eye-blink blendshape - the corrected mapping scores AUC 0.710 on those clips where the
   old one scored 0.554. Artifact: `artifacts/m3/frame_alignment.json`.
2. The classifier predicted a single gloss in every fold, on real and shuffled labels alike.
   Its WER equalled the most-frequent baseline because it was the most-frequent baseline.

Re-run with both fixed, WER is 0.920 against a most-frequent baseline of 0.920 on 1,736
tokens over 546 glosses (`artifacts/m5a/recogniser.json`). The gate is still not met. What
changed is the kind of statement: before, the experiment could not have returned anything
else; now it could have, and did not.

**What may not be claimed.** That pose is uninformative for sign identity. The corpus has
3.18 tokens per gloss; this is a statement about 200 utterances.

**What this means for every claim still marked `hypothesis`.** Any experiment that reads a
frame-level ASLLRP annotation - which includes the two that matter most, C1 on continuous
signing and a validated marker detector - must take the clip's frame rate. The conversion is
`asllrp.crop_frame_position` and it has no default rate.

### Same day, third update — C1 has now had the test M1 asked for

The claim was run on continuous signing with human frame-level markers
(`artifacts/audit/confound_audit_continuous.json`), under a design and decision rule fixed
before the first run. **No marker meets the rule**, so C1 moves from "refuted on isolated
signs, untested on continuous signing" to refuted as measured, on both.

What the row above must not be read as saying:

- Not that marker-bearing windows look the same to a FER model as marker-free ones. They
  differ more often than a placebo does, in both directions: head nods are read as less
  negative, brow furrows lean less negative in all three models, brow raises and rhetorical
  questions lean more negative in two.
- Not that the hypothesis is excluded for question marking. Only 8 and 4 clips contribute a
  yes/no- or wh-question pair inside the signing span, too few for an interval.
- Not that a stronger FER model would agree. These are compact CNNs whose read-out tracks
  true affect only weakly on these clips.

And one thing it does say that the paper should keep: the same audit without the
restriction to the signing span reports **support** for brow raise and rhetorical questions.
That version compares signing faces with resting ones. The restriction was decided before
either analysis ran.


### Same day, fourth update — a supporting claim that can be marked verified

**New supporting claim, verified.** The brow-raise marker can be read from MediaPipe
blendshapes, frame by frame, on signers the detector has not seen: within-clip AUC against
human annotation of 0.838, 0.822 and 0.878 on the three large leave-one-signer-out folds,
from the unfitted mean of three brow coefficients. Artifact:
`artifacts/m3/marker_validation.json`. This is the only visual marker in the project that
has been validated against an annotator.

**And three that cannot be claimed.** Brow furrow does not pass the same gate (0.599 on the
largest fold). Head shake and head nod are not recoverable from head pose on these crops
even by a fitted detector. Any sentence in the paper that treats the project's visual
"negation" or "wh-question" markers as measurements of those things is unsupported; only the
human annotations measure them.
