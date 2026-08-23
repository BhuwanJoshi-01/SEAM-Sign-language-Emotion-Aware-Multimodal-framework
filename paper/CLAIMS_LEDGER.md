# Claims Ledger

Every claim the paper makes must appear here **before** it appears in `main.tex`, with the
experiment that proves it and the run ID that produced the number. A claim with no run ID is a
hypothesis, not a result.

**Status values:** `hypothesis` → `in progress` → `verified` → `refuted` / `weakened`

A `refuted` claim is not a failure of the project — it is a finding. Report it. A `weakened` claim
must be rewritten in the paper to match what the evidence actually supports.

---

## Headline claims

| ID | Claim | Evidence required | Task | Run ID | Status |
|---|---|---|---|---|---|
| **C1** | Grammatical and affective non-manual signals can be explicitly factorized, and separation is measurable | Cross-prediction AUC drops from entangled baseline toward 0.5; per-loss-term ablation | T10 | — | hypothesis |
| **C2** | Factorization improves affect recognition rather than trading against it | Affect wF1 with factorization ≥ entangled baseline under LOSO | T10 | — | hypothesis |
| **C3** | A compact specialist model beats frontier MLLMs on EmoSign from video alone | wF1 > 20.76 (GPT-4o), > 18.53 (Qwen2.5-VL), > 11.03 (AffectGPT), > 22.02 (MiniGPT4) under 4-fold LOSO | T8, T9 | — | hypothesis |
| **C4** | The full emotion-aware pipeline runs on a 4 GB consumer GPU at interactive latency | Measured on RTX 3050: peak VRAM < 2500 MB, p95 < 400 ms, ≥20 FPS | T13 | — | hypothesis |

## Supporting claims

| ID | Claim | Evidence required | Task | Run ID | Status |
|---|---|---|---|---|---|
| C5 | Manual prosody (speed, amplitude, repetition, pause, jerk) carries affective information | Ablation: ±prosody channel changes affect wF1 by a margin exceeding LOSO std | T7, T8 | — | hypothesis |
| C6 | Dynamic-FER pretraining transfers to sign-video affect | Ablation: pretrained > from-scratch; linear probe > chance | T9 | — | hypothesis |
| C7 | A ≤2M-parameter keypoint model is competitive for isolated ASL recognition | Within 3 pts of published ASL Citizen pose baselines; WLASL-100 Top-1 ≥80% | T5 | — | hypothesis |
| C8 | 12 fps input preserves translation quality while cutting attention cost ~75% | BLEU-4 at 12 vs 24 fps; measured FLOPs | T12 | — | hypothesis |
| C9 | Emotion conditioning changes style while preserving meaning | Style accuracy ≥80% at BERTScore-F1 ≥0.90 | T11 | — | hypothesis |
| C10 | Emotion-modulated avatar output is preferred over neutral | Blinded pairwise human preference ≥60% with reported inter-rater agreement | T15 | — | hypothesis |
| C11 | The entangled baseline reproduces the documented hearing-non-signer error (grammatical markers read as negative affect) | Qualitative set of neutral-affect wh-question / negation clips, with per-clip predictions | T10 | — | hypothesis |
| C12 | The system generalizes partially across sign languages | INCLUDE zero-shot and few-shot Top-1 vs chance | T16 | — | hypothesis |

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
