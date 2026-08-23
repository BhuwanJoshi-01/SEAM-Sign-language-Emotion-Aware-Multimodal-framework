# Writing Guide

House style for the SEAM paper. The goal is a paper a domain reviewer trusts. In sign-language
research the reviewers frequently include Deaf researchers and linguists; vague claims and
technically-naive framing are caught immediately.

---

## 1. Non-negotiables

1. **No number without a run ID.** If it is not in `EXPERIMENT_LOG.md`, it does not go in the paper.
2. **No claim without a ledger row.** `CLAIMS_LEDGER.md` status must be `verified`.
3. **Every table caption states the split protocol.** "4-fold leave-one-signer-out, mean ± std."
4. **Every quoted baseline is cited and marked** as quoted rather than re-run.
5. **Placeholders must be loud.** Use `\TODO{}` and `\NUM` from `main.tex`. Never type a
   plausible-looking fake number "to see how the table renders" — it will survive to submission.
6. **Report the misses.** A target we did not hit goes in Limitations, not in a drawer.

## 2. Terminology — get this right

| Use | Not | Why |
|---|---|---|
| Deaf (capital D) for the cultural-linguistic community | deaf/hearing-impaired | Community convention |
| sign language *users* / signers | "the deaf", "sign language speakers" | Respectful and accurate |
| gloss | translation | A gloss is a label for a sign, not a translation |
| non-manual marker (NMM) for grammatical function | "facial expression" | The whole point of this paper is that the two differ |
| affective / emotional expression for affect | "expression" alone | Ambiguous between the two functions |
| American Sign Language (ASL) as a full natural language | "signed English" | They are different languages |
| translation (sign → spoken language) | "interpretation" | Interpretation is a human professional act |
| annotator judgments of emotion | "the signer's emotion" | We measure perceived affect, not internal state |

Do not write that sign language "lacks grammar", "is universal", or "is a visual form of English".
Each is false and each will end the review.

## 3. Voice and structure

- **Active voice, present tense** for what the paper does: "We factorize…", "Table II reports…".
- **One idea per paragraph.** Lead with the point; support after.
- **No throat-clearing.** Cut "It is important to note that", "In recent years, with the rapid
  development of deep learning".
- **Motivate every design choice in one sentence.** If you cannot, the choice is arbitrary — either
  find the reason or drop the component. (This is why YOLO left the pipeline: it had no reason.)
- **Numbers in prose get context:** "wF1 XX.X, versus 20.76 for GPT-4o from video" — never a bare
  number.
- **Define before you measure.** Cross-prediction AUC is defined in Method, used in Results.

## 4. Honesty patterns

Use these constructions; they cost nothing and buy credibility:

- "With 200 clips from 4 signers, even under leave-one-signer-out cross-validation these results
  carry substantial variance; we report standard deviation across folds and refrain from claiming
  signer-general performance."
- "Inter-annotator agreement for `surprise (negative)` (α = 0.119) and `disgust` (α = 0.166) is low,
  so per-class results for these categories should be read as indicative only."
- "Linguistic-marker supervision was derived from [ASLLRP annotations / heuristic pseudo-labels];
  in the latter case the disentanglement result should be considered a weaker form of evidence."
- "Emotion-styled paraphrases were generated with an LLM and filtered for semantic equivalence;
  they are synthetic and may not reflect natural usage."
- "Baseline numbers are quoted from [ref] and were evaluated on an 80 GB A100; we did not re-run
  them."

Avoid: "significantly" without a test, "state of the art" without stating on which benchmark and
protocol, "real-time" without a measured number, "robust" without an adversarial or shifted
evaluation.

## 5. Figures and tables

- Readable at 100% zoom in a single column; no default matplotlib legends over data.
- Colour-blind-safe palettes; never encode meaning by colour alone (accessibility matters
  especially in this venue).
- Caption = what it shows + protocol + what to conclude. A caption that only names the axes wastes
  the most-read text in the paper.
- Bold only the winning row, and only if the win exceeds the reported variance.
- Every figure/table referenced in the text; no orphans.

## 6. Reproducibility section (mandatory)

State: hardware for training and separately for the reported latency; software versions; seeds and
number of repetitions; the exact split protocol; total compute; what is released (code, weights,
labels) and what cannot be (dataset video, per the ASL Citizen and ASLLRP terms) and why.

## 7. Pre-submission pass

- [ ] Zero `\TODO` and zero `\NUM` remaining (grep for them)
- [ ] Every table caption names the protocol
- [ ] Every claim traces: ledger → run ID → log entry
- [ ] Limitations names the 4-signer constraint and the α ceiling explicitly
- [ ] Ethics names data stewardship and Deaf-community involvement honestly
- [ ] Terminology pass against §2 of this guide
- [ ] Bibliography: every `TODO: verify author list` in `refs.bib` resolved against the publisher record
- [ ] Page limit and format checked against the venue's current CFP
- [ ] Read aloud once end to end — anything you stumble over, a reviewer will too
