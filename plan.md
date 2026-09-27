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

### M1 — The confound audit ★ first headline result, needs zero emotion labels
Extract the non-manual channel (52 ARKit blendshapes + head pose + gaze) from ≥2,000 WLASL clips.
Run ≥3 pretrained non-signer FER models (EMO-AffectNet, RAF-DB, AffectNet) over it. Detect
grammatical markers by rule on the *same* blendshapes: brow-raise (yes/no Q), brow-furrow (wh-Q),
head-shake (negation), mouth-morpheme clusters. Test whether FER output shifts toward negative
affect on marker-bearing vs matched marker-free segments.

**Gate:** measured, replicated across ≥2 pretrained models on ≥500 clips, with CIs and effect
sizes, plus a marker→emotion confusion matrix.
**Kill switch:** a null result is still a paper — "non-manual markers do *not* confound non-signer
FER in ASL" is a genuine result and reframes the contribution honestly.
**Survives:** a complete first contribution plus the feature stack for every later milestone.

### M2 — Efficiency harness & the 4 GB budget, proven early
Benchmark MediaPipe-Tasks throughput and VRAM on the 3050 now, so M3+ are designed against real
numbers. ONNX export + FP32↔INT8 parity harness + a peak-VRAM enforcer that **fails loudly
rather than degrading silently**. Add `onnxruntime-gpu` for the CUDA EP.

**Gate:** measured p50/p95/p99 + peak VRAM on the 3050 for the perception stage; CI test that
fails above the ceiling.
**Kill switch:** never cut — headline claim.
**Survives:** the efficiency section.

### M3 — Linguistic-marker supervision (`L` labels)
**Track A (blocking):** rule-based marker labeller over blendshapes + syntactic analysis of the
utterance (interrogative / negation / topicalization). Every label carries a provenance flag;
heuristic labels marked `pseudo` in every downstream table and in the paper.
**Track B (async, non-blocking):** BU ASLLRP access → real SignStream non-manual XML. Filed in
M0, chased weekly.
**Grounding move:** validate the prosody feature set against the **600 free-text Deaf-annotator
cue strings** in the EmoSign CSV. The annotators named sign size, speed, repetition, emphatic
fingerspelling, brow, head-shake. Measure whether our features recover the cues they named —
turning an assumed feature set into a data-grounded one.

**Gate:** marker labels on ≥200 clips with documented provenance and agreement statistics;
feature↔cue correlation report.
**Kill switch:** Track B landing triggers re-labelling and an M4 re-run; otherwise ship Track A
as disclosed pseudo-labels.
**Survives:** the core claim at reduced evidentiary strength, honestly labeled.

### M4 — Factorized non-manual encoder + EmoSign LOSO ★ CORE CONTRIBUTION
`z_L = Enc_L(NM)`, `z_A = Enc_A(NM, P)`, each ≤1M params. Losses: `CE(head_L)`, `CE(head_A)`,
`MSE(head_VA)`, **GRL both directions**, `‖z_Lᵀ z_A‖²_F`, **vCLUB** MI upper bound. 4-fold LOSO
on the 200 clips; entangled single-branch baseline; per-λ ablation harness; frozen cross-probes →
cross-prediction AUC (0.5 = perfect separation).
**Positive control (rule 13):** a signer-embedding probe must score high, proving the metric
detects entanglement when it exists.
**Qualitative set:** neutral-affect wh-question and negation clips where the entangled baseline
calls it anger — the documented hearing-non-signer error — and the factorized model does not.

**Gate:** cross-prediction AUC ≤ 0.60 **with no drop** in affect wF1, under LOSO, positive control
passing.
**Kill switch:** if AUC improves but wF1 drops, rewrite the claim to "separation without loss" and
publish the trade-off curve.
**Survives:** the paper.

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

**Gate:** preference ≥ 60% over neutral; browser smoke test; 10 min at 30 fps without leak.
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
| **K1** | Non-signer FER bias from grammatical markers | 0 = no bias | ≠ 0, 2+ models, CIs | — | M1 |
| **K2** | Disentanglement cross-prediction AUC | 0.5 = perfect | **≤ 0.60**, no affect loss | — | M4 |
| **K3** | Positive control: signer probe AUC | — | ≥ 0.80 | — | M4 |
| **K4** | EmoSign emotion macro-F1, video-only, LOSO | **eJSL EANwH 21.09**; GPT-4o 20.76 | **> 21.09** | — | M4 |
| **K5** | Peak inference VRAM / p95 latency on the 3050 | 4096 MB hard limit | **< 2500 MB / < 400 ms** | — | M2, M7 |
| K6 | Sustained capture FPS | — | ≥ 20 | — | M2 |
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
