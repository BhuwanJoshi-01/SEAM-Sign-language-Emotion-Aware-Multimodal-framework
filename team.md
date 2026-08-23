# SEAM — Team & Squad Plan

**10 weeks · Mon 2026-08-10 → Sun 2026-10-18 · 4 humans + 1 AI implementer**

The AI does the heavy implementation. The humans decide, review, unblock, negotiate data access,
run the human evaluation, and write the paper. This file says who owns what, when, and what
"reviewed" means. Task IDs (T1–T16) and phase gates are defined in `plan.md`.

---

## 1. Roles

Each person wears one role-hat for the whole project — continuity beats flexibility over 10 weeks.
Fill in names on day 1.

### R1 — Data & Perception Lead · `________`
**Owns:** T2, T3, T4 · **Assists:** T9, T16
- Dataset acquisition, licences, and the integrity gate
- **Day-1 duty: submit the ASLLRP access request, accept EmoSign HF terms, request DFEW + MAFW.**
  This is the project's critical path; nothing else matters on day 1.
- MediaPipe Tasks perception pipeline, keypoint extraction, `.npz` shard cache
- Normalization, smoothing, augmentation, dataset statistics
- Guards the "video is transient, keypoints are the asset" storage discipline
- **Paper sections:** Datasets, Preprocessing
- **Skills:** OpenCV/MediaPipe, data engineering, patience with licence bureaucracy

### R2 — Recognition & Efficiency Lead · `________`
**Owns:** T5, T12, T13 · **Assists:** T14
- The three-model recognition ladder (GRU → ST-GCN → pose transformer) and the parameter budget
- Phase-B gloss-free continuous translation and the fps/BLEU/FLOPs trade-off
- ONNX export, INT8 quantization, ONNX Runtime, optional TensorRT
- **Owner of the 4 GB constraint.** Runs every benchmark on the actual RTX 3050 and maintains the
  CI test that fails the build if peak VRAM > 2500 MB or p95 latency > 400 ms.
- W&B sweeps on the 4060
- **Paper sections:** Recognition, Efficiency results
- **Skills:** PyTorch, GCNs/transformers, profiling, deployment runtimes

### R3 — Language & Affect Lead · `________` *(lead author)*
**Owns:** T6, T7, T8, T9, T10, T11 · **Assists:** T16
- T5-small translation, gloss→text and emotion-conditioned decoding
- Affect features, prosody channel, and the **factorized non-manual encoder** (the contribution)
- **Guardian of methodological rigor:** LOSO CV, mean±std, confidence intervals, and the rule that
  no number enters the paper without a run ID
- Says "no" when a result is too good to be true, especially on low-α emotion classes
- **Paper sections:** Abstract, Introduction, Method, Affect experiments; coordinates the whole write-up
- **Skills:** NLP/transformers, representation learning (GRL, MI bounds), experimental design

### R4 — Systems & Avatar Lead · `________`
**Owns:** T1, T14, T15 · **Assists:** T11, T16
- Repo scaffold, CI, docs, developer experience
- FastAPI + WebSocket serving, backpressure, the localhost-only/no-auth security posture
- React Three Fiber + three-vrm avatar, retargeting, blendshape mapping, emotion modulation
- Demo video, screenshots, figures
- **Schedule keeper:** tracks the cut-lines and calls them before week 9, not during it
- **Paper sections:** System design, Demo, Ethics & limitations
- **Skills:** Python web, TypeScript/React, Three.js, 3D rigging, project discipline

### The AI implementer
Writes the code, tests, configs, and doc updates for every task; follows the Agent Operating Rules
in `plan.md` §0; escalates blockers in writing rather than silently substituting approaches.
Humans never merge AI output unreviewed — see §5.

---

## 2. Sprint Calendar

| Sprint | Weeks | Dates | Focus | Primary | Support | Friday-gate demo |
|---|---|---|---|---|---|---|
| **S0** Foundations | 1–2 | Aug 10 – Aug 23 | T1–T4 | R1, R4 | R2, R3 | Perception ≥20 FPS on the 3050 + dataset readiness table |
| **S1** Recognition | 3–4 | Aug 24 – Sep 6 | T5–T6 | R2, R3 | R1 | Video → gloss → English, end to end |
| **S2** Affect | 5–6 | Sep 7 – Sep 20 | T7–T9 | R3 | R1 | Beat GPT-4o video-only on EmoSign under LOSO |
| **S3** Contribution | 7–8 | Sep 21 – Oct 4 | T10–T12 | R3, R2 | R4 | Disentanglement figure + emotion-conditioned text |
| **S4** Ship | 9–10 | Oct 5 – Oct 18 | T13–T16 | R2, R4 | all | Live avatar demo under 2500 MB + submitted paper |

### Week-by-week
| Wk | Dates | Deliverable | Owner |
|---|---|---|---|
| 1 | Aug 10–16 | **Access requests filed (day 1)**, repo + docs + paper skeleton, dataset readiness table | R1, R4 |
| 2 | Aug 17–23 | Perception pipeline benchmarked, preprocessing validated, **P1 gate Fri Aug 21** | R1 |
| 3 | Aug 24–30 | GRU + ST-GCN on WLASL-100, leaderboard v1 | R2 |
| 4 | Aug 31–Sep 6 | ASL Citizen scale-up, T5 gloss→text, **P2 gate Fri Sep 4** | R2, R3 |
| 5 | Sep 7–13 | Affect + prosody features, correlation report | R3 |
| 6 | Sep 14–20 | EmoSign LOSO baseline, FER pretraining, **P3 gate Fri Sep 18** | R3 |
| 7 | Sep 21–27 | Factorized encoder, disentanglement metrics | R3 |
| 8 | Sep 28–Oct 4 | Conditioned generation, Phase-B SLT, **P4 gate Fri Oct 2** | R3, R2 |
| 9 | Oct 5–11 | ONNX/INT8, 3050 benchmarks, serving layer | R2, R4 |
| 10 | Oct 12–18 | Avatar, human eval, repro script, paper submission, **P5 gate Fri Oct 16** | R4, all |

---

## 3. RACI

**A** = accountable (one per row) · **R** = does the work with the AI · **C** = consulted · **I** = informed

| Task | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| T1 Docs & scaffold | C | C | C | **A/R** |
| T2 Data & licences | **A/R** | C | C | I |
| T3 Perception | **A/R** | C | I | C |
| T4 Preprocessing | **A/R** | C | C | I |
| T5 Recognition ladder | C | **A/R** | C | I |
| T6 Gloss→text | I | C | **A/R** | I |
| T7 Affect features | C | I | **A/R** | I |
| T8 EmoSign LOSO baseline | C | I | **A/R** | I |
| T9 FER pretraining | R | I | **A** | I |
| T10 Factorized encoder ★ | C | C | **A/R** | I |
| T11 Conditioned generation | I | I | **A/R** | R |
| T12 Phase-B SLT | C | **A/R** | C | I |
| T13 Export & 3050 benchmark ★ | I | **A/R** | C | C |
| T14 Serving | I | C | I | **A/R** |
| T15 Avatar | I | I | C | **A/R** |
| T16 Repro & paper | R | R | **A** | R |
| Human evaluation study | R | I | **A** | R |
| Paper submission | C | C | **A** | R |

---

## 4. Ceremonies

| When | Duration | Format | Output |
|---|---|---|---|
| Mon 09:00 | 30 min | Planning against `plan.md` | Checkboxes moved, week's demo target agreed |
| Wed | async | Written blocker sweep | Every red item has an owner + date |
| **Fri 16:00** | 45 min | **Demo or it didn't happen** | KPI dashboard updated, `EXPERIMENT_LOG.md` appended, one paragraph of paper written |
| Daily | async | Standup: done / next / blocked | Thread |
| Phase gates | 60 min | Go / no-go against exit criteria | Decision recorded in `plan.md` |

**Rules of the Friday review:**
1. A demo is a running artifact, not a slide.
2. Any KPI cell filled since last week must cite a run ID.
3. If a bold KPI (K6, K10, K15, K16) is at risk, that is the only agenda item.
4. Each phase gate requires its paper section drafted — the paper is written continuously.

---

## 5. Review Gates — what "reviewed" means

The AI implements; a human signs off. Sign-off requires the reviewer to have personally checked:

**Code review (any task)**
- [ ] Tests exist, are meaningful (not tautological), and pass
- [ ] No hardcoded paths, no secrets, dependency versions pinned
- [ ] Config-driven, seeded, reproducible

**Result review (any number destined for the paper)**
- [ ] Split is signer-independent; for EmoSign, LOSO with 4 folds
- [ ] mean ± std reported, not a single fold
- [ ] Run ID logged in `EXPERIMENT_LOG.md` with commit SHA
- [ ] The number is *plausible* — a suspiciously high score on a low-α emotion class
      (surprise− α=0.119, disgust α=0.166) is treated as a bug until proven otherwise
- [ ] Baseline comparison is apples-to-apples and the reference is cited

**Efficiency review (T13)**
- [ ] Measured on the RTX 3050, not the 4060, not estimated
- [ ] Peak VRAM < 2500 MB, p95 < 400 ms, asserted by a test
- [ ] FP32↔INT8 numerical parity checked

**Ethics review (T14, T15, T16)**
- [ ] No dataset video redistributed
- [ ] Server binds to loopback only; no unauthenticated exposure of a webcam stream
- [ ] Model card written; limitations and Deaf-community statement present
- [ ] No claim of clinical, legal, or emergency-setting readiness

---

## 6. Shared Responsibilities

**Human evaluation study (T11, T15) — R3 accountable, everyone participates**
- Pre-registered rubric written *before* any output is generated
- Blinded pairwise comparison; randomized order
- Minimum 5 raters; ideally include at least one signer — if none is available, say so explicitly
  in the paper's limitations rather than glossing over it
- Report inter-rater agreement alongside the preference score

**Paper writing — R3 coordinates, each role owns its sections**
- One paragraph per person per Friday, minimum
- `CLAIMS_LEDGER.md` is filled as claims are made, not retrofitted at the end
- No section is "done" until every number in it traces to a run ID

**Cut-line decisions — R4 calls them, team ratifies**
- Cut in order: (1) INCLUDE cross-lingual ablation, (2) Phase-B SLT, (3) GLB/VRMA export
- Called by end of week 8 at the latest, never during week 10
- Never cut: T10, T8, T13, LOSO methodology, ethics statement

---

## 7. Escalation

| Situation | Action | Escalate to |
|---|---|---|
| ASLLRP access not granted by end of week 3 | Switch to the documented pseudo-label fallback for `L`; note it in the paper | R3 → team |
| DFEW/MAFW licences not granted by week 5 | Switch to RAF-DB/AffectNet or OpenFace AU features | R1 → R3 |
| Disk space insufficient for ASL Citizen | Keypoint-extract in streaming batches, delete video per batch | R1 → R2 |
| Peak VRAM > 2500 MB after T13 optimization | Drop to 12 fps, shrink T5 further, reduce keypoint subset — in that order | R2 → team |
| A bold KPI misses target | Immediate agenda item at the next Friday review; re-scope the claim honestly, do not hide it | owner → R3 |
| Any ethical concern about data or deployment | Stop the affected work immediately | anyone → R4 |
| AI implementer blocked or looping on the same failure twice | Human takes over diagnosis; change approach, do not retry variations | task owner |

---

## 8. Onboarding Checklist (day 1, each member)

- [ ] Read `project_breakdown.md` end to end — especially §2.2 (why non-manual signals matter) and
      §7.1 (the non-negotiable methodology)
- [ ] Read `plan.md` §0 (Agent Operating Rules) and your own tasks
- [ ] Read this file, put your name on your role
- [ ] `make setup && make test` green on your machine
- [ ] W&B account joined to the project
- [ ] Understand the four bold KPIs (K6, K10, K15, K16) — these are the paper
- [ ] Understand the one fact that shapes the whole project: **hearing non-signers systematically
      misread grammatical facial markers as negative emotion, and so do off-the-shelf models.
      Separating grammar from affect is the point.**
