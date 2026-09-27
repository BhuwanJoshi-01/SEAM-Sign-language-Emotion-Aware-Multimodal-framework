# SEAM — Sign-language Emotion-Aware Multimodal framework

*Factorized Non-Manual Modeling for Emotion-Aware American Sign Language Translation and
Expressive 3D Avatar Generation — on a 4 GB GPU.*

Most sign language systems read the hands and throw away the face. In ASL the face is
grammatically obligatory (raised brows mark yes/no questions, a furrowed brow marks wh-questions, a
head-shake marks negation) *and* it carries emotion. The two functions are confounded, and models —
like hearing non-signers — systematically misread grammar as negative affect. SEAM separates them,
then uses the recovered affect to condition both the translated text and an expressive avatar.

---

## Documents

| File | What it is | Read it when |
|---|---|---|
| **[project_breakdown.md](project_breakdown.md)** | The technical bible: motivation, literature synthesis, datasets, architecture, losses, evaluation protocol, risks, ethics, glossary | You want to understand *why* |
| **[plan.md](plan.md)** | The executable plan: 10 milestones with evidence gates, kill switches, KPI dashboard, revised operating rules | You want to know *what to do next* |
| **[implimentation.md](implimentation.md)** | Task-level expansion of `plan.md`: repo map, command map, per-milestone file and test breakdown | You are implementing |
| ~~[team.md](team.md)~~ | Squad roles (R1–R4), 10-week sprint calendar, RACI | **Superseded** for planning by `plan.md`; retained for role history and review-gate language |
| **[paper/PAPER_TEMPLATE.md](paper/PAPER_TEMPLATE.md)** | Section-by-section paper scaffold with word budgets and evidence requirements | You are writing |
| **[paper/main.tex](paper/main.tex)** | IEEEtran skeleton with every result table pre-stubbed | You are writing |
| **[paper/WRITING_GUIDE.md](paper/WRITING_GUIDE.md)** | House style, terminology rules, honesty patterns | You are writing |
| **[paper/CLAIMS_LEDGER.md](paper/CLAIMS_LEDGER.md)** | Every claim → evidence → run ID → verdict, plus claims we explicitly refuse to make | You are about to assert something |
| **[paper/EXPERIMENT_LOG.md](paper/EXPERIMENT_LOG.md)** | Append-only record of every run behind every number | You just ran something |
| **[paper/refs.bib](paper/refs.bib)** | Bibliography seeded with the full literature survey | You are citing |
| [signemotionaware.md](signemotionaware.md) | The original concept note, kept for provenance | Historical interest |

## The headline numbers

| KPI | Reference | Target | Milestone |
|---|---|---|---|
| Non-signer FER bias from grammatical markers | 0 = no bias | ≠ 0, 2+ models, CIs | M1 |
| Disentanglement cross-prediction AUC | 0.5 = perfect separation | **≤0.60**, no affect loss | M4 |
| EmoSign emotion macro-F1, video-only, LOSO | **21.09** eJSL EANwH · 20.76 GPT-4o | **>21.09** | M4 |
| Peak inference VRAM / p95 latency on RTX 3050 | 4096 MB hard limit | **<2500 MB / <400 ms** | M2, M7 |

Every published baseline ran on an 80 GB A100 or a 300M-parameter model. We target 4 GB, and the
efficiency claim is currently unshared.

## Schedule

**Milestone-driven, no dates.** Ten milestones (M0–M9) ordered by dependency × evidence-value ÷
cost, each closing on an *evidence gate* rather than a deadline. Each declares a kill switch and
what survives if everything after it is cut. See `plan.md`.

## Status

| Milestone | Gate | State |
|---|---|---|
| **M0** data spine | readiness table · 200/200 clips · face gate · tests green | **OPEN** |
| M1 confound audit | measured FER bias, 2+ models, CIs | next |
| M2 efficiency | 3050 p95 + peak VRAM, CI-enforced | — |
| M3 `L` labels | marker labels with provenance + cue correlation | — |
| M4 factorized encoder ★ | cross-pred AUC ≤0.60, no affect loss, positive control | — |
| M5 recognition + translation | How2Sign BLEU-4 at ≤80M params | — |
| M6 conditioned generation | style acc ≥80% @ BERTScore ≥0.90 | — |
| M7 avatar + live demo | preference ≥60%, browser smoke test | — |
| M8 cross-lingual | zero-shot Top-1 vs chance | — |
| M9 paper + release | `repro_all.sh` green, submission filed | — |

**M0 closed 2026-09-26.** The EmoSign affect benchmark is reachable: its `video_name` trailing
numeric token is the ASLLRP utterance ID, measured at **200/200**, and the face-visibility gate
passed **24/24** sampled clips (face detected in 94–100% of frames, median 30 of 52 blendshapes
carrying real temporal variance). See `paper/EXPERIMENT_LOG.md`.

## The remaining data risk

The 200 clips resolve through an **ungated re-upload of a Boston-University-controlled corpus**
(ASLLRP). Decision on record: use for research now, formalize before submission. The Ethics
section owes the reader an answer on provenance; owner is the reviewer. A parallel BU access
request runs as the clean fallback for the linguistic non-manual annotations, which the mirror
does **not** contain.

WLASL is local (3,863 clips / 668 glosses / 7.4 GB) but needs a repair pass: a 300-file sample
found 2 undecodable, truncated `.part` containers. M1 repairs before extracting.

## Ethics summary

Signer video is never redistributed (ASL Citizen forbids it; ASLLRP is access-controlled; EmoSign
releases IDs and labels only) — we release code, weights, and labels. The inference server binds to
localhost with no authentication by default because it streams webcam-derived biometric data;
exposing it beyond loopback requires a token and TLS. No claim of clinical, legal, or
emergency-setting readiness. Full statement in `project_breakdown.md` §10.
