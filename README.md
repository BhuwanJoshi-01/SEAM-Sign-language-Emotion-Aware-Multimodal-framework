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
| **[plan.md](plan.md)** | The executable checklist: 5 phases, 16 tasks, tests, definitions of done, KPI dashboard, critical path | You want to know *what to do next* |
| **[team.md](team.md)** | Squad roles (R1–R4), 10-week sprint calendar, RACI, review gates, escalation paths | You want to know *who owns what* |
| **[paper/PAPER_TEMPLATE.md](paper/PAPER_TEMPLATE.md)** | Section-by-section paper scaffold with word budgets and evidence requirements | You are writing |
| **[paper/main.tex](paper/main.tex)** | IEEEtran skeleton with every result table pre-stubbed | You are writing |
| **[paper/WRITING_GUIDE.md](paper/WRITING_GUIDE.md)** | House style, terminology rules, honesty patterns | You are writing |
| **[paper/CLAIMS_LEDGER.md](paper/CLAIMS_LEDGER.md)** | Every claim → evidence → run ID → verdict, plus claims we explicitly refuse to make | You are about to assert something |
| **[paper/EXPERIMENT_LOG.md](paper/EXPERIMENT_LOG.md)** | Append-only record of every run behind every number | You just ran something |
| **[paper/refs.bib](paper/refs.bib)** | Bibliography seeded with the full literature survey | You are citing |
| [signemotionaware.md](signemotionaware.md) | The original concept note, kept for provenance | Historical interest |

## The four numbers that are the paper

| KPI | Reference | Target |
|---|---|---|
| EmoSign emotion wF1, video-only, leave-one-signer-out | GPT-4o: **20.76** | **≥35** |
| Disentanglement cross-prediction AUC | 0.5 = perfect separation | **≤0.60** |
| End-to-end p95 latency on RTX 3050 | — | **<400 ms** |
| Peak inference VRAM | 4096 MB hard limit | **<2500 MB** |

Baselines were evaluated on an 80 GB A100. We target 4 GB.

## Timeline

10 weeks, Mon 2026-08-10 → Sun 2026-10-18. Five phases: Foundations → Recognition → Affect →
Contribution → Ship. Phase gates every second Friday. Full detail in `plan.md`.

## The single most urgent action

**Submit the ASLLRP data-access request on day 1.** ASLLRP supplies both the source video and the
*linguistic non-manual annotations* for the same 200 utterances that EmoSign labels with affect —
the only place in existence where both label types coexist. It gates the core contribution.

## Known data caveat

`/home/bhuwan/Videos/data/` is the **INCLUDE** dataset (Indian Sign Language, 4,287 videos, 263
signs), **not WLASL**, and 12 of its archives are truncated `.part` downloads — including
`Train_Test_Split`. It is now scoped as an optional cross-lingual ablation. See
`project_breakdown.md` §5.2.

## Ethics summary

Signer video is never redistributed (ASL Citizen forbids it; ASLLRP is access-controlled; EmoSign
releases IDs and labels only) — we release code, weights, and labels. The inference server binds to
localhost with no authentication by default because it streams webcam-derived biometric data;
exposing it beyond loopback requires a token and TLS. No claim of clinical, legal, or
emergency-setting readiness. Full statement in `project_breakdown.md` §10.
