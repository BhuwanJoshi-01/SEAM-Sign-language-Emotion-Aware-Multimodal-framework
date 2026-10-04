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
| **[PROJECT_BREAKDOWN.md](PROJECT_BREAKDOWN.md)** | **The map: what each stage is, what is measured, what works, what does not, and what is next** | You want to know where the project actually is |
| **[project_breakdown.md](project_breakdown.md)** | The original technical bible: motivation, literature synthesis, datasets, architecture, losses, evaluation protocol, risks, ethics, glossary. **Describes the plan, not the current state.** | You want to understand *why* |
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

> Two files here are named `project_breakdown.md` and `PROJECT_BREAKDOWN.md`. They differ only
> by case and disagree completely; on a case-insensitive filesystem they collide. Read
> **`PROJECT_BREAKDOWN.md`** for the current state and the other one for the original design.

## How to run it

```bash
make setup            # editable install + pre-commit hooks (once)
make doctor           # GPU, VRAM, MediaPipe, ffmpeg, disk, network
make test             # the whole suite

make serve            # http://127.0.0.1:8000 - two pages:
#   /         webcam demo; MediaPipe runs in the browser, video is never uploaded
#   /avatar   SMPL-X avatar viewer (needs `make demo-avatar` first)
make serve-check      # boots the server and asserts the HTTP contract of both pages

make demo-avatar      # video -> SMPLer-X -> SMPL-X mesh -> animated .glb + .mp4 + manifest
                      # ~15 min on the GPU; publishes a `status` while it runs

make readiness        # the M0 gate: every dataset, its real state, and a named owner
make wlasl-index      # WLASL on-disk index, incl. HTML-placeholder substitutions
make fetch-how2sign   # 31 How2Sign pose shards, 14.1 GB, resumable
```

`/avatar` loads the animated GLB in three.js. **three.js is vendored** under
`src/seam/web/vendor/three` and served from `/static/vendor/`, so the page needs neither
network nor a CDN. The licence-gated SMPL-X parameters are *not* vendored; point `SMPLX`
at your own copy:

```bash
make demo-avatar SMPLX=/path/to/SMPLX_NEUTRAL.npz
```

## The web front end

Four pages, one shared header, reachable from each other:

| Page | What it is |
|---|---|
| `/` | webcam demo — MediaPipe runs in the browser, video is never uploaded |
| `/avatar` | animated skinned glTF beside the per-clip measurements |
| `/api/coverage` | what the system does and does not claim, generated from the code |
| `/routes` | every route, probed live, with status and byte count |

**Light and dark, both designed.** The default follows your system preference; the toggle
overrides it and is remembered. `?theme=light` / `?theme=dark` forces one — the M7 study
harness needs it so every rater sees the same stage.

All colour, type, space and depth live in `src/seam/web/tokens.css`, which is why no page
carries a palette of its own. Contrast was measured rather than eyeballed: **body text clears
4.5:1 on every surface in both themes**, and the one value that failed (light `--ok` on its
own tint, 4.44:1) was corrected rather than tolerated. Text selection, focus rings,
scrollbars, placeholder text and tabular numerals are themed too — browser defaults belong to
no design system.

`make serve-check` asserts that all four pages load the token sheet and the pre-paint theme
boot snippet, and it fails on the patterns that were actually removed: a coloured
`border-left` on a callout, and monospace set on body text as a costume.

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

**Read this table, not the one below it.** It is kept in `plan.md` §4 as the KPI dashboard.

| Milestone | Gate | State | Evidence |
|---|---|---|---|
| **M0** data spine | readiness table · 200/200 clips · face gate · tests green | **REGRESSED to OPEN** | `make readiness` reports **6/10 resources**; see the note below |
| **M1** confound audit | measured FER bias, 2+ models, CIs | **met — hypothesis REFUTED** | 2,565 clips, 3 FER models, −0.008…+0.002, MDE 0.003–0.008 |
| **M2** efficiency | 3050 p95 + peak VRAM, CI-enforced | **met, K6 marginal** | 73.2 ms p95 · 186 MB for 6 live models · 20.4 FPS (repeats 19.7–20.4) |
| **M3** `L` labels | marker labels with provenance + cue correlation | **met** | 43,038 human non-manual annotations, 200/200 joined; negation ↔ head-shake r=0.554 |
| **M4** factorized encoder ★ | cross-pred AUC ≤0.60, no affect loss, positive control | **REFUTED — contribution WITHDRAWN** | worst-fold cross-AUC 0.7276 (heuristic `y_L`) → 0.7031 (human `y_L`), vs ≤0.60. Signer control 0.973 passes, so the failure is the model's |
| **M5a** gloss recognition | WER beats its baseline | **negative** | WER 0.916 = most-frequent baseline 0.916 (shuffled 0.911) |
| **M5b** translation | How2Sign BLEU-4 at ≤80M params | **not started** | `how2sign_mediapipe_pose` not fetched |
| **M6** conditioned generation | style acc ≥80% @ BERTScore ≥0.90 | **not started** | — |
| **M7** avatar + live demo | preference ≥60%, browser smoke test | **avatar working; study blocked on 0 raters** | animated skinned glTF, verified against a real glTF runtime |
| **M8** cross-lingual | zero-shot Top-1 vs chance | **not started** | NSL provenance/terms not established |
| **M9** paper + release | `repro_all.sh` green, submission filed | **not started** | `paper/main.pdf` predates every result in the repo |

**What this means, stated plainly.** Two of the plan's hypotheses were measured and refuted
(M1's marker-induced FER bias, M4's factorisation), and the core contribution as originally
framed was withdrawn. What survives is a **negative-results contribution with sound
instruments**: two refuted hypotheses, a measured result on how unreliable heuristic
pseudo-labels are in this domain, a reusable adversarial/probe harness that passes its own
positive control, and a working SMPL-X avatar system. See `PROJECT_BREAKDOWN.md`.

**M0 closed 2026-09-26, and has since regressed** — but for one reason only, and it was
three false negatives in the gate rather than lost data:

| resource | reported before | actually on disk |
|---|---|---|
| `wlasl_local` | "reuse path absent" | **3,863 clips, 7.43 GB** at the very path named |
| `rafdb_mediapipe` | "not fetched yet" | **2.72 GB, 6/6 shards readable** (14,329 train rows) |
| `how2sign_mediapipe_pose` | "not fetched yet" | genuinely absent; 31 shards, 14.12 GB, ungated — `make fetch-how2sign` |

The dataset spine still resolves: the EmoSign join is **200/200** and the face-visibility gate
passed **24/24** sampled clips (face in 94–100% of frames, median 30 of 52 blendshapes with
real temporal variance). What remains open is ASL Citizen (81 GB, reviewer-owned, not
started) and NSL provenance, plus 4 of 3,863 WLASL clips that are genuinely lost.

**WLASL's recorded defect was wrong.** `sources.py` said *"untrimmed `.part` downloads with no
`moov` atom"*, needing ffmpeg re-cuts, and a test asserted that guess was still recorded. An
ffprobe sweep of all 3,863 files found 92 undecodable, every one of them 813 KB of **YouTube
HTML** saved as `0.mp4` — zero moov faults. 88 of the 92 have a sibling
`N_yt.mp4.part.mp4` that decodes, so the repair is a substitution and `wlasl.index_on_disk`
now performs it: **3,771 usable clips**, up from the ~2,565 the broken indexer reported. See
`paper/EXPERIMENT_LOG.md`.

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
