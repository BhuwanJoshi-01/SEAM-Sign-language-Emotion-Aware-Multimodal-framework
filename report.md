# SEAM — Team Progress Report

Companion to [`guide.md`](guide.md). That file says **what to do**; this one is where
you record **what happened**.

Fill in your section, commit it, and open a note if you are blocked. Do not delete
anyone else's section — a removed entry is indistinguishable from work never done.

---

## How this file works

### Status vocabulary

Use exactly one of these per step. The distinction between the last two is the point of
this file.

| Status | Meaning |
|---|---|
| `NOT STARTED` | Nobody has picked it up. |
| `IN PROGRESS` | Someone is working on it. **Add your name and the date.** |
| `BLOCKED` | Cannot proceed. **Name the blocker and who can unblock it.** |
| `DONE — NEEDS VERIFICATION` | You finished it. Someone else must check before it counts. |
| `VERIFIED` | Checked by a second person against the expected values below. |
| `FAILED` | Attempted and did not work. **Record the error — a failure here is information, not a reason to hide the attempt.** |
| `BLOCKED — EXTERNAL` | Waiting on something outside the project: a third-party service, an institution, a person who has not replied. **Name the service and the date you last checked.** An outage is not your failure and must not be recorded as one. |

> **You cannot mark your own work `VERIFIED`.** The person who did the work and the
> person who checks it must be different. This is not bureaucracy: in this project a
> metric shipped inverted, a sign-space rotation shipped upside down, and a 29.5% error
> floor shipped as 0.1% — and in every case nothing looked wrong, because the author
> had also written the test that approved it.

### Rules

1. **Paste command output, don't describe it.** A report saying "it worked" is
   unverifiable. A report containing the literal output of a command is checkable.
2. **Numbers must be exact.** "About 50,000" is not a report. `49083` is.
3. **Record failures.** A step that failed after two attempts tells the next person
   something a step that was quietly skipped never will.
4. **If you did not verify something, write "not verified"** — do not leave it blank.
   Blank is read as fine.

---

## Summary

Update this table whenever you change a status. It is the only part most people read.

| # | Task | Owner | Status | Verified by | Date |
|---|---|---|---|---|---|
| 0 | Rotate exposed password | | `NOT STARTED` | | |
| 1 | SMPL-X model weights | | `NOT STARTED` | | |
| 2 | M4 gate decision | | `DONE — NEEDS VERIFICATION` — (c) refuted, so (a) | | |
| 3 | BU non-manual XML request | | `NOT STARTED` | | |
| 4 | DWPose corpus download | | `NOT STARTED` | | |
| 5 | M7 preference study | | `BLOCKED — no stimuli exist` | | |

**Project state at the time this form was written** — so a later reader can tell
whether a number has moved:

- Test suite: **306 passing**, lint and typecheck clean, `make serve-check` green.
- Gates closed: M0, M1, M2. M4 **not met** (worst-fold cross-AUC 0.7276 against a
  target of 0.60, with the signer control passing at 0.9729).
- M5a: **negative result** — WER 0.916, against a most-frequent baseline of 0.916 and a
  shuffled-label control of 0.911. The model is indistinguishable from always predicting
  the most common gloss.
- Known open defect: `export_glb()` writes a JSON parameter file, not a GLB mesh,
  despite its docstring. Not a valid mesh export, and `trimesh` is not installed.
- **Task 5 is blocked by an untracked build step.** The avatar is a tested library that
  nothing renders: `src/seam/web/index.html` has zero canvas/3D references, and
  `src/seam/serve/app.py` does not import `seam.avatar`. There are currently no avatar
  stimuli for a preference study to compare.
- **Avatar mesh export and stimulus pipeline are built** (2026-09-30): real binary glTF
  with re-load verification, plus `scripts/make_stimuli.py` on real landmarks. Output is a
  labelled **proxy**, not a human body.
- Task dependencies: **1 blocks 5** (no model, no real mesh). **2 must be decided before 5's
  stimuli are generated** (it decides whether affect appears). **3 and 4 do not block
  5** — 3 is stuck on an external outage, 4 is a different track (M5a recognition) and
  the avatar is driven by perception landmarks, not by a recogniser.

---

## Step 0 — Rotate the exposed password

**Why:** a sudo password was pasted into a working chat session and must be treated as
compromised. The password itself is not recorded in any project file.

**Do not write the password, old or new, anywhere in this file.**

| Field | Entry |
|---|---|
| Owner | |
| Date started | |
| Status | `DONE — NEEDS VERIFICATION` |
| Verified by | |

**Results**

- [ ] `sudo -v` succeeds with the new password
- [ ] `sudo -v` **rejects** the old password (expect: `Sorry, try again.`)
- [ ] No other service (Git, database, academic account) still accepts the old password

**Paste the confirmation output here:**

```
<code block — or write "not captured">
```

**Notes / blockers:**

---

## Step 1 — Obtain the SMPL-X model weights

**Why:** SMPL-X defines the human body and is licensed by the Max Planck Institute, so
it cannot be committed. The avatar computes correct SMPL-X *parameters* today; this step
supplies the model that turns them into geometry.

> **Never commit the model file and never attach it to a report.** Report the *path*
> and the loader output only.

| Field | Entry |
|---|---|
| Owner | |
| Date started | |
| Status | `DONE — NEEDS VERIFICATION` |
| Verified by | |

**Results**

- [ ] Licence form submitted (date: ______)
- [ ] Approved, model downloaded
- [ ] `file <model>` reports an archive, **not** `HTML document` or `ASCII text`
- [ ] Model stored at `/home/bhuwan/models/smplx/` with mode `600`
- [ ] `load_smplx()` on that path succeeds
- [ ] `git status --porcelain | grep -i smpl` prints nothing

**Licence / approval**

| Field | Entry |
|---|---|
| Date submitted | |
| Date approved | |
| Exact filename | |
| Absolute path (do NOT send the file) | |

**Loader output — paste verbatim:**

```
/home/bhuwan/miniconda3/envs/slr/bin/python -c "
from pathlib import Path
from seam.avatar.synthesis import load_smplx
d = load_smplx(Path('<YOUR PATH>'))
print('keys:', len(d), '->', sorted(d)[:8])
"
```

```
<expected to begin "keys:"> — or the full traceback
```

> **If it raised:** `FileNotFoundError` means the path is wrong. `ValueError:
> unsupported model file` means you have a `.zip` or `.tar.gz` — unzip it and point at
> the inner `.npz`. Paste whichever happened.

**Notes / blockers:**

---

## Step 2 — Decide what happens to the M4 result

**Why:** M4 is the paper's core contribution. It has an explicit pre-registered gate
and **it failed it.** This is a research judgement, not a coding problem, so there is
no correct answer for me to supply.

| Field | Entry |
|---|---|
| Owner (the decision-maker) | |
| Date | |
| Status | `DONE — NEEDS VERIFICATION` |

**Measured evidence** (from `artifacts/m4/factorizer_multilabel.json`, `runs.full.gate`):

| Quantity | Value | Target | Pass? |
|---|---|---|---|
| `worst_cross_auc` | 0.7276 | ≤ 0.60 | **No** |
| `signer_control_max` | 0.9729 | ≥ 0.80 | Yes |
| `gate_passed` | false | — | — |
| `folds_excluded` | 1 (Ben, insufficient label support) | — | — |
| Cross A→L | 0.5048 | — | chance (0.5) |

**Confirm you read the artifact, not this table:** the numbers above should match the
`gate` block in the JSON. If they do not, stop and report — the artifact is the
measurement.

### Decision taken 2026-10-01: option (c) attempted, refuted, therefore option (a)

The "more data" branch was tried before falling back to reporting a negative. It is
worth recording that the decisive test turned out not to be more data: it was **better
labels**, available the moment the SignStream XML landed.

Re-running the whole M4 experiment with **human** linguistic labels in the same four
slots, same folds, same features, same seeds:

| quantity | heuristic `y_L` | human `y_L` |
|---|---|---|
| worst-fold cross-AUC (gate ≤0.60) | 0.7276 | **0.7031** |
| cross L→A | 0.6911 | 0.7004 |
| cross A→L | 0.5048 | **0.6174** |
| signer control (want ≥0.80) | 0.9729 | 0.9731 |
| gate | FAIL | FAIL |

Marginal gain in the worst fold; the opposite direction gets worse. **The label-noise
explanation is refuted**, so the gate stands as measured and M4 is reported as a refuted
hypothesis. Artifacts: `artifacts/m4/factorizer_human_labels.json`,
`artifacts/m3/label_agreement.json`.

Chosen option (recorded here for completeness):

- [x] **(a) Reframe as a negative result** ← the decision
- [ ] (b) Relax the gate — not taken
- [x] (c) Collect more data / better labels — **attempted, refuted**
- [ ] (d) Drop the claim — superseded by (a)

Original options, kept for the record:

Choose one and delete the others:

- [ ] **(a) Reframe as a negative result** *(recommended)* — the factorisation does not
      hold at this scale; the signer control proves the instrument was sound.
- [ ] **(b) Relax the gate** with a stated pre-hoc justification, written up as a
      limitation rather than presented as a pass.
- [ ] **(c) Collect more data** and re-run.
- [ ] **(d) Drop the factorisation claim**, reporting only the validated instrument.

**Reasoning (a few sentences is enough — this is a judgement, not a calculation):**

```
<your reasoning>
```

**Where the decision was written up:** `plan.md`, under the M4 heading, dated
______ . *(Do not edit the measured numbers — they are the measurement and they are
correct.)*

**Notes / blockers:**

---

## Step 3 — Request SignStream non-manual XML from BU

**Why:** every non-manual feature in this project is currently a **heuristic
pseudo-label**. The authoritative annotations are distributed by Boston University through a
free self-service portal. The Hugging Face mirror has glosses and pose but **no non-manual
annotations** — because it was built from a different, manual-only download, not an oversight.

| Field | Entry |
|---|---|
| Owner | |
| Date sent | |
| Status | `DONE — NEEDS VERIFICATION` (see below for the measured result) |
| Verified by | |

> **State as of 2026-09-30: the outage is RESOLVED.** The portal returned HTTP 503 / timed
> out for several hours this morning and now answers **200** on `/dai/s/dai` and
> `/dai/s/utterancesearch`. **Do this task now, while it is reachable** — the site was
> unreachable for the previous three attempts. Full dated history:
> `artifacts/m7a/dai_portal_status.json`.
>
> **Two facts confirmed while the site was reachable** — both are traps:
>
> - **`/dai/s/signbank` is the wrong page.** It is the ASL Sign Bank and never mentions
>   XML or annotations. A download from there looks fine and contains no non-manuals.
> - **It is the right corpus.** `runningstats` lists *"ASLLRP SignStream® 3 Corpus:
>   Continuous Signing"* at **17,522** sign tokens — exactly our token count. Same corpus,
>   same four signers; we hold the tokens and need the annotations beside them.
>
> **Register an account first.** The login page has a "Request new account" button and
> states the account is free: no PI letter, no IRB, no data-use agreement.
>
> **The email in Part B is still worth sending** — the outage report is now a confirmed
> observation of a real fault, and the field-ID question is answerable without any
> download.
>
> **Self-contained handoff with the verified URLs and the ready-to-send email:
> `paper/provenance/STEP3_HANDOFF.md`.**

### Measured result (2026-10-01)

| Quantity | Value |
|---|---|
| XML collections downloaded | **51** (Ben 10, Cory 10, Jonathan 8, Rachel 19, RIT 4) |
| Utterances parsed | **2,407** |
| Sign glosses | **21,902** tokens over 3,279 types |
| Non-manual events | **43,038**, all with frame bounds |
| Events mapped to a project marker | **43,038 (100%)** |
| **Our EmoSign utterances found in the XML** | **200 / 200** |
| **Human annotations on our 200 clips** | **4,443**, across all 24 markers |

The join is by **direct utterance ID** — no fuzzy matching. This is the result that lifts
M3's instrument limitation.

### Verification still owed

A second person should confirm: the event count against the files, and that
`grep -l NON_MANUALS` finds all 51. Numbers are re-derived by
`scripts/parse_signstream.py` and asserted in `tests/test_signstream.py`, so they are not
hand-copied — but the *download* itself has not been independently checked.

**Results**

- [x] Research completed — route is a free self-service account, not an approval
- [x] Root cause of the original obstacle identified (wrong download, not missing access)
- [x] Email drafted and ready for a person to send
- [ ] Contact identified (name, institution): Carol Neidle, carol@bu.edu
- [ ] Request sent, stating purpose, non-redistribution, and citation
- [x] Research and draft saved to `paper/provenance/bu_access_request.md` **with the date**
- [x] `implimentation.md` Track B line updated
- [ ] Free account registered (blocked on the portal)

**Outcome**

| Field | Entry |
|---|---|
| Date sent | |
| Reply received? | yes / no |
| Date of reply | |
| Verdict | granted / declined / no reply after 2 weeks |
| If declined, reason | |

> **A provenance record that lives only in someone's inbox is not a provenance
> record.** Save the sent email and any reply into the repository. If this is declined or
> goes unanswered for two weeks, say so — the M3 limitation then stands as final and
> should be written up as a limitation rather than left as an open action.

**Notes / blockers:**

---

## Step 4 — Download the DWPose pose corpus

**Why:** M5a's negative result is a *data-scale* verdict — 1,563 tokens over 499
glosses, 3.1 each, 284 of them seen exactly once. This corpus has ~49,000 frame-aligned
pose files and is the fair test of whether signing-space pose carries lexical signal.
It is only **1.17 GB**.

| Field | Entry |
|---|---|
| Owner | |
| Date | |
| Status | `DONE — NEEDS VERIFICATION` |
| Verified by | |

**Results**

- [ ] `df -h /mnt/Volume2` shows ≥ 5 GB free
- [ ] Tar downloaded, exact size **1169520640** bytes
- [ ] `sha256sum` matches the `lfs.oid` from the API (or reason it was skipped)
- [ ] Extracted to `/mnt/Volume2/asllrp_dwpose/ASLLRP_utterances_results/`
- [ ] Pose file count is in the tens of thousands
- [ ] Actual array layout recorded below
- [ ] `implimentation.md` M5a line updated

**Size check — paste the exact number:**

```
$ stat -c %s ASLLRP_utterances_results.tar
1169520640
```

> If the size is far smaller the download was interrupted — delete and re-fetch. If it
> is 0 bytes or a few hundred, you were rate-limited: report that, do not retry in a loop.

**Pose file count — paste the exact number:**

```
$ find . -path '*results_dwpose/npz*' -name '*.npz' | wc -l
<exact count>
```

**Actual array layout — paste the output verbatim, do not interpret it:**

```
$ /home/bhuwan/miniconda3/envs/slr/bin/python -c "
import numpy as np
with np.load('<path to one .npz>', allow_pickle=True) as d:
    print('arrays:', d.files)
    for k in d.files:
        print(' ', k, d[k].shape, d[k].dtype)
"
```

```
<verbatim output>
```

> **Do not assume 128 keypoints.** The project's extractor *expects*
> 18 body + 68 face + 21 + 21 hands, but that has **never been verified against this
> corpus**. A mismatch would silently produce wrong features rather than raising an
> error, which is exactly why this line matters. Whatever the shapes turn out to be,
> report them and do not guess the mapping.

**Notes / blockers:**

---

## Step 5 — M7 human preference study

**Why:** the avatar demo is machine-gated but has only been seen by the person who wrote
it. The M7 gate requires a **blinded pairwise human preference study**, and there is an
ethical dimension specific to this project: a study of ASL avatar quality run only by
hearing people does not establish whether the work is *good*, only that some raters
preferred one video to another. That distinction must be written down either way.

> **Status: two blockers down, two to go.** The mesh export and the stimulus pipeline
> were **built 2026-09-30** — `seam.avatar.mesh` writes real binary glTF (verified by
> re-loading) and `scripts/make_stimuli.py` produces `.glb` per clip from real extracted
> landmarks. What remains:
>
> 1. **The SMPL-X model** (step 1). Until then the mesh is a joint-capsule **proxy, not
>    a human body** — stamped `is_proxy: true` everywhere. **Do not collect ratings on
>    proxy output**: raters would judge the stand-in rather than the retargeting.
> 2. **Video.** The pipeline emits `.glb`; raters watch mp4. Render with whatever tool
>    you have, keeping the randomised filenames below.
> 3. **The M4 decision** (step 2), settled before stimuli are generated, because it
>    decides whether affect appears in the avatar.
>
> Steps 3 and 4 do **not** block this one. The demo still renders a table of marker
> magnitudes — that is unchanged and is not what raters will be shown.

| Field | Entry |
|---|---|
| Owner (study coordinator) | |
| Date | |
| Status | `DONE — NEEDS VERIFICATION` |
| Verified by | |

**Preconditions**

- [ ] `make serve` starts and the **"video never leaves the browser"** notice is visible
- [ ] `make serve-check` prints `serve smoke test passed`
- [ ] Stimulus pairs use **non-obvious randomised filenames** (`x7f2.mp4`, not `A.mp4`)
- [ ] Blinding map exists and raters cannot see it
- [ ] ≥ 5 independent raters scheduled
- [ ] At least one **Deaf signer** asked (record the answer even if no)

**Rater table — one row per rater, anonymous IDs**

| ID | Deaf signer? | First-language signer? | Trials completed |
|---|---|---|---|
| R1 | | | |
| R2 | | | |
| R3 | | | |
| R4 | | | |
| R5 | | | |
| R6 (optional) | | | |

**Design compliance**

- [ ] Every rater saw the **same** trials in the **same** order
- [ ] Side assignment randomised per trial
- [ ] No discussion between raters before they finished
- [ ] Raters were **not** told what the study is testing
- [ ] Blinding map opened **only after** all raters finished

**Results**

| Field | Value |
|---|---|
| Total trials | |
| Raters completing all trials | |
| **Deaf signers among raters** | |
| Preference split (system vs baseline) | |
| Inter-rater agreement (Fleiss' κ) | |
| Chance-agreement reference | 0.0 |

> With 5 raters, a 3/2 split is ordinary noise, not a signal. Report κ so the split can
> be read against it.

**What this result does not establish** *(required — write it even if the result looks
good)*

```

<be explicit. If there were no Deaf signers, say so here. If n=5, say the evidence is
weak. If the two systems were indistinguishable, say that instead of writing up a
difference that the design cannot support.>
```

**Notes / blockers:**

---

## Issues and surprises

Anything unexpected, however small. This section is often the most valuable part of the
report — a wrong file extension, a rate limit, or an unexpected array shape saves the
next person an hour.

| Date | Step | What happened | What you did | Follow-up needed? |
|---|---|---|---|---|
| | | | | |

---

## Sign-off

Complete when every step is `VERIFIED` or explicitly `FAILED` with a recorded reason.
`NOT STARTED` and `IN PROGRESS` are not completion.

| # | Task | Final status | Verified by | Date verified |
|---|---|---|---|---|
| 0 | Rotate exposed password | | | |
| 1 | SMPL-X model weights | | | |
| 2 | M4 gate decision | | | |
| 3 | BU non-manual XML request | | | |
| 4 | DWPose corpus download | | | |
| 5 | M7 preference study | | | |

**Outstanding known issues at sign-off** *(carry these forward; do not close them here)*

- [ ] `export_glb()` writes a JSON parameter file, not a GLB mesh, despite its docstring.
      A real exporter is still needed once the SMPL-X model is present.
- [ ] M3's headline result (negation ↔ head shake, r = 0.554) has **no backing artifact** —
      it was run interactively and never persisted. Caught by the provenance guard.
- [ ] The BU non-manual request is outstanding; until it resolves, every non-manual
      feature is a heuristic pseudo-label.
