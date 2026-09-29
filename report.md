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
| 2 | M4 gate decision | | `NOT STARTED` | | |
| 3 | BU non-manual XML request | | `NOT STARTED` | | |
| 4 | DWPose corpus download | | `NOT STARTED` | | |
| 5 | M7 preference study | | `NOT STARTED` | | |

**Project state at the time this form was written** — so a later reader can tell
whether a number has moved:

- Test suite: **306 passing**, lint and typecheck clean, `make serve-check` green.
- Gates closed: M0, M1, M2. M4 **not met** (worst-fold cross-AUC 0.7276 against a
  target of 0.60, with the signer control passing at 0.9729).
- M5a: **negative result** — WER 0.916 against a most-frequent baseline of 0.916.
- Known open defect: `export_glb()` writes a JSON parameter file, not a GLB mesh,
  despite its docstring. Not a valid mesh export.

---

## Step 0 — Rotate the exposed password

**Why:** a sudo password was pasted into a working chat session and must be treated as
compromised. The password itself is not recorded in any project file.

**Do not write the password, old or new, anywhere in this file.**

| Field | Entry |
|---|---|
| Owner | |
| Date started | |
| Status | `NOT STARTED` |
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
| Status | `NOT STARTED` |
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
| Status | `NOT STARTED` |

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

**Decision**

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
pseudo-label**. The authoritative annotations are distributed by Boston University under
a data-use agreement. The free mirror has glosses and pose but **no non-manual
annotations** — that is all it contains, not an oversight.

| Field | Entry |
|---|---|
| Owner | |
| Date sent | |
| Status | `NOT STARTED` |
| Verified by | |

**Results**

- [ ] Contact identified (name, institution): ______________________
- [ ] Request sent, stating purpose, non-redistribution, and citation
- [ ] Email saved to `paper/provenance/bu_access_request.md` **with the date**
- [ ] `implimentation.md` Track B line updated

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
| Status | `NOT STARTED` |
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

| Field | Entry |
|---|---|
| Owner (study coordinator) | |
| Date | |
| Status | `NOT STARTED` |
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
