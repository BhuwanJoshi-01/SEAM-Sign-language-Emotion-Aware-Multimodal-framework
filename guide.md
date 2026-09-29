# SEAM — Contributor Guide

**Read this first.** These are the tasks that cannot be completed by the code alone.
Everything here is deliberately explicit: each step says what to type, what you should
see, and what to send back. If a step does not produce the stated output, **stop and
report** rather than working around it — several of these steps have "success" states
that look like failure, and a silent workaround produces a result that is wrong rather
than a result that is missing.

Every command below assumes you are in the project directory and using the project's
Python environment. Verify that first:

```bash
cd /mnt/Volume2/Sign_Language_EmotionAware
/home/bhuwan/miniconda3/envs/slr/bin/python --version   # must print Python 3.12.x
```

If that path does not exist, the environment is not set up. Stop and report; do not
install a second Python.

---

## Contents

| # | Task | Who | Time | Blocks |
|---|---|---|---|---|
| 0 | [Rotate the exposed password](#0-rotate-the-exposed-password-15-minutes) | whoever owns the machine | 15 min | security |
| 1 | [Obtain the SMPL-X model weights](#1-obtain-the-smpl-x-model-weights-1-3-hours) | you (licence form) | 1–3 h | the avatar rendering a real body |
| 2 | [Decide what happens to the M4 result](#2-decide-what-happens-to-the-m4-result-30-minutes) | you (a judgement call) | 30 min | the paper's headline claim |
| 3 | [Request SignStream non-manual XML from BU](#3-request-signstream-non-manual-xml-from-bu-20-minutes) | you (institutional email) | 20 min + waiting | M3's instrument |
| 4 | [Download the DWPose pose corpus](#4-download-the-dwpose-pose-corpus-20-minutes) | anyone | 20 min | M5a at real scale |
| 5 | [Run the M7 human preference study](#5-run-the-m7-human-preference-study-1-2-days) | 5+ people, incl. a Deaf signer if possible | 1–2 days | the last M7 gate |

**Do these in this order.** 0 and 4 are quick and unblock other work. 1 and 3 involve
waiting on other people, so start them as early as possible. 2 needs nothing but your
judgement and unblocks the writing.

---

## 0. Rotate the exposed password (15 minutes)

### Why

A sudo password was pasted into a working chat session while setting up this project.
It is therefore in the chat transcript, in any chat export, and in this project's
context history. It must be treated as compromised and changed. **The password itself is
deliberately not repeated anywhere in this file.**

### Steps

1. Change the password:

   ```bash
   passwd
   ```

   Enter the current password once, then the new one twice. You will not see characters
   as you type — this is normal.

2. Confirm the old password no longer works:

   ```bash
   sudo -v
   ```

   Type the **new** password when prompted. It should succeed with no output.

3. Confirm the old one fails:

   ```bash
   sudo -v
   ```

   Type the **old** password. It must be rejected with `Sorry, try again.` If the old
   password still works, `passwd` did not take effect — stop and report.

4. Also change it anywhere else it was reused. If the same password was used for Git,
   a database, or an academic account, change those too.

### Done when

`sudo -v` succeeds with the new password and rejects the old one, and no other service
still accepts the old password.

---

## 1. Obtain the SMPL-X model weights (1–3 hours)

### Why

The avatar retargets MediaPipe landmarks onto a real human body, and the model that
defines a human body is **SMPL-X**, licensed by the Max Planck Institute. The model
files are therefore *deliberately not in this repository* — committing them would
breach the licence. Instead the code accepts a path you supply.

Right now the avatar computes correct **SMPL-X parameters** (body pose, MANO hand
angles, FLAME expression coefficients) and they are all unit-tested. What is missing is
the model that turns those parameters into geometry. Until this step is done, there is
no verified visual output and no real 3D export.

> **Honest warning before you start.** The current `export_glb()` function in
> `src/seam/avatar/synthesis.py` does **not** write a real GLB mesh, despite its
> docstring saying it does. It writes a JSON file of SMPL-X parameters to a path ending
> in `.glb`. `trimesh` is not currently installed either. Do not report the avatar as
> "exporting a mesh" after this step — the exporter still needs to be implemented, and
> the parameter file is not a mesh. This is a known open item, listed in
> `implimentation.md`.

### Steps

1. **Register and accept the licence.** Go to the official SMPL-X page linked from
   <https://smpl-x.is.tue.mpg.de> (the "Model Download" / registration section). You
   must fill in a form stating you are a non-commercial researcher and accept the terms.
   **This is the slow part — approval is manual and is not instant.** Expect a delay.

2. **After approval**, download the model. You need the main SMPL-X model file for your
   operating system (Linux). The download is a `.npz` (or `.pkl` for the body-only
   model). Record the exact filename you get — it is usually something like
   `SMPLX_NEUTRAL.npz` or `SMPLX_MALE.npz`.

3. **Check the file is real, not an error page.** A failed download often saves an HTML
   page with a `.npz` name:

   ```bash
   ls -lh ~/Downloads/*.npz
   file ~/Downloads/SMPLX_NEUTRAL.npz
   ```

   `file` must report a NumPy/zip archive. If it says `HTML document` or `ASCII text`,
   the download failed — delete it and try again.

4. **Move it somewhere permanent and out of the repository.** Never put it in the repo
   even temporarily:

   ```bash
   mkdir -p /home/bhuwan/models/smplx
   mv ~/Downloads/SMPLX_NEUTRAL.npz /home/bhuwan/models/smplx/
   chmod 600 /home/bhuwan/models/smplx/SMPLX_NEUTRAL.npz
   ```

   `chmod 600` matters: the licence forbids redistribution, and making the file
   owner-only reduces the chance of it being copied somewhere public.

5. **Confirm the code can read it.** Run:

   ```bash
   cd /mnt/Volume2/Sign_Language_EmotionAware
   /home/bhuwan/miniconda3/envs/slr/bin/python -c "
   from pathlib import Path
   from seam.avatar.synthesis import load_smplx
   d = load_smplx(Path('/home/bhuwan/models/smplx/SMPLX_NEUTRAL.npz'))
   print('keys:', len(d), '->', sorted(d)[:8])
   "
   ```

   **Expected:** a line beginning `keys:` with a plausible count and a list of parameter
   names (things like `v_template`, `shapedirs`, `J_regressor`, `kintree_table`).

   **If it raises** `FileNotFoundError`, the path is wrong. **If it raises**
   `ValueError: unsupported model file`, you have a `.zip` or `.tar.gz` — unzip it
   first and point at the inner `.npz`.

6. **Confirm it is still not in the repository.** This must all print nothing:

   ```bash
   cd /mnt/Volume2/Sign_Language_EmotionAware
   git status --porcelain | grep -i smpl || echo "clean: nothing model-related staged"
   ```

### Done when

- You have a licence-gated `.npz` at `/home/bhuwan/models/smplx/`, mode `600`.
- `load_smplx()` on that path prints a `keys:` line.
- Nothing model-related appears in `git status`.

### Send back

The path to the file and the output of step 5. **Do not send the file itself**, and do
not commit it. Then the real exporter can be implemented against it.

---

## 2. Decide what happens to the M4 result (30 minutes)

### Why

M4 is the paper's **core contribution** — a factorised encoder that is supposed to hold
linguistic (`z_L`) and affect (`z_A`) information apart. It was given an explicit,
pre-registered gate before the experiment was run, and **it failed it.** This is a
research judgement, not a coding problem, and it is yours to make.

The gate is defined in `plan.md` and the result is in
`artifacts/m4/factorizer_multilabel.json`. Read the `gate` block:

```
separation_target  0.6      worst_cross_auc  0.7276
control_min        0.8      signer_control   0.9729
gate_passed        false    folds_excluded   1
```

**What the numbers mean, plainly:**

- `worst_cross_auc = 0.728` against a target of `0.60`. AUC of 0.5 is chance. So affect
  is leaking into the linguistic factor (or the reverse) **noticeably more than the gate
  allows** on the worst fold.
- The **positive control passes strongly** (`0.973`): the model can identify which
  signer produced an utterance far better than chance. So this is not a broken model or
  a broken feature pipeline — the instrument works, and it is measuring a real leak.
- One fold (Ben) was **excluded for insufficient label support**, so the gate was
  computed on the three remaining signers. That exclusion is legitimate but reduces the
  evidence base.
- Cross A→L sits at **0.505**, i.e. chance. The leak is one-directional: affect is
  readable from the linguistic factor, not the other way.

### Your options

**(a) Reframe as a negative result — this is the recommended path.**
Report that the factorisation *does not hold* on this corpus at this scale, with the
signer control proving the instrument was sound. This is a real, publishable, honest
finding, and it pairs naturally with M1 (which likewise refuted its own hypothesis
C1 with a measured minimum detectable effect). The paper becomes a careful negative
result with a validated instrument rather than a positive claim that does not reproduce.

**(b) Relax the gate with a stated, pre-hoc justification.**
Permissible, but you must state *why* 0.60 was the right bar and why 0.728 is now
acceptable, and it must be written as a limitation rather than presented as a pass. A
gate moved to fit a number it did not meet is the failure mode this project has
otherwise been careful to avoid.

**(c) Collect more data and re-run.**
More signers and more affect-labelled windows. Honest, but slow, and the M5a result
suggests the underlying data density is a general constraint.

**(d) Drop the factorisation claim** and report only the validated instrument (the
signer control and the GRL/adversarial machinery) as a reusable contribution.

### Steps

1. Read `plan.md`, the M4 section, and the "M4 — GATE NOT MET" status line.
2. Read the `gate` block in `artifacts/m4/factorizer_multilabel.json` and confirm the
   numbers above match what is written there.
3. Choose (a), (b), (c) or (d).
4. Write the decision into `plan.md` as a dated entry under the M4 heading, and state
   the chosen gate status.

### Done when

`plan.md` contains a dated decision, and it states plainly whether M4 is now reported as
a pass, a refuted hypothesis, or a deferred gate. Do **not** edit the measured numbers —
they are the measurement, and they are correct.

---

## 3. Request SignStream non-manual XML from BU (20 minutes + waiting)

### Why

Every non-manual feature in this project — brow raise, brow furrow, head shake, mouth
morpheme — currently comes from **heuristic pseudo-labels** derived from MediaPipe
blendshapes. That is stated in the artifacts (`marker_provenance: "heuristic
(pseudo-labels); ASLLRP SignStream XML not available"`). It is the single largest
instrument limitation in the project, and it is why M3's gate says "met on labels,
blocked on the visual instrument."

The authoritative labels are the **ASLLRP SignStream non-manual XML**, which is
distributed by Boston University under a data-use agreement. The freely downloadable
Hugging Face mirror contains the **gloss** tokens and DWPose pose, but **no
non-manual annotations** — that is not an oversight in the mirror, it is what the
mirror contains.

### Steps

1. Find the contact. The dataset is "ASLLRP" (American Sign Language Lexicon-Reinforced
   with Prosodic Features). The distribution is handled by Boston University's
   Language and Communication Lab / speech-and-language processing group. Look for the
   dataset request link on the ASLLRP project page or on the related publications
   (search `ASLLRP non-manual annotation XML download`). If the original authors no
   longer distribute it, the current maintainer will be listed on the dataset page.

2. **Email, in your own words, stating all of the following:**
   - Who you are (name, institution or "independent researcher", contact).
   - That you are requesting the **SignStream non-manual XML annotations** for
     ASLLRP — specifically the facial/body non-manual labels, not the glosses, which
     you already have.
   - The **research purpose**, in one or two sentences: measuring how facial
     non-manuals and affect markers interact in ASL, for a reproducibility study.
   - That you will **redistribute nothing** and will cite the dataset.
   - That you can sign the data-use agreement if required.

3. **Save the request.** Save the sent email and any reply into
   `paper/provenance/bu_access_request.md` (create the directory) with the date. A
   provenance record that lives only in someone's inbox is not a provenance record.

4. **Record the status** in `implimentation.md` under the Track B line, as either
   "requested YYYY-MM-DD, awaiting reply" or "declined YYYY-MM-DD, reason". If it is
   declined or unanswered after two weeks, say so — the M3 limitation then stands as
   final and should be written up as a limitation rather than left as an open action.

### Done when

`paper/provenance/bu_access_request.md` exists with the date and outcome, and
`implimentation.md` reflects it.

**This step cannot be hurried.** Start it now, do the other work while it is pending.

---

## 4. Download the DWPose pose corpus (20 minutes)

### Why

M5a was run and the result is **negative**: a pose recogniser scored WER 0.916 against a
most-frequent baseline of 0.916 — indistinguishable from always predicting the most
common gloss. That is a *data-scale* verdict, not proof that pose is uninformative. The
test corpus had **1,563 usable tokens over 499 glosses — 3.1 tokens per gloss, with 284
of the 499 glosses appearing exactly once.** A third of one signer's tokens used a gloss
the other three signers never used.

The corpus available fixes exactly that problem: it contains **49,083 DWPose pose
files, frame-aligned to the utterance crops by construction**, across ~100× more
utterances. That is the fair test of whether signing-space pose carries lexical signal.

**This is much smaller than it sounds: the whole download is about 1.2 GB** and you have
36 GB free. It is the cheapest high-value task in this list.

### Steps

1. **Check your disk first.**

   ```bash
   df -h /mnt/Volume2
   ```

   You need at least 5 GB free. If it reports less than 5 GB, stop and report — do not
   delete anything to make room without asking.

2. **Download the bundle.** It is a single tar at the top level of the dataset repo:

   ```bash
   mkdir -p /mnt/Volume2/asllrp_dwpose
   cd /mnt/Volume2/asllrp_dwpose
   curl -L -o ASLLRP_utterances_results.tar \
     "https://huggingface.co/datasets/FangSen9000/ASLLRP_utterances_results/resolve/main/ASLLRP_utterances_results.tar"
   ```

   This is roughly 1.2 GB and may take several minutes. Watch the progress bar.

3. **Verify the download is complete and not an error page.** Expected size is about
   1,169,520,640 bytes:

   ```bash
   ls -l ASLLRP_utterances_results.tar
   stat -c %s ASLLRP_utterances_results.tar
   ```

   The number must be exactly `1169520640`. If it is much smaller, the download was
   interrupted — delete the file and repeat step 2. **If it is 0 bytes or a few hundred
   bytes, you were rate-limited or blocked; report that instead of retrying in a loop.**

4. **Check the checksum against the source** (the API reports it as `lfs.oid`):

   ```bash
   curl -s "https://huggingface.co/api/datasets/FangSen9000/ASLLRP_utterances_results" \
     | /home/bhuwan/miniconda3/envs/slr/bin/python -c "
   import json,sys
   d=json.load(sys.stdin)
   s=[x for x in d['siblings'] if x['rfilename'].endswith('.tar')]
   print('lfs oid:', s[0].get('lfs',{}).get('oid') if s else 'not reported')
   "
   ```

   The `lfs oid` is the SHA256 of the file. If you want to verify:
   `sha256sum ASLLRP_utterances_results.tar` should match. **A mismatch means a corrupt
   download — delete and re-fetch, do not use the file.**

5. **Extract.** Extracting roughly doubles disk usage, so you need ~2.5 GB:

   ```bash
   cd /mnt/Volume2/asllrp_dwpose
   tar -xf ASLLRP_utterances_results.tar
   ls
   ```

   **Expected:** a directory named `ASLLRP_utterances_results/`.

6. **Confirm the pose files are really there and in the expected layout:**

   ```bash
   cd /mnt/Volume2/asllrp_dwpose/ASLLRP_utterances_results
   ls | head
   find . -path '*results_dwpose/npz*' -name '*.npz' | wc -l
   ```

   **Expected:** a large count — around 49,000 — of **`.npz`** files (note: `.npz`, not
   `.npz`), under paths shaped exactly like
   `ASLLRP_utterances_results/<utterance_id>/results_dwpose/npz/00000001.npz`.
   If the count is 0, the tar did not contain what is expected — report it.

7. **Inspect one file to find out what layout it actually uses.** The project's own
   feature extractor assumes 18 body + 68 face + 21 left-hand + 21 right-hand keypoints
   per frame, but **that has not been verified against this corpus** — do not treat it
   as a known quantity. Establish the real layout:

   ```bash
   cd /mnt/Volume2/asllrp_dwpose/ASLLRP_utterances_results
   F=$(find . -path '*results_dwpose/npz*' -name '*.npz' | sort | head -1)
   echo "file: $F"
   /home/bhuwan/miniconda3/envs/slr/bin/python -c "
   import numpy as np
   with np.load('$F', allow_pickle=True) as d:
       print('arrays:', d.files)
       for k in d.files:
           print(' ', k, d[k].shape, d[k].dtype)
   print('body+face+hands if 18+68+21+21 =', 18+68+21+21)
   "
   ```

   **Report the printed shapes verbatim — do not interpret them.** The key question is
   whether some array has 128 keypoints (optionally +3 for a per-point score), or
   something else entirely. Whatever it turns out to be, the extractor in
   `src/seam/features/signpose.py` will need to match it, and a mismatch would otherwise
   produce wrong features silently rather than raising an error.

8. **Free the tar once extraction is verified** (it is reproducible, and it is large):

   ```bash
   rm /mnt/Volume2/asllrp_dwpose/ASLLRP_utterances_results.tar
   df -h /mnt/Volume2
   ```

   **Do not delete the extracted directory.**

9. **Record it.** Add a line to `implimentation.md` under M5a noting the corpus is on
   disk, its path, and the keypoint count you confirmed in step 7.

### Done when

- `/mnt/Volume2/asllrp_dwpose/ASLLRP_utterances_results/` exists.
- `find . -path '*results_dwpose/npz*' -name '*.npz' | wc -l` returns tens of thousands.
- You have reported the actual array shapes from step 7.

### Send back

The file count and the sampled `shape:` line. That is enough to start the scaled M5a.

---

## 5. Run the M7 human preference study (1–2 days)

### Why

The avatar demo is built and automatically gated, but its quality has only been checked
by the person who wrote it. The M7 gate requires a **blinded pairwise human preference
study**: independent people compare the avatar's output against a baseline and say
which they prefer. This cannot be self-certified, and it is the last M7 gate item.

There is also an ethical dimension specific to this project. It concerns **Deaf
signers**, and a study of ASL avatar quality run only by hearing people, or judged by
people who are not Deaf signers, does not establish anything about whether the work is
*good* — only that some raters preferred one video to another. That distinction has to
be written down either way.

### Prerequisites

- At least **5 independent raters**. More is better; 5 is the floor in the plan.
- **At least one Deaf signer** if at all possible. If none participates, the study is
  still runnable, but the write-up must state that explicitly and must not describe the
  result as a sign-language-community judgement.
- Raters must be **blind** to which system produced which video.

### Steps

1. **Start the demo and confirm it works:**

   ```bash
   cd /mnt/Volume2/Sign_Language_EmotionAware
   make serve
   ```

   Open the printed URL in a browser. You should see a camera view with a
   **"video never leaves the browser"** notice. **Confirm this notice is present** — it
   is a privacy property, not decoration: only landmark numbers are sent to the server,
   never video. Grant camera permission when prompted.

2. **Confirm the server contract still holds** before involving anyone else:

   ```bash
   make serve-check
   ```

   **Expected:** `serve smoke test passed`. If this fails, stop — do not run a study on
   a build whose contract is broken.

3. **Prepare the stimulus set.** For each clip to be judged, you need two renderings:
   - **A (system):** the current avatar pipeline.
   - **B (baseline):** the comparison condition agreed in `plan.md` — read the M7
     section for what baseline was specified and use exactly that, not one invented
     later.

   Give the two files **non-obvious, randomised names** (e.g. `x7f2.mp4` / `kq91.mp4`).
   Record the mapping in a private file the raters never see. Naming them "A" and "B"
   or revealing the condition is the single easiest way to invalidate the study.

4. **Write the rating sheet.** For each trial, capture:
   - Rater ID (anonymous, e.g. `R1`…`R5`).
   - Whether the rater is a Deaf signer (yes/no) and, optionally, first-language signer
     (yes/no).
   - Trial ID.
   - Chosen clip (the randomised name).
   - Optional free-text reason.
   - Any "the avatar's hands did not match what I signed" observations — these are the
     most valuable data in the study, because they point at specific retargeting bugs.

5. **Run the trials.**
   - Every rater sees the **same trials in the same order** (fixed order, randomised
     side assignment — this is a within-subject design).
   - Raters watch each pair once and record a choice. No discussion between raters
     before they finish, so answers stay independent.
   - Raters must not be told what the study is testing beyond "which clip looks more
     like natural signing".

6. **Record the blinding map only after every rater has finished.** Unblinding early
   destroys the design.

7. **Compute the results and check they mean something:**
   - For each trial, did raters agree? Report **inter-rater agreement** (e.g. Fleiss'
     kappa). Chance agreement on a forced choice is 0.0 kappa, and with 5 raters a
     3/2 split is common noise.
   - Report the **preference split per condition**, with the count of trials.
   - Report **how many raters are Deaf signers**, separately.

8. **Write it up honestly in `implimentation.md` under the M7 line** and append the
   numbers to `paper/EXPERIMENT_LOG.md`. Include:
   - Number of raters, and the Deaf-signer count.
   - Inter-rater agreement.
   - The preference result.
   - **An explicit statement of what the result does not establish.** With 5 raters and
     no Deaf signer, it is a weak preference signal and must be described as one.

   If the avatars do **not** differ meaningfully, write that. A null result from a
   properly blinded small study is a legitimate finding, and reporting it is what makes
   the study worth having done.

### Done when

- 5+ raters have completed the blinded trials, with at least one Deaf signer if
  obtainable.
- Inter-rater agreement and the preference split are computed and written down.
- `implimentation.md` and `paper/EXPERIMENT_LOG.md` record the result **and its
  limitations**, including the Deaf-signer count whatever it is.

---

## Reporting back

Send one message with:

| Step | Report |
|---|---|
| 0 | confirmation the old password is rejected |
| 1 | the model file path, and the `keys:` line |
| 2 | which option (a/b/c/d) you chose for M4 |
| 3 | the date sent, and the outcome if known |
| 4 | the `.npz` file count and the verbatim array shapes from step 7 |
| 5 | rater count, Deaf-signer count, inter-rater agreement, preference split |

**Never send the SMPL-X model file, and never commit it.** It is licence-restricted.
