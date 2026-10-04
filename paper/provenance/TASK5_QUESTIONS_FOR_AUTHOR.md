# Task 5 questions for the author

**Status: open. These are decisions only the study author can make.**
Written 2026-10-04 because `scripts/make_study_stimuli.py` and
`scripts/build_study_spec.py` both cite this file for the baseline condition, and it did
not exist. A script pointing at a missing document is a decision that will be made by
accident, which is precisely the outcome both scripts are written to prevent.

Nothing below has been decided. No preference result exists and none may be reported
until these are answered.

---

## Q1. What is condition B? *(blocks everything else)*

The study is a **pairwise forced choice**. Right now every stimulus is written under
condition `A` only — the current avatar — because the instruction was "for now lets not
think of baseline first lets make out".

**One arm cannot support a preference study.** With only condition A there is nothing to
prefer *over*: a forced choice between two renderings of the same system is not a
measurement of quality, it is a measurement of noise. Both scripts deliberately refuse to
invent a baseline, and this question is why.

Options, with the trade-off each buys:

| Option | Baseline | What it measures |
|---|---|---|
| **B1** | Proxy mesh (joint capsules) | Does the real body beat a stand-in? Tests the rendering pipeline, not the avatar's quality. Weak, but honest and already built. |
| **B2** | Current avatar, **no landmark retargeting** — rest pose plus a scripted motion | Does landmark-driven retargeting beat a generic motion? Isolates the contribution this project actually makes. |
| **B3** | A previous/naive retargeting | Does the current rig beat an older one? Needs a second pipeline. |
| **B4** | A commercial/off-the-shelf avatar system | The comparison a reviewer will ask for. Licensing and setup cost. |

**B2 is the most informative** for this project: the claim under test is that
landmark-driven SMPL-X retargeting improves avatar fidelity, and only a baseline isolates
that. **B1 is the cheapest** and is the only one that can be run today.

**Decide:** ______

---

## Q2. Which clips, and how many trials?

Currently 4 clips x 1 condition = 4 stimuli, which is a pipeline demonstration, not a
study. The corpus offers **1,354 utterance videos** (see `implimentation.md`, step 4).

Raters sustain attention for a limited number of trials, and agreement falls as the
design gets thinner per trial. **5 raters x N trials** with a fixed trial list is the
plan; N is a call about how long a rater can be asked to sit through.

**Decide:** N trials = ______, from how many clips = ______

---

## Q3. Raters — and is any of them a Deaf signer?

The floor is **5 independent raters**; inter-rater agreement must be reported, because
with 5 raters a 3/2 split is ordinary noise rather than a result.

The ethical point is separate and does not go away by being out of scope: **a study of ASL
avatar quality run only by hearing people, or judged by people who are not Deaf signers,
does not establish whether the work is good.** It establishes that some raters preferred
one video to another. Those are different claims and the write-up must distinguish them.

**Decide:** number of raters = ______ | Deaf signers among them = ______

---

## Q4. The avatar renders a seated signer as a standing figure — is that in or out?

**Measured 2026-10-04.** `scripts/measure_posture.py` reports: *"All signers are seated
(verified against real frames)"*. The renderer draws a standing body, and MediaPipe's leg
tracking on these clips is not usable — hip-to-knee distance ranges **0.06 to 1.42
torso-lengths** across 200 clips, which is noise rather than anatomy.

So the stimuli show a seated person standing upright, with legs that do not correspond to
what the signer did. Raters will see it. It is arguably the largest visible difference
between the avatar and the source video.

Options: render only the upper body (torso and above), which is where all the sign
language is anyway; keep the full body and state the limitation; or fix the leg
retargeting, which is real work and may not be possible with this tracker.

**Decide:** ______

---

## Q5. What may be claimed if the raters cannot tell the conditions apart?

A null result from a properly blinded study is a real result and must be reported as
one. This project has two refuted hypotheses already and has been careful about that, so
the answer should be decided before the data is seen, not after.

**Decide:** ______

---

## Note on what has been produced so far

Real geometry, real stimuli, and no fabricated results:

- `artifacts/m7a/stimuli/` — 4 clips, **10,475-vertex SMPL-X meshes**, `is_proxy: false`,
  generated from real EmoSign landmarks with the licence-gated model.
- `artifacts/m7a/study/public/` — 4 rendered `.mp4`, 480x600, 40 frames, **opaque
  filenames**, with a public manifest verified to contain no condition label.
- `artifacts/m7a/study/private/blinding_key.json` — the condition map. **Never given to
  a rater.**

No ratings exist, none have been simulated, and none may be. `make_rater_kit.py` emits a
*blank* rating sheet and a private answer key, and refuses to write `trials.csv` if a
condition label would leak into it. That refusal is the correct behaviour and should stay.