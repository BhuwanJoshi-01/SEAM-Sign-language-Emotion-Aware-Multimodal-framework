# Product

<!-- impeccable:product-schema 1 -->

Durable product truth for SEAM. Visual decisions live in `src/seam/web/tokens.css` and are
not restated here. Anything below marked **open** is a decision not yet taken.

## Platform

web — four static pages served by a single FastAPI process bound to localhost. No build step,
no framework, no bundler. Front-end libraries are vendored in-repo rather than fetched.

## Purpose

Take a video of someone signing and recover two things that are normally conflated in the
face: the **linguistic** non-manual signal (raised brows mark yes/no questions, a furrowed
brow marks wh-questions, a head-shake marks negation) and the **affective** signal. In ASL
the face is grammatically obligatory *and* carries emotion, and models — like hearing
non-signers — read grammar as negative affect.

The system therefore reports **what it measured and what it refuses to claim**, and that
second half is a product feature rather than a disclaimer.

## Users and jobs

Two, equally weighted. This was asked and answered; do not optimise for one and demote the
other.

| User | Situation | Job |
|---|---|---|
| A researcher or reviewer | opening it cold, often sceptical, at a desk | See whether the pipeline actually runs, and see exactly what is and is not supported |
| A Deaf signer or SL researcher | checking whether the tool is usable and respectful of the language | Read the non-manual channel legibly, with the caveats unmissable |

The second user is why the claims panel is prominent rather than buried, and why the live
view is labelled as *pipeline measurement* at every level rather than as a result.

## What the product makes possible

- **Continuous inference.** MediaPipe Tasks runs in the browser in WASM; a sliding window of
  the 52 ARKit blendshape coefficients plus head pose is posted to the server, which returns
  marker magnitudes with per-cue reliability. At most one request is in flight, so the panel
  reports the newest window rather than a growing queue of stale ones.
- **Video never leaves the machine.** The server receives numbers, never frames. There is
  nothing to store, log, or leak, and the demo can be run on footage nobody consented to
  share.
- **An animated SMPL-X avatar** driven by a learned whole-body regressor, exported as a
  skinned glTF that any 3D viewer can open.
- **A route index** where every endpoint is listed and probed live.

## Position

Every published sign-emotion model is trained or evaluated by hearing non-signers. The
efficiency claim is unshared: published baselines ran on 80 GB A100s or 300M-parameter models,
and this targets a 4 GB laptop GPU.

## Voice

Plain, specific, and unwilling. The house style across `plan.md`, `paper/WRITING_GUIDE.md` and
every UI string: name the measured number, name its instrument, state the limitation in the
same breath. No superlatives. No claim that is not already backed by a run ID.

Two patterns are load-bearing and must not be softened:

- **A withheld number is stated as withheld, with the measurement that withheld it.** "Affect
  is not shown: M4 measured 0.497 balanced accuracy against a 0.5 reference." Not an empty
  panel, and not the number.
- **A blind instrument is distinguished from an absent signal.** "blind — cannot discriminate"
  and "no marker in window" are different states and the UI keeps them different.

## Stack

Existing codebase answers this: static HTML/CSS/ES modules served by FastAPI, three.js
vendored. **No build step and no CDN** — a page that needs the network to render is not a
product page, and a CDN dependency cannot be pinned or audited by the repo. This was
confirmed when the first `/avatar` version imported `examples/jsm/*` from a CDN's raw path,
where the bare `three` specifier cannot resolve, and the page's own error handler blamed the
CDN — which was reachable the whole time.

Typography stays on the platform sans stack. Offered a self-hosted display face and declined;
the display voice is carried by weight, scale and tracking instead. Recorded so a later pass
does not re-open it as a fresh suggestion.

## Evidence and honesty constraints

These are product requirements, not style preferences, and they bind every surface:

- **No number without an artefact.** `tests/test_provenance.py` fails on any figure in
  `paper/EXPERIMENT_LOG.md` or `paper/CLAIMS_LEDGER.md` with no run behind it.
- **No human preference data exists.** The M7 study has stimuli and a harness and **zero
  raters**. Nothing may be presented as a result about how good the avatar looks.
- **SMPLer-X is third-party and trained on mocap.** It knows nothing about sign language and
  will render plausible signing whatever the hands say. Its output is not evidence about
  linguistic content.
- **A pipeline measurement is not a quality measurement.** A confidently wrong pipeline
  produces a similar table, so every live figure is labelled as instrumentation.
- **Licence-gated assets are never redistributed.** SMPL-X parameters are not vendored;
  ASLLRP XML is gitignored; signer video is released as IDs and labels only.
- **Contrast is measured, not eyeballed.** Body text clears 4.5:1 on every surface in both
  themes, and `make serve-check` fails if a page drops the token sheet.

## Accessibility

Both themes are first-class, and the theme follows the reader's system preference with a
persistent override. `?theme=light|dark` forces one — **the M7 study harness needs it**, since
a rater on a light-mode OS seeing a different stage from the next rater is a confounded
experiment.

Browser surfaces are themed from the palette (selection, focus rings, scrollbars, placeholder
text, tabular numerals) rather than left at defaults. Every control has a visible focus ring,
a named accessible name, and a real disabled state that stays legible — a 0.45-opacity label
measures 1.2:1 against its own surface, which is a broken control rather than a disabled one.

Keyboard reachability and a screen-reader pass on the live panel: **open**, not yet done.

## Open decisions

- **Avatar preference study.** Blocked on humans. Recruitment, ≥5 raters, ideally ≥1 signer,
  inter-rater agreement. Cannot be retrofitted.
- **ASLLRP provenance.** The 200 EmoSign clips resolve through an ungated re-upload of a
  Boston-University-controlled corpus. Decision on record: use for research now, formalise
  before submission. Owner: reviewer.
- **NSL / cross-lingual terms.** Not established; M8 stays blocked. Owner: reviewer.