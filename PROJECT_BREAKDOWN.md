# SEAM — project breakdown

**Sign Language Emotion-Aware Multimodal framework.** Monocular video → 3D SMPL-X avatar,
with linguistic and non-manual analysis on the side.

This document is the map: what each stage is, what is measured, what works, and what is
still missing. Numbers here are reproducible with the commands shown; nothing is asserted
without an artefact behind it.

---

## 0. The one-paragraph version

Take a video of someone signing. A **learned regressor (SMPLer-X)** turns each frame into a
full SMPL-X body — pose, hands, face, body shape — and a **skinned glTF with a real
animation clip** is written for any 3D viewer or web page. Alongside that, **MediaPipe
landmarks** feed a linguistic/affect analyser that reports brow, head and mouth activity as
*measurements with known reliability*, not as confident labels. The honest state: the avatar
pipeline works and is verified; the linguistic claims are partly refuted; **no human
preference study has been run**, because there are no raters.

---

## 1. Architecture

```
video ─┬─► SMPLer-X (learned SMPL-X regressor) ─► SMPL-X params ─► LBS ─► skinned GLB + anim
       │                                          (per frame)              │
       │                                                              ▼
       │                                                        web viewer / 3D app
       │
       └─► MediaPipe landmarks ─► syntactic/linguistic/affect features ─► SEAM analysis
                                    (with per-cue reliability)
```

Two front ends, deliberately. SMPLer-X needs pixels and gives a body. MediaPipe gives
landmarks and drives the linguistic analysis. They are not interchangeable: the landmark
path's legs are noise (see §4), and the regressor says nothing about meaning.

---

## 2. Stage status

| Stage | What it does | State | Evidence |
|---|---|---|---|
| **M0** | Data readiness gate | **OPEN on two reviewer decisions** | `make readiness`: 7/10. Three earlier blockers were false negatives; see §2.1 |
| **M1** | Marker-induced FER bias | **not supported on continuous signing** (human frame-level markers, pre-registered rule); **head shake supported on isolated signs from 12 clips** after the 2026-10-05 head-axis fix, brows and mouth null | `artifacts/audit/confound_audit.json`, `confound_audit_continuous.json` |
| **M2** | Latency budget on RTX 3050 | **met**, marginal | 19.7–20.4 FPS, K6 |
| **M3** | Non-manual instrumentation | **met** | 2,407 utterances, 100% mapped; brow raise validated against human frames; negation ↔ head shake measured (25 of 27 negated clips vs 80 of 173) once both hands and the head's turn were actually tracked |
| **M3b** | Label quality | **mixed** | kappa 0.639 / 0.734 usable, 0.028 / 0.038 at chance |
| **M4** | Factorized encoder (linguistic vs affect) | **refuted** | worst cross-AUC 0.705 (heuristic `y_L`) / 0.758 (human `y_L`) on corrected code, vs ≤0.60 |
| **M5a** | Gloss recognition | **gate not met** | WER 0.920 vs most-frequent baseline 0.920; first run superseded (constant predictor, misaligned frames) |
| **M5b** | Translation (How2Sign) | **not started** | data on disk: 35,176 sentences, 991 shards |
| **M6** | Emotion-conditioned generation | **not started** | — |
| **M7 (retargeting)** | Landmark → SMPL-X solve | **superseded** | landmark arms replaced by SMPLer-X |
| **M7a** | Avatar rendering | **working** | 4 clips, real geometry, animated glTF |
| **M7** | Preference study | **blocked on humans** | stimuli + harness ready; **0 raters** |

Stage names follow `plan.md` §3. Two stages are refuted hypotheses and one is at its
baseline. That is the honest state and it is recorded rather than buried.

### 2.1 M0 regressed because the gate was wrong, not because data was lost

`make readiness` reported 5/10 with four blockers. All four were false negatives, and all three
errors had one shape: **the gate stated a fact it had not measured.**

| resource | the gate said | what was actually on disk |
|---|---|---|
| `wlasl_local` | `reuse path absent: /home/bhuwan/Videos/wlasl/videos` | **3,863 clips, 7.43 GB, at exactly that path** |
| `rafdb_mediapipe` | `not fetched yet` | **2.72 GB, 6/6 shards readable** — 14,329 train + 3,071 val + 3,071 test rows |
| `how2sign_mediapipe_pose` | `not fetched yet` | was absent; now on disk as 991 shards / 4.8 GB from `martinctl/how2sign-asl-landmarks`. The 31-shard, 14.12 GB repo first named holds JPEGs and a constant caption and is unusable |
| `asl_citizen_poses` | `not fetched yet` | genuinely absent — 81 GB, reviewer-owned, not started |

Three fixes, each measured:

1. **`_state_for` emitted `reuse path absent: <path>` without touching the filesystem.** It
   had no manifest for `wlasl_local` because the reuse fallback keyed on *directory
   existence*, and `seam_data/wlasl/` exists — holding 5,130 **landmark shards**, the derived
   output of the M1 extraction, and zero videos. `dest="wlasl"` was doing double duty for a
   corpus and its extracted keypoints. The fallback now keys on "no file matching this
   resource's patterns is here".
2. **A partially-downloaded resource read as complete.** With (1) fixed, How2Sign reported
   `5/5 verified, ok` with 5 of 31 shards on disk. There is now a completeness check against
   `approx_bytes`, requiring the shortfall to exceed both 2% and an absolute 50 MB — the
   floor matters, because `approx_bytes` is an order of magnitude, not a contract.
3. **`rafdb`'s real location was never declared.** Its shards sit at `seam_data/rafdb/`
   under local names differing from the HuggingFace paths in `sources`.

**WLASL's recorded defect was wrong, and the repair was never needed.** `sources.py` claimed
*"untrimmed `.part` downloads whose mp4 containers have no `moov` atom"*, needing ffmpeg
re-cuts, and a test asserted that guess was still on file. An ffprobe sweep of **all 3,863**
files (42 s, 8 workers) found 92 undecodable and **every one is 813 KB of YouTube HTML**
beginning `<!DOCTYPE html>`, saved as `0.mp4` — zero moov faults.

Worse, `_FILENAME` did not match `N_yt.mp4.part.mp4` (it required `_yt` adjacent to `.part`),
so **1,206 of 3,863 files were invisible to the indexer** and the HTML placeholders were the
only candidate for their own `(gloss, instance)`. **88 of the 92 have a sibling that decodes
cleanly**, so the repair is a substitution, not a transcode, and `index_on_disk` now performs
it and records it:

| | before | after |
|---|---|---|
| index keys | 2,661 | **3,775** |
| keys bound to an HTML error page | 92 | **4** |
| usable clips | ~2,565 | **3,771** |

The 4 genuinely lost: `beard/1`, `children/1`, `corn/0`, `decide/0`. `wlasl_repaired/` was
**empty** — the earlier attempt produced nothing and left a directory that looked like
progress. Run `make wlasl-index` to see the current count.

**M0 is still OPEN, and now correctly so:** 7/10 resources usable. ASL Citizen (81 GB) and
NSL provenance are outstanding, and 4 of 3,775 indexed WLASL clips are gone. The gate now
counts those from `wlasl.index_on_disk` over every file instead of asking for a repair pass
on the strength of a 300-file sample.

---

## 3. What works, and how it is verified

### Perception — learned SMPL-X regression

`src/seam/perception/smplerx.py`

* Runs the read-only `Sapien_Pipeline` tree as a subprocess. **That tree is never written
  to** — verified by mtime; only `main/nsl_runner.py` is executed and its JSON read.
* Fits a 3.68 GiB laptop GPU at batch 4 (ViT-B, `smpler_x_b32`).
* Measured on 4 EmoSign clips / 541 frames: detection coverage **1.00 / 1.00 / 0.99 / 1.00**.

Three bugs here were only findable by rendering and looking, and each is documented where it
lives:

1. **Upside-down bodies.** SMPLer-X emits `global_orient` in its own convention (x averages
   **129°**), putting the head at y = −0.69 m. Upstream never rotates these, so there was no
   conversion to copy. Fixed by deriving it from two anatomical invariants — head above
   pelvis, front toward the viewer — both measured before and after. A first attempt
   rendered a *convincing back view*, which is how that class of bug hides.
2. **Empty frames.** The regressor works at a real camera distance: mesh centroid **22.6 m**
   from the origin, renderer at 2.6 m. The first run produced a 2.9 KB MP4 containing
   **two unique colours** while every measurement reported success.
3. **Collapsed frames.** Detected on head-above-pelvis distance (sharply bimodal: 0.68 m for
   frames 0–39, then 0.17 m for 40–42). A median-deviation test was tried first and found
   **zero** outliers in a clip that visibly has three. Repaired frames are interpolated, and
   reported as interpolated.

### Avatar — skinned glTF with animation

`src/seam/avatar/gltf_export.py`

The old exporter wrote **one static mesh per frame** — 40 nodes, 0 skins, no animation on
clip 1372. A file that opens, shows a body, and never moves. That is a debug dump, not an
asset.

The new exporter writes what a viewer actually needs: **one mesh**, `JOINTS_0`/`WEIGHTS_0`,
**55 joint nodes** in `kintree_table` order, `inverseBindMatrices`, and an animation clip
keying every joint's TRS.

It is verified **twice, by two readers that share no code**:

| | |
|---|---|
| nodes / skins / animations | 56 / 1 / 1 |
| channels / keyframes | 110 / 85 |
| max error vs mesh pipeline | **6.4 mm** |
| 4-influence weight mass dropped | 4.7% (SMPL-X has up to 10 influences; glTF carries 4) |

The residual is the glTF 4-influence limit, and it is **reported** rather than absorbed into
a loose tolerance.

**The second reader is the one that matters**, and it was added because the first one had
been agreeing with the writer. `tests/test_gltf_conformance.py` loads the file through
**three.js's own `GLTFLoader` and its own skinning** in headless Chrome and asserts the
result is one `SkinnedMesh` with 55 bones, 110 tracks, the right duration, a human-sized
skinned pose, and vertices that actually move. Three.js is vendored, so this needs no
network. Two bugs were invisible to every offline check and obvious there:

* `JOINTS_4`/`WEIGHTS_4` are not glTF 2.0 attribute names (the spec says `JOINTS_0`/
  `WEIGHTS_0`). Every index and byte range was valid, and three.js stored them as custom
  attributes, leaving `skinWeight` undefined — so nothing could open the file.
* **All 110 samplers shared one frame-major `(n_frames, n_joints, C)` block**, so each
  sampler's output accessor held 220 values for 4 keyframes. The spec requires one value per
  keyframe *per channel*. Every index was in range; three.js paired the 4-element time
  accessor with the front of the block, so all 55 joints read frame 0 of joints 0–3 and the
  body rendered as a mangled sliver.

Both were reported as `agrees: true` by `verify_glb_animation`, because it indexed the block
with the same convention the writer used. **A verifier that shares the writer's assumptions
cannot detect either class of bug** — that is the generalisable lesson here, and it is why
the exporter now has a reader it does not control.

Six bugs in the exporter, four found by the verifier and two only by a real runtime:

* GLB container missing the chunk **type** field — plausible file, unparseable JSON.
* `byteLength` assigned after the JSON was already serialised.
* Samplers built as `[all translations] + [all rotations]` while channels index them
  interleaved — surfaces only once something animates.
* `global_transl` never written, so the body sat the wrong distance from the origin.
* `JOINTS_4`/`WEIGHTS_4` instead of `JOINTS_0`/`WEIGHTS_0` — unloadable in any viewer.
* One shared keyframe block across all samplers — spec-violating and mis-assigns every joint.

Plus two in the *verifier*: reading glTF's column-major `MAT4` as row-major, and indexing
animation rows as `frame*n_joints + joint` — the index that made the sampler bug invisible.

### Kinematics cross-check

`scripts/validate_fk_vs_blender.py`

Our forward kinematics is checked against **Blender's own armature evaluation** on a
90° transverse rotation of `spine1`:

* bones outside the pivot's subtree: **0.000°**
* bones inside it: **exactly 90.000°**
* Blender: 76–90° on the same probe, because the fitted spine curves and ours is collinear

So the rotation algebra is confirmed against an independent implementation, and the residual
is quantified: **max 13.933° on `head`**, mean 1.10°. That number is the error budget for
landmark-derived head/neck tilt and was previously unknown.

---

## 4. What does not work, and why

**Landmark legs are noise.** Hip→knee distance spans **0.06–1.42 torso-lengths** across 200
EmoSign clips. Two 2D points per bone do not determine monocular depth for a leg. This is not
a bug to tune; it is the ceiling of the method, and it is why the front end was replaced.

**Frame indices were being read at the wrong rate.** ASLLRP frame indices are on a 30 fps
timeline and 138 of the 200 clips are 24 fps, so the frame-for-frame mapping ran a quarter
fast on most of the corpus. Found by testing alignment against an independent signal
(annotated blinks vs the eye-blink blendshape: AUC 0.554 frame-for-frame, 0.710 corrected,
on the 24 fps clips). Fixed in `asllrp.crop_frame_position`, which takes the clip's frame
rate and has no default.

**Gloss recognition does not beat its baseline.** Re-run 2026-10-05 on correctly aligned
frames: WER **0.920** against a most-frequent baseline of **0.920** (shuffled control 0.910;
closed-vocab 0.881), 1,736 tokens over 546 glosses. The numbers first recorded here were not
a measurement - the model predicted one gloss in every fold - and are kept for the record:
WER 0.916 against 0.916 (shuffled control 0.911; closed-vocab 0.875). 1,563 tokens, 499
glosses, 3.1 per gloss, 284 hapax.

**Linguistic and affect information do not separate.** Worst cross-AUC 0.705 with heuristic
`y_L` and 0.758 with human `y_L` (re-run 2026-10-05 on corrected code), against ≤0.60. Signer
control is 0.98 in both arms, so the instrument can see. Over three seeds no separation loss
moves the direction the gate fails on, and neither model learns affect (balanced accuracy
0.493–0.506 against 0.5).

**No avatar preference result exists.** Both study scripts refuse to invent a baseline,
correctly. The comparison arm is now defined (§5) and both arms render, but **no human has
rated anything**, and no ratings are simulated.

---

## 5. The M7 comparison, decided

| Arm | Pose source |
|---|---|
| **A** | SMPLer-X — learned whole-body regression |
| **B** | MediaPipe landmarks → per-joint rotation solve (this project's own previous front end) |

Both arms are real existing code. A stand-in baseline would answer a question about the
renderer rather than about this project's contribution.

Fairness is enforced in code, each point because it was got wrong first:

* **One camera for both arms.** Fitting each separately gave 5.92 m vs 4.28 m, so the arms
  appeared at different scales — and a viewer asked which looks better answers "the bigger
  one".
* **Camera distance from the 99th percentile** of vertex radius over both arms. `max` let one
  stray vertex shrink both arms to specks.
* **Identical frame rate**, side-by-side truncated to the shorter arm. Resampling one arm
  silently doubles the other's playback speed.
* **No arm label drawn on any frame** — the same files feed the blinded study.

---

## 6. The web product

```bash
make demo-avatar     # video → SMPL-X → mesh → animated .glb + .mp4 + manifest
make serve           # then open http://127.0.0.1:8000/avatar
```

`/avatar` loads the **animated GLB** in three.js (skinned mesh, real clip, transport
controls, orbit camera, skeleton toggle) beside the per-clip measurements, and states
plainly at the top that **no ratings exist**. Routes:

| Route | Purpose |
|---|---|
| `/` | webcam demo; links to `/avatar` |
| `/avatar` | product page |
| `/api/demo/manifest` | what `make demo-avatar` produced, including `status` while it runs |
| `/api/demo/{name}` | one artefact; path traversal is rejected |
| `/static/vendor/{path}` | vendored three.js; path traversal is rejected |

**three.js is vendored, not fetched.** It lives in `src/seam/web/vendor/three` and is served
from `/static/vendor/`. The first version imported `examples/jsm/*` from a CDN's raw path,
where the bare `three` specifier inside those modules cannot resolve — every import failed,
and the page's own error handler reported it as "CDN unreachable", which was a
**misdiagnosis**: the CDN was fine, the specifier was wrong. A CDN import was tried first and
is wrong for a product anyway: it makes the page fail whenever the CDN is unreachable and it
cannot be pinned or audited by the repo.

`make serve-check` asserts the whole contract — both pages, all three vendored assets, both
traversal guards, and the manifest shape. `/avatar` was unlisted and unchecked for its whole
life, which is most of why a working feature read as a broken website.

`make demo-avatar` takes ~15 minutes on the GPU and now **publishes a `status` into the
manifest while it runs**. It used to write the manifest only at the end, so for the whole
run the page said "No clips. Run `make demo-avatar`" — which is exactly what it says to
someone who never ran it.

### Theming, and what it cost

Light and dark are **both designed**, not one derived from the other. Dark is not light
inverted: surfaces climb `8% → 12% → 16%` near-neutral; light goes white → `#f7f9fc` →
`#eef2f8` with a deeper accent, because 4.5:1 is far harder to hit on white. Shadows widen as
the ambient falls, and the 3D stage is lit for its own background in each theme — a light
stage needs *less* ambient or the mesh washes out and loses every silhouette cue.

`src/seam/web/tokens.css` is the only place a colour exists. `theme.js` resolves the theme
(explicit choice → `?theme=` override → system → dark) and runs from an inline snippet before
first paint, so there is no flash. `?theme=` is not test scaffolding: **the M7 study harness
needs it**, because a rater whose OS is in light mode seeing a different stage background
from the next rater is a confounded experiment.

Contrast was measured, not eyeballed. **One value failed and was corrected**: light `--ok` on
its own tint measured 4.44:1. `--fg-subtle` also measured 3.7:1 on the light stage — the
stage is *darker* than the page in that theme, which inverts the usual direction — so there
is a separate `--fg-stage`. Disabled controls use muted colours rather than `opacity`, because
a 0.45-opacity label measures **1.2:1** against its own surface.

The mechanical detector reports **0 findings** across four pages × two themes × desktop and
390 px mobile. Two patterns it did find were removed rather than re-declared: a coloured
`border-left` ≥1px on every callout, and `ui-monospace` on `body`, which made every paragraph
read as a code listing. Monospace is now scoped to identifiers and raw values.

**One product bug was found by looking at the render.** The GLB wrote no `materials` block, so
every conforming loader applied the glTF spec defaults — `metallicFactor: 1.0`,
`roughnessFactor: 1.0` — a mirror. With no environment map a metal has no diffuse term, and
the body rendered as a near-black silhouette on any background, in both themes. Legal glTF,
so every structural check passed throughout. The exporter now writes a matte material
(`metallicFactor 0.0`, `roughnessFactor 0.72`) and a test asserts it.

---

## 7. Provenance and honesty rules

* **SMPL-X parameters are licence-gated** and deliberately not vendored. The loader takes a
  path; `SMPLX ?= ...` in the Makefile is overridable.
* **SMPLer-X weights are not vendored either.** Third-party, trained on mocap, knows nothing
  about sign language. Nothing derived from it may be presented as evidence about linguistic
  content.
* **No human preference data is fabricated, ever.** `make_rater_kit.py` emits a blank sheet
  and refuses to write `trials.csv` if a condition label would leak. That refusal stays.
* **ASLLRP data is not redistributed**; `*.xml` is gitignored as a licence safeguard.
* **A provenance guard** (`tests/test_provenance.py`) fails on any number in
  `EXPERIMENT_LOG.md` / `CLAIMS_LEDGER.md` that has no artefact behind it.
* **A staleness guard** (`tests/test_artifact_staleness.py`) fails when a result artefact
  was written by code that has since changed. Each artefact carries a fingerprint of every
  `seam` module loaded when it was written; an overtaken result is moved to
  `artifacts/superseded/` and registered in `paper/artifact_status.json`, never overwritten.
* **ASLLRP frame indices are on a 30 fps timeline** and most clips are 24 fps. Anything that
  reads a frame-level annotation goes through `asllrp.crop_frame_position`, which takes the
  clip's frame rate and has no default.
* Every table above is **pipeline measurement, not quality measurement.** A confidently
  wrong pipeline produces a similar table.

---

## 8. Reproducing the numbers

```bash
make lint typecheck test                       # the whole suite; the count is whatever pytest prints
make provenance                                # untraced numbers + stale artefacts
make repro                                     # every cited artefact, 15 stages, in dependency order
make serve-check                               # HTTP contract of both pages
make demo-avatar                               # 4 clips, ~15 min (SMPLer-X on GPU)
make serve                                     # / and /avatar
PYTHONPATH=src python scripts/validate_fk_vs_blender.py    # needs blender
SEAM_SMPLX_MODEL=... pytest tests/test_gltf_export.py tests/test_gltf_conformance.py -q
```

Run from the repo root. `SEAM_SMPLX_MODEL` is only needed for the model-dependent tests; they
skip loudly without it rather than passing quietly. `tests/test_gltf_conformance.py` also
needs a `google-chrome` binary and skips without one — and when Chrome *is* present but
fails, it fails with the browser's message rather than passing quietly.