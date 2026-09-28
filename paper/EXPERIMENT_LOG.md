# Experiment Log

**Append-only.** Every run whose numbers could reach the paper gets an entry, at the time it runs —
not reconstructed later. If a number is in `main.tex` and not in this file, it does not ship.

Add entries at the bottom. Never edit or delete a past entry; if a run was wrong, add a new entry
that supersedes it and say so.

---

## Entry template

```
### <YYYY-MM-DD> · <RUN-ID> · <Task> · <one-line title>
- **Commit:** <sha>
- **Config:** <path to yaml / CLI invocation>
- **Data:** <dataset, split protocol — LOSO fold(s), signer IDs held out>
- **Hardware:** <RTX 4060 16GB / RTX 3050 4GB / Colab T4 — and which for the reported latency>
- **Seeds:** <list>
- **Metrics:** <metric = value ± std, per fold if applicable>
- **Claim(s) touched:** <C-IDs from CLAIMS_LEDGER.md>
- **Verdict:** <supports / weakens / refutes / inconclusive>
- **Notes:** <anything surprising, any fallback taken, any suspicion about the number>
```

### Rules
1. **LOSO or it doesn't count** for EmoSign. Record which signer was held out per fold.
2. **Latency and VRAM only from the RTX 3050.** If the number came from the 4060, label it clearly
   as a training-hardware measurement and do not use it in the efficiency table.
3. **Record fallbacks taken** — pseudo-labelled `L`, RAF-DB instead of DFEW, partial INCLUDE
   download. These become limitations text later; if they are not logged they get forgotten.
4. **Log suspicious results too.** An unexpectedly high score on a low-α emotion class
   (surprise− α=0.119, disgust α=0.166) is a bug signal. Write down the suspicion.
5. Failed and abandoned runs still get entries. Negative results save the next person a week.

---

## Log

### 2026-08-07 · — · T1 · Project initialized
- **Commit:** initial
- **Notes:** Documentation spine created (`project_breakdown.md`, `plan.md`, `team.md`, `paper/`).
  Dataset audit finding recorded: `/home/bhuwan/Videos/data` is **INCLUDE (ISL)**, not WLASL, with
  12 truncated `.part` archives including `Train_Test_Split`. No experiments yet.
- **Verdict:** n/a

<!-- Append new entries below this line. -->

### 2026-09-26 · — · M0 · Resource availability audit and the ASLLRP join
- **Commit:** see `git log -1`
- **Config:** n/a (no training)
- **Data:** all resources probed on the target machine
- **Hardware:** RTX 3050 4 GB, i5-12500H, 15 GB RAM (~3 GB free), swap 12.5/15.7 GB used
- **Metrics:** EmoSign join **200/200**; 200/200 clips downloaded (87 MB), 200/200 decode;
  MediaPipe face detection on sampled clips median **1.00** (min 0.94);
  blendshapes with real temporal variance median **30** of 52; max brow coefficient SD
  median **0.271**
- **Claim(s) touched:** none yet — this is the prerequisite measurement for C1/C3
- **Verdict:** supports. The affect benchmark is reachable, and the non-manual channel is
  recoverable from the clips.
- **Notes:**
  1. **The single most important finding of the project so far.** EmoSign's `video_name` is
     not a filename; its trailing numeric token is the ASLLRP utterance ID
     (`Jonathan_2012-11-27_sc93_5572615` → `5572615` → `crop_original_video.mp4`). Measured
     200/200 across all four signers. This removes the week-1 blocking access request that
     plan.md v1 identified as the critical path.
  2. EmoSign labels are **ungated** (43 KB CSV), not gated as v1 stated.
  3. The mirror contains gloss tokens (17,522 frame-aligned) but **no SignStream non-manual
     XML and no English sentences**. So `L` labels still need heuristics (M3 Track A) and
     gloss→English still needs the M5 ladder.
  4. The "Single Expression Set of 140 clips" cited by prior work is **not reconstructible**
     from the released CSV. Five subset rules tested; the closest gave 106. Our own
     single-expression rule gives 44. Reported with the full 200 alongside.
  5. **Fallback taken:** the 200 clips come from an ungated re-upload of a
     Boston-University-controlled corpus. Decision recorded rather than silently made:
     use for research, formalize before M9 submission. Owner: reviewer.
  6. Local WLASL is **not** INCLUDE as v1 claimed — `/home/bhuwan/Videos/data` no longer
     exists. The real corpus is 3,863 mp4 / 668 glosses / 7.4 GB, and it needs a repair
     pass: a 300-file sample found 2 undecodable (truncated `.part` containers with no
     `moov` atom). M1 must repair before extracting.
  7. v1 named a MediaPipe Tasks `HolisticLandmarker`, which does not exist. Tasks ships
     separate Face/Hand/Pose landmarkers; the 478-point Tasks face mesh (not 468) must be
     composed manually. The legacy `solutions.holistic` emits **no blendshapes** and is
     therefore unusable for this project.
  8. The blendshape basis emitted by `face_landmarker.task` is `_neutral` + 51 ARKit
     coefficients **including** `eyeLookUp*` and **excluding** `tongueOut`. Recorded in
     `seam/perception/tasks_api.py` and asserted on every face detection.

### 2026-09-26 · — · M2-prep · Perception stage latency on the RTX 3050
- **Commit:** see `git log -1`
- **Hardware:** RTX 3050 4 GB, CPU delegate, machine under external load (load avg 6–9)
- **Metrics:** per-module mean ms/frame at 480×320 — face+blendshapes **15.9**,
  2× hand **41.2**, pose **29.5**. Sequential full pipeline **109.7 ms/frame (9.1 FPS)**;
  **3-graph concurrent 67.3 ms/frame (14.9 FPS), a 1.63× speedup.** Best of 5 runs.
- **Verdict:** supports the efficiency claim's direction; the ≥20 FPS target is not yet met.
- **Notes:**
  1. **Hands cost more than the face** (41.2 vs 15.9 ms). The v1 budget assumed the face
     landmarker would dominate. It does not, and the optimization effort should follow the
     measurement.
  2. **Input resolution is irrelevant** across 160px–480px (78–86 ms): the bottleneck is not
     pixel throughput. `cv2.setNumThreads` ∈ {1,2,4} made no difference. The cost is fixed
     per-graph overhead, which is why concurrency helps and downscaling does not.
  3. Concurrency is safe: TFLite releases the GIL and the three graphs share no state.
     Parallel and sequential results are asserted identical in `tests/test_perception.py`.
  4. **K6 (≥20 FPS) is not met at 14.9 FPS.** Machine load is a confound (sequential
     measured 82 ms unloaded and 110 ms under load). Deferred to M2's dedicated benchmark
     on an idle machine; if it still misses, the target is relaxed and the miss reported
     rather than the measurement massaged.
  5. Found and fixed en route: the Tasks graphs require **strictly increasing timestamps per
     graph instance**, so decoding two videos back to back silently degrades tracking. There
     is now an explicit process-wide monotonic clock with a test.

<!-- Append new entries below this line. -->

### 2026-09-26 · — · M0 · Full 200-clip landmark extraction and face-coverage census
- **Commit:** see `git log -1`
- **Config:** `seam landmarks extract --dataset emosign`, 3-graph concurrent, CPU delegate
- **Data:** all 200 EmoSign clips, MediaPipe Tasks, 52 blendshapes + 553 landmarks per frame
- **Hardware:** RTX 3050 4 GB (unused — CPU delegate), 16 vCPU under external load
- **Seeds:** n/a (deterministic extraction)
- **Metrics:** **200/200 clips extracted, 0 failures**, 1744 s wall (0.10 clips/s incl. resume
  from 20 already done). 24,733 frames. 87 MB video → **71 MB shards**.
  - face-usable **200/200 (100%)**; face rate median 1.000 (min 0.871), pose median 1.000,
    hand median 0.938
  - blendshapes with real temporal variance: median 31 of 52 (min 20)
  - max brow-coefficient SD: median 0.262
  - per signer (all 100% usable): Ben 7 / 1,461 frames · Cory 87 / 10,965 · Jonathan 54 /
    6,259 · Rachel 52 / 6,048
- **Claim(s) touched:** prerequisite for C1, C1b, C3
- **Verdict:** supports. M0's evidence gate is met with margin.
- **Notes:**
  1. **The brow coefficients dominate the variance** across the whole corpus:
     `browDownRight` SD 0.329, `browDownLeft` 0.301, `browOuterUpLeft` 0.203,
     `browOuterUpRight` 0.193, against a brow-group mean activation of 0.205. ASL linguistics
     says brow-down marks a wh-question and brow-up a yes/no question — i.e. the
     highest-variance channel in the data is the one that is *grammatically* loaded. This is
     the raw material for C1, and it is the strongest prior expectation we have that the
     confound is real.
  2. **9 coefficients carry zero variance** on this corpus and can be dropped from the
     non-manual channel: `_neutral`, `cheekPuff`, `cheekSquintLeft`, `cheekSquintRight`,
     `jawForward`, `jawRight`, `mouthFrownRight`, `noseSneerLeft`, `noseSneerRight`. That is a
     free 52 → 43 dimensionality reduction. It is also a caution: `mouthFrownRight` is constant
     while `mouthFrownLeft` is not, which is the kind of asymmetry that should be checked
     against the landmark ordering before any left/right claim is made.
  3. Storage rule validated: 87 MB of video became 71 MB of shards, and the video is now
     disposable for every downstream milestone.
  4. Written to `artifacts/reports/emosign_landmarks.json`.

<!-- Append new entries below this line. -->

### 2026-09-26 · — · M2-prep · The VRAM budget is 3770 MB, not 4096
- **Metrics:** `nvidia-smi` reports 4096 MiB for the board; `torch.cuda.get_device_properties`
  reports **3770 MB**, and `mem_get_info` shows 3681 MB free at idle.
- **Verdict:** corrects the budget in `configs/base.yaml` and `project_breakdown.md` §6.8.
- **Notes:** ~326 MB is reserved by the display/EGL path before any process starts. A VRAM
  assertion written against the board figure fails for a reason unrelated to the model, and the
  2500 MB ceiling is 66% of usable rather than 61% of nominal. Both figures are now recorded
  explicitly rather than the nominal one being assumed.

<!-- Append new entries below this line. -->

### 2026-09-27 · — · M1 · WLASL manifest, integrity census, and three corrections
- **Commit:** see `git log -1`
- **Config:** `seam.data.wlasl` — annotation join, ffprobe sweep, failure classification
- **Data:** WLASL_v0.3.json (21,083 annotated instances) against the local tree
- **Metrics:** 2,657 distinct `(gloss, instance_id)` keys · **2,565 usable** · 92
  html_placeholder · 0 genuinely truncated · 189,352 frames · 665 glosses · 67 signers ·
  splits train 1,706 / val 522 / test 337. Full ffprobe sweep of 2,657 files in 19 s
  (8 threads).
- **Claim(s) touched:** prerequisite for C1
- **Verdict:** supports. The M1 substrate is larger than the gate requires (2,565 ≥ 500).
- **Notes:**
  1. **The on-disk filename is the `instance_id`, not the `video_id`.** The annotation carries
     both; the files are named `<gloss>/<instance_id>.mp4`. Joining on `video_id` matched 25 of
     3,863 files and looked like a catastrophic data loss rather than a key mismatch.
  2. **WLASL is 2,657 clips on this machine, not 3,863.** Of the 3,863 files, 2,657 are
     `<n>.mp4` and **1,206 are `_yt.mp4.part.mp4` duplicates** of instances that also have a
     good copy. The 3,863 figure in `plan.md` §1 is corrected here.
  3. **All 92 unusable clips are HTML error pages saved with an `.mp4` extension**, not
     truncated video: the first bytes are `<!DOCTYPE html>` (`CTYP`/`E html><` box). An
     ffmpeg re-cut to the annotated frame range cannot help, because there is no video in the
     file. The prior ISLR system reported these as "1,130 untrimmed, repairable"; that
     population does not exist in this copy. Failures are now *classified* —
     `html_placeholder` vs `truncated` vs `no_index_other` — because the three have different
     remedies and lumping them as "corrupt" sends you looking for a repair that cannot work.
  4. The 300-file sampled integrity check run in M0 found 2/300 undecodable; the full sweep
     puts it at **92/2,657 = 3.5%**, consistent.

### 2026-09-27 · — · M1 · Preprocessing and feature property tests
- **Commit:** see `git log -1`
- **Metrics:** 39 property tests over `preprocess/` and `features/`, plus 20 over the audit
  machinery. Suite total **132 passing**.
- **Verdict:** supports. Four real defects were found by the properties, not by inspection.
- **Notes:**
  1. **`interpolate_gaps` conflated the presence *column* index with the landmark *slice*
     offset.** Presence column 1 is the left hand; its landmark rows are 33..54. Indexing the
     landmark array with the column index filled the wrong block while every shape assertion
     still passed — the exact class of bug that produces plausible, meaningless features. The
     signature now requires `part_slices` and the reason is in its docstring.
  2. **A relative-only pause threshold reports a motionless signer as 93% *active*.** The
     threshold was `0.15 × the clip's own peak speed`, and on a still clip the "peak" is
     tracker noise, so the cut sat below the noise. `pause_stats` now takes
     `max(rel_threshold × peak, abs_floor)`; the property test that forced it is
     `test_still_signing_is_all_pause`.
  3. **Jerk is meaningless without pre-smoothing, and the default bandwidth is wrong for it.**
     The third derivative amplifies noise by 1/dt³: on raw landmarks a *linear ramp* scored
     higher jerk (6.12) than a *square wave* (4.22), so the feature was measuring the
     detector. `jerk` now smooths at `min_cutoff=6.0` rather than the interactive 1.0, because
     at 1 Hz a real 3 Hz repetition is attenuated as hard as the noise.
  4. **Two fixture traps, both instructive.** (a) Moving the two wrists in *opposite*
     directions holds their mean perfectly still, so every speed-based assertion over a
     symmetric-motion fixture is vacuous — this is a real property of two-handed signs, now
     pinned by `test_two_handed_opposed_motion_has_a_stationary_centroid`, and it is a genuine
     limitation of a centroid-based speed feature. (b) `arr[:, (0, 33)]` is *fancy* indexing and
     selects rows 0 and 33, not the range; the same column-vs-slice confusion as defect 1, in
     the test suite this time.
  5. Marker thresholds are **clip-relative** (`median + 1.5·MAD`) rather than absolute. A fixed
     cut would label a signer who holds a mild furrow throughout as marker-bearing on every
     frame, at which point the audit would be measuring the signer's face. Asserted by
     `test_marker_thresholding_is_clip_relative`.

### 2026-09-27 · — · M1 · Non-signer FER baselines: a broken instrument, diagnosed and fixed
- **Commit:** see `git log -1`
- **Config:** `scripts/train_fer.py`, RAF-DB (via `Pelmeshek/raf-db-7emotions-mediapipe-768`),
  three compact CNNs, no pretrained backbone
- **Data:** 5,164 train / 2,652 test (the dataset's own test split), 7 classes, 112×112
- **Hardware:** RTX 3050 4 GB, CPU training
- **Metrics:** first run 59.9 / 64.1 / 65.4% test accuracy. **After the front-end fix:
  61.7 / 63.7 / 66.3%.** Published RAF-DB SOTA is ~86% with ResNet-50-class backbones; these
  are small CNNs trained from scratch, so the gap is expected and is not hidden.
- **Claim(s) touched:** prerequisite for C1
- **Verdict:** **inconclusive, then informative.** The first run produced a broken instrument;
  the second run is sound and its result is a null.
- **Notes:**
  1. **The first audit was invalid and is recorded as such.** All three FER models, applied to
     sign-video face crops, predicted **one class for all 200 EmoSign clips** - sadness 200/200,
     fear 200/200, anger 200/200 - with Spearman correlation to true sentiment of **+0.03,
     -0.09, +0.16**. Per-frame probability spread *within* a clip was **0.0002** while the
     spread *across* clips was **0.283**: one output per clip. A model that emits one answer per
     clip is reading lighting, background and resolution, not a face, and cannot show a
     marker-dependent shift in a within-clip contrast however large the marker is.
  2. **The cause and the fix.** No FER front end at all: RGB crops, tight box at fit, 0.35
     margin at inference, raw [0,1] input. The fix is the conventional one - grayscale,
     histogram equalisation, per-image z-score - applied by **a single function on both the
     fitting and the inference path**. The broken models are kept at
     `artifacts/fer_v1_broken_preproc/` rather than deleted.
  3. **After the fix the instrument is live.** Predicted-label distributions on the 200 clips
     became non-degenerate: fer_cnn_a happiness 162 / sadness 19 / neutral 19; fer_cnn_b
     happiness 88 / surprise 68 / neutral 31 / sadness 13; fer_cnn_c happiness 62 / sadness 63
     / neutral 74. The **negative-mass baseline fell from 0.78 to 0.297**, which is most of the
     apparent "negative bias" in the first run - it was the model collapsing onto negative
     classes, not markers driving it there.
  4. **The recovered affect signal is weak but correctly signed.** Spearman(sentiment, negative
     mass) = +0.133 [-0.004, +0.269], +0.035 [-0.108, +0.179], +0.128 [-0.012, +0.267]. Right
     sign, two of three intervals touching zero. A compact CNN trained on RAF-DB recovers only
     weak affect from sign-video crops, and that is now a measured statement rather than an
     assumption.
  5. Two train/serve defects found en route, both now asserted by tests: `face_crop` returned
     float32 while the fitting path handed over uint8, so `equalizeHist` raised **on the
     inference side only**; and the crop geometry differed between fit and inference.
  6. **This is the same failure family as the prior ISLR system's 19-point normalization bug** -
     a model scored on differently-preprocessed input is measuring the preprocessing. Two
     independent occurrences in one project is now a rule, not an anecdote: rule 15 in
     `plan.md` §0.

### 2026-09-27 · — · M1 · The confound audit ★ — a null result, with power
- **Commit:** see `git log -1`
- **Config:** `scripts/run_confound_audit.py`; T=24 / stride 8 at 12 fps; within-clip matched
  pairs on (amplitude, speed) at 1.0 pooled sd; cluster bootstrap over clips, 4,000 resamples
- **Data:** **2,565 WLASL clips**, 189,352 frames, 665 glosses, 67 signers. 4,244 windows,
  **4,168 scorable (98%)**
- **Hardware:** RTX 3050 4 GB
- **Seeds:** 0/1/2 (one per FER model)
- **Metrics:** per model, per marker - matched shift in negative probability mass, 95% cluster
  bootstrap CI, minimum detectable effect at 80% power, Cohen's d, bootstrap p. Negative-mass
  baseline **0.297**. Marker prevalence: brow_raise 16.2%, brow_furrow 14.8%, mouth_morpheme
  22.4%, head_shake 19.1%, mouth_positive (control) 14.4% of windows.
- **Claim(s) touched:** **C1**
- **Verdict:** **refutes C1 on this corpus.** No marker-induced negative bias is detectable.
- **Notes:**
  1. **Result, fer_cnn_a / b / c:** brow_raise +0.0024 (p 0.17), **-0.0002 (p 0.92)**;
     brow_furrow -0.0024 (p 0.021), -0.0012 (p 0.61); mouth_morpheme +0.0011 (p 0.35),
     **-0.0080 (p 0.0003)**; head_shake -0.0007 (p 0.66), -0.0033 (p 0.25).
  2. **The two significant effects run the WRONG WAY** - marker-bearing windows are read as
     *less* negative - and they **do not replicate across models** (mouth_morpheme is +0.0011
     in b and -0.0080 in c). They are not evidence of a confound; they are noise at the edge of
     resolution.
  3. **The controls are clean, so the machinery is sound.** The `mouth_positive` control
     marker is null in all three models (p 0.85, 0.14, 0.61), and a uniform-random null model on
     the same matched pairs is null in all six rows (p 0.37-0.73). A planted +0.20 bias is
     recovered with a CI excluding zero, and a planted null is reported as null - both asserted
     in `tests/test_audit.py`.
  4. **Power.** MDE is **0.0030-0.0083**, i.e. **1.0-2.8% of the 0.297 baseline**, over 202-329
     contributing clips. The null therefore rules out marker-induced negative shifts larger than
     about 1-3% of baseline. This is a powered null, not an absence of measurement. Every null
     now carries its MDE; a null without one is not a finding.
  5. **The substrate is the real limitation, and it is a finding about the design.** WLASL is
     *isolated dictionary signing* - one gloss per clip, deliberately neutral and posed. The
     confound the ASL literature describes is a **discourse** phenomenon: a signer raises their
     brows *because the utterance is a question*. A dictionary video of BOOK has no discourse
     context, so the marker is not expected to be present and the contrast has nothing to
     contrast. This audit bounds the effect on isolated signing; it says nothing about continuous
     signing, where the phenomenon is expected to be strongest.
  6. **What this does to the plan.** C1 is refuted as stated, on the one corpus available at
     scale. The testable version of the claim needs *continuous* signing where markers are
     syntactically determined - ASLLRP utterances, or How2Sign. That is the substrate M3/M4
     should be pointed at, and it is a **negative result worth publishing**, not a failure: it
     localises the phenomenon, and localisation is the finding.
  7. `head_nod` has only 13 pairs over 9 clips and is reported as "too few clips" rather than
     with a degenerate interval - a one-cluster bootstrap has no variance, which is how a
     uniform null model came out "significant" at n=3 during development.

---

## M2 — ONNX export, parity, and the measurement instruments

Run ID `m2-onnx-001`. 24 parity batches / 56 real held-out faces per model.
Execution providers active: `['CUDAExecutionProvider', 'CPUExecutionProvider']`.

### Findings

1. **The torch dynamo ONNX exporter is unusable on this stack.** `torch.onnx.export` defaults to
   the dynamo path, which imports `onnxscript` and needs an op registry matching the exact torch
   build. `onnxscript` 0.5.7 has no `torch_2_11` module and this is torch 2.13. The two packages
   version independently, so this is not resolvable by installing a matching pair. Pinned
   `dynamo=False` (the TorchScript exporter), which has no such coupling and produces the same
   graph for the small conv nets exported here.

2. **`onnxruntime-gpu` could not use the GPU, and said so only in a log line.** The CUDA provider
   failed with `libcublasLt.so.13: cannot open shared object file`, while `get_available_providers()`
   still advertised `CUDAExecutionProvider`. The library was present the whole time at
   `site-packages/nvidia/cu13/lib/` — torch had installed it — but the system CUDA is v12, so the
   dynamic loader never looked in the bundled directory. `seam.export.runtime.ensure_cuda_libraries`
   pre-loads the bundled `nvidia/*/lib` trees with `RTLD_GLOBAL`; the CUDA provider then activates
   and TensorRT remains absent (`libnvinfer.so.10` not installed). Without this every latency
   number in this entry would have been a CPU number wearing a GPU label.

3. **torch cannot see ONNX Runtime's VRAM. Demonstrated, not asserted.** Running a CUDA-EP ONNX
   session 20 times: `nvidia-smi` shows the allocation, `torch.cuda.max_memory_allocated()`
   reports **0.0 MB**, because ORT allocates through its own arena and never touches torch's
   caching allocator. An ONNX model measured with torch would have passed any budget while
   measuring nothing.

4. **A baseline taken after session creation measures nothing.** The first inference builds the
   CUDA context, costing a few hundred MB, charged to whichever process created it. Sampling
   `nvidia-smi` after that first run excludes the context *and* the weights, and returned 0 MB
   for every model. The baseline must be sampled before the session exists.

5. **Parity must be measured on real faces.** The first harness synthesised random noise, which
   drove the models to near-uniform probabilities so an argmax was decided by a vanishing margin
   and flipped on numerical noise alone. On noise, `fer_cnn_a` INT8 showed 95.8% argmax
   agreement; on 7 real faces it showed 100%. Decoding was lifted into `seam.data.rafdb` so the
   trainer and the parity harness cannot drift apart — the exact failure mode that produced the
   invalid M1 v1 models.

6. **The `bbox_xyxy_768` frame was never mapped into image space.** A measured defect, not a
   guessed one. Stored images are a mix of 100x100 and 512x512; the boxes are in a 768-pixel frame.
   The old decoder clipped without rescaling. Face detection on the resulting crop, 40 rows per
   shard, over candidate mappings:

   | mapping | face detected |
   |---|---|
   | clip to image bounds (old) | 52% |
   | scale by `size/1000` | 82% |
   | scale by `size/768` | **100%** |

   The failure was silent: the crop is a plausible rectangle of plausible size, passes every shape
   check, and simply is not the face. `tests/test_rafdb_bbox.py` pins the mapping and asserts the
   ≥90% face-containment property on real rows.

7. **That defect did *not* change the headline number, and that is the useful part.** Re-training
   `fer_cnn_a` on corrected crops: **0.6136** (n_train 5316, n_test 2720) vs **0.6165** (n_train
   5164, n_test 2652) before. The difference is inside the noise band (SE ≈ 0.9 pp at p≈0.61,
   n=2720), so **M1's FER results stand** and the C1 refutation is unaffected. Data yield improved
   3% because rows with empty boxes are now counted and explained rather than conflated. M1 is not
   reopened; the correction is recorded here with its measured impact.

### Results

| model | precision | max abs dev | p95 abs dev | argmax | latency med/p95 (ms) | VRAM (MB) | on disk | verdict |
|---|---|---|---|---|---|---|---|---|
| fer_cnn_a | fp32 | 2.38e-07 | 5.96e-08 | 100.0% | 6.85 / 7.19 | 102 | 188 KB | PASS |
| fer_cnn_a | int8 | 3.76e-02 | 1.64e-02 | 95.8% | 18.87 / 19.33 | 102 | 57 KB | **FAIL** |
| fer_cnn_b | fp32 | 3.91e-04 | 5.82e-05 | 100.0% | 9.12 / 9.55 | 72 | 578 KB | PASS |
| fer_cnn_b | int8 | 4.58e-02 | 9.40e-03 | 100.0% | 31.96 / 32.89 | 72 | 158 KB | PASS |
| fer_cnn_c | fp32 | 3.58e-07 | 1.19e-07 | 100.0% | 11.58 / 11.82 | 0 | 191 KB | PASS |
| fer_cnn_c | int8 | 2.45e-02 | 8.44e-03 | 100.0% | 14.01 / 14.20 | 0 | 64 KB | PASS |

Deviation is on the probability simplex, not logits; the FP32 tolerance is calibrated from the
observed maximum (worst p95 5.82e-05) rather than carried over from a figure measured on a
different quantity in the prior ISLR system. All three FP32 graphs agree with PyTorch on the
argmax for every batch, so the exports are the same computation.

### Verdicts

8. **FP32 export is accepted for all three models.** Max probability deviation 3.91e-04, argmax
   100%, sizes 188-578 KB.
9. **INT8 is rejected for `fer_cnn_a`.** One batch of 24 disagrees (95.8%, gate 98%). This is a
   real quantization effect now that inputs are in distribution, and it is a legitimate early
   warning rather than a harness artefact. 70% size reduction does not buy a flipped prediction.
10. **INT8 is not a speed win on this hardware.** `fer_cnn_b` INT8 is *slower* than its own FP32
    graph (31.96 ms vs 9.12 ms). These models are small enough that dynamic-quantization overhead
    dominates on this GPU. Any speed claim from INT8 has to be measured per model, not assumed.
11. **VRAM is not a constraint for the FER baselines.** 102 + 72 + 0 = 174 MB against a 2500 MB
    ceiling. The first model absorbs the CUDA context; later models add weights only. The 0 MB is
    correct incremental accounting, not a failure to measure.
12. **6 models resident is not implied by 174 MB.** Each session holds its own weights; the 0 MB
    reading reflects allocator reuse in this sequence. A concurrent-6 figure needs a real
    simultaneous-load measurement before any claim is made, and K6 is still short of its 20 FPS
    target (17.8 idle), so the binding constraint is perception, not VRAM.

### Wiring and gates (M2)

13. **`seam bench` and `seam export` are wired and they fail the build.**
    `make export` exits non-zero on a failed parity gate (observed exit 2 on the
    `fer_cnn_a` INT8 rejection), and `make bench` exits non-zero on a VRAM breach.
    A gate that only prints red is not a gate. `make lint` now covers `scripts` as
    well as `src tests` — the trainer was not linted at all before, and a
    refactor broke it silently during this milestone.
14. **The perception benchmark had two defects of its own.** It drew the frame index
    and the MediaPipe timestamp from two independent `next()` calls on one iterator,
    so they advanced independently and were only accidentally ordered. And
    `--sequential` asked the process-wide `landmarker()` singleton for a sequential
    configuration, which it ignores once constructed — the "sequential" row would
    have been the concurrent path relabelled, a 1.57x error in the ablation table.
    Both fixed; a fresh `TaskLandmarker` is built for the sequential mode.
15. **The VRAM gate now enforces the larger of torch's and the driver's figures.**
    MediaPipe Tasks under the CPU delegate never touches torch, so torch reported
    0 MB while `nvidia-smi` showed 90 MB in use. A gate taking the smaller number
    would have passed a stage that was using the device. Perception now reports
    90 MB (4% of budget) instead of 0 MB.
16. **158 tests pass**, ruff and mypy clean across `src`, `tests` and `scripts`.

### K6 — sustained capture FPS, measured and not met

Concurrent, real video, RTX 3050: **median 57.9 ms/frame = 17.3 FPS**, p95 68.9 ms,
p99 69.8 ms. Earlier longer idle run: 17.8 FPS median, 18.6 FPS best. The K6 target
is **≥20 FPS**, so **K6 is not met**; the shortfall is ~15% and is stable across
runs, not noise. Sequential is 88.3 ms (11.3 FPS), so concurrency is buying 1.57x
and is necessary but not sufficient. Perception, not VRAM, is the binding
constraint for real-time signing, and the plan's own remedies (12 fps capture
target, keypoint subset) apply to perception rather than to the model budget.
Recorded as a miss with the number attached rather than a target quietly restated.

---

## M2 addendum — latency quantiles were measured over the wrong quantity

### A measurement defect, found while trying to strengthen the p99 claim

17. **`p95` and `p99` were percentiles of 5 run-means, not of frame latencies.**
    `time_fn` timed a whole run, divided by the iteration count, and then took
    percentiles across the `runs` samples. With `runs=5` that is a "p99" over five
    numbers — statistically it is very close to the maximum by construction, so it
    tracked machine noise rather than the workload, and it could not be compared
    with any other system's p99. Now every individual call is timed, and the
    quantiles are taken over `iterations * runs` samples, with the sample count
    printed next to them. Per-run means are retained and their p95 reported
    separately as `ms_p95_run_mean`, to make the difference visible rather than to
    quietly drop it.

18. **`fps_from_min` was the wrong number for the K6 verdict.** `1000/min` is a
    best case no capture loop will ever see. `fps_sustained = 1000/mean` is added
    and is what the timing line prints, because K6 asks what rate a signer
    actually gets when frames arrive back to back.

19. **Pinned by three tests**: the sample count is `iterations * runs`; a workload
    with a 25% slow tail must show `p99 > 2x median` (per-run means would dilute
    it away); and `fps_sustained == 1000/mean`.

### Blocked measurement — recorded, not worked around

20. **The two measurements this milestone still owes cannot be taken right now: the
    NVIDIA driver is not loaded in this kernel session.** `nvidia-smi` reports it
    cannot communicate with the driver, `/dev/nvidia*` does not exist, and no nvidia
    module is present. `torch.cuda.is_available()` is `False`. This is a machine
    fault, not a code fault, and it is not recoverable from this shell.

    Specifically still owed, and **not** measured:
    - the sustained-capture p95/p99 with a real sample size (the harness now
      supports it; it needs the GPU to be a GPU number);
    - the simultaneous live-stack footprint — 3 MediaPipe graphs plus 3 ONNX graphs
      all resident at once, which is what the "6 models resident" claim actually
      requires. `seam.eval.bench.resident_stack_bench` is written and wired to
      `seam bench --stack`, and is untested against hardware.

21. **A CPU run was attempted and its output is deliberately not used.** The
    harness reported `device cpu` against a configured target of the RTX 3050 and
    attached the note *"Latency and VRAM numbers from this run must not enter the
    paper's efficiency table."* It also declined to report VRAM at all. That refusal
    is the guard working: a CPU latency number is not a 3050 latency number, and
    the run refused to become evidence. The numbers it produced (min 25.2 ms,
    median 43.1 ms, n=600) are recorded here only to show they were not promoted.

22. **K6 is therefore unchanged and still a miss.** The verdict stands on the
    earlier GPU runs (17.3 FPS median, 18.6 best, against a ≥20 target). The
    corrected quantile machinery has not yet produced a fresh GPU p99, and no p99
    claim is made until it does.

23. **Provider probe hardened.** With the GPU gone, `provider_status()` returned an
    empty active-provider list, which is indistinguishable from a broken probe. It
    now retries with the CPU provider alone, mirroring how the real code path
    degrades, so the probe always reports something.

---

## M3 — Linguistic-marker supervision

Run ID `m3-markers-001`. Gate artefact `artifacts/audit/marker_labels.json`.

### What is now real

24. **200/200 EmoSign utterances join to the human-authored ASLLRP gloss map.**
    `asllrp_utterance_map` and `asllrp_gloss_tokens` (3.8 MB) were defined in
    `sources.py` but never fetched, and are marked as blocking M5. They are
    actually required by M3's syntactic track, so they were fetched. The join is
    by trailing numeric utterance ID and is complete, so every clip this project
    reasons about has real linguistic context rather than a gloss it invented.
    The gloss map holds 2,108 utterances, 1,243 token types, 16,784 tokens.

25. **The syntactic track has a genuine input, and keeps its provenance.**
    `seam.features.syntactic` labels `interrogative`, `negation`,
    `reference_establishment` from the annotation via lexical rules, and
    `topicalization` as a weak inference. Provenance is not cosmetic — the three
    lexically-grounded labels carry `lexical-rule-over-annotated-gloss`, and
    `topicalization` carries `heuristic (pseudo)` with confidence capped at 0.5,
    because ASLLRP's gloss is a flat token sequence with no constituent
    brackets and "this constituent was fronted" is not something the annotation
    states. Rates on the 200 clips: interrogative 0.230, negation 0.135,
    reference_establishment 0.465, topicalization 0.420.

26. **A plausible reading of `fs-` would have been wrong, and was checked.** The 98
    `fs-` token types look like they could encode facial or non-manual behaviour.
    They are compound sign tokens — `fs-BEACH`, `fs-LATE`, `fs-OF`, 39 occurrences
    for `fs-OF` alone. Reading the prefix as a facial signal would have put 98
    token types into the non-manual vocabulary for no reason, so the lexicon was
    read out of the corpus and `describe()` states the finding.

### The blocking finding: the visual marker set is degenerate at clip level

27. **4 of 6 visual markers fire on a large majority of clips**, which means they
    cannot discriminate between clips:

    | marker | clip prevalence | state |
    |---|---|---|
    | brow_raise | 0.855 | degenerate |
    | brow_furrow | 0.540 | usable |
    | mouth_morpheme | 0.950 | degenerate |
    | head_shake | 0.785 | degenerate |
    | head_nod | 0.170 | usable |
    | mouth_positive | 0.905 | degenerate — and this is the *control* |

    `mouth_positive` is designated in `markers.py` as the control that "should not
    read as negative". A control present on 90% of clips is not a control. It also
    means M1's confound audit compared signals that were near-constant across
    clips, which is a candidate explanation for M1's null that has nothing to do
    with marker effects being absent.

28. **This is not the threshold rule being too loose.** On iid uniform noise
    (500 x 52 coefficients) the `median + 1.5*MAD` rule fires on **0.2%** of
    frames, and 0.0% at k=2. The rule is sound; the real signals are simply
    heavy-tailed, so a k-sigma rule cannot push prevalence down. Raising k from
    1.5 to 4.5 leaves `brow_raise` at 0.43 and `mouth_positive` at 0.83. No
    defensible `k` makes these markers usable, and tuning one until the agreement
    statistic looked good would have been p-hacking, so it was not done.

29. **Two real bugs in the head path, found and fixed.** `_oscillation` thresholded
    at `head_angle / 2.0` while the field documents `head_angle` as "~20 degrees" —
    so the effective threshold was 10 degrees, inside the tracking jitter of a
    256x256 face. And the `min_reversals` test counted sign changes anywhere in the
    clip, so one qualifying wobble licensed every noisy turning point in the whole
    signal. Together these made `head_shake` fire on 0.98 of clips and `head_nod`
    on 0.65. Now thresholded at `head_angle` and requiring `min_reversals`
    *large-amplitude* turning points: `head_shake` 0.785, `head_nod` 0.170. Pinned
    by tests, including one asserting tracking jitter does not read as a shake.

### Agreement, and why two of the three pairings are not reported as results

30. The harness now **refuses to print a p-value for a degenerate marker**, and
    states why. A lift of ~1.0 is what a broken instrument and a genuine null both
    look like; conflating them is how "no effect" gets concluded when the truth is
    "no resolution".

    | pairing | observed | chance | lift | verdict |
    |---|---|---|---|---|
    | interrogative <-> brow_raise | 0.325 | 0.308 | 1.05 | not interpretable (brow_raise degenerate) |
    | interrogative <-> brow_furrow | 0.480 | 0.478 | 1.00 | **interpretable: null, p=0.96** |
    | negation <-> head_shake | 0.330 | 0.292 | 1.13 | not interpretable (head_shake degenerate) |

31. **The one interpretable pairing is a clean null**, on the marker whose
    prevalence is in the usable band. `interrogative` and `brow_furrow` are
    independent — one from human annotation, one from blendshapes — and they do not
    co-occur more than chance (lift 1.00, p=0.96). This is consistent with M1's WLASL
    null, and it is a *resolvable* null rather than an unmeasurable one.

### Verdict on M3

32. **The M3 gate is met on labels: 200 clips, every label carrying provenance,
    agreement statistics computed with base rates and a degeneracy check.** What the
    gate does *not* establish is any linguistic association, and the honest
    position is that the marker-claim half of the project is blocked on instrument
    calibration, not on data or on ideas.
33. **Next step, and it is not more thresholds.** A per-frame marker detector and a
    per-clip label are different instruments. The right fix is a clip-level marker
    definition with a stated minimum duration and amplitude, calibrated so that the
    control `mouth_positive` has low prevalence — a criterion fixed *before* looking
    at any interrogative/negation agreement, so the null above cannot be an artefact
    of the calibration and a future positive cannot be either.
