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

---

## M3 addendum — clip-level marker calibration, and a confound that nearly survived

Run `m3-markers-002`. Same 200 clips, same labels, new instrument.

### The calibration criterion was pre-registered, and it could not be met

34. The criterion fixed *before* looking at any agreement statistic: the control
    `mouth_positive` must be present in ≤25% of clips, and every marker inside
    [5%, 60%]. **No setting satisfied it**, across a 64-point sweep of
    run-duration × peak × coverage. The reason is not that the thresholds are
    wrong: it is that **`mouth_positive` is not a control.** Its peak excursion has
    a median of 27 "MADs" and a 90th percentile of 81 — mouths move while people
    sign, so a "mouth positive" signal is present in essentially every clip. A
    screen that demands a control be rare rejects this signal for being real.

### Two scale bugs, and a statistic that was measuring clip length

35. **MAD is the wrong scale for these signals.** A blendshape coefficient spends
    most frames near its floor and peaks occasionally, so the MAD is dominated by
    the flat part and `(x - median) / MAD` explodes at the peak. The result was a
    27:0 spread in peak evidence between `mouth_positive` and `head_nod`, across
    which no single threshold is meaningful. Normalising instead by the clip's own
    dynamic range, `p95 - p50`, brought every blendshape marker into 0–5 and makes
    the peak threshold a scale-free "fraction of this clip's own range".

36. **The first result was substantially clip length, not head movement.** With
    the integral of the excursion as the statistic, all three pairings looked
    strong and correctly directed: interrogative/brow_raise r=+0.41 (p=0.015),
    interrogative/brow_furrow r=+0.66 (p=0.0001), negation/head_shake r=+0.71
    (p=0.003). **Every syntactic label correlates with clip duration on this corpus**
    — interrogative clips run 5.77 s against 4.48 s (r_pb = +0.68), negation +0.54 —
    because questions and negated statements are longer utterances. The magnitude was
    an integral, so it inherited that mechanically. Switching to a per-frame mean
    collapsed interrogative/brow_raise to **−0.045**: that association was entirely
    clip duration and would have been reported as a finding.

37. **The fix is a partial correlation against log-duration**, which keeps the label
    binary. The obvious implementation — residualise the 0/1 label on the covariate,
    then re-split at zero — silently regroups the clips and returned a uniform
    zero; it was tried first and is why the test now keeps `y` binary.

38. **A per-frame bar in the wrong units made an instrument always read zero.**
    `clip_presence` compared the new clip-range evidence against
    `MarkerThresholds.k` (1.5, a MAD-scale constant), which is unreachable for
    evidence bounded near 1.0. All four blendshape markers therefore reported zero
    coverage and zero duration, and the whole clip-level view read "absent"
    regardless of settings — which is what made the pre-registered sweep appear to
    have found nothing. A bar in the wrong units does not fail loudly.

### Result

| pairing | r (partial, duration-controlled) | perm p | ×3 | MDE | n+ |
|---|---|---|---|---|---|
| interrogative <-> brow_raise | −0.084 | 0.621 | 1.000 | 0.331 | 46 |
| interrogative <-> brow_furrow | 0.199 | 0.244 | 0.733 | 0.336 | 46 |
| **negation <-> head_shake** | **0.554** | **0.011** | **0.033** | 0.411 | 27 |

39. **The canonical ASL negation marker is associated with negated utterances**, and
    it is the one effect that survives: partial r = 0.554, permutation p = 0.011,
    Bonferroni-corrected 0.033 over the three pairings tested, against an MDE of
    0.411 — the effect is larger than the smallest one this design could detect.
    Negated utterances carry 0.55 SD stronger head-shake evidence than others, from
    a lexical label read off human annotation and a visual pseudo-label read off head
    pose, with clip length controlled.
40. **The two interrogative pairings are powered nulls.** `brow_raise` is flat
    (−0.084) and `brow_furrow` is 0.199 with an MDE of 0.336, so effects below
    roughly 0.34 SD are not ruled out — which is a different statement from "there is
    no effect", and is why the MDE is printed beside the p-value.
41. **Why interrogative/brow failed while negation/head_shake passed, and why that
    is not yet a finding.** Head shake is a discrete, high-amplitude, directional
    event that a head-pose signal represents directly. A brow raise in continuous
    signing is graded and co-occurs with topic, contrast and emphasis, so a binary
    question/declaration contrast may simply not be the right conditioning for it.
    That is a hypothesis for M4, not a conclusion here.
42. **This does not reopen M1.** M1 tested marker-induced *shifts in a FER
    prediction* on isolated WLASL signs; this is marker–syntax co-occurrence on
    continuous EmoSign utterances with a duration control. The substrates differ and
    the two are compatible: an absent effect on isolated dictionary signing, and a
    present effect where negation is syntactically determined.
43. **Remaining caveats, stated because they bound the claim.** n=200 clips with 27
    negation positives; both variables are pseudo-labels, carrying the provenance
    strings already recorded; the duration control is linear in log-duration; the
    sequence was analyse → find a strong result → find the confound → re-analyse, and
    while the confound was identified from a *label–duration* correlation measured
    independently of the marker, the fact that the analysis changed after seeing
    results is part of the record.

---

## M2 close-out — the efficiency numbers, on a machine whose clock was the variable

Runs `m2-perception-*` and `m2-stack-*`, 2026-09-28, governor `performance`,
`cpu_clock_mhz_during` recorded in every artefact.

### What the earlier numbers actually were

45. **The 118 ms figure was a measurement of the CPU governor, not of the pipeline.**
    The host had rebooted and come up in `powersave` with the core clock at
    **1.14 GHz against a 4.5 GHz maximum**. MediaPipe runs on the CPU delegate, so
    perception is bound by exactly that clock. Before the reboot the same benchmark
    read 57.9 ms. Two figures, one machine, and **no field in the report distinguished
    them** — the artefact recorded device, load and memory, never the clock.
46. **Two candidate explanations were measured and rejected before blaming the clock**,
    because both would have invalidated the earlier number too: the frame mix
    (first-30-frame median 100.5 ms vs full-300 97.9 ms, so not a sampling artefact)
    and hand detection (97.7 ms with hands present vs 101.2 ms without, so the
    hand landmarker is not the tail).

### The gate now judges the run, and two of its three checks were wrong first

47. **`Timing.noise_ratio` cannot see a busy machine.** It compares runs *within* one
    session, so three equally contended runs look clean — it reported 1.01–1.11
    through the whole contended period. Added a load gate: load per core ≤0.75 and
    ≥2 GB available memory, recorded from `/proc/meminfo` and `getloadavg`.
48. **Sampling the clock *after* the run discarded a passing measurement.** Under the
    performance governor the CPU falls to ~700 MHz when idle, so the post-run reading
    reported 16% of peak for a run genuinely at 87%, and a **20.5 FPS** result was
    rejected as unreportable. The clock is now sampled *inside* the measured window
    and reported as `cpu_clock_mhz_during`.
49. **Gating on 80% of `cpuinfo_max_freq` flagged a healthy run.** That figure is a
    *single-core* turbo maximum; this i5-12500H sustains ~2.6–3.3 GHz under a
    multi-threaded MediaPipe load, 58–73% of it. The floor is now 40%, a sanity check
    for a misconfigured or throttled host, and the *actionable* gate is the governor.
    A threshold that flags the machine rather than the measurement is the same error
    as the idle sample, in the opposite direction.
50. **The governor was checked only in one branch**, so a run that happened to catch a
    full clock under `powersave` was accepted — precisely the non-reproducible case,
    since the same invocation minutes later reports the idle clock and 2.5x the
    latency. It is now checked first and unconditionally.

### Results — gate met

51. **Perception, 3 MediaPipe graphs concurrent**, RTX 3050, governor `performance`,
    quantiles over 900 individually-timed calls (300 frames × 3 runs):

    | statistic | value |
    |---|---|
    | min | 25.9 ms |
    | **p50** | **41.8 ms** |
    | **p95** | **73.2 ms** |
    | **p99** | **78.1 ms** |
    | sustained rate | **20.4 FPS** |

    Independent repeat runs: 19.7, 20.0, 20.3 FPS. VRAM 90 MB.

52. **Sequential baseline**, same instrument: p50 **73.0 ms**, p95 103.0, p99 111.2,
    12.8 FPS. **Concurrency is worth 1.75x** on the median (41.8 vs 73.0 ms), measured
    the same way in the same session — larger than the 1.57x previously recorded from
    run-means, and reproducible.
53. **K6 is met, marginally.** Four reportable runs give 19.7, 20.0, 20.3, 20.4 FPS
    against a ≥20 target — straddling it, not clearing it. Stated as "meets the target
    at the median and does not clear it at the low end" rather than as a pass, because
    a single 20.4 FPS run would have been the dishonest way to report it.
54. **Six models resident, all live at once**: 3 MediaPipe graphs plus 3 ONNX FER
    graphs, **186 MB peak against the 2500 MB ceiling (7% of budget)**, at 41.4 ms p50
    / 20.3 FPS — statistically indistinguishable from perception alone. The FER graphs
    execute on the GPU while perception runs on the CPU, so they overlap; this is the
    measurement that the earlier sequential 174 MB could not license.
55. **VRAM is not the binding constraint anywhere.** 186 MB for the whole live stack
    against 2500 MB. The perception stage is the limit and it is a *CPU* limit, which
    is why the governor mattered more than any model choice in this milestone.

---

## M3 close-out — grounding the feature set in what Deaf annotators wrote

Run `m3-cues-001`. Artefact `artifacts/audit/cue_grounding.json`.

56. **The corpus is 600 free-text strings over 200 clips**, three annotator columns
    each. The plan's figure is confirmed. The feature set in `seam.features` was
    assembled by assumption; this tests it against what the annotators actually
    wrote.

### The annotators write two different kinds of thing

57. **Motor cues and affective interpretations are mixed in the same fields, and they
    are not the same kind of evidence.** "raised eye brow", "bared teeth", "head
    shake" describe a movement, and a movement is something a feature can be expected
    to recover. "conveys a sense of surprise", "signifies worry" are the annotator's
    *reading* of an affect — and no feature in `prosody` or `markers` is a claim about
    affect, because those are FER's job. Treating an affective interpretation as
    validation of a brow-raise feature would be a category error, and it would
    manufacture agreement on a face the FER model is explicitly trained to read.

    So the two are parsed separately. **482/600 strings (80%) contain at least one
    motor cue; 118 are affective-only** and are excluded from the motor test by
    construction rather than counted as misses. Coverage is reported, because a
    vocabulary that silently discards what it cannot parse reports a clean result on
    the remainder and hides the selection.

58. **The vocabulary was mined from the corpus, not assumed.** A first pass matched
    459/600 and its 141 misses were read before the vocabulary was extended. Those
    misses contained real observable cues the patterns had overlooked — "eyes
    popped", "widened eyes", "bared teeth", "grimace", "lower than usual" — and
    extending the patterns raised motor coverage to 80%.

### Result: the feature set is not validated

15 cue/feature tests, all partial correlations on log-duration, permutation p, MDE
reported, Bonferroni across the 15:

| cue | feature | n+ | r | p | MDE |
|---|---|---|---|---|---|
| head_shake | head_shake | 59 | **+0.275** | 0.077 | 0.31 |
| repetition | repetition | 11 | +0.433 | 0.173 | 0.62 |
| speed_slow | speed | 19 | **−0.398** | 0.111 | 0.49 |
| head_nod | head_nod | 31 | — | **BLIND** | — |
| smile | mouth_positive | 30 | +0.221 | 0.268 | 0.39 |
| pause_hesitation | pause_ratio / pause_mean | 9 | +0.198 / +0.207 | 0.574 / 0.566 | 0.68 / 0.70 |
| emphasis | jerk / volume / amplitude | 115 | −0.133 / +0.055 / +0.090 | 0.36–0.71 | 0.28 |
| sign_size | amplitude | 40 | +0.102 | 0.569 | 0.35 |
| mouth_shape | mouth_morpheme | 101 | −0.153 | 0.277 | 0.28 |
| brow_furrow | brow_furrow | 46 | +0.061 | 0.715 | 0.33 |
| brow_raise | brow_raise | 29 | −0.045 | 0.799 | 0.40 |
| speed_fast | speed | 25 | −0.032 | 0.884 | 0.43 |

59. **Nothing survives Bonferroni.** The grounding move returns a negative result: the
    assumed feature set does not demonstrably recover the cues the annotators named.
    That is the point of running it — an assumed feature set confirmed by tests
    designed after the fact is not a validated one.

60. **The one channel with independent corroboration is the head channel.**
    `head_shake` here is r=+0.275 (p=0.077) against annotator text, and in the
    previous run it was r=+0.554 (p=0.011) against *lexical negation* from the ASLLRP
    gloss. Two independent ground truths — a Deaf annotator's free text and a
    human-authored linguistic annotation — agree on the same channel, which is
    stronger evidence than either alone. The brow and mouth channels show nothing
    here, and are exactly the channels M3 measured as firing on 79–95% of clips.

61. **One test is reported as BLIND, not as a null.** `head_nod` is zero on **83%** of
    clips, so it cannot discriminate and no conclusion is available from it. Reporting
    its non-separation as evidence would be indistinguishable from saying the
    annotators were wrong, when nothing was measured. The same check ran on all 15
    tests: 14 are informative, 1 is blind.

62. **The features were checked for dynamic range before the nulls were believed.**
    Three earlier findings in this project came from an instrument reporting a
    plausible number with no signal in it, so each feature's interquartile spread and
    zero fraction were measured first. The features do vary across clips
    (e.g. `amplitude` IQR 0.285–0.536, `pause_ratio` 0.406–0.944), so 14 of the nulls
    are real nulls rather than instrument failures.

### What this specifies for M4

63. **Six cue categories that Deaf annotators actually used have no feature behind
    them at all**: `head_tilt`, `eye_widen`, `blink_close`, `gaze_shift`,
    `fingerspelling`, `body_posture`. This is a specification derived from the corpus
    rather than a list of preferences, and it is the concrete content of the M4 feature
    set. `eye_widen` and `blink_close` are both recoverable from blendshape
    coefficients that are already being computed and simply are not being read.
64. **The two saturated channels need recalibration, not more of the same.** The brow
    and mouth markers fire on most clips, which is why both their cue tests and their
    syntactic tests are null. Whatever M3 concluded about the interrogative/brow
    pairing is therefore a statement about a saturated instrument.
65. **The test's own calibration is asserted.** A permutation test that reports small
    p on data with no signal is worse than no test, so a null-data calibration runs in
    the suite: 20 independent draws on pure noise must give a median p near 0.5 and no
    more than 4 below 0.05.

---

## M4 — Factorized encoder: the gate is NOT met, and four instruments were wrong first

Run `m4-factorizer-001`. Artefact `artifacts/m4/factorizer.json`.
4 LOSO folds, 314 trainable windows from 200 clips, 30 epochs, 1 seed.

### The plan's vCLUB does not work, and that is a measured result

66. **The minibatch CLUB estimate was implemented and rejected.** With a bilinear
    critic, 256 samples, 8 dimensions, plus input normalisation, weight decay and a
    bounded critic output, the critic-minimised bound came out:

    | data | bound (nats) |
    |---|---|
    | independent | −18.29 |
    | weak dependence | −38.44 |
    | identical | −41.70 |

    **The ordering is inverted** — more dependence gave a *lower* bound. An
    unconstrained critic is rewarded for driving the expression to −∞. Replaced with a
    Jensen–Shannon MI bound via a joint-vs-product discriminator, which is stable.

67. **A dependence detector on this data must be evaluated held-out, and the size of
    the effect is the headline finding.** The same discriminator, scored on the rows it
    trained on, gave **0.9375 accuracy between two independent variables** — it had
    memorised 256 pairs. Scored on held-out draws from the same process: independent
    **0.4996**, weak 0.9679, dependent 0.9969. A probe fitted and scored on the same
    windows would have inflated the cross-prediction AUC that the M4 gate depends on,
    in the direction that makes a broken model look good.

68. **A first attempt at the held-out helper broke that guarantee in a new way.** The
    helpers resampled the two factors' rows *independently*, which destroys the joint —
    so the "positive" half of the evaluation was a product sample and the function
    measured nothing it claimed to. Now the holdout is an aligned split of the real
    pairs. A JSD value above `log 2` (physically impossible) is what exposed it.

### Three more instrument bugs, all caught by refusing to believe a number

69. **`auc()` silently returned values above 1 on multiclass labels.** Affect is
    8-class; the function treated anything that was not 0 as negative and anything not 1
    as positive, so classes 2–7 consumed rank space without being counted and the
    Mann-Whitney numerator stopped being bounded by the denominator. The first real M4
    run reported **AUC 3.996**. The function now raises on a non-binary label, and
    `cross_prediction_multiclass` does one-vs-rest per class with the **worst** class as
    the headline — separation must hold for every class to mean the factors share no
    affect, and a mean would let seven good classes carry one entangled one.
70. **The probe's own optimiser diverged.** A hand-rolled normalised gradient step of
    `0.5·√n` for 200 iterations produced scores that fed the same out-of-range AUC.
    Replaced with scikit-learn's convex solver, standardising on the fit split only.
71. **Neither adversarial term was in the objective.** `factorizer_loss` added the two
    direct losses and commented that the GRL heads' loss "is not minimised — its
    gradient is reversed", excluding them. The reversal was in the graph but not in the
    loss, so **the GRL heads were untrained**. This also dissolved an apparent finding:
    an earlier run showed "the adversarial head reads affect at 0.121 while a frozen
    probe reads the same factor at 0.871", which looked like a statement about
    adversarial defeat being achievable while the information remained. It was an
    untrained head. Both cross terms are now in the total.
72. **Two advertised ablation levers did not exist.** `w_affect_from_l` and
    `w_affect_from_p` were config fields the loss never read, so the `no_affect_from_p`
    variant reproduced `full` byte for byte and looked like a clean null. Replaced with
    `lambda_grl_a`, which is read, and the separate GRL directions are now ablatable.

### Result: the gate fails, and the failure is real

73. | variant | cross L→A | cross A→L | GRL head acc | signer control |
    |---|---|---|---|---|
    | **full** | **0.879** | 0.695 | 0.143 | 0.971 |
    | no_mi | 0.878 | 0.700 | 0.124 | 0.973 |
    | no_grl_on_a | 0.877 | 0.681 | 0.140 | 0.973 |
    | no_orthogonality | 0.903 | 0.665 | 0.143 | 0.976 |
    | no_separation | 0.898 | 0.617 | 0.162 | 0.979 |

    **The gate is cross-prediction AUC ≤ 0.60. Measured worst: 0.879. FAIL.** The
    signer control passes at 0.971, so the metric is sensitive here and the failure is
    the model's, not the instrument's.
74. **The separation terms work, monotonically, and the orthogonality penalty carries
    the effect.** Removing orthogonality moves cross L→A from 0.879 to 0.903; removing
    all separation pressure gives 0.898 with the GRL head rising from 0.143 to 0.162.
    That is a consistent, correctly-signed effect — and it is far short of the target.
75. **Neither task is learned, which is the binding constraint.** Balanced accuracy
    0.547 on the linguistic task (chance 0.50) and 0.173 on affect (chance 0.125 for
    8 classes). **Plain accuracy looks far better and is meaningless here: the
    majority-class predictor scores 0.863 and 0.334, and beats both this model and the
    entangled baseline.** Without a majority reference, "linguistic accuracy 0.809" in
    an earlier run would have read as a strong result. The gate metric is weighted F1,
    and both are now reported.
76. **The data cannot support the affect task as posed.** Requiring a single committed
    emotion per clip leaves **314 trainable windows from 200 clips** — 1,451 of 1,765
    windows are dropped because the annotators did not commit to one dominant
    expression. EmoSign's affect labels are multi-label by construction, and M4's
    8-class single-expression framing discards 82% of the available windows. This is a
    framing problem, not a capacity problem, and it is the first thing to fix.
77. **Honest status: the M4 claim is not established.** A factorized encoder, a working
    LOSO protocol, a falsifiable gate metric with a passing positive control, and an
    ablation in which every lever moves the target metric in the right direction — and
    a model that does not learn either task and does not separate. The negative result
    is on the *current configuration and data*, and the most likely cause is the label
    framing in (76), not the architecture. The next step is a multi-label affect
    objective and the 1,451 windows currently thrown away, not more separation terms.

---

## M4 rerun — multi-label affect. The data fix moved the metric a long way; the gate still fails.

Run `m4-factorizer-002`. Artefact `artifacts/m4/factorizer_multilabel.json`.
Same folds, same instruments, same schedule. **Only the affect target changed.**

78. **The framing was the bug, and fixing it recovered 82% of the data.** Requiring a
    single committed emotion per clip dropped **1,451 of 1,765 windows**, leaving 314.
    EmoSign's affect annotation is multi-label by construction — the clips routinely
    carry two or three emotions above threshold. Multi-hot targets with BCE over eight
    independent binary outputs keeps every clip: **1,765 windows, mean positive rate
    0.248** per label.

79. **Affect is still not learnable here, and now that is visible rather than hidden.**
    Micro-F1 **0.359** (0.391 at threshold 0.3), macro-F1 0.297 (0.371), balanced
    accuracy **0.497** — against a balanced-accuracy reference of 0.5, because a
    majority-class predictor scores exactly 0.5 by construction. Linguistic balanced
    accuracy 0.529 (chance 0.5). Neither task is learned.
80. **A reporting error caught in this run.** The first multi-label output printed
    "affect-majority-balanced 0.760" as the reference for a balanced accuracy of 0.497.
    That 0.760 is the majority *rate*, and no predictor's balanced accuracy can be
    compared to it — a majority predictor is at 0.5 by definition. Renamed
    `mean_majority_rate` and reported as class-imbalance context, with the 0.5
    reference stated in the output. This is the same failure shape as the earlier
    majority-class-accuracy trap, one level deeper.

81. **Separation improved substantially, and the fix is attributable.**

    | | single-expression | multi-label |
    |---|---|---|
    | trainable windows | 314 | **1,765** |
    | cross A→L (linguistic from z_A) | 0.695 | **0.505** |
    | cross L→A (affect from z_L) | 0.879 | **0.691** |
    | affect micro-F1 | n/a (314 windows) | **0.359** |
    | signer control | 0.971 | 0.973 |

    **cross A→L at 0.505 is chance.** The affect factor no longer carries linguistic
    information, and that is a real result: the affect-side gradient reversal is doing
    exactly what it was added to do. Affect remains partially readable from `z_L`
    (0.691), so the gate's ≤0.60 is still not met.

82. **Ablation, and every lever now moves the metric in the right direction:**

    | variant | cross L→A | cross A→L | micro-F1 | ling. balanced |
    |---|---|---|---|---|
    | **full** | **0.691** | **0.505** | **0.359** | 0.529 |
    | no_mi | 0.700 | 0.581 | 0.315 | 0.521 |
    | no_grl_on_a | 0.690 | 0.621 | 0.333 | 0.507 |
    | no_orthogonality | 0.686 | 0.584 | 0.315 | 0.499 |
    | no_separation | 0.707 | 0.585 | 0.299 | 0.490 |

    The full model is best on **all three** target metrics simultaneously — worst
    cross AUC, and best micro-F1. That is the first configuration where the separation
    terms do not cost task performance, which was the plan's kill-switch condition
    ("if AUC improves but wF1 drops, rewrite the claim"). It does not drop: removing any
    separation term lowers micro-F1 by 0.024–0.060.

83. **The gate is computed on the worst fold and still fails.**

    | fold | n_test | cross L→A | cross A→L | affect labels used | micro-F1 |
    |---|---|---|---|---|---|
    | Ben | 112 | **0.884** | 0.628 | 3 | 0.313 |
    | Cory | 788 | 0.647 | 0.546 | 8 | 0.338 |
    | Jonathan | 443 | 0.686 | **0.255** | 7 | 0.378 |
    | Rachel | 422 | 0.728 | 0.657 | 8 | 0.392 |

    The weighted mean cross L→A is 0.691; the **worst fold is 0.884** and that is what
    the gate uses. Ben is the problem, for a reason already measured: 7 clips, 112 test
    windows, and only **3 of 8 affect labels** reach the support threshold. A fold that
    cannot estimate most of its own labels cannot support a claim about them, and the
    signier control passes (0.973) so the failure is not a blind metric.
84. **Honest verdict: the gate is still not met, and the reason has moved.** The single
    biggest obstacle is gone — the framing bug is fixed, 5.6x more data is in use, and
    cross A→L reached chance. What remains is (a) the Ben fold, which is too small to
    estimate an 8-label cross-prediction and needs either a merged small-signer fold or
    an explicit "insufficient support" verdict rather than a number, and (b) genuine
    residual affect information in `z_L` at 0.691. Neither is fixed by more separation
    terms. (a) is a reporting decision; (b) may be real, given that M3 measured a
    head-shake/negation association and brow and mouth channels remain saturated.

---

## M5a — the ASLLRP token table parses cleanly and its frame indices do not fit our clips

### The data layer

78. **The file is not valid CSV, and the fix was a parser rather than a skip.** Gloss
    labels contain bare inner double quotes — `"5"ok, hey""` — which RFC 4180 requires to
    be doubled. Every standard reader fails on ~2 rows in 6,000. Skipping them would
    remove the most unusual glosses, which are exactly what a recognition model finds
    hardest, and the token count would still look plausible. `_split_row` opens a
    quoted region at a `"` after a delimiter and closes it at a `"` before one, treating
    any other `"` as data. Result: **17,522 tokens, 2,127 utterances, 1,956 gloss
    types, 0 malformed rows**, with all 761 quoted glosses preserved.

79. **One signer appears only in a `1-Name-topic` form.** Deriving the signer name set
    from the `Cory_2013-6-27_sc115` form left **all 4,389 of Ben's tokens unassigned**,
    which would have pooled them into a single anonymous leave-one-out group — the exact
    leak signer-disjoint splits exist to prevent, arriving through a parse that looked
    clean. Both collection forms are now resolved, giving **Ben 4,389, Cory 5,313,
    Jonathan 3,236, Rachel 4,584** tokens, 0 unmapped collections.

80. **A documented limitation of the dashed form.** The rule reads the token after the
    collection index, so in isolation `1-Introduction-x` yields `Introduction`; a name and
    a topic word are indistinguishable in one string. The guard therefore sits on the
    *derived set* — a leaked topic word would appear as an extra signer, splitting one
    person into two groups. Reported as `signer_set_plausible` and asserted to be exactly
    the four people.

### The blocker: token frame indices do not index the EmoSign clips

81. **Measured, and it would have produced a plausible CER about the wrong frames.** The
    ASLLRP token table's frame indices are **absolute positions in a long session
    recording**: the `containing utterance` start values run monotonically upward (5000,
    287, 1372, 1491, 1599, 1710, 3287, …) and the median utterance span is **5,123
    frames, about 2.8 minutes at 30 fps**. The EmoSign clips are **4.6-second excerpts**
    with a median of 109 frames.
82. **It is not a recoverable rescale.** The ratio of ASLLRP index to extracted frame
    count runs from **1.00 to 222.33** across 200 utterances with a coefficient of
    variation above 0.9, so no constant factor maps one to the other, and the excerpt's
    offset into its session is recorded nowhere in the data available.
83. **Consequence: 1,725 of 1,738 EmoSign-overlapping tokens have a frame range entirely
    outside the extracted video** — 99.3%. Only 1,738 of 17,522 tokens overlap at all,
    and those 1,738 are misaligned. A recogniser trained on them would learn the wrong
    frames and its error rate would describe the misalignment rather than the model.
84. **This is now a checked property, not a caveat in prose.** `asllrp.check_alignment`
    returns an `AlignmentReport` whose `aligned` flag must be consulted before these
    labels are used for training, with two tests: one asserting the current mismatch (and
    documenting that a pass would mean the data changed and the plan should be updated),
    and one asserting the check can say *yes*, so it is a check and not a constant.
85. **The route forward is the other column.** The table carries
    `Sign video filename`: **17,522 isolated sign clips, each with a single gloss and its
    own frame bounds**. Those are downloadable and alignable by construction — no session
    offset to recover — and are the correct substrate for M5a. Cost is a download of the
    sign-clip set plus MediaPipe extraction over short single-sign videos, which is hours
    of work and not minutes, so it is scheduled rather than started silently.

---

## M7 — a runnable demo, and two things it deliberately refuses to do

`seam serve` / `make serve`. 21 tests in `tests/test_serve.py` plus `make serve-check`,
which boots the server and asserts the HTTP contract.

86. **Video never leaves the browser tab.** MediaPipe Tasks runs client-side in WASM and
    the page posts the 52 blendshape coefficients and the head transform. The server
    receives numbers, so there is nothing to store or leak. The cost of that choice is
    stated rather than hidden: **this process cannot verify what the client measured**,
    so every response carries the provenance of its inputs and the UI displays it.
87. **Marker magnitudes come from the same code the experiments use**
    (`seam.features.markers`), so a demo figure and a paper figure cannot come from
    different implementations. Measured server latency **9 ms** per window.
88. **The demo refuses to show predictions this build cannot support, and says why.** The
    panel reads "affect — withheld: M4's factorized encoder was measured at 0.497
    balanced accuracy against a 0.5 reference, so it does not learn" and "gloss
    recognition — withheld: M5a's labels are misaligned". An empty panel is honest; a
    confident wrong number is not, and a demo that displayed 0.497-derived affect labels
    would be presenting a measurement that carries no information.
89. **A blind feature is shown as blind, not as absent.** The response carries each
    marker's zero fraction, so `head_nod` (zero on 83% of clips) renders as "blind —
    cannot discriminate" rather than as a low magnitude, keeping "no marker" and "this
    instrument cannot tell" distinct on screen.
90. **Malformed windows are rejected with an actionable message, not repaired.** A
    51-coefficient window is a specific trap: MediaPipe emits 52 starting with
    `_neutral`, and an implementation that dropped it would shift every marker threshold
    by one while looking healthy. All four bad cases return **400** with a message
    (`51 coeffs`, `3 frames`, `fps=0`, unknown utterance) rather than 500 or a
    confidently wrong result.
91. **A missing head pose stays missing.** Zero-filling it is not a neutral default, it
    is a *plausible* head pose, and it makes "no tracking" indistinguishable from
    "perfectly still" — which silently pins head_shake and head_nod to zero.
92. **One bug found by the smoke test, which is why it exists.** The web asset path
    resolved to `src/` instead of the package, so the app started cleanly, logged
    "app built", and served a **44-byte 404 page**. A demo that starts and shows
    nothing is worse than one that refuses to start, so `make serve-check` now boots
    the server and asserts the page is real HTML carrying both halves of the client
    contract (`tasks-vision` and `api/analyse`), plus the health endpoint's declared
    coefficient count.
93. **Verified end to end over HTTP:** `GET /` 9,126 bytes of real HTML;
    `POST /api/analyse` → 6 markers in 9 ms; linguistic annotation resolved for
    utterance 24363254 (`GROW TALK of GROW SIGN DCL:B"pathway of sign language"`, 9
    tokens) from the real ASLLRP annotation; every malformed case 400.

### What the demo is, precisely

A live monitor of non-manual signal with its linguistic context attached: real
MediaPipe perception in the browser, real marker magnitudes from the research code,
real ASLLRP annotation for a supplied utterance id, and explicit refusals with measured
reasons for the two things this build cannot yet do. It is not a claim about affect
recognition or sign recognition, because neither is established.
