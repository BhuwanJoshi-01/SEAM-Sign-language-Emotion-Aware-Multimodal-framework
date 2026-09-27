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
