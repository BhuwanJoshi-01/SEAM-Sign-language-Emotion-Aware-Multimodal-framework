"""M3 grounding report: do our features recover the cues the annotators named?

The feature set in `seam.features` was assembled by assumption. This tests it
against the 600 free-text cue strings the three Deaf annotators wrote for the 200
EmoSign clips: for each cue they named, does the feature that ought to respond to
it actually separate those clips from the rest?

Design constraints, all of them earned:

* **The cue-to-feature map is declared in `seam.features.cues`, not chosen here.**
  The expected feature for each cue is fixed before any number is computed, so a cue
  whose feature shows nothing is a statement about the detector rather than a search
  for a feature that happens to work.

* **Duration is controlled.** Learned the hard way in M3: every syntactic label
  tracks clip length (interrogative r_pb = +0.68), and a long clip moves every
  time-normalised feature. A cue like "speed" is the obvious case - longer clips
  give more room to look fast - so the association is a partial correlation against
  log-duration.

* **Both sides are pseudo-labels, and the report says so.** The cues come from
  annotator free text parsed by a regex vocabulary, not from a controlled
  non-manual annotation. That is the best available ground truth and it is still
  soft: it validates the feature set against human attention, not against a
  linguistically defined label.

* **Nothing is reported without its MDE.** A cue with no separation is reported as
  a null with the size of the effect that was ruled out, and a cue with too few
  clips is reported as too few rather than as a p-value.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:  # pragma: no cover - bootstrap
    sys.path.insert(0, str(_ROOT / "src"))

from seam.features import cues as C
from seam.features import markers as VM
from seam.features import prosody as PR
from seam.logging import get, setup
from seam.paths import artifacts_root, default_data_root

log = get("cue_grounding")

from seam.perception.tasks_api import PART_SLICES as PART_SLICES_LOCAL

#: Bonferroni across the number of testable cues actually run.
ALPHA = 0.05


@dataclass(slots=True)
class ClipFeatures:
    """Everything extracted for one clip, keyed by the feature names the cue map uses."""

    utterance_id: str
    duration_s: float
    marker_magnitude: dict[str, float]
    prosody: dict[str, float]
    hand_visible: float


def _pose_slice() -> tuple[int, int]:
    """The pose block, which ``prosody.hand_centroids`` uses to find the wrist.

    ``summarize`` takes the *full* landmark array plus the pose slice: hand
    centroids are computed wrist-relative, so the wrist index comes from pose and
    the hand region alone is not a valid input.
    """
    from seam.perception.tasks_api import PART_SLICES

    return PART_SLICES["pose"]


def clip_features(shard: Path) -> ClipFeatures | None:
    """Extract the feature set for one landmark shard.

    Returns ``None`` when no hand is visible enough to define a trajectory, because
    the whole prosody family is hand-trajectory based and reporting it over a clip
    with no hand would be reporting a number with no referent.
    """
    data = np.load(shard)
    bs = np.asarray(data["blendshapes"], dtype=np.float64)
    lm = np.asarray(data["landmarks"], dtype=np.float64)
    rot = np.asarray(data["head_rotation"], dtype=np.float64)
    fps = 25.0
    side = shard.with_suffix(".json")
    if side.is_file():
        meta = json.loads(side.read_text())
        fps = float(meta.get("native_fps") or fps)

    magnitude = VM.clip_magnitude(VM.signals(bs, rotation=rot, fps=fps), fps)
    presence = np.asarray(data["presence"], dtype=bool) if "presence" in data else None
    hand_vis = float(presence[:, 1:3].mean()) if presence is not None else 1.0

    n = len(lm)
    out: dict[str, float] = {}
    hand_lo, hand_hi = PART_SLICES_LOCAL["left_hand"]
    visible = lm[:, hand_lo:hand_hi, :]
    vis = np.isfinite(visible[:, :, 0]) & (visible[:, :, 0] != 0)
    if float(vis.mean()) < 0.25:
        # The whole prosody family is hand-trajectory based; reporting it over a clip
        # with no hand would be reporting a number with no referent.
        return None

    # Gap-fill across the full landmark array by carrying the last observation, so a
    # single dropped frame does not make the trajectory NaN and silently zero every
    # downstream feature. Points never seen stay at 0, which the hand-visibility
    # fraction above has already bounded.
    finite = np.isfinite(lm[:, :, 0]) & (lm[:, :, 0] != 0)
    filled = np.where(finite[..., None], lm, np.nan)
    for i in range(filled.shape[1]):
        col = filled[:, i, :]
        mask = np.isnan(col[:, 0])
        if mask.all():
            filled[:, i, :] = 0.0
            continue
        idx = np.where(~mask, np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        filled[:, i, :] = col[idx]

    # summarize() takes the full landmark array and the hand slice; the
    # gap-filled array is passed so a single dropped frame does not zero the
    # whole feature family.
    summ = PR.summarize(filled, _pose_slice(), fps).as_dict()
    out["speed"] = float(summ["speed"])
    out["peak_speed"] = float(summ["peak_speed"])
    out["amplitude"] = float(summ["amplitude"])
    out["volume"] = float(summ["volume"])
    out["repetition"] = float(summ["repetition"])
    out["jerk"] = float(summ["jerk"])
    out["pause_ratio"] = float(summ["pause_fraction"])
    out["pause_mean"] = float(summ["pause_mean"])

    return ClipFeatures(
        utterance_id=shard.stem,
        duration_s=n / fps,
        marker_magnitude=magnitude,
        prosody=out,
        hand_visible=hand_vis,
    )


def _pb(y: np.ndarray, x: np.ndarray) -> float:
    a, b = x[y > 0], x[y == 0]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    return float((a.mean() - b.mean()) / np.sqrt(pooled)) if pooled > 0 else 0.0


def _partial_pb(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    """Point-biserial, controlling for ``z``; degenerate to plain when ``z`` is flat."""
    r_xy = _pb(y, x)
    if z.std() == 0 or x.std() == 0 or y.std() == 0:
        return r_xy
    r_xz = float(np.corrcoef(x, z)[0, 1])
    r_yz = float(np.corrcoef(y, z)[0, 1])
    return float((r_xy - r_xz * r_yz) / np.sqrt(max((1 - r_xz**2) * (1 - r_yz**2), 1e-12)))


def test_cue(
    records: list[ClipFeatures],
    present: list[bool],
    feature: str,
    *,
    n_perm: int = 20_000,
    seed: int = 0,
) -> dict[str, object]:
    """One cue against one feature, duration-controlled, by label permutation."""

    def get_value(r: ClipFeatures) -> float:
        if feature in VM.MARKERS:
            return float(r.marker_magnitude.get(feature, 0.0))
        return float(r.prosody.get(feature, np.nan))

    vals = np.array([get_value(r) for r in records], dtype=float)
    ok = np.isfinite(vals)
    y = np.array(present, dtype=float)[ok]
    x = vals[ok]
    dur = np.array([r.duration_s for r in records], dtype=float)[ok]
    logd = np.log(np.maximum(dur, 1e-6))

    n = len(x)
    base = {
        "feature": feature,
        "n": n,
        "n_present": int(y.sum()),
        "n_absent": int((y == 0).sum()),
    }
    if n < 12 or y.sum() < 5 or (y == 0).sum() < 5:
        return {**base, "interpretable": False, "note": "too few clips in one arm"}
    if x.std() == 0:
        return {**base, "interpretable": False, "note": f"{feature} is constant across clips"}

    # Can this feature discriminate at all? A cue whose feature is saturated or
    # zero-inflated cannot test anything, and reporting its null as evidence would
    # be indistinguishable from reporting that the annotators were wrong. M3
    # already established that the blendshape markers fire on 79-95% of clips, so
    # this is a known blind spot rather than a surprise - but it has to be checked
    # per feature, not assumed.
    zero_frac = float(np.mean(np.isclose(x, 0.0)))
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    med = float(np.median(x))
    saturated = iqr <= 1e-9 or (med != 0.0 and iqr / abs(med) < 0.05)
    zero_inflated = zero_frac > 0.60
    blind = saturated or zero_inflated
    if blind:
        why = (
            f"feature is zero on {zero_frac:.0%} of clips"
            if zero_inflated
            else f"feature has no interquartile spread (IQR={iqr:.3g})"
        )
        return {
            **base,
            "interpretable": False,
            "note": f"cannot discriminate: {why}",
            "zero_fraction": round(zero_frac, 4),
            "iqr": round(iqr, 6),
        }

    obs = _partial_pb(y, x, logd)
    rng = np.random.default_rng(seed)
    null = np.array([_partial_pb(rng.permutation(y), x, logd) for _ in range(n_perm)])
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    mde = 1.96 * float(null.std(ddof=1))
    return {
        **base,
        "interpretable": True,
        "zero_fraction": round(zero_frac, 4),
        "iqr": round(iqr, 6),
        "r_partial": round(obs, 4),
        "r_uncontrolled": round(_pb(y, x), 4),
        "perm_p": round(p, 5),
        "mde_80pct": round(mde, 4),
        "median_present": round(float(np.median(x[y > 0])), 4),
        "median_absent": round(float(np.median(x[y == 0])), 4),
    }


def build(landmark_dir: Path) -> dict[str, object]:
    cov = C.coverage()
    clip_cue_map = C.clip_cues()
    feats: dict[str, ClipFeatures] = {}
    for shard in sorted(landmark_dir.glob("*.npz")):
        f = clip_features(shard)
        if f is not None:
            feats[f.utterance_id] = f
    log.info("features for %d/%d clips", len(feats), len(clip_cue_map))

    rows: list[dict[str, object]] = []
    skipped: list[str] = []
    for spec in C.MOTOR_CUES:
        if not spec.expect:
            skipped.append(spec.name)
            continue
        ids = sorted(set(feats) & set(clip_cue_map))
        present = [clip_cue_map[i].has(spec.name) for i in ids]
        recs = [feats[i] for i in ids]
        for feature in spec.expect:
            row = test_cue(recs, present, feature)
            row["cue"] = spec.name
            row["rationale"] = spec.rationale
            rows.append(row)

    m = sum(1 for r in rows if r.get("interpretable"))
    for r in rows:
        if r.get("interpretable") and m > 1:
            r["p_bonferroni"] = round(min(1.0, float(r["perm_p"]) * m), 5)
            r["recovered"] = bool(r["p_bonferroni"] < ALPHA)

    return {
        "coverage": cov,
        "n_clips_with_features": len(feats),
        "expectations": C.expectations(),
        "cue_tests": rows,
        "n_tests": m,
        "alpha": ALPHA,
        "unmapped_cues": C.unmapped_cues(),
        "unmapped_note": (
            "Cues the Deaf annotators actually used that no current feature can test. "
            "This is a specification derived from the data, not a list of wishes."
        ),
        "blind_channels": {
            "explanation": (
                "Cues whose expected feature cannot discriminate across clips. These are "
                "not nulls: a saturated or zero-inflated feature cannot test a cue, and "
                "reporting its non-separation as evidence would be indistinguishable from "
                "claiming the annotators were wrong."
            ),
            "cues": sorted(
                {
                    str(r["cue"])
                    for r in rows  # type: ignore[union-attr]
                    if not r.get("interpretable") and "cannot discriminate" in str(r.get("note"))
                }
            ),
        },
        "caveat": (
            "Both sides are soft. The cue labels come from annotator free text matched "
            "by a regex vocabulary, not from controlled non-manual annotation, so a pass "
            "validates the feature set against human attention rather than against a "
            "linguistic definition. Every statistic is a partial correlation on "
            "log-duration with a permutation test, because longer clips move every "
            "time-normalised feature."
        ),
    }


def render(rep: dict[str, object]) -> str:
    cov = rep["coverage"]  # type: ignore[index]
    lines = [
        f"annotator strings: {cov['n_strings']}  |  with a motor cue: {cov['n_with_motor_cue']} "
        f"({cov['motor_coverage']:.0%})  |  affective-only: {cov['n_affective_only']}",
        f"clips with extracted features: {rep['n_clips_with_features']}",
        "",
        f"{'cue':<18} {'feature':<12} {'n+':>4} {'n-':>4} {'r':>7} {'p':>8} {'x m':>8} {'MDE':>6}",
    ]
    for r in rep["cue_tests"]:  # type: ignore[union-attr]
        if not r.get("interpretable"):
            lines.append(
                f"  {r['cue']:<16} {r['feature']:<12} {r['n_present']:>4} {r['n_absent']:>4}"
                f"   -- not interpretable: {r.get('note', '')}"
            )
            continue
        mark = " *" if r.get("recovered") else ""
        lines.append(
            f"  {r['cue']:<16} {r['feature']:<12} {r['n_present']:>4} {r['n_absent']:>4} "
            f"{r['r_partial']:>7.3f} {r['perm_p']:>8.4f} {r.get('p_bonferroni', 0):>8.4f} "
            f"{r['mde_80pct']:>6.2f}{mark}"
        )
    blind = rep.get("blind_channels", {}).get("cues", [])  # type: ignore[union-attr]
    if blind:
        lines += [
            "",
            "BLIND (feature cannot discriminate, so no conclusion is available):",
            "  " + ", ".join(blind),
        ]
    if rep["unmapped_cues"]:  # type: ignore[union-attr]
        lines += [
            "",
            "named by annotators, no feature to test them:",
            "  " + ", ".join(rep["unmapped_cues"]),  # type: ignore[union-attr]
        ]
    lines += ["", f"caveat: {rep['caveat']}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--landmark-dir", default=str(default_data_root() / "emosign" / "landmarks"))
    ap.add_argument("--out", default=str(artifacts_root() / "audit" / "cue_grounding.json"))
    args = ap.parse_args()
    setup("INFO")

    rep = build(Path(args.landmark_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print()
    print(render(rep))
    print()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
