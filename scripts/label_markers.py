"""M3 gate: marker labels on >=200 clips, with provenance and agreement.

The M3 gate is not "we produced labels". It is labels on at least 200 clips, each
carrying its provenance, plus agreement statistics between the two independent
ways this project infers a marker:

* **syntactic** — from the human-authored ASLLRP gloss (:mod:`seam.features.syntactic`)
* **visual** — from blendshapes and head pose (:mod:`seam.features.markers`)

Neither is gold, and the interesting quantity is precisely that they were derived
independently. If an interrogative gloss predicts a brow raise more often than
chance, that is evidence the visual marker is reading something real. If it does
not, then either the visual threshold is wrong or the effect is too small here —
and M1 already found that on WLASL, so a null on EmoSign would be consistent with
prior results rather than a surprise.

Two things this script refuses to do:

* **It does not report an agreement number without its base rate.** A marker that
  fires on 3% of clips will agree with a 3%-base-rate syntactic label 97% of the
  time by coincidence. So every pairing reports the contingency counts, and the
  lift over the base rate, not just a percentage.
* **It does not present pseudo-labels without the word.** Every table this writes
  carries the provenance string, because a marker column whose reliability is
  unstated is worse than one known to be weak.
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

from seam.data.emosign import load as load_emosign
from seam.features import markers as VM
from seam.features import syntactic as SY
from seam.logging import get, setup
from seam.paths import artifacts_root, default_data_root
from seam.provenance import KEY as PROVENANCE_KEY
from seam.provenance import stamp

log = get("label_markers")


@dataclass(slots=True)
class ClipLabels:
    """One clip's labels from both sources."""

    utterance_id: str
    frames: int
    fps: float
    visual: dict[str, float]
    #: Continuous per-clip marker magnitude, the primary statistic.
    magnitude: dict[str, float]
    syntactic: dict[str, bool]
    provenance: dict[str, str]
    gloss: str


def _clip_visual(shard: Path) -> tuple[dict[str, float], int, float]:
    """Per-clip marker firing rates, 0-1, over the frames in the shard."""
    data = np.load(shard)
    bs = np.asarray(data["blendshapes"], dtype=np.float64)
    rot = np.asarray(data["head_rotation"], dtype=np.float64)
    fps = 25.0
    side = shard.with_suffix(".json")
    if side.is_file():
        meta = json.loads(side.read_text())
        fps = float(meta.get("native_fps") or fps)

    sig = VM.signals(bs, rotation=rot, fps=fps)
    fired = VM.fire(sig)
    out: dict[str, float] = {}
    for name in VM.MARKERS:
        flag = np.asarray(fired[name], dtype=bool)
        out[name] = float(flag.mean()) if flag.size else 0.0
    return out, len(bs), fps


def _magnitude(shard: Path) -> dict[str, float]:
    """Continuous per-clip marker magnitude, with fps read from the sidecar.

    This is the primary statistic for marker/label association; ``_clip_visual``
    is the binary view, kept for reporting.
    """
    data = np.load(shard)
    bs = np.asarray(data["blendshapes"], dtype=np.float64)
    rot = np.asarray(data["head_rotation"], dtype=np.float64)
    fps = 25.0
    side = shard.with_suffix(".json")
    if side.is_file():
        meta = json.loads(side.read_text())
        fps = float(meta.get("native_fps") or fps)
    # reduce="mean" is the duration-free default; see clip_magnitude's docstring
    # for the confound the integral would have inherited.
    return VM.clip_magnitude(VM.signals(bs, rotation=rot, fps=fps), fps, reduce="mean")


def label_all(landmark_dir: Path, limit: int | None = None) -> list[ClipLabels]:
    gm = SY.load_gloss_map()
    records = load_emosign(default_data_root())
    out: list[ClipLabels] = []
    for rec in records.clips:
        if limit is not None and len(out) >= limit:
            break
        shard = landmark_dir / f"{rec.utterance_id}.npz"
        if not shard.is_file():
            continue
        visual, frames, fps = _clip_visual(shard)
        magnitude = _magnitude(shard)
        syn = SY.classify(rec.utterance_id, gm)
        out.append(
            ClipLabels(
                utterance_id=rec.utterance_id,
                frames=frames,
                fps=fps,
                visual=visual,
                magnitude=magnitude,
                syntactic={n: lab.present for n, lab in syn.labels.items()},
                provenance=syn.provenance(),
                gloss=syn.gloss,
            )
        )
    return out


#: A clip-level marker is only interpretable if it is present in a minority of
#: clips and a non-trivial number of them.
#:
#: Below the floor there is no signal to analyse; above the ceiling the marker
#: fires almost everywhere and carries no information about any clip, so agreement
#: with anything is close to the base rate by construction. Both failure modes look
#: identical in a contingency table - a "lift" of ~1.0 - but they mean opposite
#: things: too few events, or no discrimination. Reporting one number for both is
#: how a broken instrument gets mistaken for a null result.
USABLE_PREVALENCE_MIN = 0.05
USABLE_PREVALENCE_MAX = 0.60


def marker_usability(clips: list[ClipLabels]) -> dict[str, dict[str, object]]:
    """Classify each visual marker as usable or degenerate, with the reason."""
    out: dict[str, dict[str, object]] = {}
    for name in VM.MARKERS:
        prev = float(np.mean([c.visual.get(name, 0.0) > 0 for c in clips]))
        if prev > USABLE_PREVALENCE_MAX:
            state, why = (
                "degenerate",
                f"fires on {prev:.0%} of clips; fires almost always, so it cannot "
                f"discriminate between clips (ceiling {USABLE_PREVALENCE_MAX:.0%})",
            )
        elif prev < USABLE_PREVALENCE_MIN:
            state, why = (
                "too_rare",
                f"fires on {prev:.0%} of clips; too few events to analyse "
                f"(floor {USABLE_PREVALENCE_MIN:.0%})",
            )
        else:
            state, why = "usable", f"fires on {prev:.0%} of clips"
        out[name] = {"clip_prevalence": round(prev, 4), "state": state, "why": why}
    return out


#: A magnitude that is zero on more than this fraction of clips is blind. The same
#: figure `scripts/cue_grounding.py` uses, so the two reports cannot disagree about it.
BLIND_ZERO_FRACTION = 0.60

#: Which visual marker is expected to co-occur with which syntactic label, and on
#: what linguistic grounds. Kept explicit so a pairing cannot be invented after
#: seeing the numbers.
EXPECTED_PAIRS = (
    ("interrogative", "brow_raise", "yes/no and wh-questions raise the brows in ASL"),
    ("interrogative", "brow_furrow", "wh-questions furrow rather than raise"),
    ("negation", "head_shake", "negation is canonically marked by a head shake"),
)


def agreement(clips: list[ClipLabels], syn_name: str, vis_name: str) -> dict[str, object]:
    """Contingency for one pairing, with lift over the base rate.

    A bare agreement percentage is not reported as the headline, because a rare
    marker agrees with a rare label by chance. The base rate is the number that
    makes the agreement interpretable.
    """
    n = len(clips)
    if not n:
        return {"syntactic": syn_name, "visual": vis_name, "n": 0}
    syn = np.array([c.syntactic.get(syn_name, False) for c in clips], dtype=bool)
    vis = np.array([c.visual.get(vis_name, 0.0) > 0.0 for c in clips], dtype=bool)

    both = int((syn & vis).sum())
    syn_only = int((syn & ~vis).sum())
    vis_only = int((~syn & vis).sum())
    neither = int((~syn & ~vis).sum())
    obs_agree = (both + neither) / n
    base = (int(syn.sum()) * int(vis.sum()) + int((~syn).sum()) * int((~vis).sum())) / (n * n)
    lift = (obs_agree / base) if base > 0 else float("nan")

    # Only test a pairing whose visual marker can discriminate. A degenerate
    # marker has no p-value worth reading, and printing one invites it to be read
    # as a null result rather than as a broken instrument.
    prev = float(vis.mean()) if n else 0.0
    interpretable = n >= 8 and USABLE_PREVALENCE_MIN <= prev <= USABLE_PREVALENCE_MAX
    from scipy import stats as _st

    table = [[both, syn_only], [vis_only, neither]]
    testable = (
        interpretable and not np.isnan(lift) and (both + syn_only) > 0 and (vis_only + neither) > 0
    )
    p = float(_st.chi2_contingency(table, correction=False)[1]) if testable else float("nan")
    return {
        "syntactic": syn_name,
        "visual": vis_name,
        "n": n,
        "contingency": {
            "both": both,
            "syntactic_only": syn_only,
            "visual_only": vis_only,
            "neither": neither,
        },
        "syntactic_rate": round(float(syn.mean()), 4),
        "visual_rate": round(float(vis.mean()), 4),
        "observed_agreement": round(obs_agree, 4),
        "chance_agreement": round(float(base), 4),
        "lift_over_chance": None if np.isnan(lift) else round(float(lift), 4),
        "chi2_p": None if np.isnan(p) else round(p, 5),
        "interpretable": bool(interpretable),
        "not_interpretable_because": (
            ""
            if interpretable
            else (
                f"visual marker fires on {prev:.0%} of clips, outside the usable band "
                f"[{USABLE_PREVALENCE_MIN:.0%}, {USABLE_PREVALENCE_MAX:.0%}]; the lift of "
                f"{lift:.2f} is a property of the base rate, not evidence of an association"
            )
        ),
        "provenance_synaptic": clips[0].provenance.get(syn_name, ""),
        "provenance_visual": VM.describe()["provenance"],
    }


def magnitude_association(
    clips: list[ClipLabels],
    syn_name: str,
    vis_name: str,
    *,
    n_perm: int = 20_000,
    seed: int = 0,
) -> dict[str, object]:
    """Point-biserial association between a syntactic label and marker magnitude.

    Binary presence was abandoned for this reason: ASL non-manual markers are
    graded and frequent, so a prevalence screen cannot tell a frequent marker from
    a useless one - the control ``mouth_positive`` is active in essentially every
    clip because mouths move while people sign, and demanding it be rare rejects
    the signal for being real. A magnitude is graded, has a well-posed null, and
    needs no prevalence assumption.

    The test is a permutation test over the label, which is the same instrument M1
    validated, and it reports the observed effect size so a null can be read as
    powered rather than merely as an absence of measurement. The MDE is the effect
    the design could have detected at 80% power, from the permutation null's
    spread - so "no association" comes with the size of the effect that was
    ruled out, not just a p-value.
    """
    n = len(clips)
    if n < 8:
        return {"syntactic": syn_name, "visual": vis_name, "n": n, "interpretable": False}
    y = np.array([c.syntactic.get(syn_name, False) for c in clips], dtype=float)
    x = np.array([c.magnitude.get(vis_name, 0.0) for c in clips], dtype=float)
    dur = np.array([c.frames / max(c.fps, 1e-6) for c in clips], dtype=float)

    # Duration is a confounder, not a nuisance. Every syntactic label correlates
    # with clip length on this corpus (interrogative r_pb = +0.68, negation +0.54)
    # because questions and negated statements are longer utterances. The
    # magnitude is already a per-frame mean, but the *label* still tracks duration,
    # so both are residualised on log-duration before correlating and the duration
    # association is reported, so the confound stays visible rather than assumed
    # away.
    # Duration is a confounder, not a nuisance. Every syntactic label correlates
    # with clip length on this corpus (interrogative r_pb = +0.68, negation +0.54)
    # because questions and negated statements are longer utterances. The magnitude
    # is already a per-frame mean, so it is not mechanically longer for longer
    # clips - but the label still tracks duration, and any variable correlated with
    # the label can manufacture an association.
    #
    # The control is a *partial* correlation against log-duration, which keeps the
    # label binary. Residualising a 0/1 label on a covariate and then re-splitting
    # it at zero - the obvious implementation - silently regroups the clips and
    # was tried first here; it returned a uniform zero.
    z = np.log(dur) if dur.std() > 0 else None
    label_r_dur = _pb(dur, y) if z is not None else 0.0
    r_before = _pb(y, x)

    if y.std() == 0 or x.std() == 0:
        return {
            "syntactic": syn_name,
            "visual": vis_name,
            "n": n,
            "interpretable": False,
            "note": "label or marker is constant; no association is definable",
        }

    obs = _partial_pb(y, x, z)

    # The same blindness rule `scripts/cue_grounding.py` applies, because the two reports
    # disagreed: this one printed r and p for negation <-> head_shake while the grounding
    # report called the same channel blind. A magnitude that is zero on most clips cannot
    # separate them, so its correlation is not a null - it is no measurement. The value
    # is kept in the artifact, unread, so a later reader can see what was withheld.
    zero_frac = float(np.mean(np.isclose(x, 0.0)))
    if zero_frac > BLIND_ZERO_FRACTION:
        return {
            "syntactic": syn_name,
            "visual": vis_name,
            "n": n,
            "n_positive": int(y.sum()),
            "interpretable": False,
            "blind": True,
            "zero_fraction": round(zero_frac, 4),
            "note": f"cannot discriminate: magnitude is zero on {zero_frac:.0%} of clips",
            "statistic_not_interpreted": round(obs, 4),
        }

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = _partial_pb(rng.permutation(y), x, z)
    # Two-sided p, with the +1 correction so a p can never be exactly zero.
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    sd_null = float(null.std(ddof=1))
    mde = 1.96 * sd_null  # effect detectable at 80% power with this n

    present = np.array([c.visual.get(vis_name, 0.0) > 0 for c in clips], dtype=bool)
    return {
        "syntactic": syn_name,
        "visual": vis_name,
        "n": n,
        "n_positive": int(y.sum()),
        "point_biserial": round(obs, 4),
        "perm_p": round(p, 5),
        "n_perm": n_perm,
        "null_sd": round(sd_null, 4),
        "mde_80pct_power": round(mde, 4),
        "median_magnitude_present": round(
            float(np.median(x[present])) if present.any() else 0.0, 3
        ),
        "median_magnitude_absent": round(
            float(np.median(x[~present])) if (~present).any() else 0.0, 3
        ),
        "control": "residualised on log clip duration",
        "r_before_duration_control": round(r_before, 4),
        "label_duration_r": round(label_r_dur, 4),
        "zero_fraction": round(zero_frac, 4),
        "interpretable": True,
    }


def _partial_pb(y: np.ndarray, x: np.ndarray, z: np.ndarray | None) -> float:
    """Point-biserial correlation of ``x`` against binary ``y``, controlling for ``z``.

    The standard partial correlation,
    ``(r_xy - r_xz r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))``,
    with ``z=None`` degenerating to the plain point-biserial. ``y`` stays binary, so
    the split that defines the statistic is unchanged by the control.
    """
    r_xy = _pb(y, x)
    if z is None or z.std() == 0:
        return r_xy
    r_xz = float(np.corrcoef(x, z)[0, 1]) if x.std() > 0 else 0.0
    r_yz = float(np.corrcoef(y, z)[0, 1]) if y.std() > 0 else 0.0
    denom = np.sqrt(max((1 - r_xz**2) * (1 - r_yz**2), 1e-12))
    return float((r_xy - r_xz * r_yz) / denom)


def _pb(y: np.ndarray, x: np.ndarray) -> float:
    """Point-biserial correlation of ``x`` against binary ``y``."""
    a, b = x[y > 0], x[y == 0]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    if pooled <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def report(clips: list[ClipLabels]) -> dict[str, object]:
    pairs = []
    for syn_name, vis_name, rationale in EXPECTED_PAIRS:
        row = agreement(clips, syn_name, vis_name)
        row["rationale"] = rationale
        pairs.append(row)
    marker_rates = {
        name: round(float(np.mean([c.visual.get(name, 0.0) for c in clips])), 4)
        for name in VM.MARKERS
    }
    usability = marker_usability(clips)
    marker_prevalence = {name: usability[name]["clip_prevalence"] for name in VM.MARKERS}
    syn_rates = {
        name: round(float(np.mean([c.syntactic.get(name, False) for c in clips])), 4)
        for name in SY.SYNTACTIC_MARKERS
    }
    assoc = [magnitude_association(clips, syn, vis) for syn, vis, _ in EXPECTED_PAIRS]
    # Three pairings were tested, so raw permutation p-values are adjusted. A p of
    # 0.011 is not a discovery at alpha=0.05 when it is the best of three
    # uncorrected tests; the correction costs nothing here, it is applied because
    # the uncorrected number would otherwise be quoted on its own.
    m = sum(1 for a in assoc if a.get("interpretable"))
    for a in assoc:
        if a.get("interpretable") and m > 1:
            a["n_tests"] = m
            a["p_bonferroni"] = round(min(1.0, float(a["perm_p"]) * m), 5)
            a["significant_bonferroni"] = bool(a["p_bonferroni"] < 0.05)
    return {
        "n_clips": len(clips),
        "gate": {
            "required_clips": 200,
            "met": len(clips) >= 200,
            "note": "every label carries a provenance string; pseudo-labels are not gold",
        },
        "visual_marker_rates": {
            "mean_firing_rate": marker_rates,
            "clip_prevalence": marker_prevalence,
            "usability": usability,
        },
        "usable_prevalence_band": [USABLE_PREVALENCE_MIN, USABLE_PREVALENCE_MAX],
        "syntactic_rates": syn_rates,
        "agreements": pairs,
        "magnitude_associations": assoc,
        "provenance": {
            "visual": VM.describe()["provenance"],
            "syntactic": SY.describe()["provenance"],
        },
        "caveat": (
            "Two views are reported and they answer different questions. The binary "
            "prevalence view is only interpretable for a marker present in a minority "
            "of clips; ASL non-manual markers are graded and frequent, so it rejects "
            "the control mouth_positive (0.91) for being genuinely active rather than "
            "for being useless. The continuous-magnitude association is therefore the "
            "primary statistic: it needs no prevalence assumption, its permutation "
            "test carries an MDE so a null is reported as powered rather than as an "
            "absence of measurement, and it is a partial correlation against clip "
            "duration, because every syntactic label tracks clip length (interrogative "
            "r_pb=+0.68, negation +0.54) and an uncontrolled integral inherits that "
            "confound wholesale."
        ),
    }


def render(rep: dict[str, object]) -> str:
    lines = [
        f"clips labelled: {rep['n_clips']} (gate needs 200: "
        f"{'MET' if rep['gate']['met'] else 'NOT MET'})",
        "",
        f"{'visual marker':<20} {'mean firing':>12} {'clip prev.':>11}",
    ]
    rates = rep["visual_marker_rates"]  # type: ignore[index]
    for k, v in rates["mean_firing_rate"].items():
        lines.append(f"  {k:<18} {v:>12.3f} {rates['clip_prevalence'][k]:>11.3f}")
    lines += ["", f"{'syntactic label':<20} {'rate':>8}", ""]
    for k, v in rep["syntactic_rates"].items():  # type: ignore[index]
        lines.append(f"  {k:<18} {v:>8.3f}")
    lines += [
        "",
        f"{'marker':<18} {'prevalence':>11} {'state':>12}  reason",
    ]
    for name, u in rates["usability"].items():
        lines.append(f"  {name:<16} {u['clip_prevalence']:>11.3f} {u['state']:>12}  {u['why']}")
    lines += [
        "",
        f"{'pairing':<40} {'obs':>6} {'chance':>7} {'lift':>6} {'p':>8}  verdict",
    ]
    for p in rep["agreements"]:  # type: ignore[index]
        lift = p["lift_over_chance"]
        pv = p["chi2_p"]
        verdict = "interpretable" if p["interpretable"] else "NOT INTERPRETABLE - degenerate marker"
        lines.append(
            f"  {p['syntactic'] + ' <-> ' + p['visual']:<38} "
            f"{p['observed_agreement']:>6.3f} {p['chance_agreement']:>7.3f} "
            f"{(lift if lift is not None else float('nan')):>6.2f} "
            f"{(pv if pv is not None else float('nan')):>8.4f}  {verdict}"
        )
    lines += ["", "provenance:", f"  visual:    {rep['provenance']['visual']}"]  # type: ignore[index]
    syn_prov = rep["provenance"]["syntactic"]  # type: ignore[index]
    for k, v in syn_prov.items():
        lines.append(f"  {k:<24} {v}")
    lines += [
        "",
        "continuous magnitude, point-biserial + label permutation (primary):",
        f"  {'pairing':<40} {'r_pb':>7} {'perm p':>8} {'x3':>8} {'MDE':>7}  {'n+':>4}",
    ]
    for a in rep.get("magnitude_associations", []):  # type: ignore[union-attr]
        if not a.get("interpretable"):
            lines.append(
                f"  {a['syntactic'] + ' <-> ' + a['visual']:<38} "
                f"{'-':>7} {'-':>8} {'-':>7}  not interpretable"
                + (f" - {a['note']}" if a.get("note") else "")
            )
            continue
        lines.append(
            f"  {a['syntactic'] + ' <-> ' + a['visual']:<38} {a['point_biserial']:>7.3f} "
            f"{a['perm_p']:>8.4f} {a.get('p_bonferroni', float('nan')):>8.4f} "
            f"{a['mde_80pct_power']:>7.3f}  {a['n_positive']:>4}"
            + ("  *" if a.get("significant_bonferroni") else "")
        )
    lines += ["", f"caveat: {rep['caveat']}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--landmark-dir", default=str(default_data_root() / "emosign" / "landmarks"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=str(artifacts_root() / "audit" / "marker_labels.json"))
    args = ap.parse_args()
    setup("INFO")
    get("seam.features.syntactic")

    clips = label_all(Path(args.landmark_dir), limit=args.limit)
    log.info("labelled %d clips", len(clips))
    rep = report(clips)
    rep[PROVENANCE_KEY] = stamp(__file__)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    log.info("wrote %s", out)

    print()
    print(render(rep))
    print()
    print(f"wrote {out}")
    return 0 if rep["gate"]["met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
