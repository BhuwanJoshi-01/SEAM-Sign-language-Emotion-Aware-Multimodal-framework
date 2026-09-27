"""The M0 face-visibility gate.

The EmoSign clips reach us through a mirror that stores them as
``crop_original_video.mp4``. Nobody has looked at them. If those crops are
signer-centred, or framed on the hands, then the 52 blendshape coefficients -
which are the entire non-manual channel, and therefore the entire empirical basis
for M1, M3 and M4 - are simply not recoverable from them, and every downstream
milestone has to change shape.

So this gate runs before the extraction pipeline is committed, not after. It
answers one question per sampled clip: *is the face usable?* A clip passes when
MediaPipe finds a face in a large enough share of frames, and when the resulting
blendshapes actually vary over time. A face detected in 90% of frames that never
moves is not a usable affect channel, and a mean-variance check catches it where a
detection-rate check would not.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from seam.data import emosign as emosign_mod
from seam.logging import get
from seam.perception.extract import ExtractionError, extract_clip
from seam.perception.tasks_api import BLENDSHAPE_INDEX

log = get(__name__)

#: A clip is face-usable when the face is found in at least this share of frames.
MIN_FACE_RATE = 0.50
#: ...and at least this many of the 52 coefficients show real temporal variance.
#: Below this the face is present but static, which carries no affect signal.
MIN_ACTIVE_BLENDSHAPES = 6
#: Variance floor per coefficient. Blendshape scores are in [0, 1]; a coefficient
#: that never leaves a 0.05 band is noise, not an expression.
MIN_BLENDSHAPE_STD = 0.02

#: The brow coefficients whose coordinated movement marks a grammatical marker
#: rather than an emotion. Their variance is what M1's audit depends on, so it is
#: reported separately from the whole-basis variance.
_BROW = ("browInnerUp", "browOuterUpLeft", "browOuterUpRight", "browDownLeft", "browDownRight")


@dataclass(slots=True)
class ClipVerdict:
    """Per-clip result of the gate."""

    utterance_id: str
    signer: str
    frames: int
    face_rate: float
    pose_rate: float
    hand_rate: float
    active_blendshapes: int
    max_brow_std: float
    passed: bool
    detail: str = ""

    def as_row(self) -> list[str]:
        return [
            self.utterance_id,
            self.signer,
            str(self.frames),
            f"{self.face_rate:.2f}",
            f"{self.pose_rate:.2f}",
            f"{self.hand_rate:.2f}",
            str(self.active_blendshapes),
            f"{self.max_brow_std:.3f}",
            "PASS" if self.passed else "FAIL",
            self.detail,
        ]


@dataclass(slots=True)
class GateReport:
    """Aggregate gate result."""

    verdicts: list[ClipVerdict] = field(default_factory=list)
    sample: int = 0

    @property
    def passed_count(self) -> int:
        return sum(1 for v in self.verdicts if v.passed)

    @property
    def passed(self) -> bool:
        """The gate passes when most sampled clips are usable.

        "Most" rather than "all": these are third-party lab recordings, and a
        minority of unusable clips is a sampling problem to report, not a reason
        to stop. A majority failure means the mirror's framing is wrong and the
        plan needs to change.
        """
        if not self.verdicts:
            return False
        return self.passed_count / len(self.verdicts) >= 0.80

    def render(self) -> str:
        header = (
            f"{'utterance':>10} {'signer':>9} {'frames':>6} {'face':>5} {'pose':>5} "
            f"{'hand':>5} {'active':>6} {'browSD':>7}  verdict"
        )
        lines = [header, "-" * len(header)]
        lines += ["  ".join(v.as_row()) for v in self.verdicts]

        lines.append("")
        lines.append(f"clips sampled          {len(self.verdicts)}")
        pct = 100 * self.passed_count / max(len(self.verdicts), 1)
        lines.append(
            f"face-usable            {self.passed_count}/{len(self.verdicts)} ({pct:.0f}%)"
        )
        if self.verdicts:
            rates = [v.face_rate for v in self.verdicts]
            actives = [v.active_blendshapes for v in self.verdicts]
            brows = [v.max_brow_std for v in self.verdicts]
            lines.append(
                f"face detection rate    median {statistics.median(rates):.2f}"
                f"  min {min(rates):.2f}  max {max(rates):.2f}"
            )
            lines.append(
                f"active blendshapes     median {statistics.median(actives):.0f}"
                f"  min {min(actives)}  max {max(actives)}"
            )
            lines.append(f"max brow coefficient SD median {statistics.median(brows):.3f}")
        lines.append("")
        verdict = (
            "OPEN - the non-manual channel is recoverable"
            if self.passed
            else "CLOSED - the mirror framing does not support the affect milestones"
        )
        lines.append(f"GATE: {verdict}")
        return "\n".join(lines)


def judge_clip(utterance_id: str, signer: str, arrays: dict[str, np.ndarray]) -> ClipVerdict:
    """Apply the gate to one extracted clip."""
    presence = arrays["presence"]
    part_index = {"pose": 0, "left_hand": 1, "right_hand": 2, "face": 3}
    face_rate = float(presence[:, part_index["face"]].mean())
    pose_rate = float(presence[:, part_index["pose"]].mean())
    hand_rate = float(
        (presence[:, part_index["left_hand"]] | presence[:, part_index["right_hand"]]).mean()
    )

    shapes = arrays["blendshapes"].astype(np.float64)
    # A single-frame clip has no variance to measure; report it as static rather
    # than crashing, so it fails the gate for the right reason.
    stds = shapes.std(axis=0) if shapes.shape[0] > 1 else np.zeros(shapes.shape[1])
    active = int((stds > MIN_BLENDSHAPE_STD).sum())
    brow_std = max(float(stds[BLENDSHAPE_INDEX[n]]) for n in _BROW)

    reasons: list[str] = []
    if face_rate < MIN_FACE_RATE:
        reasons.append(f"face rate {face_rate:.2f} < {MIN_FACE_RATE}")
    if active < MIN_ACTIVE_BLENDSHAPES:
        reasons.append(f"{active} varying coefficients < {MIN_ACTIVE_BLENDSHAPES}")
    if brow_std < MIN_BLENDSHAPE_STD:
        reasons.append(f"brow variance {brow_std:.3f} < {MIN_BLENDSHAPE_STD}")

    return ClipVerdict(
        utterance_id=utterance_id,
        signer=signer,
        frames=int(shapes.shape[0]),
        face_rate=face_rate,
        pose_rate=pose_rate,
        hand_rate=hand_rate,
        active_blendshapes=active,
        max_brow_std=brow_std,
        passed=not reasons,
        detail="; ".join(reasons),
    )


def run_gate(data_root: Path, *, sample: int = 24, seed: int = 0) -> GateReport:
    """Sample EmoSign clips and judge whether the face channel is usable.

    The sample is spread across signers rather than drawn uniformly, because the
    four signers have very different clip counts (87 / 54 / 52 / 7) and a uniform
    draw would under-represent Ben and over-represent Cory. A gate that only ever
    sees one signer is not a gate.
    """
    labels = emosign_mod.load(data_root)
    video_root = data_root / "emosign" / "video"

    available = [c for c in labels if (video_root / f"{c.utterance_id}.mp4").is_file()]
    if not available:
        raise ExtractionError(
            f"no EmoSign videos under {video_root}. Run `seam data fetch --all` first."
        )

    rng = np.random.default_rng(seed)
    by_signer: dict[str, list] = {}
    for clip in available:
        by_signer.setdefault(clip.signer, []).append(clip)

    per = max(1, sample // max(len(by_signer), 1))
    chosen: list = []
    for signer in sorted(by_signer):
        pool = sorted(by_signer[signer], key=lambda c: c.utterance_id)
        take = min(per, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        chosen.extend(pool[int(i)] for i in sorted(idx))

    if len(chosen) < sample:  # top up from whatever is left
        remaining = [c for c in available if c not in chosen]
        extra = min(sample - len(chosen), len(remaining))
        if extra:
            idx = rng.choice(len(remaining), size=extra, replace=False)
            chosen.extend(remaining[int(i)] for i in sorted(idx))

    report = GateReport(sample=len(chosen))
    for clip in sorted(chosen, key=lambda c: c.utterance_id):
        path = video_root / f"{clip.utterance_id}.mp4"
        try:
            arrays, _meta = extract_clip(path, utterance_id=clip.utterance_id)
        except ExtractionError as exc:
            report.verdicts.append(
                ClipVerdict(
                    clip.utterance_id, clip.signer, 0, 0.0, 0.0, 0.0, 0, 0.0, False, str(exc)
                )
            )
            continue
        report.verdicts.append(judge_clip(clip.utterance_id, clip.signer, arrays))

    log.info("face gate: %d/%d clips usable", report.passed_count, len(report.verdicts))
    return report
