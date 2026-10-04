"""The readiness table: what is actually usable, and what blocks what.

This is the M0 evidence gate. Its job is to make the gap between "the plan says we
have this dataset" and "we can actually read it" impossible to miss, and to give
every blocked row a named owner so nothing sits red without an owner.

It is a table rather than a boolean because the interesting states are the middle
ones: a resource that is present, decodable, and *not licensed for the use we
intend* is the case that gets discovered three milestones too late otherwise.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from seam.data import fetch as fetch_mod
from seam.data.manifest import Manifest
from seam.data.sources import (
    RESOURCES,
    Resource,
    Verify,
)


class State(StrEnum):
    OK = "ok"
    REUSED = "ok (reused local)"
    PARTIAL = "partial"
    DEGRADED = "degraded"
    BLOCKED = "blocked"
    MISSING = "missing"


@dataclass(slots=True)
class Row:
    """One resource's readiness."""

    resource_id: str
    role: str
    license: str
    files: int
    bytes_: int
    integrity: str
    state: State
    blocks: str
    blocker: str
    owner: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "resource": self.resource_id,
            "role": self.role,
            "license": self.license,
            "files": self.files,
            "bytes": self.bytes_,
            "integrity": self.integrity,
            "state": self.state.value,
            "blocks": self.blocks,
            "blocker": self.blocker,
            "owner": self.owner,
        }


#: Who owns what when a resource is not usable. Named, not "TBD" - an unowned
#: blocker is the thing that survives the whole project.
OWNERS: dict[str, str] = {
    "emosign_labels": "implementer",
    "emosign_video": "implementer",
    "asllrp_gloss_tokens": "implementer",
    "asllrp_utterance_map": "implementer",
    "wlasl_local": "implementer",
    "mediapipe_task_models": "implementer",
    "rafdb_mediapipe": "implementer",
    "how2sign_mediapipe_pose": "implementer",
    "asl_citizen_poses": "reviewer",
    "nsl_local": "reviewer",
}

#: Why a resource is not usable, when it is not.
BLOCKERS: dict[str, str] = {
    "nsl_local": "provenance and terms not established; M8 blocked until they are",
}


def _describe_reuse_path(resource: Resource) -> tuple[bool, str]:
    """Actually look at the reuse path. ``(present, description)``.

    This used to report ``reuse path absent: <path>`` whenever a resource had no manifest,
    without ever touching the filesystem. It was wrong: `wlasl_local` was reported absent
    while 667 gloss directories and 7.0 GB sat at exactly the path named in the message. A
    gate that states a fact it did not measure is worse than no gate, because every
    downstream decision inherits the error — and here it sent the plan to treat 7 GB of
    already-downloaded data as a fresh 7 GB fetch.
    """
    rp = resource.reuse_path
    if not rp:
        return False, "no reuse path declared"
    p = Path(rp)
    if not p.exists():
        return False, f"declared reuse path does not exist: {rp}"
    if p.is_file():
        return True, f"{p.name}, {p.stat().st_size / 1e6:.1f} MB"
    try:
        entries = list(p.iterdir())
    except OSError as e:  # pragma: no cover - permissions
        return False, f"reuse path unreadable: {e}"
    if not entries:
        return False, f"reuse path is empty: {rp}"
    return True, f"present at {rp} ({len(entries)} entries)"


#: A resource is short of its declared size by more than this fraction when it is
#: called incomplete. 2% absorbs shard-count rounding and container overhead without
#: letting a half-finished download read as complete.
_SIZE_TOLERANCE = 0.02

#: ...and the shortfall must also exceed this many bytes in absolute terms.
#:
#: `approx_bytes` is an order-of-magnitude estimate, not a contract, so a small resource with
#: a rough estimate would otherwise be reported incomplete forever: a 384-byte stand-in CSV
#: against a declared 67 KB is 99% short, and that is a fixture, not a broken download. What
#: this check exists to catch is an *interrupted bulk transfer* — how2sign reading `ok` with
#: 5 of its 31 shards on disk — and that is always large in bytes as well as in proportion.
_SIZE_FLOOR_BYTES = 50_000_000


def _state_for(resource: Resource, manifest: Manifest | None) -> tuple[State, str, str]:
    """Return ``(state, integrity_summary, blocker)`` for one resource."""
    if not resource.usable:
        return (
            State.BLOCKED,
            "n/a",
            BLOCKERS.get(resource.id, f"license status is {resource.license_status.value}"),
        )

    if manifest is None or not manifest.files:
        present, desc = _describe_reuse_path(resource)
        if not resource.reuse_path:
            return State.MISSING, desc, "not fetched yet"
        if present:
            # The bytes are there and no manifest has been built over them yet. That is a
            # different state from "absent", and a much cheaper one to resolve.
            return (
                State.PARTIAL,
                desc,
                "no manifest over the reuse path; the data is present but unverified "
                f"(owner: {OWNERS.get(resource.id, 'implementer')})",
            )
        return State.MISSING, desc, f"reuse path absent: {desc}"

    counts = manifest.summary()
    total = len(manifest.files)
    checked = total - counts.get("unchecked", 0)
    bad = sum(n for s, n in counts.items() if s not in ("ok", "unchecked"))

    if resource.verify is Verify.VIDEO_SAMPLE:
        # Say plainly how much was decoded. "3863 files" and "some unknown
        # fraction decode" are different facts and the table must not blur them.
        #
        # `checked` is the number of files carrying a verification status, which includes
        # the ones that failed, so it is the denominator - never the numerator. Printing
        # `checked/checked decoded` alongside `bad undecodable of {checked} sampled`
        # produced "300/300 decoded, 2 undecodable of 300 sampled": one sentence asserting
        # a fact and its negation. A gate that contradicts itself in its own output is
        # worse than one that is vague, because a reviewer cannot tell which half to
        # believe.
        if bad:
            good = checked - bad
            return (
                State.PARTIAL,
                (f"{good}/{checked} sampled decode cleanly, {bad} do not; {total} files total"),
                (
                    f"{bad} of {checked} sampled files do not decode; a repair pass is "
                    f"required before extraction (owner: {OWNERS.get(resource.id, 'implementer')})"
                ),
            )
        return (
            State.REUSED if resource.reuse_path else State.OK,
            (f"{checked}/{checked} sampled decode cleanly of {total} files"),
            "",
        )

    if resource.id == "emosign_video":
        if bad == 0:
            return State.OK, f"{manifest.total_bytes / 1e6:.0f} MB, all {total} decodable", ""
        return (
            State.PARTIAL,
            f"{bad}/{total} failed verification",
            (f"{bad} clips missing or undecodable; re-run `seam data fetch --all`"),
        )

    # Is what is here all of what there should be? Without this, a resource mid-download
    # reads as `ok` because every file present verifies: how2sign showed "5/5 verified, ok"
    # with 5 of its 31 shards on disk. A gate that cannot tell finished from unfinished
    # reports progress as completion.
    want = int(resource.approx_bytes or 0)
    shortfall = want - manifest.total_bytes
    if want and shortfall > want * _SIZE_TOLERANCE and shortfall > _SIZE_FLOOR_BYTES:
        got_pct = 100 * manifest.total_bytes / want
        return (
            State.PARTIAL,
            f"{manifest.total_bytes / 1e9:.2f} GB of {want / 1e9:.2f} GB expected "
            f"({got_pct:.0f}%), {total} files verified",
            f"incomplete: {100 - got_pct:.0f}% of the declared {want / 1e9:.2f} GB is absent "
            f"(owner: {OWNERS.get(resource.id, 'implementer')})",
        )

    integrity = f"{total - bad}/{total} verified"
    if bad == 0:
        state = State.REUSED if resource.reuse_path else State.OK
        return state, integrity, ""
    return State.PARTIAL, integrity, f"{bad} files failed: {sorted(counts)[1:]}"


def build(data_root: Path) -> list[Row]:
    """Build the readiness table by verifying every registered resource."""
    rows: list[Row] = []
    for resource in RESOURCES:
        manifest = None
        if resource.usable:
            try:
                manifest = fetch_mod.manifest_for(resource.id, data_root)
            except OSError as exc:  # unreadable mount, permissions
                rows.append(
                    Row(
                        resource.id,
                        resource.role.value,
                        resource.license_status.value,
                        0,
                        0,
                        f"unreadable: {exc}",
                        State.MISSING,
                        ",".join(resource.blocking),
                        str(exc),
                        OWNERS.get(resource.id, "implementer"),
                    )
                )
                continue

        state, integrity, blocker = _state_for(resource, manifest)
        rows.append(
            Row(
                resource_id=resource.id,
                role=resource.role.value,
                license=resource.license_status.value,
                files=len(manifest.files) if manifest else 0,
                bytes_=manifest.total_bytes if manifest else 0,
                integrity=integrity,
                state=state,
                blocks=",".join(resource.blocking),
                blocker=blocker,
                owner=OWNERS.get(resource.id, "implementer") if blocker else "",
            )
        )
    return rows


_COLUMNS = (
    "resource",
    "role",
    "license",
    "files",
    "bytes",
    "integrity",
    "state",
    "blocks",
    "blocker",
    "owner",
)


def render(rows: list[Row]) -> str:
    """Render the table as aligned text for the terminal."""
    widths = {c: max(len(c), *(len(str(r.as_dict()[c])) for r in rows)) for c in _COLUMNS}
    header = "  ".join(c.ljust(widths[c]) for c in _COLUMNS)
    rule = "  ".join("-" * widths[c] for c in _COLUMNS)
    body = "\n".join(
        "  ".join(str(r.as_dict()[c]).ljust(widths[c]) for c in _COLUMNS) for r in rows
    )
    ready = sum(1 for r in rows if r.state in (State.OK, State.REUSED))
    footer = f"\n{ready}/{len(rows)} resources usable"
    return f"{header}\n{rule}\n{body}{footer}"


def write(rows: list[Row], out_dir: Path) -> tuple[Path, Path]:
    """Write ``readiness.md`` and ``readiness.csv``. Returns both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)

    md = out_dir / "readiness.md"
    lines = [
        "# Data readiness",
        "",
        "Generated by `seam data verify --all --table`. Do not edit by hand: it is the",
        "executable form of `plan.md` section 1, and a hand edit is a claim with no",
        "evidence behind it.",
        "",
        "```",
        render(rows),
        "```",
        "",
    ]
    blocked = [r for r in rows if r.blocker]
    if blocked:
        lines += ["## Blockers", "", "| resource | blocker | owner | blocks |", "|---|---|---|---|"]
        lines += [f"| `{r.resource_id}` | {r.blocker} | {r.owner} | {r.blocks} |" for r in blocked]
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")

    csv_path = out_dir / "readiness.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_COLUMNS))
        writer.writeheader()
        writer.writerows(r.as_dict() for r in rows)

    return md, csv_path


def gate_open(rows: list[Row], *, milestone: str = "M0") -> bool:
    """Whether ``milestone`` can start, given the current table.

    M0's own bar is deliberately narrow: the resources that M0's gate names
    (EmoSign labels, EmoSign video, MediaPipe task models) plus nothing blocked
    without an owner. Later milestones are not gated here - they are gated by
    their own milestone review.
    """
    required = {"emosign_labels", "emosign_video", "mediapipe_task_models"}
    if milestone == "M0":
        for row in rows:
            if row.resource_id in required and row.state not in (State.OK, State.REUSED):
                return False
    unowned = [r for r in rows if r.blocker and not r.owner]
    return not unowned
