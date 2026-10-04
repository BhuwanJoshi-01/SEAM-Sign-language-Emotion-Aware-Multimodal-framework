"""WLASL manifest: join the official annotation to what is actually on disk.

The annotation is a list of gloss records, each with instances carrying
``video_id``, ``instance_id``, ``signer_id``, ``split`` and the annotated
``frame_start`` / ``frame_end``. On disk the files are named ``<id>.mp4``,
``<id>_yt.mp4`` or ``<id>_yt.mp4.part.mp4`` inside a directory named after the
gloss. That naming is not documented anywhere, so the mapping is derived here and
asserted rather than assumed.

The frame range matters for more than bookkeeping: a large fraction of the local
tree is untrimmed - a whole YouTube video where only a second or two carries the
sign - and for those the container has no ``moov`` atom and does not decode. The
annotated range is what makes them repairable, which is the difference between
1,130 usable clips and 168 dead ones in the prior system.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from seam.logging import get

log = get(__name__)

#: The on-disk filename encodes the *instance id*, not the video id:
#: ``<gloss>/<instance_id>.mp4`` or ``<gloss>/<instance_id>_yt.mp4`` for a clip
#: downloaded from YouTube, with a trailing ``.part`` when the download was
#: interrupted. Measured over all 3,863 local files:
#:
#:     2,657  <n>.mp4                  isolated, already trimmed
#:     1,206  <n>_yt.mp4.part.mp4      YouTube excerpt, download truncated
#:
#: so ``(gloss, instance_id)`` is the join key and the ``_yt``/``.part`` suffix
#: predicts exactly the class that needs a repair pass.
# Measured shapes in the local corpus, all 3,863 files:
#   N.mp4                  2,657   pre-trimmed isolated signs
#   N_yt.mp4.part.mp4      1,206   YouTube excerpts whose download was interrupted
# The inner `.mp4` matters. A pattern expecting `_yt` adjacent to `.part` matches only the
# first shape, which left all 1,206 YouTube excerpts invisible to the indexer and made the
# 92 HTML placeholders the only candidate for their own (gloss, instance).
_FILENAME = re.compile(r"^(?P<instance>\d+)(?P<yt>_yt)?(?P<inner>\.mp4)?(?P<part>\.part)?\.mp4$")

#: Probe concurrency. ffprobe is IO-bound on a warm page cache; 8 workers keeps
#: the 3,863-file sweep in the tens of seconds without saturating a machine that
#: is already running a desktop and a browser.
PROBE_WORKERS = 8


class WlaslError(RuntimeError):
    """Raised when the annotation is missing or malformed."""


#: How a clip can fail to decode. These are different problems with different
#: remedies, and lumping them together as "corrupt" sends you looking for a
#: repair that cannot work.
#:
#: ``html_placeholder``  the downloader saved an HTML error page under an .mp4
#:                       name. Measured on this machine: 92 of the 3,863 local files
#:                       are 813 KB of YouTube HTML beginning ``<!DOCTYPE html>``.
#:                       **88 of the 92 are recoverable** — a sibling
#:                       ``<n>_yt.mp4.part.mp4`` holds the real, decodable clip — so this
#:                       is a substitution, not a re-download. Only 4 have no sibling and
#:                       are genuinely lost. `index_on_disk` performs the substitution.
#: ``truncated``        a real mp4 whose ``moov`` atom is missing because the
#:                       download stopped before the end. Repairable when the
#:                       annotation bounds the sign inside a longer video.
#: ``no_index_other``   anything else, including genuinely absent media.
FAILURE_HTML = "html_placeholder"
FAILURE_TRUNCATED = "truncated"
FAILURE_OTHER = "no_index_other"

_HTML_SIGNS = (b"<!DOCTYPE", b"<html", b"<?xml", b"<!doctype")


def classify_failure(head: bytes, stderr: str) -> str:
    """Name the failure so the remedy matches the cause."""
    if any(head.startswith(sig) for sig in _HTML_SIGNS) or b"<html" in head[:200].lower():
        return FAILURE_HTML
    if "moov atom not found" in stderr or "moov atom" in stderr:
        return FAILURE_TRUNCATED
    return FAILURE_OTHER


@dataclass(slots=True)
class WlaslClip:
    """One WLASL instance, as annotated and as found."""

    gloss: str
    video_id: str
    instance_id: int
    signer_id: int
    split: str
    frame_start: int
    frame_end: int
    bbox: tuple[int, int, int, int] | None
    fps: float
    #: Relative path under the dataset root, or None when absent from disk.
    path: str | None = None
    #: "ok" | "undecodable" | "missing" | "repaired" | "unrepairable"
    status: str = "missing"
    detail: str = ""
    native_fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    #: True when the annotation says this is an excerpt of a longer video, which
    #: is exactly the class that needs a repair pass.
    needs_trim: bool = False
    #: Filename carried a ``_yt`` marker: the clip is an excerpt of a YouTube
    #: video rather than a pre-trimmed isolated sign.
    is_youtube: bool = False
    #: Filename carried ``.part``: the download was interrupted.
    is_partial: bool = False
    #: One of FAILURE_HTML / FAILURE_TRUNCATED / FAILURE_OTHER when undecodable.
    failure: str = ""

    @property
    def usable(self) -> bool:
        return self.status in ("ok", "repaired")

    @property
    def has_frame_range(self) -> bool:
        """Whether the annotation bounds the sign inside a longer video.

        ``frame_end == -1`` means the annotation covers the whole file, so there
        is nothing to trim and nothing to repair.
        """
        return self.frame_end > 0 and self.frame_start > 0


def load_annotation(path: Path) -> list[WlaslClip]:
    """Parse ``WLASL_v0.3.json`` into a flat clip list."""
    if not path.is_file():
        raise WlaslError(f"WLASL annotation not found at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise WlaslError(f"{path}: expected a list of gloss records")

    clips: list[WlaslClip] = []
    for entry in raw:
        gloss = entry["gloss"]
        for inst in entry.get("instances", []):
            bbox = inst.get("bbox")
            clips.append(
                WlaslClip(
                    gloss=gloss,
                    video_id=str(inst["video_id"]),
                    instance_id=int(inst.get("instance_id", 0)),
                    signer_id=int(inst.get("signer_id", -1)),
                    split=inst.get("split", "train"),
                    frame_start=int(inst.get("frame_start", 1)),
                    frame_end=int(inst.get("frame_end", -1)),
                    bbox=tuple(bbox) if bbox else None,  # type: ignore[arg-type]
                    fps=float(inst.get("fps", 25.0) or 25.0),
                )
            )
    if not clips:
        raise WlaslError(f"{path} contained no instances")
    return clips


def is_placeholder(path: Path) -> bool:
    """Whether the file is an HTML error page saved under an ``.mp4`` name.

    A content test, not a name test. The downloader wrote 813 KB of YouTube HTML to 92
    files named ``0.mp4``, so any rule keyed on ``.part`` in the name misses every one of
    them - and all 92 are *not* lost: 88 have a sibling ``<n>_yt.mp4.part.mp4`` that decodes
    cleanly, which is a recoverable substitution rather than a re-download.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(256)
    except OSError:
        return False
    return classify_failure(head, "") == FAILURE_HTML


def index_on_disk(
    root: Path, substitutions: dict[str, str] | None = None
) -> dict[tuple[str, int], Path]:
    """Map ``(gloss, instance_id)`` -> file, by scanning the gloss directories.

    Keyed on the gloss directory plus the instance id in the filename, because that is what
    is actually on disk.

    Two preferences, in order, and the first one is the load-bearing one:

    1. **A real container beats an HTML placeholder.** `0.mp4` sorts before
       `0_yt.mp4.part.mp4`, so a naive scan binds each of the 92 affected (gloss, instance)
       keys to the error page and never sees the valid sibling beside it. Measured on this
       corpus: 92 placeholders, 88 recoverable this way, 4 genuinely lost.
    2. A non-``.part`` file beats a ``.part`` one, since a ``.part`` is by definition the
       interrupted download. This is a tiebreak only - the recoverable donor *is* a
       ``.part``, so applying it first would discard the only good copy.

    `substitutions`, if given, is filled with ``placeholder -> donor`` relative paths so the
    substitution is recorded rather than happening silently.
    """
    index: dict[tuple[str, int], Path] = {}
    if not root.is_dir():
        return index
    placeholders: dict[tuple[str, int], Path] = {}
    for gloss_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(gloss_dir.iterdir()):
            m = _FILENAME.match(f.name)
            if not m or not f.is_file():
                continue
            key = (gloss_dir.name, int(m.group("instance")))
            if is_placeholder(f):
                placeholders.setdefault(key, f)
                continue
            existing = index.get(key)
            if existing is None or (".part" in existing.name and ".part" not in f.name):
                index[key] = f

    for key, bad in placeholders.items():
        if key in index:
            if substitutions is not None:
                substitutions[str(bad.relative_to(root))] = str(index[key].relative_to(root))
            continue
        # No usable sibling. Keep the placeholder so the clip is reported as a failure with
        # a reason, rather than silently disappearing from the index.
        index[key] = bad
    return index


def attach(clips: list[WlaslClip], root: Path) -> None:
    """Fill in ``path`` and the initial status for every clip."""
    index = index_on_disk(root)
    for clip in clips:
        found = index.get((clip.gloss, clip.instance_id))
        if found is None:
            clip.status = "missing"
            clip.detail = "no file on disk for this (gloss, instance_id)"
            continue
        clip.path = str(found.relative_to(root))
        clip.needs_trim = clip.has_frame_range
        clip.is_youtube = ".part" in found.name or "_yt" in found.name
        clip.is_partial = ".part" in found.name
        clip.status = "unprobed"
    log.info("annotated %d clips, %d matched to disk", len(clips), len(index))


def _head(path: Path, n: int = 256) -> bytes:
    try:
        with path.open("rb") as fh:
            return fh.read(n)
    except OSError:
        return b""


def _probe_one(args: tuple[WlaslClip, Path, Path]) -> WlaslClip:
    """Probe one clip. ``path`` is the file to read; ``root`` resolves ``clip.path``."""
    clip, path, root = args
    path = root / path
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,avg_frame_rate,width,height",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        clip.status = "undecodable"
        clip.detail = "ffprobe timed out"
        return clip

    if proc.returncode != 0:
        clip.status = "undecodable"
        clip.detail = f"rc={proc.returncode}: {proc.stderr.strip()[:120]}"
        clip.failure = classify_failure(_head(path), proc.stderr)
        return clip

    try:
        streams = json.loads(proc.stdout).get("streams", [])
    except json.JSONDecodeError:
        clip.status = "undecodable"
        clip.detail = "unparseable ffprobe output"
        return clip

    if not streams:
        clip.status = "undecodable"
        clip.detail = "no video stream"
        clip.failure = classify_failure(_head(path), "no video stream")
        return clip

    s = streams[0]
    clip.width = int(s.get("width") or 0)
    clip.height = int(s.get("height") or 0)
    try:
        num, den = (s.get("avg_frame_rate") or "0/1").split("/")
        clip.native_fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        clip.native_fps = 0.0
    try:
        clip.frame_count = int(s.get("nb_frames") or 0)
    except (TypeError, ValueError):
        clip.frame_count = 0

    clip.status = "ok"
    return clip


def probe(clips: list[WlaslClip], root: Path, *, workers: int = PROBE_WORKERS) -> None:
    """Probe every clip that has a file. Undecodable clips keep their reason."""
    todo = [(c, Path(c.path), root) for c in clips if c.status == "unprobed" and c.path]
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(_probe_one, todo), start=1):
            if i % 500 == 0:
                log.info("probed %d/%d", i, len(todo))


def repair(
    clips: list[WlaslClip],
    root: Path,
    out_root: Path,
    *,
    workers: int = 4,
) -> list[WlaslClip]:
    """Re-cut undecodable clips to their annotated frame range with ffmpeg.

    Only clips whose annotation bounds the sign inside a longer video are
    attempted. A clip with no frame range that will not decode is genuinely lost,
    and is reported as such rather than guessed at.
    """
    candidates = [
        c
        for c in clips
        if c.status == "undecodable" and c.path and c.has_frame_range and c.failure != FAILURE_HTML
    ]
    skipped = sum(1 for c in clips if c.status == "undecodable" and c.failure == FAILURE_HTML)
    if skipped:
        log.info(
            "skipping %d html-placeholder files: there is no video in them to re-cut",
            skipped,
        )
        for c in clips:
            if c.status == "undecodable" and c.failure == FAILURE_HTML:
                c.status = "html_placeholder"
    if not candidates:
        log.info("nothing to repair")
        return []

    out_root.mkdir(parents=True, exist_ok=True)
    log.info("repairing %d undecodable clips with a frame range", len(candidates))

    def _one(clip: WlaslClip) -> WlaslClip:
        assert clip.path is not None  # filtered by the comprehension above
        src = root / clip.path
        dest = out_root / clip.gloss.replace("/", "_") / f"{clip.video_id}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Re-encode rather than stream-copy: the source has no index, so -c copy
        # cannot seek. 224x224 keeps the repair set small and is ample for
        # landmark extraction, which is all these clips are for.
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            str(max(clip.frame_start - 1, 0) / max(clip.fps, 1e-6)),
            "-i",
            str(src),
            "-frames:v",
            str(max(clip.frame_end - clip.frame_start + 1, 1)),
            "-vf",
            "scale=224:224:force_original_aspect_ratio=decrease,pad=224:224:(ow-iw)/2:(oh-ih)/2",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            str(dest),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
        except subprocess.TimeoutExpired:
            clip.status = "unrepairable"
            clip.detail = "ffmpeg timed out"
            return clip
        if proc.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
            clip.status = "unrepairable"
            clip.detail = f"ffmpeg rc={proc.returncode}: {proc.stderr.strip()[:120]}"
            dest.unlink(missing_ok=True)
            return clip

        # A repair that produced an undecodable file is not a repair.
        check = _probe_one((WlaslClip(**{**asdict(clip), "status": "unprobed"}), dest, out_root))
        if check.status != "ok":
            clip.status = "unrepairable"
            clip.detail = f"re-cut file does not decode: {check.detail}"
            dest.unlink(missing_ok=True)
            return clip

        clip.status = "repaired"
        clip.detail = f"re-cut to frames {clip.frame_start}-{clip.frame_end}"
        clip.path = str(dest.relative_to(out_root)) if dest.is_relative_to(out_root) else str(dest)
        clip.native_fps = check.native_fps
        clip.frame_count = check.frame_count
        clip.width, clip.height = check.width, check.height
        return clip

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_one, candidates))
    repaired = sum(1 for c in results if c.status == "repaired")
    log.info("repaired %d/%d", repaired, len(candidates))
    return results


def save(clips: list[WlaslClip], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(c) for c in clips], indent=1), encoding="utf-8")


def load(path: Path) -> list[WlaslClip]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for d in raw:
        fields = set(WlaslClip.__dataclass_fields__)
        out.append(WlaslClip(**{k: v for k, v in d.items() if k in fields}))
    return out


def summarize(clips: list[WlaslClip]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for c in clips:
        counts[c.status] = counts.get(c.status, 0) + 1
    usable = [c for c in clips if c.usable]
    return {
        "total": len(clips),
        "by_status": dict(sorted(counts.items())),
        "usable": len(usable),
        "usable_glosses": len({c.gloss for c in usable}),
        "usable_signers": len({c.signer_id for c in usable}),
        "usable_frames": sum(c.frame_count for c in usable),
        "usable_splits": _count(c.split for c in usable),
        "glosses_fully_unusable": len({c.gloss for c in clips} - {c.gloss for c in usable}),
    }


def _count(values: Iterable[object]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


def render(summary: dict[str, object]) -> str:
    lines = [f"WLASL: {summary['total']} annotated instances"]
    by_status: dict[str, int] = summary["by_status"]  # type: ignore[assignment]
    for status, n in by_status.items():
        lines.append(f"  {status:<14} {n:>5}")
    lines += [
        f"  {'---':<14} {'---':>5}",
        f"  {'usable':<14} {summary['usable']:>5}"
        f"  across {summary['usable_glosses']} glosses,"
        f" {summary['usable_signers']} signers,"
        f" {summary['usable_frames']} frames",
        f"  glosses with nothing usable: {summary['glosses_fully_unusable']}",
        f"  splits: {summary['usable_splits']}",
    ]
    return "\n".join(lines)
