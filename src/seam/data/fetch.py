"""Fetching, reusing, and verifying dataset resources.

Three strategies, chosen per resource rather than per call:

``reuse``     hardlink or symlink an existing local path. 7 GB of WLASL and
              8.6 GB of NSL are already on this machine, and the 27 MB of
              MediaPipe task bundles are cached in a sibling repo. Re-downloading
              any of them is pure waste on a 36 GB volume.
``download``  a single file over HTTPS.
``stream``    a folder tree on Hugging Face, fetched one file at a time. Used for
              the EmoSign video, where the file list is derived from the verified
              label join rather than hardcoded, so it cannot drift.

Downloads are resumable at the file level: a resource that fails halfway is
re-verified on the next run and only the missing or damaged files are refetched.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from seam.data import emosign as emosign_mod
from seam.data.manifest import (
    FileRecord,
    Manifest,
    free_bytes,
    scan_tree,
    verify_file,
)
from seam.data.sources import Resource, ResourceKind, Verify
from seam.data.sources import get as get_resource
from seam.logging import get

log = get(__name__)

_HF_FILE_TEMPLATE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
_UA = "seam/0.1 (research; contact via repo owner)"
_TIMEOUT = 120
_CHUNK = 1 << 18


class Action(StrEnum):
    REUSED = "reused"
    DOWNLOADED = "downloaded"
    PRESENT = "present"
    MISSING = "missing"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(slots=True)
class Outcome:
    """What happened to one resource."""

    resource_id: str
    action: Action
    files: int = 0
    bytes_: int = 0
    detail: str = ""

    def __str__(self) -> str:
        size = f"{self.bytes_ / 1e6:.1f} MB" if self.bytes_ else "-"
        return (
            f"{self.resource_id:<26} {self.action.value:<11} "
            f"{self.files:>6} files  {size:>10}  {self.detail}"
        )


def _download(url: str, dest: Path, *, overwrite: bool = False) -> int:
    """Download ``url`` to ``dest`` atomically. Returns bytes written.

    Writes to a temp file in the destination directory and renames on success,
    so an interrupted run can never leave a half-written file that a later
    size check would pass.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite and dest.stat().st_size > 0:
        return dest.stat().st_size

    request = urllib.request.Request(url, headers={"User-Agent": _UA})
    tmp_fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(tmp_fd)
    tmp = Path(tmp_name)
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, _CHUNK)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest.stat().st_size


def _link_or_copy(src: Path, dest: Path) -> int:
    """Hardlink ``src`` into place, falling back to copy across filesystems.

    ``/home`` and ``/mnt`` are separate mounts here, so a hardlink frequently
    fails; a symlink is tried before a copy because a 7 GB copy to save a
    terabyte is not a trade worth making.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return 0
    try:
        os.link(src, dest)
    except OSError:
        try:
            dest.symlink_to(src)
        except OSError:
            shutil.copy2(src, dest)
    return 0


def _count_files(root: Path, patterns: Iterable[str] = ("*",)) -> tuple[int, int]:
    if root.is_symlink():
        root = root.resolve()
    if not root.exists():
        return 0, 0
    seen: set[Path] = set()
    for pattern in patterns:
        for p in root.rglob(pattern):
            if p.is_file():
                seen.add(p)
    return len(seen), sum(p.stat().st_size for p in seen)


# --------------------------------------------------------------------------
# Per-resource strategies
# --------------------------------------------------------------------------


def _fetch_single_files(resource: Resource, data_root: Path) -> Outcome:
    """Fetch every statically-known source file, skipping ones already present."""
    files = 0
    total = 0
    for source in resource.sources:
        dest = resource.source_path(data_root, source)
        if dest.exists() and dest.stat().st_size > 0:
            record = verify_file(dest.name, dest, expect_sha256=source.sha256)
            if record.status == "ok":
                files += 1
                total += record.bytes_
                continue
            log.warning(
                "%s/%s: %s (%s) - refetching",
                resource.id,
                source.path,
                record.status,
                record.detail,
            )
        total += _download(source.url, dest)
        files += 1
    action = Action.PRESENT if files and total else Action.FAILED
    return Outcome(resource.id, action, files, total, f"{len(resource.sources)} declared sources")


def _fetch_reuse(resource: Resource, data_root: Path) -> Outcome:
    """Link an existing local tree into the data root, or report it absent."""
    root = resource.local_path(data_root)
    if not resource.reuse_path:
        return Outcome(resource.id, Action.MISSING, 0, 0, "no reuse path declared")

    src = Path(resource.reuse_path)
    if not src.exists():
        return Outcome(resource.id, Action.MISSING, 0, 0, f"reuse path absent: {src}")

    if resource.kind is ResourceKind.MODELS:
        # A bundle of .task files: link the directory, do not copy 27 MB.
        root.parent.mkdir(parents=True, exist_ok=True)
        if not root.exists():
            try:
                root.symlink_to(src)
            except OSError:
                shutil.copytree(src, root)
        files, total = _count_files(root, ("*.task",))
        return Outcome(resource.id, Action.REUSED, files, total, f"linked {src}")

    # A dataset tree: link a manifest of file paths rather than the tree, so the
    # fetch stays a cheap, inspectable artifact instead of a giant symlink farm.
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".reused_from"
    if not marker.exists():
        marker.write_text(f"{src}\n", encoding="utf-8")

    patterns = ("*.mp4",) if resource.kind is ResourceKind.VIDEO else ("*",)
    files, total = _count_files(src, patterns)
    return Outcome(resource.id, Action.REUSED, files, total, f"reuse path {src}")


def _fetch_emosign_video(resource: Resource, data_root: Path) -> Outcome:
    """Stream the 200 EmoSign clips, deriving the list from the verified join.

    There is deliberately no hardcoded URL list: each URL is built from the
    utterance ID parsed out of the EmoSign label file, so the two cannot drift
    apart. If the join breaks, this fails loudly instead of quietly fetching the
    wrong 200 videos.
    """
    if not resource.usable:
        return Outcome(resource.id, Action.BLOCKED, 0, 0, f"license {resource.license_status}")

    root = resource.local_path(data_root)
    try:
        labels = emosign_mod.load(data_root)
    except emosign_mod.EmoSignError as exc:
        return Outcome(resource.id, Action.FAILED, 0, 0, f"label join failed: {exc}")

    needed = free_bytes(root)
    if needed < 200_000_000:
        log.warning(
            "only %.0f MB free under %s; the 200 clips need ~120 MB", needed / 1e6, root.parent
        )

    root.mkdir(parents=True, exist_ok=True)
    fetched = present = failed = 0
    total = 0
    failures: list[str] = []

    for clip in labels.with_video():
        dest = root / f"{clip.utterance_id}.mp4"
        if dest.exists() and dest.stat().st_size > 0:
            present += 1
            total += dest.stat().st_size
            continue
        url = _HF_FILE_TEMPLATE.format(
            repo="FangSen9000/ASLLRP_utterances_results",
            path=f"ASLLRP_utterances_results/{clip.utterance_id}/crop_original_video.mp4",
        )
        try:
            total += _download(url, dest)
            fetched += 1
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
            failed += 1
            if len(failures) < 5:
                failures.append(f"{clip.utterance_id}: {exc}")

    detail = f"{fetched} fetched, {present} already present, {failed} failed"
    if failures:
        detail += f" | {failures[0]}"
    action = Action.DOWNLOADED if fetched else Action.PRESENT
    if failed and not (fetched or present):
        action = Action.FAILED
    return Outcome(resource.id, action, fetched + present, total, detail)


_STRATEGY = {
    "emosign_video": _fetch_emosign_video,
}


def fetch(
    resource_id: str,
    data_root: Path,
    *,
    force: bool = False,
) -> Outcome:
    """Materialise one resource under ``data_root`` and report what happened."""
    resource = get_resource(resource_id)

    if not resource.usable:
        return Outcome(
            resource.id, Action.BLOCKED, 0, 0, f"license status is {resource.license_status}"
        )

    custom = _STRATEGY.get(resource.id)

    try:
        if custom is not None:
            return custom(resource, data_root)
        if resource.reuse_path and Path(resource.reuse_path).exists():
            if force:
                log.info("--force ignored for reused resource %s", resource.id)
            return _fetch_reuse(resource, data_root)
        if resource.sources:
            return _fetch_single_files(resource, data_root)
        if resource.reuse_path:
            return _fetch_reuse(resource, data_root)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        return Outcome(resource.id, Action.FAILED, 0, 0, f"{type(exc).__name__}: {exc}")

    return Outcome(resource.id, Action.MISSING, 0, 0, "no strategy applies")


def _has_matching_files(root: Path, patterns: Iterable[str]) -> bool:
    """Whether ``root`` holds at least one file matching any of ``patterns``.

    Used to decide whether the declared local path actually holds the resource, as opposed
    to merely existing. `seam_data/wlasl/` exists and holds 5,130 *landmark shards* — the
    derived output of the M1 extraction — and zero videos. "The directory exists" was
    therefore true while "the corpus is here" was false, so the readiness gate reported a
    7 GB corpus that was sitting on disk as absent.
    """
    if not root.is_dir():
        return root.is_file()
    return any(next(root.rglob(p), None) is not None for p in patterns)


def manifest_for(resource_id: str, data_root: Path) -> Manifest:
    """Verify a resource's local footprint and return its manifest.

    A single-file resource is verified as one file. Scanning it as a directory
    yields an empty manifest, which the readiness table would then report as
    "missing" immediately after a successful fetch - a false negative on the one
    resource the whole affect milestone depends on.

    The reuse fallback triggers on "no matching files here", not on "no directory here".
    A declared `dest` can end up holding a *derived* product of the same pipeline - the
    landmark shards beside the corpus they were extracted from - and keying the fallback on
    directory existence then hides the corpus entirely.
    """
    resource = get_resource(resource_id)
    patterns = _scan_patterns(resource)
    root = resource.local_path(data_root)
    # A fetcher may have recorded a different reuse location than the registry
    # declares; the marker on disk is the more recent truth.
    marker = resource.local_path(data_root) / ".reused_from"
    if marker.is_file():
        recorded = Path(marker.read_text(encoding="utf-8").strip())
        if recorded.is_dir():
            root = recorded
    elif resource.reuse_path and not _has_matching_files(root, patterns):
        reuse = Path(resource.reuse_path)
        if _has_matching_files(reuse, patterns):
            root = reuse

    if resource.is_single_file:
        manifest = Manifest(resource_id=resource_id, root=str(root))
        if root.is_file():
            manifest.files.append(verify_file(root.name, root, kind=_probe_kind(resource)))
        return manifest

    kind = _probe_kind(resource)

    if resource.verify is Verify.VIDEO_SAMPLE:
        return _sampled_scan(resource, root, kind=kind, patterns=patterns)

    return scan_tree(resource.id, root, kind=kind, patterns=patterns)


def _probe_kind(resource: Resource) -> str:
    if resource.verify in (Verify.VIDEO, Verify.VIDEO_SAMPLE, Verify.SHARD):
        return "video"
    if resource.verify is Verify.ARCHIVE:
        return "archive"
    return "file"


def _scan_patterns(resource: Resource) -> tuple[str, ...]:
    if resource.verify in (Verify.VIDEO, Verify.VIDEO_SAMPLE):
        return ("*.mp4", "*.mov", "*.mkv", "*.webm")
    if resource.kind in (ResourceKind.VIDEO, ResourceKind.LOCAL):
        return ("*",)
    return ("*",)


def _sampled_scan(
    resource: Resource, root: Path, *, kind: str, patterns: tuple[str, ...]
) -> Manifest:
    """Existence-check every file, but fully decode a deterministic sample.

    The distinction the table must not blur: "3863 files present" and "some
    unknown fraction of them are decodable" are different facts, and only the
    second one is useful. A seeded sample makes the number reproducible without
    paying an hour of ffprobe on every readiness check.
    """
    import numpy as np

    manifest = Manifest(resource_id=resource_id_of(resource), root=str(root))
    files = sorted(p for pat in patterns for p in root.rglob(pat) if p.is_file())
    if not files:
        return manifest

    rng = np.random.default_rng(0)
    n = min(resource.sample_size, len(files))
    idx = {int(i) for i in rng.choice(len(files), size=n, replace=False)}

    for i, path in enumerate(files):
        if i in idx:
            manifest.files.append(verify_file(str(path.relative_to(root)), path, kind=kind))
        else:
            size = path.stat().st_size
            manifest.files.append(
                FileRecord(str(path.relative_to(root)), size, "", "unchecked", "existence only")
            )
    return manifest


def resource_id_of(resource: Resource) -> str:
    return resource.id
