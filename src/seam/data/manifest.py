"""SHA256 manifests, integrity verification, and truncation detection.

The v1 lesson this module encodes: a truncated download is not a missing file, it
is a file that is present, opens, and silently trains a model on half a video. A
byte-count check misses it; a hash check catches it only if the expected hash is
known. So verification is three-layered, and a resource passes only if all three
layers agree:

1. the file exists and is non-empty;
2. its size matches the manifest;
3. its content decodes as what it claims to be (an mp4 is probed, not assumed).

Layer 3 is the one that matters most, and it is the one the v1 plan omitted.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

_CHUNK = 1 << 20


class IntegrityError(RuntimeError):
    """Raised when a file fails verification and the caller demanded strictness."""


@dataclass(slots=True)
class FileRecord:
    """One verified file."""

    relpath: str
    bytes_: int
    sha256: str
    #: "ok", "missing", "size_mismatch", "hash_mismatch", "undecodable"
    status: str = "ok"
    detail: str = ""


@dataclass(slots=True)
class Manifest:
    """A set of file records under a root, persisted as JSON."""

    resource_id: str
    root: str
    files: list[FileRecord] = field(default_factory=list)

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "resource_id": self.resource_id,
            "root": self.root,
            "files": [asdict(f) for f in self.files],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Manifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            resource_id=payload["resource_id"],
            root=payload["root"],
            files=[FileRecord(**f) for f in payload.get("files", [])],
        )

    # -- queries -----------------------------------------------------------
    @property
    def total_bytes(self) -> int:
        return sum(f.bytes_ for f in self.files)

    def by_status(self, status: str) -> list[FileRecord]:
        return [f for f in self.files if f.status == status]

    @property
    def healthy(self) -> bool:
        return bool(self.files) and all(f.status == "ok" for f in self.files)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.files:
            counts[f.status] = counts.get(f.status, 0) + 1
        return counts


def sha256_file(path: Path) -> str:
    """Stream a file through SHA256. Never loads the whole file into RAM."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def probe_video(path: Path) -> tuple[bool, str]:
    """Return ``(decodable, detail)`` for a video file.

    Uses ffprobe when available because it reads the container, not the codec -
    an mp4 whose ``moov`` atom was never written is a valid-looking file to
    ``os.path.getsize`` and an undecodable file to every downstream stage.
    """
    if shutil.which("ffprobe") is None:
        # Without ffprobe, fall back to a container-magic check rather than
        # declaring everything undecodable. A truncated mp4 usually still has
        # the ftyp box, so this cannot detect truncation - it only avoids
        # producing a false alarm on a machine without ffmpeg.
        with path.open("rb") as fh:
            head = fh.read(12)
        if head[4:8] == b"ftyp":
            return True, "ftyp box present; ffprobe unavailable, truncation NOT checked"
        return False, "no ftyp box and ffprobe unavailable"

    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_frames,avg_frame_rate",
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
        return False, "ffprobe timed out"

    if proc.returncode != 0:
        return False, f"ffprobe rc={proc.returncode}: {proc.stderr.strip()[:200]}"

    try:
        streams = json.loads(proc.stdout).get("streams", [])
    except json.JSONDecodeError:
        return False, "ffprobe emitted unparseable output"
    if not streams:
        return False, "no video stream in container"

    s = streams[0]
    detail = (
        f"{s.get('codec_name')} {s.get('width')}x{s.get('height')} "
        f"frames={s.get('nb_frames')} fps={s.get('avg_frame_rate')}"
    )
    return True, detail


def probe_archive(path: Path) -> tuple[bool, str]:
    """Return ``(readable, detail)`` for a zip/tar archive.

    Video probing catches a truncated mp4 because the container's ``moov`` atom
    is missing. An archive needs its own check: a half-written zip is still a
    valid-looking file to a size test, and ``zipfile`` only notices when it
    actually reads the central directory, which is why this opens the archive
    rather than stat-ing it.
    """
    if zipfile.is_zipfile(path):
        try:
            with zipfile.ZipFile(path) as zf:
                bad = zf.testzip()
            if bad is not None:
                return False, f"corrupt member: {bad}"
            return True, "zip central directory readable, all members pass CRC"
        except (zipfile.BadZipFile, OSError) as exc:
            return False, f"zip unreadable: {exc}"
    try:
        with tarfile.open(path) as tf:
            names = tf.getnames()
        return True, f"tar readable, {len(names)} members"
    except (tarfile.TarError, OSError) as exc:
        return False, f"not a readable zip or tar: {exc}"


def verify_file(
    relpath: str,
    path: Path,
    *,
    expect_bytes: int | None = None,
    expect_sha256: str | None = None,
    kind: str = "file",
) -> FileRecord:
    """Verify one file across all three layers and return its record."""
    if not path.exists():
        return FileRecord(relpath, 0, "", "missing", str(path))
    if not path.is_file():
        return FileRecord(relpath, 0, "", "missing", f"not a regular file: {path}")

    size = path.stat().st_size
    if size == 0:
        return FileRecord(relpath, size, "", "undecodable", "zero bytes")

    if expect_bytes is not None and size != expect_bytes:
        return FileRecord(
            relpath, size, "", "size_mismatch", f"expected {expect_bytes}, got {size}"
        )

    digest = sha256_file(path)

    if expect_sha256 is not None and digest != expect_sha256:
        return FileRecord(relpath, size, digest, "hash_mismatch", "content differs from manifest")

    if kind == "video":
        ok, detail = probe_video(path)
        if not ok:
            return FileRecord(relpath, size, digest, "undecodable", detail)
        return FileRecord(relpath, size, digest, "ok", detail)

    if kind == "archive":
        ok, detail = probe_archive(path)
        if not ok:
            return FileRecord(relpath, size, digest, "undecodable", detail)
        return FileRecord(relpath, size, digest, "ok", detail)

    return FileRecord(relpath, size, digest, "ok", "")


def scan_tree(
    resource_id: str,
    root: Path,
    *,
    kind: str = "file",
    patterns: tuple[str, ...] = ("*",),
) -> Manifest:
    """Verify every matching file under ``root`` and return a manifest."""
    seen: set[Path] = set()
    for pattern in patterns:
        for p in sorted(root.rglob(pattern)):
            if p.is_file():
                seen.add(p)

    manifest = Manifest(resource_id=resource_id, root=str(root))
    for p in sorted(seen):
        manifest.files.append(verify_file(str(p.relative_to(root)), p, kind=kind))
    return manifest


def free_bytes(path: Path) -> int:
    """Free space on the filesystem holding ``path``, walking up to a mount."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError:
        return 0
    return usage.free
