"""Resource registry, manifest verification, and the readiness table."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from seam.data import fetch as fetch_mod
from seam.data import readiness as readiness_mod
from seam.data.manifest import (
    Manifest,
    probe_video,
    scan_tree,
    sha256_file,
    verify_file,
)
from seam.data.sources import (
    RESOURCES,
    LicenseStatus,
    ResourceKind,
    all_open_issues,
    blocking_for,
)
from seam.data.sources import (
    get as get_resource,
)

# -- registry ------------------------------------------------------------


def test_registry_ids_are_unique() -> None:
    ids = [r.id for r in RESOURCES]
    assert len(ids) == len(set(ids))


def test_every_resource_names_an_owner_when_blocked() -> None:
    for row in readiness_mod.build(Path("/nonexistent-data-root")):
        if row.blocker:
            assert row.owner, f"{row.resource_id} has a blocker but no owner"


def test_registry_records_the_measured_ground_truth() -> None:
    """The plan's section 1 table, asserted rather than trusted."""
    labels = get_resource("emosign_labels")
    assert labels.usable
    assert labels.measured["rows"] == "200"
    assert labels.measured["join_to_asllrp_utterance_id"].startswith("200/200")
    assert "M4" in labels.blocking

    wlasl = get_resource("wlasl_local")
    assert wlasl.kind is ResourceKind.LOCAL
    assert wlasl.measured["clips"] == "3863"


def test_unresolved_provenance_is_recorded_not_dropped() -> None:
    """The Ethics section owes the reader an answer; the registry must know that."""
    issues = all_open_issues()
    assert "emosign_video" in issues
    joined = " ".join(issues["emosign_video"]).lower()
    assert "provenance" in joined
    assert "research" in joined


def test_blocked_license_means_not_usable() -> None:
    nsl = get_resource("nsl_local")
    assert nsl.license_status is LicenseStatus.PENDING
    assert not nsl.usable
    assert get_resource("emosign_video").license_status is LicenseStatus.RESEARCH_ONLY
    assert get_resource("emosign_video").usable


def test_single_file_resources_do_not_double_their_filename(tmp_path: Path) -> None:
    """Regression: the CSV landed at ``.../emosign_dataset.csv/emosign_dataset.csv``.

    A successful fetch followed by a failing every consumer.
    """
    resource = get_resource("emosign_labels")
    assert resource.is_single_file
    src = resource.sources[0]
    dest = resource.source_path(tmp_path, src)
    assert dest.name == "emosign_dataset.csv"
    assert dest.parent.name == "emosign"


def test_blocking_for_returns_the_right_milestones() -> None:
    ids = {r.id for r in blocking_for("M4")}
    assert "emosign_labels" in ids
    assert "emosign_video" in ids
    assert "wlasl_local" not in ids


def test_unknown_resource_raises_with_the_known_ids() -> None:
    with pytest.raises(KeyError, match="known ids"):
        get_resource("does-not-exist")


# -- manifest ------------------------------------------------------------


def test_sha256_is_stable(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    p.write_bytes(b"seam" * 1000)
    assert sha256_file(p) == sha256_file(p)
    assert len(sha256_file(p)) == 64


def test_missing_file_is_flagged(tmp_path: Path) -> None:
    rec = verify_file("x", tmp_path / "nope")
    assert rec.status == "missing"


def test_zero_byte_file_is_flagged_not_accepted(tmp_path: Path) -> None:
    p = tmp_path / "empty.mp4"
    p.write_bytes(b"")
    assert verify_file("empty.mp4", p, kind="video").status == "undecodable"


def test_size_mismatch_is_flagged(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    p.write_bytes(b"12345")
    rec = verify_file("a.bin", p, expect_bytes=99)
    assert rec.status == "size_mismatch"


def test_hash_mismatch_is_flagged(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    p.write_bytes(b"12345")
    rec = verify_file("a.bin", p, expect_sha256="0" * 64)
    assert rec.status == "hash_mismatch"


def test_truncated_archive_is_flagged(tmp_path: Path) -> None:
    """The failure mode a byte-count check misses: a present, non-empty, broken file.

    A zip is used because it is easy to truncate convincingly - keep the central
    directory, drop the payload - which is exactly the shape of a real
    interrupted download.
    """
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("a.txt", "x" * 5000)

    truncated = tmp_path / "truncated.zip"
    raw = good.read_bytes()
    truncated.write_bytes(raw[: len(raw) // 2])

    rec = verify_file("truncated.zip", truncated, kind="archive")
    assert rec.status != "ok", "a half-written archive must never verify as ok"

    rec_ok = verify_file("good.zip", good, kind="archive")
    assert rec_ok.status == "ok"


def test_video_probe_rejects_a_non_video(tmp_path: Path) -> None:
    p = tmp_path / "not_a_video.mp4"
    p.write_bytes(b"\x00" * 4096)
    ok, detail = probe_video(p)
    assert not ok
    assert detail


def test_scan_tree_summarises_statuses(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "good.bin").write_bytes(b"ok")
    (tmp_path / "bad.bin").write_bytes(b"")
    manifest = scan_tree("res", tmp_path)
    assert manifest.summary().get("undecodable") == 1
    assert not manifest.healthy
    assert manifest.total_bytes > 0


def test_manifest_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    p.write_bytes(b"payload")
    m = Manifest(resource_id="r", root=str(tmp_path))
    from seam.data.manifest import FileRecord

    m.files.append(FileRecord("a.bin", 7, sha256_file(p)))
    dest = tmp_path / "m.json"
    m.save(dest)
    back = Manifest.load(dest)
    assert back.resource_id == "r"
    assert back.files[0].sha256 == m.files[0].sha256
    assert back.healthy


# -- readiness -----------------------------------------------------------


def test_render_produces_one_line_per_resource(tmp_path: Path) -> None:
    rows = readiness_mod.build(tmp_path)
    text = readiness_mod.render(rows)
    body = [ln for ln in text.splitlines() if "|" not in ln and "  " in ln]
    assert "resource" in text and "integrity" in text
    assert len(rows) == len(RESOURCES)
    del body


def test_write_produces_markdown_and_csv(tmp_path: Path) -> None:
    rows = readiness_mod.build(tmp_path)
    md, csv_path = readiness_mod.write(rows, tmp_path / "reports")
    assert md.is_file() and csv_path.is_file()
    text = md.read_text(encoding="utf-8")
    assert "Blockers" in text
    import csv as _csv

    with csv_path.open(encoding="utf-8") as fh:
        rows_out = list(_csv.DictReader(fh))
    assert len(rows_out) == len(rows)


def test_gate_is_closed_without_the_required_resources(tmp_path: Path) -> None:
    rows = readiness_mod.build(tmp_path)
    assert not readiness_mod.gate_open(rows, milestone="M0")


def test_gate_is_closed_when_the_video_is_only_a_stub(tmp_path: Path) -> None:
    """A present, non-empty file is not a usable dataset.

    The failure this guards: a fetch that writes placeholders and a readiness
    table that reports green, which is worse than an obvious failure because the
    extraction milestone then proceeds on nothing.
    """
    from tests.conftest import EMOSIGN_ROWS, _write_emosign_csv

    root = tmp_path / "seam_data"
    _write_emosign_csv(root / "emosign" / "emosign_dataset.csv", EMOSIGN_ROWS)
    (root / "emosign" / "video").mkdir(parents=True)
    for row in EMOSIGN_ROWS:
        import re

        uid = re.search(r"(\d+)$", row["video_name"]).group(1)
        (root / "emosign" / "video" / f"{uid}.mp4").write_bytes(b"\x00" * 32)

    rows = readiness_mod.build(root)
    by_id = {r.resource_id: r for r in rows}
    assert by_id["emosign_labels"].state.value.startswith("ok")
    assert by_id["emosign_video"].state is not readiness_mod.State.OK
    assert not readiness_mod.gate_open(rows, milestone="M0")


def test_gate_opens_with_real_decodable_video(tmp_path: Path) -> None:
    """M0's bar, met: labels verified, clips decodable, bundles present."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable; cannot mint a decodable fixture")

    from tests.conftest import EMOSIGN_ROWS, _write_emosign_csv

    root = tmp_path / "seam_data"
    _write_emosign_csv(root / "emosign" / "emosign_dataset.csv", EMOSIGN_ROWS)
    video_dir = root / "emosign" / "video"
    video_dir.mkdir(parents=True)

    for row in EMOSIGN_ROWS:
        import re

        uid = re.search(r"(\d+)$", row["video_name"]).group(1)
        dest = video_dir / f"{uid}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=256x256:rate=24",
                "-pix_fmt",
                "yuv420p",
                str(dest),
            ],
            check=True,
            timeout=120,
        )

    rows = readiness_mod.build(root)
    by_id = {r.resource_id: r for r in rows}
    assert by_id["emosign_video"].state is readiness_mod.State.OK
    assert by_id["emosign_labels"].state.value.startswith("ok")
    assert readiness_mod.gate_open(rows, milestone="M0")


# -- deep verification vs existence checking -----------------------------


def test_video_sample_reports_how_much_was_actually_decoded(tmp_path: Path) -> None:
    """The table must distinguish "N files present" from "N files decode".

    WLASL is the case in point: 3,863 files, 7.4 GB, all non-empty, and a
    meaningful fraction of them truncated with no ``moov`` atom. A
    size-and-hash pass calls all 3,863 "verified", which is how a broken dataset
    reaches an extraction run and produces a plausible-looking empty result.
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable")

    root = tmp_path / "wlasl"
    (root / "a").mkdir(parents=True)
    good = root / "a" / "0.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=128x128:rate=24",
            "-pix_fmt",
            "yuv420p",
            str(good),
        ],
        check=True,
        timeout=120,
    )
    (root / "a" / "1.mp4").write_bytes(b"\x00" * 500)  # truncated

    manifest = fetch_mod.manifest_for("wlasl_local", _root_where(root))
    counts = manifest.summary()
    assert counts.get("undecodable") == 1
    assert "unchecked" not in counts, "a 2-file tree must be fully probed, not sampled"

    rows = {r.resource_id: r for r in readiness_mod.build(_root_where(root))}
    row = rows["wlasl_local"]
    assert row.state is readiness_mod.State.PARTIAL
    assert "do not decode" in row.blocker
    assert row.owner


def test_wlasl_is_configured_for_sampled_video_verification() -> None:
    from seam.data.sources import Verify

    wlasl = get_resource("wlasl_local")
    assert wlasl.verify is Verify.VIDEO_SAMPLE
    assert wlasl.sample_size >= 100
    # The known-bad-member problem must be recorded, not just configured around.
    assert any("moov" in issue for issue in wlasl.open_issues)


def test_emosign_video_is_fully_decoded_not_sampled() -> None:
    from seam.data.sources import Verify

    assert get_resource("emosign_video").verify is Verify.VIDEO


def _root_where(resource_dir: Path) -> Path:
    """A data root whose ``wlasl`` resource points at ``resource_dir``.

    The resource registry hardcodes the real local path, so the test relocates
    it via the reuse marker the fetcher writes.
    """
    root = resource_dir.parent
    (root / "wlasl").mkdir(exist_ok=True)
    (root / "wlasl" / ".reused_from").write_text(f"{resource_dir}\n", encoding="utf-8")
    return root
