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
    # The known-bad-member problem must be recorded, not just configured around. The
    # recorded cause was wrong for months: it said "untrimmed .part downloads with no moov
    # atom", and a full ffprobe sweep of all 3,863 files found the failure is 92 files of
    # 813 KB YouTube **HTML** saved as 0.mp4 - zero of them moov problems. So the assertion
    # is on the measured cause, and it fails if the note drifts back to the wrong one.
    issues = " ".join(wlasl.open_issues).lower()
    assert "html" in issues, (
        f"the recorded WLASL defect is not the measured one: {wlasl.open_issues}"
    )
    assert "moov" not in issues, (
        "the WLASL open issue still claims a missing moov atom; a full sweep measured 0 of "
        "3,863 files with that fault and 92 HTML placeholders"
    )


# ── the on-disk index: placeholder substitution ───────────────────────────────


def _fake_mp4(path: Path, payload: bytes = b"\x00\x00\x00\x18ftypisom") -> Path:
    path.write_bytes(payload + b"\x00" * 64)
    return path


def _html_mp4(path: Path) -> Path:
    path.write_bytes(b"<!DOCTYPE html><html><body>video unavailable</body></html>" + b" " * 800)
    return path


def test_filename_pattern_matches_the_shapes_actually_on_disk() -> None:
    """`N_yt.mp4.part.mp4` is 1,206 of the 3,863 local files.

    A pattern that misses it leaves every YouTube excerpt invisible to the indexer, which is
    how 92 HTML placeholders ended up as the only candidate for their own (gloss, instance).
    """
    from seam.data.wlasl import _FILENAME

    for name, want_instance in (
        ("0.mp4", 0),
        ("10.mp4", 10),
        ("0_yt.mp4.part.mp4", 0),
        ("137_yt.mp4.part.mp4", 137),
    ):
        m = _FILENAME.match(name)
        assert m is not None, f"{name!r} did not match the filename pattern"
        assert int(m.group("instance")) == want_instance


def test_a_valid_sibling_beats_an_html_placeholder(tmp_path: Path) -> None:
    """`0.mp4` sorts first, so a naive scan binds the key to the error page.

    Measured on the real corpus: 92 placeholders, 88 with a decodable
    `0_yt.mp4.part.mp4` beside them, 4 with nothing.
    """
    from seam.data.wlasl import index_on_disk

    gloss = tmp_path / "about"
    gloss.mkdir()
    _html_mp4(gloss / "0.mp4")
    donor = _fake_mp4(gloss / "0_yt.mp4.part.mp4")

    subs: dict[str, str] = {}
    index = index_on_disk(tmp_path, subs)
    assert index[("about", 0)] == donor, "the placeholder won over the real container"
    assert subs == {"about/0.mp4": "about/0_yt.mp4.part.mp4"}


def test_a_placeholder_with_no_sibling_still_appears_in_the_index(tmp_path: Path) -> None:
    """Reported as a failure with a reason, rather than silently vanishing.

    A clip that disappears from the index is indistinguishable from an annotation that was
    never written, which is how 4 genuinely lost clips could have gone unnoticed.
    """
    from seam.data.wlasl import index_on_disk, is_placeholder

    gloss = tmp_path / "corn"
    gloss.mkdir()
    bad = _html_mp4(gloss / "0.mp4")

    index = index_on_disk(tmp_path)
    assert ("corn", 0) in index
    assert index[("corn", 0)] == bad
    assert is_placeholder(bad)


def test_placeholder_detection_is_a_content_test_not_a_name_test(tmp_path: Path) -> None:
    """The defect has nothing to do with the filename.

    Every one of the 92 is named `0.mp4` with no `.part` anywhere, so a rule keyed on `.part`
    misses all of them.
    """
    from seam.data.wlasl import is_placeholder

    html = _html_mp4(tmp_path / "0.mp4")
    assert ".part" not in html.name
    assert is_placeholder(html)

    real = _fake_mp4(tmp_path / "1.mp4")
    assert not is_placeholder(real)

    # A truncated .part with a real container is not a placeholder.
    part = _fake_mp4(tmp_path / "2_yt.mp4.part.mp4")
    assert not is_placeholder(part)


def test_the_partial_file_is_still_demoted_among_real_containers(tmp_path: Path) -> None:
    """The `.part` tiebreak survives, it is just no longer the first rule.

    Applied first it would discard the only good copy, because the recoverable donor *is* a
    `.part`.
    """
    from seam.data.wlasl import index_on_disk

    gloss = tmp_path / "again"
    gloss.mkdir()
    partial = _fake_mp4(gloss / "3_yt.mp4.part.mp4")
    _fake_mp4(gloss / "3.mp4")
    assert index_on_disk(tmp_path)[("again", 3)] == gloss / "3.mp4"
    assert partial.is_file()


def test_a_partially_fetched_resource_is_not_reported_ok(tmp_path: Path) -> None:
    """Every file present verifying is not the same as the resource being complete.

    Measured: how2sign read "5/5 verified, ok" with 5 of its 31 shards on disk, and
    "ok (reused local)" for a corpus the gate had never looked at. A gate that cannot
    distinguish finished from in-progress reports progress as completion.

    Sizes are scaled down and written sparse, so the declared size stays above the absolute
    floor without materialising anything. Asserting this against the real resource would mean
    writing its declared 14 GB, which is its own kind of mistake.
    """
    from dataclasses import replace

    from seam.data.readiness import (
        _SIZE_FLOOR_BYTES,
        _SIZE_TOLERANCE,
        _state_for,
    )
    from seam.data.sources import Verify

    want = _SIZE_FLOOR_BYTES * 4
    res = replace(get_resource("how2sign_mediapipe_pose"), approx_bytes=want, verify=Verify.EXISTS)
    assert res.approx_bytes > 0, "declared size is required for the completeness check"

    def manifest(fraction: float) -> Manifest:
        m = Manifest(resource_id=res.id, root=str(tmp_path))
        # Sparse: st_size reports the full length, no blocks are allocated.
        for i, part in enumerate((0.6 * fraction, 0.4 * fraction)):
            p = tmp_path / f"train-{i:05d}-of-00031.parquet"
            with p.open("wb") as fh:
                fh.truncate(int(want * part))
            m.files.append(verify_file(p.name, p))
        assert abs(m.total_bytes / want - fraction) < 0.02
        return m

    done_state, _, done_blocker = _state_for(res, manifest(1.0))
    part_state, integrity, part_blocker = _state_for(res, manifest(0.13))

    # `REUSED` is as complete as `OK` — the resource declares a reuse path, so the gate
    # reports that rather than a fetch. Asserting on the exact enum member here broke the
    # moment a reuse path was added, which says nothing about completeness.
    assert done_state in (readiness_mod.State.OK, readiness_mod.State.REUSED), (
        f"a complete resource reported {done_state}: {done_blocker}"
    )
    assert not done_blocker
    assert part_state is readiness_mod.State.PARTIAL, (
        f"a 13%-fetched resource reported {part_state}; the completeness check is not firing"
    )
    assert "incomplete" in part_blocker
    assert "%" in integrity
    assert _SIZE_TOLERANCE < 1.0


def test_a_rough_estimate_on_a_small_resource_is_not_called_incomplete(tmp_path: Path) -> None:
    """The absolute floor exists so a fixture is not mistaken for a broken download.

    `approx_bytes` is an order of magnitude, not a contract: 384 bytes against a declared
    67 KB is 99% short, and flagging it would make the gate permanently red for the M0 tests
    and for any small resource whose estimate was rough.
    """
    from dataclasses import replace

    from seam.data.readiness import _SIZE_FLOOR_BYTES, _state_for
    from seam.data.sources import Verify

    res = replace(get_resource("emosign_labels"), approx_bytes=67_018, verify=Verify.EXISTS)
    p = tmp_path / "emosign_dataset.csv"
    p.write_text("utterance_id,sentiment\n1,positive\n")
    m = Manifest(resource_id=res.id, root=str(tmp_path))
    m.files.append(verify_file(p.name, p))
    state, _, blocker = _state_for(res, m)
    assert m.total_bytes < _SIZE_FLOOR_BYTES
    assert state is not readiness_mod.State.PARTIAL, blocker
    assert "incomplete" not in blocker


def test_an_unknown_size_cannot_be_called_incomplete(tmp_path: Path) -> None:
    """`approx_bytes = 0` means "not declared", not "zero bytes expected"."""
    from dataclasses import replace

    from seam.data.readiness import _state_for
    from seam.data.sources import Verify

    res = replace(get_resource("emosign_labels"), approx_bytes=0, verify=Verify.EXISTS)
    p = tmp_path / "emosign_dataset.csv"
    p.write_text("utterance_id,sentiment\n1,positive\n")
    m = Manifest(resource_id=res.id, root=str(tmp_path))
    m.files.append(verify_file(p.name, p))
    state, _, blocker = _state_for(res, m)
    assert state is not readiness_mod.State.PARTIAL
    assert "incomplete" not in blocker


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
