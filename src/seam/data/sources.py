"""The resource registry: every dataset SEAM depends on, as typed records.

This module is the executable form of ``plan.md`` section 1. The plan's first job
is to replace the assumptions inherited from plan.md v1 with what was actually
measured on 2026-09-26, and a table in a markdown file cannot be enforced by a
test. Putting it here means a resource cannot be silently used before it exists,
and its absence shows up as a failing ``seam data verify`` rather than as a crash
three milestones later.

Three rules are encoded here rather than left to discipline:

* **No resource is usable before its license status is ``granted``.** The
  registry's ``usable`` property is what loaders check.
* **Local data is reused, never re-downloaded.** 7 GB of WLASL and 8.6 GB of NSL
  are already on this machine; a fetcher that re-pulls them wastes an hour and
  risks filling a 36 GB volume.
* **A resource with no verified provenance is marked as such.** The EmoSign video
  arrives via an ungated re-upload of a Boston-University-controlled corpus. That
  is a real open item (plan.md section 6), and it is recorded on the record
  itself so it cannot be dropped from the ethics section by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class LicenseStatus(StrEnum):
    """Whether the terms permit research use from this machine."""

    GRANTED = "granted"
    PENDING = "pending"
    RESEARCH_ONLY = "research_only"
    BLOCKED = "blocked"


class ResourceKind(StrEnum):
    CSV = "csv"
    VIDEO = "video"
    KEYPOINTS = "keypoints"
    LANDMARKS = "landmarks"
    MODELS = "models"
    LOCAL = "local"


class Verify(StrEnum):
    """How deeply a resource's files are checked.

    The distinction matters because a size-and-hash pass cannot tell a valid
    mp4 from a truncated one. WLASL's raw tree is full of untrimmed ``.part``
    downloads whose containers have no ``moov`` atom: non-empty, plausible size,
    completely undecodable. Reporting those as "verified" is the specific failure
    mode the manifest exists to prevent, so anything video-shaped is decoded.
    """

    #: Exists, is a regular file, is non-empty.
    EXISTS = "exists"
    #: Decode the container with ffprobe.
    VIDEO = "video"
    #: Read the archive's central directory.
    ARCHIVE = "archive"
    #: Decode a deterministic sample; the rest are existence-checked only.
    VIDEO_SAMPLE = "video_sample"
    #: Check the sidecar's frame count against the array.
    SHARD = "shard"


class Role(StrEnum):
    """Which milestone a resource unblocks. Used to order the readiness table."""

    EMOTION_BENCHMARK = "emotion_benchmark"
    LINGUISTIC_LABELS = "linguistic_labels"
    RECOGNITION = "recognition"
    TRANSLATION = "translation"
    FER_PRETRAIN = "fer_pretrain"
    CONFOUND_AUDIT = "confound_audit"
    CROSS_LINGUAL = "cross_lingual"
    INFRASTRUCTURE = "infrastructure"


# Hugging Face endpoints. `datasets/` prefix for dataset repos, `models/` for
# model repos; both resolve through the same CDN.
_HF = "https://huggingface.co/datasets"
_HF_MODEL = "https://huggingface.co"


@dataclass(frozen=True, slots=True)
class Source:
    """One file inside a resource, with the provenance needed to verify it."""

    path: str
    url: str
    required: bool = True
    #: Filled in after a successful fetch. Compared on every later verify.
    sha256: str | None = None
    bytes_: int | None = None


@dataclass(frozen=True, slots=True)
class Resource:
    """A dataset or asset SEAM depends on."""

    id: str
    kind: ResourceKind
    role: Role
    license_status: LicenseStatus
    #: Where it lands under the data root.
    dest: str
    #: An existing path on this machine that can be hardlinked instead of
    #: downloaded. Checked for existence at fetch time, not assumed.
    reuse_path: str | None = None
    sources: tuple[Source, ...] = ()
    #: Approximate download size, for the disk-space precheck.
    approx_bytes: int = 0
    #: Human-readable note rendered in the readiness table.
    note: str = ""
    #: Unresolved provenance / licensing questions. Non-empty means the Ethics
    #: section owes the reader an answer.
    open_issues: tuple[str, ...] = ()
    #: Milestones that cannot start without this resource.
    blocking: tuple[str, ...] = ()
    #: Empirical findings, kept next to the resource that produced them.
    measured: dict[str, str] = field(default_factory=dict)
    #: How to verify. Defaults to EXISTS, which is only right for text/tabular
    #: assets whose integrity is fully captured by their hash.
    verify: Verify = Verify.EXISTS
    #: Sample size for VIDEO_SAMPLE. 3863 ffprobe calls is over an hour, which is
    #: too slow for a readiness check that should run in seconds.
    sample_size: int = 150

    @property
    def usable(self) -> bool:
        """Whether the terms permit research use of this resource."""
        return self.license_status in (
            LicenseStatus.GRANTED,
            LicenseStatus.RESEARCH_ONLY,
        )

    @property
    def is_single_file(self) -> bool:
        """Whether ``dest`` names one file rather than a directory."""
        return self.dest.endswith((".csv", ".txt", ".json"))

    def local_path(self, data_root: Path) -> Path:
        return data_root / self.dest

    def source_path(self, data_root: Path, source: Source) -> Path:
        """Where one source file lands.

        A single-file resource already names its file in ``dest``, so the source
        path must not be appended again - doing so produces
        ``emosign/emosign_dataset.csv/emosign_dataset.csv``, which reads as a
        successful fetch and then fails every consumer.
        """
        root = self.local_path(data_root)
        return root if self.is_single_file else root / source.path


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------

_EMO_VIDEOS = "https://huggingface.co/datasets/FangSen9000/ASLLRP_utterances_results/resolve/main"
_ASLLRP = "https://huggingface.co/datasets/FangSen9000/ASLLRP_utterances_results/resolve/main"


RESOURCES: tuple[Resource, ...] = (
    Resource(
        id="emosign_labels",
        kind=ResourceKind.CSV,
        role=Role.EMOTION_BENCHMARK,
        license_status=LicenseStatus.GRANTED,
        dest="emosign/emosign_dataset.csv",
        sources=(
            Source(
                path="emosign_dataset.csv",
                url=f"{_HF}/catfang/emosign/resolve/main/emosign_dataset.csv",
            ),
        ),
        approx_bytes=44_000,
        note="200 utterances x sentiment-7 + 10 emotion intensities + 3 free-text cue columns.",
        measured={
            "rows": "200",
            "signers": "Cory 87 / Jonathan 54 / Rachel 52 / Ben 7",
            "gated": "no",
            "join_to_asllrp_utterance_id": "200/200 via trailing numeric token of video_name",
        },
        blocking=("M1", "M3", "M4"),
    ),
    Resource(
        id="emosign_video",
        kind=ResourceKind.VIDEO,
        role=Role.EMOTION_BENCHMARK,
        license_status=LicenseStatus.RESEARCH_ONLY,
        dest="emosign/video",
        reuse_path="/mnt/Volume2/Sign_Language_Recognition/data/landmarks",
        sources=(),
        approx_bytes=120_000_000,
        note=(
            "The 200 EmoSign clips, resolved one at a time from the ASLLRP mirror as "
            "<utterance_id>/crop_original_video.mp4. Fetcher expands the per-clip list "
            "from the verified join, so there is no static URL list to drift."
        ),
        open_issues=(
            "Provenance: ungated re-upload of a Boston-University-controlled corpus "
            "(ASLLRP). Decision 2026-09-26: use for research, formalize before M9 "
            "submission. Owner: reviewer.",
            "Framing of crop_original_video.mp4 is unvalidated; M0's face-visibility "
            "gate determines whether the non-manual channel is recoverable.",
        ),
        measured={"join": "200/200", "sampled_mp4_present": "14/14", "mean_mp4_bytes": "567 KB"},
        verify=Verify.VIDEO,
        blocking=("M1", "M3", "M4"),
    ),
    Resource(
        id="asllrp_gloss_tokens",
        kind=ResourceKind.CSV,
        role=Role.RECOGNITION,
        license_status=LicenseStatus.RESEARCH_ONLY,
        dest="asllrp/asllrp_sentence_signs_2025_06_28.csv",
        sources=(
            Source(
                path="asllrp_sentence_signs_2025_06_28.csv",
                url=f"{_ASLLRP}/asllrp_sentence_signs_2025_06_28.csv",
            ),
        ),
        approx_bytes=3_730_000,
        note=(
            "17,522 frame-aligned sign tokens with gloss labels, handshape annotations, "
            "and signer-tagged source collections. Real gloss supervision and real "
            "temporal alignment."
        ),
        open_issues=("Same ASLLRP provenance question as emosign_video.",),
        measured={"tokens": "17522", "columns": "20"},
        blocking=("M5",),
    ),
    Resource(
        id="asllrp_utterance_map",
        kind=ResourceKind.CSV,
        role=Role.RECOGNITION,
        license_status=LicenseStatus.RESEARCH_ONLY,
        dest="asllrp/ASLLRP_utterances_mapping.txt",
        sources=(
            Source(
                path="ASLLRP_utterances_mapping.txt",
                url=f"{_ASLLRP}/ASLLRP_utterances_mapping.txt",
            ),
        ),
        approx_bytes=130_000,
        note="2,108 utterance IDs mapped to gloss sequences. The join target for EmoSign.",
        measured={"utterances": "2108"},
        blocking=("M5",),
    ),
    Resource(
        id="wlasl_local",
        kind=ResourceKind.LOCAL,
        role=Role.CONFOUND_AUDIT,
        license_status=LicenseStatus.GRANTED,
        dest="wlasl",
        reuse_path="/home/bhuwan/Videos/wlasl/videos",
        approx_bytes=7_000_000_000,
        note=(
            "3,863 mp4 across 668 glosses. Used for the recognition backbone and, more "
            "importantly, as the label-free substrate for the M1 confound audit."
        ),
        measured={"clips": "3863", "gloss_dirs": "668", "bytes": "7.43 GB"},
        verify=Verify.VIDEO_SAMPLE,
        sample_size=300,
        open_issues=(
            "Measured by an ffprobe sweep of all 3,863 files, not sampled: 92 are "
            "undecodable, and every one is 813 KB of YouTube HTML saved as 0.mp4, not a "
            "truncated .part container as previously recorded here. 88 of the 92 have a "
            "sibling <n>_yt.mp4.part.mp4 holding the real decodable clip, which "
            "`wlasl.index_on_disk` substitutes automatically; 4 are genuinely lost "
            "(beard/1, children/1, corn/0, decide/0). Usable clips: 3,771.",
        ),
        blocking=("M1", "M5"),
    ),
    Resource(
        id="mediapipe_task_models",
        kind=ResourceKind.MODELS,
        role=Role.INFRASTRUCTURE,
        license_status=LicenseStatus.GRANTED,
        dest="mediapipe_tasks",
        reuse_path="/mnt/Volume2/Sign_Language_Recognition/artifacts/mp_tasks",
        approx_bytes=27_000_000,
        note=(
            "FaceLandmarker / HandLandmarker / PoseLandmarker .task bundles. Already "
            "cached locally, so blendshape extraction needs no download at all. This is "
            "what makes the 52 ARKit coefficients available."
        ),
        measured={"files": "4", "bytes": "27 MB", "source": "local cache"},
        blocking=("M0", "M1", "M4"),
    ),
    Resource(
        id="rafdb_mediapipe",
        kind=ResourceKind.KEYPOINTS,
        role=Role.FER_PRETRAIN,
        license_status=LicenseStatus.GRANTED,
        dest="rafdb_mediapipe",
        # Already on disk, complete and readable, under a different directory name than the
        # one `dest` declares. Declaring it here rather than re-downloading 2.7 GB is the
        # point of a reuse path; the readiness gate reported this resource as "not fetched
        # yet" while all six shards sat in the data root.
        reuse_path="/mnt/DevProd/seam_data/rafdb",
        sources=(
            Source(
                path="data/train-00000-of-00004.parquet",
                url=f"{_HF}/Pelmeshek/raf-db-7emotions-mediapipe-768/resolve/main/data/train-00000-of-00004.parquet",
            ),
        ),
        approx_bytes=2_700_000_000,
        note=(
            "RAF-DB with MediaPipe already extracted, for non-signer FER pretraining and "
            "as one of the models the M1 audit indicts. On disk as six shards under "
            "seam_data/rafdb with local names (train_0-of-4 ... test.parquet) that differ "
            "from the HuggingFace paths in `sources`; the reuse path is the truth here."
        ),
        measured={
            "shards": "6",
            "train_rows": "14329",
            "val_rows": "3071",
            "test_rows": "3071",
            "bytes": "2.6 GB",
            "all_shards_readable": "yes, via pyarrow",
        },
        blocking=("M1", "M4"),
    ),
    Resource(
        id="how2sign_mediapipe_pose",
        kind=ResourceKind.KEYPOINTS,
        role=Role.TRANSLATION,
        license_status=LicenseStatus.GRANTED,
        dest="how2sign_pose",
        sources=(
            Source(
                path="README.md",
                url=f"{_HF}/Kavitha/how2sign_user3_mediapipe_pose/resolve/main/README.md",
            ),
        ),
        approx_bytes=14_120_432_425,
        note=(
            "How2Sign MediaPipe pose keypoints, published separately from the 80 h of "
            "video. Carries the official English captions, which is what makes it the "
            "translation substrate: ASLLRP English is not in the mirror. 31 parquet shards; "
            "fetch with `scripts/fetch_how2sign.py`, which resumes a dropped connection "
            "rather than restarting 14 GB."
        ),
        measured={
            "shards": "31",
            "bytes": "14.12 GB",
            "gated": "False",
            "remote_sizes_sum": "14120432425 (queried from the remote 2026-10-04)",
        },
        blocking=("M5",),
    ),
    Resource(
        id="asl_citizen_poses",
        kind=ResourceKind.LANDMARKS,
        role=Role.RECOGNITION,
        license_status=LicenseStatus.GRANTED,
        dest="asl_citizen_poses",
        approx_bytes=81_000_000_000,
        note=(
            "ASL Citizen as pre-extracted MediaPipe .pose files, ungated. Removes the "
            "Microsoft Download Center 403 and the 80 GB video download entirely. "
            "NO blendshapes - usable for the manual channel only. Fetched per batch "
            "(1000 files) and streamed, never materialised."
        ),
        measured={"storage_bytes": "81249515024", "files_per_batch": "1000"},
        blocking=("M5",),
    ),
    Resource(
        id="nsl_local",
        kind=ResourceKind.LOCAL,
        role=Role.CROSS_LINGUAL,
        license_status=LicenseStatus.PENDING,
        dest="nsl",
        reuse_path="/home/bhuwan/Videos/Sign_Language",
        approx_bytes=8_600_000_000,
        note=(
            "Indian Sign Language material: Olenepal, FULL_NSL_DATA, Rabin/Bidhayeka "
            "annotations. plan.md v1 reported 12 truncated .part archives in a path that "
            "no longer exists, so integrity must be re-audited before M8."
        ),
        open_issues=(
            "Provenance and terms for the NSL corpora are not established; M8 is blocked "
            "on this until they are.",
        ),
        measured={"bytes": "8.6 GB"},
        verify=Verify.VIDEO_SAMPLE,
        sample_size=120,
        blocking=("M8",),
    ),
)


_REGISTRY: dict[str, Resource] = {r.id: r for r in RESOURCES}


def get(resource_id: str) -> Resource:
    """Return a resource by id, raising on an unknown id."""
    try:
        return _REGISTRY[resource_id]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown resource {resource_id!r}; known ids: {known}") from None


def by_role(role: Role) -> tuple[Resource, ...]:
    return tuple(r for r in RESOURCES if r.role is role)


def blocking_for(milestone: str) -> tuple[Resource, ...]:
    """Resources that must be usable before ``milestone`` can start."""
    return tuple(r for r in RESOURCES if milestone in r.blocking)


def all_open_issues() -> dict[str, tuple[str, ...]]:
    """Unresolved provenance/licensing questions, keyed by resource id."""
    return {r.id: r.open_issues for r in RESOURCES if r.open_issues}
