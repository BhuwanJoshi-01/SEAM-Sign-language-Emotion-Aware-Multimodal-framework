"""The ``seam`` command line.

Subcommands map to milestone gates, so ``make readiness`` and ``make facegate`` are
the two commands whose output decides whether M0 is done.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from seam import __version__, paths
from seam.logging import get, setup

log = get(__name__)

DEFAULT_DATA_ROOT = paths.default_data_root()


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report the environment. Cheap enough to run before anything else."""
    import shutil

    print(f"SEAM {__version__}")
    print(f"python      {sys.version.split()[0]}  ({sys.executable})")
    print(f"project     {paths.project_root()}")
    print(f"data root   {DEFAULT_DATA_ROOT}")
    print(f"artifacts   {paths.artifacts_root()}")

    print("\n-- gpu --")
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mb = props.total_memory / 1024**2
                print(
                    f"  [{i}] {props.name}  {total_mb:.0f} MB  "
                    f"cc {props.major}.{props.minor}  sm_{props.major}{props.minor}"
                )
                print(f"      torch {torch.__version__} cuda {torch.version.cuda}")
            free, total = torch.cuda.mem_get_info()
            print(f"      free {free / 1024**2:.0f} MB / {total / 1024**2:.0f} MB")
        else:
            print("  no CUDA device visible")
    except ImportError:
        print("  torch not installed")

    print("\n-- perception --")
    try:
        import mediapipe as mp

        print(f"  mediapipe {mp.__version__}")
        from mediapipe.tasks.python import vision

        for name in ("FaceLandmarkerOptions", "HandLandmarkerOptions", "PoseLandmarkerOptions"):
            print(f"  {name}: {'yes' if hasattr(vision, name) else 'MISSING'}")
        holistic = hasattr(mp.solutions, "holistic")
        print(f"  legacy solutions.holistic: {'present (do not use)' if holistic else 'absent'}")
    except ImportError as exc:
        print(f"  mediapipe not available: {exc}")

    print("\n-- tools --")
    for tool in ("ffprobe", "ffmpeg", "latexmk", "git", "node"):
        path = shutil.which(tool)
        print(f"  {tool:<10} {path or 'MISSING'}")

    print("\n-- disk --")
    for candidate in (Path("/mnt/Volume2"), Path("/mnt/DevProd"), Path("/")):
        if candidate.is_dir():
            try:
                usage = shutil.disk_usage(candidate)
                print(
                    f"  {candidate!s:<16} {usage.free / 1024**3:>7.1f} GB free "
                    f"of {usage.total / 1024**3:.0f} GB"
                )
            except OSError:
                pass

    return 0


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def cmd_data_fetch(args: argparse.Namespace) -> int:
    from seam.data import fetch as fetch_mod
    from seam.data.sources import RESOURCES

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    ids = [args.resource] if args.resource else [r.id for r in RESOURCES]
    outcomes = []
    for resource_id in ids:
        outcome = fetch_mod.fetch(resource_id, data_root, force=args.force)
        outcomes.append(outcome)
        print(outcome)

    failed = [o for o in outcomes if o.action.value == "failed"]
    blocked = [o for o in outcomes if o.action.value == "blocked"]
    if failed or blocked:
        print(f"\n{len(failed)} failed, {len(blocked)} blocked - see `seam data verify --all`")
    return 1 if failed else 0


def cmd_data_verify(args: argparse.Namespace) -> int:
    from seam.data import readiness as readiness_mod
    from seam.data.sources import RESOURCES

    data_root = Path(args.data_root)
    ids = {args.resource} if args.resource else {r.id for r in RESOURCES}

    rows = [r for r in readiness_mod.build(data_root) if r.resource_id in ids]
    if not rows:
        print("no matching resources", file=sys.stderr)
        return 2

    if args.table or args.json:
        if args.json:
            print(json.dumps([r.as_dict() for r in rows], indent=2))
        else:
            print(readiness_mod.render(rows))
    if args.write or args.table:
        out_dir = paths.artifacts_root() / "reports"
        md, csv_path = readiness_mod.write(rows, out_dir)
        print(f"\nwrote {md}\nwrote {csv_path}")

    open_gate = readiness_mod.gate_open(rows, milestone=args.gate)
    print(f"\n{args.gate} gate: {'OPEN' if open_gate else 'CLOSED'}")
    if not open_gate:
        for row in rows:
            if row.blocker:
                print(f"  blocked: {row.resource_id} - {row.blocker} (owner: {row.owner})")
    return 0 if open_gate else 1


def cmd_data_emosign(args: argparse.Namespace) -> int:
    from seam.data import emosign as emosign_mod

    labels = emosign_mod.load(Path(args.data_root))
    print(labels.describe())

    video_root = Path(args.data_root) / "emosign" / "video"
    if video_root.is_dir():
        clips = [c for c in labels if (video_root / f"{c.utterance_id}.mp4").is_file()]
        print(f"\nvideo on disk: {len(clips)}/{len(labels)}")
    return 0


# --------------------------------------------------------------------------
# stubs for later milestones
# --------------------------------------------------------------------------


def cmd_landmarks_extract(args: argparse.Namespace) -> int:
    from seam.perception.runner import extract_emosign

    return extract_emosign(
        data_root=Path(args.data_root),
        limit=args.limit,
        force=args.force,
    )


def cmd_landmarks_face_gate(args: argparse.Namespace) -> int:
    from seam.perception.face_gate import run_gate

    report = run_gate(data_root=Path(args.data_root), sample=args.sample)
    print(report.render())
    return 0 if report.passed else 1


def cmd_bench(args: argparse.Namespace) -> int:
    """Run the M2 latency / VRAM harness over real video.

    Returns non-zero when the harness cannot run, or when a measured stage breaches
    the VRAM ceiling, so ``make bench`` fails a build rather than printing a red
    line nobody reads.
    """
    from seam.eval import bench
    from seam.paths import artifacts_root

    if args.videos:
        videos = [Path(v) for v in args.videos]
    else:
        videos = sorted(Path(args.data_root, "emosign", "video").glob("*.mp4"))[: args.limit]
    if not videos:
        print("no videos to benchmark; pass --videos or fetch the EmoSign clips first")
        return 2

    # One mode per invocation. Measuring both from one process would race for the
    # perception singleton, and the second run would silently reuse the first run's
    # graph, so the "sequential" row would be the concurrent path under a
    # sequential label.
    mode = False if args.sequential else None
    print(
        f"benchmarking on {len(videos)} clip(s), {'sequential' if mode is False else 'concurrent'}"
    )
    try:
        report = bench.perception_bench(
            videos, iterations=args.iterations, runs=args.runs, parallel=mode
        )
    except bench.BenchError as exc:
        print(f"benchmark failed: {exc}")
        return 2

    print(report.summary())
    dest = Path(args.out) if args.out else artifacts_root() / "bench" / "perception.json"
    bench.save(report, dest)
    print(f"\nwrote {dest}")

    if report.vram is not None and not report.vram.ok:
        print("VRAM ceiling breached")
        return 1
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export the FER baselines to ONNX and check FP32/INT8 parity (M2).

    Delegates to the driver script rather than reimplementing it, so ``make
    export`` and ``seam export`` cannot diverge.
    """
    from seam.paths import project_root

    script = project_root() / "scripts" / "export_onnx.py"
    if not script.is_file():
        print(f"export driver not found at {script}")
        return 2
    argv = [sys.executable, str(script)]
    if args.inputs:
        argv.append(f"--inputs={args.inputs}")
    return int(subprocess.run(argv).returncode)


def _not_yet(name: str) -> Callable[[argparse.Namespace], int]:
    def handler(_: argparse.Namespace) -> int:
        print(f"seam {name} is not implemented yet - see plan.md for the milestone it belongs to")
        return 3

    return handler


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seam", description=__doc__)
    parser.add_argument("--version", action="version", version=f"seam {__version__}")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="environment report")
    p_doctor.set_defaults(func=cmd_doctor)

    p_data = sub.add_parser("data", help="dataset acquisition and verification")
    data_sub = p_data.add_subparsers(dest="data_command", required=True)

    p_fetch = data_sub.add_parser("fetch", help="fetch or reuse resources")
    p_fetch.add_argument("resource", nargs="?", help="resource id; default all")
    p_fetch.add_argument("--all", action="store_true", help="fetch every resource")
    p_fetch.add_argument("--force", action="store_true")
    p_fetch.set_defaults(func=cmd_data_fetch)

    p_verify = data_sub.add_parser("verify", help="verify integrity and print readiness")
    p_verify.add_argument("resource", nargs="?", help="resource id; default all")
    p_verify.add_argument("--all", action="store_true")
    p_verify.add_argument("--table", action="store_true", help="render the table")
    p_verify.add_argument("--json", action="store_true")
    p_verify.add_argument("--write", action="store_true", help="write reports to artifacts/")
    p_verify.add_argument("--gate", default="M0", help="milestone gate to evaluate")
    p_verify.set_defaults(func=cmd_data_verify)

    p_emo = data_sub.add_parser("emosign", help="label distribution and video join")
    p_emo.set_defaults(func=cmd_data_emosign)

    p_lm = sub.add_parser("landmarks", help="landmark and blendshape extraction")
    lm_sub = p_lm.add_subparsers(dest="lm_command", required=True)

    p_ex = lm_sub.add_parser("extract", help="extract for the EmoSign clips")
    p_ex.add_argument("--dataset", default="emosign")
    p_ex.add_argument("--limit", type=int, default=None)
    p_ex.add_argument("--force", action="store_true")
    p_ex.set_defaults(func=cmd_landmarks_extract)

    p_fg = lm_sub.add_parser("face-gate", help="M0 face-visibility gate")
    p_fg.add_argument("--sample", type=int, default=24)
    p_fg.set_defaults(func=cmd_landmarks_face_gate)

    p_bench = sub.add_parser("bench", help="latency / VRAM harness on the RTX 3050 (M2)")
    p_bench.add_argument("--videos", nargs="*", help="explicit video paths")
    p_bench.add_argument("--limit", type=int, default=5, help="clips to use when none given")
    p_bench.add_argument("--iterations", type=int, default=40)
    p_bench.add_argument("--runs", type=int, default=5)
    p_bench.add_argument(
        "--sequential", action="store_true", help="measure the sequential baseline"
    )
    p_bench.add_argument("--out", help="report path")
    p_bench.set_defaults(func=cmd_bench)

    p_exp = sub.add_parser("export", help="ONNX export + FP32/INT8 parity (M2)")
    p_exp.add_argument("--inputs", type=int, help="parity batches to use")
    p_exp.set_defaults(func=cmd_export)

    for name, help_text in (
        ("train", "train from configs/ (M1+)"),
        ("serve", "FastAPI + WebSocket server (M7)"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=_not_yet(name))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup("DEBUG" if args.verbose else "INFO", quiet=args.command == "doctor")

    if args.command == "doctor":
        # doctor imports torch and probes CUDA; keep its output unadorned.
        return cmd_doctor(args)

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except subprocess.CalledProcessError as exc:
        log.error("subprocess failed: %s", exc)
        return exc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
