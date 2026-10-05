#!/usr/bin/env python
"""Draw the report's figures from the result artifacts.

Every figure in `README.md` is generated here from a JSON file under `artifacts/`, so a
chart cannot show a number the experiments did not produce. Run it after `make repro`;
it writes PNGs to `docs/figures/`.

Colour is assigned by job, from one validated three-colour categorical set (blue, orange,
aqua). Identity is never carried by colour alone: every multi-series figure has a legend,
and text is always ink, never the series colour.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts"
OUT = REPO / "docs" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
NEUTRAL = "#8a8983"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)


def _title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.965, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.905, subtitle, fontsize=10, color=INK_2, va="top")


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print(f"wrote docs/figures/{name}")


def frame_alignment() -> None:
    d = json.loads((ART / "m3" / "frame_alignment.json").read_text())["by_clip_fps"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    fig.subplots_adjust(top=0.80, left=0.09, right=0.97, bottom=0.14)
    for fps, colour in (("24", SERIES[0]), ("30", SERIES[1])):
        g = d[fps]
        xs = [float(k) for k in g["by_scale"]]
        ys = list(g["by_scale"].values())
        ax.plot(
            xs,
            ys,
            color=colour,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=f"{fps} fps clips ({g['n_clips']})",
        )
        k = int(np.argmax(ys))
        ax.annotate(
            f"peak {ys[k]:.2f} at {xs[k]:.2f}",
            (xs[k], ys[k]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            color=INK,
        )
    ax.axhline(0.5, color=NEUTRAL, linewidth=1)
    ax.text(1.10, 0.505, "chance", ha="right", va="bottom", fontsize=9, color=INK_2)
    ax.set_xlabel("scale applied to the annotated frame offset (1.00 = frame-for-frame)")
    ax.set_ylabel("AUC: blink annotation vs eye-blink signal")
    ax.set_ylim(0.45, 0.9)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    _title(
        fig,
        "Annotations only line up when rescaled by the clip's frame rate",
        "ASLLRP frame numbers are on a 30 fps timeline; 138 of our 200 clips are 24 fps "
        "(24/30 = 0.80)",
    )
    _save(fig, "frame_alignment.png")


def marker_validation() -> None:
    d = json.loads((ART / "m3" / "marker_validation.json").read_text())["markers"]
    markers = ["brow_raise", "brow_furrow", "head_shake", "head_nod"]
    labels = ["Brow raise", "Brow furrow", "Head shake", "Head nod"]
    folds = ["Cory", "Jonathan", "Rachel"]
    # The detector the live page runs. For the brows that is the blendshape mean; for the
    # head it is the revised detector on the corrected axes, where one exists.
    current = {
        m: d[m].get(
            "revised_page_within_clip_auc_by_fold", d[m]["heuristic_within_clip_auc_by_fold"]
        )
        for m in markers
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    fig.subplots_adjust(top=0.78, left=0.16, right=0.97, bottom=0.15)
    for j, (fold, colour) in enumerate(zip(folds, SERIES, strict=True)):
        ys = np.arange(len(markers))[::-1] + (j - 1) * 0.18
        ax.scatter(
            [current[m][fold] for m in markers],
            ys,
            s=70,
            color=colour,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=3,
            label=f"held-out signer: {fold}",
        )
        # Where the head markers stood before the axis correction: hollow, same colour.
        for m, y in zip(markers, ys, strict=True):
            if "revised_page_within_clip_auc_by_fold" in d[m]:
                ax.scatter(
                    d[m]["heuristic_within_clip_auc_by_fold"][fold],
                    y,
                    s=58,
                    facecolor=SURFACE,
                    edgecolor=colour,
                    linewidth=1.6,
                    zorder=2,
                )
    ax.scatter(
        [],
        [],
        s=58,
        facecolor=SURFACE,
        edgecolor=INK_2,
        linewidth=1.6,
        label="before: wrong head axis",
    )
    ax.axvline(0.5, color=NEUTRAL, linewidth=1)
    ax.axvline(0.8, color=INK, linewidth=1)
    ax.text(0.5, 3.62, "chance", ha="center", fontsize=9, color=INK_2)
    ax.text(0.8, 3.62, "validation gate 0.80", ha="center", fontsize=9, color=INK)
    ax.set_yticks(np.arange(len(markers))[::-1], labels)
    ax.set_ylim(-0.6, 3.55)
    ax.set_xlim(0.42, 0.95)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("within-clip AUC against human frame-level annotation")
    ax.legend(loc="lower right", fontsize=9)
    _title(
        fig,
        "Brow raise is validated; head shake is visible once the right axis is read",
        "Live-page detector vs human SignStream annotation, leave one signer out, 200 utterances",
    )
    _save(fig, "marker_validation.png")


def c1_continuous() -> None:
    d = json.loads((ART / "audit" / "confound_audit_continuous.json").read_text())
    markers = [
        m
        for m in d["verdicts"]
        if all(v is not None for v in d["verdicts"][m]["mde_by_model"].values())
    ]
    names = {
        "brow_raise": "Brow raise",
        "brow_furrow": "Brow furrow",
        "head_shake": "Head shake",
        "head_nod": "Head nod",
        "negation": "Negation",
        "topic": "Topic",
        "conditional": "Conditional",
        "question_rhetorical": "Rhetorical question",
    }
    models = sorted(d["models"])
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    fig.subplots_adjust(top=0.78, left=0.22, right=0.97, bottom=0.12)
    base = np.arange(len(markers))[::-1]
    for j, (model, colour) in enumerate(zip(models, SERIES, strict=True)):
        ys = base + (1 - j) * 0.22
        rows = [d["models"][model][m] for m in markers]
        xs = np.array([r["mean_shift"] for r in rows])
        lo = xs - np.array([r["ci_low"] for r in rows])
        hi = np.array([r["ci_high"] for r in rows]) - xs
        ax.errorbar(
            xs,
            ys,
            xerr=[lo, hi],
            fmt="o",
            color=colour,
            markersize=6.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            elinewidth=1.6,
            capsize=0,
            label=f"FER model {model[-1].upper()}",
            zorder=3,
        )
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(base, [names[m] for m in markers])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(
        "shift in negative-emotion probability, marker-bearing minus marker-free (95% CI)"
    )
    ax.text(0.004, len(markers) - 0.42, "read as MORE negative →", fontsize=9, color=INK_2)
    ax.text(
        -0.004, len(markers) - 0.42, "← read as LESS negative", fontsize=9, color=INK_2, ha="right"
    )
    ax.set_ylim(-0.6, len(markers) - 0.15)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=3,
        fontsize=9.5,
        handletextpad=0.3,
        columnspacing=1.6,
    )
    _title(
        fig,
        "No grammatical marker is consistently read as negative emotion",
        "200 continuous ASL utterances, human frame-level markers, three non-signer FER models",
    )
    _save(fig, "c1_continuous.png")


def m4_gate() -> None:
    d = json.loads((ART / "m4" / "factorizer_multilabel_ablation.json").read_text())["runs"]
    order = ["full", "no_mi", "no_orthogonality", "no_grl_on_a", "no_separation"]
    labels = ["Full model", "No MI term", "No orthogonality", "No GRL on affect", "No separation"]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    fig.subplots_adjust(top=0.76, left=0.21, right=0.97, bottom=0.16)
    base = np.arange(len(order))[::-1]
    for y, name in zip(base, order, strict=True):
        rows = d[name]["per_fold"]
        folds = sorted({r["held_out"] for r in rows})
        seeds = [rows[i : i + len(folds)] for i in range(0, len(rows), len(folds))]
        worst = [
            max(max(r["cross_l_to_a"], r["cross_a_to_l"]) for r in s if r["fold_coverage_ok"])
            for s in seeds
        ]
        ax.scatter(
            worst,
            y + np.linspace(-0.12, 0.12, len(worst)),
            s=60,
            color=SERIES[0],
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=3,
        )
        ax.text(
            max(worst) + 0.012,
            y,
            f"mean {np.mean(worst):.3f}",
            va="center",
            fontsize=9.5,
            color=INK,
        )
    ax.axvline(0.60, color=INK, linewidth=1)
    ax.axvline(0.50, color=NEUTRAL, linewidth=1)
    ax.text(0.604, len(order) - 0.42, "target ≤ 0.60", ha="left", fontsize=9, color=INK)
    ax.text(0.496, len(order) - 0.42, "perfect 0.50", ha="right", fontsize=9, color=INK_2)
    ax.set_yticks(base, labels)
    ax.set_ylim(-0.6, len(order) - 0.15)
    ax.set_xlim(0.46, 0.90)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("worst-fold cross-prediction AUC (one dot per random seed)")
    _title(
        fig,
        "The factorized encoder never separates grammar from affect",
        "Leave-one-signer-out on EmoSign; signer positive control passes at 0.98, so "
        "the instrument can see",
    )
    _save(fig, "m4_gate.png")


def main() -> int:
    frame_alignment()
    marker_validation()
    c1_continuous()
    m4_gate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
