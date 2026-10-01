#!/usr/bin/env python
"""How good are the heuristic linguistic labels, now that human ones exist?

M4's whole result hinges on `y_L`, and `y_L` is built from `seam.features.syntactic`,
which produces **heuristic** pseudo-labels. The ASLLRP SignStream XML downloaded on
2026-10-01 carries human annotations for the same categories on the same 200 clips. This
script measures the agreement, because a factorisation gate that fails is very hard to
interpret while the labels feeding it are of unknown quality.

Writes `artifacts/m3/label_agreement.json`.

The question it answers is deliberately narrow: *does the heuristic label predict the human
label?* Cohen's kappa, not accuracy. Accuracy is dominated by the majority class and would
make three chance-level classifiers look like two good ones and one great one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from seam.data.emosign import load as load_emosign
from seam.data.signstream import parse_directory
from seam.eval.labels import cohen_kappa
from seam.features import syntactic as SY
from seam.paths import default_data_root

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "m3" / "label_agreement.json"

#: (human markers that count as this category, heuristic label name, label for display)
#: (stable key, human markers, heuristic label, human row key, heuristic row key, display)
#:
#: The key is separate from the heuristic label because two comparisons share the
#: heuristic `interrogative`. Keying a consumer on that alone reads the rhetorical-question
#: row (kappa 0.73) in place of the wh/yes-no row (kappa 0.03) - the exact substitution
#: that would hide the finding.
PAIRS = (
    (
        "negation",
        ("negation",),
        "negation",
        "h_neg",
        "p_neg",
        "negation",
    ),
    (
        "interrogative_wh_yesno",
        ("question_wh", "question_yn"),
        "interrogative",
        "h_q",
        "p_interrogative",
        "interrogative (wh + yes/no)",
    ),
    (
        "topic",
        ("topic",),
        "topicalization",
        "h_topic",
        "p_topicalization",
        "topicalization ~ topic/focus",
    ),
    (
        "conditional",
        ("conditional",),
        "reference_establishment",
        "h_cond",
        "p_reference",
        "reference-establishment ~ conditional/when",
    ),
    (
        "interrogative_rhetorical",
        ("question_rhetorical",),
        "interrogative",
        "h_rhet",
        "p_interrogative",
        "interrogative ~ rhetorical question",
    ),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", default=str(default_data_root() / "asllrp_signstream_xml" / "raw"))
    args = ap.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.is_dir():
        print(f"no SignStream XML at {xml_path}")
        return 1

    utterances, rep = parse_directory(xml_path)
    by_id = {u.utterance_id: u for u in utterances}

    ds = load_emosign(default_data_root())
    gloss_map = SY.load_gloss_map()

    rows: list[dict[str, bool]] = []
    missing = 0
    for rec in ds.clips:
        u = by_id.get(rec.utterance_id)
        if u is None:
            missing += 1
            continue
        human = u.markers_present
        syn = SY.classify(rec.utterance_id, gloss_map).labels
        row = {
            "h_neg": "negation" in human,
            "h_q": ("question_wh" in human) or ("question_yn" in human),
            "h_topic": "topic" in human,
            "h_cond": "conditional" in human,
            "h_rhet": "question_rhetorical" in human,
            "h_brow_raise": "brow_raise" in human,
            "h_brow_furrow": "brow_furrow" in human,
            "h_head_shake": "head_shake" in human,
            "h_head_nod": "head_nod" in human,
        }
        for key, hname in (
            ("p_neg", "negation"),
            ("p_interrogative", "interrogative"),
            ("p_topicalization", "topicalization"),
            ("p_reference", "reference_establishment"),
        ):
            row[key] = bool(syn[hname].present)
        rows.append(row)

    n = len(rows)
    print(f"clips compared: {n}/{len(ds.clips)}  (no XML match: {missing})")
    if not n:
        return 1

    comparisons = []
    for key, markers, heuristic, hkey, pkey, display in PAIRS:
        h = np.array([r[hkey] for r in rows])
        p = np.array([r[pkey] for r in rows])
        k = cohen_kappa(h, p)
        rec = {
            "key": key,
            "category": display,
            "human_markers": list(markers),
            "heuristic_label": heuristic,
            "n_human_positive": int(h.sum()),
            "n_heuristic_positive": int(p.sum()),
            "accuracy": round(float(np.mean(h == p)), 4),
            "kappa": None if np.isnan(k) else round(k, 4),
            "tp": int(np.sum(h & p)),
            "fp": int(np.sum(h & ~p)),
            "fn": int(np.sum(~h & p)),
            "tn": int(np.sum(~h & ~p)),
            "verdict": (
                "useless"
                if not np.isnan(k) and k < 0.1
                else "weak"
                if not np.isnan(k) and k < 0.4
                else "usable"
            ),
        }
        comparisons.append(rec)
        kk = "n/a" if rec["kappa"] is None else f"{rec['kappa']:+.3f}"
        print(
            f"  {display:44} human+{rec['n_human_positive']:4d} "
            f"heuristic+{rec['n_heuristic_positive']:4d} "
            f"acc={rec['accuracy']:.3f} kappa={kk:>7}  {rec['verdict']}"
        )

    prevalence = {k: int(sum(1 for r in rows if r[k])) for k in rows[0] if k.startswith("h_")}
    print("\nhuman marker prevalence (clips):")
    for k, v in prevalence.items():
        print(f"   {k[2:]:16s} {v:4d}/{n}")

    out = {
        "n_clips": n,
        "n_without_xml": missing,
        "source_xml": str(xml_path),
        "corpus_events": rep.as_dict()["non_manual_events"],
        "comparisons": comparisons,
        "human_prevalence": prevalence,
        "interpretation": (
            "Cohen's kappa, not accuracy: accuracy is dominated by the majority class and "
            "would make a chance-level classifier look strong. A heuristic label at "
            "chance cannot support a claim about linguistic/affect separation, because "
            "the target it defines is not the construct the name denotes."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
