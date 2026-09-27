"""Train the non-signer FER baselines on RAF-DB.

Run as a script. Writes one checkpoint per spec into ``artifacts/fer/`` plus a
``metrics.json`` recording each model's accuracy **on its own training
distribution** - the non-signer test set. That number is what licenses the
audit's central sentence: "this is a working facial-emotion model, and it is
wrong on sign language in this specific, measured way."
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np

from seam.data import rafdb
from seam.eval.fer import (
    EMOSIGN_LABELS,
    FerModelSpec,
    build_model,
)
from seam.logging import get, setup
from seam.paths import artifacts_root
from seam.seed import set_seed

log = get(__name__)

RAFDB_DIR = Path("/mnt/DevProd/seam_data/rafdb")

#: Decoding and split loading live in ``seam.data.rafdb`` so the trainer and the
#: ONNX parity harness cannot drift apart. They did drift once, and the resulting
#: accuracy discrepancy was misread as a model result. See that module's docstring.
load_split = rafdb.load_split


def train_one(spec: FerModelSpec, xt, yt, xv, yv, *, epochs: int, batch: int) -> dict:
    import torch
    from torch import nn

    set_seed(spec.seed, deterministic_torch=False)
    model = build_model(spec)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=max(1, epochs * (len(xt) // batch + 1))
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    xtr = torch.from_numpy(xt)
    ytr = torch.from_numpy(yt)
    xv_t = torch.from_numpy(xv)
    yv_t = torch.from_numpy(yv)

    best = {"acc": -1.0, "state": None, "epoch": -1}
    rng = np.random.default_rng(spec.seed)
    started = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(xtr))
        total = 0.0
        for i in range(0, len(order), batch):
            sel = torch.from_numpy(order[i : i + batch])
            # Mild augmentation: horizontal flip and a little translation. No
            # colour jitter - the audit depends on the model being calibrated on
            # ordinary photographic faces, and a model trained on aggressively
            # augmented faces is a weaker stand-in for a plain FER model.
            xb = xtr[sel]
            flip = torch.rand(len(xb)) < 0.5
            xb[flip] = torch.flip(xb[flip], dims=[3])
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, ytr[sel])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # OneCycleLR raises once total_steps is exhausted, which happens on
            # the final partial batch. Stepping past the end is not an error.
            with contextlib.suppress(ValueError):
                sched.step()
            total += float(loss) * len(sel)

        acc = evaluate(model, xv_t, yv_t)
        if acc > best["acc"]:
            best = {
                "acc": acc,
                "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
            }
        log.info(
            "%s epoch %d loss %.4f test-acc %.4f (best %.4f @%d)",
            spec.name,
            epoch,
            total / max(len(order), 1),
            acc,
            best["acc"],
            best["epoch"],
        )

    model.load_state_dict(best["state"])
    model.eval()
    return {
        "spec": {
            "name": spec.name,
            "seed": spec.seed,
            "width": spec.width,
            "blocks": spec.blocks,
            "dropout": spec.dropout,
        },
        "params": n_params,
        "non_signer_test_acc": best["acc"],
        "best_epoch": best["epoch"],
        "n_train": len(xt),
        "n_test": len(xv),
        "seconds": time.perf_counter() - started,
        "_state": best["state"],
    }


def evaluate(model, x, y, batch: int = 256) -> float:
    import torch

    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            logits = model(x[i : i + batch])
            correct += int((logits.argmax(dim=1) == y[i : i + batch]).sum())
    return correct / max(len(x), 1)


SPECS = (
    FerModelSpec("fer_cnn_a", seed=0, width=32, blocks=3, dropout=0.3),
    FerModelSpec("fer_cnn_b", seed=1, width=48, blocks=4, dropout=0.4),
    FerModelSpec("fer_cnn_c", seed=2, width=24, blocks=5, dropout=0.2),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--specs", type=int, default=None, help="train only the first N specs (smoke runs)"
    )
    args = ap.parse_args()

    setup("INFO")
    out = Path(args.out) if args.out else artifacts_root() / "fer"
    out.mkdir(parents=True, exist_ok=True)

    xt, yt, xv, yv = load_split(args.limit)
    log.info("train %d  test %d  classes %s", len(xt), len(xv), EMOSIGN_LABELS)

    import torch

    specs = SPECS if args.specs is None else SPECS[: args.specs]
    report = {"dataset": "RAF-DB", "labels": list(EMOSIGN_LABELS), "models": []}
    for spec in specs:
        res = train_one(spec, xt, yt, xv, yv, epochs=args.epochs, batch=args.batch)
        # Checkpoint before the model is reported, so a failure in the reporting
        # path cannot cost a completed training run.
        state = res.pop("_state")
        torch.save(
            {"state_dict": state, "spec": res["spec"], "n_classes": len(EMOSIGN_LABELS)},
            out / f"{spec.name}.pt",
        )
        report["models"].append(res)
        print(
            f"{spec.name}: {res['params']:,} params, non-signer test acc "
            f"{res['non_signer_test_acc']:.4f} ({res['seconds']:.0f}s)"
        )

    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
