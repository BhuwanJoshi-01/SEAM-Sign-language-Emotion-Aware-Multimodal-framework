#!/usr/bin/env python
"""Is the FER read-out measuring the face, or the scene?

M1's audit came back with per-window shifts of ~0.004 on a negative-mass baseline
of 0.78, and the per-frame probability spread *within* a WLASL clip was 0.0002
while the spread *across* clips was 0.283. One output per clip is what you get
from a model reading background, resolution and lighting rather than expression,
and such a model cannot show a marker-dependent shift in a within-clip contrast
no matter how large the marker is.

This script decides between the two explanations, and its answer determines
whether M1's design is valid or needs replacing:

  within-video output spread  vs  between-video output spread, on RAF-DB

If within-video spread is near zero while the model still scores well above
chance, the model is keying on something that is constant within a video, and the
within-clip contrast is the wrong instrument for it.
"""

from __future__ import annotations

import collections
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_model(ckpt: Path):
    import torch

    from seam.eval import fer

    blob = torch.load(ckpt, weights_only=False)
    spec = fer.FerModelSpec(**blob["spec"])
    model = fer.build_model(spec, n_classes=blob["n_classes"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def main() -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from PIL import Image

    from seam.eval import fer

    n = 600
    src = Path("/mnt/DevProd/seam_data/rafdb/train_1-of-4.parquet")
    pf = pq.ParquetFile(src)
    batch = next(
        pf.iter_batches(batch_size=n, columns=["image", "bbox_xyxy_768", "label_name", "sample_id"])
    )
    tbl = pa.Table.from_batches([batch])
    imgs = tbl.column("image").to_pylist()
    boxes = tbl.column("bbox_xyxy_768").to_pylist()
    labels = [str(x) for x in tbl.column("label_name").to_pylist()]
    sids = [str(x) for x in tbl.column("sample_id").to_pylist()]
    del tbl, batch, pf

    crops, keep_lab, keep_sid = [], [], []
    for im, bx, lab, sid in zip(imgs, boxes, labels, sids, strict=True):
        if not bx or len(bx) != 4 or not im:
            continue
        try:
            img = Image.open(io.BytesIO(im["bytes"])).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        x0, y0, x1, y1 = bx
        px0, py0 = max(int(x0), 0), max(int(y0), 0)
        px1, py1 = min(int(x1), w), min(int(y1), h)
        if px1 - px0 < 8 or py1 - py0 < 8:
            continue
        crops.append(np.asarray(img.crop((px0, py0, px1, py1)), dtype=np.uint8))
        keep_lab.append(lab)
        keep_sid.append(sid)
    del imgs, boxes
    print(f"loaded {len(crops)} crops from {src.name}")

    ckpt = Path("artifacts/fer/fer_cnn_c.pt")
    model = load_model(ckpt)
    x = fer.preprocess_faces(crops)
    with torch.no_grad():
        P = torch.softmax(model(torch.from_numpy(x)), dim=1).numpy()
    del x, crops
    print(f"probability matrix: {P.shape}")

    groups = collections.defaultdict(list)
    for i, s in enumerate(keep_sid):
        groups[s].append(i)
    multi = {k: v for k, v in groups.items() if len(v) >= 3}
    covered = sum(len(v) for v in multi.values())
    print(f"groups with >=3 frames: {len(multi)} covering {covered} images")

    within, between = [], []
    for idxs in multi.values():
        sub = P[idxs]
        within.append(sub.max(0) - sub.min(0))
        rest = np.setdiff1d(np.arange(len(P)), idxs)
        if len(rest):
            between.append(P[rest].max(0) - P[idxs].mean(0))
    w = np.array(within)
    b = np.array(between)
    print(f"\nmean within-group spread per class : {np.round(w.mean(0), 4)}")
    print(f"mean group-mean-to-rest spread     : {np.round(b.mean(0), 4)}")
    ratio = w.mean() / max(b.mean(), 1e-9)
    print(f"ratio within / between             : {ratio:.2f}")
    print("  (< 0.2 means the model is nearly constant inside a video)")

    pred = P.argmax(1)
    acc = float(
        np.mean([fer.EMOSIGN_LABELS[pp] == lab for pp, lab in zip(pred, keep_lab, strict=True)])
    )
    print(f"\naccuracy on this sample: {acc:.4f}  (chance {1 / len(fer.EMOSIGN_LABELS):.4f})")

    pure = sum(1 for idxs in multi.values() if len({keep_lab[i] for i in idxs}) == 1)
    share = 100 * pure / len(multi) if multi else float("nan")
    print(f"groups whose frames all share one label: {pure}/{len(multi)} ({share:.0f}%)")
    print(
        "\nIf a high share of groups are label-constant, subject/clip identity is "
        "\nsufficient to predict the label, and accuracy alone cannot show the "
        "\nmodel reads expression."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
