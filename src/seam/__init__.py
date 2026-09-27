"""SEAM - Sign-language Emotion-Aware Multimodal framework.

Factorized non-manual modeling for emotion-aware ASL translation and expressive
avatar generation, engineered to run inference on a 4 GB consumer GPU.

The package is organised as a pipeline, one subpackage per stage:

    data       acquisition, integrity, manifests, splits
    perception landmarks, blendshapes, head pose, gaze
    preprocess normalization, interpolation, smoothing, windowing, augmentation
    features   manual (M), non-manual (NM), prosody (P) channels
    models     recognition backbones
    affect     the factorized non-manual encoder (core contribution)
    translate  gloss->text, pose->text, emotion conditioning
    avatar     retargeting, blendshape mapping, emotion modulation
    serve      FastAPI + WebSocket, backpressure, auth guard
    export     ONNX, quantization, VRAM guard
    eval       metrics, LOSO, benchmarks, tables
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
