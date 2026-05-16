"""Utility helpers adapted from the official ROAD implementation."""

from __future__ import annotations

import numpy as np

use_device = "cpu"


def set_device(device_str: str) -> None:
    global use_device
    use_device = device_str


def normalize_map(saliency: np.ndarray) -> np.ndarray:
    epsilon = 1e-5
    saliency = saliency.astype(np.float32, copy=False)
    return (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + epsilon)


def rescale_channel(explanation: np.ndarray) -> np.ndarray:
    """Return a normalized 2D saliency map.

    The official ROAD code expects HWC arrays. This adapter also accepts CHW and
    already-collapsed HW maps so it can consume this repo's attribution tensors
    without reshaping at every call site.
    """

    explanation = np.asarray(explanation, dtype=np.float32)
    if explanation.ndim == 2:
        saliency = explanation
    elif explanation.ndim == 3:
        if explanation.shape[0] <= 4 and explanation.shape[-1] > 4:
            saliency = np.sum(explanation, axis=0)
        else:
            saliency = np.sum(explanation, axis=-1)
    else:
        raise ValueError(f"Expected 2D/3D explanation map, got shape {explanation.shape}")
    return normalize_map(np.abs(saliency))
