"""Minimal ROAD adapter based on the official ROAD repository.

Source: https://github.com/tleemann/road_evaluation (MIT)
This local copy keeps only the no-retraining path used by exp12.
"""

from .imputations import BaseImputer, BlurImputer, ChannelMeanImputer, NoisyLinearImputer, ZeroImputer
from .imputed_dataset import (
    ExplanationCache,
    ImputedDataset,
    build_explanation_cache,
    estimate_explanation_cache_bytes,
    resolve_explanation_cache_mode,
)
from .road import run_road
from .utils import rescale_channel, set_device, use_device

__all__ = [
    "BaseImputer",
    "BlurImputer",
    "ChannelMeanImputer",
    "ExplanationCache",
    "ImputedDataset",
    "NoisyLinearImputer",
    "ZeroImputer",
    "build_explanation_cache",
    "estimate_explanation_cache_bytes",
    "rescale_channel",
    "resolve_explanation_cache_mode",
    "run_road",
    "set_device",
    "use_device",
]
