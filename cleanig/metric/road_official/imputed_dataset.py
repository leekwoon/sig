"""Imputed dataset wrapper adapted from the official ROAD implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .imputations import BaseImputer, ChannelMeanImputer
from .utils import rescale_channel


def _mask_hw(shape) -> tuple[int, int]:
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 3 and shape[0] <= 4 and shape[-1] > 4:
        return int(shape[1]), int(shape[2])
    if len(shape) == 3:
        return int(shape[0]), int(shape[1])
    raise ValueError(f"Unsupported mask shape: {shape}")


def _percentage_key(value: float) -> str:
    return f"{float(value):.6f}"


def estimate_explanation_cache_bytes(mask, percentages) -> dict[str, int]:
    mask_np = np.asarray(mask, dtype=np.float32)
    if mask_np.ndim != 3:
        raise ValueError(f"Expected explanation masks with shape [N, H, W], got {mask_np.shape}")

    sample_count, height, width = mask_np.shape
    pixels = int(height * width)
    order_bytes = int(sample_count * pixels * np.dtype(np.int32).itemsize)
    keep_mask_bytes = int(len(percentages) * sample_count * pixels * np.dtype(np.uint8).itemsize)
    return {
        "order_bytes": order_bytes,
        "keep_mask_bytes": keep_mask_bytes,
    }


@dataclass(frozen=True)
class ExplanationCache:
    mode: str
    sample_count: int
    height: int
    width: int
    percentages: tuple[float, ...]
    estimated_bytes: int
    salient_orders: np.ndarray | None = None
    keep_masks: dict[str, np.ndarray] | None = None

    def get_keep_mask(self, index: int, percentage: float) -> np.ndarray | None:
        if not self.keep_masks:
            return None
        return self.keep_masks[_percentage_key(percentage)][index]

    def get_order(self, index: int) -> np.ndarray | None:
        if self.salient_orders is None:
            return None
        return self.salient_orders[index]


def resolve_explanation_cache_mode(
    requested_mode: str | None,
    mask,
    percentages,
    memory_budget_bytes: int | None = None,
) -> str:
    requested = str(requested_mode or "auto").strip().lower()
    if requested in {"none", "order", "mask"}:
        return requested
    if requested != "auto":
        raise ValueError(f"Unsupported ROAD cache mode: {requested_mode}")

    estimates = estimate_explanation_cache_bytes(mask, percentages)
    if memory_budget_bytes is not None and memory_budget_bytes > 0:
        if estimates["keep_mask_bytes"] <= memory_budget_bytes:
            return "mask"
        if estimates["order_bytes"] <= memory_budget_bytes:
            return "order"
        return "none"
    return "mask"


def build_explanation_cache(
    mask,
    percentages,
    cache_mode: str | None = "auto",
    memory_budget_bytes: int | None = None,
) -> ExplanationCache:
    mask_np = np.asarray(mask, dtype=np.float32)
    if mask_np.ndim != 3:
        raise ValueError(f"Expected explanation masks with shape [N, H, W], got {mask_np.shape}")

    sample_count, height, width = mask_np.shape
    percentages = tuple(float(value) for value in percentages)
    estimates = estimate_explanation_cache_bytes(mask_np, percentages)
    resolved_mode = resolve_explanation_cache_mode(
        cache_mode,
        mask_np,
        percentages,
        memory_budget_bytes=memory_budget_bytes,
    )

    if resolved_mode == "none":
        return ExplanationCache(
            mode=resolved_mode,
            sample_count=sample_count,
            height=height,
            width=width,
            percentages=percentages,
            estimated_bytes=0,
        )

    random_v = 1e-4 * np.random.randn(height, width).astype(np.float32)
    flat_pixels = height * width
    salient_orders = np.empty((sample_count, flat_pixels), dtype=np.int32)
    for index in range(sample_count):
        saliency = rescale_channel(mask_np[index]) + random_v
        salient_orders[index] = np.argsort(-saliency.reshape(-1), kind="stable").astype(np.int32, copy=False)

    if resolved_mode == "order":
        return ExplanationCache(
            mode=resolved_mode,
            sample_count=sample_count,
            height=height,
            width=width,
            percentages=percentages,
            estimated_bytes=estimates["order_bytes"],
            salient_orders=salient_orders,
        )

    row_indices = np.arange(sample_count, dtype=np.int64)[:, None]
    keep_masks: dict[str, np.ndarray] = {}
    for percentage in percentages:
        keep_count = int(flat_pixels * percentage)
        keep_mask = np.zeros((sample_count, flat_pixels), dtype=np.uint8)
        if keep_count > 0:
            keep_mask[row_indices, salient_orders[:, :keep_count]] = 1
        keep_masks[_percentage_key(percentage)] = keep_mask.reshape(sample_count, height, width)

    return ExplanationCache(
        mode=resolved_mode,
        sample_count=sample_count,
        height=height,
        width=width,
        percentages=percentages,
        estimated_bytes=estimates["keep_mask_bytes"],
        salient_orders=salient_orders,
        keep_masks=keep_masks,
    )


class ImputedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        mask,
        th_p: float = 1.0,
        remove: bool = True,
        imputation: BaseImputer = ChannelMeanImputer(),
        transform=None,
        target_transform=None,
        prediction=None,
        use_cache: bool = False,
        explanation_cache: ExplanationCache | None = None,
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.img_mask = mask
        self.th_p = float(th_p)
        self.remove = bool(remove)
        self.prediction = prediction
        self.imputation = imputation
        self.use_cache = bool(use_cache)
        self.transform = transform
        self.target_transform = target_transform
        self.explanation_cache = explanation_cache
        self.cached_img = {}
        self.cached_target = {}
        self.cached_pred = {}

        height, width = _mask_hw(np.asarray(self.img_mask[0]).shape)
        self.height = height
        self.width = width
        if self.explanation_cache is not None and self.explanation_cache.mode != "none":
            self.random_v = None
        else:
            self.random_v = 1e-4 * np.random.randn(height, width).astype(np.float32)

    def _cached_bitmask(self, index: int, height: int, width: int) -> torch.Tensor | None:
        if self.explanation_cache is None:
            return None

        keep_mask = self.explanation_cache.get_keep_mask(index, self.th_p)
        if keep_mask is not None:
            mask_tensor = torch.from_numpy(keep_mask)
            if self.remove:
                return 1 - mask_tensor
            return mask_tensor

        salient_order = self.explanation_cache.get_order(index)
        if salient_order is None:
            return None

        bitmask = np.ones(height * width, dtype=np.uint8)
        cutoff = int(height * width * self.th_p)
        if self.remove:
            bitmask[salient_order[:cutoff]] = 0
        else:
            bitmask[salient_order[cutoff:]] = 0
        return torch.from_numpy(bitmask.reshape(height, width))

    def __getitem__(self, index: int):
        if not self.use_cache or index not in self.cached_img:
            img, target = self.base_dataset[index]
            pred = int(self.prediction[index]) if self.prediction is not None else 0

            height, width = img.size(-2), img.size(-1)
            bitmask = self._cached_bitmask(index, height, width)
            if bitmask is None:
                explanation = np.asarray(self.img_mask[index], dtype=np.float32)
                saliency = rescale_channel(explanation) + self.random_v
                salient_order = np.argsort(-saliency.reshape(-1), kind="stable").astype(np.int32, copy=False)
                bitmask_np = np.ones(height * width, dtype=np.uint8)
                cutoff = int(height * width * self.th_p)
                if self.remove:
                    bitmask_np[salient_order[:cutoff]] = 0
                else:
                    bitmask_np[salient_order[cutoff:]] = 0
                bitmask = torch.from_numpy(bitmask_np.reshape(height, width))

            img = self.imputation(img.clone(), bitmask)

            if self.use_cache:
                self.cached_img[index] = img
                self.cached_target[index] = target
                self.cached_pred[index] = pred
        else:
            img = self.cached_img[index]
            target = self.cached_target[index]
            pred = self.cached_pred[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, pred

    def __len__(self) -> int:
        return len(self.base_dataset)
