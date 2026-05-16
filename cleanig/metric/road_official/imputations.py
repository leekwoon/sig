"""Minimal imputation operators adapted from the official ROAD implementation."""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


class BaseImputer:
    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def batched_call(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def _gaussian_blur_tensor(img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    import torchvision.transforms.functional as TF

    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(kernel_size, 3)
    return TF.gaussian_blur(img, [kernel_size, kernel_size], float(sigma))


class ChannelMeanImputer(BaseImputer):
    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = img.clone()
        for channel_idx in range(len(out)):
            mean_val = out[channel_idx].mean()
            channel = out[channel_idx]
            channel[mask == 0] = mean_val
        return out

    def batched_call(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        channel_mean = img.view(img.shape[0], img.shape[1], -1).mean(dim=2)
        channel_mean = channel_mean[:, :, None, None].expand_as(img)
        return (channel_mean * (1.0 - mask[:, None])) + img * mask[:, None]


class ZeroImputer(BaseImputer):
    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return img * mask.unsqueeze(0)

    def batched_call(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return img * mask.unsqueeze(1)


class BlurImputer(BaseImputer):
    def __init__(self, kernel_size: int = 11, sigma: float = 5.0):
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        missing = mask <= 0.5
        if not torch.any(missing):
            return img.clone()
        blurred = _gaussian_blur_tensor(img, self.kernel_size, self.sigma)
        out = img.clone()
        out[:, missing] = blurred[:, missing]
        return out

    def batched_call(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        missing = (mask <= 0.5).unsqueeze(1)
        if not torch.any(missing):
            return img.clone()
        blurred = _gaussian_blur_tensor(img, self.kernel_size, self.sigma)
        return torch.where(missing, blurred, img)


NEIGHBOR_WEIGHTS = [
    ((1, 1), 1 / 12),
    ((0, 1), 1 / 6),
    ((-1, 1), 1 / 12),
    ((1, -1), 1 / 12),
    ((0, -1), 1 / 6),
    ((-1, -1), 1 / 12),
    ((1, 0), 1 / 6),
    ((-1, 0), 1 / 6),
]


class NoisyLinearImputer(BaseImputer):
    def __init__(self, noise: float = 0.01, weighting=NEIGHBOR_WEIGHTS):
        self.noise = float(noise)
        self.weighting = weighting

    @staticmethod
    def add_offset_to_indices(indices: np.ndarray, offset, mask_shape) -> tuple[np.ndarray, np.ndarray]:
        coord_x = indices % mask_shape[1]
        coord_y = indices // mask_shape[1]
        coord_y += offset[0]
        coord_x += offset[1]
        invalid = (coord_y < 0) | (coord_x < 0) | (coord_y >= mask_shape[0]) | (coord_x >= mask_shape[1])
        return ~invalid, indices + offset[0] * mask_shape[1] + offset[1]

    @staticmethod
    def setup_sparse_system(mask: np.ndarray, img: np.ndarray, weighting) -> tuple[lil_matrix, np.ndarray]:
        flat_mask = mask.flatten()
        flat_img = img.reshape((img.shape[0], -1))
        indices = np.argwhere(flat_mask == 0).flatten()
        coord_to_var_idx = np.zeros(len(flat_mask), dtype=int)
        coord_to_var_idx[indices] = np.arange(len(indices))

        num_equations = len(indices)
        system = lil_matrix((num_equations, num_equations))
        rhs = np.zeros((num_equations, img.shape[0]), dtype=np.float32)
        sum_neighbors = np.ones(num_equations, dtype=np.float32)

        for offset, weight in weighting:
            valid, new_coords = NoisyLinearImputer.add_offset_to_indices(indices, offset, mask.shape)
            valid_coords = new_coords[valid]
            valid_ids = np.argwhere(valid == 1).flatten()

            known_coords = valid_coords[flat_mask[valid_coords] > 0.5]
            known_ids = valid_ids[flat_mask[valid_coords] > 0.5]
            rhs[known_ids, :] -= weight * flat_img[:, known_coords].T

            missing_coords = valid_coords[flat_mask[valid_coords] < 0.5]
            variable_ids = coord_to_var_idx[missing_coords]
            missing_ids = valid_ids[flat_mask[valid_coords] < 0.5]
            system[missing_ids, variable_ids] = weight

            invalid_ids = np.argwhere(valid == 0).flatten()
            if len(invalid_ids) > 0:
                sum_neighbors[invalid_ids] -= weight

        system[np.arange(num_equations), np.arange(num_equations)] = -sum_neighbors
        return system, rhs

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        flat_mask = mask.reshape(-1)
        indices_linear = np.argwhere(flat_mask.cpu().numpy() == 0).flatten()
        if len(indices_linear) == 0:
            return img.clone()

        system, rhs = self.setup_sparse_system(mask.cpu().numpy(), img.cpu().numpy(), self.weighting)
        solution = torch.tensor(spsolve(csc_matrix(system), rhs), dtype=img.dtype)

        filled = img.reshape(img.shape[0], -1).clone()
        filled[:, indices_linear] = solution.t()
        if self.noise > 0:
            filled[:, indices_linear] += self.noise * torch.randn_like(filled[:, indices_linear])
        return filled.reshape_as(img)

    def batched_call(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        in_device = img.device
        outputs = [self.__call__(img[idx].cpu(), mask[idx].cpu()) for idx in range(len(img))]
        return torch.stack(outputs).to(in_device)


def _from_str(imputer_str: str) -> BaseImputer:
    if imputer_str == "linear":
        return NoisyLinearImputer()
    if imputer_str == "fixed":
        return ChannelMeanImputer()
    if imputer_str == "zero":
        return ZeroImputer()
    if imputer_str == "blur":
        return BlurImputer()
    raise ValueError(f"Unknown imputer string: {imputer_str}")
