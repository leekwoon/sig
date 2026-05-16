"""Minimal no-retraining ROAD runner adapted from the official ROAD implementation."""

from __future__ import annotations

import torch

from .imputations import NoisyLinearImputer, _from_str
from .imputed_dataset import ExplanationCache, ImputedDataset
from .retraining import road_eval


def run_road(
    model,
    dataset_test,
    explanations_test,
    transform_test,
    percentages,
    morf: bool = True,
    batch_size: int = 32,
    imputation=None,
    num_workers: int = 8,
    device: str | None = None,
    explanation_cache: ExplanationCache | None = None,
):
    if imputation is None:
        imputation = NoisyLinearImputer(noise=0.01)
    if isinstance(imputation, str):
        imputation = _from_str(imputation)

    res_acc = torch.zeros(len(percentages), dtype=torch.float32)
    prob_acc = torch.zeros(len(percentages), dtype=torch.float32)

    for idx, percentage in enumerate(percentages):
        ds_test_imputed = ImputedDataset(
            dataset_test,
            mask=explanations_test,
            th_p=percentage,
            remove=morf,
            imputation=imputation,
            transform=transform_test,
            prediction=None,
            use_cache=False,
            explanation_cache=explanation_cache,
        )
        testloader = torch.utils.data.DataLoader(
            ds_test_imputed,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        acc_avg, prob_avg = road_eval(model, testloader, device=device)
        res_acc[idx] = float(acc_avg)
        prob_acc[idx] = float(prob_avg)

    return res_acc, prob_acc
