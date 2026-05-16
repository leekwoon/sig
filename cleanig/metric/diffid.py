import numpy as np
from typing import Optional, List, Tuple, Union

import torch
import torch.nn.functional as F


def _gaussian_blur_flat_images(
    flat_images: torch.Tensor,
    original_shape: tuple,
    kernel_size: int = 11,
    sigma: float = 5.0,
) -> torch.Tensor:
    import torchvision.transforms.functional as TF

    images_reshaped = flat_images.view(original_shape)
    blurred = TF.gaussian_blur(
        images_reshaped,
        [kernel_size, kernel_size],
        sigma=[sigma, sigma],
    )
    return blurred.view(flat_images.shape[0], -1)


def compute_diffid_metrics(
    model: torch.nn.Module,
    images: torch.Tensor,
    attributions: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    ratios: Optional[List[float]] = None,
    baseline_method: str = 'mean',
    use_soft_metric: bool = False,
    return_curves: bool = False
) -> dict:
    if ratios is None:
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    device = images.device
    batch_size = images.shape[0]

    # Get original predictions if labels not provided
    if labels is None:
        with torch.no_grad():
            outputs = model(images)
            labels = torch.argmax(outputs, dim=1)

    # Get original confidences for soft metric
    if use_soft_metric:
        with torch.no_grad():
            original_outputs = model(images)
            original_probs = F.softmax(original_outputs, dim=1)
            # Get probability of correct class for each sample
            original_confidences = original_probs[torch.arange(batch_size), labels]

    # Flatten for easier manipulation
    # Use reshape instead of view to handle non-contiguous tensors (e.g., from WebP compression)
    num_pixels = images.shape[1] * images.shape[2] * images.shape[3]
    flat_images = images.reshape(batch_size, -1)
    flat_attributions = torch.abs(attributions).reshape(batch_size, -1)

    insertion_scores = []
    deletion_scores = []
    per_example_insertion_scores = []
    per_example_deletion_scores = []

    for ratio in ratios:
        num_perturb = int(num_pixels * ratio)

        if use_soft_metric:
            deletion_score = _evaluate_perturbation_soft_batch(
                model, flat_images, flat_attributions, labels,
                num_perturb, images.shape, original_confidences,
                descending=True, baseline_method=baseline_method
            )

            insertion_score = _evaluate_perturbation_soft_batch(
                model, flat_images, flat_attributions, labels,
                num_perturb, images.shape, original_confidences,
                descending=False, baseline_method=baseline_method
            )
        else:
            deletion_score = _evaluate_perturbation_batch(
                model, flat_images, flat_attributions, labels,
                num_perturb, images.shape, descending=True, baseline_method=baseline_method
            )

            insertion_score = _evaluate_perturbation_batch(
                model, flat_images, flat_attributions, labels,
                num_perturb, images.shape, descending=False, baseline_method=baseline_method
            )

        insertion_np = insertion_score.detach().cpu().numpy()
        deletion_np = deletion_score.detach().cpu().numpy()
        per_example_insertion_scores.append(insertion_np)
        per_example_deletion_scores.append(deletion_np)
        insertion_scores.append(float(insertion_np.mean()))
        deletion_scores.append(float(deletion_np.mean()))

    per_example_insertion_scores = np.stack(per_example_insertion_scores, axis=1)
    per_example_deletion_scores = np.stack(per_example_deletion_scores, axis=1)
    per_example_diffid_scores = per_example_insertion_scores - per_example_deletion_scores

    per_example_diffid = per_example_diffid_scores.mean(axis=1)
    per_example_insertion_auc = per_example_insertion_scores.mean(axis=1)
    per_example_deletion_auc = per_example_deletion_scores.mean(axis=1)

    diffid_scores = (per_example_diffid_scores.mean(axis=0)).tolist()

    return {
        'diffid': float(per_example_diffid.mean()),
        'insertion_auc': float(per_example_insertion_auc.mean()),
        'deletion_auc': float(per_example_deletion_auc.mean()),
        'ratios': list(ratios),
        'insertion_scores': insertion_scores,
        'deletion_scores': deletion_scores,
        'diffid_scores': diffid_scores,
        'metric_type': 'soft' if use_soft_metric else 'binary',
        'per_example_insertion_scores': per_example_insertion_scores,
        'per_example_deletion_scores': per_example_deletion_scores,
        'per_example_diffid_scores': per_example_diffid_scores,
        'per_example_diffid': per_example_diffid,
        'per_example_insertion_auc': per_example_insertion_auc,
        'per_example_deletion_auc': per_example_deletion_auc,
    }


def compute_diffid_score(
    model: torch.nn.Module,
    images: torch.Tensor,
    attributions: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    ratios: Optional[List[float]] = None,
    baseline_method: str = 'mean',
    use_soft_metric: bool = False,
    return_curves: bool = False
) -> Union[float, Tuple[float, dict]]:
    metrics = compute_diffid_metrics(
        model=model,
        images=images,
        attributions=attributions,
        labels=labels,
        ratios=ratios,
        baseline_method=baseline_method,
        use_soft_metric=use_soft_metric,
    )

    if return_curves:
        return metrics['diffid'], {
            'ratios': metrics['ratios'],
            'insertion_scores': metrics['insertion_scores'],
            'deletion_scores': metrics['deletion_scores'],
            'diffid_scores': metrics['diffid_scores'],
            'metric_type': metrics['metric_type'],
        }

    return metrics['diffid']


def _evaluate_perturbation_batch(
    model: torch.nn.Module,
    flat_images: torch.Tensor,
    flat_attributions: torch.Tensor,
    labels: torch.Tensor,
    num_perturb: int,
    original_shape: tuple,
    descending: bool,
    baseline_method: str = 'mean'
) -> float:
    """
    Evaluate model accuracy after perturbing pixels based on attribution importance.

    Args:
        model: Classification model
        flat_images: Flattened images [batch_size, num_pixels]
        flat_attributions: Flattened attribution maps [batch_size, num_pixels]
        labels: True labels
        num_perturb: Number of pixels to perturb
        original_shape: Original image shape for reshaping
        descending: If True, perturb most important pixels (deletion)
                   If False, perturb least important pixels (insertion)
        baseline_method: Method for replacing pixels

    Returns:
        Accuracy vector after perturbation [batch_size]
    """
    batch_size = flat_images.shape[0]
    device = flat_images.device

    # Sort pixels by attribution importance
    sorted_indices = torch.argsort(flat_attributions, dim=1, descending=descending)
    perturb_indices = sorted_indices[:, :num_perturb]

    # Create baseline values
    if baseline_method == 'mean':
        # Use mean of remaining pixels
        perturb_mask = torch.ones_like(flat_images)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        perturb_mask[batch_indices, perturb_indices] = 0

        # Calculate mean of non-perturbed pixels
        sum_preserved = (flat_images * perturb_mask).sum(dim=1, keepdim=True)
        count_preserved = perturb_mask.sum(dim=1, keepdim=True)
        baseline_values = sum_preserved / (count_preserved + 1e-8)

    elif baseline_method == 'zero':
        baseline_values = torch.zeros(batch_size, 1, device=device)

    elif baseline_method == 'blur':
        baseline_values = _gaussian_blur_flat_images(flat_images, original_shape)
    else:
        raise ValueError(f"Unknown baseline method: {baseline_method}")

    # Apply perturbation
    perturbed_images = flat_images.clone()
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)

    if baseline_method == 'blur':
        perturbed_images[batch_indices, perturb_indices] = baseline_values[batch_indices, perturb_indices]
    else:
        perturbed_images[batch_indices, perturb_indices] = baseline_values.expand_as(perturbed_images)[batch_indices, perturb_indices]

    # Reshape and evaluate
    perturbed_images = perturbed_images.view(original_shape)

    with torch.no_grad():
        outputs = model(perturbed_images)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float()

    return accuracy


def _evaluate_perturbation(
    model: torch.nn.Module,
    flat_images: torch.Tensor,
    flat_attributions: torch.Tensor,
    labels: torch.Tensor,
    num_perturb: int,
    original_shape: tuple,
    descending: bool,
    baseline_method: str = 'mean'
) -> float:
    return float(
        _evaluate_perturbation_batch(
            model=model,
            flat_images=flat_images,
            flat_attributions=flat_attributions,
            labels=labels,
            num_perturb=num_perturb,
            original_shape=original_shape,
            descending=descending,
            baseline_method=baseline_method,
        ).mean().item()
    )


def _evaluate_perturbation_soft_batch(
    model: torch.nn.Module,
    flat_images: torch.Tensor,
    flat_attributions: torch.Tensor,
    labels: torch.Tensor,
    num_perturb: int,
    original_shape: tuple,
    original_confidences: torch.Tensor,
    descending: bool,
    baseline_method: str = 'mean'
) -> float:
    """
    Evaluate model confidence retention after perturbing pixels (SOFT METRIC).

    Instead of binary accuracy, this measures how much the confidence
    in the correct class is retained after perturbation.

    Args:
        model: Classification model
        flat_images: Flattened images [batch_size, num_pixels]
        flat_attributions: Flattened attribution maps [batch_size, num_pixels]
        labels: True labels
        num_perturb: Number of pixels to perturb
        original_shape: Original image shape for reshaping
        original_confidences: Original confidence scores for correct classes
        descending: If True, perturb most important pixels (deletion)
                   If False, perturb least important pixels (insertion)
        baseline_method: Method for replacing pixels

    Returns:
        Confidence retention ratio vector [batch_size]
    """
    batch_size = flat_images.shape[0]
    device = flat_images.device

    # Sort pixels by attribution importance
    sorted_indices = torch.argsort(flat_attributions, dim=1, descending=descending)
    perturb_indices = sorted_indices[:, :num_perturb]

    # Create baseline values
    if baseline_method == 'mean':
        # Use mean of remaining pixels
        perturb_mask = torch.ones_like(flat_images)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        perturb_mask[batch_indices, perturb_indices] = 0

        # Calculate mean of non-perturbed pixels
        sum_preserved = (flat_images * perturb_mask).sum(dim=1, keepdim=True)
        count_preserved = perturb_mask.sum(dim=1, keepdim=True)
        baseline_values = sum_preserved / (count_preserved + 1e-8)

    elif baseline_method == 'zero':
        baseline_values = torch.zeros(batch_size, 1, device=device)

    elif baseline_method == 'blur':
        baseline_values = _gaussian_blur_flat_images(flat_images, original_shape)
    else:
        raise ValueError(f"Unknown baseline method: {baseline_method}")

    # Apply perturbation
    perturbed_images = flat_images.clone()
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)

    if baseline_method == 'blur':
        perturbed_images[batch_indices, perturb_indices] = baseline_values[batch_indices, perturb_indices]
    else:
        perturbed_images[batch_indices, perturb_indices] = baseline_values.expand_as(perturbed_images)[batch_indices, perturb_indices]

    # Reshape and evaluate
    perturbed_images = perturbed_images.view(original_shape)

    with torch.no_grad():
        outputs = model(perturbed_images)
        probs = F.softmax(outputs, dim=1)
        # Get probability of correct class after perturbation
        perturbed_confidences = probs[torch.arange(batch_size), labels]

    # Calculate confidence retention ratio
    # Ratio of confidence after perturbation to original confidence
    confidence_retention = perturbed_confidences / (original_confidences + 1e-8)

    # Clamp to [0, 1] range (in case of numerical issues)
    confidence_retention = confidence_retention.clamp(0.0, 1.0)

    return confidence_retention


def _evaluate_perturbation_soft(
    model: torch.nn.Module,
    flat_images: torch.Tensor,
    flat_attributions: torch.Tensor,
    labels: torch.Tensor,
    num_perturb: int,
    original_shape: tuple,
    original_confidences: torch.Tensor,
    descending: bool,
    baseline_method: str = 'mean'
) -> float:
    return float(
        _evaluate_perturbation_soft_batch(
            model=model,
            flat_images=flat_images,
            flat_attributions=flat_attributions,
            labels=labels,
            num_perturb=num_perturb,
            original_shape=original_shape,
            original_confidences=original_confidences,
            descending=descending,
            baseline_method=baseline_method,
        ).mean().item()
    )
