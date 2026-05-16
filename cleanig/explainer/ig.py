"""
Reference:
    Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
"""
import torch

from cleanig.explainer.path_utils import LinearPathGenerator


class IGExplainer:
    def __init__(self, model, baseline_method, num_steps, device, exp_obj='prob', preprocess_fn=None):
        self.model = model
        self.baseline_method = baseline_method
        self.num_steps = num_steps
        self.device = device
        self.exp_obj = exp_obj
        if preprocess_fn is not None:
            self.preprocess_fn = preprocess_fn
        else:
            self.preprocess_fn = lambda x: x

        self.path_generator = LinearPathGenerator(
            baseline_method=self.baseline_method,
            preprocess_fn=self.preprocess_fn,
            device=self.device,
            num_steps=self.num_steps,
        )

    def get_attributions(self, inputs, labels=None, return_paths=False):
        paths = self.path_generator.get_paths(inputs, labels)
        attributions = compute_ig(self.model, paths, labels, self.exp_obj)
        if return_paths:
            return attributions, paths
        else:
            return attributions    


def compute_ig(model, paths, labels=None, exp_obj='prob'):
    """
    Compute attributions along paths.
        
    Args:
        model: The model to compute gradients for
        paths: Tensor of shape [B, num_steps, C, H, W] representing the path
        labels: Target labels [B]
        exp_obj: Objective function ('prob' or 'logit')
    
    Returns:
        attributions: Attribution maps [B, C, H, W]
    """    
    # Get gradients at each point along the path
    grads = get_grads(model, paths, labels, exp_obj)
    
    # Vectorized computation of attributions
    # Compute all deltas at once: differences between consecutive steps
    deltas = paths[:, 1:] - paths[:, :-1]  # [B, num_steps-1, C, H, W]
    
    # Use gradients from all positions except the last
    grads_for_deltas = grads[:, :-1]  # [B, num_steps-1, C, H, W]
    
    # Element-wise multiply and sum along the path dimension
    attributions = (deltas * grads_for_deltas).sum(dim=1)  # [B, C, H, W]
    
    return attributions


def get_grads(model, paths, labels=None, exp_obj='prob'):
    """Original implementation - processes each step separately."""
    device = paths.device

    grads = torch.zeros(paths.shape).float().to(device)

    for i in range(paths.shape[1]):
        particular_slice = paths[:, i]
        particular_slice.requires_grad = True

        output = model(particular_slice)
        if labels is None:
            labels = output.max(1, keepdim=False)[1]

        if exp_obj == 'logit':
            output = output[torch.arange(output.shape[0]), labels]   
        elif exp_obj == 'prob':
            output = torch.softmax(output, dim=-1)
            output = output[torch.arange(output.shape[0]), labels]
        else:
            raise ValueError(f'Invalid objective function: {exp_obj}')

        grad = torch.autograd.grad(output.sum(), particular_slice)[0].detach()

        grads[:, i, :] = grad

    return grads
