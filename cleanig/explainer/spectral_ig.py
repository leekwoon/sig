import torch

from cleanig.explainer.ig import IGExplainer, compute_ig
from cleanig.explainer.path_utils import SpectralPathGenerator


class SpectralIGExplainer(IGExplainer):
    def __init__(
        self,
        model,
        baseline_method='zero',
        num_steps=200,
        device='cuda',
        exp_obj='prob',
        preprocess_fn=None,
        overlap=0.5,
        spectral_mode='svd',
        channel_mode='per_channel',
        gating_schedule='linear',
        gating_sigmoid_k=12.0,
        wavelet_levels=4,
        laplacian_levels=4,
    ):
        self.model = model
        self.baseline_method = baseline_method
        self.num_steps = num_steps
        self.device = device
        self.exp_obj = exp_obj
        self.overlap = overlap
        self.spectral_mode = spectral_mode
        self.channel_mode = channel_mode
        self.gating_schedule = gating_schedule
        self.gating_sigmoid_k = gating_sigmoid_k
        self.wavelet_levels = wavelet_levels
        self.laplacian_levels = laplacian_levels

        if preprocess_fn is not None:
            self.preprocess_fn = preprocess_fn
        else:
            self.preprocess_fn = lambda x: x

        self.path_generator = SpectralPathGenerator(
            baseline_method=self.baseline_method,
            preprocess_fn=self.preprocess_fn,
            device=self.device,
            num_steps=self.num_steps,
            overlap=self.overlap,
            spectral_mode=self.spectral_mode,
            channel_mode=self.channel_mode,
            gating_schedule=self.gating_schedule,
            gating_sigmoid_k=self.gating_sigmoid_k,
            wavelet_levels=self.wavelet_levels,
            laplacian_levels=self.laplacian_levels,
        )

    def get_attributions(self, inputs, labels=None, return_paths=False):
        paths = self.path_generator.get_paths(inputs, labels)
        attributions = compute_ig(self.model, paths, labels, self.exp_obj)
        if return_paths:
            return attributions, paths
        return attributions
