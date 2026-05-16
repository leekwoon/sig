"""
Reference:
    Xu et al., "Attribution in Scale and Space", CVPR 2020
"""
from cleanig.explainer.ig import IGExplainer
from cleanig.explainer.path_utils import BlurPathGenerator


class BIGExplainer(IGExplainer):
    """Blur Integrated Gradients explainer using blur path from blurred to original."""

    def __init__(
        self,
        model,
        num_steps=100,
        device='cuda',
        exp_obj='prob',
        preprocess_fn=None,
        undo_preprocess_fn=None,
        max_sigma=50,
        sqrt=False,
    ):
        self.model = model
        self.num_steps = num_steps
        self.device = device
        self.exp_obj = exp_obj
        self.max_sigma = max_sigma
        self.sqrt = sqrt

        if preprocess_fn is not None:
            self.preprocess_fn = preprocess_fn
        else:
            self.preprocess_fn = lambda x: x

        self.undo_preprocess_fn = undo_preprocess_fn

        self.path_generator = BlurPathGenerator(
            preprocess_fn=self.preprocess_fn,
            device=self.device,
            num_steps=self.num_steps,
            max_sigma=self.max_sigma,
            sqrt=self.sqrt,
            undo_preprocess_fn=self.undo_preprocess_fn,
        )
