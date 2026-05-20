import math
import numpy as np
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


def _gaussian_blur_baseline(input_tensor, kernel_size=11, sigma=5.0):
    import torchvision.transforms.functional as TF

    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(int(kernel_size), 3)
    return TF.gaussian_blur(input_tensor, [kernel_size, kernel_size], sigma)


def build_reference_baseline(input_tensor, baseline_method, preprocess_fn=None, device='cuda'):
    """
    Build a reference baseline for already-preprocessed inputs.

    Supported baselines:
    - zero: black image in raw space, then preprocessed
    - mean: dataset-mean image, which is zero in normalized space
    - blur: Gaussian-blurred input
    """
    if preprocess_fn is None:
        preprocess_fn = lambda x: x

    if baseline_method == 'zero':
        baseline = preprocess_fn(torch.zeros_like(input_tensor).float().to(device))
    elif baseline_method == 'mean':
        baseline = torch.zeros_like(input_tensor).float().to(device)
    elif baseline_method == 'blur':
        baseline = _gaussian_blur_baseline(input_tensor.to(device))
    else:
        raise ValueError(f'Invalid baseline method: {baseline_method}')

    return baseline.to(device)


def slerp(t, v0, v1, dot_threshold=0.9995):
    """
    Spherical linear interpolation between two vectors.
    Falls back to lerp when vectors are nearly parallel.
    """
    v0_flat = v0.reshape(-1).float()
    v1_flat = v1.reshape(-1).float()
    
    norm0 = torch.norm(v0_flat)
    norm1 = torch.norm(v1_flat)
    
    if norm0 < 1e-9 or norm1 < 1e-9:
        return v0 * (1 - t) + v1 * t
    
    v0_unit = v0_flat / norm0
    v1_unit = v1_flat / norm1
    
    dot = torch.clamp(torch.sum(v0_unit * v1_unit), -1.0, 1.0)
    
    if torch.abs(dot) > dot_threshold:
        return v0 * (1 - t) + v1 * t
    
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    
    result = s0 * v0 + s1 * v1
    return result


class PathGenerator(ABC):
    """
    Abstract class for generating paths.
    [!] Assume that the input tensor is already preprocessed.
    """
    def __init__(self, baseline_method=None, preprocess_fn=None, device='cuda'):
        self.baseline_method = baseline_method
        self.device = device
        if preprocess_fn is not None:
            self.preprocess_fn = preprocess_fn
        else:
            self.preprocess_fn = lambda x: x

    def get_baselines(self, input_tensor):
        return build_reference_baseline(
            input_tensor=input_tensor,
            baseline_method=self.baseline_method,
            preprocess_fn=self.preprocess_fn,
            device=self.device,
        )

    @abstractmethod
    def get_paths(self, inputs, labels=None):
        pass


class LinearPathGenerator(PathGenerator):
    """
    Generate linear paths between the input and the baseline.
    [!] Assume that the input tensor is already preprocessed.
    """
    def __init__(self, baseline_method, preprocess_fn, device, num_steps):
        super().__init__(baseline_method, preprocess_fn, device)

        self.num_steps = num_steps

    def get_paths(self, inputs, labels=None):
        baselines = self.get_baselines(inputs)

        batch_size = inputs.shape[0]
        input_dims = list(inputs.size())[1:]
        num_input_dims = len(input_dims)

        baselines = baselines.unsqueeze(1).repeat(1, self.num_steps, *[1] * num_input_dims)
        if self.num_steps == 1:
            alpha = torch.cat([torch.Tensor([1.0]) for _ in range(batch_size)]).to(self.device)
        else:
            alpha = torch.cat([torch.linspace(0, 1, self.num_steps) for _ in range(batch_size)]).to(self.device)

        shape = [batch_size, self.num_steps] + [1] * num_input_dims
        interp_coef = alpha.view(*shape).to(self.device)

        end_point_baselines = (1.0 - interp_coef) * baselines
        inputs_expand_mult = inputs.unsqueeze(1)
        end_point_inputs = interp_coef * inputs_expand_mult

        paths = end_point_inputs + end_point_baselines.to(self.device)
        return paths


class GuidedPathGenerator(PathGenerator):
    """
    Generate guided paths based on Guided Integrated Gradients algorithm.
    The path adaptively selects features with lowest gradients at each step.
    [!] Assume that the input tensor is already preprocessed.
    """
    
    def __init__(
        self,
        baseline_method,
        preprocess_fn,
        model,
        device,
        num_steps,
        fraction=0.1,
        exp_obj='prob',
    ):
        """
        Initialize Guided Path Generator.
        
        Args:
            baseline_method: Method for generating baselines
            preprocess_fn: Preprocessing function
            device: Device to run on
            num_steps: Number of integration steps
            fraction: Fraction of features to select at each step
            max_dist: Maximum relative L1 distance from straight path
            model: Model for computing gradients
            exp_obj: Objective function ('prob' or 'logit')
        """
        super().__init__(baseline_method, preprocess_fn, device)
        self.num_steps = num_steps
        self.fraction = fraction
        self.model = model
        self.exp_obj = exp_obj

    def _l1_distance(self, x1, x2):
        """Returns L1 distance between two tensors."""
        return torch.abs(x1 - x2).sum()

    def _get_gradients(self, x, labels=None):
        """Compute gradients for current position."""
        x = x.clone().detach().requires_grad_(True)
        
        output = self.model(x)
        if labels is None:
            labels = output.max(1, keepdim=False)[1]
        
        if self.exp_obj == 'logit':
            output = output[torch.arange(output.shape[0]), labels]
        elif self.exp_obj == 'prob':
            output = torch.softmax(output, dim=-1)
            output = output[torch.arange(output.shape[0]), labels]
        else:
            raise ValueError(f'Invalid objective function: {self.exp_obj}')
        
        grad = torch.autograd.grad(output.sum(), x)[0].detach()
        return grad

    def _translate_alpha_to_x(self, alpha, x_input, x_baseline):
        """Translates alpha to point coordinates within interval."""
        return x_baseline + (x_input - x_baseline) * alpha

    def _translate_x_to_alpha(self, x, x_input, x_baseline):
        """Translates a point on path to its corresponding alpha value."""
        # Avoid division by zero
        diff = x_input - x_baseline
        alpha = torch.where(
            diff != 0,
            (x - x_baseline) / diff,
            torch.zeros_like(x)
        )
        return alpha

    def get_paths(self, inputs, labels=None):
        # EPSILON for numerical stability
        EPSILON = 1e-9

        batch_size = inputs.shape[0]
        all_paths = []

        for b in range(batch_size):
            x_input = inputs[b]
            x_baseline = self.get_baselines(inputs[b:b+1]).squeeze(0)

            # Initialize
            x = x_baseline.clone()
            l1_total = self._l1_distance(x_input, x_baseline)

            paths = []

            for step in range(self.num_steps):
                # Store current position in path
                if step == self.num_steps - 1:
                    x = x_input.clone()
                    paths.append(x)
                    break
                else:
                    paths.append(x.clone())

                # Get gradients at current position
                grad_actual = self._get_gradients(x[None], labels)[0]
                grad = grad_actual.clone()

                # Unbounded GIG
                alpha_min, alpha_max = 0.0, 1.0 
                x_min, x_max = x_baseline, x_input

                # Target L1 distance for this step
                l1_target = l1_total * (1 - (step + 1) / self.num_steps)

                gamma = np.inf
                while gamma > 1.0:
                    # Translate current x to alpha space
                    x_alpha = self._translate_x_to_alpha(x, x_input, x_baseline)

                    # Handle NaN values (when x_input == x_baseline for some features)
                    # These features should be set to alpha_max
                    x_alpha = torch.where(
                        torch.isnan(x_alpha),
                        torch.tensor(alpha_max).to(self.device),
                        x_alpha
                    )

                    # Ensure x stays within bounds - features behind should catch up
                    # x = torch.where(x_alpha < alpha_min, x_min, x)
                    debug = x_alpha < alpha_min
                    assert debug.sum() < 1
                    # print(debug.sum())

                    # Calculate current L1 distance
                    l1_current = self._l1_distance(x, x_input)

                    # Check if we're close enough to target
                    close_enough = torch.isclose(
                        l1_target, l1_current, 
                        rtol=EPSILON, atol=EPSILON
                    )
                    if close_enough:
                        break

                    # Features that reached `x_max` should not be included in the selection.
                    # Assign very high gradients to them so they are excluded.
                    at_max = torch.abs(x - x_max) < EPSILON
                    grad = torch.where(at_max, torch.tensor(float('inf')).to(self.device), grad)

                    abs_grad = grad.abs()
                    threshold = torch.quantile(abs_grad.reshape(-1), self.fraction, interpolation="lower")

                    # Select features with gradients below threshold
                    s = (torch.abs(grad) <= threshold) & (grad != float('inf'))

                    # Compute how much we can move selected features
                    l1_s = (torch.abs(x - x_max) * s).sum()

                    # Calculate ratio `gamma` that show how much the selected features should
                    # be changed toward `x_max` to close the gap between current L1 and target
                    # L1.
                    if l1_s > 0:
                        gamma = (l1_current - l1_target) / l1_s
                    else:
                        gamma = np.inf

                    if gamma > 1.0:
                        # Move selected features as much as possible toward target
                        x = torch.where(s, x_max, x)
                    else:
                        # Tiny negative gamma can appear from floating-point drift near the target L1 budget.
                        if gamma <= 0:
                            gamma_value = float(gamma)
                            if abs(gamma_value) <= 1e-6:
                                break
                            raise AssertionError(f"Gamma should be positive, got {gamma}")

                        # Move selected features by gamma fraction toward target
                        # x_new = x + gamma * (x_max - x
                        x_new = self._translate_alpha_to_x(
                            torch.tensor(gamma).to(self.device), x_max, x
                        )
                        x = torch.where(s, x_new, x)

            paths = torch.stack(paths, dim=0)  # [num_steps, C, H, W]
            all_paths.append(paths)

        # Stack paths
        all_paths = torch.stack(all_paths, dim=0)  # [B, num_steps, C, H, W]

        return all_paths


class LatentGuidedPathGenerator(GuidedPathGenerator):
    """
    Generate guided paths in VAE latent space.
    Similar to GuidedPathGenerator but operates in latent space and decodes to pixel space.
    """
    def __init__(
        self,
        vae,
        baseline_method,
        preprocess_fn,
        model,
        device,
        num_steps,
        fraction=0.1,
        exp_obj='prob',
        use_slerp=False,
    ):
        super().__init__(baseline_method, preprocess_fn, model, device, num_steps, fraction, exp_obj)
        self.vae = vae
        self.use_slerp = use_slerp
    
    def _slerp_update(self, z_current, z_target, gamma, selection_mask):
        z_new = z_current.clone()
        
        if selection_mask.sum() == 0:
            return z_new
        
        z_sel = z_current[selection_mask]
        z_tgt = z_target[selection_mask]
        
        z_new[selection_mask] = slerp(gamma, z_sel, z_tgt)
        
        return z_new

    def _get_latent_gradients(self, z, labels=None):
        """Compute gradients with respect to latent space."""
        z = z.clone().detach().requires_grad_(True)
        
        # Decode latent to image space
        x = self.vae.decode(z)
        
        # Get model output
        output = self.model(x)
        if labels is None:
            labels = output.max(1, keepdim=False)[1]
        
        if self.exp_obj == 'logit':
            output = output[torch.arange(output.shape[0]), labels]
        elif self.exp_obj == 'prob':
            output = torch.softmax(output, dim=-1)
            output = output[torch.arange(output.shape[0]), labels]
        else:
            raise ValueError(f'Invalid objective function: {self.exp_obj}')
        
        # Compute gradient with respect to latent z
        grad = torch.autograd.grad(output.sum(), z)[0].detach()
        return grad

    def get_paths(self, inputs, labels=None):
        # EPSILON for numerical stability
        EPSILON = 1e-9

        batch_size = inputs.shape[0]
        all_paths = []

        for b in range(batch_size):
            x_input = inputs[b:b+1]  # Keep batch dimension for VAE
            x_baseline = self.get_baselines(inputs[b:b+1])

            # Encode to latent space
            z_input = self.vae.encode(x_input).squeeze(0)
            z_baseline = self.vae.encode(x_baseline).squeeze(0)

            # Initialize in latent space
            z = z_baseline.clone()
            l1_total = self._l1_distance(z_input, z_baseline)

            paths = []

            for step in range(self.num_steps):
                # Store current position in path (decoded to image space)
                if step == self.num_steps - 1:
                    z = z_input.clone()
                    x = self.vae.decode(z.unsqueeze(0)).squeeze(0)
                    paths.append(x)
                    break
                else:
                    x = self.vae.decode(z.unsqueeze(0)).squeeze(0)
                    paths.append(x.clone())

                # Get gradients in latent space
                grad_actual = self._get_latent_gradients(z[None], labels)[0]
                grad = grad_actual.clone()

                # Unbounded GIG in latent space
                alpha_min, alpha_max = 0.0, 1.0
                z_min, z_max = z_baseline, z_input

                # Target L1 distance for this step (in latent space)
                l1_target = l1_total * (1 - (step + 1) / self.num_steps)

                gamma = np.inf
                while gamma > 1.0:
                    # Translate current z to alpha space
                    z_alpha = self._translate_x_to_alpha(z, z_input, z_baseline)

                    # Handle NaN values (when z_input == z_baseline for some features)
                    z_alpha = torch.where(
                        torch.isnan(z_alpha),
                        torch.tensor(alpha_max).to(self.device),
                        z_alpha
                    )

                    # # Ensure z stays within bounds
                    # debug = z_alpha < alpha_min
                    # assert debug.sum() < 1

                    # Calculate current L1 distance in latent space
                    l1_current = self._l1_distance(z, z_input)

                    # Check if we're close enough to target
                    close_enough = torch.isclose(
                        l1_target, l1_current,
                        rtol=EPSILON, atol=EPSILON
                    )
                    if close_enough:
                        break

                    # Features that reached z_max should not be included in the selection
                    at_max = torch.abs(z - z_max) < EPSILON
                    grad = torch.where(at_max, torch.tensor(float('inf')).to(self.device), grad)

                    abs_grad = grad.abs()
                    threshold = torch.quantile(abs_grad.reshape(-1), self.fraction, interpolation="lower")

                    # Select features with gradients below threshold
                    s = (torch.abs(grad) <= threshold) & (grad != float('inf'))

                    # Compute how much we can move selected features in latent space
                    l1_s = (torch.abs(z - z_max) * s).sum()

                    # Calculate ratio gamma
                    if l1_s > 0:
                        gamma = (l1_current - l1_target) / l1_s
                    else:
                        gamma = np.inf

                    if gamma > 1.0:
                        # Move selected features fully to z_max
                        z = torch.where(s, z_max, z)
                    else:
                        # Move selected features by gamma fraction toward target
                        # assert gamma > 0, f"Gamma should be positive, got {gamma}"
                        if self.use_slerp:
                            z = self._slerp_update(z, z_max, gamma, s)
                        else:
                            z_new = self._translate_alpha_to_x(
                                torch.tensor(gamma).to(self.device), z_max, z
                            )
                            z = torch.where(s, z_new, z)

            paths = torch.stack(paths, dim=0)  # [num_steps, C, H, W]
            all_paths.append(paths)

        # Stack paths
        all_paths = torch.stack(all_paths, dim=0)  # [B, num_steps, C, H, W]

        return all_paths


class LatentLinearPathGenerator(PathGenerator):
    """
    Generate linear paths in VAE latent space, then decode to pixel space.
    Path: baseline_latent → input_latent (linear interpolation) → decode each step

    This is used for Enhanced Integrated Gradients (EIG).
    """

    def __init__(
        self,
        vae,
        baseline_method,
        preprocess_fn,
        device,
        num_steps,
        use_slerp=True,
    ):
        """
        Initialize Latent Linear Path Generator.

        Args:
            vae: VAE model with encode() and decode() methods
            baseline_method: Method for generating baselines ('zero')
            preprocess_fn: Preprocessing function
            device: Device to run on
            num_steps: Number of interpolation steps
        """
        super().__init__(baseline_method, preprocess_fn, device)
        self.vae = vae
        self.num_steps = num_steps
        self.use_slerp = use_slerp

    def get_paths(self, inputs, labels=None):
        """
        Generate linear path in latent space and decode to pixel space.

        1. Encode inputs and baselines to latent space
        2. Linear interpolation in latent space
        3. Decode each latent point to pixel space

        Args:
            inputs: Input images [B, C, H, W]
            labels: Target labels [B] (not used, for API compatibility)

        Returns:
            paths: Tensor [B, num_steps, C, H, W]
        """
        batch_size = inputs.shape[0]
        all_paths = []

        for b in range(batch_size):
            x_input = inputs[b:b+1]
            x_baseline = self.get_baselines(x_input)

            # Encode to latent space
            z_input = self.vae.encode(x_input)
            z_baseline = self.vae.encode(x_baseline)

            paths = []
            for i in range(self.num_steps):
                alpha = i / (self.num_steps - 1) if self.num_steps > 1 else 1.0
                if self.use_slerp:
                    z = slerp(alpha, z_baseline, z_input)
                else:
                    z = z_baseline + alpha * (z_input - z_baseline)
                x = self.vae.decode(z).squeeze(0)
                paths.append(x)

            paths = torch.stack(paths, dim=0)  # [num_steps, C, H, W]
            all_paths.append(paths)

        return torch.stack(all_paths, dim=0)  # [B, num_steps, C, H, W]


class GeodesicPathGenerator(PathGenerator):
    """
    Generate geodesic paths in VAE latent space using energy minimization.
    The geodesic path minimizes the path energy on the VAE manifold.

    Reference: Jha et al., "Manifold Integrated Gradients", ICML 2024
    
    The algorithm finds a geodesic curve γ(t) on the data manifold by minimizing
    the path energy E[γ] = ∫||γ''(t)||² dt, where the acceleration is computed
    using the Jacobian of the decoder.
    """

    def __init__(
        self,
        vae,
        baseline_method,
        preprocess_fn,
        device,
        num_steps,
        alpha=0.01,
        max_iterations=10,
        epsilon=1e-5,
    ):
        """
        Initialize Geodesic Path Generator.

        Args:
            vae: VAE model with encode() and decode() methods
            baseline_method: Method for generating baselines ('zero')
            preprocess_fn: Preprocessing function
            device: Device to run on
            num_steps: Number of interpolation points (T)
            alpha: Learning rate for geodesic optimization
            max_iterations: Maximum iterations for geodesic path optimization
            epsilon: Convergence threshold for energy
        """
        super().__init__(baseline_method, preprocess_fn, device)
        self.vae = vae
        self.num_steps = num_steps
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def compute_etta(self, z, z_minus, z_plus, dt):
        """
        Compute acceleration (etta) in latent space using VJP of decoder.
        
        This computes: etta_i = -J^T @ (g(z+) - 2*g(z) + g(z-)) / dt
        
        where J is the Jacobian of the decoder at z, and the finite difference
        approximates the second derivative of the decoded path in image space.

        Args:
            z: Current latent point [1, D] or [1, C, H, W]
            z_minus: Previous latent point
            z_plus: Next latent point
            dt: Time step (1/(T-1) where T is num_steps)

        Returns:
            etta: Acceleration vector in latent space, same shape as z
        """
        # Compute decoded images
        g_minus = self.vae.decode(z_minus)
        g = self.vae.decode(z)
        g_plus = self.vae.decode(z_plus)

        # Finite difference approximation of second derivative in image space
        # This represents the "acceleration" of the path in image space
        finite_diff = (g_plus - 2 * g + g_minus) / dt

        # Compute VJP (Vector-Jacobian Product): J^T @ finite_diff
        # This projects the image-space acceleration back to latent space
        vjp_result = torch.autograd.functional.vjp(
            self.vae.decode, z, finite_diff
        )
        # vjp_result[0] is the forward pass output (g(z))
        # vjp_result[1] is J^T @ finite_diff
        etta = -vjp_result[1]

        # Clean up intermediate tensors
        del g_minus, g, g_plus, finite_diff, vjp_result
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return etta

    def compute_energy(self, z_collection, dt):
        """
        Compute total path energy (sum of squared etta norms).
        
        Energy E = Σ ||etta_i||² measures the total "curvature" of the path.
        A geodesic minimizes this energy.

        Args:
            z_collection: List of latent points [z_0, z_1, ..., z_{T-1}]
            dt: Time step

        Returns:
            Total energy (float)
        """
        energy = 0.0
        for j in range(1, len(z_collection) - 1):
            etta_j = self.compute_etta(
                z_collection[j],
                z_collection[j - 1],
                z_collection[j + 1],
                dt
            )
            energy += etta_j.norm().pow(2).item()
            del etta_j
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            
        return energy

    def geodesic_path_algorithm(self, z_collection):
        """
        Optimize path to minimize energy (find geodesic).
        
        Uses gradient descent on the path energy. At each iteration,
        each intermediate point z_i is updated by moving in the direction
        that reduces the local curvature.

        Args:
            z_collection: List of latent points (initial linear interpolation)

        Returns:
            Optimized z_collection (geodesic path)
        """
        T = len(z_collection)
        dt = 1.0 / (T - 1) if T > 1 else 1.0

        for iteration in range(self.max_iterations):
            # Compute current energy
            energy = self.compute_energy(z_collection, dt)
            
            # Check convergence
            if energy < self.epsilon:
                break

            # Update each intermediate point (not endpoints)
            for i in range(1, T - 1):
                etta_i = self.compute_etta(
                    z_collection[i],
                    z_collection[i - 1],
                    z_collection[i + 1],
                    dt
                )
                # Gradient descent step: move in negative gradient direction
                z_collection[i] = z_collection[i] - self.alpha * etta_i
                del etta_i
                
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        return z_collection

    def get_paths(self, inputs, labels=None):
        """
        Generate geodesic path in latent space and decode to pixel space.

        Process:
        1. Encode input and baseline to latent space
        2. Initialize path with linear interpolation in latent space
        3. Optimize path using geodesic algorithm (energy minimization)
        4. Decode optimized latent path to pixel space

        Args:
            inputs: Input images [B, C, H, W]
            labels: Target labels [B] (not used, for API compatibility)

        Returns:
            paths: Tensor [B, num_steps, C, H, W]
        """
        batch_size = inputs.shape[0]
        all_paths = []

        for b in range(batch_size):
            x_input = inputs[b:b+1]
            x_baseline = self.get_baselines(x_input)

            # Encode to latent space
            z_input = self.vae.encode(x_input)
            z_baseline = self.vae.encode(x_baseline)

            # Initialize with linear interpolation in latent space
            z_collection = []
            for i in range(self.num_steps):
                t = i / (self.num_steps - 1) if self.num_steps > 1 else 1.0
                z = z_baseline + t * (z_input - z_baseline)
                z_collection.append(z.clone())

            # Optimize to find geodesic path
            z_collection = self.geodesic_path_algorithm(z_collection)

            # Decode to pixel space
            paths = []
            for z in z_collection:
                x = self.vae.decode(z).squeeze(0)
                paths.append(x)

            paths = torch.stack(paths, dim=0)  # [num_steps, C, H, W]
            all_paths.append(paths)

        return torch.stack(all_paths, dim=0)  # [B, num_steps, C, H, W]


class BlurPathGenerator(PathGenerator):
    """
    Generate blur paths for Blur Integrated Gradients.
    Path: fully blurred (max_sigma) → original (sigma=0)

    Reference: https://arxiv.org/abs/2004.03383
    
    IMPORTANT: Gaussian blur must be applied in raw image space (0-1), not 
    normalized space. If inputs are normalized, provide undo_preprocess_fn to:
    1. Convert normalized → raw (0-1) before blur
    2. Apply Gaussian blur in raw space
    3. Convert raw → normalized after blur
    """

    def __init__(
        self,
        preprocess_fn,
        device,
        num_steps,
        max_sigma=50,
        sqrt=False,
        undo_preprocess_fn=None,
    ):
        """
        Initialize Blur Path Generator.

        Args:
            preprocess_fn: Preprocessing function (raw → normalized)
            device: Device to run on
            num_steps: Number of blur steps
            max_sigma: Maximum Gaussian blur kernel sigma
            sqrt: Use sqrt spacing for sigma values
            undo_preprocess_fn: Function to undo preprocessing (normalized → raw).
                                If provided, blur is applied in raw space.
        """
        super().__init__(baseline_method=None, preprocess_fn=preprocess_fn, device=device)
        self.num_steps = num_steps
        self.max_sigma = max_sigma
        self.sqrt = sqrt
        self.undo_preprocess_fn = undo_preprocess_fn

    def gaussian_blur(self, image, sigma):
        """
        Apply Gaussian blur to image tensor.
        If undo_preprocess_fn is set, blur is applied in raw image space.
        """
        import torchvision.transforms.functional as TF

        if sigma == 0:
            return image.clone()

        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(kernel_size, 3)

        if self.undo_preprocess_fn is not None:
            # Convert to raw space, blur, convert back
            raw_image = self.undo_preprocess_fn(image.unsqueeze(0)).squeeze(0)
            blurred_raw = TF.gaussian_blur(raw_image.unsqueeze(0), [kernel_size, kernel_size], sigma).squeeze(0)
            return self.preprocess_fn(blurred_raw.unsqueeze(0)).squeeze(0)
        else:
            return TF.gaussian_blur(image.unsqueeze(0), [kernel_size, kernel_size], sigma).squeeze(0)

    def get_paths(self, inputs, labels=None):
        """
        Generate blur path from fully blurred to original.

        Args:
            inputs: Input images [B, C, H, W]
            labels: Target labels [B] (not used, for API compatibility)

        Returns:
            paths: Tensor [B, num_steps+1, C, H, W]
        """
        import math

        batch_size = inputs.shape[0]
        all_paths = []

        # Calculate sigmas (0 → max_sigma)
        if self.sqrt:
            sigmas = [math.sqrt(float(i) * self.max_sigma / float(self.num_steps))
                      for i in range(self.num_steps + 1)]
        else:
            sigmas = [float(i) * self.max_sigma / float(self.num_steps)
                      for i in range(self.num_steps + 1)]

        for b in range(batch_size):
            x_input = inputs[b]
            paths = []

            # Path: max_sigma → 0 (blurred → original)
            # Reverse order: start from most blurred (max_sigma), end at original (sigma=0)
            for i in range(self.num_steps + 1):
                sigma = sigmas[self.num_steps - i]  # Reverse order
                blurred = self.gaussian_blur(x_input, sigma)
                paths.append(blurred)

            paths = torch.stack(paths, dim=0)  # [num_steps+1, C, H, W]
            all_paths.append(paths)

        return torch.stack(all_paths, dim=0)  # [B, num_steps+1, C, H, W]


class SpectralPathGenerator(PathGenerator):
    _dct_cache = {}

    def __init__(
        self,
        baseline_method,
        preprocess_fn,
        device,
        num_steps,
        overlap=0.5,
        spectral_mode='svd',
        channel_mode='per_channel',
        gating_schedule='linear',
        gating_sigmoid_k=12.0,
        wavelet_levels=4,
        laplacian_levels=4,
    ):
        super().__init__(baseline_method, preprocess_fn, device)
        self.num_steps = num_steps
        self.overlap = float(max(0.0, min(1.0, overlap)))
        self.spectral_mode = spectral_mode
        self.channel_mode = channel_mode
        self.gating_schedule = gating_schedule
        self.gating_sigmoid_k = float(gating_sigmoid_k)
        self.wavelet_levels = int(wavelet_levels)
        self.laplacian_levels = int(laplacian_levels)

    def get_paths(self, inputs, labels=None):
        batch_size = inputs.shape[0]
        all_paths = []

        for b in range(batch_size):
            x_input = inputs[b]
            x_baseline = self.get_baselines(inputs[b:b+1]).squeeze(0)
            diff = x_input - x_baseline
            state = self._build_state(diff)

            paths = []
            for step in range(self.num_steps):
                alpha = 1.0 if self.num_steps <= 1 else step / (self.num_steps - 1)
                diff_scaled = self._reconstruct_from_state(state, alpha)
                x_step = x_baseline + diff_scaled
                paths.append(x_step)

            paths = torch.stack(paths, dim=0)
            all_paths.append(paths)

        return torch.stack(all_paths, dim=0)

    def _build_state(self, diff):
        if self.spectral_mode == 'svd':
            if self.channel_mode == 'joint':
                return self._build_joint_svd_state(diff)
            if self.channel_mode != 'per_channel':
                raise ValueError(f"Unsupported channel_mode for SVD: {self.channel_mode}")
            return self._build_per_channel_svd_state(diff)

        if self.spectral_mode == 'dct':
            return self._build_dct_state(diff)
        if self.spectral_mode == 'wavelet':
            return self._build_wavelet_state(diff)
        if self.spectral_mode == 'laplacian':
            return self._build_laplacian_state(diff)

        raise ValueError(f"Unsupported spectral_mode: {self.spectral_mode}")

    def _reconstruct_from_state(self, state, alpha):
        if state['kind'] == 'svd_per_channel':
            scale = self._component_scale(state['strength'], alpha)
            s_scaled = state['s'] * scale
            return torch.matmul(state['U'] * s_scaled.unsqueeze(1), state['Vh'])

        if state['kind'] == 'svd_joint':
            scale = self._component_scale(state['strength'], alpha)
            s_scaled = state['s'] * scale
            diff_2d = torch.matmul(state['U'] * s_scaled.unsqueeze(0), state['Vh'])
            return diff_2d.reshape(state['shape'])

        if state['kind'] == 'dct':
            scale = self._component_scale(state['strength'], alpha).unsqueeze(0)
            return self._idct2(state['coeffs'] * scale)

        if state['kind'] == 'component_stack':
            scale = self._component_scale(state['strength'], alpha).view(-1, 1, 1, 1)
            return (state['components'] * scale).sum(dim=0)

        raise ValueError(f"Unsupported reconstruction state: {state['kind']}")

    def _build_per_channel_svd_state(self, diff):
        U, s, Vh = torch.linalg.svd(diff, full_matrices=False)
        s_max = s[:, :1].clamp_min(1e-10)
        strength = (s / s_max).clamp(0.0, 1.0)
        return {
            'kind': 'svd_per_channel',
            'U': U,
            's': s,
            'Vh': Vh,
            'strength': strength,
        }

    def _build_joint_svd_state(self, diff):
        C, H, W = diff.shape
        diff_2d = diff.reshape(C * H, W)
        U, s, Vh = torch.linalg.svd(diff_2d, full_matrices=False)
        s_max = s[:1].clamp_min(1e-10)
        strength = (s / s_max).clamp(0.0, 1.0)
        return {
            'kind': 'svd_joint',
            'U': U,
            's': s,
            'Vh': Vh,
            'strength': strength,
            'shape': diff.shape,
        }

    def _build_dct_state(self, diff):
        coeffs = self._dct2(diff)
        _, H, W = diff.shape

        y = torch.linspace(0.0, 1.0, steps=H, device=diff.device, dtype=diff.dtype)
        x = torch.linspace(0.0, 1.0, steps=W, device=diff.device, dtype=diff.dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        radius = torch.sqrt(xx.square() + yy.square())
        radius = radius / radius.max().clamp_min(1e-10)
        strength = (1.0 - radius).clamp(0.0, 1.0)

        return {
            'kind': 'dct',
            'coeffs': coeffs,
            'strength': strength,
        }

    @classmethod
    def _get_dct_matrix(cls, size, device, dtype):
        key = (size, str(device), str(dtype))
        cached = cls._dct_cache.get(key)
        if cached is not None:
            return cached

        n = torch.arange(size, device=device, dtype=dtype).unsqueeze(0)
        k = torch.arange(size, device=device, dtype=dtype).unsqueeze(1)
        matrix = torch.cos(math.pi / size * (n + 0.5) * k)
        matrix[0] *= math.sqrt(1.0 / size)
        if size > 1:
            matrix[1:] *= math.sqrt(2.0 / size)
        cls._dct_cache[key] = matrix
        return matrix

    def _dct2(self, x):
        _, H, W = x.shape
        dct_h = self._get_dct_matrix(H, x.device, x.dtype)
        dct_w = self._get_dct_matrix(W, x.device, x.dtype)
        out = torch.matmul(dct_h, x)
        return torch.matmul(out, dct_w.transpose(0, 1))

    def _idct2(self, coeffs):
        _, H, W = coeffs.shape
        dct_h = self._get_dct_matrix(H, coeffs.device, coeffs.dtype)
        dct_w = self._get_dct_matrix(W, coeffs.device, coeffs.dtype)
        out = torch.matmul(dct_h.transpose(0, 1), coeffs)
        return torch.matmul(out, dct_w)

    def _build_laplacian_state(self, diff):
        current = diff
        residuals = []

        levels = 0
        while levels < self.laplacian_levels and min(current.shape[-2:]) >= 2:
            low = F.avg_pool2d(current.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)
            up = F.interpolate(
                low.unsqueeze(0),
                size=current.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            residuals.append(current - up)
            current = low
            levels += 1

        zero_residuals = [torch.zeros_like(residual) for residual in residuals]
        components = [self._laplacian_reconstruct(current, zero_residuals)]
        for residual_idx in range(len(residuals)):
            selected_residuals = []
            for current_idx, residual in enumerate(residuals):
                if current_idx == residual_idx:
                    selected_residuals.append(residual)
                else:
                    selected_residuals.append(torch.zeros_like(residual))
            components.append(
                self._laplacian_reconstruct(torch.zeros_like(current), selected_residuals)
            )

        stacked = torch.stack(components, dim=0)
        strength = torch.linspace(
            1.0,
            0.0,
            steps=stacked.shape[0],
            device=diff.device,
            dtype=diff.dtype,
        )
        return {
            'kind': 'component_stack',
            'components': stacked,
            'strength': strength,
        }

    def _laplacian_reconstruct(self, low_res, residuals):
        current = low_res
        for residual in reversed(residuals):
            current = F.interpolate(
                current.unsqueeze(0),
                size=residual.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            current = current + residual
        return current

    def _build_wavelet_state(self, diff):
        ll = diff
        details = []

        for _ in range(self.wavelet_levels):
            if min(ll.shape[-2:]) < 2 or ll.shape[-2] % 2 != 0 or ll.shape[-1] % 2 != 0:
                break
            ll, detail = self._haar_forward(ll)
            details.append(detail)

        zero_details = [
            tuple(torch.zeros_like(band) for band in detail)
            for detail in details
        ]

        components = [self._haar_reconstruct(ll, zero_details)]
        for level_idx, detail in enumerate(details):
            for band_idx in range(3):
                selected_details = []
                for current_idx, current_detail in enumerate(details):
                    if current_idx == level_idx:
                        selected_detail = tuple(
                            band if idx == band_idx else torch.zeros_like(band)
                            for idx, band in enumerate(current_detail)
                        )
                    else:
                        selected_detail = tuple(torch.zeros_like(band) for band in current_detail)
                    selected_details.append(selected_detail)
                components.append(
                    self._haar_reconstruct(torch.zeros_like(ll), selected_details)
                )

        stacked = torch.stack(components, dim=0)
        strength = torch.linspace(
            1.0,
            0.0,
            steps=stacked.shape[0],
            device=diff.device,
            dtype=diff.dtype,
        )
        return {
            'kind': 'component_stack',
            'components': stacked,
            'strength': strength,
        }

    def _haar_forward(self, x):
        x00 = x[:, 0::2, 0::2]
        x01 = x[:, 0::2, 1::2]
        x10 = x[:, 1::2, 0::2]
        x11 = x[:, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) / 2.0
        lh = (x00 - x01 + x10 - x11) / 2.0
        hl = (x00 + x01 - x10 - x11) / 2.0
        hh = (x00 - x01 - x10 + x11) / 2.0
        return ll, (lh, hl, hh)

    def _haar_inverse(self, ll, lh, hl, hh):
        C, H, W = ll.shape
        out = torch.zeros((C, H * 2, W * 2), device=ll.device, dtype=ll.dtype)
        out[:, 0::2, 0::2] = (ll + lh + hl + hh) / 2.0
        out[:, 0::2, 1::2] = (ll - lh + hl - hh) / 2.0
        out[:, 1::2, 0::2] = (ll + lh - hl - hh) / 2.0
        out[:, 1::2, 1::2] = (ll - lh - hl + hh) / 2.0
        return out

    def _haar_reconstruct(self, ll, details):
        current = ll
        for detail in reversed(details):
            current = self._haar_inverse(current, *detail)
        return current

    def _component_scale(self, strength, alpha):
        alpha_tensor = torch.as_tensor(alpha, device=strength.device, dtype=strength.dtype)
        width = max(self.overlap, 0.0)
        start_alpha = (1.0 - width) * (1.0 - strength)

        if width <= 1e-8:
            return (alpha_tensor >= start_alpha).to(strength.dtype)

        progress = ((alpha_tensor - start_alpha) / width).clamp(0.0, 1.0)

        if self.gating_schedule == 'linear':
            return progress
        if self.gating_schedule == 'cosine':
            return 0.5 - 0.5 * torch.cos(progress * math.pi)
        if self.gating_schedule == 'sigmoid':
            gain = self.gating_sigmoid_k
            raw = torch.sigmoid(gain * (progress - 0.5))
            low = torch.sigmoid(torch.tensor(-0.5 * gain, device=strength.device, dtype=strength.dtype))
            high = torch.sigmoid(torch.tensor(0.5 * gain, device=strength.device, dtype=strength.dtype))
            return ((raw - low) / (high - low + 1e-10)).clamp(0.0, 1.0)
        if self.gating_schedule == 'step':
            return (progress >= 0.5).to(strength.dtype)

        raise ValueError(f"Unsupported gating_schedule: {self.gating_schedule}")
    

