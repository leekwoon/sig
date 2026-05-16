"""
Reference:
    Li et al., "Path Choice Matters for Clear Attributions in Path Methods"
    Official code: https://github.com/ZhengWenSEC2023/SAMP-for-Path-Method
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
import numpy as np


def _batch_l1_dist(input1, input2=None):
    bs = input1.size(0)
    if input2 is None:
        diff = torch.abs(input1).view(bs, -1)
    else:
        diff = torch.abs(input1 - input2).view(bs, -1)
    return torch.sum(diff, dim=-1)


def _gkern(ch, klen, ksig):
    inp = np.zeros((klen, klen))
    inp[klen // 2, klen // 2] = 1
    k = gaussian_filter(inp, ksig)
    kern = np.zeros((ch, ch, klen, klen))
    for i in range(ch):
        kern[i, i] = k
    return torch.from_numpy(kern.astype("float32"))


def _gaussian_blur(img, klen, ksig):
    device = img.device
    ch = img.size(1)
    kern = _gkern(ch, klen, ksig).to(device)
    img_out = F.conv2d(img, kern, padding=klen // 2)
    return img_out


class SAMPExplainer:
    def __init__(
        self,
        model,
        device='cuda',
        exp_obj='prob',
        preprocess_fn=None,
        step=5,
        n_frag=5,
        klen=11,
        ksig=5,
        momen=None,
        line_types=None,
        reduction='sum',
        insertion_baseline='blur',
    ):
        """
        Args:
            model: Classifier model to explain
            device: Device to run on
            exp_obj: Objective function ('prob' or 'logit')
            preprocess_fn: Function to preprocess inputs for classifier
            step: Number of pixels to select per iteration (as fraction: step / (H*W))
            n_frag: Number of fragments to divide the path into (controls step bound)
            klen: Gaussian blur kernel length
            ksig: Gaussian blur kernel sigma
            momen: Momentum for gradient smoothing (None to disable)
            line_types: List of path types to use (['deletion', 'insertion'] by default)
            reduction: How to combine multiple paths ('sum' or 'norm')
            insertion_baseline: Baseline for insertion path ('blur' or 'black')
        """
        self.model = model
        self.device = device
        self.exp_obj = exp_obj
        self.step = step
        self.n_frag = n_frag
        self.klen = klen
        self.ksig = ksig
        self.momen = momen
        self.reduction = reduction
        self.insertion_baseline = insertion_baseline
        
        if line_types is None:
            self.line_types = ['deletion', 'insertion']
        else:
            self.line_types = line_types
        
        if preprocess_fn is not None:
            self.preprocess_fn = preprocess_fn
        else:
            self.preprocess_fn = lambda x: x

    def _get_gradients(self, x, labels):
        x = x.clone().detach().requires_grad_(True)
        output = self.model(x)
        
        if self.exp_obj == 'logit':
            scores = output[torch.arange(output.shape[0], device=x.device), labels]
        elif self.exp_obj == 'prob':
            probs = torch.softmax(output, dim=-1)
            scores = probs[torch.arange(output.shape[0], device=x.device), labels]
        else:
            raise ValueError(f'Invalid objective function: {self.exp_obj}')
        
        grads = torch.autograd.grad(scores.sum(), x)[0]
        return grads.detach()

    def _generate_ref_point(self, img, line_type):
        if line_type == 'deletion':
            return self.preprocess_fn(torch.zeros_like(img))
        elif line_type == 'insertion':
            if self.insertion_baseline == 'blur':
                return _gaussian_blur(img, klen=self.klen, ksig=self.ksig)
            else:
                return self.preprocess_fn(torch.zeros_like(img))
        else:
            raise ValueError(f'Invalid line type: {line_type}')

    def _generate_cam(self, img, labels, ref_point, line_type):
        is_inverse = (line_type == 'deletion')
        
        BS, C, H, W = img.shape
        step = self.step
        
        cur_point = img.clone() if is_inverse else ref_point.clone()
        end_point = ref_point if is_inverse else img.clone()
        total_l1_dist = _batch_l1_dist(cur_point, end_point)
        step_bound = total_l1_dist / self.n_frag
        cam = torch.zeros(BS, H, W, device=self.device)
        
        grads = None
        
        selected_mask = ~torch.isclose(cur_point, end_point)
        selected_mask = torch.any(selected_mask, dim=1)
        
        while torch.any(selected_mask):
            batch_mask = torch.any(selected_mask.view(BS, -1), dim=-1)
            N = int(torch.sum(batch_mask).item())
            selected_mask_sub = selected_mask[batch_mask]
            
            sub_cur_point = cur_point[batch_mask]
            actual_grads = self._get_gradients(sub_cur_point, labels[batch_mask])
            
            if (grads is not None) and (self.momen is not None):
                grads[batch_mask] = (self.momen * actual_grads + (1 - self.momen) * grads[batch_mask]).detach()
            else:
                if grads is None:
                    grads = torch.empty_like(cur_point)
                grads[batch_mask] = actual_grads.detach()
            sub_grads = grads[batch_mask]
            
            with torch.no_grad():
                delta_direc = end_point[batch_mask] - sub_cur_point.detach()
                projection = torch.sum(sub_grads * delta_direc, dim=1)
                
                q = step / (H * W)
                if is_inverse:
                    projection[~selected_mask_sub] = float('inf')
                    cur_quantile = torch.quantile(
                        projection.view(N, -1), q=q, dim=-1, 
                        interpolation='lower', keepdim=True
                    ) + 1e-6
                    update_mask = projection < cur_quantile[:, :, None]
                else:
                    projection[~selected_mask_sub] = float('-inf')
                    cur_quantile = torch.quantile(
                        projection.view(N, -1), q=1-q, dim=-1,
                        interpolation='higher', keepdim=True
                    ) - 1e-6
                    update_mask = projection > cur_quantile[:, :, None]
                
                update_mask = update_mask.unsqueeze(1).expand(-1, C, -1, -1)
                move_full_step = delta_direc * update_mask
                move_l1_dist = _batch_l1_dist(move_full_step)
                
                is_outrange = move_l1_dist > step_bound[batch_mask]
                adjust_l1_ratio = step_bound[batch_mask] / (move_l1_dist + 1e-8)
                move_full_step[is_outrange] *= adjust_l1_ratio[is_outrange, None, None, None]
                
                cur_point[batch_mask] = cur_point[batch_mask] + move_full_step
                cam[batch_mask] += torch.sum(move_full_step * sub_grads, dim=1)
                
                selected_mask = ~torch.isclose(cur_point, end_point)
                selected_mask = torch.any(selected_mask, dim=1)
        
        cam = -cam if is_inverse else cam
        return cam

    def _postproc_cam(self, cam):
        if self.reduction == 'sum':
            return cam
        elif self.reduction == 'norm':
            BS, H, W = cam.size()
            cam_flat = cam.view(BS, H * W)
            max_v = cam_flat.max(dim=-1, keepdim=True)[0]
            min_v = cam_flat.min(dim=-1, keepdim=True)[0]
            cam_flat = (cam_flat - min_v) / (max_v - min_v + 1e-8)
            return cam_flat.view(BS, H, W)
        else:
            raise ValueError(f'Invalid reduction type: {self.reduction}')

    def get_attributions(self, inputs, labels=None):
        """
        Args:
            inputs: Input images [B, C, H, W] (preprocessed)
            labels: Target labels [B]. If None, uses model's predictions.
        
        Returns:
            attributions: Attribution maps [B, C, H, W]
        """
        self.model.eval()
        
        BS, C, H, W = inputs.shape
        inputs = inputs.to(self.device)
        
        if labels is None:
            with torch.no_grad():
                output = self.model(inputs)
                labels = torch.argmax(output, dim=-1)
        else:
            labels = labels.to(self.device)
        
        cam_list = []
        for line_type in self.line_types:
            ref_point = self._generate_ref_point(inputs, line_type)
            cam = self._generate_cam(inputs, labels, ref_point, line_type)
            cam = self._postproc_cam(cam)
            cam_list.append(cam)
        
        combined_cam = torch.stack(cam_list).sum(dim=0)
        
        attributions = combined_cam.unsqueeze(1).expand(-1, C, -1, -1)
        
        return attributions
