# src/losses.py
# (project root)/src/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Charbonnier ----
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    def forward(self, x, y):
        diff = torch.sqrt((x - y) ** 2 + self.eps ** 2)
        if self.reduction == 'mean': return diff.mean()
        if self.reduction == 'sum':  return diff.sum()
        return diff

# ---- Composite separation loss (max-margin from negative sample) ----
def composite_separation_loss(y_hat, y_neg, margin=0.1):
    return torch.clamp(margin - (y_hat - y_neg).abs().mean(), min=0.0)

# ---- Wavelet frequency-aware loss ----
def wavelet_frequency_loss(L_pred, H_pred, L_gt, H_gt, lambda_L=0.5, lambda_H=0.25):
    l = F.l1_loss(L_pred, L_gt)
    h = sum(F.l1_loss(hp, hg) for hp,hg in zip(H_pred, H_gt)) / len(H_pred)
    return lambda_L*l + lambda_H*h



# ---- Blur consistency loss (self-contained; no dependency on degradations.py) ----
import math

def _motion_psf(length=15, angle=0.0, device='cpu', dtype=torch.float32):
    """
    Tiny motion PSF built in pure PyTorch (no cv2).
    Starts as a 1-pixel-thick horizontal line centered in an LxL kernel,
    then rotates it by 'angle' degrees using affine_grid + grid_sample.
    The kernel is L1-normalized.
    """
    # force odd length >= 3
    L = int(max(3, round(length))) | 1
    k = torch.zeros((L, L), device=device, dtype=dtype)
    k[L // 2, :] = 1.0  # horizontal line through center

    if (angle % 360) != 0:
        theta = torch.tensor([
            [ math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0.0],
            [ math.sin(math.radians(angle)),  math.cos(math.radians(angle)), 0.0],
        ], device=device, dtype=dtype)[None, ...]  # (1,2,3)
        grid = F.affine_grid(theta, size=(1, 1, L, L), align_corners=False)
        k = F.grid_sample(
            k[None, None], grid,
            mode='bilinear', padding_mode='zeros', align_corners=False
        )[0, 0]

    k = k / (k.sum() + 1e-8)
    return k

def motion_blur_kernel(length=15, angle=0.0, device='cpu'):
    # thin wrapper returning a Torch tensor kernel
    return _motion_psf(length, angle, device=device, dtype=torch.float32)

def blur_consistency(y_hat, x_in, length=15, angle=0.0):
    """
    Data-fidelity term for deblurring:
    encourages x ≈ k * ŷ via depthwise convolution with a simple motion PSF.
    """
    B, C, H, W = y_hat.shape
    k = motion_blur_kernel(length, angle, device=y_hat.device)     # (L, L)
    k = k[None, None].repeat(C, 1, 1, 1)                           # (C,1,L,L)
    y_blur = F.conv2d(y_hat, k, padding=k.shape[-1] // 2, groups=C)
    return F.l1_loss(y_blur, x_in)





# ---- CLIP feature loss ----
def clip_feature_loss(fy_hat, fy_gt):
    return F.mse_loss(fy_hat, fy_gt)

# ---- Uncertainty-weighted multi-task ----
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
    def forward(self, losses):
        # losses: list of scalars
        total = 0.0
        for i, L in enumerate(losses):
            total += torch.exp(-self.log_vars[i]) * L + self.log_vars[i]
        return total
