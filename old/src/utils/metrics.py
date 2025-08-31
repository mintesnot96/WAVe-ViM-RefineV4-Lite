# src/utils/metrics.py
# (project root)/src/utils/metrics.py

import torch
import torch.nn.functional as F
import numpy as np
from math import log10

def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return 100.0
    return 10.0 * log10((max_val ** 2) / mse)

def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    # simple SSIM (not multi-scale) for validation
    # channels-first BCHW in [0,1]
    mu_x = img1.mean(dim=(2,3), keepdim=True)
    mu_y = img2.mean(dim=(2,3), keepdim=True)
    sigma_x = ((img1 - mu_x)**2).mean(dim=(2,3), keepdim=True)
    sigma_y = ((img2 - mu_y)**2).mean(dim=(2,3), keepdim=True)
    sigma_xy = ((img1 - mu_x)*(img2 - mu_y)).mean(dim=(2,3), keepdim=True)
    ssim_map = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2))/((mu_x**2+mu_y**2 + C1)*(sigma_x+sigma_y + C2))
    return ssim_map.mean().item()
