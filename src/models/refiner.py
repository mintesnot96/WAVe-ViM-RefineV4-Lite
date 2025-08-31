# src/models/refiner.py
# (project root)/src/models/refiner.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyRefinerUNet(nn.Module):
    """
    Very small UNet operating on the low-frequency band only.
    Conditioning: concat bottleneck summary + normalized CLIP image embedding as extra channels (via 1x1).
    """
    def __init__(self, in_c=1, cond_dim=512, base_c=64):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, base_c)
        self.enc1 = nn.Sequential(nn.Conv2d(in_c+base_c, base_c, 3,1,1), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv2d(base_c, base_c*2, 3,2,1), nn.GELU())
        self.mid = nn.Sequential(nn.Conv2d(base_c*2, base_c*2, 3,1,1), nn.GELU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(base_c*2, base_c, 2,2,0), nn.GELU())
        self.dec1 = nn.Conv2d(base_c, in_c, 3,1,1)

    def forward(self, L, cond_vec):
        B, C, H, W = L.shape
        cond = self.cond_proj(cond_vec)[:, :, None, None].expand(B, -1, H, W)
        x = torch.cat([L, cond], dim=1)
        x = self.enc1(x)
        e2 = self.enc2(x)
        m  = self.mid(e2)
        d2 = self.dec2(m)
        out = self.dec1(d2)
        return out

class TinyDiffusionRefiner(nn.Module):
    """
    Deterministic 5-step refiner: L_{t-1} = L_t + UNet(L_t, cond)
    """
    def __init__(self, in_c=1, cond_dim=512, base_c=64, steps=5):
        super().__init__()
        self.net = TinyRefinerUNet(in_c=in_c, cond_dim=cond_dim, base_c=base_c)
        self.steps = steps

    def forward(self, L_init, cond_vec):
        L = L_init
        for _ in range(self.steps):
            delta = self.net(L, cond_vec)
            L = L + 0.2 * delta
        return L
from src.losses import blur_consistency

class LDO(nn.Module):
    """
    Local Dynamic Optimization:
    a tiny per-image refinement on the predicted image to reduce residual
    blur and artifacts. It optimizes y for a few steps using blur consistency
    and TV regularization. Safe and fast (no weight updates).
    """
    def __init__(self, steps=3, lr=0.25, tv_weight=0.01):
        super().__init__()
        self.steps = int(max(0, steps))
        self.lr = float(lr)
        self.tvw = float(tv_weight)

    def _tv(self, x):
        dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        return dx + dy

    @torch.no_grad()
    def forward(self, x_in, y_init):
        if self.steps <= 0:
            return y_init
        y = y_init.clone().detach().requires_grad_(True)
        opt = torch.optim.SGD([y], lr=self.lr, momentum=0.0)
        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            Lb = blur_consistency(y, x_in, length=15, angle=0.0)
            Ltv = self._tv(y) * self.tvw
            (Lb + Ltv).backward()
            opt.step()
            y.data.clamp_(0, 1)
        return y.detach()

