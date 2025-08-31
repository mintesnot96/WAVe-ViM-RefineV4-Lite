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
