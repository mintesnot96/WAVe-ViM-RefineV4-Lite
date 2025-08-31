# src/models/wavelet.py
# (project root)/src/models/wavelet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def _outer(a: torch.Tensor, b: torch.Tensor):
    # a, b: shape [2] each -> returns 2x2
    return torch.einsum('i,j->ij', a, b).contiguous()

class _HaarKernels(nn.Module):
    """
    Prepare orthonormal 2x2 Haar kernels for LL, LH, HL, HH (registered as buffers).
    Forward uses stride=2, padding=0 (no padding); inverse uses conv_transpose2d with stride=2.
    """
    def __init__(self):
        super().__init__()
        # 1D orthonormal Haar
        h0 = torch.tensor([2**-0.5,  2**-0.5], dtype=torch.float32)  # low-pass
        h1 = torch.tensor([2**-0.5, -2**-0.5], dtype=torch.float32)  # high-pass

        # 2D separable filters
        kLL = _outer(h0, h0)  # 2x2
        kLH = _outer(h0, h1)
        kHL = _outer(h1, h0)
        kHH = _outer(h1, h1)

        # register as buffers (shape 2x2 each)
        self.register_buffer('kLL', kLL)
        self.register_buffer('kLH', kLH)
        self.register_buffer('kHL', kHL)
        self.register_buffer('kHH', kHH)

    def make_weight(self, kernel_2x2: torch.Tensor, C: int):
        """
        Expand a single 2x2 kernel to depthwise conv weight shape [C,1,2,2]
        """
        return kernel_2x2.view(1,1,2,2).repeat(C,1,1,1)

class HaarDWT(nn.Module):
    """
    Orthonormal 2D Haar DWT:
      Input  : x  (B,C,H,W)  (H,W must be even)
      Output : (LL, LH, HL, HH) each (B,C,H/2,W/2)
    """
    def __init__(self):
        super().__init__()
        self.k = _HaarKernels()

    def _dwconv(self, x, k2d):
        C = x.shape[1]
        w = self.k.make_weight(k2d, C)
        # stride=2, padding=0 (no padding) -> exact downsample by 2
        return F.conv2d(x, w, bias=None, stride=2, padding=0, groups=C)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4 and x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0, \
            f"HaarDWT expects even H,W, got {tuple(x.shape)}"
        LL = self._dwconv(x, self.k.kLL)
        LH = self._dwconv(x, self.k.kLH)
        HL = self._dwconv(x, self.k.kHL)
        HH = self._dwconv(x, self.k.kHH)
        return LL, LH, HL, HH

class HaarIDWT(nn.Module):
    """
    Inverse of the above DWT using conv_transpose2d with matched kernels.
      Input  : (LL, LH, HL, HH) each (B,C,H/2,W/2)
      Output : y (B,C,H,W)
    """
    def __init__(self):
        super().__init__()
        self.k = _HaarKernels()

    def _dwdeconv(self, x, k2d):
        C = x.shape[1]
        # Use the same 2x2 filters for transpose conv (no flip needed with orthonormal Haar).
        wT = self.k.make_weight(k2d, C)
        return F.conv_transpose2d(x, wT, bias=None, stride=2, padding=0, groups=C)

    def forward(self, LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor):
        # Recompose by summation of transpose-conv results (perfect reconstruction)
        y = self._dwdeconv(LL, self.k.kLL)
        y = y + self._dwdeconv(LH, self.k.kLH)
        y = y + self._dwdeconv(HL, self.k.kHL)
        y = y + self._dwdeconv(HH, self.k.kHH)
        return y
