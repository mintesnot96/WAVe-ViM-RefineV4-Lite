# src/models/blocks.py
# (project root)/src/models/blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ----------------------------
# Small utility layers
# ----------------------------

class ConvBNAct(nn.Module):
    # 3x3 conv -> norm -> GELU
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p)
        # self.bn = nn.BatchNorm2d(c_out)
        self.bn = LayerNorm2d(c_out) 
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DepthwiseConv(nn.Module):
    def __init__(self, c, k=5, s=1, p=2):
        super().__init__()
        self.conv = nn.Conv2d(c, c, k, s, p, groups=c)
    def forward(self, x): return self.conv(x)

class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps
    def forward(self, x):
        # per-channel over HxW
        var = x.var(dim=(2,3), keepdim=True, unbiased=False)
        mean = x.mean(dim=(2,3), keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight[None,:,None,None]*x + self.bias[None,:,None,None]
        return x

# ----------------------------
# Pseudo-Mamba block (non-causal global-ish modeling)
# Idea: depthwise conv + GRU-style gating on channels.
# ----------------------------

class PseudoMambaBlock(nn.Module):
    """
    Channels: C -> C (keep size). Complexity: O(HW*C*k)
    - Depthwise conv captures local sequential correlations
    - Gating uses pointwise conv to mix channels (global-ish)
    - Residual and LayerNorm2d stabilize
    """
    def __init__(self, c, k=5):
        super().__init__()
        self.norm = LayerNorm2d(c)
        self.dw = DepthwiseConv(c, k=k, p=k//2)
        self.gate_in = nn.Conv2d(c, c, 1)
        self.gate_out = nn.Conv2d(c, c, 1)
        self.ffn = nn.Sequential(
            nn.Conv2d(c, 4*c, 1),
            nn.GELU(),
            nn.Conv2d(4*c, c, 1)
        )
    def forward(self, x):
        idn = x
        x = self.norm(x)
        z = self.dw(x)
        g = torch.sigmoid(self.gate_in(x))
        x = g * z + (1.0 - g) * x
        x = x + self.ffn(self.norm(x))
        return x + idn

# ----------------------------
# Windowed self-attention block (simplified)
# For low-res stages, window attention helps detail and dynamic spatial mixing.
# ----------------------------

class WindowAttention(nn.Module):
    """
    Simple window MHSA
    - Split feature map into non-overlapping windows of size W
    - Apply MHSA in each window independently
    """
    def __init__(self, c, heads=4, window_size=8):
        super().__init__()
        assert c % heads == 0
        self.c = c
        self.h = heads
        self.w = window_size
        self.qkv = nn.Conv2d(c, c*3, 1)
        self.proj = nn.Conv2d(c, c, 1)
        self.scale = (c // heads) ** -0.5
        self.norm = LayerNorm2d(c)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.w == 0 and W % self.w == 0, "H,W must be divisible by window_size"
        x = self.norm(x)
        qkv = self.qkv(x)                                    # B,3C,H,W
        q, k, v = torch.chunk(qkv, 3, dim=1)                 # each B,C,H,W

        heads = self.h
        head_dim = C // heads
        ws = self.w
        Nh, Nw = H // ws, W // ws  # number of windows along H and W

        # [B, C, H, W] -> [B, Nh, Nw, heads, ws*ws, head_dim]
        qw = rearrange(q, 'b (h d) (nh ws1) (nw ws2) -> b nh nw h (ws1 ws2) d',
                       h=heads, d=head_dim, nh=Nh, nw=Nw, ws1=ws, ws2=ws)
        kw = rearrange(k, 'b (h d) (nh ws1) (nw ws2) -> b nh nw h (ws1 ws2) d',
                       h=heads, d=head_dim, nh=Nh, nw=Nw, ws1=ws, ws2=ws)
        vw = rearrange(v, 'b (h d) (nh ws1) (nw ws2) -> b nh nw h (ws1 ws2) d',
                       h=heads, d=head_dim, nh=Nh, nw=Nw, ws1=ws, ws2=ws)

        # attention inside each window
        attn = (qw @ kw.transpose(-1, -2)) * self.scale      # B, Nh, Nw, h, T, T
        attn = attn.softmax(dim=-1)
        out = attn @ vw                                      # B, Nh, Nw, h, T, d

        # merge windows back to [B, C, H, W]
        out = rearrange(out, 'b nh nw h (ws1 ws2) d -> b (h d) (nh ws1) (nw ws2)',
                        nh=Nh, nw=Nw, ws1=ws, ws2=ws, h=heads, d=head_dim)
        out = self.proj(out)
        return out


class AttnBlock(nn.Module):
    def __init__(self, c, heads=4, window_size=8):
        super().__init__()
        self.attn = WindowAttention(c, heads, window_size)
        self.ffn = nn.Sequential(
            LayerNorm2d(c),
            nn.Conv2d(c, 4*c, 1),
            nn.GELU(),
            nn.Conv2d(4*c, c, 1),
        )
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x

# ----------------------------
# FiLM (DSM modulation)
# ----------------------------

class FiLM(nn.Module):
    """
    Produces per-channel gamma/beta from a small vector (DSM code).
    """
    def __init__(self, in_dim, c_out):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2*c_out),
            nn.GELU(),
            nn.Linear(2*c_out, 2*c_out)
        )
        self.c_out = c_out
    def forward(self, feat, code):
        B,C,H,W = feat.shape
        gb = self.mlp(code)                        # B, 2C
        gamma, beta = gb[:, :self.c_out], gb[:, self.c_out:]
        gamma = gamma[:, :, None, None]
        beta  = beta[:, :, None, None]
        return gamma * feat + beta

# ----------------------------
# Bottleneck cross-attention (one-shot, low-res)
# ----------------------------

class BottleneckCrossAttention(nn.Module):
    """
    Q: visual tokens; K,V: text tokens (here we tile a single embedding)
    """
    def __init__(self, c, d_text=512, heads=4):
        super().__init__()
        assert c % heads == 0
        self.h = heads
        self.dk = c // heads
        self.Wq = nn.Conv2d(c, c, 1)
        self.Wk = nn.Linear(d_text, c)
        self.Wv = nn.Linear(d_text, c)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x, e_text):
        B,C,H,W = x.shape
        q = self.Wq(x)                              # B,C,H,W
        q = rearrange(q, 'b (h d) h1 w1 -> b h (h1 w1) d', h=self.h, d=self.dk)
        # tile text into tokens
        k = self.Wk(e_text)                         # B,C
        v = self.Wv(e_text)                         # B,C
        k = rearrange(k, 'b (h d) -> b h 1 d', h=self.h, d=self.dk)
        v = rearrange(v, 'b (h d) -> b h 1 d', h=self.h, d=self.dk)
        attn = (q @ k.transpose(-1,-2)) / (self.dk ** 0.5)   # B,h,T,1
        attn = attn.softmax(dim=-2)                         # along tokens
        out = attn @ v                                      # B,h,T,d
        out = rearrange(out, 'b h t d -> b (h d) t')
        out = rearrange(out, 'b c (h1 w1) -> b c h1 w1', h1=H, w1=W)
        out = self.proj(out)
        return x + out

# ----------------------------
# Encoder/Decoder
# ----------------------------

class EncoderStage(nn.Module):
    """
    Each stage: N blocks (PseudoMamba or Attn), downsample at end (except last).
    """
    def __init__(self, c_in, c_out, depth, use_mamba=True, use_window_attn=False, heads=4, win=8, mamba_k=5, down=True):
        super().__init__()
        blocks = []
        for _ in range(depth):
            if use_window_attn:
                blocks += [AttnBlock(c_in, heads=heads, window_size=win)]
            elif use_mamba:
                blocks += [PseudoMambaBlock(c_in, k=mamba_k)]
            else:
                blocks += [ConvBNAct(c_in, c_in)]
        self.blocks = nn.Sequential(*blocks)
        self.down = down
        self.downsample = nn.Conv2d(c_in, c_out, 3, 2, 1) if down else nn.Conv2d(c_in, c_out, 1, 1, 0)

    def forward(self, x):
        x = self.blocks(x)
        return self.downsample(x), x

class DecoderStage(nn.Module):
    def __init__(self, c_in, c_skip, c_out, depth, use_mamba=True, use_window_attn=False, heads=4, win=8, mamba_k=5):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, 2, 2, 0)
        blocks = []
        c = c_out + c_skip
        for _ in range(depth):
            if use_window_attn:
                blocks += [AttnBlock(c, heads=heads, window_size=win)]
            elif use_mamba:
                blocks += [PseudoMambaBlock(c, k=mamba_k)]
            else:
                blocks += [ConvBNAct(c, c)]
        self.blocks = nn.Sequential(*blocks)
        self.fuse = nn.Conv2d(c, c_out, 1)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.blocks(x)
        return self.fuse(x)
