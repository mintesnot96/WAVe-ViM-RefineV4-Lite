# src/models/dsm.py
# (project root)/src/models/dsm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class DegradationSemanticModulator(nn.Module):
    """
    DSM:
      - frozen CLIP (image+text) encoders provided externally
      - memory of degradation texts -> embeddings M (K, dtxt)
      - mixture weights alpha(x) from image embedding vs memory
      - FiLM heads per stage
      - optional bottleneck cross-attn module instantiated in restorer
    """
    def __init__(self, d_img, d_txt, mem_txt_embed, n_stages_channels: List[int], temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.M = nn.Parameter(mem_txt_embed, requires_grad=False)  # (K, d_txt)
        self.Q = nn.Linear(d_img, d_txt, bias=False)
        # FiLM per stage
        self.films = nn.ModuleList([nn.Sequential(
            nn.Linear(d_img + d_txt + self.M.shape[0], 2*c),
            nn.GELU(),
            nn.Linear(2*c, 2*c)
        ) for c in n_stages_channels])

    def forward_alpha(self, e_img):
        # alpha = softmax( (Q e_img) M^T / tau )
        q = self.Q(e_img)                                # (B, d_txt)
        sims = torch.matmul(q, self.M.t()) / self.temperature  # (B, K)
        alpha = sims.softmax(dim=-1)
        return alpha

    def film_params(self, stage_idx, e_img, e_txt, alpha):
        code = torch.cat([e_img, e_txt, alpha], dim=-1)  # (B, d_img+d_txt+K)
        gb = self.films[stage_idx](code)                # (B, 2C)
        return gb

    @torch.no_grad()
    def explain(self, alpha, mem_names):
        # returns list of (topk name, weight) per sample
        out = []
        for a in alpha:
            idx = torch.argsort(a, descending=True)
            top = [(mem_names[i], float(a[i])) for i in idx[:5]]
            out.append(top)
        return out
