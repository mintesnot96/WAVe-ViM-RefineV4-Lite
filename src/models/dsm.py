# src/models/dsm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# optional: sparse multi-label routing if you have entmax installed
try:
    from entmax import entmax15
    _HAS_ENTMAX = True
except Exception:
    _HAS_ENTMAX = False


class DegradationSemanticModulator(nn.Module):
    """
    DSM:
      - image embedding e_img (from CLIP)
      - memory M of degradation text embeddings (K, d_txt)
      - mixture alpha(x) computed against M
      - FiLM heads per stage (constructed in Restorer)
      - supports multi-label routing (sigmoid/entmax) and EMA-updatable memory
    """
    def __init__(
        self,
        d_img: int,
        d_txt: int,
        mem_txt_embed: torch.Tensor,
        n_stages_channels: List[int],
        temperature: float = 0.07,
        multi_label: bool = True,
        train_memory: bool = False,
        ema_momentum: float = 0.98,
        use_entmax: bool = False,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.multi_label = bool(multi_label)
        self.ema_momentum = float(ema_momentum)
        self.use_entmax = bool(use_entmax and _HAS_ENTMAX)

        # memory M (K, d_txt)
        if train_memory:
            self.M = nn.Parameter(mem_txt_embed.detach().clone().float(), requires_grad=True)
        else:
            self.register_buffer("M", mem_txt_embed.detach().clone().float(), persistent=True)

        self.Q = nn.Linear(d_img, d_txt, bias=False)

        # FiLM parameter generators per stage (gamma,beta)
        K = mem_txt_embed.shape[0]
        self.films = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_img + d_txt + K, 2*c),
                nn.GELU(),
                nn.Linear(2*c, 2*c)
            ) for c in n_stages_channels
        ])

    def forward_alpha(self, e_img: torch.Tensor) -> torch.Tensor:
        """
        Returns alpha (B,K): multi-label (sigmoid/entmax) or softmax if single-label
        """
        q = self.Q(e_img)                                     # (B, d_txt)
        sims = torch.matmul(q, self.M.t()) / self.temperature # (B, K)
        if self.multi_label:
            if self.use_entmax:
                alpha = entmax15(sims, dim=-1)
            else:
                alpha = sims.sigmoid()
        else:
            alpha = sims.softmax(dim=-1)
        return alpha

    def film_params(self, stage_idx: int, e_img: torch.Tensor, e_txt: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        code = torch.cat([e_img, e_txt, alpha], dim=-1)       # (B, d_img + d_txt + K)
        gb = self.films[stage_idx](code)                      # (B, 2C)
        return gb

    @torch.no_grad()
    def explain(self, alpha: torch.Tensor, mem_names: List[str]):
        out = []
        for a in alpha:
            idx = torch.argsort(a, descending=True)
            top = [(mem_names[i], float(a[i])) for i in idx[:5]]
            out.append(top)
        return out

    # ---------- dynamic memory utilities ----------

    @torch.no_grad()
    def ema_update(self, indices: torch.Tensor, targets: torch.Tensor):
        """
        EMA-refine selected memory vectors.
        indices: (B,) long
        targets: (B, d_txt) L2-normalized
        """
        m = self.ema_momentum
        if isinstance(self.M, nn.Parameter):
            for i, t in zip(indices.tolist(), targets):
                self.M.data[i].mul_(m).add_(t*(1-m))
        else:
            for i, t in zip(indices.tolist(), targets):
                self.M[i].mul_(m).add_(t*(1-m))

    @torch.no_grad()
    def append_text_slots(self, new_txt: torch.Tensor):
        """
        Append extra text embeddings to M (optional; not required if you don't want temp prompts).
        new_txt: (K_new, d_txt) L2-normalized
        """
        new_txt = new_txt.to(dtype=self.M.dtype, device=self.M.device)
        Mnew = torch.cat([self.M, new_txt], dim=0)
        if isinstance(self.M, nn.Parameter):
            self.M = nn.Parameter(Mnew, requires_grad=True)
        else:
            self.register_buffer("M", Mnew, persistent=True)
