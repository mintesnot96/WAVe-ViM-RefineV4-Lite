# src/models/restorer.py
# (project root)/src/models/restorer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from .blocks import ConvBNAct, EncoderStage, DecoderStage, FiLM, SpatialFiLM, BottleneckCrossAttention
from .wavelet import HaarDWT, HaarIDWT
from .refiner import TinyDiffusionRefiner, LDO
from .dsm import DegradationSemanticModulator

# src/models/restorer.py
import torch
import torch.nn as nn

class Restorer(nn.Module):
    def __init__(self, cfg,
                 clip_img_dim=512,
                 clip_txt_dim=512,
                 mem_txt_embed=None,   # (K, d_txt) or None
                 mem_names=None):
        super().__init__()  # <-- MUST be first

        # Introspectable attributes for eval/infer scripts
        self.clip_img_dim = int(clip_img_dim)
        self.clip_txt_dim = int(clip_txt_dim)
        self.mem_names = list(mem_names) if mem_names is not None else []

        # Make sure the attribute exists even if there’s no memory passed
        if mem_txt_embed is None:
            # empty (0, d_txt) buffer so shape[1] is valid and it moves with .to(device)
            self.register_buffer(
                "mem_txt_embed",
                torch.empty(0, self.clip_txt_dim, dtype=torch.float32)
            )
        else:
            if not torch.is_tensor(mem_txt_embed):
                mem_txt_embed = torch.tensor(mem_txt_embed)
            # detach/clone so it's not tied to any graph, and let .to(device) move it
            self.register_buffer(
                "mem_txt_embed",
                mem_txt_embed.detach().clone().float()
            )

        
        
        # super().__init__()
        mcfg = cfg["model"]
        C0 = mcfg["base_channels"]
        self.mem_names = mem_names or []
        self.use_refiner = mcfg["use_tiny_refiner"]
        self.tiny_steps = mcfg["tiny_refiner_steps"]

        # DWT/IDWT
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()

        # Shallow stem on Low-Frequency band (treat LF as 1-channel proxy)
        # NOTE: We still process RGB L band by channel-wise approach: pack back to 3 channels with 1x1 conv.
        self.lf_proj = nn.Conv2d(3, C0, 3,1,1)

        # Encoder: 3 stages
        depths = mcfg["depths"]
        use_mamba = mcfg["use_mamba_blocks"]
        use_attn  = mcfg["use_window_attn"]
        heads = mcfg["window_attn_heads"]
        win   = mcfg["window_size"]
        mk    = mcfg["mamba_kernel_size"]

        self.enc1 = EncoderStage(C0, C0*2, depth=depths[0], use_mamba=use_mamba, use_window_attn=False, down=True, mamba_k=mk)
        self.enc2 = EncoderStage(C0*2, C0*4, depth=depths[1], use_mamba=use_mamba, use_window_attn=False, down=True, mamba_k=mk)
        # lowest stage: allow attn for better global mixing at small spatial size
        self.enc3 = EncoderStage(C0*4, C0*8, depth=depths[2], use_mamba=not use_attn, use_window_attn=use_attn, heads=heads, win=win, down=False, mamba_k=mk)

        # DSM
        self.dsm = DegradationSemanticModulator(
            d_img=clip_img_dim, d_txt=clip_txt_dim, mem_txt_embed=self.mem_txt_embed,
            n_stages_channels=[C0, C0*2, C0*4],
            temperature=cfg["model"]["clip"]["temperature"],
            multi_label=bool(cfg["model"].get("dsm_multi_label", True)),
            train_memory=bool(cfg["model"]["clip"].get("train_memory", False)),
            ema_momentum=float(cfg["model"]["clip"].get("ema_momentum", 0.98)),
            use_entmax=bool(cfg["model"].get("dsm_use_entmax", False)),
        )


        # FiLM adapters per stage
        Film = SpatialFiLM if bool(mcfg.get("film_spatial", True)) else FiLM
        self.film1 = Film(clip_img_dim + clip_txt_dim + len(self.mem_names), C0)
        self.film2 = Film(clip_img_dim + clip_txt_dim + len(self.mem_names), C0*2)
        self.film3 = Film(clip_img_dim + clip_txt_dim + len(self.mem_names), C0*4)


        self.use_bottleneck_attn = mcfg["bottleneck_cross_attn"]
        if self.use_bottleneck_attn:
            self.bca = BottleneckCrossAttention(C0*8, d_text=clip_txt_dim, heads=4)

        # Decoder
        # enc2 skip has C0*2 (pre-downsample), not C0*4
        self.dec2 = DecoderStage(C0*8, C0*2, C0*4, depth=2, use_mamba=True, use_window_attn=False, mamba_k=mk)
        # enc1 skip has C0 (pre-downsample), not C0*2
        self.dec1 = DecoderStage(C0*4, C0,   C0*2, depth=2, use_mamba=True, use_window_attn=False, mamba_k=mk)

        self.head = nn.Sequential(
            nn.Conv2d(C0*2, C0, 3,1,1),
            nn.GELU(),
            nn.Conv2d(C0, 3, 1,1,0)
        )

        # High-frequency light path
        self.hf_path = nn.Sequential(
            nn.Conv2d(3*3, C0, 3,1,1),
            nn.GELU(),
            nn.Conv2d(C0, C0, 3,1,1)
        )
        self.hf_cfe = Film(clip_img_dim + clip_txt_dim + len(self.mem_names), C0)  # CFE on HF
        self.hf_fuse = nn.Conv2d(C0 + C0*2, C0*2, 1)


        # Tiny Refiner on LF band
        if self.use_refiner:
            self.refiner = TinyDiffusionRefiner(in_c=3, cond_dim=clip_img_dim, base_c=mcfg["tiny_refiner_channels"], steps=self.tiny_steps)
            self.ldo = LDO(steps=int(mcfg.get("ldo_steps", 0)), lr=float(mcfg.get("ldo_lr", 0.25)))


    def forward(self, x, e_img, e_txt):
        # 1) Wavelet split
        L, Hh, Hv, Hd = self.dwt(x)              # each BCHW, same C(=3)
        # project LF to features
        z = self.lf_proj(L)

        # 2) DSM mixture and FiLM codes
        alpha = self.dsm.forward_alpha(e_img)    # (B,K)
        # dynamic text vector from memory when caller passes zeros
        use_caller_txt = (e_txt is not None) and (e_txt.abs().sum() > 0)
        e_txt_dyn = torch.matmul(alpha, self.dsm.M)     # (B, d_txt)
        e_txt = e_txt if use_caller_txt else e_txt_dyn
        code = lambda idx: torch.cat([e_img, e_txt, alpha], dim=-1)

        # 3) Encoder
        z = self.film1(z, code(0))               # FiLM stage 1
        z2, skip1 = self.enc1(z)                 # down1

        z2 = self.film2(z2, code(1))             # FiLM stage 2
        z3, skip2 = self.enc2(z2)                # down2

        z3 = self.film3(z3, code(2))             # FiLM stage 3
        z4, bottleneck = self.enc3(z3)           # no further down; bottleneck features

        if self.use_bottleneck_attn:
            z4 = self.bca(z4, e_txt)             # single cross-attn at bottleneck

        # 4) Decoder + HF fusion
        d2 = self.dec2(z4, skip2)                # up1
        d1 = self.dec1(d2, skip1)                # up2

        hf = torch.cat([Hh, Hv, Hd], dim=1)   # (B, 9, H/2, W/2)
        hf = self.hf_path(hf)
        hf = self.hf_cfe(hf, code(2))            # spatially-aware HF enhancement
        d1 = self.hf_fuse(torch.cat([d1, hf], dim=1))


        y0 = self.head(d1)                       # pre-refine RGB at LF resolution: matches L size
        # reconstruct to full-res using IDWT with HF bands
        # we treat y0 as corrected LF bands; combine with HF from light path:
        y = self.idwt(y0, Hh, Hv, Hd)

        # 5) Tiny refiner on LF only (optional)
        if self.use_refiner:
            # refine LF of current y with CLIP image embedding as conditioning
            L_hat, *_ = self.dwt(y)
            L_ref = self.refiner(L_hat, e_img)
            y = self.idwt(L_ref, Hh, Hv, Hd)

        return y, alpha
    @torch.no_grad()
    def tta_refine(self, x_in, y_pred):
        try:
            return self.ldo(x_in, y_pred)
        except Exception:
            return y_pred



# ---- add below to the end of src/models/restorer.py -------------------------
import types
import torch
from typing import List, Optional
from src.utils.clip_utils import load_open_clip, IdentityImageEncoder
from src.utils.train_utils import load_ckpt

# same normalization used in train.py
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# (project root)/src/models/restorer.py

import torch.nn.functional as F

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)



from src.utils.clip_utils import encode_image_openclip

def _encode_images_for_clip_infer(openclip_model, imgs: torch.Tensor):
    return encode_image_openclip(imgs, openclip_model)

# --- NEW: size helpers and lcm ---
def _pad_to_multiple(x: torch.Tensor, mult: int):
    """Replicate-pad BCHW tensor to H,W that are multiples of `mult`."""
    import torch.nn.functional as F
    H, W = x.shape[-2:]
    newH = ((H + mult - 1) // mult) * mult
    newW = ((W + mult - 1) // mult) * mult
    pad_h = newH - H
    pad_w = newW - W
    # F.pad uses (left, right, top, bottom)
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    return x_pad, (0, pad_w, 0, pad_h), (H, W)

def _crop_from_pad(y: torch.Tensor, pad, orig_hw):
    """Crop BCHW back to original H,W after replicate padding."""
    left, right, top, bottom = pad
    H, W = y.shape[-2:]
    if right or bottom:
        y = y[..., 0:H-bottom if bottom>0 else H, 0:W-right if right>0 else W]
    oh, ow = orig_hw
    return y[..., :oh, :ow]

def _lcm(a: int, b: int) -> int:
    import math
    return abs(a*b) // math.gcd(a, b) if a and b else max(a, b)
# 
def _build_memory_text_embeddings(openclip_model, tokenizer, mem_names: List[str], device):
    """
    Returns text memory M: (K, d_txt)
    """
    if openclip_model is None:
        # fallback random (kept fixed to avoid training here)
        torch.manual_seed(0)
        return torch.randn(len(mem_names), 512, device=device)
    texts = [f"a photo with {name} degradation" for name in mem_names]
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        e = openclip_model.encode_text(tokens)
        e = e / e.norm(dim=-1, keepdim=True)
    return e

def load_restorer(cfg, ckpt_path: Optional[str] = None, device: Optional[torch.device] = None):
    """
    Build a Restorer with CLIP+memory, load weights, and attach a .infer(x, text_prompt=None) method.
    Usage in your Gradio cell:
        model = load_restorer(cfg, ckpt_path=CKPT, device=torch.device("cuda"))
        y = model.infer(x, text_prompt="low light + rain")   # x is BCHW [0,1]
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load (frozen) CLIP
    clip_cfg = cfg["model"]["clip"]
    clip_model, _clip_preproc, clip_tok = load_open_clip(
        model_name=clip_cfg["model_name"],
        pretrained=clip_cfg["pretrained"],
        device=device
    )
    mem_names = clip_cfg["mem_degradations"]

    # 2) Build text memory embeddings M
    M = _build_memory_text_embeddings(clip_model, clip_tok, mem_names, device=device)  # (K, d_txt)
    d_txt = M.shape[1] if M.dim() == 2 else 512
    d_img = 512  # open_clip image projection dim we use in train/infer

    # --- NEW: choose a safe spatial multiple for padding ---
    # DWT halves once (×1/2) and you downsample twice more (×1/4) → total /8.
    # If windowed attention uses window_size w, also align to that window.
    win = int(cfg["model"].get("window_size", 1) or 1)
    base_down = 8
    base_multiple = _lcm(base_down, win)  # e.g., lcm(8, 8)=8; lcm(8,12)=24
    use_amp = (cfg.get("precision", "fp32") == "amp")


    # 3) Instantiate model
    model = Restorer(cfg, clip_img_dim=d_img, clip_txt_dim=d_txt, mem_txt_embed=M, mem_names=mem_names).to(device)
    
    # 4) Load checkpoint (if provided)
    if ckpt_path is not None and len(str(ckpt_path)) > 0:
        try:
            load_ckpt(ckpt_path, model, map_location=device)
            print(f"[load_restorer] Loaded checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"[load_restorer] WARN: could not load ckpt ({ckpt_path}): {e}")

    model.eval()

    # # 5) Attach a convenience .infer(x, text_prompt=None) method
    # def _infer(self, x: torch.Tensor, text_prompt: Optional[str] = None):
    #     """
    #     x: BCHW, float in [0,1] on the same device as model
    #     text_prompt: optional string; when None -> zero text embeddings (blind)
    #     returns: y (BCHW, [0,1] float)
    #     """
    #     with torch.no_grad():
    #         # image embedding (always used)
    #         e_img = _encode_images_for_clip_infer(clip_model, x)

    #         # text embedding (optional)
    #         if (text_prompt is not None) and (clip_model is not None) and (len(text_prompt.strip()) > 0):
    #             tokens = clip_tok([text_prompt]).to(x.device)
    #             e_txt = clip_model.encode_text(tokens)
    #             e_txt = e_txt / e_txt.norm(dim=-1, keepdim=True)
    #         else:
    #             e_txt = torch.zeros((x.size(0), d_txt), device=x.device)

    #         y, _alpha = self(x, e_img, e_txt)
    #     return y
    def _infer(self, x: torch.Tensor, text_prompt: Optional[str] = None):
        """
        x: BCHW float in [0,1] on same device as model. Returns y with SAME H×W as x.
        """
        with torch.no_grad():
            # 0) replicate-pad to model-friendly multiple (DWT/downsample/window)
            x_pad, pad, orig_hw = _pad_to_multiple(x, base_multiple)

            # 1) CLIP embeddings (compute on padded x to keep batch shapes consistent;
            #    CLIP preproc internally resizes anyway, so this does not change output size)
            e_img = _encode_images_for_clip_infer(clip_model, x_pad)

            if (text_prompt is not None) and (clip_model is not None) and text_prompt.strip():
                tokens = clip_tok([text_prompt]).to(x.device)
                e_txt = clip_model.encode_text(tokens)
                e_txt = e_txt / e_txt.norm(dim=-1, keepdim=True)
            else:
                e_txt = torch.zeros((x_pad.size(0), d_txt), device=x.device)

            # 2) forward pass (AMP if enabled)
            if use_amp and torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    y_pad, _ = self(x_pad, e_img, e_txt)
            else:
                y_pad, _ = self(x_pad, e_img, e_txt)

            # 3) crop back to original size
            y = _crop_from_pad(y_pad, pad, orig_hw)

        return y

    model.infer = types.MethodType(_infer, model)
    return model
# -----------------------------------------------------------------------------

