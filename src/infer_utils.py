# src/infer_utils.py

import types
from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F

from src.models.restorer import Restorer
from src.utils.clip_utils import load_open_clip, IdentityImageEncoder
from src.utils.train_utils import load_ckpt

# same normalization as train.py
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


# def _encode_images_for_clip_infer(openclip_model, imgs: torch.Tensor):
#     """
#     imgs: BCHW in [0,1] on correct device
#     returns: (B, d_img)
#     """
#     if openclip_model is None:
#         enc = IdentityImageEncoder(out_dim=512).to(imgs.device)
#         with torch.no_grad():
#             return enc(imgs)
#     mean = torch.tensor(_CLIP_MEAN, device=imgs.device)[None,:,None,None]
#     std  = torch.tensor(_CLIP_STD,  device=imgs.device)[None,:,None,None]
#     x = (imgs - mean) / std
#     with torch.no_grad():
#         e = openclip_model.encode_image(x)
#         e = e / e.norm(dim=-1, keepdim=True)
#     return e
from src.utils.clip_utils import encode_image_openclip

def _encode_images_for_clip_infer(openclip_model, imgs: torch.Tensor):
    return encode_image_openclip(imgs, openclip_model)



def _build_memory_text_embeddings(openclip_model, tokenizer, mem_names: List[str], device):
    """
    Returns text memory M: (K, d_txt)
    """
    if openclip_model is None:
        torch.manual_seed(0)
        return torch.randn(len(mem_names), 512, device=device)
    texts = [f"a photo with {name} degradation" for name in mem_names]
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        e = openclip_model.encode_text(tokens)
        e = e / e.norm(dim=-1, keepdim=True)
    return e


def _pad_to_multiple(x: torch.Tensor, mult: int) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
    """
    Zero-pad tensor x (B,C,H,W) so that H and W are multiples of `mult`.
    Returns padded tensor and pad tuple (left, right, top, bottom) for later cropping.
    """
    B, C, H, W = x.shape
    newH = ((H + mult - 1) // mult) * mult
    newW = ((W + mult - 1) // mult) * mult
    pad_h = newH - H
    pad_w = newW - W
    # F.pad takes (left, right, top, bottom)
    padded = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    return padded, (0, pad_w, 0, pad_h)


def _crop_from_pad(y: torch.Tensor, pad: Tuple[int,int,int,int], orig_hw: Tuple[int,int]) -> torch.Tensor:
    """
    Remove the padding and crop back to original H,W.
    """
    _, _, H, W = y.shape
    left, right, top, bottom = pad
    y = y[:, :, 0:H-bottom if bottom>0 else H, 0:W-right if right>0 else W]
    oh, ow = orig_hw
    return y[:, :, :oh, :ow]


def load_restorer(cfg, ckpt_path: Optional[str] = None, device: Optional[torch.device] = None):
    """
    Build a Restorer with CLIP+memory, load weights, and attach .infer(x, text_prompt=None).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) CLIP + tokenizer
    clip_cfg = cfg["model"]["clip"]
    clip_model, _clip_preproc, clip_tok = load_open_clip(
        model_name=clip_cfg["model_name"],
        pretrained=clip_cfg["pretrained"],
        device=device
    )
    mem_names = clip_cfg["mem_degradations"]

    # 2) memory bank
    M = _build_memory_text_embeddings(clip_model, clip_tok, mem_names, device=device)
    d_txt = int(M.shape[1]) if M.ndim == 2 else 512
    d_img = 512

    # 3) model
    model = Restorer(cfg, clip_img_dim=d_img, clip_txt_dim=d_txt, mem_txt_embed=M, mem_names=mem_names).to(device)

    # 4) weights
    if ckpt_path:
        try:
            load_ckpt(ckpt_path, model, map_location=device)
            print(f"[load_restorer] Loaded checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"[load_restorer] WARN: could not load ckpt ({ckpt_path}): {e}")

    model.eval()

    # 5) attach .infer
    win = int(cfg["model"]["window_size"])
    base_multiple = 8 * max(1, win)   # DWT (/2) + two downs (/4) -> /8; lowest stage needs mod window_size
    use_amp = (cfg.get("precision", "fp32") == "amp")

    def _infer(self, x: torch.Tensor, text_prompt: Optional[str] = None):
        """
        x: BCHW float in [0,1] on same device as model
        text_prompt: optional string; None => blind
        """
        with torch.no_grad():
            # 0) pad to multiple
            orig_hw = (x.shape[2], x.shape[3])
            x_pad, pad = _pad_to_multiple(x, base_multiple)

            # 1) embeddings
            e_img = _encode_images_for_clip_infer(clip_model, x_pad)
            if (text_prompt is not None) and (clip_model is not None) and text_prompt.strip():
                tokens = clip_tok([text_prompt]).to(x.device)
                e_txt = clip_model.encode_text(tokens)
                e_txt = e_txt / e_txt.norm(dim=-1, keepdim=True)
            else:
                e_txt = torch.zeros((x_pad.size(0), d_txt), device=x.device)

            # 2) forward
        #     if use_amp:
        #         with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
        #             y_pad, _ = self(x_pad, e_img, e_txt)
        #     else:
        #         y_pad, _ = self(x_pad, e_img, e_txt)

        #     # 3) unpad / crop back to original size
        #     y = _crop_from_pad(y_pad, pad, orig_hw)
        # return y
        
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    y_pad, _ = self(x_pad, e_img, e_txt)
            else:
                y_pad, _ = self(x_pad, e_img, e_txt)

            # 3) unpad / crop back to original size
            y = _crop_from_pad(y_pad, pad, orig_hw)
        return y


    model.infer = types.MethodType(_infer, model)
    return model
