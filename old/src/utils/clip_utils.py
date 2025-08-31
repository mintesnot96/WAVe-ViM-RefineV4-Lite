# src/utils/clip_utils.py
# (project root)/src/utils/clip_utils.py
import torch.nn.functional as F
import torch
import torch.nn as nn


def load_open_clip(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"):
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(model_name)
        return model, preprocess, tokenizer
    except Exception as e:
        print("[WARN] open_clip not available, CLIP features will be stubbed.", e)
        return None, None, None

class IdentityImageEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((16,16))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*16*3, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        return self.proj(self.pool(x))
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# Bellow is added for V2
def encode_image_openclip(imgs, model):
    """
    imgs: BCHW in [0,1]; returns L2-normalized embeddings (B, 512) for ViT-B/32.
    If model is None, falls back to IdentityImageEncoder(512).
    """
    if model is None:
        enc = IdentityImageEncoder(out_dim=512).to(imgs.device)
        with torch.no_grad():
            return enc(imgs)

    # resolve CLIP native size (e.g., 224)
    try:
        h, w = model.visual.image_size
        if isinstance(h, (list, tuple)):
            h, w = h[0], w[1]
    except Exception:
        h = w = 224

    if imgs.shape[-2:] != (h, w):
        imgs = F.interpolate(imgs, size=(h, w), mode='bicubic', align_corners=False)

    mean = torch.tensor(_CLIP_MEAN, device=imgs.device)[None, :, None, None]
    std  = torch.tensor(_CLIP_STD,  device=imgs.device)[None, :, None, None]
    x = (imgs - mean) / std

    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        e = model.encode_image(x.to(dtype=dtype))
        e = e / e.norm(dim=-1, keepdim=True)
    return e


def encode_image_openclip(imgs, model):
    """
    imgs: BCHW in [0,1]; returns L2-normalized embeddings (B, d_img).
    Falls back to IdentityImageEncoder(512) if model is None.
    """
    if model is None:
        enc = IdentityImageEncoder(out_dim=512).to(imgs.device)
        with torch.no_grad():
            return enc(imgs)

    # Resolve CLIP-native size safely (int or (H,W))
    size = getattr(model.visual, "image_size", 224)
    if isinstance(size, (tuple, list)):
        if len(size) >= 2:
            h, w = int(size[0]), int(size[1])
        else:
            h = w = int(size[0])
    else:
        h = w = int(size)

    # Resize if needed
    if imgs.shape[-2:] != (h, w):
        imgs = F.interpolate(imgs, size=(h, w), mode='bicubic', align_corners=False)

    # CLIP normalization
    mean = torch.tensor(_CLIP_MEAN, device=imgs.device)[None, :, None, None]
    std  = torch.tensor(_CLIP_STD,  device=imgs.device)[None, :, None, None]
    x = (imgs - mean) / std

    # Match model dtype safely
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        dtype = x.dtype

    with torch.no_grad():
        e = model.encode_image(x.to(dtype=dtype))
        e = e / e.norm(dim=-1, keepdim=True)
    return e

