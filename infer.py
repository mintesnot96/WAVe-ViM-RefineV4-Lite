# (project root)/infer.py
import os, argparse, glob
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from src.utils.config import load_config
from src.utils.clip_utils import load_open_clip
from src.utils.train_utils import load_ckpt
from src.models.restorer import Restorer

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def _letterbox_with_info(img: Image.Image, size):
    """Replicate-pad to `size`, return canvas AND geometry to undo later."""
    th, tw = int(size[0]), int(size[1])
    w, h = img.size
    s = min(tw / w, th / h)
    nw, nh = max(1, int(round(w*s))), max(1, int(round(h*s)))

    imr = img.resize((nw, nh), Image.BICUBIC)
    arr = np.array(imr, dtype=np.uint8)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    offx, offy = (tw - nw)//2, (th - nh)//2
    canvas[offy:offy+nh, offx:offx+nw] = arr
    if offy > 0:  canvas[:offy, offx:offx+nw] = arr[0:1].repeat(offy, axis=0)
    if th-(offy+nh) > 0: canvas[offy+nh:, offx:offx+nw] = arr[-1:].repeat(th-(offy+nh), axis=0)
    if offx > 0:  canvas[:, :offx] = canvas[:, offx:offx+1]
    if tw-(offx+nw) > 0: canvas[:, offx+nw:] = canvas[:, offx+nw-1:offx+nw]

    info = dict(
        orig_w=w, orig_h=h,
        tw=tw, th=th,
        offx=offx, offy=offy, nw=nw, nh=nh
    )
    return Image.fromarray(canvas), info

def load_image_and_info(path, device, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        canvas, info = _letterbox_with_info(img, size=size)
        x = to_tensor(canvas).unsqueeze(0).to(device)
        return x, info
    else:
        x = to_tensor(img).unsqueeze(0).to(device)
        info = dict(orig_w=img.size[0], orig_h=img.size[1],
                    tw=img.size[0], th=img.size[1], offx=0, offy=0, nw=img.size[0], nh=img.size[1])
        return x, info

def uncrop_to_original(y_hat: torch.Tensor, info: dict):
    """
    y_hat: (B,3,th,tw) prediction on the letterboxed canvas.
    Returns (B,3,orig_h,orig_w) cropped to content and resized back.
    """
    B, C, H, W = y_hat.shape
    offx, offy, nw, nh = info["offx"], info["offy"], info["nw"], info["nh"]
    crop = y_hat[:, :, offy:offy+nh, offx:offx+nw]
    # back to original resolution
    out = F.interpolate(crop, size=(info["orig_h"], info["orig_w"]),
                        mode="bicubic", align_corners=False)
    return out

def list_images(root):
    if os.path.isdir(root):
        files = [p for p in glob.glob(os.path.join(root, "**", "*.*"), recursive=True)
                 if os.path.splitext(p)[1].lower() in IMG_EXT]
        files.sort()
        return files
    return [root]

def rel_out_path(in_path, in_root, out_root):
    rel = os.path.relpath(in_path, start=in_root) if os.path.isdir(in_root) else os.path.basename(in_path)
    outp = os.path.join(out_root, rel)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    return outp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="./configs/default.yaml")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="./outputs/infer")
    ap.add_argument("--size", type=int, nargs=2, default=None, help="H W for letterbox; crop back to original before saving")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build CLIP + memory + Restorer
    clip_cfg = cfg["model"]["clip"]
    clip_model, clip_preproc, clip_tok = load_open_clip(
        model_name=clip_cfg["model_name"],
        pretrained=clip_cfg["pretrained"],
        device=device
    )
    mem_names = clip_cfg["mem_degradations"]
    if clip_model is None:
        M = torch.randn(len(mem_names), 512, device=device)
        d_txt = 512
    else:
        tokens = clip_tok([f"a photo with {n} degradation" for n in mem_names]).to(device)
        with torch.no_grad():
            M = clip_model.encode_text(tokens)
            M = M / M.norm(dim=-1, keepdim=True)
        d_txt = M.shape[1]

    model = Restorer(cfg, clip_img_dim=512, clip_txt_dim=d_txt,
                     mem_txt_embed=M, mem_names=mem_names).to(device)
    load_ckpt(args.ckpt, model, map_location=device)
    model.eval()

    in_root = args.input
    files = list_images(in_root)
    os.makedirs(args.outdir, exist_ok=True)

    with torch.no_grad():
        for p in files:
            x, info = load_image_and_info(p, device, size=args.size)

            # CLIP image embedding
            if clip_model is None:
                e_img = torch.randn(x.size(0), 512, device=device)
            else:
                try:
                    sz = getattr(clip_model.visual, "image_size", 224)
                    if isinstance(sz, (list, tuple)):
                        clip_h, clip_w = int(sz[0]), int(sz[1])
                    else:
                        clip_h = clip_w = int(sz)
 # ensure pair
                except Exception:
                    clip_h = clip_w = 224
                x_clip = F.interpolate(x, size=(clip_h, clip_w), mode="bicubic", align_corners=False) \
                         if x.shape[-2:] != (clip_h, clip_w) else x
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)[None,:,None,None]
                std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)[None,:,None,None]
                x_clip = (x_clip - mean) / std
                e_img = clip_model.encode_image(x_clip.to(dtype=next(clip_model.parameters()).dtype))
                e_img = e_img / e_img.norm(dim=-1, keepdim=True)

            e_txt = torch.zeros((x.size(0), d_txt), device=device)

            y_hat, alpha = model(x, e_img, e_txt)         # (B,3,th,tw)
            
            
            # uncertainty-aware fallback → LDO refine (helps OOD/composites)
            def _alpha_uncertainty(a, multi_label=True):
                if multi_label:
                    # 4*p*(1-p) in [0,1], averaged across codes
                    return (4*a*(1-a)).mean(dim=1)
                p = a / (a.sum(dim=1, keepdim=True) + 1e-8)
                H = -(p * (p+1e-8).log()).sum(dim=1)
                return H / np.log(a.shape[1])

            u = _alpha_uncertainty(alpha, multi_label=cfg["model"].get("dsm_multi_label", True)).mean().item()
            if u > 0.60:   # tweak 0.55–0.70 if you like
                y_hat = model.tta_refine(x, y_hat)

            
            
            
            
            y_crop = uncrop_to_original(y_hat, info)      # (B,3,orig_h,orig_w)

            out_path = rel_out_path(p, in_root, args.outdir)
            out = to_pil_image(y_crop[0].clamp(0,1).cpu())
            out.save(out_path)

            # optional: print top-3 per image
            top = torch.argsort(alpha[0], descending=True)[:3].tolist()
            print(os.path.basename(p), "→", os.path.relpath(out_path, args.outdir),
                  "| top:", [mem_names[i] for i in top])

if __name__ == "__main__":
    main()
