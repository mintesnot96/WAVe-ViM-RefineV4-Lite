# tools/evaluate_cdd.py
import os, argparse, glob, csv, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from collections import defaultdict
from src.models.restorer import Restorer
from src.utils.train_utils import load_ckpt
from src.utils.config import load_config
from src.utils.clip_utils import load_open_clip, encode_image_openclip
from src.utils.metrics import psnr, ssim

# --- same letterbox + mask as training ---
import numpy as np
def _letterbox_replicate_pil(img, size):
    img = img.convert("RGB")
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
    mask = np.zeros((th, tw), dtype=np.uint8)
    mask[offy:offy+nh, offx:offx+nw] = 1
    return Image.fromarray(canvas), mask

def _load_pair(lq_path, gt_path, size):
    lq = Image.open(lq_path).convert("RGB")
    gt = Image.open(gt_path).convert("RGB")
    lq_pad, m1 = _letterbox_replicate_pil(lq, size)
    gt_pad, m2 = _letterbox_replicate_pil(gt, size)
    x  = to_tensor(lq_pad).unsqueeze(0)            # 1x3xHxW
    y  = to_tensor(gt_pad).unsqueeze(0)
    m  = torch.from_numpy((np.minimum(m1, m2)[None, None, ...]).astype(np.float32)) # 1x1xHxW
    return x, y, m

def _masked_avg(a, m):  # a: BxCxHxW, m: Bx1xHxW
    C = a.shape[1]; mC = m.expand(-1, C, -1, -1)
    num = (a*mC).flatten(1).sum(-1); den = mC.flatten(1).sum(-1).clamp_min(1.0)
    return (num/den).mean()

def _build_restorer(cfg, ckpt, device):
    clip_cfg = cfg["model"]["clip"]
    clip_model, clip_preproc, clip_tok = load_open_clip(
        model_name=clip_cfg["model_name"], pretrained=clip_cfg["pretrained"], device=device
    )
    mem_names = clip_cfg["mem_degradations"]
    if clip_model is None:
        M = torch.randn(len(mem_names), 512, device=device); d_txt = 512
    else:
        toks = clip_tok([f"a photo with {n} degradation" for n in mem_names]).to(device)
        with torch.no_grad():
            M = clip_model.encode_text(toks); M = M / M.norm(dim=-1, keepdim=True)
        d_txt = M.shape[1]
    model = Restorer(cfg, clip_img_dim=512, clip_txt_dim=d_txt, mem_txt_embed=M, mem_names=mem_names).to(device)
    load_ckpt(ckpt, model, map_location=device)
    model.eval()
    return model, clip_model, mem_names, d_txt

def _encode_img_for_clip(clip_model, x):
    if clip_model is None:
        return torch.randn(x.size(0), 512, device=x.device)
    size = getattr(clip_model.visual, "image_size", 224)
    if isinstance(size, (tuple, list)): ch, cw = int(size[0]), int(size[1])
    else: ch = cw = int(size)
    xc = F.interpolate(x, size=(ch, cw), mode="bicubic", align_corners=False) if x.shape[-2:]!=(ch,cw) else x
    mean = torch.tensor([0.48145466,0.4578275,0.40821073], device=x.device)[None,:,None,None]
    std  = torch.tensor([0.26862954,0.26130258,0.27577711], device=x.device)[None,:,None,None]
    xc = (xc - mean)/std
    with torch.no_grad():
        e = clip_model.encode_image(xc.to(dtype=next(clip_model.parameters()).dtype))
        e = e / e.norm(dim=-1, keepdim=True)
    return e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--gt_clear_dir", required=True, help="path/to/root/clear (GT)")
    ap.add_argument("--deg_dir", required=True, help="path/to/root/<deg> OR path/to/root/mixed")
    ap.add_argument("--size", type=int, nargs=2, default=[256,256])
    ap.add_argument("--mode", choices=["run-model","pred-dir"], default="run-model")
    ap.add_argument("--pred_dir", default="", help="if mode=pred-dir, folder with model outputs matching deg_dir filenames")
    ap.add_argument("--tag", default="mixed_or_degname", help="tag to report (e.g., haze, fog, mixed)")
    ap.add_argument("--csv_out", default="./eval_results.csv")
    ap.add_argument("--json_out", default="./eval_summary.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    H,W = args.size

    # list of files under deg_dir matched with GT in clear_dir
    exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    lq_list = sorted([p for p in glob.glob(os.path.join(args.deg_dir,"**","*.*"), recursive=True) if p.lower().endswith(exts)])

    if args.mode == "run-model":
        model, clip_model, mem_names, d_txt = _build_restorer(cfg, args.ckpt, device)

    rows = []
    ps, ss = [], []
    with torch.no_grad():
        for lp in lq_list:
            name = os.path.basename(lp)
            gp = os.path.join(args.gt_clear_dir, name)
            if not os.path.exists(gp):  # skip if no GT
                continue

            if args.mode == "run-model":
                x, y, m = _load_pair(lp, gp, (H,W))
                x = x.to(device); y = y.to(device); m = m.to(device)
                
                e_img = _encode_img_for_clip(clip_model, x)
                # e_txt = torch.zeros((x.size(0), model.mem_txt_embed.shape[1]), device=device)
                
                d_txt = getattr(model, "clip_txt_dim",
                model.mem_txt_embed.shape[1] if getattr(model, "mem_txt_embed", None) is not None else 512)
                e_txt = torch.zeros((x.size(0), d_txt), device=device)

                y_hat, _ = model(x, e_img, e_txt)
            else:  # pred-dir
                # read saved prediction, pad GT to same size, compute mask
                pred_path = os.path.join(args.pred_dir, name)
                if not os.path.exists(pred_path): 
                    continue
                y_hat = to_tensor(Image.open(pred_path).convert("RGB")).unsqueeze(0).to(device)
                # letterbox GT to the same size as prediction
                Hh, Ww = y_hat.shape[-2:]
                y, m = None, None
                gt = Image.open(gp).convert("RGB")
                gt_pad, mgt = _letterbox_replicate_pil(gt, (Hh, Ww))
                y = to_tensor(gt_pad).unsqueeze(0).to(device)
                m = torch.from_numpy(mgt[None,None,...].astype(np.float32)).to(device)

            m3 = m.expand(-1,3,-1,-1)
            cur_ps = psnr((y_hat*m3).clamp(0,1), (y*m3).clamp(0,1))
            cur_ss = ssim((y_hat*m3).clamp(0,1), (y*m3).clamp(0,1))
            ps.append(cur_ps); ss.append(cur_ss)
            rows.append([name, args.tag, float(cur_ps), float(cur_ss)])

    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file","tag","psnr","ssim"]); w.writerows(rows)

    summary = {
        "tag": args.tag,
        "count": len(rows),
        "psnr_mean": float(np.mean(ps)) if ps else float("nan"),
        "ssim_mean": float(np.mean(ss)) if ss else float("nan"),
    }
    with open(args.json_out, "w") as f: json.dump(summary, f, indent=2)
    print("[SUMMARY]", summary)

if __name__ == "__main__":
    main()
