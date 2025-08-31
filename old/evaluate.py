# evaluate.py
# (project root)/evaluate.py

import os, glob, argparse
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from src.utils.metrics import psnr, ssim

def load_image(path):
    return to_tensor(Image.open(path).convert("RGB")).unsqueeze(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--gt_dir", type=str, required=True)
    args = ap.parse_args()

    # preds = sorted(glob.glob(os.path.join(args.pred_dir, "*.*")))
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    preds = sorted(p for p in glob.glob(os.path.join(args.pred_dir, "*.*")) if p.lower().endswith(exts))

    ps, ss = [], []
    for p in preds:
        name = os.path.basename(p)
        gt_p = os.path.join(args.gt_dir, name)
        if not os.path.exists(gt_p): continue
        yp = load_image(p)
        yg = load_image(gt_p)
        ps.append(psnr(yp, yg))
        ss.append(ssim(yp, yg))
    print(f"PSNR: {sum(ps)/len(ps):.2f} | SSIM: {sum(ss)/len(ss):.3f} on {len(ps)} images")

if __name__ == "__main__":
    main()
