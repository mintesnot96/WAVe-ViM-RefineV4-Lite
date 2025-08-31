# old/infer_cli.py
import os, sys, argparse, torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

THIS = os.path.dirname(__file__)
sys.path.insert(0, THIS)  # prefer old/src

from src.utils.config import load_config          # OLD codebase
from src.infer_utils import load_restorer         # OLD codebase

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_restorer(cfg, ckpt_path=args.ckpt, device=device)
    model.eval()

    im = Image.open(args.input).convert("RGB")
    x = to_tensor(im).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model.infer(x, text_prompt=None)

    out = to_pil_image(y.clamp(0,1)[0].cpu())
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.save(args.output)

if __name__ == "__main__":
    main()
