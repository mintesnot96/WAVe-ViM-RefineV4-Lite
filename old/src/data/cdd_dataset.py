# (root)\src\data\cdd_dataset.py
import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

def _letterbox_replicate_pil(img, size):
    img = img.convert("RGB")
    th, tw = int(size[0]), int(size[1])
    w, h = img.size
    s = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    imr = img.resize((nw, nh), Image.BICUBIC)
    arr = np.array(imr, dtype=np.uint8)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    offx, offy = (tw - nw) // 2, (th - nh) // 2
    canvas[offy:offy+nh, offx:offx+nw] = arr
    if offy > 0:  canvas[:offy, offx:offx+nw] = arr[0:1].repeat(offy, axis=0)
    if th-(offy+nh) > 0: canvas[offy+nh:, offx:offx+nw] = arr[-1:].repeat(th-(offy+nh), axis=0)
    if offx > 0:  canvas[:, :offx] = canvas[:, offx:offx+1]
    if tw-(offx+nw) > 0: canvas[:, offx+nw:] = canvas[:, offx+nw-1:offx+nw]
    mask = np.zeros((th, tw), dtype=np.uint8)
    mask[offy:offy+nh, offx:offx+nw] = 1
    return Image.fromarray(canvas), mask

# ---- ADD this helper (letterbox keep-aspect, no stretch) -------------------
def _to_fixed(img, size=(256, 256), fill=(0, 0, 0)):
    """
    Letterbox to target size without stretching.
    """
    img = img.convert("RGB")
    th, tw = int(size[0]), int(size[1])
    w, h = img.size
    s = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    img_resized = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (tw, th), fill)
    canvas.paste(img_resized, ((tw - nw) // 2, (th - nh) // 2))
    return canvas

class CDD11Dataset(Dataset):
    def __init__(self, root_dir, size=None, transform=None):
        """
        root_dir: path to dataset split (train or test) directory
                  Example: /.../CDD-11-30/CDD-11_train
                  Inside should be subfolders: clear, haze, fog, rain, etc.
        size: tuple (H, W) to resize images, or None to keep original size
        transform: optional torchvision transform
        """
        self.size = size or (256, 256)
        self.root_dir = root_dir
        self.gt_dir = os.path.join(root_dir, "clear")
        if not os.path.exists(self.gt_dir):
            raise FileNotFoundError(f"Ground truth folder not found: {self.gt_dir}")

        # Collect degradation folders (exclude "clear")
        self.degradations = sorted(
            [d for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d)) and d.lower() != "clear"]
        )
        self.deg_to_id = {name: idx for idx, name in enumerate(self.degradations)}

        # Collect all samples: (deg_image_path, gt_image_path, deg_name)
        self.samples = []
        for deg_name in self.degradations:
            deg_path = os.path.join(root_dir, deg_name)
            deg_images = sorted(glob.glob(os.path.join(deg_path, "*.*")))
            for img_path in deg_images:
                fname = os.path.basename(img_path)
                gt_path = os.path.join(self.gt_dir, fname)
                if os.path.exists(gt_path):
                    self.samples.append((img_path, gt_path, deg_name))
                else:
                    print(f"[WARN] No GT found for {img_path}")

        # Default transforms
        # if transform is None:
        #     if size:
        #         self.transform = T.Compose([
        #             T.Resize(size),
        #             T.ToTensor()
        #         ])
        #     else:
        #         self.transform = T.ToTensor()
        # else:
        #     self.transform = transform
        
        # Default transforms (LETTERBOX keep-aspect → no stretching)
        if transform is None:
            if size:
                self.transform = lambda im: T.ToTensor()(_to_fixed(im, size=size, fill=(0,0,0)))
            else:
                self.transform = T.ToTensor()
        else:
            self.transform = transform


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        deg_path, gt_path, deg_name = self.samples[index]
        d_img = Image.open(deg_path).convert("RGB")
        g_img = Image.open(gt_path).convert("RGB")
        d_canvas, d_mask = _letterbox_replicate_pil(d_img, size=self.size)
        g_canvas, g_mask = _letterbox_replicate_pil(g_img,  size=self.size)

        m = (np.minimum(d_mask, g_mask).astype(np.float32))[None, ...]  # 1xHxW
        d_arr = np.asarray(d_canvas).astype(np.float32)/255.0
        g_arr = np.asarray(g_canvas).astype(np.float32)/255.0

        # negative sample
        neg_index = index
        while neg_index == index:
            neg_index = random.randint(0, len(self.samples)-1)
        n_path, _, _ = self.samples[neg_index]
        n_img = Image.open(n_path).convert("RGB")
        n_canvas, _ = _letterbox_replicate_pil(n_img, size=self.size)
        n_arr = np.asarray(n_canvas).astype(np.float32)/255.0

        meta = {"deg_name": deg_name, "deg_path": deg_path, "gt_path": gt_path}
        return (
            torch.from_numpy(d_arr).permute(2,0,1).float(),
            torch.from_numpy(g_arr).permute(2,0,1).float(),
            torch.from_numpy(n_arr).permute(2,0,1).float(),
            torch.from_numpy(m).float(),
            meta
        )
