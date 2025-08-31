# src/data/dataset.py  (REPLACED: clean, no synthetic code)

import os
import re
import glob
import random
import difflib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
# ----------------- helpers: mask-aware letterbox (keep-aspect) --------------

def _letterbox_replicate_pil(img: Image.Image, size):
    """
    Keep-aspect letterbox using replicated borders (no black).
    Returns (PIL_RGB_canvas, mask_np_uint8[H,W]) where mask==1 on real content.
    """
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
    # replicate top/bottom
    if offy > 0:
        canvas[:offy, offx:offx+nw] = arr[0:1].repeat(offy, axis=0)
    if th - (offy+nh) > 0:
        canvas[offy+nh:, offx:offx+nw] = arr[-1:].repeat(th-(offy+nh), axis=0)
    # replicate left/right (including corners)
    if offx > 0:
        canvas[:, :offx] = canvas[:, offx:offx+1]
    if tw - (offx+nw) > 0:
        canvas[:, offx+nw:] = canvas[:, offx+nw-1:offx+nw]
    mask = np.zeros((th, tw), dtype=np.uint8)
    mask[offy:offy+nh, offx:offx+nw] = 1
    return Image.fromarray(canvas), mask

def _to_chw_float01(np_img_hwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np_img_hwc).permute(2, 0, 1).float()

def _load_fixed_with_mask(path: str, size) -> tuple[np.ndarray, np.ndarray]:
    im = Image.open(path).convert("RGB")
    canvas, mask = _letterbox_replicate_pil(im, size=size)
    arr = np.asarray(canvas).astype(np.float32) / 255.0  # HWC in [0,1]
    m = mask.astype(np.float32)[..., None]               # HxWx1 (float)
    return arr, m


# ----------------- helpers: safe letterbox (keep-aspect) --------------------

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def _norm_deg_tag(name: str) -> str:
    s = str(name).lower().replace(" ", "_").replace("-", "_")
    s = s.replace("__", "_")
    if "motion" in s and "blur" in s: s = "motion_blur"
    if s == "blur": s = "defocus_blur"
    return s

# ---------- robust stem normalization & pairing -----------------------------

_copy_tokens = re.compile(r"(?:\s*\(\d+\)|\s*copy\s*\d*|\s*copy|\s*-\s*copy\s*\d*)$", re.IGNORECASE)
_suffix_num  = re.compile(r"[_\- ]\d+$")
def _basic_stem(p): return os.path.splitext(os.path.basename(p))[0].lower()

def _strip_suffix_num(stem: str):
    # remove trailing _1, -2, ' 3', '(1)', 'copy' suffixes
    s = _copy_tokens.sub("", stem)
    s = _suffix_num.sub("", s)
    return s

def _best_match(stem_lq, gt_map: Dict[str, str], pairing_cfg: Dict[str, Any]):
    """Return GT path or None using cascade: exact -> strip_suffix_num -> prefix -> fuzzy."""
    order = pairing_cfg.get("strategy_order", ["exact", "strip_suffix_num", "prefix", "fuzzy"])
    fuzzy_thr = float(pairing_cfg.get("fuzzy_threshold", 0.86))

    s0  = stem_lq
    s1  = _strip_suffix_num(s0)

    if "exact" in order and s0 in gt_map: return gt_map[s0]
    if "exact" in order and s1 in gt_map: return gt_map[s1]

    if "prefix" in order:
        for gs, gp in gt_map.items():
            if s0.startswith(gs) or gs.startswith(s0) or s1.startswith(gs) or gs.startswith(s1):
                return gp

    if "fuzzy" in order:
        cands = list(gt_map.keys())
        m1 = difflib.get_close_matches(s0, cands, n=1, cutoff=fuzzy_thr)
        if m1: return gt_map[m1[0]]
        if s1 != s0:
            m2 = difflib.get_close_matches(s1, cands, n=1, cutoff=fuzzy_thr)
            if m2: return gt_map[m2[0]]

    return None

# ---------- PairedFolderDataset: <root>/<task>/gt, lq ----------------------

class PairedFolderDataset(Dataset):
    """
    Layout:
      root/
        <taskA>/
          gt/*.*   (clean)
          lq/*.*   (degraded)
        <taskB>/
          gt/*.*   (clean)
          lq/*.*   (degraded)

    Matches by filename stem with robust rules. `task` name becomes a degradation tag.
    """
    def __init__(self, paired_root, size, tasks=None, return_neg=True,
                 mix_strategy="shuffle", pairing_cfg=None):
        super().__init__()
        self.root = paired_root
        self.size = tuple(size)
        self.return_neg = return_neg
        self.pairing_cfg = pairing_cfg or {}

        # discover tasks if not given
        if not tasks:
            tasks = []
            for name in sorted(os.listdir(self.root)):
                tdir = os.path.join(self.root, name)
                if os.path.isdir(os.path.join(tdir, "gt")) and os.path.isdir(os.path.join(tdir, "lq")):
                    tasks.append(name)
        self.tasks = tasks

        self.items, gt_pool = [], []
        for task in self.tasks:
            gt_dir = os.path.join(self.root, task, "gt")
            lq_dir = os.path.join(self.root, task, "lq")
            if not (os.path.isdir(gt_dir) and os.path.isdir(lq_dir)):
                continue

            gt_map = {
                _basic_stem(p): p
                for p in glob.glob(os.path.join(gt_dir, "**", "*.*"), recursive=True)
                if p.lower().endswith(IMG_EXT)
            }

            for lp in glob.glob(os.path.join(lq_dir, "**", "*.*"), recursive=True):
                if not lp.lower().endswith(IMG_EXT): continue
                stem = _basic_stem(lp)
                gp = _best_match(stem, gt_map, self.pairing_cfg)
                if gp is None: 
                    continue
                self.items.append({"lq": lp, "gt": gp, "task": _norm_deg_tag(task)})
                gt_pool.append(gp)

        if mix_strategy == "shuffle":
            random.shuffle(self.items)
        self._gt_pool = gt_pool if len(gt_pool) > 0 else [it["gt"] for it in self.items]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        x, mx = _load_fixed_with_mask(it["lq"], self.size)
        y, my = _load_fixed_with_mask(it["gt"], self.size)
        m = np.minimum(mx, my)  # HxWx1 float (safe intersect)

        if self.return_neg and len(self._gt_pool) > 1:
            j = random.randrange(len(self._gt_pool))
            if self._gt_pool[j] == it["gt"]:
                j = (j + 1) % len(self._gt_pool)
            y_neg, _ = _load_fixed_with_mask(self._gt_pool[j], self.size)
        else:
            y_neg = y.copy()

        meta = {"degradations": [it["task"]], "path": it["lq"]}
        return (
            _to_chw_float01(x),
            _to_chw_float01(y),
            _to_chw_float01(y_neg),
            torch.from_numpy(m).permute(2,0,1).float(),  # 1xHxW
            meta
        )


# ---------- CDDLikeDataset: root/{clear, haze, ...} ------------------------

class CDDLikeDataset(Dataset):
    """
    Layout:
      root/
        clear/           # GT
        <degA>/
        <degB>/
        ...

    We pair each file in every LQ folder to clear/<stem>.* using robust rules.
    """
    def __init__(self, root, size, include_folders=None, return_neg=True, pairing_cfg=None):
        super().__init__()
        self.root = root
        self.size = tuple(size)
        self.return_neg = return_neg
        self.pairing_cfg = pairing_cfg or {}

        gt_dir = os.path.join(root, "clear")
        assert os.path.isdir(gt_dir), f"[CDDLikeDataset] missing GT folder: {gt_dir}"

        gt_map = {
            _basic_stem(p): p
            for p in glob.glob(os.path.join(gt_dir, "**", "*.*"), recursive=True)
            if p.lower().endswith(IMG_EXT)
        }
        self._gt_pool = list(gt_map.values())

        # discover LQ subfolders
        all_subs = [d for d in sorted(os.listdir(root))
                    if os.path.isdir(os.path.join(root, d)) and d != "clear"]

        use_subs = (include_folders or []) or all_subs

        self.items = []
        for deg in use_subs:
            lq_dir = os.path.join(root, deg)
            for lp in glob.glob(os.path.join(lq_dir, "**", "*.*"), recursive=True):
                if not lp.lower().endswith(IMG_EXT): continue
                stem = _basic_stem(lp)
                gp = _best_match(stem, gt_map, self.pairing_cfg)
                if gp is None:
                    continue
                self.items.append({"lq": lp, "gt": gp, "task": _norm_deg_tag(deg)})

        random.shuffle(self.items)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        x, mx = _load_fixed_with_mask(it["lq"], self.size)
        y, my = _load_fixed_with_mask(it["gt"], self.size)
        m = np.minimum(mx, my)

        if self.return_neg and len(self._gt_pool) > 1:
            j = random.randrange(len(self._gt_pool))
            if self._gt_pool[j] == it["gt"]:
                j = (j + 1) % len(self._gt_pool)
            y_neg, _ = _load_fixed_with_mask(self._gt_pool[j], self.size)
        else:
            y_neg = y.copy()

        meta = {"degradations": [it["task"]], "path": it["lq"]}
        return (
            _to_chw_float01(x),
            _to_chw_float01(y),
            _to_chw_float01(y_neg),
            torch.from_numpy(m).permute(2,0,1).float(),  # 1xHxW
            meta
        )


from .cdd_dataset import CDD11Dataset

# ---------- factory ---------------------------------------------------------

def _concat_if_many(datasets: List[Dataset]):
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)

def _build_single(cfg_block: Dict[str, Any], is_train: bool) -> Dataset:
    """
    cfg_block:
      dataset_type: "cdd_11" | "cdd_like" | "paired_generic"
      size: [H, W]
      For cdd_like:
        root_dir: "<path>"    OR paired_roots: ["<path1>", "<path2>", ...]
        include_folders: [...]
        pairing: {strategy_order: [...], fuzzy_threshold: 0.86}
      For paired_generic:
        paired_root: "<path>" OR paired_roots: [...]
        paired_tasks: [...]
        pairing: {...}
    """
    ds_type = (cfg_block.get("dataset_type") or "cdd_like").lower()
    size = tuple(cfg_block.get("size") or (256, 256))

    if ds_type == "cdd_11":
        # Single root_dir with clear + 11 degradation folders
        root_dir = cfg_block["root_dir"]
        return CDD11Dataset(root_dir=root_dir, size=size)

    elif ds_type == "cdd_like":
        roots = cfg_block.get("paired_roots") or [cfg_block.get("root_dir")]
        include_folders = cfg_block.get("include_folders", [])
        pairing_cfg = cfg_block.get("pairing", {})
        datasets = [
            CDDLikeDataset(root=r, size=size, include_folders=include_folders,
                           return_neg=is_train, pairing_cfg=pairing_cfg)
            for r in roots if r
        ]
        if not datasets:
            raise RuntimeError("[cdd_like] No valid roots in config.")
        return _concat_if_many(datasets)

    elif ds_type == "paired_generic":
        roots = cfg_block.get("paired_roots") or [cfg_block.get("paired_root")]
        tasks = cfg_block.get("paired_tasks", [])
        pairing_cfg = cfg_block.get("pairing", {})
        mix_strategy = "shuffle" if is_train else "sequential"
        datasets = [
            PairedFolderDataset(paired_root=r, size=size, tasks=tasks,
                                return_neg=is_train, mix_strategy=mix_strategy,
                                pairing_cfg=pairing_cfg)
            for r in roots if r
        ]
        if not datasets:
            raise RuntimeError("[paired_generic] No valid roots in config.")
        return _concat_if_many(datasets)

    else:
        raise ValueError(f"Unsupported dataset_type: {ds_type}")

def build_datasets(cfg: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
    """
    Reads cfg['train_data'] and cfg['val_data'] and returns (train_ds, val_ds).
    """
    train_ds = _build_single(cfg["train_data"], is_train=True)
    val_ds   = _build_single(cfg["val_data"],   is_train=False)
    return train_ds, val_ds
