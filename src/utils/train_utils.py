# src/utils/train_utils.py
# (project root)/src/utils/train_utils.py

import os
import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def build_optimizer(model, cfg):
    opt = AdamW(model.parameters(), lr=cfg["optim"]["lr"],
                betas=tuple(cfg["optim"]["betas"]),
                weight_decay=cfg["optim"]["weight_decay"])
    return opt

# def build_scheduler(optimizer, cfg):
#     warmup = cfg["optim"]["warmup_iters"]
#     total = cfg["optim"]["epochs"] * cfg["optim"]["iters_per_epoch"]
#     def lr_lambda(step):
#         if step < warmup:
#             return float(step) / float(max(1, warmup))
#         return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (total - warmup))))
#     return LambdaLR(optimizer, lr_lambda)


def build_scheduler(optimizer, cfg):
    warmup_cfg = int(cfg["optim"].get("warmup_iters", 0))
    total = int(cfg["optim"]["epochs"]) * int(cfg["optim"]["iters_per_epoch"])

    # Clamp warmup so it's always < total (or 0 if total is tiny)
    if total <= 1:
        warmup = 0
    else:
        warmup = min(max(0, warmup_cfg), total - 1)

    T = max(1, total - warmup)  # remaining steps for cosine

    # (Optional) one-time notice if we had to clamp
    if warmup_cfg != warmup:
        print(f"[Scheduler] warmup_iters clamped from {warmup_cfg} to {warmup} (total train steps = {total}).")

    def lr_lambda(step):
        # step is 0-based
        if warmup > 0 and step < warmup:
            return float(step) / float(max(1, warmup))
        # cosine on the remainder
        progress = max(0, step - warmup)
        progress = min(progress, T)  # don’t exceed the planned horizon
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(progress) / float(T))))

    return LambdaLR(optimizer, lr_lambda)
#####################


def save_ckpt(path, step, model, optimizer, scaler=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)

def load_ckpt(path, model, optimizer=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None and "opt" in ckpt:
        optimizer.load_state_dict(ckpt["opt"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("step", 0)
