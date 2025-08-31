# (project root)/train.py
import os, sys, math, time, random, json, shutil, argparse
import yaml, pprint
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.utils import save_image
from tqdm import tqdm
from collections import defaultdict
from src.utils.config import load_config
from src.utils.dist import init_distributed, is_primary, barrier
from src.utils.train_utils import build_optimizer, build_scheduler, save_ckpt, load_ckpt
from src.utils.metrics import psnr, ssim
from src.models.restorer import Restorer
from src.models.wavelet import HaarDWT
# with this:
from src.losses import (
    UncertaintyWeightedLoss,
    composite_separation_loss,
    blur_consistency,
    clip_feature_loss,
)
from torchvision.utils import save_image, make_grid  # add make_grid here

def masked_charbonnier(y_hat, y, m, eps=1e-3):
    # y_hat,y: BxCxHxW in [0,1]; m: Bx1xHxW
    C = y.shape[1]
    mC = m.expand(-1, C, -1, -1)                          # BxCxHxW
    diff = torch.sqrt((y_hat - y) ** 2 + eps ** 2) * mC    # BxCxHxW
    num = diff.flatten(1).sum(-1)                          # B
    den = mC.flatten(1).sum(-1).clamp_min(1.0)             # B
    return (num / den).mean()                              # scalar

def l1_mask(a, b, m):
    # a,b: BxCxHxW; m: Bx1xHxW
    C = a.shape[1]
    mC = m.expand(-1, C, -1, -1)
    num = (a - b).abs().mul(mC).flatten(1).sum(-1)         # B
    den = mC.flatten(1).sum(-1).clamp_min(1.0)             # B
    return (num / den).mean()                              # scalar



def down2_mask(m):
    # average-pool the binary mask for wavelet-sized maps
    return F.avg_pool2d(m, kernel_size=2, stride=2)




# -------------------------
# helpers
# -------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def collate_keep_meta(batch):
    # batch: list of tuples (x, y, y_neg, mask, meta)
    xs, ys, ynegs, masks, metas = zip(*batch)
    xs    = torch.stack(xs, dim=0)
    ys    = torch.stack(ys, dim=0)
    ynegs = torch.stack(ynegs, dim=0)
    masks = torch.stack(masks, dim=0)  # Bx1xHxW
    metas = list(metas)
    return xs, ys, ynegs, masks, metas


def build_memory_text_embeddings(openclip_model, tokenizer, mem_names, device):
    if openclip_model is None:
        return torch.randn(len(mem_names), 512, device=device)
    texts = [f"a photo with {name} degradation" for name in mem_names]
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        e = openclip_model.encode_text(tokens)
        e = e / e.norm(dim=-1, keepdim=True)
    return e  # (K, d_txt)


from src.utils.clip_utils import encode_image_openclip, load_open_clip

def encode_images_for_clip(openclip_model, preprocess, imgs):
    return encode_image_openclip(imgs, openclip_model)


# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/default.yaml")
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint to resume; if empty, start from scratch")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["optim"]["epochs"] = args.epochs

    set_seed(cfg["seed"])
    ddp, rank, world = init_distributed(cfg["ddp"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # -------------------------
    # datasets / loaders
    # -------------------------
    from src.data.dataset import build_datasets
    train_ds, val_ds = build_datasets(cfg)

    if is_primary():
        print(f"[DATA] train samples: {len(train_ds)} | val samples: {len(val_ds)}")
        tx, ty, tneg, tmask, tmeta = train_ds[0]
        print(f"[DATA] sample shapes: x={tuple(tx.shape)} y={tuple(ty.shape)} neg={tuple(tneg.shape)}")
        print(f"[DATA] meta example: {tmeta}")
        print(f"[DATA] mask shape: {tuple(tmask.shape)} | meta: {tmeta}")


    if ddp:
        train_samp = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        val_samp   = DistributedSampler(val_ds,   num_replicas=world, rank=rank, shuffle=False, drop_last=False)
    else:
        train_samp = None
        val_samp   = None

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=(train_samp is None),
        num_workers=cfg["num_workers"], sampler=train_samp, drop_last=True,
        pin_memory=True, collate_fn=collate_keep_meta
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, cfg["batch_size"]//2), shuffle=False,
        num_workers=cfg["num_workers"], sampler=val_samp, drop_last=False,
        pin_memory=True, collate_fn=collate_keep_meta
    )

    # -------------------------
    # CLIP / memory bank
    # -------------------------
    clip_cfg = cfg["model"]["clip"]
    clip_model, clip_preproc, clip_tok = load_open_clip(
        model_name=clip_cfg["model_name"], pretrained=clip_cfg["pretrained"], device=device
    )
    mem_names = clip_cfg["mem_degradations"]
    M = build_memory_text_embeddings(clip_model, clip_tok, mem_names, device=device)  # (K, d_txt)
    
    if is_primary():
        print(f"[CLIP] mem_names={len(mem_names)} | txt_dim={M.shape[1]}")

    # -------------------------
    # model / optim
    # -------------------------
    restorer = Restorer(cfg, clip_img_dim=512, clip_txt_dim=M.shape[1], mem_txt_embed=M, mem_names=mem_names).to(device)
    if ddp:
        restorer = torch.nn.parallel.DistributedDataParallel(
            restorer,
            device_ids=[rank % torch.cuda.device_count()],
            find_unused_parameters=True
        )

    opt = build_optimizer(restorer, cfg)
    sched = build_scheduler(opt, cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg["precision"]=="amp" or args.amp))

    # -------------------------
    # output dir & logging (ALL under cfg["output_dir"])
    # -------------------------
    outdir = os.path.join(cfg["output_dir"], cfg["experiment_name"])
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "train.log")
    if is_primary():
        with open(log_path, "a") as f:
            f.write(json.dumps({"event": "start",
                                "time": time.time(),
                                "config": cfg}, default=str) + "\n")
        print(f"[IO] outputs → {outdir}")
        print(f"[IO] log     → {log_path}")

    # -------------------------
    # resume (ONLY from --resume)
    # -------------------------
    start_step = 0
    resume_path = args.resume
    if resume_path and os.path.isfile(resume_path):
        start_step = load_ckpt(resume_path, restorer, opt, scaler, map_location=device)
        if is_primary():
            print(f"[Resume] loaded from {resume_path} at step {start_step}")
    else:
        if resume_path:
            print(f"[Resume] WARN: resume path not found: {resume_path}; starting from scratch.")
        if is_primary():
            print("[Resume] starting from scratch.")
            

    # -------------------------
    # losses
    # -------------------------

    uw = UncertaintyWeightedLoss(n_tasks=5).to(device) if cfg["loss"]["use_uncertainty"] else None
    dwt = HaarDWT().to(device)

    iters_per_epoch = cfg["optim"]["iters_per_epoch"]
    global_step = start_step
    # total_steps = cfg["optim"]["epochs"] * iters_per_epoch

    # ---- compute starting epoch & in-epoch offset for resumed runs ----
    start_epoch = global_step // iters_per_epoch
    start_iter_in_epoch = global_step % iters_per_epoch

    # -------------------------
    # training loop
    # -------------------------
    for epoch in range(start_epoch, cfg["optim"]["epochs"]):
        if ddp and train_samp is not None:
            train_samp.set_epoch(epoch)

        restorer.train()
        pbar = tqdm(total=iters_per_epoch, disable=not is_primary(),
                    desc=f"Epoch {epoch+1}/{cfg['optim']['epochs']}")

        it = iter(train_loader)
        # skip already processed iterations in resumed epoch
        for _ in range(start_iter_in_epoch):
            try:
                next(it)
            except StopIteration:
                it = iter(train_loader)
                next(it)
        start_iter_in_epoch = 0  # reset after first epoch boundary

        for _ in range(iters_per_epoch):
            try:
                x, y, y_neg, mask, meta = next(it)
            except StopIteration:
                it = iter(train_loader)
                x, y, y_neg, mask, meta = next(it)
                # 
            # x = x.to(device); y = y.to(device); y_neg = y_neg.to(device)
            # 
            x = x.to(device); y = y.to(device); y_neg = y_neg.to(device); mask = mask.to(device)
            m3 = mask.expand(-1, 3, -1, -1)  # Bx3xHxW


            with torch.no_grad():
                e_img = encode_images_for_clip(clip_model, clip_preproc, x)
                e_txt = torch.zeros((x.size(0), M.shape[1]), device=device)

            with torch.amp.autocast('cuda', enabled=(cfg["precision"]=="amp" or args.amp)):
                y_hat, alpha = restorer(x, e_img, e_txt)

                # reconstruction
                L_rec = masked_charbonnier(y_hat, y, mask) * cfg["loss"]["w_rec"]

                # wavelet subband
                # (L_pred_L, L_pred_Hh, L_pred_Hv, L_pred_Hd) = dwt(y_hat)
                # (L_true_L, L_true_Hh, L_true_Hv, L_true_Hd) = dwt(y)
                # L_wav = wavelet_frequency_loss(
                #     L_pred_L, [L_pred_Hh, L_pred_Hv, L_pred_Hd],
                #     L_true_L, [L_true_Hh, L_true_Hv, L_true_Hd],
                #     lambda_L=cfg["loss"]["w_wav_L"], lambda_H=cfg["loss"]["w_wav_H"]
                # )
                (L_pred_L, L_pred_Hh, L_pred_Hv, L_pred_Hd) = dwt(y_hat)
                (L_true_L, L_true_Hh, L_true_Hv, L_true_Hd) = dwt(y)
                m2 = down2_mask(mask)
                L_low  = l1_mask(L_pred_L, L_true_L, m2)
                L_high = (l1_mask(L_pred_Hh, L_true_Hh, m2) +
                        l1_mask(L_pred_Hv, L_true_Hv, m2) +
                        l1_mask(L_pred_Hd, L_true_Hd, m2)) / 3.0
                L_wav = cfg["loss"]["w_wav_L"] * L_low + cfg["loss"]["w_wav_H"] * L_high


                # composite contrast
                L_comp = composite_separation_loss(y_hat * m3, y_neg * m3, margin=cfg["loss"]["comp_margin"]) * cfg["loss"]["w_comp"]

                # blur-consistency (only if motion/blur present in meta)
                def _is_motion_blur(m):
                    if "degradations" in m:
                        return any(("motion_blur" in d) or ("blur" in d) for d in m["degradations"])
                    name = m.get("deg_name", "")
                    return ("motion" in name.lower() and "blur" in name.lower()) or (name.lower() == "blur")
                has_mblur = any(_is_motion_blur(m) for m in meta)
                if has_mblur:
                    L_blur = blur_consistency(y_hat * m3, x * m3, length=15, angle=0.0) * cfg["loss"]["w_blur"]
                else:
                    L_blur = torch.tensor(0.0, device=device)


                with torch.no_grad():
                    fy = encode_images_for_clip(clip_model, clip_preproc, (y * m3).clamp(0,1))
                fy_hat = encode_images_for_clip(clip_model, clip_preproc, (y_hat * m3).clamp(0,1))
                L_clip = clip_feature_loss(fy_hat, fy) * cfg["loss"]["w_clip"]

                
                
                if uw is not None:
                    total_loss = uw([L_rec, L_wav, L_comp, L_blur, L_clip])
                else:
                    total_loss = L_rec + L_wav + L_comp + L_blur + L_clip

            scaler.scale(total_loss).backward()

            if (global_step + 1) % cfg["grad_accum_steps"] == 0:
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

            # logging
            if is_primary() and (global_step % cfg["log_interval"] == 0):
                with torch.no_grad():
                    val_psnr = psnr((y_hat*m3).clamp(0,1), (y*m3).clamp(0,1))
                    val_ssim = ssim((y_hat*m3).clamp(0,1), (y*m3).clamp(0,1))

                msg = f"step {global_step} | loss {total_loss.item():.4f} | psnr {val_psnr:.2f} | ssim {val_ssim:.3f}"
                tqdm.write(msg)
                with open(log_path, "a") as f:
                    f.write(json.dumps({
                        "event": "log",
                        "time": time.time(),
                        "step": int(global_step),
                        "loss": float(total_loss.item()),
                        "psnr": float(val_psnr),
                        "ssim": float(val_ssim)
                    }) + "\n")

            # periodic save (ALL under outdir)
            if is_primary() and (global_step % cfg["save_interval"] == 0):
                ckpt_path = os.path.join(outdir, f"ckpt_{global_step}.pt")
                save_ckpt(ckpt_path, global_step, restorer, opt, scaler)
                
                
                # save_image(y_hat.clamp(0,1), os.path.join(outdir, f"train_pred_{global_step}.png"))
                # save_image(x.clamp(0,1),     os.path.join(outdir, f"train_lq_{global_step}.png"))
                # save_image(y.clamp(0,1),     os.path.join(outdir, f"train_gt_{global_step}.png"))
                
                
                
                from torchvision.utils import save_image, make_grid  # ensure import

                B = y_hat.size(0)
                for b in range(B):
                    panel = make_grid(
                        [x[b].clamp(0,1), y_hat[b].clamp(0,1), y[b].clamp(0,1)],
                        nrow=3, padding=2, pad_value=1.0  # <- was 255
                        )

                    # panel = make_grid(
                    #     [x[b].clamp(0,1), y_hat[b].clamp(0,1), y[b].clamp(0,1)],
                    #     nrow=3,               # 3 panels in one row
                    #     padding=2,            # set 0 if you don't want any gap
                    #     pad_value=255         # white gap; use 0 for black
                    # )
                    save_image(panel, os.path.join(outdir, f"train_triplet_{global_step}_{b}.png"))

                
                ######
                
                
                
                with open(log_path, "a") as f:
                    f.write(json.dumps({"event":"save","time":time.time(),"step":int(global_step),
                                        "ckpt": ckpt_path}) + "\n")

            global_step += 1
            pbar.update(1)


        if is_primary():
            restorer.eval()
            ps, ss = [], []
            by_deg = defaultdict(lambda: {"ps": [], "ss": []})
            with torch.no_grad():
                for vx, vy, _, vmask, vmeta in val_loader:
                    vmask = vmask.to(device)
                    vm3 = vmask.expand(-1,3,-1,-1)

                    vx = vx.to(device); vy = vy.to(device)
                    e_img = encode_images_for_clip(clip_model, clip_preproc, vx)
                    e_txt = torch.zeros((vx.size(0), M.shape[1]), device=device)
                    vy_hat, _ = restorer(vx, e_img, e_txt)
                    p = psnr((vy_hat*vm3).clamp(0,1), (vy*vm3).clamp(0,1))
                    s = ssim((vy_hat*vm3).clamp(0,1), (vy*vm3).clamp(0,1))
                    ps.append(p); ss.append(s)
                    # aggregate by degradation tag
                    for i, m in enumerate(vmeta):
                        tag = (m.get("deg_name") or (m.get("degradations") or ["mixed"])[0])
                        by_deg[tag]["ps"].append(float(p))
                        by_deg[tag]["ss"].append(float(s))
            mean_psnr = float(np.mean(ps)) if len(ps) else float("nan")
            mean_ssim = float(np.mean(ss)) if len(ss) else float("nan")
            print(f"[VAL] epoch {epoch} | PSNR {mean_psnr:.2f} | SSIM {mean_ssim:.3f}")
            for k, d in by_deg.items():
                if d["ps"]:
                    print(f"   • {k:<14} PSNR {np.mean(d['ps']):.2f}  SSIM {np.mean(d['ss']):.3f}")
            with open(log_path, 'a') as f:
                f.write(json.dumps({"event":"val","time":time.time(),"epoch":int(epoch),
                                    "psnr": mean_psnr, "ssim": mean_ssim,
                                    "by_deg": {k: {"ps": float(np.mean(v['ps'])), "ss": float(np.mean(v['ss']))} for k,v in by_deg.items() if v['ps']}
                                }) + "\n")

        # if ddp: 
        #     barrier()

            # final save
            # end-of-epoch safety checkpoint (kept to survive crashes between save_interval)
            if is_primary():
                final_ckpt = os.path.join(outdir, "ckpt_final.pt")  # rolling alias
                save_ckpt(final_ckpt, global_step, restorer, opt, scaler)

                # more informative console + log line
                msg = f"[FINAL EPOCH SAVE] epoch={epoch+1} step={global_step} → {final_ckpt}"
                print(msg)
                with open(log_path, "a") as f:
                    f.write(json.dumps({
                        "event": "final_epoch_save",
                        "time": time.time(),
                        "epoch": int(epoch + 1),
                        "step": int(global_step),
                        "final_ckpt": final_ckpt,
                        "total_steps": int(global_step)
                    }) + "\n")
        if ddp:
            barrier()
            
if __name__ == "__main__":
    main()
