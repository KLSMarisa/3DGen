#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
from pathlib import Path

import torch
from torch import autocast
from contextlib import nullcontext

from PIL import Image
import numpy as np

from omegaconf import OmegaConf
from torchvision import transforms
from einops import rearrange

# --- Make sure we can import `ldm` from threestudio's extern ---
def add_ldm_to_syspath(threestudio_dir: str):
    import sys
    ts = Path(threestudio_dir).resolve()
    # threestudio/extern/ldm_zero123 contains the `ldm/` package
    sys.path.insert(0, str(ts / "extern" / "ldm_zero123"))
    # sometimes configs refer to modules relative to threestudio root
    sys.path.insert(0, str(ts))

def load_model_from_config(config, ckpt, device, verbose=False):
    from ldm.util import instantiate_from_config
    print(f"[INFO] Loading ckpt: {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"[INFO] Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose and missing:
        print("[WARN] missing keys:", missing)
    if verbose and unexpected:
        print("[WARN] unexpected keys:", unexpected)

    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def preprocess_rgba_or_rgb(pil_img: Image.Image, size=256) -> np.ndarray:
    """
    Returns float32 HxWx3 in [0,1], with white background compositing if RGBA.
    """
    pil_img = pil_img.convert("RGBA") if pil_img.mode != "RGBA" else pil_img
    pil_img = pil_img.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0  # HxWx4 in [0,1]
    alpha = arr[:, :, 3:4]
    white = np.ones_like(arr)
    arr = alpha * arr + (1.0 - alpha) * white
    arr = arr[:, :, 0:3]  # HxWx3
    return arr

@torch.no_grad()
def sample_model(
    input_im, model, sampler,
    precision, h, w,
    ddim_steps, n_samples,
    scale, ddim_eta,
    polar_deg, azim_deg, radius_offset
):
    """
    input_im: 1x3xHxW in [-1,1]
    returns: n_samplesx3xHxW in [0,1] on CPU
    """
    precision_scope = autocast if precision == "autocast" else nullcontext
    with precision_scope("cuda" if input_im.is_cuda else "cpu"):
        with model.ema_scope():
            # image conditioning (CLIP image embedding)
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)

            # camera conditioning
            T = torch.tensor([
                math.radians(polar_deg),
                math.sin(math.radians(azim_deg)),
                math.cos(math.radians(azim_deg)),
                float(radius_offset),
            ], device=c.device, dtype=c.dtype)

            T = T[None, None, :].repeat(n_samples, 1, 1)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)

            cond = {
                "c_crossattn": [c],
                "c_concat": [
                    model.encode_first_stage(input_im.to(c.device)).mode().detach()
                    .repeat(n_samples, 1, 1, 1)
                ],
            }

            if scale != 1.0:
                uc = {
                    "c_concat": [torch.zeros(n_samples, 4, h // 8, w // 8, device=c.device, dtype=c.dtype)],
                    "c_crossattn": [torch.zeros_like(c, device=c.device, dtype=c.dtype)],
                }
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples, _ = sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                x_T=None,
            )

            x_samples = model.decode_first_stage(samples)
            out = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0).cpu()
            return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input image path (prefer RGBA with transparent bg)")
    ap.add_argument("--outdir", default="./outputs_zero123", help="output directory")

    ap.add_argument("--polar", type=float, default=0.0, help="polar/elevation in degrees (仰角)")
    ap.add_argument("--azim", type=float, default=0.0, help="azimuth in degrees (方位角)")
    ap.add_argument("--radius", type=float, default=0.0, help="radius offset (缩放/推拉), typically in [-0.5,0.5]")

    ap.add_argument("--n_samples", type=int, default=1, help="how many images to sample")
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps")
    ap.add_argument("--scale", type=float, default=3.0, help="CFG scale")
    ap.add_argument("--eta", type=float, default=1.0, help="DDIM eta (0.0 deterministic, 1.0 more stochastic)")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--precision", choices=["fp32", "autocast"], default="fp32")

    ap.add_argument("--threestudio_dir",
                    default="/home/linzhuohang/3DGen/modules/threestudio",
                    help="path to threestudio repo (for extern/ldm_zero123)")
    ap.add_argument("--config_yaml",
                    default="/home/linzhuohang/3DGen/modules/threestudio/load/zero123/sd-objaverse-finetune-c_concat-256.yaml")
    ap.add_argument("--ckpt",
                    default="/home/linzhuohang/3DGen/modules/threestudio/load/zero123/stable_zero123.ckpt")

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    add_ldm_to_syspath(args.threestudio_dir)

    from ldm.models.diffusion.ddim import DDIMSampler

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available, use --device cpu")

    # reproducibility
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # load model
    config = OmegaConf.load(args.config_yaml)
    model = load_model_from_config(config, args.ckpt, device=device)

    # load & preprocess image
    pil = Image.open(args.image)
    img01 = preprocess_rgba_or_rgb(pil, size=256)  # HxWx3 [0,1]
    x = transforms.ToTensor()(img01).unsqueeze(0).to(device)  # 1x3x256x256 [0,1]
    x = x * 2.0 - 1.0  # [-1,1]
    x = transforms.functional.resize(x, [args.h, args.w])

    sampler = DDIMSampler(model)

    # sample
    out = sample_model(
        input_im=x,
        model=model,
        sampler=sampler,
        precision=args.precision,
        h=args.h, w=args.w,
        ddim_steps=args.steps,
        n_samples=args.n_samples,
        scale=args.scale,
        ddim_eta=args.eta,
        polar_deg=args.polar,
        azim_deg=args.azim,
        radius_offset=args.radius,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # save
    for i in range(out.shape[0]):
        im = (255.0 * rearrange(out[i].numpy(), "c h w -> h w c")).astype(np.uint8)
        save_path = outdir / f"view_p{args.polar:.1f}_a{args.azim:.1f}_r{args.radius:.2f}_s{args.seed}_{i}.png"
        Image.fromarray(im).save(save_path)
        print("[OK] saved:", save_path)

if __name__ == "__main__":
    main()