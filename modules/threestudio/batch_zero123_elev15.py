#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image


# -----------------------------
# Utilities to instantiate Zero123 LDM from yaml
# -----------------------------
def get_obj_from_str(string: str, reload: bool = False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model_from_config(config, ckpt_path: str, device: str, keep_decoder: bool = True):
    """
    keep_decoder=True: 推理需要 decode_first_stage，所以必须保留 decoder。
    """
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd

    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)

    # Zero123 / Stable-Zero123 常见：有 EMA 权重，加载后可删掉以省显存
    if getattr(model, "use_ema", False):
        model.model_ema.copy_to(model.model)
        del model.model_ema

    if not keep_decoder:
        # 训练 SDS 时可以删 decoder；推理不要删
        if hasattr(model, "first_stage_model") and hasattr(model.first_stage_model, "decoder"):
            del model.first_stage_model.decoder
            torch.cuda.empty_cache()

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# -----------------------------
# Image IO
# -----------------------------
def load_rgba_as_rgb_256(image_path: str) -> torch.Tensor:
    """
    返回 [1,3,256,256]，范围 [0,1]。
    如果输入无 alpha，则 alpha=1。
    透明区域用白底合成（和 threestudio 的 zero123 处理一致）。
    """
    img = Image.open(image_path).convert("RGBA")
    img = img.resize((256, 256), Image.BICUBIC)
    rgba = np.array(img).astype(np.float32) / 255.0

    rgb = rgba[..., :3]
    a = rgba[..., 3:4]
    rgb = rgb * a + (1.0 - a)

    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
    return rgb_t


def is_hex32_dirname(name: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{32}", name))


def parse_xy_png(filename: str) -> Optional[Tuple[int, int]]:
    """
    解析 x_y.png -> (x, y)
    支持负号，如 -30_45.png
    """
    m = re.fullmatch(r"(-?\d+)_(-?\d+)\.png", filename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# -----------------------------
# Stable Zero123 Inference
# -----------------------------
class StableZero123Infer:
    """
    纯推理版：
    - prepare_embeddings(image): 得到 c_crossattn / c_concat
    - set_reference_view(cond_elev, cond_azim, cond_dist)
    - sample_one(target_elev, target_azim, target_dist)
    """

    def __init__(
        self,
        ckpt_path: str,
        model_config_path: str,
        device: str = "cuda",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 100,
    ):
        self.device = device
        self.cfg = OmegaConf.load(model_config_path)
        self.model = load_model_from_config(self.cfg, ckpt_path, device=device, keep_decoder=True)

        # 与 threestudio/stable_zero123_guidance.py 一致的构造方式（旧版 diffusers 可用这种 positional 写法）
        num_train_timesteps = self.cfg.model.params.timesteps
        linear_start = self.cfg.model.params.linear_start
        linear_end = self.cfg.model.params.linear_end
        self.scheduler = DDIMScheduler(
            num_train_timesteps,
            linear_start,
            linear_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.guidance_scale = float(guidance_scale)
        self.num_inference_steps = int(num_inference_steps)

        # reference view (输入图视角)
        self.cond_elevation_deg = 0.0
        self.cond_azimuth_deg = 0.0
        self.cond_camera_distance = 1.2

        # embeddings cache for current input image
        self.c_crossattn = None  # [1,1,768]
        self.c_concat = None     # [1,4,32,32]

    def set_reference_view(self, cond_elev: float, cond_azim: float, cond_dist: float = 1.2):
        self.cond_elevation_deg = float(cond_elev)
        self.cond_azimuth_deg = float(cond_azim)
        self.cond_camera_distance = float(cond_dist)

    @torch.no_grad()
    def prepare_embeddings(self, image_path: str):
        rgb_256 = load_rgba_as_rgb_256(image_path).to(self.device)

        img = rgb_256 * 2.0 - 1.0  # [-1,1]
        c_crossattn = self.model.get_learned_conditioning(img.float())
        c_concat = self.model.encode_first_stage(img.float()).mode()

        self.c_crossattn = c_crossattn
        self.c_concat = c_concat

    @torch.no_grad()
    def get_cond(self, elevation_deg: float, azimuth_deg: float, camera_distance: float) -> Dict[str, Any]:
        assert self.c_crossattn is not None and self.c_concat is not None, "Call prepare_embeddings() first."

        elevation = torch.tensor([elevation_deg], device=self.device, dtype=torch.float32)
        azimuth = torch.tensor([azimuth_deg], device=self.device, dtype=torch.float32)

        # 与 threestudio 的 stable_zero123_guidance 公式一致：用相对视角构造 T（注意 polar=90-elev）
        T = torch.stack(
            [
                torch.deg2rad((90.0 - elevation) - (90.0 - torch.tensor(self.cond_elevation_deg, device=self.device))),
                torch.sin(torch.deg2rad(azimuth - self.cond_azimuth_deg)),
                torch.cos(torch.deg2rad(azimuth - self.cond_azimuth_deg)),
                torch.deg2rad(90.0 - torch.full_like(elevation, self.cond_elevation_deg)),
            ],
            dim=-1,
        )[:, None, :]  # [1,1,4]

        clip_emb = self.model.cc_projection(torch.cat([self.c_crossattn, T], dim=-1))  # [1,1,768]

        cond = {
            "c_crossattn": [torch.cat([torch.zeros_like(clip_emb), clip_emb], dim=0)],
            "c_concat": [torch.cat([torch.zeros_like(self.c_concat), self.c_concat], dim=0)],
        }
        return cond

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        image = self.model.decode_first_stage(latents)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image

    @torch.no_grad()
    def sample_one(
        self,
        target_elevation_deg: float,
        target_azimuth_deg: float,
        target_camera_distance: float = 1.2,
        seed: int = 0,
    ) -> Image.Image:
        cond = self.get_cond(target_elevation_deg, target_azimuth_deg, target_camera_distance)

        g = torch.Generator(device=self.device).manual_seed(int(seed))

        # 256 输入 -> latent 32x32, 4 通道
        latents = torch.randn((1, 4, 32, 32), generator=g, device=self.device, dtype=torch.float32)

        self.scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(self.device)

        for t in timesteps:
            x_in = torch.cat([latents, latents], dim=0)
            t_in = torch.cat([t.reshape(1), t.reshape(1)], dim=0).to(self.device)

            noise_pred = self.model.apply_model(x_in, t_in, cond)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]

        img = self.decode_latents(latents)[0]  # [3,H,W]
        img = (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(img)


# -----------------------------
# Batch job
# -----------------------------
@dataclass
class ViewSpec:
    x: int
    y: int


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/mnt/hdd3/linzhuohang/3DGen/high_score_25k", help="high_score_25k 根目录")
    parser.add_argument("--output_root", default="./output", help="输出根目录")
    parser.add_argument("--num_items", type=int, default=50, help="抽样物品数量")
    parser.add_argument("--seed", type=int, default=123, help="随机种子（抽样 + 生成）")

    parser.add_argument("--ckpt", required=True, help="stable_zero123.ckpt 路径")
    parser.add_argument("--model_config", required=True, help="sd-objaverse-finetune-c_concat-256.yaml 路径")

    parser.add_argument("--target_elev", type=float, default=15.0, help="目标仰角（度）")
    parser.add_argument("--target_dist", type=float, default=1.2, help="相机距离（一般 1.2/3.8 看你训练配置）")
    parser.add_argument("--steps", type=int, default=100, help="DDIM steps")
    parser.add_argument("--scale", type=float, default=7.5, help="CFG guidance scale")
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")

    args = parser.parse_args()

    # 确保能 import extern.ldm_zero123...
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # 轻微加速（A6000 是 Ampere）
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 你指定要拿这三张
    needed_views = [ViewSpec(0, 45), ViewSpec(0, 60), ViewSpec(0, 75)]

    # 找到满足三张都存在的物品
    all_ids: List[str] = []
    for name in os.listdir(args.data_root):
        if not is_hex32_dirname(name):
            continue
        item_dir = os.path.join(args.data_root, name)
        if not os.path.isdir(item_dir):
            continue
        ok = True
        for v in needed_views:
            if not os.path.exists(os.path.join(item_dir, f"{v.x}_{v.y}.png")):
                ok = False
                break
        if ok:
            all_ids.append(name)

    if len(all_ids) == 0:
        raise RuntimeError(f"在 {args.data_root} 下找不到同时包含 0_45/0_60/0_75.png 的物品文件夹")

    random.seed(args.seed)
    random.shuffle(all_ids)
    chosen = all_ids[: min(args.num_items, len(all_ids))]

    os.makedirs(args.output_root, exist_ok=True)

    infer = StableZero123Infer(
        ckpt_path=args.ckpt,
        model_config_path=args.model_config,
        device=args.device,
        guidance_scale=args.scale,
        num_inference_steps=args.steps,
    )

    print(f"[INFO] eligible items: {len(all_ids)}")
    print(f"[INFO] chosen items:   {len(chosen)}")
    print(f"[INFO] output_root:   {os.path.abspath(args.output_root)}")

    for idx, item_id in enumerate(chosen, 1):
        src_dir = os.path.join(args.data_root, item_id)
        out_dir = os.path.join(args.output_root, item_id)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[{idx}/{len(chosen)}] {item_id}")

        for v in needed_views:
            src_img = os.path.join(src_dir, f"{v.x}_{v.y}.png")
            dst_img = os.path.join(out_dir, f"{v.x}_{v.y}.png")
            shutil.copy2(src_img, dst_img)

            # reference view = 输入图本身视角 (x,y)
            infer.set_reference_view(cond_elev=v.y, cond_azim=v.x, cond_dist=args.target_dist)
            infer.prepare_embeddings(src_img)

            # target view：同 azim（0），仰角变成 15°
            # 你说的“0_45 偏转 -30”在这里就是从 45 -> 15（模型内部用 polar 差来编码）
            out = infer.sample_one(
                target_elevation_deg=float(args.target_elev),
                target_azimuth_deg=float(v.x),
                target_camera_distance=float(args.target_dist),
                seed=args.seed + idx * 1000 + v.y,
            )

            gen_name = f"{v.x}_{int(args.target_elev)}_from_{v.x}_{v.y}.png"
            out_path = os.path.join(out_dir, gen_name)
            out.save(out_path)
            print(f"  - {os.path.basename(dst_img)}  ->  {os.path.basename(out_path)}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()