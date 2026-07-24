import argparse
import importlib
import os
import sys
from typing import Any, Dict, Tuple

import numpy as np
import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image


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


def load_model_from_config(config, ckpt_path: str, device: str, vram_O: bool = True):
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd

    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)

    # 省显存：不需要 decoder 时可删（threestudio 里也是这么做的）:contentReference[oaicite:2]{index=2}
    if getattr(model, "use_ema", False):
        model.model_ema.copy_to(model.model)
        del model.model_ema

    if vram_O:
        # 只在你确认推理只需要 decode 时不要删；我们推理需要 decode，所以不删 decoder
        pass

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_rgba_as_rgb_256(image_path: str) -> torch.Tensor:
    """
    返回 [1,3,256,256]，范围 [0,1]。
    如果输入无 alpha，就当 alpha=1。
    """
    img = Image.open(image_path).convert("RGBA")
    img = img.resize((256, 256), Image.BICUBIC)
    rgba = np.array(img).astype(np.float32) / 255.0
    rgb = rgba[..., :3]
    a = rgba[..., 3:4]
    rgb = rgb * a + (1.0 - a)  # 透明背景按白底合成（threestudio 同逻辑）:contentReference[oaicite:3]{index=3}
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
    return rgb_t


class StableZero123Infer:
    def __init__(
        self,
        ckpt_path: str,
        config_path: str,
        device: str = "cuda",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 100,
        cond_elevation_deg: float = 0.0,
        cond_azimuth_deg: float = 0.0,
        cond_camera_distance: float = 1.2,
    ):
        self.device = device
        self.cfg = OmegaConf.load(config_path)
        self.model = load_model_from_config(self.cfg, ckpt_path, device=device, vram_O=False)

        # DDIM scheduler（threestudio 用 diffusers 的 DDIMScheduler）:contentReference[oaicite:4]{index=4}
        timesteps = self.cfg.model.params.timesteps
        linear_start = self.cfg.model.params.linear_start
        linear_end = self.cfg.model.params.linear_end
        self.scheduler = DDIMScheduler(
            timesteps,
            linear_start,
            linear_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

        # 输入图对应的“参考视角”（如果你输入图不是正面，就要改这里）
        self.cond_elevation_deg = cond_elevation_deg
        self.cond_azimuth_deg = cond_azimuth_deg
        self.cond_camera_distance = cond_camera_distance

        self.c_crossattn = None
        self.c_concat = None

    @torch.no_grad()
    def prepare_embeddings(self, image_path: str):
        rgb_256 = load_rgba_as_rgb_256(image_path).to(self.device)

        # threestudio: img in [-1,1] -> get_learned_conditioning & encode_first_stage:contentReference[oaicite:5]{index=5}
        img = rgb_256 * 2.0 - 1.0
        c_crossattn = self.model.get_learned_conditioning(img.float())
        c_concat = self.model.encode_first_stage(img.float()).mode()

        self.c_crossattn = c_crossattn
        self.c_concat = c_concat

    @torch.no_grad()
    def get_cond(
        self,
        elevation_deg: float,
        azimuth_deg: float,
        camera_distance: float,
    ) -> Dict[str, Any]:
        assert self.c_crossattn is not None and self.c_concat is not None, "Call prepare_embeddings() first."

        # 这段就是 threestudio stable_zero123_guidance.py 的相机 embedding 方式（相对视角）:contentReference[oaicite:6]{index=6}
        elevation = torch.tensor([elevation_deg], device=self.device, dtype=torch.float32)
        azimuth = torch.tensor([azimuth_deg], device=self.device, dtype=torch.float32)

        T = torch.stack(
            [
                torch.deg2rad((90 - elevation) - (90 - torch.tensor(self.cond_elevation_deg, device=self.device))),
                torch.sin(torch.deg2rad(azimuth - self.cond_azimuth_deg)),
                torch.cos(torch.deg2rad(azimuth - self.cond_azimuth_deg)),
                torch.deg2rad(90 - torch.full_like(elevation, self.cond_elevation_deg)),
            ],
            dim=-1,
        )[:, None, :]

        # cc_projection 将 [clip_embed, T] 拼起来投影成 cross-attn conditioning:contentReference[oaicite:7]{index=7}
        clip_emb = self.model.cc_projection(torch.cat([self.c_crossattn, T], dim=-1))

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

        g = torch.Generator(device=self.device)
        g.manual_seed(seed)

        # latent 尺寸：stable-zero123 的 config 是 256 输入，对应 latent 32x32（4 通道）
        latents = torch.randn((1, 4, 32, 32), generator=g, device=self.device, dtype=torch.float32)

        self.scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(self.device)

        for t in timesteps:
            x_in = torch.cat([latents] * 2, dim=0)
            t_in = torch.cat([t.reshape(1), t.reshape(1)], dim=0).to(self.device)

            noise_pred = self.model.apply_model(x_in, t_in, cond)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]

        img = self.decode_latents(latents)[0]  # [3,H,W]
        img = (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="path to stable_zero123.ckpt")
    parser.add_argument("--config", required=True, help="path to sd-objaverse-finetune-c_concat-256.yaml")
    parser.add_argument("--input", required=True, help="input image (最好 RGBA 去背景)")
    parser.add_argument("--output", required=True, help="output image path, e.g. out.png")

    parser.add_argument("--target_elev", type=float, default=0.0)
    parser.add_argument("--target_azim", type=float, default=30.0)
    parser.add_argument("--target_dist", type=float, default=1.2)

    parser.add_argument("--cond_elev", type=float, default=0.0)
    parser.add_argument("--cond_azim", type=float, default=0.0)
    parser.add_argument("--cond_dist", type=float, default=1.2)

    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    # 确保能 import 到 config target 的模块（通常在 threestudio 根目录运行就行）
    # 如果你的 ldm 在 extern/ 里，额外加一下：
    repo_root = os.path.dirname(os.path.abspath(__file__))
    extern_dir = os.path.join(repo_root, "extern")
    if extern_dir not in sys.path:
        sys.path.insert(0, extern_dir)

    infer = StableZero123Infer(
        ckpt_path=args.ckpt,
        config_path=args.config,
        device=args.device,
        guidance_scale=args.scale,
        num_inference_steps=args.steps,
        cond_elevation_deg=args.cond_elev,
        cond_azimuth_deg=args.cond_azim,
        cond_camera_distance=args.cond_dist,
    )
    infer.prepare_embeddings(args.input)
    out_img = infer.sample_one(
        target_elevation_deg=args.target_elev,
        target_azimuth_deg=args.target_azim,
        target_camera_distance=args.target_dist,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_img.save(args.output)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()