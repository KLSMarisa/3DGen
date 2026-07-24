import contextlib
import importlib
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from omegaconf import OmegaConf

import threestudio
from threestudio.systems.base import BaseSystem


def get_obj_from_str(string: str):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model_from_config_trainable(config, ckpt_path: str):
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    sd = pl_sd["state_dict"] if isinstance(pl_sd, dict) and "state_dict" in pl_sd else pl_sd

    # 检查是否所有的 key 都以 "model." 开头
    # 确保 sd 不为空，避免 empty dict 的 all() 返回 True
    if len(sd) > 0 and all(k.startswith("model.") for k in sd.keys()):
        # 批量去掉最前面的 "model." (6个字符)
        sd = {k[6:]: v for k, v in sd.items()}

    model = instantiate_from_config(config.model)

    # 改为 strict=True，严格检查权重名称是否完全匹配
    model.load_state_dict(sd, strict=False)

    return model


@threestudio.register("zero123-finetune-system")
class Zero123FinetuneSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        pretrained_model_name_or_path: str = "./load/zero123/stable_zero123.ckpt"
        pretrained_config: str = "./load/zero123/sd-objaverse-finetune-c_concat-256.yaml"

        # --- 训练模块开关 (仅训练 UNet) ---
        train_unet: bool = True
        train_cc_projection: bool = False
        train_cond_stage: bool = False
        train_first_stage: bool = False

        lr: float = 1e-5
        betas: Tuple[float, float] = (0.9, 0.999)
        weight_decay: float = 1e-2
        lr_t_max: int = 10000
        lr_eta_min: float = 1e-6

        val_ddim_steps: int = 50
        val_guidance_scale: float = 7.5
        val_ddim_eta: float = 1.0
        val_save_generated: bool = True
        val_filename_with_batch_item_suffix: bool = True

    cfg: Config

    def configure(self) -> None:
        self.ldm_cfg = OmegaConf.load(self.cfg.pretrained_config)
        self.model = load_model_from_config_trainable(
            self.ldm_cfg, self.cfg.pretrained_model_name_or_path
        )

        # 1. 冻结所有参数
        for p in self.model.parameters():
            p.requires_grad_(False)

        # 2. 根据配置仅解冻 UNet
        if self.cfg.train_unet and hasattr(self.model, "model"):
            for p in self.model.model.diffusion_model.parameters():
                p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        threestudio.info(f"[Zero123Finetune] trainable params: {trainable}/{total}")

    def preprocess_data(self, batch, stage):
        pass

    def _get_ldm_batch(self, batch):
        """将 threestudio 的 batch 转换为原版 ddpm.py 期望的字典格式，做维度对齐"""
        # ddpm.py 内部期待 h w c 格式，其 get_input 会再转回 c h w
        tgt_hwc = rearrange(batch["tgt"], "b c h w -> b h w c")
        ref_hwc = rearrange(batch["ref"], "b c h w -> b h w c")

        # 构造相机相对参数 T
        T = torch.stack(
            [
                torch.deg2rad((90.0 - batch["tgt_elev"]) - (90.0 - batch["ref_elev"])),
                torch.sin(torch.deg2rad(batch["tgt_azim"] - batch["ref_azim"])),
                torch.cos(torch.deg2rad(batch["tgt_azim"] - batch["ref_azim"])),
                torch.deg2rad(90.0 - batch["ref_elev"]),
            ],
            dim=-1,
        )

        ldm_batch = {
            self.model.first_stage_key: tgt_hwc,
            self.model.cond_stage_key: ref_hwc,
            "T": T,
        }
        return ldm_batch

    def on_train_start(self):
        # 如果需要定期保存完整的 state_dict，可在此实现
        return

    def _save_generated_images(self, pred_01: torch.Tensor, batch, batch_idx: int):
        out_dir = os.path.join(
            self.get_save_dir(), f"it{self.true_global_step}-val-noise2img"
        )
        os.makedirs(out_dir, exist_ok=True)

        ref_01 = torch.clamp((batch["ref"] + 1.0) / 2.0, 0.0, 1.0).to(pred_01.dtype)
        tgt_01 = torch.clamp((batch["tgt"] + 1.0) / 2.0, 0.0, 1.0).to(pred_01.dtype)

        if ref_01.shape[-2:] != pred_01.shape[

            -2:]:
            ref_01 = F.interpolate(
                ref_01,
                size=pred_01.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if tgt_01.shape[-2:] != pred_01.shape[-2:]:
            tgt_01 = F.interpolate(
                tgt_01,
                size=pred_01.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        stitched_01 = torch.cat([ref_01, tgt_01, pred_01], dim=-1)

        for i in range(stitched_01.shape[0]):
            img = stitched_01[i].detach().cpu().permute(1, 2, 0).numpy()
            img = (img * 255.0).clip(0, 255).astype("uint8")
            if "filename" in batch:
                if isinstance(batch["filename"], (list, tuple)):
                    name = str(batch["filename"][i])
                else:
                    name = str(batch["filename"])
                base = os.path.splitext(os.path.basename(name))[0]
            elif "sample_id" in batch:
                if isinstance(batch["sample_id"], (list, tuple)):
                    sid = str(batch["sample_id"][i])
                else:
                    sid = str(batch["sample_id"])
                base = sid
            elif "index" in batch:
                base = f"{int(batch['index'][i]):06d}"
            else:
                base = "sample"
            if self.cfg.val_filename_with_batch_item_suffix:
                filename = f"{base}_b{batch_idx:06d}_i{i:02d}.png"
            else:
                filename = f"{base}.png"
            Image.fromarray(img).save(os.path.join(out_dir, filename))

    def training_step(self, batch, batch_idx):
        self.model.logvar = self.model.logvar.to(self.device)
        ldm_batch = self._get_ldm_batch(batch)

        # 1. 使用原版逻辑获取输入，包含特征编码、内置的 CFG Dropout
        z_tgt, cond = self.model.get_input(ldm_batch, self.model.first_stage_key)

        # 2. 直接前向传播拿 Loss
        loss, loss_dict = self.model(z_tgt, cond)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.logvar = self.model.logvar.to(self.device)
        ldm_batch = self._get_ldm_batch(batch)

        z_tgt, cond = self.model.get_input(ldm_batch, self.model.first_stage_key)
        loss, loss_dict = self.model(z_tgt, cond)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        with torch.no_grad():
            B = z_tgt.shape[0]
            T = ldm_batch["T"].to(z_tgt.device)
            ref_tensor = batch["ref"].to(z_tgt.device).float()

            # --- 构造 Cond ---
            clip_emb = self.model.get_learned_conditioning(ref_tensor).detach()
            c_concat = self.model.encode_first_stage(ref_tensor).mode().detach()
            cond_emb = self.model.cc_projection(torch.cat([clip_emb, T[:, None, :]], dim=-1))
            c = {"c_crossattn": [cond_emb], "c_concat": [c_concat]}

            # --- 构造 Uncond (文本置空/图像置零，保留相机 T) ---
            null_prompt = self.model.get_learned_conditioning([""] * B).detach()
            uncond_concat = torch.zeros_like(c_concat)
            uncond_emb = self.model.cc_projection(torch.cat([null_prompt.repeat(B, 1, 1), T[:, None, :]], dim=-1))
            uc = {"c_crossattn": [uncond_emb], "c_concat": [uncond_concat]}

            # --- 采样 ---
            samples, _ = self.model.sample_log(
                cond=c,
                batch_size=B,
                ddim=True,
                ddim_steps=self.cfg.val_ddim_steps,
                eta=self.cfg.val_ddim_eta,
                unconditional_guidance_scale=self.cfg.val_guidance_scale,
                unconditional_conditioning=uc,
            )

            # 解码
            pred_01 = self.model.decode_first_stage(samples)
            pred_01 = torch.clamp((pred_01 + 1.0) / 2.0, 0.0, 1.0)

            # 存图评估
            tgt_01 = torch.clamp((batch["tgt"] + 1.0) / 2.0, 0.0, 1.0).to(pred_01.dtype)
            if pred_01.shape[-2:] != tgt_01.shape[-2:]:
                pred_01 = F.interpolate(
                    pred_01, size=tgt_01.shape[-2:], mode="bilinear", align_corners=False
                )
            noise2img_mse = F.mse_loss(pred_01, tgt_01)
            self.log("val/noise2img_mse", noise2img_mse, prog_bar=True, on_step=False, on_epoch=True)

            if self.cfg.val_save_generated:
                self._save_generated_images(pred_01, batch, batch_idx)

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        self.model.logvar = self.model.logvar.to(self.device)
        ldm_batch = self._get_ldm_batch(batch)
        z_tgt, cond = self.model.get_input(ldm_batch, self.model.first_stage_key)
        loss, loss_dict = self.model(z_tgt, cond)
        self.log("test/loss", loss, prog_bar=True)
        return {"test_loss": loss}

    def predict_step(self, batch, batch_idx):
        return {}

    def configure_optimizers(self):
        # 此时 self.model 中只有 UNet 被设为了 requires_grad=True
        params = [p for p in self.model.parameters() if p.requires_grad]
        assert len(params) > 0, "No trainable parameters found."

        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.lr,
            betas=self.cfg.betas,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.lr_t_max,
            eta_min=self.cfg.lr_eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }