import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import threestudio
from threestudio import register
from threestudio.utils.config import parse_structured


def load_rgba_as_rgb(
    image_path: str, image_size: int = 256, resize_to_foreground_90: bool = False
) -> torch.Tensor:
    img = Image.open(image_path).convert("RGBA").resize((image_size, image_size), Image.BICUBIC)
    rgba_u8 = np.array(img).astype(np.uint8)

    if resize_to_foreground_90:
        rgb_u8 = rgba_u8[..., :3]
        alpha_u8 = rgba_u8[..., 3]
        bg_mask = (rgb_u8[..., 0] > 250) & (rgb_u8[..., 1] > 250) & (rgb_u8[..., 2] > 250)
        fg_mask = (~bg_mask) & (alpha_u8 > 0)

        if fg_mask.any():
            ys, xs = np.where(fg_mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            box_h = max(1, y1 - y0)
            box_w = max(1, x1 - x0)

            target_size = max(1.0, 0.9 * float(image_size))
            scale = target_size / float(max(box_h, box_w))
            if scale > 1.0:
                new_h = max(1, int(round(box_h * scale)))
                new_w = max(1, int(round(box_w * scale)))
                new_h = min(image_size, new_h)
                new_w = min(image_size, new_w)

                patch = Image.fromarray(rgba_u8[y0:y1, x0:x1], mode="RGBA")
                patch = patch.resize((new_w, new_h), Image.BICUBIC)

                canvas = np.zeros_like(rgba_u8)
                top = (image_size - new_h) // 2
                left = (image_size - new_w) // 2
                canvas[top : top + new_h, left : left + new_w] = np.array(patch)
                rgba_u8 = canvas

    rgba = rgba_u8.astype(np.float32) / 255.0
    rgb = rgba[..., :3]
    a = rgba[..., 3:4]
    rgb = rgb * a + (1.0 - a)  # white background
    rgb = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()  # [3,H,W], [0,1]
    rgb = rgb * 2.0 - 1.0  # [-1,1]
    return rgb


@dataclass
class Zero123PairDataModuleConfig:
    train_manifest: str = ""
    val_manifest: Optional[str] = None
    image_size: int = 256
    batch_size: int = 1
    num_workers: int = 4
    shuffle: bool = True
    default_target_elev: Optional[float] = None
    resize_to_foreground_90: bool = False
    # Reference angle constraints. Azimuth is clamped; elevation filters records.
    ref_azim_min: Optional[float] = None  # clamp ref azimuth minimum
    ref_azim_max: Optional[float] = None  # clamp ref azimuth maximum
    ref_elev_min: Optional[float] = None  # filter ref elevations below this value
    ref_elev_max: Optional[float] = None  # filter ref elevations above this value


class Zero123PairDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        image_size: int = 256,
        default_target_elev: Optional[float] = None,
        resize_to_foreground_90: bool = False,
        ref_azim_min: Optional[float] = None,
        ref_azim_max: Optional[float] = None,
        ref_elev_min: Optional[float] = None,
        ref_elev_max: Optional[float] = None,
    ):
        super().__init__()
        assert os.path.exists(manifest_path), f"manifest not found: {manifest_path}"
        self.image_size = image_size
        self.default_target_elev = default_target_elev
        self.resize_to_foreground_90 = resize_to_foreground_90
        self.ref_azim_min = ref_azim_min
        self.ref_azim_max = ref_azim_max
        self.ref_elev_min = ref_elev_min
        self.ref_elev_max = ref_elev_max
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.records = [json.loads(x) for x in f if x.strip()]
        manifest_record_count = len(self.records)

        # filter out records with missing images
        def _has_valid_images(r):
            ref_img = self._get_ref_path(r)
            tgt_img = self._get_tgt_path(r)
            return os.path.exists(ref_img) and os.path.exists(tgt_img)

        self.records = [r for r in self.records if _has_valid_images(r)]

        def _has_valid_ref_elevation(r):
            ref_elev = float(r.get("ref_elev", r.get("elev", 0.0)))
            if self.ref_elev_min is not None and ref_elev < self.ref_elev_min:
                return False
            if self.ref_elev_max is not None and ref_elev > self.ref_elev_max:
                return False
            return True

        image_valid_record_count = len(self.records)
        self.records = [r for r in self.records if _has_valid_ref_elevation(r)]
        elevation_filtered_count = image_valid_record_count - len(self.records)
        threestudio.info(
            f"[Zero123PairDataset] manifest={manifest_path} "
            f"records={manifest_record_count} image_valid={image_valid_record_count} "
            f"elevation_filtered={elevation_filtered_count} kept={len(self.records)} "
            f"ref_elev_range=[{self.ref_elev_min}, {self.ref_elev_max}]"
        )
        assert len(self.records) > 0, f"empty manifest after filtering: {manifest_path}"

    def __len__(self):
        return len(self.records)

    def _get_ref_path(self, r):
        if "ref_img" in r:
            return r["ref_img"]
        if "orig_copy" in r:
            return r["orig_copy"]
        if "img_path" in r:
            return r["img_path"]
        raise KeyError("need one of: ref_img / orig_copy / img_path")

    def _get_tgt_path(self, r):
        if "tgt_img" in r:
            return r["tgt_img"]
        if "target_img" in r:
            return r["target_img"]
        raise KeyError("need tgt_img or target_img in manifest")

    def __getitem__(self, idx):
        r = self.records[idx]

        ref_img = self._get_ref_path(r)
        tgt_img = self._get_tgt_path(r)

        ref_yaw = float(r.get("ref_yaw", r.get("yaw", 0.0)))
        ref_elev = float(r.get("ref_elev", r.get("elev", 0.0)))
        tgt_yaw = float(r.get("tgt_yaw", ref_yaw))

        # Azimuth bounds retain their historical clamp behavior. Elevation bounds
        # are applied as record filters in __init__, so the manifest label remains intact.
        if self.ref_azim_min is not None:
            ref_yaw = max(ref_yaw, self.ref_azim_min)
        if self.ref_azim_max is not None:
            ref_yaw = min(ref_yaw, self.ref_azim_max)

        if "tgt_elev" in r:
            tgt_elev = float(r["tgt_elev"])
        elif self.default_target_elev is not None:
            tgt_elev = float(self.default_target_elev)
        else:
            raise KeyError("need tgt_elev in manifest or set data.default_target_elev")

        sample_id = str(r.get("sample_id", idx))
        filename = str(
            r.get(
                "filename",
                os.path.splitext(os.path.basename(ref_img))[0],
            )
        )

        return {
            "sample_id": sample_id,
            "filename": filename,
            "ref": load_rgba_as_rgb(
                ref_img,
                self.image_size,
                resize_to_foreground_90=self.resize_to_foreground_90,
            ),
            "tgt": load_rgba_as_rgb(
                tgt_img,
                self.image_size,
                resize_to_foreground_90=self.resize_to_foreground_90,
            ),
            "ref_azim": torch.tensor(ref_yaw, dtype=torch.float32),
            "ref_elev": torch.tensor(ref_elev, dtype=torch.float32),
            "tgt_azim": torch.tensor(tgt_yaw, dtype=torch.float32),
            "tgt_elev": torch.tensor(tgt_elev, dtype=torch.float32),
        }


@register("zero123-pair-datamodule")
class Zero123PairDataModule(pl.LightningDataModule):
    cfg: Zero123PairDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(Zero123PairDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = Zero123PairDataset(
                self.cfg.train_manifest,
                image_size=self.cfg.image_size,
                default_target_elev=self.cfg.default_target_elev,
                resize_to_foreground_90=self.cfg.resize_to_foreground_90,
                ref_azim_min=self.cfg.ref_azim_min,
                ref_azim_max=self.cfg.ref_azim_max,
                ref_elev_min=self.cfg.ref_elev_min,
                ref_elev_max=self.cfg.ref_elev_max,
            )
        if stage in [None, "fit", "validate"]:
            val_manifest = self.cfg.val_manifest or self.cfg.train_manifest
            self.val_dataset = Zero123PairDataset(
                val_manifest,
                image_size=self.cfg.image_size,
                default_target_elev=self.cfg.default_target_elev,
                resize_to_foreground_90=self.cfg.resize_to_foreground_90,
                ref_azim_min=self.cfg.ref_azim_min,
                ref_azim_max=self.cfg.ref_azim_max,
                ref_elev_min=self.cfg.ref_elev_min,
                ref_elev_max=self.cfg.ref_elev_max,
            )
        if stage in [None, "test", "predict"]:
            val_manifest = self.cfg.val_manifest or self.cfg.train_manifest
            self.test_dataset = Zero123PairDataset(
                val_manifest,
                image_size=self.cfg.image_size,
                default_target_elev=self.cfg.default_target_elev,
                resize_to_foreground_90=self.cfg.resize_to_foreground_90,
                ref_azim_min=self.cfg.ref_azim_min,
                ref_azim_max=self.cfg.ref_azim_max,
                ref_elev_min=self.cfg.ref_elev_min,
                ref_elev_max=self.cfg.ref_elev_max,
            )

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, self.cfg.num_workers // 2),
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, self.cfg.num_workers // 2),
            pin_memory=True,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
