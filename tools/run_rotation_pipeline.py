#!/usr/bin/env python3
"""
端到端图片旋转 pipeline：
  1. 预测输入图片的仰角（angle_predictor）
  2. 构建 JSONL manifest 供 threestudio Zero123 使用
  3. 运行 Zero123 验证，生成目标仰角的旋转视图
  4. 后处理：resize 到 1024 + meanshift 平滑 -> 直接输出到 output_dir

用法:
  python tools/run_rotation_pipeline.py \
      --input_dir /path/to/images \
      --output_dir /path/to/output

  # 仅预测角度 + 生成 manifest（不跑 Zero123）
  python tools/run_rotation_pipeline.py \
      --input_dir /path/to/images \
      --output_dir /path/to/output \
      --skip_zero123
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.angle_predictor import ElevationRegressorNet as PipelineAnglePredictorNet
from modules.trellis2.trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

THREESTUDIO_DIR = PROJECT_ROOT / "modules" / "threestudio"
DEFAULT_CONFIG = str(THREESTUDIO_DIR / "configs" / "zero123-generate.yaml")
DEFAULT_WEIGHTS = "/data/home/2024120101018/linzhuohang/3DGen/data/angle_predictor.bin"


# ---------------------------------------------------------------------------
# 角度预测
# ---------------------------------------------------------------------------

def load_image(path: str, size: int = 512) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def build_predictor(weights_path: str, device: str = "cuda"):
    dino_extractor = DinoV3FeatureExtractor(model_name="dinov3", image_size=512)
    dino_extractor.model.eval()
    for p in dino_extractor.model.parameters():
        p.requires_grad = False

    model = PipelineAnglePredictorNet(dino_extractor)
    ckpt = torch.load(weights_path, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = {k.replace("model.", ""): v for k, v in ckpt.items()}

    cur = model.state_dict()
    casted = {}
    for k, v in ckpt.items():
        if k not in cur:
            casted[k] = v
            continue
        if torch.is_tensor(v):
            casted[k] = v.to(device=cur[k].device, dtype=cur[k].dtype)
        else:
            casted[k] = v
    model.load_state_dict(casted, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    model.register_buffer("img_mean", mean, persistent=False)
    model.register_buffer("img_std", std, persistent=False)

    return model, dino_extractor


@torch.no_grad()
def predict_elevation(model, dino_extractor, img_tensor: torch.Tensor, device: str = "cuda"):
    img_tensor = F.interpolate(img_tensor, size=(512, 512), mode="bicubic", align_corners=False)
    x_cast = img_tensor.to(device=device, dtype=next(dino_extractor.model.parameters()).dtype)
    x_norm = (x_cast - model.img_mean) / model.img_std
    tokens = dino_extractor.extract_features(x_norm)
    raw = model(x_cast, dino_tokens=tokens).float().view(-1)
    if raw.abs().max() <= 1.5:
        return (raw * 90.0).item()
    return raw.item()


def step1_predict_angles(input_dir: str, weights_path: str, device: str) -> dict:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    image_files = sorted([
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in exts and p.is_file()
    ])
    if not image_files:
        raise RuntimeError(f"No images found in {input_dir}")

    print(f"[Step 1] Predicting elevation for {len(image_files)} images...")
    model, dino_extractor = build_predictor(weights_path, device)
    model = model.to(device)
    dino_extractor.model = dino_extractor.model.to(device)

    results = {}
    for i, img_path in enumerate(image_files):
        img = load_image(str(img_path))
        elev = predict_elevation(model, dino_extractor, img, device)
        results[img_path.name] = elev
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(image_files)}] {img_path.name} -> {elev:.2f}°")

    print(f"[Step 1] Done. Elevation range: {min(results.values()):.2f}° ~ {max(results.values()):.2f}°")
    return results


# ---------------------------------------------------------------------------
# 构建 manifest
# ---------------------------------------------------------------------------

def step2_build_manifest(input_dir: str, angles: dict, output_dir: str, target_elev: float) -> str:
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Step 2] Building manifest -> {manifest_path}")

    records = []
    for fname, pred_elev in angles.items():
        img_abs = str(Path(input_dir).resolve() / fname)
        records.append({
            "sample_id": Path(fname).stem,
            "ref_img": img_abs,
            "tgt_img": img_abs,
            "ref_yaw": 0,
            "ref_elev": pred_elev,
            "tgt_yaw": 0,
            "tgt_elev": target_elev,
        })

    with open(manifest_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[Step 2] Done. Wrote {len(records)} records.")
    return manifest_path


# ---------------------------------------------------------------------------
# Zero123 生成
# ---------------------------------------------------------------------------

def step3_run_zero123(manifest_path: str, config_path: str, output_dir: str, gpu: str = "0") -> str:
    print(f"[Step 3] Running Zero123 validation...")

    cmd = [
        sys.executable, "launch.py",
        "--config", config_path,
        "--validate",
        "--gpu", gpu,
        f"data.val_manifest={manifest_path}",
        "tag=rotation-pipeline",
        f"exp_root_dir={output_dir}",
    ]

    print(f"  CWD: {THREESTUDIO_DIR}")
    print(f"  CMD: {' '.join(cmd)}")

    result = subprocess.run(
        cmd, cwd=str(THREESTUDIO_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in result.stdout.splitlines():
        print(f"  [zero123] {line}")

    if result.returncode != 0:
        print(f"[Step 3] WARNING: non-zero exit code ({result.returncode}), continuing...")

    zero123_outputs = sorted(
        Path(output_dir).glob("zero123-ft/rotation-pipeline*/save/it*-val-noise2img"),
        key=os.path.getmtime, reverse=True,
    )
    if not zero123_outputs:
        raise RuntimeError(
            f"Zero123 output not found under {output_dir}/zero123-ft/rotation-pipeline*/save/"
        )
    zero_dir = str(zero123_outputs[0])
    print(f"[Step 3] Zero123 output: {zero_dir}")
    return zero_dir


# ---------------------------------------------------------------------------
# 后处理：resize 1024 + crop 右 1/3 + meanshift → 直接写入 output_dir
# ---------------------------------------------------------------------------

def step4_postprocess(zero_dir: str, output_dir: str, workers: int = 50, sp: int = 40, sr: int = 13):
    from tools.eval_docker.resize_img import resize_images_in_dir

    tmp_dir = os.path.join(output_dir, ".tmp_1024")
    print(f"[Step 4a] Resize + crop right 1/3 -> {tmp_dir}")
    os.makedirs(tmp_dir, exist_ok=True)
    resize_images_in_dir(zero_dir, tmp_dir, target_size=(1024, 1024))

    print(f"[Step 4b] Meanshift smoothing -> {output_dir}")
    subprocess.run([
        sys.executable, "-m", "tools.eval_docker.smooth_img",
        "-i", tmp_dir,
        "-o", output_dir,
        "--mode", "meanshift",
        "--meanshift-name", ".tmp_meanshift",
        "--sp", str(sp),
        "--sr", str(sr),
        "-w", str(workers),
    ], cwd=str(PROJECT_ROOT), check=True)

    # 直接移动到 output_dir
    tmp_meanshift = os.path.join(output_dir, ".tmp_meanshift")
    for fname in os.listdir(tmp_meanshift):
        shutil.move(os.path.join(tmp_meanshift, fname), os.path.join(output_dir, fname))
    shutil.rmtree(tmp_meanshift)

    # 清理临时目录
    shutil.rmtree(tmp_dir)
    print(f"[Step 4] Done. Final images in: {output_dir}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="端到端图片旋转 pipeline")
    parser.add_argument("--input_dir", required=True, help="输入图片目录")
    parser.add_argument("--output_dir", required=True, help="输出目录（最终结果直接放在这里）")
    parser.add_argument("--target_elev", type=float, default=35.0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--sp", type=int, default=40)
    parser.add_argument("--sr", type=int, default=13)
    parser.add_argument("--skip_zero123", action="store_true")
    parser.add_argument("--skip_postprocess", action="store_true")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    input_dir = os.path.abspath(args.input_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Rotation Pipeline")
    print(f"  Input:       {input_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  Target elev: {args.target_elev}°")
    print(f"  GPU:         {args.gpu}")
    print("=" * 60)

    # Step 1 + 2: 预测角度 + 构建 manifest
    t0 = time.time()
    angles = step1_predict_angles(input_dir, args.weights, args.device)
    manifest_path = step2_build_manifest(input_dir, angles, output_dir, args.target_elev)

    if args.skip_zero123:
        print(f"\n[SKIP] Zero123 not run. Manifest at: {manifest_path}")
        print(f"Manual: cd {THREESTUDIO_DIR} && python launch.py --config {args.config} "
              f"--validate --gpu {args.gpu} data.val_manifest={manifest_path}")
        return

    # Step 3: Zero123
    zero_dir = step3_run_zero123(manifest_path, args.config, output_dir, args.gpu)

    if args.skip_postprocess:
        print(f"\n[SKIP] Postprocess not run. Raw output at: {zero_dir}")
        return

    # Step 4: 后处理 → output_dir
    step4_postprocess(zero_dir, output_dir, args.workers, args.sp, args.sr)

    # 清理 Zero123 中间产物（threestudio outputs）
    threestudio_outputs = os.path.join(output_dir, "zero123-ft")
    if os.path.isdir(threestudio_outputs):
        shutil.rmtree(threestudio_outputs)
        print(f"[Cleanup] Removed intermediate: {threestudio_outputs}")

    print("=" * 60)
    print(f"Done! Total time: {time.time() - t0:.1f}s")
    print(f"  Final output: {output_dir}")
    print(f"  Manifest:     {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
