#!/bin/bash
gpus=7
version=5
python -m torch.distributed.launch \
    --master_port 114  \
    --nproc_per_node $gpus \
    --use_env main.py \
    --exp flux \
    --config "configs/base.yaml" \
    --mode train \
    --batch_size 1 \
    --name flux_training  \
    --savedir "/mnt/hdd3/linzhuohang/3DGen/ckptv"$version \
    --devices $gpus \
    --batch_size 1 \
    --nodes 1 \
    --init_step 0 \
    --resume /mnt/hdd3/linzhuohang/3DGen/ckptv4/checkpoints/flux_training-step=5000.ckpt