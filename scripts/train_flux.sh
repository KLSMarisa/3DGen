#!/bin/bash
gpus=7
version=7
#python -m flux_modules.OAFluxKontextPipeline2 --version $version
python -m torch.distributed.launch \
    --master_port 115  \
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
    --version $version
    #--resume /mnt/hdd3/linzhuohang/3DGen/ckptv4/checkpoints/flux_training-step=5000.ckpt