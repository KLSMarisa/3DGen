#!/bin/bash
gpus=1
version=11
#python -m flux_modules.OAFluxKontextPipeline2 --version $version
#python model_converter.py --step 6500 --version $version
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
    --version $version \
    --resume '/mnt/hdd3/linzhuohang/3DGen/ckptv'$version'/checkpoints/flux_training-step=1000.ckpt'