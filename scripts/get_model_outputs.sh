#!/bin/bash
MAX_STEP=5000
export CUDA_VISIBLE_DEVICES=7
for STEP in $(seq 4000 500 $MAX_STEP); do
    echo "Processing step: $STEP"
    #python /home/linzhuohang/3DGen/model_converter.py --step $STEP
    python -m torch.distributed.launch --master_port 99  --nproc_per_node 1 --use_env main.py  --init_step $STEP --exp flux --config configs/base.yaml --mode inference --batch_size 1 --name flux_training  --savedir '/mnt/hdd3/linzhuohang/3DGen/ckpt' --device 1 --nodes 1 --resume '/mnt/hdd3/linzhuohang/3DGen/ckptv4/checkpoints/flux_training-step='$STEP'.ckpt'
done
