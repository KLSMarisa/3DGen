#!/bin/bash
MAX_STEP = 6500
for STEP in $(seq 500 500 $MAX_STEP); do
    python /home/linzhuohang/3DGen/model_converter.py --step $STEP
    python -m torch.distributed.launch --master_port 101  --nproc_per_node 1 --use_env main.py  --init_step 0 --exp flux --config configs/base.yaml --mode inference --batch_size 1 --name flux_training  --savedir '/mnt/hdd3/linzhuohang/3DGen/ckpt' --device 1 --nodes 1
done
