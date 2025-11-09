#!/bin/bash
gpus=2
version=17
#python -m flux_modules.OAFluxKontextPipeline2 --version $version
#python model_converter.py --step 16500 --version 16
python -m torch.distributed.launch \
    --master_port 115  \
    --nproc_per_node $gpus \
    --use_env main.py \
    --exp flux \
    --config "configs/base.yaml" \
    --mode train \
    --batch_size 1 \
    --name flux_training  \
    --devices $gpus \
    --batch_size 1 \
    --nodes 1 \
    --init_step 0 \
    #--version $version \
    #--resume latest