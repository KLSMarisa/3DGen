# CUDA_LAUNCH_BLOCKING=1 
# python main.py --nodes 1 --devices 8 --stage 1 --name spatial_finetune --dataset text2obj --batch_size 4
# Xvfb :99 -screen 0 1024x768x24 &
# export DISPLAY=:99 &
# python -m torch.distributed.launch   --nproc_per_node 8 --use_env main.py --stage 3 --name stage2_sd --savedir '/mnt/hdd1/caixiao/3dgen_sd2.1_stage3_fintune_ctx' --dataset text2obj --batch_size 4 --nodes 1  --resume '/mnt/hdd1/caixiao/3dgen_sd2.1_stage3_fintune_ctx/checkpoints/stage2_sd-step=40000.ckpt'
python -m torch.distributed.launch   --nproc_per_node 8 --use_env main.py --stage 3 --name turbo --savedir '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/turbo_reflow2' --dataset text2obj --batch_size 2 --nodes 1  --resume '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/turbo_reflow2/checkpoints/turbo-step=1000.ckpt'
# setsid nohup python -m torch.distributed.launch   --nproc_per_node 8 --use_env main.py --stage 1 --name turbo --dataset text2obj --batch_size 4 --nodes 1  > output.log 2>&1
# python main.py --stage 3 --exp diffusion --name turbo --config configs/base.yaml --savedir '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/turbo' --dataset text2obj --batch_size 2 --nodes 1 --devices 1 