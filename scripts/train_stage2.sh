# CUDA_LAUNCH_BLOCKING=1 
python main.py --nodes $1 --devices $2 --stage 2 --name multiview_training --dataset webvid --batch_size 2
