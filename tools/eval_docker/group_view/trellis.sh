cd /data/home/2024120101018/linzhuohang/3DGen/3DGen_new

root=/data/home/2024120101018/linzhuohang/3DGen/data/testset/group_view
python -m tools.eval_docker.trellis_batch_to15_render \
 --input_dirs $root/HighElevation_0Roll $root/HighElevation_RandomRoll $root/LowElevation_0Roll $root/LowElevation_RandomRoll $root/Rotated \
   --output_root $root/trellis \
   --gpus 0 \
   --skip_exist

bash /data/home/2024120101018/linzhuohang/3DGen/3DGen_new/tools/eval_docker/compare_glb_trellis.sh
