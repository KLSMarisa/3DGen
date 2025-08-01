import torch
from safetensors.torch import save_file
import os
from transformers.modeling_utils import shard_checkpoint
from safetensors.torch import save_file
original_model_path = "/home/linzhuohang/3DGen/flux_modules/oa_transformer.pth"
sharded_save_directory = "/mnt/hdd3/linzhuohang/3DGen/oa_transfomer" # 例如: ./oa_transformer_sharded
max_shard_size = "5GB" # 您可以根据需要调整这个值
# ---------------------

print(f"Loading original state_dict from {original_model_path}...")
# 1. 将原始模型加载到CPU内存
state_dict = torch.load(original_model_path, map_location="cpu")

print("Sharding the state_dict in memory...")
# 2. 使用 transformers 的工具函数进行分片
#    它会返回分片后的小 state_dict 以及索引文件内容
shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size)

# 3. 创建保存目录
os.makedirs(sharded_save_directory, exist_ok=True)

print(f"Saving {len(shards)} shards to {sharded_save_directory}...")
# 4. 遍历分片并使用 save_file 逐个保存
for shard_file_name, shard_tensors in shards.items():
    shard_file_path = os.path.join(sharded_save_directory, shard_file_name)
    # 调用 save_file 保存单个分片，这里不再需要 max_shard_size 参数
    save_file(shard_tensors, shard_file_path)
    print(f"  - Saved {shard_file_name}")
