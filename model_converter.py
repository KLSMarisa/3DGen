import argparse
from itertools import chain
import os
import sys
from typing import OrderedDict
import torch
import accelerate
from diffusers import FluxKontextPipeline 
from flux_modules import OAFluxTransformer2DModel
import subprocess  # 新增导入

step = 6000

def bin2safetensors(bin_file_path, output_dir):
    config = {
        'patch_size': 1,
        'in_channels': 64,
        'out_channels': None,
        'num_layers': 19,
        'num_single_layers': 38,
        'attention_head_dim': 128,
        'num_attention_heads': 24,
        'joint_attention_dim': 4096,
        'pooled_projection_dim': 768,
        'guidance_embeds': True,
        'axes_dims_rope': [16, 56, 56],
        '_class_name': 'FluxTransformer2DModel',
        '_diffusers_version': '0.34.0.dev0',
        '_name_or_path': '/mnt/hdd3/linzhuohang/3DGen/hf/hub/models--black-forest-labs--FLUX.1-Kontext-dev/snapshots/af58063aa431f4d2bbc11ae46f57451d4416a170/transformer'
    }
    
    oa_transformer = OAFluxTransformer2DModel.OAFluxTransformer2DModel.from_pretrained('/mnt/hdd3/linzhuohang/3DGen/hf/hub/models--black-forest-labs--FLUX.1-Kontext-dev/snapshots/af58063aa431f4d2bbc11ae46f57451d4416a170/transformer',strict=False,low_cpu_mem_usage=True)
    state_dict = torch.load(bin_file_path+f'/{step}.bin')
    #print(state_dict.keys())
    embedding = torch.load('/home/linzhuohang/3DGen/embedding.bin')
    embedding = {'time_text_embed.'+k: v for k, v in embedding.items()}
    #print(embedding.keys())
    new_weights = OrderedDict(chain(embedding.items(), state_dict.items()))
    new_weights = {k.replace("transformer.", ""): v for k, v in new_weights.items()}
    print(new_weights.keys())
    oa_transformer.load_state_dict(new_weights,assign= True,strict = False)
    print('model loaded')
    oa_transformer.save_pretrained(output_dir+f'/{step}/', safe_serialization=True,  max_shard_size = '3GB')

def ckpt2bin(ckpt_file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # 使用os.makedirs替代mkdir -p
    command = ['python', 'utils/zero_to_fp32.py', ckpt_file_path, f'{output_dir}/{step}.bin']
    subprocess.run(command, check=True)  # 使用subprocess.run确保命令执行完成

def convert_model(ckpt_file_path, bin_output_dir, safetensors_output_dir):
    if not os.path.exists(f'{bin_output_dir}/{step}.bin'):
        ckpt2bin(ckpt_file_path, bin_output_dir)
    print('bin file generated')
    bin2safetensors(bin_output_dir, safetensors_output_dir)
    print('safetensors file generated')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, help='Checkpoint step to convert')
    parser.add_argument('--version',type=int)
    args = parser.parse_args()
    step = args.step
    print('converting model: version{} step{}'.format(args.version,step))
    ckpt_file_path = f"/mnt/hdd3/linzhuohang/3DGen/ckptv{args.version}/checkpoints/flux_training-step={step}.ckpt"
    bin_output_dir = f"/mnt/hdd3/linzhuohang/3DGen/ckptv{args.version}/bin"
    safetensors_output_dir = f"/mnt/hdd3/linzhuohang/3DGen/ckptv{args.version}/safetensors"
    if os.path.exists(safetensors_output_dir+f'/{step}/'):
        print(f"Output directory {safetensors_output_dir}/{step} already exists. Skipping conversion.")
        sys.exit(0)
    convert_model(ckpt_file_path, bin_output_dir, safetensors_output_dir)
    #bin2safetensors(bin_output_dir, safetensors_output_dir)
    print('Conversion completed successfully.')