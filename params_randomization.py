import torch
import os
from flux_modules import OAFluxTransformer2DModel
import accelerate 
def randomize_model(model,module_name_to_randomize, save_directory, max_shard_size="3GB"):
    # 禁用梯度计算，以加速并减少内存使用
    transformer_blocks = model.transformer_blocks
    with torch.no_grad():
        for i, block in enumerate(transformer_blocks):
            # 获取目标模块，这里假设是每个 block 下的 'attn' 或 'self_attn' 模块
            # 这是您提到的 "oa_attn" 最可能的对应物
            if hasattr(block, module_name_to_randomize):
                target_module = getattr(block, module_name_to_randomize)
                print(f"正在处理 Block {i} 中的 '{module_name_to_randomize}' 模块...")
                
                # 遍历该模块的所有参数 (权重和偏置)
                for name, param in target_module.named_parameters():
                    # 创建一个与原参数形状、类型、设备都相同的新随机张量
                    # torch.randn_like 会生成一个服从标准正态分布的随机张量
                    new_random_param = (torch.randn_like(param)/50).clamp(-0.5,0.5)
                    
                    # 使用 .copy_() 方法将新生成的随机数据复制到原参数的内存中
                    # 这是修改模型参数的正确方式
                    param.copy_(new_random_param)
                    print(f"  - 已随机化参数: {name} (大小: {list(param.shape)})")
            else:
                print(f"警告: Block {i} 中未找到名为 '{module_name_to_randomize}' 的模块。")

    print("\n所有指定模块的参数已随机化。")

    # --- 3. 分片保存修改后的模型 ---
    print(f"\n正在将修改后的模型分片保存到 '{save_directory}'...")
    print(f"每个分片最大为: {max_shard_size}")

    # 确保保存目录存在
    os.makedirs(save_directory, exist_ok=True)

    try:
        # 使用 save_pretrained 进行保存，并设置最大分片大小
        model.save_pretrained(
            save_directory,
            max_shard_size=max_shard_size
        )
        print("模型和 Tokenizer 保存成功！")
        print(f"文件已保存至: {os.path.abspath(save_directory)}")
    except Exception as e:
        print(f"保存模型失败: {e}")

if __name__ == "__main__":
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
    with accelerate.init_empty_weights():
        oa_transformer = OAFluxTransformer2DModel.OAFluxTransformer2DModel(**config)
    print('loading transformer weights...')
    #state_dict = load_file('/mnt/hdd3/linzhuohang/3DGen/oa_transfomer')
    #oa_transformer.load_state_dict(state_dict)
    ckpt_path = '/mnt/hdd3/linzhuohang/3DGen/ckptv2/safetensors/500'
    print('oa transformer loaded')
    if ckpt_path is not None and not ckpt_path.endswith('.bin'):
        oa_transformer = oa_transformer.from_pretrained(
            ckpt_path,
            device_map='cuda', 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
    randomize_model(oa_transformer, 'ortho_attn', '/mnt/hdd3/linzhuohang/3DGen/ckptv3/safetensors/0', max_shard_size="3GB")