from flux_modules import OAFluxKontextPipeline2 as OAFluxKontextPipeline
from flux_modules import OAFluxTransformer2DModel
import torch
from flux_modules import OAFluxKontextPipeline2 as OAFluxKontextPipeline
from flux_modules import OAFluxTransformer2DModel
import torch

@torch.no_grad()
def inspect_transformer_blocks(transformer):
    """
    Inspect each block in the transformer and print the overall absolute mean 
    for ff.weight, ff.bias, oa_attn.to_q.weight, and oa_attn.to_out.0.bias.
    """
    if not hasattr(transformer, 'oa_transformer_blocks'):
        print("The provided transformer does not have 'oa_transformer_blocks'.")
        return

    # Lists to store the abs_mean of each parameter tensor across all blocks
    attn_bias_means = []
    attn_q_weight_means = []
    ff_weight_means = []
    ff_bias_means = []

    for i, block in enumerate(transformer.oa_transformer_blocks):
        # Inspect ortho_attn if it exists
        if hasattr(block, 'ortho_attn'):
            for name, param in block.ortho_attn.named_parameters():
                if name == "to_out.0.bias":
                    attn_bias_means.append(param.abs().mean().cpu().item())
                elif name == "to_q.weight":
                    attn_q_weight_means.append(param.abs().mean().cpu().item())

        # Inspect ff (FeedForward) if it exists
        if hasattr(block, 'ff'):
            for name, param in block.ff.named_parameters():
                mean = param.abs().mean().cpu().item()
                if 'weight' in name:
                    ff_weight_means.append(mean)
                elif 'bias' in name:
                    ff_bias_means.append(mean)
    
    # Initialize return values
    total_attn_bias_mean = 0
    total_attn_q_weight_mean = 0
    total_ff_weight_mean = 0
    total_ff_bias_mean = 0

    print("\n--- Overall Absolute Mean Statistics ---")

    # Calculate and print overall statistics
    if attn_bias_means:
        total_attn_bias_mean = sum(attn_bias_means) / len(attn_bias_means)
        print(f"oa_attn.to_out.0.bias: abs_mean={total_attn_bias_mean:.12f}")

    if attn_q_weight_means:
        total_attn_q_weight_mean = sum(attn_q_weight_means) / len(attn_q_weight_means)
        print(f"oa_attn.to_q.weight  : abs_mean={total_attn_q_weight_mean:.12f}")
    
    if ff_weight_means:
        total_ff_weight_mean = sum(ff_weight_means) / len(ff_weight_means)
        print(f"ff.weight            : abs_mean={total_ff_weight_mean:.12f}")
    
    if ff_bias_means:
        total_ff_bias_mean = sum(ff_bias_means) / len(ff_bias_means)
        print(f"ff.bias              : abs_mean={total_ff_bias_mean:.12f}")
        
    return total_attn_bias_mean, total_attn_q_weight_mean, total_ff_weight_mean, total_ff_bias_mean
if __name__ == "__main__":
    init_step = 500
    pipe = OAFluxKontextPipeline.get_pipeline(ckpt_path= f'/mnt/hdd3/linzhuohang/3DGen/ckptv2/safetensors/{init_step}',Train = True)
    print("--- 模型加载成功 ---")
    inspect_transformer_blocks(pipe)
    inspect_transformer_blocks(pipe)
