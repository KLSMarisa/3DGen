from flux_modules import OAFluxKontextPipeline2 as OAFluxKontextPipeline
from flux_modules import OAFluxTransformer2DModel
import torch

@torch.no_grad()
def inspect_transformer_blocks(transformer):
    """
    Inspect weights and return a dict of abs_mean values.
    """
    results = {}

    if not hasattr(transformer, 'oa_transformer_blocks'):
        print("The provided transformer does not have 'oa_transformer_blocks'.")
        return results

    attn_bias_means = []
    attn_q_weight_means = []
    ff_weight_means = []
    ff_bias_means = []
    time_liner_weighted_sum = 0.0
    time_liner_param_count = 0

    for block in transformer.oa_transformer_blocks:
        # ortho_attn
        if hasattr(block, 'ortho_attn'):
            for name, param in block.ortho_attn.named_parameters():
                if name == "to_out.0.bias":
                    attn_bias_means.append(param.abs().mean().cpu().item())
                elif name == "to_q.weight":
                    attn_q_weight_means.append(param.abs().mean().cpu().item())

        # feed-forward
        if hasattr(block, 'ff'):
            for name, param in block.ff.named_parameters():
                mean = param.abs().mean().cpu().item()
                if 'weight' in name:
                    ff_weight_means.append(mean)
                elif 'bias' in name:
                    ff_bias_means.append(mean)

        # time_liner
        if hasattr(block, 'time_liner'):
            for _, param in block.time_liner.named_parameters():
                abs_mean = param.abs().mean().cpu().item()
                numel = param.numel()
                time_liner_weighted_sum += abs_mean * numel
                time_liner_param_count += numel

    # 汇总
    if attn_bias_means:
        results["weight/oa_attn.to_out.0.bias"] = sum(attn_bias_means) / len(attn_bias_means)
    if attn_q_weight_means:
        results["weight/oa_attn.to_q.weight"] = sum(attn_q_weight_means) / len(attn_q_weight_means)
    if ff_weight_means:
        results["weight/ff.weight"] = sum(ff_weight_means) / len(ff_weight_means)
    if ff_bias_means:
        results["weight/ff.bias"] = sum(ff_bias_means) / len(ff_bias_means)
    if time_liner_param_count > 0:
        results["weight/time_liner"] = time_liner_weighted_sum / time_liner_param_count

    # norm_out.linear
    if hasattr(transformer, "norm_out") and hasattr(transformer.norm_out, "linear"):
        if hasattr(transformer.norm_out.linear, "weight"):
            results["weight/norm_out.linear.weight"] = transformer.norm_out.linear.weight.abs().mean().cpu().item()
        if hasattr(transformer.norm_out.linear, "bias") and transformer.norm_out.linear.bias is not None:
            results["weight/norm_out.linear.bias"] = transformer.norm_out.linear.bias.abs().mean().cpu().item()

    # proj_out
    if hasattr(transformer, "proj_out"):
        if isinstance(transformer.proj_out, torch.nn.Parameter):
            results["weight/proj_out"] = transformer.proj_out.abs().mean().cpu().item()
        elif hasattr(transformer.proj_out, "weight"):
            results["weight/proj_out"] = transformer.proj_out.weight.abs().mean().cpu().item()

    return results


@torch.no_grad()
def inspect_transformer_grads(transformer):
    """
    Inspect gradients and return a dict of abs_mean values.
    """
    results = {}

    if not hasattr(transformer, 'oa_transformer_blocks'):
        print("The provided transformer does not have 'oa_transformer_blocks'.")
        return results

    attn_bias_grads = []
    attn_q_weight_grads = []
    ff_weight_grads = []
    ff_bias_grads = []
    time_liner_weighted_sum = 0.0
    time_liner_param_count = 0

    for block in transformer.oa_transformer_blocks:
        if hasattr(block, 'ortho_attn'):
            for name, param in block.ortho_attn.named_parameters():
                if param.grad is None:
                    continue
                if name == "to_out.0.bias":
                    attn_bias_grads.append(param.grad.abs().mean().cpu().item())
                elif name == "to_q.weight":
                    attn_q_weight_grads.append(param.grad.abs().mean().cpu().item())

        if hasattr(block, 'ff'):
            for name, param in block.ff.named_parameters():
                if param.grad is None:
                    continue
                mean = param.grad.abs().mean().cpu().item()
                if 'weight' in name:
                    ff_weight_grads.append(mean)
                elif 'bias' in name:
                    ff_bias_grads.append(mean)

        if hasattr(block, 'time_liner'):
            for _, param in block.time_liner.named_parameters():
                if param.grad is None:
                    continue
                abs_mean = param.grad.abs().mean().cpu().item()
                numel = param.numel()
                time_liner_weighted_sum += abs_mean * numel
                time_liner_param_count += numel

    if attn_bias_grads:
        results["grad/oa_attn.to_out.0.bias"] = sum(attn_bias_grads) / len(attn_bias_grads)
    if attn_q_weight_grads:
        results["grad/oa_attn.to_q.weight"] = sum(attn_q_weight_grads) / len(attn_q_weight_grads)
    if ff_weight_grads:
        results["grad/ff.weight"] = sum(ff_weight_grads) / len(ff_weight_grads)
    if ff_bias_grads:
        results["grad/ff.bias"] = sum(ff_bias_grads) / len(ff_bias_grads)
    if time_liner_param_count > 0:
        results["grad/time_liner"] = time_liner_weighted_sum / time_liner_param_count

    if hasattr(transformer, "norm_out") and hasattr(transformer.norm_out, "linear"):
        if hasattr(transformer.norm_out.linear, "weight") and transformer.norm_out.linear.weight.grad is not None:
            results["grad/norm_out.linear.weight"] = transformer.norm_out.linear.weight.grad.abs().mean().cpu().item()
        if hasattr(transformer.norm_out.linear, "bias") and transformer.norm_out.linear.bias is not None and transformer.norm_out.linear.bias.grad is not None:
            results["grad/norm_out.linear.bias"] = transformer.norm_out.linear.bias.grad.abs().mean().cpu().item()

    if hasattr(transformer, "proj_out"):
        if isinstance(transformer.proj_out, torch.nn.Parameter):
            if transformer.proj_out.grad is not None:
                results["grad/proj_out"] = transformer.proj_out.grad.abs().mean().cpu().item()
        elif hasattr(transformer.proj_out, "weight") and transformer.proj_out.weight.grad is not None:
            results["grad/proj_out"] = transformer.proj_out.weight.grad.abs().mean().cpu().item()

    return results


if __name__ == "__main__":
    init_step = 500
    pipe = OAFluxKontextPipeline.get_pipeline(
        ckpt_path=f'/mnt/hdd3/linzhuohang/3DGen/ckptv2/safetensors/{init_step}',
        Train=True
    )
    print("--- 模型加载成功 ---")
    inspect_transformer_blocks(pipe)
