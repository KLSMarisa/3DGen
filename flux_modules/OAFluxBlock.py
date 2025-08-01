from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
from .OrthogonalAttention import OrthogonalAttentionModule
import torch
import torch.nn as nn
import math
import deepspeed

# 假设您的 OrthogonalAttentionModule 模块已在别处定义
# from your_project.modules import OrthogonalAttentionModule 

class OAFluxTransformerBlock(FluxTransformerBlock):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, **kwargs):
        # 父类 __init__ 会创建 self.attn, self.ff, self.norm1, self.norm2 等原生模块
        super().__init__(dim=dim, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim)

        # 添加我们自己的正交注意力模块
        self.ortho_attn = OrthogonalAttentionModule(dim, num_attention_heads)
        
        # 为了 reshape，我们需要知道 latent 的边长
        # 注意：这需要根据模型的实际配置来确定。对于FLUX，通常是固定的。
        self.latent_size = 56 # 示例值, 请根据您的模型配置调整
        self.num_patches = self.latent_size * self.latent_size

    def forward(self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        **kwargs):
        #return deepspeed.checkpointing.checkpoint(self._forward, hidden_states,encoder_hidden_states,temb,**kwargs).requires_grad_(True)
        return self._forward(hidden_states, encoder_hidden_states, temb, **kwargs)
    def _forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        **kwargs # 使用 **kwargs 捕获并传递所有其他参数
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # --- 1. 对两个流进行 AdaLayerNormZero (与原生代码一致) ---
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        
        # --- 2. 执行标准注意力 ---
        # 这一步同时计算图像的自注意力和与文本的交叉注意力
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            **kwargs, # 传递 image_rotary_emb 等参数
        )
        
        # 分离出图像和文本的注意力结果
        attn_output = attention_outputs[0]
        context_attn_output = attention_outputs[1]

        # --- 3. 关键：在图像流上插入我们的正交注意力 ---
        # a. 准备输入
        B_times_N, seq_len, C = hidden_states.shape
        seq_len = seq_len // 2
        sqrt_seq_len = math.sqrt(seq_len)
        if sqrt_seq_len != int(sqrt_seq_len):
            raise ValueError(f"Sequence length {seq_len} is not a perfect square. Cannot determine latent_size.")
        
        latent_size = int(sqrt_seq_len)
        N = 3
        B = B_times_N // N
        S = latent_size
        
        ortho_input = hidden_states[:,:latent_size*latent_size,:].reshape(B, N, S, S, C)
        
        # b. 执行正交注意力
        p_xy, p_xz, p_yz = ortho_input.unbind(dim=1)
        out_xy, out_xz, out_yz = self.ortho_attn(p_xy, p_xz, p_yz)
        
        # c. 将结果 reshape 回序列格式
        ortho_output_spatial = torch.stack([out_xy, out_xz, out_yz], dim=1)
        ortho_output_seq = ortho_output_spatial.reshape(B_times_N, seq_len, C)

        # --- 4. 完成 hidden_states (图像流) 的更新 ---
        # a. 添加标准注意力的残差
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        # b. 再添加我们的正交注意力的残差
        hidden_states[:,:latent_size*latent_size,:] = hidden_states[:,:latent_size*latent_size,:] + ortho_output_seq
        
        # c. 执行前馈网络
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        # --- 5. 完成 encoder_hidden_states (文本流) 的更新 ---
        # (这部分逻辑与原生代码完全相同，直接复用)
        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        # --- 6. 返回符合原生契约的元组 ---
        return encoder_hidden_states, hidden_states