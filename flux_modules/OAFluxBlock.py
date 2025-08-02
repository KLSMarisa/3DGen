from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
from .OrthogonalAttention import OrthogonalAttentionModule
import torch
import torch.nn as nn
import math
import deepspeed
from typing import Any, Dict, Optional, Tuple, Union

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
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None):
        # The main forward call now points to our modified _forward method
        return self._forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs)

    def _checkpointed_forward_part1(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        This part of the forward pass is wrapped in a checkpoint.
        It computes everything up to the creation of ortho_input.
        """
        # --- 1. AdaLayerNormZero for both streams ---
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        print('norm_hidden_states.shape:', norm_hidden_states.shape)
        print('norm_encoder_hidden_states.shape:', norm_encoder_hidden_states.shape)

        # --- 2. Standard Attention ---
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs
        )
        attn_output = attention_outputs[0]
        context_attn_output = attention_outputs[1]

        # --- 3. Prepare input for Orthogonal Attention ---
        B_times_N, seq_len, C = hidden_states.shape
        seq_len_half = seq_len // 2
        sqrt_seq_len = math.sqrt(seq_len_half)
        if sqrt_seq_len != int(sqrt_seq_len):
            raise ValueError(f"Sequence length {seq_len_half} is not a perfect square. Cannot determine latent_size.")
        
        latent_size = int(sqrt_seq_len)
        N = 3
        B = B_times_N // N
        S = latent_size
        
        ortho_input = hidden_states[:, :latent_size*latent_size, :].reshape(B, N, S, S, C)
        
        # Return all necessary tensors for the next stage
        return (
            hidden_states, encoder_hidden_states, ortho_input, gate_msa, attn_output, 
            context_attn_output, c_gate_msa, shift_mlp, scale_mlp, gate_mlp, 
            c_shift_mlp, c_scale_mlp, c_gate_mlp, B_times_N, seq_len_half, C, latent_size
        )

    def _forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # --- Execute the first part within the checkpoint wrapper ---
        (
            hidden_states, encoder_hidden_states, ortho_input, gate_msa, attn_output, 
            context_attn_output, c_gate_msa, shift_mlp, scale_mlp, gate_mlp, 
            c_shift_mlp, c_scale_mlp, c_gate_mlp, B_times_N, seq_len_half, C, latent_size
        ) = deepspeed.checkpointing.checkpoint(
            self._checkpointed_forward_part1,
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb,
            joint_attention_kwargs
        )
        
        # --- Continue with the rest of the logic using the results from the checkpointed part ---

        # b. Execute orthogonal attention
        p_xy, p_xz, p_yz = ortho_input.unbind(dim=1)
        out_xy, out_xz, out_yz = self.ortho_attn(p_xy, p_xz, p_yz)
        
        # c. Reshape result back to sequence format
        ortho_output_spatial = torch.stack([out_xy, out_xz, out_yz], dim=1)
        ortho_output_seq = ortho_output_spatial.reshape(B_times_N, seq_len_half, C)

        # --- 4. Complete hidden_states (image stream) update ---
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        hidden_states[:, :latent_size*latent_size, :] = hidden_states[:, :latent_size*latent_size, :] + ortho_output_seq
        
        # c. Feed-forward network
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        # --- 5. Complete encoder_hidden_states (text stream) update ---
        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        # --- 6. Return the final tuple ---
        return encoder_hidden_states, hidden_states