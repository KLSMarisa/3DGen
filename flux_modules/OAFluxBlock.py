from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
from einops import rearrange
from .OrthogonalAttention import OrthogonalAttentionModule
import torch
import torch.nn as nn
import math
import deepspeed
from typing import Any, Dict, Optional, Tuple, Union
from modules.StableDiffusion.attention import MemoryEfficientCrossAttention_tri
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    FluxAttnProcessor2_0,
    FluxAttnProcessor2_0_NPU,
    FusedFluxAttnProcessor2_0,
)
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class OAFluxTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int,scale_factor, **kwargs):
        super().__init__()

        self.enable_gate_control = True
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # NOTE: 这里的输入维度是 dim
        self.time_liner = FeedForward(dim=dim, dim_out=dim*4,mult=1/scale_factor)
        
        # --- [修改] 新增一个可学习的位置嵌入，用于区分组内的3个样本 ---
        self.position_emb = nn.Embedding(3, dim)
        # --- 修改结束 ---

        self.ff =  FeedForward(dim=dim, dim_out=dim,mult=min(1,2.0/scale_factor))
        
        self.ortho_attn = MemoryEfficientCrossAttention_tri(
            query_dim=dim,
            dim_head=attention_head_dim,
        )
        self.use_checkpoint = False
        self.enable_oa = True

    def forward(self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs = None):
        if(self.use_checkpoint):
            return deepspeed.checkpointing.checkpoint(
                self._forward,
                hidden_states,
                temb,
                image_rotary_emb
            )
        return self._forward(hidden_states, temb, image_rotary_emb)

    def debug_print(self, message: str, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor):
        pass

    def _forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None 
    ):
        if self.enable_gate_control:
            
            
            # 1. 准备输入和维度信息
            B = hidden_states.shape[0] // 3
            C = hidden_states.shape[2]
            
            # 2. 生成独一无二的条件嵌入
            # 假设传入的temb是(3*B, C)，且每3个都相同，我们先取出去重的共享temb
            #print('temb: ', temb.shape)
            temb_shared = temb[::3] # Shape: (B, C)
            #print('temb shared: ', temb_shared.shape)
            # 获取位置嵌入
            pos_emb = self.position_emb.weight # Shape: (3, C)
            
            # 通过广播将共享temb和位置嵌入相加
            temb_expanded = temb_shared.unsqueeze(1)      # Shape: (B, 1, C)
            pos_emb_expanded = pos_emb.unsqueeze(0) # Shape: (1, 3, C)
            combined_emb = temb_expanded + pos_emb_expanded # Shape: (B, 3, C)
            
            # 展平以送入MLP
            combined_emb_flat = combined_emb.view(B * 3, C) # Shape: (3*B, C)
            
            # 使用包含位置信息的嵌入生成调制参数
            modulation = self.time_liner(combined_emb_flat)
            gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(4, dim=-1)

            # 3. 正交注意力模块
            norm_hidden_states_flat = self.norm1(hidden_states)
            attn_output_flat = self.ortho_attn(norm_hidden_states_flat[:,:int(hidden_states.shape[1]/2), :], image_rotary_emb)

            # 4. 分组并应用Attention门控
            latent_size = int(math.sqrt(hidden_states.shape[1]/2))
            
            # 正确地对hidden_states和attn_output进行分组
            hidden_states_grouped = hidden_states.view(B, 3, *hidden_states.shape[1:])
            attn_output_grouped = attn_output_flat.view(B, 3, *attn_output_flat.shape[1:])
            
            # 正确地对门控参数进行分组
            gate_msa_grouped = gate_msa.view(B, 3, -1) # Shape: (B, 3, C)
            
            # 使用unsqueeze进行正确的广播来应用门控
            gated_attn = attn_output_grouped * gate_msa_grouped.unsqueeze(2) # unsqueeze adds seq_len dim for broadcasting
            
            # 残差连接
            ortho_combined = hidden_states_grouped[:,:, :latent_size*latent_size, :] + gated_attn
            hidden_states_grouped = torch.cat((ortho_combined, hidden_states_grouped[:, :, latent_size*latent_size:, :]), dim=2)
            
            # 5. 前馈网络模块 (在分组状态下进行)
            norm_hidden_states_grouped = self.norm2(hidden_states_grouped)
            
            # 正确地对scale和shift参数进行分组
            scale_mlp_grouped = scale_mlp.view(B, 3, -1) # Shape: (B, 3, C)
            shift_mlp_grouped = shift_mlp.view(B, 3, -1) # Shape: (B, 3, C)

            # 使用unsqueeze进行正确的广播来实现AdaLN
            scale = scale_mlp_grouped.unsqueeze(2)
            shift = shift_mlp_grouped.unsqueeze(2)
            norm_hidden_states_modulated = norm_hidden_states_grouped * (1 + scale) + shift

            # 计算FFN输出
            ff_output = self.ff(norm_hidden_states_modulated)
            
            # 应用FFN门控
            gate_mlp_grouped = gate_mlp.view(B, 3, -1) # Shape: (B, 3, C)
            ff_output = ff_output * gate_mlp_grouped.unsqueeze(2)
            
            # 第二个残差连接
            hidden_states = hidden_states_grouped + ff_output
            
            # 6. 还原形状
            hidden_states = hidden_states.view(B * 3, *hidden_states.shape[2:])

        else:
            # 原始的else逻辑保持不变
            norm_hidden_states = self.norm1(hidden_states)
            # NOTE: The original logic here assumes time_liner output can be chunked into 2.
            # You might need to adjust the output dimension of time_liner if enable_gate_control can be toggled.
            # For now, assuming a different MLP or configuration for this case.
            # A simple fix could be to have another MLP for this branch.
            # For this example, I'll assume self.time_liner's output is C*2 in this case.
            scale,shift = self.time_liner(temb).chunk(2, dim=-1) # This might fail if time_liner output is C*4
            attn_input = norm_hidden_states[:,:int(hidden_states.shape[1]/2), :] * (1 + scale[:, None, :]) + shift[:, None, :]
            attn_output = self.ortho_attn(attn_input,image_rotary_emb)
            latent_size = int(math.sqrt(hidden_states.shape[1]/2))
            ortho_combined =hidden_states[:, :latent_size*latent_size, :] + attn_output
            hidden_states = torch.cat((ortho_combined, hidden_states[:, latent_size*latent_size:, :]), dim=1)
            # c. Feed-forward network
            norm_hidden_states = self.norm2(hidden_states)
            ff_output = self.ff(norm_hidden_states)
            hidden_states = hidden_states + ff_output
            
        return hidden_states