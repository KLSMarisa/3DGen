from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
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
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int,enable_gate_control=False, **kwargs):
        # 父类 __init__ 会创建 self.attn, self.ff, self.norm1, self.norm2 等原生模块
        super().__init__()


        self.enable_gate_control = enable_gate_control
        self.norm1 = AdaLayerNormZero(dim) if enable_gate_control else nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        #self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # 添加我们自己的正交注意力模块
        #self.ortho_attn = OrthogonalAttentionModule(dim, num_attention_heads)
        self.ortho_attn = MemoryEfficientCrossAttention_tri(
            query_dim=dim,
            #context_dim=dim,
            dim_head = attention_head_dim,

        )
        self.use_checkpoint = False
        self.enable_oa = True

    def forward(self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs = None):
        # The main forward call now points to our modified _forward method
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
        #if torch.distributed.get_rank() == 0:  # Only print if local_rank is 0
        #    print(f"{message}: encoder:({encoder_hidden_states.min().cpu().item(), encoder_hidden_states.max().cpu().item()}), "
        #          f"hidden:({hidden_states.min().cpu().item(), hidden_states.max().cpu().item()})")

    def _forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None 
    ):
        """
        This part of the forward pass is wrapped in a checkpoint.
        It computes everything up to the creation of ortho_input.
        """
        # --- 1. AdaLayerNormZero for both streams ---
        
        #norm_hidden_states = self.norm1(hidden_states)
        #print(hidden_states.shape)
        if self.enable_gate_control:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
            
            attn_output = self.ortho_attn(norm_hidden_states[:,:int(hidden_states.shape[1]/2), :],image_rotary_emb)
            latent_size = int(math.sqrt(hidden_states.shape[1]/2))
            ortho_combined =hidden_states[:, :latent_size*latent_size, :] + attn_output*gate_msa[:latent_size*latent_size].unsqueeze(1)
            hidden_states = torch.cat((ortho_combined, hidden_states[:, latent_size*latent_size:, :]), dim=1)
            #self.debug_print('leaving oa_attn', hidden_states)
            # c. Feed-forward network
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp.unsqueeze(1) * ff_output
            hidden_states = hidden_states +   ff_output
        else:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.ortho_attn(norm_hidden_states[:,:int(hidden_states.shape[1]/2), :],image_rotary_emb)
            latent_size = int(math.sqrt(hidden_states.shape[1]/2))
            ortho_combined =hidden_states[:, :latent_size*latent_size, :] + attn_output
            hidden_states = torch.cat((ortho_combined, hidden_states[:, latent_size*latent_size:, :]), dim=1)
            #self.debug_print('leaving oa_attn', hidden_states)
            # c. Feed-forward network
            norm_hidden_states = self.norm2(hidden_states)
            ff_output = self.ff(norm_hidden_states)
            hidden_states = hidden_states + ff_output
        return hidden_states