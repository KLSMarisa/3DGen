import math
from typing import Any, Dict, Optional, Tuple, Union
import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from diffusers import FluxTransformer2DModel
from flux_modules.OAFluxBlock import OAFluxTransformerBlock
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
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

logger = logging.get_logger(__name__)

class OAFluxTransformer2DModel(FluxTransformer2DModel):

    def __init__(self, *args, **kwargs):
        # 1. 首先，完整地调用父类的构造函数
        # 这会创建出所有原生模块，包括一个由原生 FluxTransformerBlock 组成的列表
        #print('Initializing OAFluxTransformer2DModel with args:', args, 'and kwargs:', kwargs)
        kwargs['guidance_embeds'] = True  # 强制启用 guidance_embeds
        super().__init__(*args, **kwargs)
        attention_head_dim = kwargs.get('attention_head_dim',128)
        # 2. 现在，我们将替换核心的 transformer_blocks
        # 我们从 self.config 中获取所有必要的参数来构建我们的自定义 Block
        self.oa_emb = FluxPosEmbed(theta=10000,axes_dim=(64,64))
        # 从配置中获取Block所需的参数
        self.num_layers = self.config.num_layers
        self.num_single_layers = self.config.num_single_layers
        num_attention_heads = self.config.num_attention_heads
        
        # 3. 创建一个由我们自定义 Block 组成的新 ModuleList
        block_cnt = self.num_layers+self.num_single_layers
        scale_factor_list = [8]*block_cnt
        scale_factor_list[0]=scale_factor_list[-1]=scale_factor_list[1]=scale_factor_list[-2]=1
        scale_factor_list[2]=scale_factor_list[-3]=2
        scale_factor_list[3]=scale_factor_list[-4]=4
        
        enable_ca_list = [False]*block_cnt
        self.oa_transformer_blocks = nn.ModuleList(
            [
                OAFluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    scale_factor=scale_factor_list[i],
                    enable_ca=enable_ca_list[i],
                )
                for i in range(block_cnt)
            ]
        )
        self.gradient_checkpointing = True
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        def prepare_latent_image_ids( height, width, device, dtype):
            latent_image_ids = torch.zeros(height, width, 2)
            latent_image_ids[..., 0] = latent_image_ids[..., 0] + torch.arange(height)[:, None]
            latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(width)[None, :]

            latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

            latent_image_ids = latent_image_ids.reshape(
                latent_image_id_height * latent_image_id_width, latent_image_id_channels
            )

            return latent_image_ids.to(device=device, dtype=dtype)
        
        #print("hidden states shape:", hidden_states.shape)
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)
        sz =int(math.sqrt(hidden_states.shape[1]//2))
        oa_rotary_emb = self.oa_emb(prepare_latent_image_ids(sz,sz,hidden_states.device,hidden_states.dtype))
        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        for index_block, block in enumerate(self.transformer_blocks):
            #print('Processing transformer block:', index_block)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = deepspeed.checkpointing.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
            hidden_states = self.oa_transformer_blocks[index_block](hidden_states, temb, oa_rotary_emb)
            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = deepspeed.checkpointing.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
            hidden_states = self.oa_transformer_blocks[self.num_layers+index_block](hidden_states, temb, oa_rotary_emb) 
            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
