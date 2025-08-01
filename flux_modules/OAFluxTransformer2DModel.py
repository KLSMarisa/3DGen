import torch.nn as nn
from diffusers import FluxTransformer2DModel
from flux_modules.OAFluxBlock import OAFluxTransformerBlock

class OAFluxTransformer2DModel(FluxTransformer2DModel):

    def __init__(self, *args, **kwargs):
        # 1. 首先，完整地调用父类的构造函数
        # 这会创建出所有原生模块，包括一个由原生 FluxTransformerBlock 组成的列表
        super().__init__(*args, **kwargs)
        attention_head_dim = kwargs.get('attention_head_dim',128)
        # 2. 现在，我们将替换核心的 transformer_blocks
        # 我们从 self.config 中获取所有必要的参数来构建我们的自定义 Block
        
        # 从配置中获取Block所需的参数
        num_layers = self.config.num_layers
        num_attention_heads = self.config.num_attention_heads
        
        # 3. 创建一个由我们自定义 Block 组成的新 ModuleList
        self.transformer_blocks = nn.ModuleList(
            [
                OAFluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim
                )
                for _ in range(num_layers)
            ]
        )
