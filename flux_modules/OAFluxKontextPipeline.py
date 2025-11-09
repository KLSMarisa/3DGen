import torch
from diffusers import FluxKontextPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from typing import Union, List, Optional, Callable
from PIL import Image
import OAFluxTransformer2DModel
import env_set
import deepspeed
import json
# 假设我们已经定义好了 CustomFluxTransformerBlock_Ortho 和 OrthogonalAttentionModule
# 以及用于创建 CustomFluxTransformer2DModel_Ortho 的代码
# (这些核心模块代码与之前的回复相同)

class OAFluxKontextPipeline(FluxKontextPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        N: int = 3, # <--- MODIFICATION: 新增参数 N，默认为3
        image: Optional[Image.Image] = None, # <--- MODIFICATION: 为 I2I 显式添加 image 参数
        strength: float = 0.7, # <--- MODIFICATION: 为 I2I 添加 strength 参数
        num_inference_steps: int = 50,
        guidance_scale: float = 0.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ):
        # 1. Standard setup (检查输入, 获取尺寸)
        height = self.vae.config.sample_size
        width = self.vae.config.sample_size
        device = self._execution_device
        
        if num_images_per_prompt != 1:
            raise ValueError("num_images_per_prompt must be 1 for this custom pipeline.")
        if N != 3:
            print(f"Warning: This pipeline's orthogonal attention was designed for N=3, but got N={N}.")

        # 2. Encode prompt
        prompt_embeds = self.encode_prompt(prompt,prompt)[0]
        
        # <--- MODIFICATION: 将文本编码复制 N 份
        prompt_embeds = prompt_embeds.repeat(N, 1, 1)

        # 3. Prepare latents
        # <--- MODIFICATION: 分支处理 Text-to-Image 和 Image-to-Image
        if image is None:
            # Text-to-Image Path
            batch_size = 1 # We handle one prompt at a time
            latents_shape = (
                batch_size * N,
                self.transformer.config.in_channels,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )
            latents = torch.randn(latents_shape, generator=generator, device=device, dtype=self.transformer.dtype)
            
            # 设置调度器的总步数
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

        else:
            # Image-to-Image Path
            image_tensor = self.image_processor.preprocess(image).to(device=device, dtype=self.vae.dtype)
            init_latents = self.vae.encode(image_tensor).latent_dist.sample(generator)
            init_latents = self.vae.config.scaling_factor * init_latents
            
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            start_step = min(int(num_inference_steps * strength), num_inference_steps)
            timesteps = self.scheduler.timesteps[num_inference_steps - start_step :]
            
            noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=init_latents.dtype)
            # 在正确的 timestep 添加噪声
            latents = self.scheduler.add_noise(init_latents, noise, timesteps[0])
            
            # 将加噪后的 latents 复制 N 份
            latents = latents.repeat(N, 1, 1, 1)

        # 4. Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            # 这里无需修改，因为 self.transformer 已经是我们的自定义版本
            noise_pred = self.transformer(
                hidden_states=latents, 
                timestep=t, 
                encoder_hidden_states=prompt_embeds
            ).sample
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # 5. Decode latents
        image = self.vae.decode(latents / self.vae.config.scaling_factor).sample

        # 6. Post-process and return
        image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

def get_pipeline():
    print('connecting to pipeline...')
    base_pipe = FluxKontextPipeline.from_pretrained(
        "/mnt/hdd3/linzhuohang/3DGen/hf/hub/models--black-forest-labs--FLUX.1-Kontext-dev/snapshots/af58063aa431f4d2bbc11ae46f57451d4416a170",
        torch_dtype=torch.bfloat16,
    )

    print('loading transformer...')
    config = base_pipe.transformer.config
    oa_transformer = OAFluxTransformer2DModel.OAFluxTransformer2DModel(**config)
    
    oa_transformer =oa_transformer.from_pretrained('/mnt/hdd3/linzhuohang/3DGen/oa_transfomer',device_map="cuda",torch_dtype=torch.bfloat16)
    # 初始化DeepSpeed引擎，使用简化的配置
    print('deepspeed initializing...')
    oa_transformer = deepspeed.init_inference(
        model=oa_transformer,
        mp_size=1,
        dtype=torch.bfloat16,
        replace_with_kernel_inject=True
    )
    print('loading pipeline...')
    pipe = FluxKontextPipeline(
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        tokenizer=base_pipe.tokenizer,
        scheduler=base_pipe.scheduler,
        transformer=oa_transformer.module,  # 使用.module访问实际模型
        text_encoder_2=base_pipe.text_encoder_2,
        tokenizer_2=base_pipe.tokenizer_2,
    )
    
    return pipe

