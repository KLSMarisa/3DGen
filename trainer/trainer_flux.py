import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from deepspeed.runtime.lr_schedules import WarmupLR
import warnings
import clip
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, PNDMScheduler, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from modules.networks import EMAModel, EMAModel_for_deepspeed_stage3
from modules.StableDiffusion import UNetModel, FrozenCLIPEmbedder, AutoencoderKL, NormResBlock
from torchvision.transforms import ToPILImage
from diffusers import FluxKontextPipeline
from flux_modules import OAFluxKontextPipeline2 as OAFluxKontextPipeline
from typing import Optional, Union, List
from PIL import Image
class Flux_Trainer(pl.LightningModule):
    def __init__(self):
        super(Flux_Trainer, self).__init__()

        self.pipeline = OAFluxKontextPipeline.get_pipeline(Train =True)
        self.pipeline.to('cuda')
        self.pipeline.frozen_parameters()
        self.transformer = self.pipeline.transformer
        for block in self.transformer.transformer_blocks:
            block.ortho_attn.requires_grad_(True)
        #self.transformer.transformer_blocks[0].requires_grad_(True)
        self.save_hyperparameters()
        self.scheduler = self.pipeline.scheduler

    def configure_optimizers(self):
        params = self.transformer.parameters()
        #print('params:',len(params))
        opt = DeepSpeedCPUAdam(
            filter(lambda p: p.requires_grad, params),
            lr=1e-3,#self.config["base_learning_rate"] if self.stage < 3 else self.config["base_learning_rate"]/5, 
            betas=(0.9, 0.9), 
            weight_decay=0.03
        )
        return opt

    



    @property
    def dtype(self):
        return next(self.parameters()).data.dtype


    def predict(
        self,
        image,
        gt_images,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]],
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[Image.Image] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[Image.Image] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs = None,
        callback_on_step_end = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        max_area: int = 1024**2,
        _auto_resize: bool = True,
        ):
        with torch.no_grad():
            device = self.pipeline._execution_device
            image.to(device)


            height = height or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor
            width = width or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor

            original_height, original_width = height, width
            aspect_ratio = width / height
            width = round((max_area * aspect_ratio) ** 0.5)
            height = round((max_area / aspect_ratio) ** 0.5)

            multiple_of = self.pipeline.vae_scale_factor * 2
            width = width // multiple_of * multiple_of
            height = height // multiple_of * multiple_of

            if height != original_height or width != original_width:
                print(
                    f"Generation `height` and `width` have been adjusted to {height} and {width} to fit the model requirements."
                )

            # 1. Check inputs. Raise error if not correct


            self.pipeline._guidance_scale = guidance_scale
            self.pipeline._joint_attention_kwargs = joint_attention_kwargs
            self.pipeline._current_timestep = None
            self.pipeline._interrupt = False

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            

            lora_scale = (
                self.pipeline.joint_attention_kwargs.get("scale", None) if self.pipeline.joint_attention_kwargs is not None else None
            )
            has_neg_prompt = negative_prompt is not None or (
                negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
            ) 
            do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = self.pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            if do_true_cfg:
                (
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    negative_text_ids,
                ) = self.pipeline.encode_prompt(
                    prompt=negative_prompt,
                    prompt_2=negative_prompt_2,
                    prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    lora_scale=lora_scale,
                )

            # 3. Preprocess image
            gt_images =  rearrange(gt_images,'b n c h w -> (b n) c h w')
            if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.pipeline.latent_channels):
                img = image[0] if isinstance(image, list) else image
                image_height, image_width = self.pipeline.image_processor.get_default_height_width(img)
                aspect_ratio = image_width / image_height
                if _auto_resize:
                    # Kontext is trained on specific resolutions, using one of them is recommended
                    _, image_width, image_height = min(
                        (abs(aspect_ratio - w / h), w, h) for w, h in  OAFluxKontextPipeline.PREFERRED_KONTEXT_RESOLUTIONS
                    )
                image_width = image_width // multiple_of * multiple_of
                image_height = image_height // multiple_of * multiple_of
                image = self.pipeline.image_processor.resize(image, image_height, image_width)
                image = self.pipeline.image_processor.preprocess(image, image_height, image_width)
                gt_images = self.pipeline.image_processor.resize(gt_images, image_height, image_width)
                gt_images = self.pipeline.image_processor.preprocess(gt_images, image_height, image_width)
                

            # 4. Prepare latent variables
            num_channels_latents = self.transformer.config.in_channels // 4
            _, image_latents, latent_ids, image_ids = self.pipeline.prepare_latents(
                image,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device
            )
            
            _,gt_images_latents,_,_ = self.pipeline.prepare_latents(
                gt_images,
                batch_size*3,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device
            )
            print('gt_images_latents shape:',gt_images_latents.shape)
            print('image_latents shape:',image_latents.shape)
            if image_ids is not None:
                latent_ids = torch.cat([latent_ids, image_ids], dim=0)  # dim 0 is sequence dimension

            # 5. Prepare timesteps
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas




            # handle guidance
            if self.transformer.config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                guidance = guidance.expand(image.shape[0])
            else:
                guidance = None

            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
                negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
            ):
                negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
                negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

            elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
                negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
            ):
                ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
                ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

            if self.pipeline.joint_attention_kwargs is None:
                self.pipeline._joint_attention_kwargs = {}

            image_embeds = None
            negative_image_embeds = None
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                image_embeds = self.pipeline.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                )
            if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
                negative_image_embeds = self.pipeline.prepare_ip_adapter_image_embeds(
                    negative_ip_adapter_image,
                    negative_ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                )

            
            # modified for OA
            def expand_3d(arr):
                return torch.repeat_interleave(arr, repeats=3, dim=0)
            
            noise = torch.randn_like(gt_images_latents).to(device)
            t = torch.randint(0, self.scheduler.num_train_timesteps, (batch_size,), dtype=torch.int64, device=self.device)
            t_reshaped = t.view(batch_size, *([1] * (gt_images_latents.dim() - 1))).to(device)
            latents = t_reshaped * gt_images_latents + (1 - t_reshaped) * noise
            target_vector = gt_images_latents - noise
            pooled_prompt_embeds = expand_3d(pooled_prompt_embeds)
            image_latents = expand_3d(image_latents) if image_latents is not None else None
            #text_ids = expand_3d(text_ids)
            prompt_embeds = expand_3d(prompt_embeds)
            #latent_ids = expand_3d(latent_ids) if image_ids is not None else None
            if image_embeds is not None:
                self.pipeline._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
            latent_model_input = latents
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)
            
            print(latents.shape)
            print(latent_model_input.shape)
            print('text ids shape:',text_ids.shape)
            print('latent ids shape:',latent_ids.shape)
            
        predict_vector = self.transformer(
            hidden_states=latent_model_input,
            timestep=t / 1000,
            guidance=None, #modfied
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            joint_attention_kwargs=self.pipeline.joint_attention_kwargs,
            return_dict=False,
        )[0]
        predict_vector = predict_vector[:, : latents.size(1)]

        if do_true_cfg:
            if negative_image_embeds is not None:
                self.pipeline._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
            neg_noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=t / 1000,
                guidance=None, #modfied
                pooled_projections=negative_pooled_prompt_embeds,
                encoder_hidden_states=negative_prompt_embeds,
                txt_ids=negative_text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs=self.pipeline.joint_attention_kwargs,
                return_dict=False,
            )[0]
            neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
            predict_vector = neg_noise_pred + true_cfg_scale * (predict_vector - neg_noise_pred)
        return predict_vector,target_vector



    def training_step(self, batch, batch_idx):
        ### Step1: reconstruct batch for multi dataset training including noise preparing
        # print(_batch['images'].shape)
        # exit()
        ### Step2: Predict the noise residual
        print('start training step')
        print('img shape',batch['img'].shape)
        print('rgb shape',batch['rgb'].shape)
        predict_vector,target_vector = self.predict(batch['img'],batch['rgb'],batch['caption'],batch['caption'])
        print('calculate loss')
        ### Step3: Compute loss
        loss = 0.0
        
        loss_images = F.mse_loss(predict_vector.float().flatten(1), target_vector.float().flatten(1))
        self.log('loss_images', loss_images, prog_bar=True, sync_dist=True)
        print('end training step')
        loss += loss_images
        
        return loss

