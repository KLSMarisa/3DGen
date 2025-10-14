import datetime
import json
import os
import sys
import time
import PIL
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
from torchvision.transforms import ToPILImage
from diffusers import FluxKontextPipeline
from flux_modules import OAFluxKontextPipeline2 as OAFluxKontextPipeline
from typing import Optional, Union, List
from PIL import Image
import params_inspect


from utils import measures
class Flux_Trainer(pl.LightningModule):
    def __init__(self,init_step,config):
        super(Flux_Trainer, self).__init__()
        version = config.version
        self.cpu_opt = config.cpu_offload
        self.inference_saving_path = f'/home/linzhuohang/train_outputs_v{version}/{init_step}'
        self.val_saving_path = f'/home/linzhuohang/val_outputs_v{version}/'
        self.loss_list = []
        self.log_interval = config.log_interval
        ckpt_path= f'{config.data_dir}/ckptv{version}/safetensors/{init_step}'
        if not os.path.exists(ckpt_path):
            print('using 0 ckpt')
            
            ckpt_path= f'{config.data_dir}/ckptv{version}/safetensors/0'
        if not os.path.exists(ckpt_path):
            print('using last version ckpt')
            ckpt_path= f'{config.data_dir}/ckptv{version-1}/safetensors/{init_step}'
        
        self.pipeline = OAFluxKontextPipeline.get_pipeline(ckpt_path,config=config,Train =True)
        self.pipeline.frozen_parameters()
        self.transformer = self.pipeline.transformer
        print(self.transformer)
        del self.pipeline.transformer
        torch.cuda.empty_cache()
        block_lenth = len(self.transformer.oa_transformer_blocks)
        self.transformer.norm_out.requires_grad_(True)
        self.transformer.proj_out.requires_grad_(True)
        self.transformer.gradient_checkpointing = config.gradient_checkpointing
        for i,block in enumerate(self.transformer.oa_transformer_blocks):
            block.requires_grad_(True)
            if i!=0 and config.gradient_checkpointing:  block.use_checkpoint = True
            #print(f'{i} :{block.enable_oa}')
            #block.enable_oa = False
        #self.transformer.transformer_blocks[0].requires_grad_(True)
        


    def configure_optimizers(self):
        params = self.transformer.parameters()
        grad_params = filter(lambda p: p.requires_grad, params)
        optimizer = DeepSpeedCPUAdam if self.cpu_opt else FusedAdam
        opt = optimizer(
            grad_params,
            lr=4e-5,
            betas=(0.9, 0.99),
            weight_decay=0.02
        ) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=16000, eta_min=0)
        scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": opt,  "lr_scheduler": scheduler_config}

    



    @property
    def dtype(self):
        return next(self.parameters()).data.dtype


    def predict(
        self,
        image,
        gt_images,
        prompt: Union[str, List[str]],
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
        joint_attention_kwargs = None,
        max_sequence_length: int = 512,
        max_area: int = 1024**2,
        _auto_resize: bool = True,
        use_caption = True
        ):



        with torch.no_grad():
            #print('func: ',self.transformer.time_text_embed)
            def expand_3d(arr):
                return torch.repeat_interleave(arr, repeats=3, dim=0)
            def print_tensor_info(tensor, name):    
                #print(f'{name} info:    min={tensor.min().item()}  max={tensor.max().item()}   mean={tensor.mean().item()} std={tensor.std().item()}    shape={tensor.shape}')
                pass
            device = self.pipeline._execution_device
            image.to(device)
            image = (image+1.0)/2.0
            gt_images = (gt_images+1.0)/2.0
            height = height or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor
            width = width or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor

            #original_height, original_width = height, width
            #aspect_ratio = width / height
            #width = round((max_area * aspect_ratio) ** 0.5)
            #height = round((max_area / aspect_ratio) ** 0.5)
            #multiple_of = self.pipeline.vae_scale_factor * 2
            #width = width // multiple_of * multiple_of
            #height = height // multiple_of * multiple_of
            #if height != original_height or width != original_width:
            #    print(
            #        f"Generation `height` and `width` have been adjusted to {height} and {width} to fit the model requirements."
            #    )

            # 1. Check inputs. Raise error if not correct


            self.pipeline._guidance_scale = guidance_scale
            self.pipeline._joint_attention_kwargs = joint_attention_kwargs
            self.pipeline._current_timestep = None
            self.pipeline._interrupt = False

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
                prompt = [prompt]
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]
            if use_caption: 
                prompt_3d = []
                for i in range(batch_size):
                    prompt_3d.append('front view relative to the input image:'+prompt[i])
                    prompt_3d.append('upper view relative to the input image:'+prompt[i])
                    prompt_3d.append('side view relative to the input image:'+prompt[i])
            else: prompt_3d = [' ',' ',' ']
            
            prompt_2 = prompt = prompt_3d
            #print(prompt)
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
                #image_width = image_width // multiple_of * multiple_of
                #image_height = image_height // multiple_of * multiple_of
                image = self.pipeline.image_processor.resize(image, image_height, image_width)
                image = self.pipeline.image_processor.preprocess(image, image_height, image_width)
                gt_images = self.pipeline.image_processor.resize(gt_images, image_height, image_width)
                gt_images = self.pipeline.image_processor.preprocess(gt_images, image_height, image_width)
            print_tensor_info(gt_images, 'gt_images')
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
            #print('gt_images_latents shape:',gt_images_latents.shape)
            #print('image_latents shape:',image_latents.shape)
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
            
            
            noise = torch.randn_like(gt_images_latents).to(device)
            indices = torch. torch.randint(0, len(self.pipeline.scheduler.timesteps), (batch_size,), device='cpu')
# Get the actual timestep values from the scheduler's list using the random indices
            t = self.pipeline.scheduler.timesteps[indices].to(device)
            t_reshaped = t.view(batch_size, *([1] * (gt_images_latents.dim() - 1)))
            t_reshaped = expand_3d(t_reshaped)
            latents = []
            print_tensor_info(gt_images_latents, 'gt_images_latents')
            print_tensor_info(image_latents, 'image_latents')
            for i in range(batch_size*3):
                latents.append(self.pipeline.scheduler.scale_noise(gt_images_latents[i],t_reshaped[i],noise[i]))
            #print('item shape: ',latents[0].shape)
            #print(len(latents))
            latents = torch.stack(latents,dim=0)
            print_tensor_info(latents, 'latents')
            #print('latent shape: ',latents.shape)
            #latents = t_reshaped*noise+(1-t_reshaped)*gt_images_latents
            target_vector = noise - gt_images_latents
            print_tensor_info(target_vector, 'target_vector')
            image_latents = expand_3d(image_latents) if image_latents is not None else None
            #text_ids = text_ids.to(image_latents.dtype)
            #latent_ids = latent_ids.to(image_latents.dtype) if image_ids is not None else None
            if image_embeds is not None:
                self.pipeline._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
            latent_model_input = latents
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)
            #print('image latent shape:',image_latents.shape)
            #print('latents shape:',latents.shape)
            #print(latent_model_input.shape)
            #print('text ids shape:',text_ids.shape)
            #print('latent ids shape:',latent_ids.shape)
            
        predict_vector = self.transformer(
            hidden_states=latent_model_input,
            timestep=t / 1000,
            guidance=guidance, 
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            joint_attention_kwargs=self.pipeline.joint_attention_kwargs,
            return_dict=False,
            
        )[0]
        predict_vector = predict_vector[:, : latents.size(1)]
        print_tensor_info(predict_vector, 'predict_vector')
        if do_true_cfg:
            if negative_image_embeds is not None:
                self.pipeline._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
            neg_noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=t / 1000,
                guidance=guidance, 
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

    def inference(self,image,prompt):
        result = self.pipeline(image,prompt,prompt,transformer = self.transformer,height=OAFluxKontextPipeline.input_size,width=OAFluxKontextPipeline.input_size)
        return result


    def numpy_to_pil(self,images: np.ndarray) :
        r"""
        Convert a numpy image or a batch of images to a PIL image.

        Args:
            images (`np.ndarray`):
                The image array to convert to PIL format.

        Returns:
            `List[PIL.Image.Image]`:
                A list of PIL images.
        """
        images = images.squeeze(0).detach().to(torch.float).cpu().numpy()
        images = (images * 0.5 + 0.5).clip(0, 1)
        images = (images * 255).round().astype("uint8")
        print('images shape',images.shape)
        images = np.transpose(images,(1,2,0))
        print(images.shape)
        pil_images = Image.fromarray(images)
        pil_images = pil_images.resize((OAFluxKontextPipeline.input_size,OAFluxKontextPipeline.input_size))
        return pil_images
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        save_path = os.path.join(self.val_saving_path, str(self.global_step),str(batch_idx))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #print('batch img shape:',batch['img'].shape)
        with open(f'{save_path}/caption.txt', 'w') as f:
            f.write(batch['caption'][0])
        self.numpy_to_pil(batch['img']).save(save_path+'/src.jpg')
        for i in range(len(batch['rgb'][0])):
            self.numpy_to_pil(batch['rgb'][0][i]).save(f'{save_path}/src_{i}.jpg')
        result = self.inference(batch['img'],batch['caption'])
        #print('result:',len(result))
        #print(result)
        result_np = []
        for i in range(len(result)):
            result[i].save(f'{save_path}/{i}.jpg')
            result_np.append(np.array(result[i].resize((OAFluxKontextPipeline.input_size,OAFluxKontextPipeline.input_size)),dtype=float)/127.5-1)
        loss = F.mse_loss(torch.tensor(result_np).flatten(1), batch['rgb'][0].cpu().flatten(1))
        self.log('val_loss',loss.item(),on_step = True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        if batch_idx>20:
            sys.exit(0)
        for i in range(2):
            save_path = os.path.join(self.inference_saving_path, str(batch_idx),str(i))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            #print('batch img shape:',batch['img'].shape)
            with open(f'{save_path}/caption.txt', 'w') as f:
                f.write(batch['caption'][0])
            self.numpy_to_pil(batch['img']).save(save_path+'/src.jpg')
            for i in range(len(batch['rgb'][0])):
                self.numpy_to_pil(batch['rgb'][0][i]).save(f'{save_path}/src_{i}.jpg')
            result = self.inference(batch['img'],batch['caption'])
            #print('result:',len(result))
            #print(result)
            for i in range(len(result)):
                result[i].save(f'{save_path}/{i}.jpg')
            compare_path = os.path.join(save_path, 'compare.json')
            compare_results = {}
            if os.path.exists(compare_path):
                with open(compare_path, 'r') as f:
                    compare_results = json.load(f)
            for i in range(len(result)):
                src_img = np.array(self.numpy_to_pil(batch['img'][0]))
                gen_img = np.array(result[i])
                psnr = measures.compare_psnr(src_img, gen_img)
                ssim = measures.compare_ssim(src_img, gen_img, multichannel=True)
                compare_results[str(i)] = {'psnr':psnr, 'ssim':ssim}
            with open(compare_path, 'w') as f:
                json.dump(compare_results, f)

    
    
    
    def training_step(self, batch, batch_idx):
        ### Step1: reconstruct batch for multi dataset training including noise preparing
        # print(_batch['images'].shape)
        # exit()
        ### Step2: Predict the noise residual
        #print('start training step')
        #print('img shape',batch['img'].shape)
        #print('rgb shape',batch['rgb'].shape)
        try:
            local_rank = int(os.environ["LOCAL_RANK"])
        except:
            local_rank = 0

        #print(f'global_step: {self.global_step}, rank: {local_rank}')
        predict_vector,target_vector = self.predict(batch['img'],batch['rgb'],batch['caption'],width=OAFluxKontextPipeline.input_size,height=OAFluxKontextPipeline.input_size,use_caption=False)
        #print('calculate loss')
        ### Step3: Compute loss
        if(batch_idx%(self.log_interval)==0 and local_rank==0): 
            #t1 = time.time()
            results= params_inspect.inspect_transformer_blocks(self.transformer)
            #grads =  params_inspect.inspect_transformer_grads(self.transformer)
            #t2 = time.time()
            #print('inspect time:',t2-t1)
            self.log_dict(results,on_step=True,logger=True, on_epoch=False)
            #self.log_dict(grads, on_step=True, on_epoch=False, logger=True)
        loss = F.mse_loss(predict_vector.float().flatten(1), target_vector.float().flatten(1))
        #print('end training step')
        print('loss:',loss.detach())
        self.loss_list.append(loss.detach())
        if(batch_idx+1)%self.log_interval==0: 
            mean = torch.stack(self.loss_list).mean()
            self.loss_list = []
            print('loss mean',mean)
            self.log('loss',mean.cpu().item(),on_step=True)
            #if os.environ['LOCAL_RANK'] == '0':
            #    for name, param in self.transformer.named_parameters():
            #        if(param.requires_grad):
            #            print(f'{name} grad: {param.grad}')
        #if(self.global_step+1)%self.image_interval==0  and local_rank==0:
        #    print(f"global_step: {self.global_step}, generating sample")
        #    result =  self.inference(batch['img'][0].unsqueeze(0),batch['caption'][0])
        #    for i in range(3):
        #        result[i].save(f'/home/linzhuohang/train_outputs/{self.global_step}_{i}.jpg')
        return loss

    def on_after_backward(self):
        if self.global_step % 5 == 0:  # 每 500 step 记录一次
            st_time = datetime.datetime.now()
            grads = params_inspect.inspect_transformer_grads(self.transformer)
            ed_time = datetime.datetime.now()
            print('grad inspect time:',(ed_time-st_time))
            if grads:
                # 一次性写入 logger，on_step=True 保证逐步记录
                self.log_dict(grads, on_step=True, on_epoch=False, logger=True)

