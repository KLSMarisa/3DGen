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

class StableDiffusionTrainer(pl.LightningModule):
    def __init__(self, config):
        super(StableDiffusionTrainer, self).__init__()

        self.config = config
        self.save_hyperparameters()
        self.T = 12 # img num

        # self.text_clip = FrozenOpenCLIPEmbedder(layer='penultimate')
        # ckpt_path = '/home/caixiao/projects/ldm3d/text_encoder/model.ckpt'
        # self.text_clip.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)

        self.text_clip = FrozenCLIPEmbedder()
        ckpt_path = '/mnt/hdd1/caixiao/datasets/l3dm4c/text_encoder/pytorch_model.bin'
        self.text_clip.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)


        self.scale_factor = 0.18215
        self.ae = AutoencoderKL(
            {
                'double_z': True,
                'z_channels': 4,
                'resolution': 256,
                'in_channels': 4,
                'out_ch': 4,
                'ch': 128,
                'ch_mult': (1, 2, 4, 4),
                'num_res_blocks': 2,
                'attn_resolutions': [],
                'dropout': 0.0,
            },
            {'target': 'torch.nn.Identity'},
            4,
        )
        self.ae.eval()
        self.ae.requires_grad_(False)

        # ckpt_path = '/XXX/hub/huggingface/StableDiffusion/v2-1_512-ema-pruned-autoencoder.ckpt'
        ckpt_path = '/home/caixiao/projects/ldm3d/vae/diffusion_pytorch_model_rename3.ckpt'
        self.ae.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

        self.unet = UNetModel(
            use_checkpoint = True,
            use_fp16 = True,
            image_size = 32, # unused
            in_channels = 4,
            out_channels = 4,
            model_channels = 320,
            attention_resolutions = [ 4, 2, 1 ],
            num_res_blocks = 2,
            channel_mult = [ 1, 2, 4, 4 ],
            num_heads = 8, # need to fix for flash-attn
            use_spatial_transformer = True,
            use_linear_in_transformer = True,
            transformer_depth = 1,
            context_dim = 768,
            legacy = False,
            spatial_finetune_blocks = 1,
        )
        ckpt_path = '/home/caixiao/projects/ldm3d/unet/diffusion_pytorch_model_2_2.ckpt'
        # with open('/home/caixiao/projects/3d_lib/ckpt/unet_layers.txt', 'w') as f:
        #     for name, parameters in self.unet.named_parameters():
        #         f.write(f'{name} ++++++++++ {parameters.size()} \n')
        self.unet.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.unet.requires_grad_(False)



        self.stage = config.stage
        assert self.stage >= 1 and self.stage <= 3

        ## Stage 1, train added spatial
        self.unet.apply_extra_spatial(['text2obj'])

        ## Stage 2, train added temporal
        if self.stage > 1:
            self.unet.requires_grad_(False)
            _sd = {}
            ckpt_path = ''
            sd = torch.load(ckpt_path, map_location='cpu')
            for k, v in sd.items():
                if '_forward_module.unet.' in k:
                    _sd[k.replace('_forward_module.unet.', '')] = v
            self.unet.load_state_dict(_sd, strict=False)
            self.unet.apply_temporal()

        ## Stage 3, finetune spatial added in stage 1
        if self.stage > 2:
            self.unet.requires_grad_(False)
            _sd = {}
            ckpt_path = ''
            sd = torch.load(ckpt_path, map_location='cpu')
            for k, v in sd.items():
                if '_forward_module.unet.' in k:
                    _sd[k.replace('_forward_module.unet.', '')] = v
            self.unet.load_state_dict(_sd, strict=False)
            for module in self.unet.modules():
                if isinstance(module, NormResBlock):
                    module.requires_grad_(True)

        ## Stage 4
        if self.stage > 3:
            self.unet.requires_grad_(True)

        if config['ema']:
            self.ema_model = EMAModel(self.unet)
        else:
            self.ema_model = None

        # self.scheduler = DDIMScheduler(
        #     beta_start         =     0.00085,
        #     beta_end           =     0.0120,
        #     beta_schedule      =     'scaled_linear',
        #     clip_sample        =     False,
        #     set_alpha_to_one   =     False,
        #     steps_offset       =     1,
        # )

        self.scheduler = DDIMScheduler(
            beta_end=0.012,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            clip_sample=False,
            clip_sample_range=1.0,
            dynamic_thresholding_ratio=0.995,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            rescale_betas_zero_snr=False,
            sample_max_value=1.0,
            set_alpha_to_one=False,
            # skip_prk_steps= True,
            steps_offset=1,
            thresholding=False,
            timestep_spacing="leading",
            trained_betas=None
        )

    def configure_optimizers(self):
        opt = DeepSpeedCPUAdam(
            filter(lambda p: p.requires_grad, self.unet.parameters()), 
            lr=self.config["base_learning_rate"] if self.stage < 3 else self.config["base_learning_rate"]/5, 
            betas=(0.9, 0.9), 
            weight_decay=0.03
        )
        return opt

    @torch.no_grad()
    def encode_text(self, text):
        # text : ['a dog', 'a cat', 'a bird', ...]
        outputs = self.text_clip.encode(text)
        return outputs # b * l * c

    @torch.no_grad()
    def encode_images(self, images, unflatten=True):
        image = rearrange(images, 'B C T H W -> (B T) C H W')
        z = self.ae.encode(image).sample() # (N*T, C, H, W)
        if unflatten:
            z = rearrange(z, '(B T) C H W -> B C T H W', B=images.shape[0])
        return z * self.scale_factor

    @torch.no_grad()
    def decode_images(self, z, unflatten=False):
        if unflatten:
            z = rearrange(z, 'B C T H W -> (B T) C H W')
        z = z * (1. / self.scale_factor)
        images = self.ae.decode(z) # (N*T, C, H, W)
        return images

    @torch.no_grad()
    def intergrate_batch(self, batch):
        _batch = {}
        text_latent = self.encode_text(batch['caption']) # (N, 77, C)
        _batch['text'] = text_latent.to(self.dtype)
        mimage_latent = self.encode_images(batch['rgbd']) # N * C * T * H * W
        # print(mimage_latent.shape)
        # exit()
        if self.stage == 1:
            noise = torch.randn_like(mimage_latent)
        else:
            alpha = 1.0
            noise = torch.randn_like(mimage_latent[:, :, :1]) * ((alpha**2/(1+alpha**2))**0.5) \
                  + torch.randn_like(mimage_latent) * ((1/(1+alpha**2))**0.5)
        t = torch.randint(0, self.scheduler.num_train_timesteps, (mimage_latent.shape[0],), dtype=torch.int64, device=self.device)
        xt_images = self.scheduler.add_noise(mimage_latent, noise, t)
        _batch['images_timestep'] = t
        _batch['images_target'] = noise.to(self.dtype)
        _batch['images'] = xt_images.to(self.dtype)
        # _batch['images'] = batch['rgbd']
        _batch['dataset'] = batch['dataset']

        return _batch

    @property
    def dtype(self):
        return next(self.parameters()).data.dtype

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_model is not None:
            self.ema_model(self.unet)

    def training_step(self, batch, batch_idx):
        ### Step1: reconstruct batch for multi dataset training including noise preparing
        _batch = self.intergrate_batch(batch)
        # print(_batch['images'].shape)
        # exit()
        ### Step2: Predict the noise residual
        img_out = self.unet(
            _batch['images'], 
            _batch['images_timestep'], 
            _batch['text'], 
            _batch['dataset']
        )

        ### Step3: Compute loss
        loss = 0.0
        
        loss_images = F.mse_loss(img_out.float().flatten(1), _batch['images_target'].float().flatten(1))
        self.log('loss_images', loss_images, prog_bar=True, sync_dist=True)
        loss += loss_images

        return loss

    @torch.no_grad()
    def inference(self, 
        text=None, 
        num_inference_steps=50, 
        do_classifier_free_guidance=True, 
        guidance_scale=7.0, 
        use_dataset='text2obj',
        **extra_step_kwargs
    ):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        if isinstance(text, list) or isinstance(text, tuple):
            batch_size = len(text)
        elif isinstance(text, str):
            batch_size = 1
            text = [text]
        else:
            raise NotImplementedError

        if not do_classifier_free_guidance:
            text_latent = self.encode_text(text).to(self.dtype)
        else:
            text = text + [''] * batch_size
            text_latent = self.encode_text(text).to(self.dtype)
        
        # images_latent = torch.randn((batch_size, 4, self.T, self.config['diff_latent_res_H'], self.config['diff_latent_res_W']), dtype=self.dtype, device=self.device)
        alpha = 1.0
        images_latent = torch.randn((batch_size, 4, 1, self.config['diff_latent_res_H'], self.config['diff_latent_res_W']), dtype=self.dtype, device=self.device) * ((alpha**2/(1+alpha**2))**0.5) \
                + torch.randn((batch_size, 4, self.T, self.config['diff_latent_res_H'], self.config['diff_latent_res_W']), dtype=self.dtype, device=self.device) * ((1/(1+alpha**2))**0.5)


        for t in timesteps:
            if do_classifier_free_guidance:
                input_images_latent = torch.cat([images_latent, images_latent], 0)
            else:
                input_images_latent = images_latent
    
            timestep = t.repeat(input_images_latent.shape[0]).contiguous()

            dataset = [use_dataset] * input_images_latent.shape[0]
            noise_pred = self.unet(
                input_images_latent, 
                timestep, 
                text_latent,
                dataset
            )

            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_cond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            images_latent = self.scheduler.step(noise_pred, t, images_latent).prev_sample

        result = {}
        result['images'] = self.decode_images(images_latent, unflatten=True)
        result['images'] = rearrange(result['images'], '(B T) C H W -> B T C H W', B=batch_size)

        out_unet = out_unet[0, :, :, :, :]

        out_unet = ((out_unet + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu()
        # out_unet = out_unet.numpy()

        depth_unet = out_unet[0, 3, :, :]
        img_unet = out_unet[0, :3, :, :]

        # depth_unet = out_unet[0, 0, 3, :, :]
        # img_unet = out_unet[0, 0, :3, :, :]

        # to_img = ToPILImage()
        # img_unet = to_img(img_unet)
        # depth_unet = to_img(depth_unet)
        # img_unet.save(f"/home/caixiao/projects/3DGen/test_unet/unet_out.png")
        # depth_unet.save(f"/home/caixiao/projects/3DGen/test_unet/dep_unet_out.png")
        return result

