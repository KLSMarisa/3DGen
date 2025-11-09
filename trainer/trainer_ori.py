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
from modules.StableDiffusion import UNetModel, FrozenOpenCLIPEmbedder, AutoencoderKL, NormResBlock


class StableDiffusionTrainer(pl.LightningModule):
    def __init__(self, config):
        super(StableDiffusionTrainer, self).__init__()

        self.config = config
        self.save_hyperparameters()
        self.T = config['timesteps'] // config['splits']

        self.text_clip = FrozenOpenCLIPEmbedder(layer='penultimate')
        ckpt_path = '/XXX/hub/huggingface/StableDiffusion/v2-1_512-ema-pruned-openclip.ckpt'
        self.text_clip.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)

        self.scale_factor = 0.18215
        self.ae = AutoencoderKL(
            {
                'double_z': True,
                'z_channels': 4,
                'resolution': 256,
                'in_channels': 3,
                'out_ch': 3,
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
        ckpt_path = '/XXX/hub/huggingface/StableDiffusion/v2-1_512-ema-pruned-autoencoder.ckpt'
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
            num_head_channels = 64, # need to fix for flash-attn
            use_spatial_transformer = True,
            use_linear_in_transformer = True,
            transformer_depth = 1,
            context_dim = 1024,
            legacy = False,
            spatial_finetune_blocks = 1,
        )
        ckpt_path = '/XXX/hub/huggingface/StableDiffusion/v2-1_512-ema-pruned-diffusionmodel.ckpt'
        self.unet.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.unet.requires_grad_(False)

        self.stage = config.stage
        assert self.stage >= 1 and self.stage <= 3

        ## Stage 1, train added spatial
        self.unet.apply_extra_spatial(['webvid', 'hdvg'])

        ## Stage 2, train added temporal
        if self.stage > 1:
            self.unet.requires_grad_(False)
            _sd = {}
            ckpt_path = '/XXX/deeplearning/text-to-VA/checkpoints/spatial_finetune_superwide-step=70000.ckpt/model.pth'
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
            ckpt_path = 'XXX/deeplearning/text-to-VA/checkpoints/temporal_training_superwide-step=200000.ckpt/model.pth'
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

        self.scheduler = DDIMScheduler(
            beta_start         =     0.00085, 
            beta_end           =     0.0120, 
            beta_schedule      =     'scaled_linear', 
            clip_sample        =     False,
            set_alpha_to_one   =     False,
            steps_offset       =     1,
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
    def encode_video(self, video, unflatten=True):
        image = rearrange(video, 'B C T H W -> (B T) C H W')
        z = self.ae.encode(image).sample() # (N*T, C, H, W)
        if unflatten:
            z = rearrange(z, '(B T) C H W -> B C T H W', B=video.shape[0])
        return z * self.scale_factor

    @torch.no_grad()
    def decode_video(self, z, unflatten=False):
        if unflatten:
            z = rearrange(z, 'B C T H W -> (B T) C H W')
        z = z * (1. / self.scale_factor)
        video = self.ae.decode(z) # (N*T, C, H, W)
        return video

    @torch.no_grad()
    def intergrate_batch(self, batch):
        _batch = {}

        text_latent = self.encode_text(batch['text']) # (N, 77, C)
        _batch['text'] = text_latent.to(self.dtype)
        video_latent = self.encode_video(batch['video']) # N * C * T * H * W
        if self.stage == 1:
            noise = torch.randn_like(video_latent)
        else:
            alpha = 1.0
            noise = torch.randn_like(video_latent[:, :, :1]) * ((alpha**2/(1+alpha**2))**0.5) \
                  + torch.randn_like(video_latent) * ((1/(1+alpha**2))**0.5)
        t = torch.randint(0, self.scheduler.num_train_timesteps, (video_latent.shape[0],), dtype=torch.int64, device=self.device)
        xt_video = self.scheduler.add_noise(video_latent, noise, t)
        _batch['video_timestep'] = t
        _batch['video_target'] = noise.to(self.dtype)
        _batch['video'] = xt_video.to(self.dtype)

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

        ### Step2: Predict the noise residual
        video_out = self.unet(
            _batch['video'], 
            _batch['video_timestep'], 
            _batch['text'], 
            _batch['dataset']
        )

        ### Step3: Compute loss
        loss = 0.0
        
        loss_video = F.mse_loss(video_out.float().flatten(1), _batch['video_target'].float().flatten(1))
        self.log('loss_video', loss_video, prog_bar=True, sync_dist=True)
        loss += loss_video

        return loss

    @torch.no_grad()
    def inference(self, 
        text=None, 
        num_inference_steps=50, 
        do_classifier_free_guidance=True, 
        guidance_scale=7.0, 
        use_dataset='hdvg',
        fixed_noise=False,
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
        
        # video_latent = torch.randn((batch_size, 4, self.T, self.config['diff_latent_res_H'], self.config['diff_latent_res_W']), dtype=self.dtype, device=self.device)
        alpha = 1.0
        if not fixed_noise or not hasattr(self, 'fixed_noise'):
            video_latent = torch.randn((batch_size, 4, 1, self.config['diff_latent_res_H'], self.config['diff_latent_res_W']), dtype=self.dtype, device=self.device) * ((alpha**2/(1+alpha**2))**0.5) \
                    + torch.randn((batch_size, 4, self.T, self.config['diff_latent_res_H'], self.config['diff_latent_res_W']), dtype=self.dtype, device=self.device) * ((1/(1+alpha**2))**0.5)
        else:
            video_latent = self.fixed_noise

        if fixed_noise and not hasattr(self, 'fixed_noise'):
            self.fixed_noise = video_latent

        for t in timesteps:
            if do_classifier_free_guidance:
                input_video_latent = torch.cat([video_latent, video_latent], 0)
            else:
                input_video_latent = video_latent
    
            timestep = t.repeat(input_video_latent.shape[0]).contiguous()

            dataset = [use_dataset] * input_video_latent.shape[0]
            noise_pred = self.unet(
                input_video_latent, 
                timestep, 
                text_latent,
                dataset
            )

            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_cond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            video_latent = self.scheduler.step(noise_pred, t, video_latent, **extra_step_kwargs).prev_sample

        result = {}
        result['video'] = self.decode_video(video_latent, unflatten=True)
        result['video'] = rearrange(result['video'], '(B T) C H W -> B T C H W', B=batch_size)

        return result

