import os
import csv
from typing import Iterable
from data.text2obj_dataset import Text2ObjDataset
from torch.utils.data import Dataset, IterableDataset, ChainDataset
from torch.utils.data import DataLoader
from modules.StableDiffusion import UNetModel, FrozenCLIPEmbedder, AutoencoderKL, NormResBlock
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from deepspeed.runtime.lr_schedules import WarmupLR
import warnings
import clip
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, PNDMScheduler, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
# from diffusers import StableDiffusionLDM3DPipeline
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from modules.networks import EMAModel, EMAModel_for_deepspeed_stage3
from torchvision.transforms import ToPILImage

def load_text2obj_dataset(
        *,
        data_dir,
        center_crop=True,
        random_flip=False,
):
    """
    For a dataset, create a generator over text2obj (text-objaverse pair) data samples.
    The format of data is a dict. The meaning of each element is as follows,
    "video": the video, a NFCHW dtype tensor
    "video_caption": tokenized text, a tensor
    "video_text": the original text
    "dataset": the dataset name
    "audio": None (TODO)
    "audio_caption": None (TODO)
    "audio_text": None (TODO)

    :param annotation_path: annotation file.
    :param data_dir: video dir
    :param video_size: [frane_num, channnel, height, width]
    :param video_fps: loading fps of the video
    :param frame_gap: after loading the video with video_fps, sample under this frame gap
    :param center_crop/random_flip: data augmentation
    :param p_uncond: the rate of "" for classifier free guidance

    :param audio_size, audio_fps (TODO)
    """

    img_infos = []

    path_lists = os.listdir(os.path.join(data_dir, 'views_release'))
    path_lists.sort()
    for path_list in path_lists:
        path = os.path.join(data_dir, 'views_release', path_list)
        img_info = dict()
        img_info['filename'] = path
        img_infos.append(img_info)

    # img_infos = sorted(img_infos, key=lambda x: x['filename'])
    print(f"load {len(img_infos)} objs in Text2ObjDataset")

    return Text2ObjDataset(
        data_dir=data_dir,
        img_list=img_infos,
        random_flip=random_flip,
        center_crop=center_crop,
    )


class ConcatDataset(IterableDataset):
    r"""Dataset for concating multiple :class:`IterableDataset` s.

    This class is useful to assemble different existing dataset streams. The
    concating operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """

    def __init__(self, datasets: Iterable[Dataset], length_cut='max') -> None:
        super(ConcatDataset, self).__init__()
        self.datasets = datasets
        self.length_cut = length_cut
        assert length_cut in ['max', 'min']
        self.len = len(self)

    def __iter__(self):
        datasets_iter = [iter(ds) for ds in self.datasets]
        for i in range(self.len):
            idx = i % len(self.datasets)
            try:
                x = next(datasets_iter[idx])
            except:
                datasets_iter[idx] = iter(self.datasets[idx])
                x = next(datasets_iter[idx])
            yield x

    def __len__(self):
        lengths = []
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            lengths.append(len(d))
        if self.length_cut == 'max':
            total = max(lengths) * len(lengths)
        elif self.length_cut == 'min':
            total = min(lengths) * len(lengths)
        return total


def UnifiedDataset(config):
    _datasets = []

    datasets = config.datasets.split(',')
    assert len(datasets) > 0, "No dataset specified"
    for dataset in datasets:
        if dataset == 'text2obj':
            _datasets.append(load_text2obj_dataset(
                data_dir=config.dataset.text2obj.data_dir
            ))
        else:
            '''
            TODO
            '''
            raise NotImplementedError

    # return ChainDataset(_datasets)
    return ConcatDataset(_datasets)


def identity_collate_fn(x):
    '''Identity function to be passed as collate function to DataLoader'''
    return x


def create_dataloader(config):
    dataset = UnifiedDataset(config)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=None,
        pin_memory=False,
    )
    return dataloader


text_clip = FrozenCLIPEmbedder()
# print(text_clip)
ckpt_path = '/mnt/hdd1/caixiao_hdd1/datasets/l3dm4c/text_encoder/pytorch_model.bin'
text_clip.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
text_clip.cuda()

ae = AutoencoderKL(
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
# with open('/home/caixiao/projects/3d_lib/weight/our_model_vae.txt', 'w') as f:
#     for name, module in ae.named_modules():
#         f.write(f'{name} {module}\n')
ae.cuda()
# ae.half()
ae.eval()
ae.requires_grad_(False)

# ckpt_path = '/XXX/hub/huggingface/StableDiffusion/v2-1_512-ema-pruned-autoencoder.ckpt'
# ckpt_path = '/home/caixiao/projects/ldm3d/vae/diffusion_pytorch_model_rename3.ckpt'
ckpt_path = '/home/caixiao/projects/3DGen/models/diffusion_pytorch_model_1_3.bin'
ae.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

unet = UNetModel(
    use_checkpoint=True,
    use_fp16=True,
    image_size=32,  # unused
    in_channels=4,
    out_channels=4,
    model_channels=320,
    attention_resolutions=[4, 2, 1],
    num_res_blocks=2,
    channel_mult=[1, 2, 4, 4],
    num_head_channels=8,  # need to fix for flash-attn
    use_spatial_transformer=True,
    use_linear_in_transformer=True,
    transformer_depth=1,
    context_dim=768,
    legacy=False,
    spatial_finetune_blocks=1,
)




# ckpt_path = '/home/caixiao/projects/ldm3d/unet/diffusion_pytorch_model_2_2.ckpt'
ckpt_path = '/home/caixiao/projects/3DGen/models/unet/diffusion_pytorch_model_1_4.bin'
unet.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
# with open('/home/caixiao/projects/3d_lib/weight/our_model_unet.txt', 'w') as f:
#     for layer, module in unet.named_modules():
#         f.write(f'{layer}\n ')
#         f.write('-------------\n')
#         for name, param in module.named_parameters():
#             f.write(f'{layer}{name} {param.size()}\n')
#             f.write(f'{param.data}\n')
#             f.write('\n\n')
# with open('/home/caixiao/projects/3d_lib/weight/weights_our2.txt', 'w') as f:
#     for name, param in unet.named_parameters():
#         if param.requires_grad:
#             f.write(f'{name} {param.size()}\n')
#             f.write(f'{param.data}\n')
#             f.write('\n\n')

unet.half()
unet.cuda()

unet.requires_grad_(False)
# unet.apply_temporal()

scheduler = PNDMScheduler(
    beta_start=0.00085,
    beta_end=0.0120,
    num_train_timesteps=1000,
    beta_schedule='scaled_linear',
    # clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
    skip_prk_steps=True,
    # trained_betas=null,
)

root = '/home/caixiao/projects/3d_lib/obj'
dataset = load_text2obj_dataset(data_dir=root)
dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=4)
data = iter(dataloader)
# # info =next(data)
# # print(info)
#

# print(dtype)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
counter = 0
guidance_scale = 7.0
max_samples = 2
do_classifier_free_guidance = True
dtype = next(unet.parameters()).data.dtype
scale_factor = 0.18215


#ldm3d
# pipe = StableDiffusionLDM3DPipeline.from_pretrained("/mnt/hdd1/caixiao/datasets/l3dm4c")
# # print(dir(pipe))
#
# tokenizer = pipe.tokenizer
# # print(text_tokenizer)
# text_encoder_P = pipe.text_encoder
# # print(text_clip_P)
# # exit()
# text_encoder_P.cuda()
# text_encoder_P.requires_grad_(False)
#
# ae_P = pipe.vae
# ae_P.cuda()
# ae_P.requires_grad_(False)
#
#
# unet_P = pipe.unet
#
# unet_P.to("cuda")
# unet_P.requires_grad_(False)
# dtype = next(unet_P.parameters()).data.dtype


def encode_text(text):
    # text : ['a dog', 'a cat', 'a bird', ...]
    outputs = text_clip.encode(text)
    # text_inputids = tokenizer(text, padding=True, return_tensors="pt")['input_ids'].cuda()
    # text_masks = tokenizer(text, padding=True, return_tensors="pt")['attention_mask'].cuda()
    # outputs = text_encoder_P(text_inputids, text_masks)['last_hidden_state']
    return outputs


def decode_images(z, unflatten=False):
    if unflatten:
        z = rearrange(z, 'B C T H W -> (B T) C H W').to(torch.float32)
    z = z * (1. / scale_factor)
    images = ae.decode(z) # (N*T, C, H, W)
    return images

while True:
    try:
        info = next(data)
        scheduler.set_timesteps(50, device=device)
        timesteps = scheduler.timesteps
        # print(timesteps)
        # print(info['depth'].shape)
        # print(info['rgb'].shape)
        # 处理图像和标签
        images = info['rgbd'].cuda()
        text = info['caption']

        # images = images.squeeze(0)
        # image = rearrange(images, 'C T H W -> (T) C H W')

        # print(text.size)
        # text = "A picture of some lemons on a table"
        if isinstance(text, list) or isinstance(text, tuple):
            batch_size = len(text)
        elif isinstance(text, str):
            batch_size = 1
            text = [text]
        else:
            raise NotImplementedError

        if not do_classifier_free_guidance:
            text_latent = encode_text(text).to(dtype)
        else:
            text = text + [''] * batch_size
            text_latent = encode_text(text).to(dtype)


        alpha = 1.0
        images_latent = torch.randn((batch_size, 4, 1, 64, 64), dtype=dtype, device=device) * ((alpha**2/(1+alpha**2))**0.5) \
                + torch.randn((batch_size, 4, 12, 64, 64), dtype=dtype, device=device) * ((1/(1+alpha**2))**0.5)
        # images_latent = torch.randn((batch_size, 4, 12, 64, 64), dtype=dtype, device=device)

        for t in timesteps:
            if do_classifier_free_guidance:
                input_images_latent = torch.cat([images_latent, images_latent], 0)
            else:
                input_images_latent = images_latent

            timestep = t.repeat(input_images_latent.shape[0]).contiguous()

            dataset = ['text2obj'] * input_images_latent.shape[0]
            noise_pred = unet(
                input_images_latent,
                timestep,
                text_latent,
                dataset)

            # input_images_latent = input_images_latent[:,:,0,:,:]
            # print(input_images_latent.shape)
            # noise_pred = unet_P(input_images_latent,timestep,text_latent,dataset).sample
            # print(noise_pred.shape)

            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_cond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # images_latent = self.scheduler.step(noise_pred, t, images_latent, **extra_step_kwargs).prev_sample
            images_latent = scheduler.step(noise_pred, t, images_latent).prev_sample

        #     images_latent = rearrange(images_latent, 'B C  H W -> (B ) C H W',B=batch_size).to(torch.float32)

        # z = rearrange(z, 'B T C H W ->(B C) T H W', B=batch_size)
        # out_ae = ae.decode(z)
        #     out_unet = ae.decode(images_latent)
        #     images_latent = images_latent * (1. / scale_factor)
        #     out_unet = ae_P.decode(images_latent).sample
        out_unet = decode_images(images_latent, unflatten=True)
        out_unet = rearrange(out_unet, '(B T) C H W ->B T C H W', B=batch_size)
        out_unet = ((out_unet + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu() # (t, h, w, c)
        print(out_unet.shape)
        # out_unet = out_unet.numpy()
            # print(out_unet)
            # exit()
        # images_latent = rearrange(images_latent, '(B ) C H W -> B C  H W', B=batch_size).to(dtype)
            # images_latent = rearrange(images_latent, '(B T) C H W -> B C T H W', B=batch_size).to(dtype)
        # out_ae = rearrange(out_ae, '(B T) C H W ->B T C H W',B=batch_size)




            # out_unet = rearrange(out_unet, '(B T) C H W ->B T C H W',B=batch_size)
        #
        # out = ae.decode(z)  # (N*T, C, H, W)
        # depth_ae = out_ae[0, 0, 3, :, :]
        # img_ae = out_ae[0, 0, :3, :, :]

        depth_unet = out_unet[0, 0, 3, :, :]
        img_unet = out_unet[0, 0, :3, :, :]

        to_img = ToPILImage()
        # img_ae = to_img(img_ae)
        # depth_ae = to_img(depth_ae)
        img_unet = to_img(img_unet)
        depth_unet = to_img(depth_unet)
            # img_unet.save(f"/home/caixiao/projects/3DGen/test_unet/unet_{t}.png")
            # depth_unet.save(f"/home/caixiao/projects/3DGen/test_unet/dep_unet_{t}.png")
            # print(img.shape)
        print(counter)
        # img_ae.save(f"/home/caixiao/projects/3DGen/test_unet/ae{counter}.png")
        # depth_ae.save(f"/home/caixiao/projects/3DGen/test_unet/dep_ae{counter}.png")

        img_unet.save(f"/home/caixiao/projects/3DGen/test_unet/unet{counter}.png")
        depth_unet.save(f"/home/caixiao/projects/3DGen/test_unet/dep_unet{counter}.png")
        # print(img.shape)
        counter += 1  # 更新计数器
        if counter >= max_samples:  # 如果已经返回了足够的样本
            break  # 提前终止迭代
    except StopIteration:
        break  # 如果已经到达了数据集的末尾，终止迭代