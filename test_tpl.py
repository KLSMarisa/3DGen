import os
import csv
from typing import Iterable
from data.text2obj_dataset import Text2ObjDataset
from torch.utils.data import Dataset, IterableDataset, ChainDataset
from torch.utils.data import DataLoader
from modules.StableDiffusion import UNetModel, FrozenCLIPEmbedder, AutoencoderKL, NormResBlock, FrozenOpenCLIPEmbedder
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
import warnings
import clip
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, PNDMScheduler, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionLDM3DPipeline
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from modules.networks import EMAModel, EMAModel_for_deepspeed_stage3
from torchvision.transforms import ToPILImage
import time
import random
import json

# load data
def load_text2obj_dataset(
    *,
    data_dir,
    # view_num,
    center_crop=True,
    random_flip=False,
):

    """
    For a dataset, create a generator over text2obj (text-objaverse pair) data samples.
    The format of data is a dict. The meaning of each element is as follows,
    "video": the video, a NFCHW float tensor
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
    
    with open('/home/linzhuohang/3DGen/data/id.json', 'r') as f:
    # with open('/mnt/hdd2/caixiao/data/objaverse/metadata/merged_data.json', 'r') as f:
        data3 = json.load(f)
    # with open('/mnt/hdd1/caixiao/data/objaverse_1.0/utils/data_select/sd_scores.json', 'r') as f:
    #     data3 = json.load(f)
    
    with open('/mnt/hdd1/caixiao/data/objaverse_1.0/caption/merged.json', 'r') as f:
        data2 = json.load(f)

        
        
    
    with open(data_dir,'r') as f:
        data = json.load(f)
        print("data_dir: ",data_dir)
        for line in data:
            #print(line)
    
            img_info = dict()

            # img_info['filename'] = line.split('\n')[0]
            id = line.split('/')[-1]
            # print(line)
            #id = line['obj_id']
            # caption = line['cap3d']
            # caption = line['3dtopia']
            # print(img_info['filename'])
            # exit()
            # white test
            # file_path = img_info['filename'].replace("/3dgen/nerf/white_test2", "/data/pv_views_v2")
            if not id in data3:
                continue
            if not str(id) in data2:
                continue
            caption = data2[id]
            # print(caption)
            # if len(caption)>96:
            #     continue
            # img_info['filename'] = img_info['filename'].replace("mnt/hdd1/caixiao/data/pv_views_v2", "mnt/hdd1/data")
            img_info['filename'] = os.path.join("/mnt/hdd1/data", data3[id].split('pv_views_v2/')[1])
            # print(img_info['filename'])
            # img_info['filename'] = img_info['filename']
            img_info['caption'] = caption
            # print(img_info['filename'])
            # print(caption)
            # exit()
            img_infos.append(img_info)
            
            ## black
            # img_info['filename'] = img_info['filename'].replace("/3dgen/nerf/white_test2", "/data/pv_views_v2")
            # ## ori
            # if not img_info['filename'] in data2:
            #     continue
            # caption = data2[img_info['filename']]
            # if len(caption)>96:
            #     continue
            # img_info['caption'] = caption
            # # print(img_info['filename'])
            # # exit()
            # img_infos.append(img_info)
    # img_infos = img_infos[:6500]
    # img_infos = sorted(img_infos, key=lambda x: x['filename'])
    print(f"load {len(img_infos)} objs in Text2ObjDataset")
    # exit()

    return Text2ObjDataset(
        data_dir=data_dir,
        # view_num=view_num,
        img_list = img_infos,
        random_flip = random_flip,
        center_crop = center_crop,
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



##### 测试unet


##  载入各网络模型
## clip
text_clip = FrozenCLIPEmbedder()
# ckpt_path = '/mnt/hdd1/liujianzhi/ckpt/stable-diffusion-v1-4/text_encoder/pytorch_model.bin'
# ckpt_path = '/mnt/hdd1/caixiao/3dgen/models/StableDiffusion/text_encoder/pytorch_model.bin'
ckpt_path = '/mnt/hdd1/caixiao/deeplearning/StableDiffusion/models/StableDiffusion/text_encoder/pytorch_model.bin'
text_clip.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
text_clip.half()
text_clip.cuda()

# text_clip = FrozenOpenCLIPEmbedder(layer='penultimate')
# ckpt_path = '/mnt/hdd1/caixiao/3dgen/models/StableDiffusion/v2-1_512-ema-pruned-openclip.ckpt'
# text_clip.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
# text_clip.half()
# text_clip.cuda()

## vae
ae = AutoencoderKL(
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

# ckpt_path = '/home/caixiao/projects/3DGen/models/diffusion_pytorch_model_1_3.bin'
# ckpt_path = '/mnt/hdd1/caixiao/3dgen/models/StableDiffusion/v2-1_512-ema-pruned-autoencoder.ckpt'
ckpt_path = '/mnt/hdd1/caixiao/deeplearning/StableDiffusion/models/StableDiffusion/v2-1_512-ema-pruned-autoencoder.ckpt'
ae.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

ae.half()
ae.cuda()
ae.eval()
ae.requires_grad_(False)




## unet
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
    num_head_channels=64,  # need to fix for flash-attn
    # num_heads=8,
    use_spatial_transformer=True,
    use_linear_in_transformer=True,
    transformer_depth=1,
    context_dim=1024,
    legacy=False,
    spatial_finetune_blocks=1,
)



# ckpt_path = '/home/caixiao/projects/3DGen/models/unet/diffusion_pytorch_model_1_4.bin'
# ckpt_path = '/mnt/hdd1/caixiao/3dgen/models/StableDiffusion/v2-1_512-ema-pruned-diffusionmodel.ckpt'
# unet.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
# ckpt_path = '/mnt/hdd1/caixiao/3dgen/models/StableDiffusion/v2-1_512-ema-pruned-diffusionmodel.ckpt'
#         # unet.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
# unet.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
# # unet.requires_grad_(False)
# print('ori')

unet.apply_extra_spatial(['text2obj'])

# _sd = {}
# # ckpt_path = '/mnt/hdd1/caixiao/3dgen_sd2.1/checkpoints/stage1_sd-step=195000.ckpt/pytorch_model.bin'
# # ckpt_path = '/mnt/hdd1/caixiao/3dgen/nerf/3dgen_sd_bw_test_b/checkpoints/stage1_sd-step=40000.ckpt/40k.bin'
# # ckpt_path = '/mnt/hdd1/caixiao/qilian/MV_SD/45K.bin'
# ckpt_path = '/mnt/hdd1/caixiao/3dgen_sd2.1/checkpoints/stage1_sd-step=195000.ckpt/pytorch_model.bin'
# # ckpt_path = '/home/caixiao/projects/3DGen/test/stage1-step=125000.ckpt/pytorch_model.bin'
# sd = torch.load(ckpt_path, map_location='cpu')

  
# # unet.load_state_dict(_sd, strict=True)
# for k, v in sd.items():
#     if '_forward_module.unet.' in k:
#         _sd[k.replace('_forward_module.unet.', '')] = v
# unet.load_state_dict(_sd, strict=True)

unet.apply_extra_triattn()

ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/TPL_finetune/checkpoints/TPL-step=130000.ckpt/130k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/stage1_with_sd/45k.bin'
sd = torch.load(ckpt_path, map_location='cpu')

_sd={}
# for k, v in sd.items():
#     if 'unet.' in k:
#         _sd[k.replace('unet.', '')] = v
# unet.load_state_dict(_sd, strict=True)
for k, v in sd.items():
    if 'unet.' in k:
        
        _sd[k.replace('unet.', '')] = v
    
    
unet.load_state_dict(_sd, strict=True)
unet.eval()
unet.cuda()
unet.requires_grad_(False)
unet.to(torch.bfloat16)

scheduler = DDIMScheduler(
            beta_start         =     0.00085,
            beta_end           =     0.0120,
            beta_schedule      =     'scaled_linear',
            num_train_timesteps=     1000,
            clip_sample        =     False,
            set_alpha_to_one   =     False,
            steps_offset       =     1,
            rescale_betas_zero_snr=True
        )





##### 以下可理解为main函数
## 读取测试数据
os.environ['WORLD_SIZE']='1'
os.environ['LOCAL_RANK']='0'
# root = '/home/caixiao/projects/3d_lib/obj/pv_views'
root = '/mnt/hdd1/caixiao/data/objaverse_1.0/utils/data_select/rgb2.json'
dataset = load_text2obj_dataset(data_dir=root)
dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=4)
data = iter(dataloader)


# 一些全局参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
guidance_scale = 7.0
do_classifier_free_guidance = True
scale_factor = 0.18215
dtype = torch.float16

# while循环指示器和循环轮数设定
max_samples = 10 #只进行max次循环，即读取几个数据，最多8个，测试数据只放了8个
counter = 0

## 决定是否用pipe替换我们的网络
is_pipe_clip = 0
is_pipe_unet = 0
is_pipe_ae = 0


def encode_text(text, is_pipe):
    # text : ['a dog', 'a cat', 'a bird', ...]
    if is_pipe:
        text_inputids = tokenizer(text, padding=True, return_tensors="pt")['input_ids'].cuda()
        text_masks = tokenizer(text, padding=True, return_tensors="pt")['attention_mask'].cuda()
        outputs = text_encoder_P(text_inputids, text_masks)['last_hidden_state']
    else:
        outputs = text_clip.encode(text)

    return outputs


def decode_images(z, is_pipe_unet, is_pipe_ae, unflatten=False):
    if unflatten:
        if is_pipe_unet:
            z = rearrange(z, 'B C  H W -> (B ) C H W').to(torch.float16)
        else:
            z = rearrange(z, 'B C T H W -> (B T) C H W').to(torch.float16)
    z = z * (1. / scale_factor)
    if is_pipe_ae:
        images = ae_P.decode(z).sample  # (N*T, C, H, W)
    else:
        images = ae.decode(z)  # (N*T, C, H, W)
    return images

timestamp = int(time.time())
while True:
    try:
        info = next(data)
        scheduler.set_timesteps(50, device=device)
        timesteps = scheduler.timesteps
        print(scheduler.num_train_timesteps)
        
        
        # 从数据集中随机获取
        #text = info['caption']
        # print(text)
        # print(info['file'])
        
        # 或者可以手动指定
        # text = "a wooden cart with wheels"
        text = 'a young beautiful anime girl with blue eyes, long black hair, wearing a yellow dress, anime style,highly detailed'
        if isinstance(text, list) or isinstance(text, tuple):
            batch_size = len(text)
        elif isinstance(text, str):
            batch_size = 1
            text = [text]
        else:
            raise NotImplementedError

        if not do_classifier_free_guidance:
            text_latent = encode_text(text, is_pipe=is_pipe_clip).to(dtype)
        else:
            text = text + [''] * batch_size
            text_latent = encode_text(text, is_pipe=is_pipe_clip).to(dtype)

        # print(text_latent.shape)
        alpha = 1.0
        images_latent = torch.randn((batch_size, 4, 1, 64, 64), dtype=dtype, device=device) * ((alpha**2/(1+alpha**2))**0.5) \
                + torch.randn((batch_size, 4, 3, 64, 64), dtype=dtype, device=device) * ((1/(1+alpha**2))**0.5)
        # images_latent = torch.randn((batch_size, 4, 12, 64, 64), dtype=dtype, device=device)

        # pipe只接受最多4D输入
        if is_pipe_unet:
            images_latent = images_latent[:,:,0,:,:] # B*4*64*64
        # print(timesteps)
        # diffusion
        for t in timesteps:
            if do_classifier_free_guidance:
                input_images_latent = torch.cat([images_latent, images_latent], 0)
            else:
                input_images_latent = images_latent
            # t = t*100
            timestep = t.repeat(input_images_latent.shape[0]).contiguous()
            dataset = ['text2obj'] * input_images_latent.shape[0]
            # print(timestep)
            # exit()
            # print('input:',input_images_latent.shape)
            
            unet = unet.to(torch.bfloat16)
            input_images_latent = input_images_latent.to(torch.bfloat16)
            text_latent = text_latent.to(torch.bfloat16)
            timestep = timestep.to(torch.bfloat16)
            #print(timestep.type(),dataset.type())
            if is_pipe_unet:
                noise_pred = unet_P(input_images_latent,timestep,text_latent,dataset).sample
            else:
                noise_pred = unet(
                    input_images_latent,
                    timestep,
                    text_latent,
                    dataset)

            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_cond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            # t = t //100
            # images_latent = self.scheduler.step(noise_pred, t, images_latent, **extra_step_kwargs).prev_sample
            images_latent = scheduler.step(noise_pred, t, images_latent).prev_sample

        # print(images_latent)
        out_unet = decode_images(images_latent, is_pipe_unet=is_pipe_unet, is_pipe_ae=is_pipe_ae, unflatten=True)
        if not is_pipe_unet:
            out_unet = rearrange(out_unet, '(B T) C H W ->B T C H W', B=batch_size)
            out_unet = out_unet[0,:,:,:,:]
        # print(out_unet[0,3,...])
        out_unet = ((out_unet + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu()
        # out_unet = out_unet.numpy()
        print(out_unet.shape)
        # print(out_unet)
        # exit()
        # depth_unet = out_unet[0, 3, :, :]
        
        random_number = random.randint(0, 99999)
        # os.makedirs(f"/mnt/hdd1/caixiao/3dgen/test/stage2/fine_35k", exist_ok=True)
        # os.makedirs(f"/mnt/hdd1/caixiao/test/3dgen/ablatiaon/with_sd/4k", exist_ok=True)
        os.makedirs(f"path_to_save", exist_ok=True)
        # os.makedirs(f"/mnt/nfs/VGG/caixiao/data_test/test/stage2/fine_70k", exist_ok=True)
        for i in range(3):
            img_unet = out_unet[i, :3, :, :]
            # print(img_unet.shape)
            # print(i)
            # depth_unet = out_unet[0, 0, 3, :, :]
            # img_unet = out_unet[0, 0, :3, :, :]
            # exit()
            to_img = ToPILImage()
            img_unet = to_img(img_unet)
            # depth_unet = to_img(depth_unet)
            
            # os.makedirs(f"/mnt/nfs/caixiao/test/3DGEN/stage1/step_125000", exist_ok=True)
            # img_unet.save(f"/mnt/nfs/caixiao/test/3DGEN/stage1/step_125000/unet_ori_{counter}_{timestamp}_{text}.png")
            # depth_unet.save(f"/home/caixiao/projects/3DGen/test/test_stage1/dep_50000_{counter}_{timestamp}_{text}_{info['id']}.png")
            # img_unet.save(f"/mnt/hdd1/caixiao/test/3dgen/ablatiaon/with_sd/4k/{timestamp}_{counter}_{i}_{text}.png")
            img_unet.save(f"outputs/{timestamp}_{counter}_{i}_{text}.png")
            # exit()
        

        print(counter)
        # exit()
        counter += 1  # 更新计数器
        if counter >= max_samples:  # 如果已经返回了足够的样本
            break  # 提前终止迭代
    except StopIteration:
        break  # 如果已经到达了数据集的末尾，终止迭代