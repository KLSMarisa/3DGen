import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from torch.utils.data import Dataset, IterableDataset, ChainDataset
from torch.utils.data import DataLoader
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from deepspeed.runtime.lr_schedules import WarmupLR
import warnings
import clip
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, PNDMScheduler, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from modules.mipnerf.ray_utils import Rays, convert_to_ndc, namedtuple_map
# from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, PNDMScheduler, DDPMScheduler
# from transformers import CLIPTokenizer, CLIPTextModel
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from modules.networks import EMAModel, EMAModel_for_deepspeed_stage3
from modules.StableDiffusion import AutoencoderKL,FrozenCLIPEmbedder
from modules.StableDiffusion.StableDiffusionModel import UNetModel
from modules.mipnerf.nplanes_test_model import MipNeRF
from modules.mipnerf.pose_utils import visualize_depth, visualize_normals, to8b
from modules.mipnerf.loss import NeRFLoss, mse_to_psnr

from modules.instantmesh.encoder.dino_wrapper import DinoWrapper
# from modules.instantmesh.decoder.transformer_ori import TriplaneTransformer
from modules.instantmesh.decoder.transformer import TriplaneTransformer
from modules.instantmesh.renderer.synthesizer_test import TriplaneSynthesizer
from modules.instantmesh.geometry.camera.perspective_camera import PerspectiveCamera
# from modules.instantmesh.geometry.render.neural_render import NeuralRender
from modules.instantmesh.geometry.rep_3d.flexicubes_geometry import FlexiCubesGeometry
# from modules.instantmesh.utils.mesh_util import xatlas_uvmap

from torchvision.transforms import Resize, ToPILImage, ToTensor
from data.obj2render_dataset import read_obj
import time
import lpips
import collections
import imageio
from data.obj2render_instant_test import Obj2Render_Dataset
import json
import random
import rembg
from PIL import Image
from torchvision import transforms as Trans
import mcubes
import trimesh

os.environ['WORLD_SIZE']='1'
os.environ['LOCAL_RANK']='0'
# os.environ["U2NET_HOME"] = ''
def load_obj2render_dataset(
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
    # df = pd.read_csv('/home/caixiao/projects/3d_lib/data/select_all.csv')
    # # len(df)
    # for i in range(len(df)):
    #     # 选择你想要的行，例如第3行
    #     row = df.iloc[i]
    #     # 将行数据转换为列表
    #     data = row.tolist()
    #     if data[1] != -1:
    #         img_infos.append(data)
    # addition = '/mnt/hdd1/caixiao/data/pv_views_64'
    with open('/mnt/hdd1/caixiao/data/objaverse_1.0/caption/merged.json', 'r') as f:
        data2 = json.load(f)
    
    with open(data_dir,'r') as f:
        data = json.load(f)
    
    # keys = data2.keys()
    # key_list = list(keys) 
    for line in data:
        
        id = line.split('/')[-1]
        # exit()
        img_info = dict()
        img_info['filename'] = line
        
        if not str(id) in data2:
            # print(img_info['filename'])
            continue
        
        caption = data2[id]
        
        # if len(caption)>96:
        #     continue
        # if len(img_info['plane'])==0:
        #     continue
        # print(img_info['filename'])
        # exit()
        # img_info['filename'] = img_info['filename'].replace("/mnt/hdd1/caixiao/data/pv_views", "/mnt/hdd1/caixiao/mount2")
        img_info['caption'] = caption
        img_infos.append(img_info)
    # path_lists.sort()
    
    # with open('/home/caixiao/projects/3d_lib/caption/filter_data_dir.txt','w') as file:
    #     for path_list in tqdm(path_lists):
    #         if path_list in Filter:
    #             continue
    #         path = os.path.join(data_dir, path_list)
    #         file.write(path+'\n')
    #         img_info = dict()
    #         img_info['filename'] = path
    #         # print(path)
    #         img_infos.append(img_info)
    
    
    # path_lists2 = os.listdir(os.path.join(addition))
    # # path_lists.sort()
    # for path_list in path_lists:
    #     path = os.path.join(data_dir, path_list)
    #     img_info = dict()
    #     img_info['filename'] = path
    #     # print(path)
    #     img_infos.append(img_info)
    # with open('/mnt/nfs/caixiao/datasets/objaverse/hf-objaverse-v1/downloaded.txt','r') as f:
    # with open('/home/caixiao/projects/3d_lib/caption/filter_data_dir.txt','r') as f:
    # with open('/home/caixiao/projects/3d_lib/img/filter_data_dir.txt','r') as f:
    #     for line in f:
    #         # print(line.split('\n'))
    
    #         img_info = dict()
    #         img_info['filename'] = line.split('\n')[0]
    #         # print(img_info['filename'])
    #         # exit()
    #         img_infos.append(img_info)
    # img_infos = img_infos[3000:4000]
    # img_infos = sorted(img_infos, key=lambda x: x['filename'])
    print(f"load {len(img_infos)} objs in Obj2Render_Dataset")
    # exit()

    return Obj2Render_Dataset(
        data_dir=data_dir,
        # view_num=view_num,
        img_list = img_infos,
        random_flip = random_flip,
        center_crop = center_crop,
    )


grid_res = 128
grid_scale = 2.0
deformation_multiplier = 4.0
input_size = 320#320
render_size = 192

encoder = DinoWrapper(
    model_name='/mnt/hdd1/caixiao/deeplearning/instant/dino',
    freeze=True,
)

_encoder = {}
# ckpt_path = '/mnt/hdd1/caixiao/mount/instant_nerf_large.ckpt'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/instant_nerf_ori_rgbgt_en_de/checkpoints/renderer_instantmesh-step=8000.ckpt/8k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/instant_nerf_ori_4attn/checkpoints/renderer_instantmesh-step=19000.ckpt/19k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/instant_nerf_ori_3attn/checkpoints/renderer_instantmesh-step=13000.ckpt/13k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/instant_nerf_ori_2attn/checkpoints/renderer_instantmesh-step=30000.ckpt/30k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/instant_nerf_ori_2attn/checkpoints/renderer_instantmesh-step=25000.ckpt/25k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/instant_nerf_ori_2attn/checkpoints/renderer_instantmesh-step=20000.ckpt/20k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_stage1/34k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_stage2/19k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_stage1/z40k.bin'
#ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_stage1/30k_v1.bin'
ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_stage3/2k.bin'
ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_stage3/6k.bin'
ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_stage3/15k_v1cos.bin'
ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_all/48k.bin'
ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_all/5k.bin'
ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_all/50k.bin'
ckpt_path = '/mnt/hdd2/caixiao/deeplearning/ckpt/3dgen/AVS/checkpoints/renderer_instantmesh-step=94000.ckpt/94k.bin'
ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/instant_nerf_ori_attn_all+renderer_rgb/checkpoints/renderer_instantmesh-step=66000.ckpt/66k.bin'
# ckpt_path = '/mnt/hdd2/caixiao/deeplearning/ckpt/3dgen/AVS_goodcase3/checkpoints/renderer_instantmesh-step=10000.ckpt/10k.bin'
# ckpt_path = '/mnt/hdd2/caixiao/deeplearning/ckpt/3dgen/AVS_goodcase3/checkpoints/renderer_instantmesh-step=1000.ckpt/1k.bin'
# state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
state_dict = torch.load(ckpt_path, map_location='cpu')

for k, v in state_dict.items():
    # print(k)
    # if 'lrm_generator.encoder.' in k:
    #    _encoder[k.replace('lrm_generator.encoder.', '')] = v
    # if '_forward_module.encoder.' in k:
    #    _encoder[k.replace('_forward_module.encoder.', '')] = v
    if 'encoder.model.encoder.' in k:
        _encoder[k.replace('encoder.model.encoder.', 'model.encoder.')] = v
    elif 'encoder.' in k and 'text_clip.' not in k:
        _encoder[k.replace('encoder.', '')] = v
# print(_encoder)
# exit()
# state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
encoder.load_state_dict(_encoder, strict=True)

encoder.to(torch.bfloat16)
encoder.eval()
encoder.cuda()
encoder.requires_grad_(False)


transformer = TriplaneTransformer(
    inner_dim=1024, #1280,#1024 
    num_layers=16, #24, #16
    num_heads=16, #20,#16
    image_feat_dim=768,
    triplane_low_res=32, 
    triplane_high_res=64, 
    triplane_dim=80,
)


_decoder = {}
# ckpt_path = '/mnt/hdd1/caixiao/mount/instant_nerf_large.ckpt'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/instant_nerf_ori_rgbgt_en_de/checkpoints/renderer_instantmesh-step=8000.ckpt/8k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/instant_nerf_ori_4attn/checkpoints/renderer_instantmesh-step=19000.ckpt/19k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/instant_nerf_ori_3attn/checkpoints/renderer_instantmesh-step=13000.ckpt/13k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/instant_nerf_ori_2attn/checkpoints/renderer_instantmesh-step=30000.ckpt/30k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/instant_nerf_ori_2attn/checkpoints/renderer_instantmesh-step=25000.ckpt/25k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/instant_nerf_ori_2attn/checkpoints/renderer_instantmesh-step=20000.ckpt/20k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_stage1/34k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_stage2/19k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_stage1/z40k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_stage1/10k_v1.bin'
# ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_stage1/30k_v2.bin'
# ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_stage1/20k_v3.bin'
ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_stage3/2k.bin'
ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_stage3/6k.bin'
ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_stage3/15k_v1cos.bin'
ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_all/48k.bin'
ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_all/5k.bin'
ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_all/50k.bin'
ckpt_path = '/mnt/hdd2/caixiao/deeplearning/ckpt/3dgen/AVS/checkpoints/renderer_instantmesh-step=94000.ckpt/94k.bin'
ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/instant_nerf_ori_attn_all+renderer_rgb/checkpoints/renderer_instantmesh-step=66000.ckpt/66k.bin'
# ckpt_path = '/mnt/hdd2/caixiao/deeplearning/ckpt/3dgen/AVS_goodcase3/checkpoints/renderer_instantmesh-step=10000.ckpt/10k.bin'
# ckpt_path = '//mnt/hdd2/caixiao/deeplearning/ckpt/3dgen/AVS_goodcase3/checkpoints/renderer_instantmesh-step=1000.ckpt/1k.bin'
# state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
state_dict = torch.load(ckpt_path, map_location='cpu')

for k, v in state_dict.items():
    # if 'lrm_generator.transformer.' in k:
    #     _decoder[k.replace('lrm_generator.transformer.', '')] = v
    if 'transformer.' in k and 'text_clip.' not in k:
       _decoder[k.replace('transformer.', '')] = v
# state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
transformer.load_state_dict(_decoder, strict=True)
transformer.to(torch.bfloat16)
transformer.eval()
transformer.cuda()
transformer.requires_grad_(False)

## nerf_module
renderer = TriplaneSynthesizer(
    triplane_dim=80, 
    samples_per_ray=128,
)


_renderer = {}
ckpt_path = '/mnt/hdd1/caixiao/mount/instant_nerf_large.ckpt'
ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_all/5k.bin'
ckpt_path = '/mnt/hdd1/caixiao/3dgen/ckpt/new_renderer_all/50k.bin'
ckpt_path = '/mnt/hdd2/caixiao/deeplearning/ckpt/3dgen/AVS/checkpoints/renderer_instantmesh-step=94000.ckpt/94k.bin'
ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/instant_nerf_ori_attn_all+renderer_rgb/checkpoints/renderer_instantmesh-step=66000.ckpt/66k.bin'
# ckpt_path = '/mnt/hdd2/caixiao/deeplearning/ckpt/3dgen/MOE/animal/checkpoints/renderer_instantmesh-step=1000.ckpt/1k.bin'
# ckpt_path = '/mnt/hdd2/caixiao/deeplearning/ckpt/3dgen/MOE/animal/checkpoints/renderer_instantmesh-step=1000-v1.ckpt/1k.bin'
# ckpt_path = '/mnt/hdd2/caixiao/deeplearning/ckpt/3dgen/AVS_goodcase3/checkpoints/renderer_instantmesh-step=10000.ckpt/10k.bin'
# ckpt_path = '/mnt/hdd2/caixiao/deeplearning/ckpt/3dgen/AVS_goodcase3/checkpoints/renderer_instantmesh-step=1000.ckpt/1k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/instant_nerf_ori_rgbgt_en_de/checkpoints/renderer_instantmesh-step=8000.ckpt/8k.bin'
# state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
state_dict = torch.load(ckpt_path, map_location='cpu')

for k, v in state_dict.items():
    # if 'lrm_generator.synthesizer.' in k:
    #     _renderer[k.replace('lrm_generator.synthesizer.', '')] = v
    if 'renderer.' in k:
        _renderer[k.replace('renderer.', '')] = v
renderer.load_state_dict(_renderer, strict=True)
renderer.eval()
renderer.cuda() 
renderer.to(torch.float16)
renderer.requires_grad_(False)


text_clip = FrozenCLIPEmbedder()
# ckpt_path = '/mnt/hdd1/liujianzhi/ckpt/stable-diffusion-v1-4/text_encoder/pytorch_model.bin'
# ckpt_path = '/mnt/hdd1/caixiao/3dgen/models/StableDiffusion/text_encoder/pytorch_model.bin'
ckpt_path = '/mnt/hdd1/caixiao/deeplearning/StableDiffusion/models/StableDiffusion/text_encoder/pytorch_model.bin'
text_clip.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
text_clip.to(torch.bfloat16)
text_clip.cuda()


scale_factor = 0.18215
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

ae.requires_grad_(False)
ckpt_path = '/mnt/hdd1/caixiao/deeplearning/StableDiffusion/models/StableDiffusion/v2-1_512-ema-pruned-autoencoder.ckpt'
# ckpt_path = '/mnt/hdd1/model/StableDiffusion/v2-1_512-ema-pruned-autoencoder.ckpt'
ae.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
ae.eval()
ae.cuda()
ae.to(torch.bfloat16)

## unet
unet = UNetModel(
    use_checkpoint=True,
    use_bf16=True,
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


## Stage 1, train added spatial
unet.apply_extra_spatial(['text2obj'])

## Stage 2, train added tri_atten
# if self.stage > 1:
#     # self.unet.requires_grad_(False)
#     _sd = {}
#     ckpt_path = ''
#     sd = torch.load(ckpt_path, map_location='cpu')
#     for k, v in sd.items():
#         if '_forward_module.unet.' in k:
#             _sd[k.replace('_forward_module.unet.', '')] = v
#     self.unet.load_state_dict(_sd, strict=False)
unet.apply_extra_triattn()

## Stage 3, finetune spatial triplane and renderer
# if self.stage > 2:
# self.unet.requires_grad_(False)
_sd = {}
# ckpt_path = '/mnt/hdd1/caixiao/qilian/MV_SD/45K.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/stage3_zerosnr_finetune/zero_snr_finetune/checkpoints/stage2_sd-step=70000.ckpt/70k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/turbo_reflow2/checkpoints/turbo-step=2000.ckpt/2k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/turbo_reflow1/checkpoints/turbo-step=6000.ckpt/6k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/turbo_reflow1/checkpoints/turbo-step=1000-v1.ckpt/1k.bin'
ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/stage3_zerosnr_finetune/zero_snr_finetune/checkpoints/stage2_sd-step=70000.ckpt/70k.bin'
ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/TPL_finetune/checkpoints/TPL-step=130000.ckpt/130k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/qilian_v1_2/checkpoints/qilian_v1-step=79000.ckpt/79k.bin'
# ckpt_path = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/qilian_v1_c4+76_hlr2/checkpoints/qilian_v1-step=2500.ckpt/2500.bin'
sd = torch.load(ckpt_path, map_location='cpu')



for k, v in sd.items():
    if 'unet.' in k:
        ## only first train added
        # if '_forward_module.unet.out.2.weight' in k:
        #     new_weights = torch.randn(80, 320, 3, 3) / 100.
        #     new_weights[:4] = v
        #     print(k)
        #     v = new_weights
        # if '_forward_module.unet.out.2.bias' in k:
        #     new_weights = torch.randn(80) / 100.
        #     new_weights[:4] = v
        #     print(k)
        #     v = new_weights
        _sd[k.replace('unet.', '')] = v
    # if '_forward_module.unet.' in k:
    #     ## only first train added
    #     # if '_forward_module.unet.out.2.weight' in k:
    #     #     new_weights = torch.randn(80, 320, 3, 3) / 100.
    #     #     new_weights[:4] = v
    #     #     print(k)
    #     #     v = new_weights
    #     # if '_forward_module.unet.out.2.bias' in k:
    #     #     new_weights = torch.randn(80) / 100.
    #     #     new_weights[:4] = v
    #     #     print(k)
    #     #     v = new_weights
    #     _sd[k.replace('_forward_module.unet.', '')] = v
    
unet.load_state_dict(_sd, strict=True)
unet.eval()
unet.cuda()
unet.requires_grad_(False)
unet.to(torch.bfloat16)
# if config['ema']:
#     self.ema_model = EMAModel(self.renderer)
# else:
#     self.ema_model = None

# from modules.src.scheduler_perflow import PeRFlowScheduler
# self.scheduler = PeRFlowScheduler(
#     num_train_timesteps=1000,
#     beta_start = 0.00085,
#     beta_end = 0.012,
#     beta_schedule = "scaled_linear",
#     prediction_type= 'epsilon',
#     t_noise = 1,
#     t_clean = 0,
#     num_time_windows= 4,
# )
scheduler = DDIMScheduler(
    beta_start         =     0.00085,
    beta_end           =     0.0120,
    beta_schedule      =     'scaled_linear',
    # num_train_timesteps=     1000,
    clip_sample        =     False,
    set_alpha_to_one   =     False,
    steps_offset       =     1,
    rescale_betas_zero_snr=True
)


# 一些全局参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
guidance_scale = 7.0
do_classifier_free_guidance = True
scale_factor = 0.18215
dtype = torch.bfloat16

# while循环指示器和循环轮数设定
max_samples = 1000 #只进行max次循环，即读取几个数据，最多8个，测试数据只放了8个
counter = 0


root = '/mnt/hdd1/caixiao/3dgen/data/v1.json'
root = '/home/caixiao/projects/3DGen/data/objaverse_qilian_building.json'
root = '/home/caixiao/projects/3DGen/data/rgb2.json'
root = '/mnt/hdd1/caixiao/data/objaverse_1.0/utils/data_select/rgb2.json'
# root = '/home/caixiao/projects/3d_lib/img/filter_data_v2.json'
# root = '/mnt/hdd1/caixiao/3dgen/nerf/white_test/white.json' ## test white bg
dataset = load_obj2render_dataset(data_dir=root)
dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=4)
data = iter(dataloader)

@torch.no_grad()
def encode_text(text):
    # text : ['a dog', 'a cat', 'a bird', ...]
    # if is_pipe:
    #     text_inputids = tokenizer(text, padding=True, return_tensors="pt")['input_ids'].cuda()
    #     text_masks = tokenizer(text, padding=True, return_tensors="pt")['attention_mask'].cuda()
    #     outputs = text_encoder_P(text_inputids, text_masks)['last_hidden_state']
    # else:
    outputs = text_clip.encode(text)

    return outputs

def get_trans_matrix(azimuth, elev):
    # elev = camera['elev']
    # azimuth = camera['azimuth']
    # print(azimuth)
    # print(elev)
    RT, K = instant_RTK(azimuth, elev, 1, 30)
    # C = np.asarray(camera['camera_position'][0], dtype=np.float32)
    # norm = np.linalg.norm(C)
    return RT, K

import torch.nn.functional as F
# import numpy as np


def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics


def center_looking_at_camera_pose(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """
    Create OpenGL camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4) or (3, 4)
    """
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # OpenGL camera: z-backward, x-right, y-up
    z_axis = camera_position - look_at
    z_axis = F.normalize(z_axis, dim=-1).float()
    x_axis = torch.linalg.cross(up_world, z_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1).float()
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis = F.normalize(y_axis, dim=-1).float()

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics


def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5):
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    xs = radius * np.cos(elevations) * np.cos(azimuths)
    ys = radius * np.cos(elevations) * np.sin(azimuths)
    zs = radius * np.sin(elevations)

    cam_locations = np.stack([xs, ys, zs], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()

    c2ws = center_looking_at_camera_pose(cam_locations)
    return c2ws


def get_circular_camera_poses(M=120, radius=2.5, elevation=30.0):
    # M: number of circular views
    # radius: camera dist to center
    # elevation: elevation degrees of the camera
    # return: (M, 4, 4)
    assert M > 0 and radius > 0

    elevation = np.deg2rad(elevation)

    camera_positions = []
    for i in range(M):
        azimuth = 2 * np.pi * i / M
        x = radius * np.cos(elevation) * np.cos(azimuth)
        y = radius * np.cos(elevation) * np.sin(azimuth)
        z = radius * np.sin(elevation)
        camera_positions.append([x, y, z])
    camera_positions = np.array(camera_positions)
    camera_positions = torch.from_numpy(camera_positions).float()
    extrinsics = center_looking_at_camera_pose(camera_positions)
    return extrinsics


def FOV_to_intrinsics(fov, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """
    focal_length = 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics
def instant_RTK(azimuths, elevations, cam_distance, fov):
    # azimuths = np.array([30, 90, 150, 210, 270, 330])
    # elevations = np.array([20, -10, 20, -10, 20, -10])
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    x = cam_distance * np.cos(elevations) * np.cos(azimuths)
    y = cam_distance * np.cos(elevations) * np.sin(azimuths)
    z = cam_distance * np.sin(elevations)

    cam_locations = np.stack([x, y, z], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()
    c2ws = center_looking_at_camera_pose(cam_locations)
    c2ws = c2ws.float()
    Ks = FOV_to_intrinsics(fov).float()

    # render_c2ws = get_circular_camera_poses(M=8, radius=cam_distance, elevation=20.0)
    # render_Ks = FOV_to_intrinsics(self.fov).unsqueeze(0).repeat(render_c2ws.shape[0], 1, 1)
    # self.render_c2ws = render_c2ws.float()
    # self.render_Ks = render_Ks.float()

    return c2ws, Ks

@torch.no_grad()
def decode_images(z, unflatten=False):
    if unflatten:
        z = rearrange(z, 'B C T H W -> (B T) C H W')
    z = z * (1. / scale_factor)
    images = ae.decode(z) # (N*T, C, H, W)
    return images

@torch.no_grad()
def triplane_out( _batch):
    B = _batch['images'].shape[0]
    device = _batch['images'].device
    scheduler.set_timesteps(50, device=device)
    timesteps = scheduler.timesteps
    alpha = 1.0
    guidance_scale = 7.0
    images_latent = torch.randn((_batch['images'].shape[0], 4, 1, 64, 64), dtype=dtype, device=device) * ((alpha**2/(1+alpha**2))**0.5) \
            + torch.randn((_batch['images'].shape[0], 4, 3, 64, 64), dtype=dtype, device=device) * ((1/(1+alpha**2))**0.5)
    # _batch['images'] = images_latent
    # do_classifier_free_guidance = do_classifier_free_guidance
    # print('timestep:',timesteps)
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
        # print('text:',_batch['text'].shape)
        # input_images_latent = input_images_latent[:,:4]
        # exit()
        # print(_batch['text_latent'].shape)
        # print(input_images_latent.shape)
        # print(_batch['dataset'])
        # exit()
        noise_pred = unet(
            input_images_latent,
            timestep,
            _batch['text_latent'],
            dataset)
        
        
        # 
        if do_classifier_free_guidance:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_cond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        # unet_feat = noise_pred[:,4:]
        # print("unet_feat",unet_feat.requires_grad)
        # print(unet_feat.shape)
        # print(noise_pred.shape)
        # t = t //100
        # images_latent = self.scheduler.step(noise_pred, t, images_latent, **extra_step_kwargs).prev_sample
        # images_latent = self.scheduler.step(noise_pred, t, images_latent).prev_sample
        # images_latent = torch.randn((_batch['pixels'].shape[0], 80, 1, 64, 64), dtype=self.dtype, device=device) * ((alpha**2/(1+alpha**2))**0.5) \
        #     + torch.randn((_batch['pixels'].shape[0], 80, 3, 64, 64), dtype=self.dtype, device=device) * ((1/(1+alpha**2))**0.5)
        
        images_latent = scheduler.step(noise_pred, t, images_latent).prev_sample
    
    out_unet = decode_images(images_latent, unflatten=True)
    # if not is_pipe_unet:
    #     out_unet = rearrange(out_unet, '(B T) C H W ->B T C H W', B=batch_size)
    #     out_unet = out_unet[0,:,:,:,:]
    # print(out_unet.shape)
    
    to_img = ToPILImage()
    out_unet = ((out_unet + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().unsqueeze(0)
    # out_unet = out_unet.numpy()
    # print(out_unet.shape)
    # rgb = (out_unet[0][0])
    # rgb = to_img(rgb)
    # rgb.save(os.path.join(log_dir, f'input.png'))
    # exit()
    B, N, C, H, W = out_unet.shape
    # exit()
    images = out_unet.reshape(-1, C, H, W)
    
    # # 创建一个Resize对象
    # resizer = Resize((320, 320))

    # 使用Resize对象来调整图像矩阵的大小
    # images = resizer(images)
    # images = F.interpolate(images, size=(self.input_size, self.input_size)).clamp(0, 1)
    resize_imgs = []
    for i in range(B*N):

        to_pil_image = ToPILImage()

        # 创建一个Resize对象
        resizer = Resize((320, 320))

        # 创建一个ToTensor对象，用于将PIL Image转换回Tensor
        to_tensor = ToTensor()

        # 使用ToPILImage对象将Tensor转换为PIL Image
        image_pil = to_pil_image(images[i])
        
        # print()
        # 使用Resize对象来调整PIL Image的大小
        resized_image_pil = resizer(image_pil)
        image_rmb = rembg.remove(resized_image_pil)

        # 创建一个白色背景图片
        white_bg = Image.new("RGBA", image_rmb.size, "white")

        # 将移除背景后的图片与白色背景合并
        white_img = Image.alpha_composite(white_bg, image_rmb)
        # 使用ToTensor对象将调整大小后的PIL Image转换回Tensor
        image = to_tensor(white_img)
        # print(image.shape)
        resize_imgs.append(image[:3])
    images = torch.stack(resize_imgs,0).reshape(B,N,C,input_size,input_size).to(device)
    return images
    
@torch.no_grad()
def intergrate_batch(batch, caption, test=False):
    print("------intergrate_batch--------")
    input_size = 320
    render_size = 192
    _batch = {}
    # print(batch['caption'])
    # batch['caption'] = ['A red and white sneaker with black laces.\nA white sole.\nA black heel.\nBlack laces.\nA black heel.']
    # batch['caption'] = ['A simple, minimalist chair with a red seat and a black frame.\nA flat, two-dimensional style with a limited color palette.']
    # batch['caption'] = ['A globe of the Earth, centered on Africa, with a stand underneath it.']
    # batch['caption'] = ['A row of six spherical objects, each with a distinct color and texture, are arranged in a straight line, with each one slightly offset from the one before it. The colors of the objects, from left to right, are green, gray, blue, red, and yellow. The objects appear to be made of a shiny material, possibly metal or plastic, and they reflect light in a way that suggests a smooth surface.']
    # caption = batch['caption']
    caption = caption
    print(caption)
    # caption = 'A globe of the Earth, centered on Africa, with a stand underneath it.'
    # caption = 'A three-dimensional model of a yellow sports car with a sleek design and a shiny finish. Positioned at a slight angle, allowing a view of its front and side.'
    # print(caption)
    # caption = 'A blue chair with a metal frame and a blue seat.'
    # caption = 'A wooden model of a plane with a propeller and wheels.'
    # caption = 'A ring with a diamond on it.'
    # caption = 'A red apple with a shiny surface has a stem on top.'
    # caption = 'A three-dimensional rendering of a wooden axe with a metallic head.\nA brown wooden handle and a silver-colored head with a sharp edge.'
    # caption = 'A bun is brown and appears to be toasted.\nA patty is brown and has grill marks on it.\nSome cheese is yellow and appears to be melted.\nA patty is brown and has grill marks on it.\nA bun is brown and appears to be toasted.\nSome cheese is yellow and appears to be melted.'
    # caption = 'Blue and gray car with a red light on top. Car has a logo on the side that reads \"\u041c\u041e\u0421\u041a\u0412\u0410\" and \"\u0412\u0410\u041b\u0410\u041e\".'
    # caption = 'A red fire hydrant with a shiny surface.'
    # caption = 'a fox head with a red and white color scheme. A black muzzle and green eyes. A black muzzle. A purple tongue, open mouth. Pointed upwards, ears.'
    # # caption = 'A pixelated hammer with a gray head and a brown handle. The handle has a black band around it.'
    # caption = 'A blue car with a black roof. A license plate that reads \"K0900K\".'
    # # caption = 'A red, three-dimensional object with a shiny, metallic sphere on top.'
    # caption = 'A blue and black car with a sleek and futuristic design.'
    # caption 
    text_connect = '_'.join(caption.split(' '))
    text_con = text_connect.replace('.', '_').replace(',', '_')
    if len(text_con) > 50:
        text_con = text_con[:50]
    caption_lat = encode_text(caption)
    _batch['caption'] = caption_lat
    render_gt = {}   # for supervision
    obj_path = batch['file'][0].split('/')[-2:]
    _batch['id'] = obj_path
    # input images
    images = batch['input_images']
    # print("input_images",images.shape)
    # images = images.to(torch.bfloat16)
    # print(images.shape)
    # exit()
    # global_step = trainer.global_step
    # device = torch.cuda.current_device()
    # log_dir = '/mnt/hdd1/caixiao/deeplearning/ckpt/3dgen/instant_nerf/test'
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # to_img = ToPILImage()
    
    B, N, C, H, W = images.shape
    # B, N, C, H, W = images.shape
    # text = batch['caption']
    text = [caption]
    if not do_classifier_free_guidance:
        text_latent = encode_text(text)
    else:
        text = text + [''] * B
        text_latent = encode_text(text)
    _batch['text_latent'] = text_latent
    # N_in = _batch['images'].shape[1]
    # rgb = (images[0][0] * 255. ).clamp(0, 255).to(torch.uint8).cpu()
    # rgb = to_img(rgb)
    # rgb.save(os.path.join(log_dir, f'{global_step}_{device}_in.png'))

    images = images.reshape(-1, C, H, W)
    images = F.interpolate(images, size=(input_size, input_size), mode='bilinear', align_corners=False).clamp(0, 1)

    # images = v2.functional.resize(
    #     images, input_size, interpolation=3, antialias=True).clamp(0, 1)
    # print(images.shape)
    _batch['images'] = images.reshape(B,N,C,input_size,input_size).to(device).to(torch.bfloat16)
    # _batch['images'] = images.reshape(B,N,C,input_size,input_size).to(device)
    # rgb = ((_batch['images'][0][0]) * 255. ).clamp(0, 255).to(torch.uint8).cpu()
    # rgb = to_img(rgb)
    # rgb.save(os.path.join(log_dir, f'{global_step}_{device}_gt.png'))
    # exit()
    # rgb = ((render_out['render_images'][0][0]) * 255. ).clamp(0, 255).to(torch.uint8).cpu()
    # rgb = to_img(rgb)
    # rgb.save(os.path.join(log_dir, f'{global_step}_{device}_out.png'))
    # exit()
    # input cameras and render cameras
    
    input_c2ws = batch['input_c2ws'].flatten(-2)
    input_Ks = batch['input_Ks'].flatten(-2)
    target_c2ws = batch['target_c2ws'].flatten(-2)
    # print(target_c2ws.shape)
    target_Ks = batch['target_Ks'].flatten(-2)
    
    ##custom camera
    target_c2ws = []
    target_Ks = []
    print(batch['input_c2ws'].shape)
    # for elev in range(-60, 60, 10):
    #     # 仰角采样
    #     RT, K = get_trans_matrix(150, elev)
    #     target_c2ws.append(RT)
    #     target_Ks.append(K)
    for angle in range(0, 360, 30):
        # xuanzhuan角采样
        RT, K = get_trans_matrix(-angle,  30)
        target_c2ws.append(RT)
        target_Ks.append(K)
    # for i in range(1):
    #     angle = random.randint(0, 360)
    #     elev = random.randint(-90, 90)
    #     # xuanzhuan角采样
    #     RT, K = get_trans_matrix(angle, elev)
    #     target_c2ws.append(RT)
    #     target_Ks.append(K)
    idx = counter
    # target_c2ws = torch.stack(target_c2ws)[[1,4,7,10]].unsqueeze(0).flatten(-2)
    # target_Ks = torch.stack(target_Ks)[[1,4,7,10]].unsqueeze(0).flatten(-2)
    target_c2ws = torch.stack(target_c2ws)[8:12].unsqueeze(0).flatten(-2)
    target_Ks = torch.stack(target_Ks)[8:12].unsqueeze(0).flatten(-2)
    # target_c2ws = torch.stack(target_c2ws)[idx:idx+1].unsqueeze(0).flatten(-2)
    # target_Ks = torch.stack(target_Ks)[idx:idx+1].unsqueeze(0).flatten(-2)
    print(target_c2ws.shape)
    
    
    # render_cameras_input = torch.cat([input_c2ws, input_Ks], dim=-1)
    render_cameras_target = torch.cat([target_c2ws, target_Ks], dim=-1)
    # render_cameras = torch.cat([render_cameras_input, render_cameras_target], dim=1)

    input_extrinsics = input_c2ws[:, :, :12]
    input_intrinsics = torch.stack([
        input_Ks[:, :, 0], input_Ks[:, :, 4], 
        input_Ks[:, :, 2], input_Ks[:, :, 5],
    ], dim=-1)
    cameras = torch.cat([input_extrinsics, input_intrinsics], dim=-1)

    # add noise to input cameras
    # cameras = cameras + torch.rand_like(cameras) * 0.04 - 0.02
    cameras = cameras.to(torch.bfloat16) 
    # cameras = cameras

    _batch['cameras'] = cameras.to(device)
    _batch['render_cameras'] = render_cameras_target.to(device).to(torch.bfloat16)
    # _batch['render_cameras'] = render_cameras_target.to(device)

    # # target images
    # target_images = torch.cat([batch['input_images'], batch['target_images']], dim=1)
    # target_depths = torch.cat([batch['input_depths'], batch['target_depths']], dim=1)
    # target_alphas = torch.cat([batch['input_alphas'], batch['target_alphas']], dim=1)
    # target images
    target_images = torch.cat([batch['target_images']], dim=1)
    target_depths = torch.cat([batch['target_depths']], dim=1)
    target_alphas = torch.cat([batch['target_alphas']], dim=1)

    # random crop
    # render_size = np.random.randint(render_size, 513)
    # render_size = 512
    target_images = target_images.reshape(-1, C, H, W)
    target_images = F.interpolate(target_images, size=(render_size, render_size), mode='bilinear', align_corners=False).clamp(0, 1)
    target_images = target_images.reshape(B, -1, C, render_size, render_size)
    # target_images = v2.functional.resize(
    #     target_images, render_size, interpolation=3, antialias=True).clamp(0, 1)
    # print("target_depths",target_depths.shape)
    target_depths = target_depths.reshape(-1, 1,H, W)
    target_depths = F.interpolate(target_depths, size=(render_size, render_size))
    target_depths = target_depths.reshape(B, -1, 1, render_size, render_size)

    target_alphas = target_alphas.reshape(-1, 1, H, W)
    target_alphas = F.interpolate(target_alphas, size=(render_size, render_size))
    target_alphas = target_alphas.reshape(B, -1, 1, render_size, render_size)
    # target_depths = v2.functional.resize(
    #     target_depths, render_size, interpolation=0, antialias=True)
    # target_alphas = v2.functional.resize(
    #     target_alphas, render_size, interpolation=0, antialias=True)

    # crop_params = v2.RandomCrop.get_params(
    #     target_images, output_size=(render_size, render_size))
    # target_images = v2.functional.crop(target_images, *crop_params)
    # # print("target_images_crop",target_images.shape)
    # target_depths = v2.functional.crop(target_depths, *crop_params)[:, :, 0:1]
    # target_alphas = v2.functional.crop(target_alphas, *crop_params)[:, :, 0:1]
    # print("target_depths",target_depths.shape)
    # print("target_alphas",target_alphas.shape)

    _batch['render_size'] = render_size
    # _batch['crop_params'] = crop_params

    # render_gt['target_images'] = target_images.to(device).to(torch.bfloat16)
    # render_gt['target_depths'] = target_depths.to(device).to(torch.bfloat16)
    # render_gt['target_alphas'] = target_alphas.to(device).to(torch.bfloat16)
    render_gt['target_images'] = target_images.to(device)
    render_gt['target_depths'] = target_depths.to(device)
    render_gt['target_alphas'] = target_alphas.to(device)

    return _batch, render_gt, text_con


def get_texture_prediction( planes, tex_pos, hard_mask=None):
    '''
    Predict Texture given triplanes
    :param planes: the triplane feature map
    :param tex_pos: Position we want to query the texture field
    :param hard_mask: 2D silhoueete of the rendered image
    '''
    tex_pos = torch.cat(tex_pos, dim=0)
    if not hard_mask is None:
        tex_pos = tex_pos * hard_mask.float()
    batch_size = tex_pos.shape[0]
    tex_pos = tex_pos.reshape(batch_size, -1, 3)
    ###################
    # We use mask to get the texture location (to save the memory)
    if hard_mask is not None:
        n_point_list = torch.sum(hard_mask.long().reshape(hard_mask.shape[0], -1), dim=-1)
        sample_tex_pose_list = []
        max_point = n_point_list.max()
        expanded_hard_mask = hard_mask.reshape(batch_size, -1, 1).expand(-1, -1, 3) > 0.5
        for i in range(tex_pos.shape[0]):
            tex_pos_one_shape = tex_pos[i][expanded_hard_mask[i]].reshape(1, -1, 3)
            if tex_pos_one_shape.shape[1] < max_point:
                tex_pos_one_shape = torch.cat(
                    [tex_pos_one_shape, torch.zeros(
                        1, max_point - tex_pos_one_shape.shape[1], 3,
                        device=tex_pos_one_shape.device, dtype=torch.float32)], dim=1)
            sample_tex_pose_list.append(tex_pos_one_shape)
        tex_pos = torch.cat(sample_tex_pose_list, dim=0)

    tex_feat = torch.utils.checkpoint.checkpoint(
        renderer.get_texture_prediction,
        planes, 
        tex_pos,
        use_reentrant=False,
    )

    if hard_mask is not None:
        final_tex_feat = torch.zeros(
            planes.shape[0], hard_mask.shape[1] * hard_mask.shape[2], tex_feat.shape[-1], device=tex_feat.device)
        expanded_hard_mask = hard_mask.reshape(hard_mask.shape[0], -1, 1).expand(-1, -1, final_tex_feat.shape[-1]) > 0.5
        for i in range(planes.shape[0]):
            final_tex_feat[i][expanded_hard_mask[i]] = tex_feat[i][:n_point_list[i]].reshape(-1)
        tex_feat = final_tex_feat

    return tex_feat.reshape(planes.shape[0], hard_mask.shape[1], hard_mask.shape[2], tex_feat.shape[-1])


def extract_mesh(
    # self, 
    planes: torch.Tensor, 
    mesh_resolution: int = 256, 
    mesh_threshold: int = 10.0, 
    use_texture_map: bool = False, 
    texture_resolution: int = 1024,
    **kwargs,
):
    '''
    Extract a 3D mesh from triplane nerf. Only support batch_size 1.
    :param planes: triplane features
    :param mesh_resolution: marching cubes resolution
    :param mesh_threshold: iso-surface threshold
    :param use_texture_map: use texture map or vertex color
    :param texture_resolution: the resolution of texture map
    '''
    assert planes.shape[0] == 1
    device = planes.device

    grid_out = renderer.forward_grid(
        planes=planes,
        grid_size=mesh_resolution,
    )
    
    vertices, faces = mcubes.marching_cubes(
        grid_out['sigma'].to(torch.float32).squeeze(0).squeeze(-1).cpu().numpy(), 
        mesh_threshold,
    )
    vertices = vertices / (mesh_resolution - 1) * 2 - 1

    if not use_texture_map:
        # query vertex colors
        # import ipdb
        # ipdb.set_trace()
        vertices_tensor = torch.tensor(vertices, dtype=torch.float32, device=device).unsqueeze(0)
        vertices_colors = renderer.forward_points(
            planes, vertices_tensor)['rgb'].squeeze(0).cpu().numpy()
        vertices_colors = (vertices_colors * 255).astype(np.uint8)

        return vertices, faces, vertices_colors
    
    # use x-atlas to get uv mapping for the mesh
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(faces.astype(int), dtype=torch.long, device=device)

    ctx = dr.RasterizeCudaContext(device=device)
    uvs, mesh_tex_idx, gb_pos, tex_hard_mask = xatlas_uvmap(
        ctx, vertices, faces, resolution=texture_resolution)
    tex_hard_mask = tex_hard_mask.float()

    # query the texture field to get the RGB color for texture map
    tex_feat = get_texture_prediction(
        planes, [gb_pos], tex_hard_mask)
    background_feature = torch.zeros_like(tex_feat)
    img_feat = torch.lerp(background_feature, tex_feat, tex_hard_mask)
    texture_map = img_feat.permute(0, 3, 1, 2).squeeze(0)

    return vertices, faces, uvs, mesh_tex_idx, texture_map

def save_obj(pointnp_px3, facenp_fx3, colornp_px3, fpath):

    pointnp_px3 = pointnp_px3 @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    facenp_fx3 = facenp_fx3[:, [2, 1, 0]]

    mesh = trimesh.Trimesh(
        vertices=pointnp_px3, 
        faces=facenp_fx3, 
        vertex_colors=colornp_px3,
    )
    mesh.export(fpath, 'obj')
    
    #     # 加载 mesh.obj 文件
    # mesh = trimesh.load(fpath)

    # # 定义中心点和保留范围（例如，中心点为 (0, 0, 0)，范围为 10）
    # center = np.array([0, 0, 0])  # 中心点坐标
    # radius = 10  # 保留范围的半径

    # # 计算每个顶点到中心点的距离
    # distances = np.linalg.norm(mesh.vertices - center, axis=1)

    # # 找到在范围内的顶点索引
    # inside_indices = np.where(distances <= radius)[0]

    # # 根据索引保留范围内的顶点和面
    # new_mesh = mesh.submesh([mesh.faces[np.all(np.isin(mesh.faces, inside_indices), axis=1)]], append=True)
    # import ipdb
    # ipdb.set_trace()
    # # 保存清理后的网格
    # new_mesh.export(fpath, 'obj')
##### 以下可理解为main函数
## 读取测试数据
# os.environ['WORLD_SIZE']='1'
# os.environ['LOCAL_RANK']='0'
# root = '/home/caixiao/projects/3d_lib/obj/pv_views'
# root = '/mnt/hdd1/caixiao/data/pv_views_64_1'
# dataset = load_text2obj_dataset(data_dir=root)
# dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=4)
# data = iter(dataloader)



txt_file = "/mnt/hdd2/caixiao/deeplearning/bench/prompt/all.txt"
txt_file = "/mnt/hdd2/caixiao/deeplearning/test/bench/prompt/sample_prompts.txt"
# txt_file = "/mnt/hdd2/caixiao/deeplearning/test/3dgen/moe/prompt/animal.txt"
# txt_file = "/mnt/hdd2/caixiao/deeplearning/test/3dgen/iccv/fantacy/fancy.txt"
# txt_file = '/mnt/hdd2/caixiao/projects/3dbench/prompt/sample_prompts.txt'
# path_list = os.listdir(base)

with open(txt_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
prompt_id = 0
leng = len(lines)
ti2 = int(time.time())
while True:
    try:
        # info = next(data)
        
        # prompt_in = lines[prompt_id%leng]
        prompt_in = lines[prompt_id]
        # prompt_in = "A rainbow over a waterfall."
        
        # scheduler.set_timesteps(50, device=device)
        # timesteps = scheduler.timesteps
        info = next(data)
        # 处理图像和caption
        _batch, render_gt, text_con = intergrate_batch(info, prompt_in)
        B = _batch['images'].shape[0]
        device = _batch['images'].device
        # pimgs = []
        # aimgs = []
        # input_size = 192
        # for i in range(3):
        #     # idx = i.split('_')[1].split('.')[0]
            
        #     # img_path = os.path.join(img_dir, f'rgb/rgb_{i}.png')
        #     img_path = os.path.join('/mnt/hdd1/caixiao/3dgen/data/tri', f'img{i+1}.png')
        #     # print(img_path)
        #     # exit()
        #     # bad_case = is_black_and_white(img_path)
        #     # if bad_case:
        #     #     return None, None
        #     # ti1 = time.time()
        #     pimg = Image.open(img_path)
        #     pimg = pimg.resize((512, 512))
        #     pimg = np.asarray(pimg)
        #     pimg = pimg  / 255.
        #     # pimg = pimg * 1.0 / 127.5 - 1
        #     tensor_image = Trans.ToTensor()(pimg)
        #     # import ipdb
        #     # ipdb.set_trace()
            
        #     # latent_img = ae.encode(tensor_image)
        #     # print(tensor_image.shape)
        #     # exit()
        #     # ti2 = time.time()
        #     # print('img',ti2-ti1)
        #     # exit()
        #     pimgs.append(tensor_image[:3])
        #     aimgs.append(tensor_image[3:])
            
        # _batch['images'] = torch.stack(pimgs)
        # images = _batch['images'].reshape(-1, 3, 512, 512)
        # _batch['images'] = F.interpolate(images, size=(input_size, input_size), mode='bilinear', align_corners=False).clamp(0, 1)
        # alpha = torch.stack(aimgs)
        # print(_batch['images'].shape)
        # print(encoder)
        # exit()
        with torch.no_grad():
            tri_prior = triplane_out(_batch)
            _batch['images'] = tri_prior
            image_feats = encoder(_batch['images'], _batch['cameras'])
            image_feats = image_feats.to(torch.bfloat16)
            # image_feats = image_feats
            plane_feat = rearrange(image_feats, '(b v) l d -> b (v l) d', b=B)
            # b, v, l, d = plane_feat.shape
            # H = int(np.sqrt(l))
            # plane_feat = rearrange(plane_feat, 'b v (h w) d -> b d v h w', h=H)
            print("image_feats",image_feats.shape)
            
            print("image_feats",image_feats.dtype)
            # exit()
            ## triplane process
            # plane_feat = _batch['images']
            text_emb = _batch['caption'].to(torch.bfloat16)
            # text_emb = _batch['caption']
            print("text_emb",text_emb.shape)
            # tri_plane = transformer(plane_feat)
            tri_plane = transformer(plane_feat, text_emb)
            # get mesh
            
            # tri_plane = transformer(plane_feat, text_emb).to(torch.float32)
            # tri_plane = rearrange(tri_plane, 'b d v h w -> b v d h w')
            print("tri_plane",tri_plane.shape)
            # exit()
            ## render
            for j in range(6):
                target_c2ws = []
                target_Ks = []
                # print(batch['input_c2ws'].shape)
                # for elev in range(-60, 60, 10):
                #     # 仰角采样
                #     RT, K = get_trans_matrix(150, elev)
                #     target_c2ws.append(RT)
                #     target_Ks.append(K)
                for angle in range(0, 360, 15):
                    # xuanzhuan角采样
                    RT, K = get_trans_matrix(-angle,  30)
                    target_c2ws.append(RT)
                    target_Ks.append(K)
                # for i in range(1):
                #     angle = random.randint(0, 360)
                #     elev = random.randint(-90, 90)
                #     # xuanzhuan角采样
                #     RT, K = get_trans_matrix(angle, elev)
                #     target_c2ws.append(RT)
                #     target_Ks.append(K)
                # idx = counter
                # target_c2ws = torch.stack(target_c2ws)[[1,4,7,10]].unsqueeze(0).flatten(-2)
                # target_Ks = torch.stack(target_Ks)[[1,4,7,10]].unsqueeze(0).flatten(-2)
                target_c2ws = torch.stack(target_c2ws)[j * 4:j*4+4].unsqueeze(0).flatten(-2)
                target_Ks = torch.stack(target_Ks)[j * 4:j*4+4].unsqueeze(0).flatten(-2)
                # target_c2ws = torch.stack(target_c2ws)[idx:idx+1].unsqueeze(0).flatten(-2)
                # target_Ks = torch.stack(target_Ks)[idx:idx+1].unsqueeze(0).flatten(-2)
                print(target_c2ws.shape)
                
                
                # render_cameras_input = torch.cat([input_c2ws, input_Ks], dim=-1)
                render_cameras_target = torch.cat([target_c2ws, target_Ks], dim=-1)
            
            
                device = torch.cuda.current_device()
                # target_pose = _batch['render_cameras']
                target_pose = render_cameras_target.to(device).to(torch.bfloat16)
                # print("target_pose",target_pose.shape)
                
                target_size = _batch['render_size']
                print("target_size",target_size)
                # exit()
                # crop_params = _batch['crop_params']
                render_results = renderer(
                    tri_plane,
                    target_pose,
                    target_size,
                    # crop_params
                )

                render_images = torch.clamp(render_results['images_rgb'], 0.0, 1.0)
                render_depths = render_results['images_depth']
                render_alphas = torch.clamp(render_results['images_weight'], 0.0, 1.0)

                render_out = {
                    'render_images': render_images,
                    'render_depths': render_depths,
                    'render_alphas': render_alphas,
                }
        # print(render_results)
        # print("images_rgb",render_images.shape)
        # print("images_depth",render_depths.shape)
        # exit()
        # loss, loss_dict = compute_loss(render_out, render_gt)
        # print(loss)
        # exit()
        # log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                global_step = counter
                
                log_dir = '/mnt/hdd1/caixiao/test/3dgen/instant_nerf_ori/our_en_de_re_5k_400'
                log_dir = '/mnt/hdd1/caixiao/test/3dgen/instant_nerf_ori/our_en_de_re_50k_select+case'
                # text_connect = '_'.join(text.split(' '))
                # text_con = text_connect.replace('.', '_').replace(',', '_')
                # if len(text_con) > 50:
                #     text_con = text_con[:50]
                log_dir = '/mnt/hdd2/caixiao/deeplearning/test/3dgen/tpl_130k+avs_66k/test_git_textin'
                log_dir = f'/mnt/hdd2/caixiao/deeplearning/bench/baseline/semv3d/prompt_200_4/{prompt_id}'
                log_dir = f'/mnt/hdd2/caixiao/deeplearning/bench/baseline/semv3d/prompt_sample_4/{prompt_id}'
                log_dir = f'/mnt/hdd2/caixiao/deeplearning/test/3dgen/iccv/fantacy/prompt_test2/{prompt_id}'
                log_dir = f'/mnt/hdd2/caixiao/deeplearning/test/bench/baseline/semv3d/with_mesh/{prompt_id}'
                # log_dir = f'/mnt/hdd2/caixiao/deeplearning/test/3dgen/moe/animal_test/1_1k/{prompt_id}'
                # log_dir = f'/mnt/hdd2/caixiao/deeplearning/test/3dgen/moe/animal_test/ori/{prompt_id}'
                # log_dir = f'/mnt/hdd2/caixiao/deeplearning/test/3dgen/moe/animal_test/2_1k/{prompt_id}'
                # log_dir = '/mnt/hdd2/caixiao/deeplearning/test/3dgen/tpl_130k+avs_66k/red_shoes'
                # log_dir = '/mnt/hdd2/caixiao/deeplearning/test/3dgen/tpl_130k+avs_66k/yellow_car'
                # log_dir = '/mnt/hdd1/caixiao/test/3dgen/instant_nerf_ori/6v_in_alldata_ori'
                # log_dir = '/mnt/hdd1/caixiao/test/3dgen/instant_nerf_ori/6v_in_w_cls'
                N_in = render_gt['target_images'].shape[1]
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                to_img = ToPILImage()
                part = _batch['id'][0]
                id = _batch['id'][1]
                
                rgb = (_batch['images'][0][0] * 255. ).clamp(0, 255).to(torch.uint8).cpu()
                rgb = to_img(rgb)
                rgb.save(os.path.join(log_dir, f'{global_step}_{device}_{1}_in.png'))
                
                rgb = (_batch['images'][0][1] * 255. ).clamp(0, 255).to(torch.uint8).cpu()
                rgb = to_img(rgb)
                rgb.save(os.path.join(log_dir, f'{global_step}_{device}_{2}_in.png'))
                
                rgb = (_batch['images'][0][2] * 255. ).clamp(0, 255).to(torch.uint8).cpu()
                rgb = to_img(rgb)
                rgb.save(os.path.join(log_dir, f'{global_step}_{device}_{3}_in.png'))
                    
                # torch.save(tri_plane, os.path.join(log_dir,'triplane.pt'))
                for i in range(4):
                    # B, N, C, H, W = _batch['images'].shape
                    
                    # rgb = ((render_gt['target_images'][0][i]) * 255. ).clamp(0, 255).to(torch.uint8).cpu()
                    # rgb = to_img(rgb)
                    # rgb.save(os.path.join(log_dir, f'{global_step}_{i}_{part}_{id}_gt.png'))

                    rgb = ((render_out['render_images'][0][i]) * 255. ).clamp(0, 255).to(torch.uint8).cpu()
                    rgb = to_img(rgb)
                    rgb.save(os.path.join(log_dir, f'{i+4*j}.png'))
                
            # rgb = ((alpha[i][0]) ).clamp(0, 1).to(torch.uint8).cpu()
            # rgb = to_img(rgb)
            # rgb.save(os.path.join(log_dir, f'{global_step}_{i}_alpha.png'))
            
            mesh_path_idx = os.path.join(log_dir, f'{prompt_id}.obj')

            # # import ipdb
            # # ipdb.set_trace()
            mesh_out = extract_mesh(
                tri_plane.to(torch.float32),
                # use_texture_map=args.export_texmap,
                # **infer_config,
            )
            # if args.export_texmap:
            #     vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            #     save_obj_with_mtl(
            #         vertices.data.cpu().numpy(),
            #         uvs.data.cpu().numpy(),
            #         faces.data.cpu().numpy(),
            #         mesh_tex_idx.data.cpu().numpy(),
            #         tex_map.permute(1, 2, 0).data.cpu().numpy(),
            #         mesh_path_idx,
            #     )
            # else:
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, mesh_path_idx)
            
            # print(f"Mesh saved to {mesh_path_idx}")
            # rgb = ((render_gt['target_alphas'][0][i])* 255. ).to(torch.uint8).cpu()
            # rgb = to_img(rgb)
            # rgb.save(os.path.join(log_dir, f'{global_step}_{i}_gt_alpha.png'))
            
            # rgb = ((render_gt['target_depths'][0][i])* 255. ).to(torch.uint8).cpu()
            # rgb = to_img(rgb)
            # rgb.save(os.path.join(log_dir, f'{global_step}_{i}_gt_depth.png'))

            # rgb = ((lat_pixels[0].reshape(img_nums,dh *8,dh *8,-1)[0].permute(2,0,1)+1) * 255. ).clamp(0, 255).to(torch.uint8).cpu()
            # rgb = to_img(rgb)
            # rgb.save(os.path.join(log_dir, f'{global_step}_{device}_lat_gt.png'))
            # input_images = v2.functional.resize(
            #     _batch['images'], (H, W), interpolation=3, antialias=True).clamp(0, 1)
            # input_images = torch.cat(
            #     [input_images, torch.ones(B, N-N_in, C, H, W).to(input_images)], dim=1)
            # print(input_images.shape)
            # input_images = rearrange(
            #     input_images, 'b n c h w -> b c h (n w)')
            # target_images = rearrange(
            #     render_gt['target_images'], 'b n c h w -> b c h (n w)')
            # print(target_images.shape)
            # render_images = rearrange(
            #     render_out['render_images'], 'b n c h w -> b c h (n w)')
            # target_alphas = rearrange(
            #     repeat(render_gt['target_alphas'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            # render_alphas = rearrange(
            #     repeat(render_out['render_alphas'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            # target_depths = rearrange(
            #     repeat(render_gt['target_depths'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            # render_depths = rearrange(
            #     repeat(render_out['render_depths'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            # MAX_DEPTH = torch.max(target_depths)
            # target_depths = target_depths / MAX_DEPTH * target_alphas
            # render_depths = render_depths / MAX_DEPTH

            # grid = torch.cat([
            #     input_images, 
            #     target_images, render_images, 
            #     target_alphas, render_alphas, 
            #     target_depths, render_depths,
            # ], dim=-2)
            # grid = make_grid(grid, nrow=target_images.shape[0], normalize=True, value_range=(0, 1))

            # save_image(grid, os.path.join(logdir, f'train_{global_step:07d}.png'))
            prompt_id = prompt_id +1
        print(counter)
        # exit()
        counter += 1  # 更新计数器
        if counter >= max_samples:  # 如果已经返回了足够的样本
            break  # 提前终止迭代
    except StopIteration:
        break  # 如果已经到达了数据集的末尾，终止迭代



# depth_unet = out_unet[0, 0, 3, :, :]
# img_unet = out_unet[0, 0, :3, :, :]

# to_img = ToPILImage()
# img_unet = to_img(img_unet)
# depth_unet = to_img(depth_unet)
# img_unet.save(f"/home/caixiao/projects/3DGen/test_unet/unet_out.png")
# depth_unet.save(f"/home/caixiao/projects/3DGen/test_unet/dep_unet_out.png")
