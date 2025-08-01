import os
import csv
from typing import Iterable
from data.text2obj_dataset import Text2ObjDataset
from torch.utils.data import Dataset, IterableDataset, ChainDataset
from torch.utils.data import DataLoader
from modules.StableDiffusion import UNetModel, FrozenOpenCLIPEmbedder, AutoencoderKL, NormResBlock
from modules.StableDiffusion.consistencydecoder import ConsistencyDecoder
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
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from modules.networks import EMAModel, EMAModel_for_deepspeed_stage3
from torchvision.transforms import ToPILImage
import numpy as np
from PIL import Image
from torchvision import transforms as Trans

def load_text2obj_dataset(
        *,
        data_dir,
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
    addition = '/mnt/hdd1/caixiao/data/pv_views_64'
    path_lists = os.listdir(os.path.join(data_dir))
    # path_lists.sort()
    for path_list in path_lists:
        path = os.path.join(data_dir, path_list)
        img_info = dict()
        img_info['filename'] = path
        # print(path)
        img_infos.append(img_info)
    path_lists2 = os.listdir(os.path.join(addition))
    # path_lists.sort()
    for path_list in path_lists2:
        path = os.path.join(addition, path_list)
        img_info = dict()
        img_info['filename'] = path
        # print(path)
        img_infos.append(img_info)
    # with open('/mnt/nfs/caixiao/datasets/objaverse/hf-objaverse-v1/downloaded.txt','r') as f:
    #     for line in f:
    #         # print(line.split('\n'))
    
    #         img_info = dict()
    #         img_info['filename'] = line.split('\n')[0]
    #         # print(img_info['filename'])
    #         # exit()
    #         img_infos.append(img_info)
    # img_infos = img_infos[:100]
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



# ## vae
# ae = AutoencoderKL(
#     {
#         'double_z': True,
#         'z_channels': 4,
#         'resolution': 256,
#         'in_channels': 3,
#         'out_ch': 3,
#         'ch': 128,
#         'ch_mult': (1, 2, 4, 4),
#         'num_res_blocks': 2,
#         'attn_resolutions': [],
#         'dropout': 0.0,
#     },
#     {'target': 'torch.nn.Identity'},
#     4,
# )

# ckpt_path = '/home/caixiao/projects/3DGen/models/diffusion_pytorch_model_1_3.bin'
# ae.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

# ae.half()
# ae.cuda()
# ae.eval()
# ae.requires_grad_(False)


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
ae.eval()
ae.cuda()
ae.requires_grad_(False)
ckpt_path = '/mnt/hdd1/caixiao/3dgen/models/StableDiffusion/v2-1_512-ema-pruned-autoencoder.ckpt'
ae.load_state_dict(torch.load(ckpt_path, map_location='cpu'))


consis_decoder = ConsistencyDecoder()
consis_decoder.cuda()
consis_decoder.requires_grad_(False)
# ckpt_path = '/mnt/hdd1/caixiao/3dgen/models/StableDiffusion/v2-1_512-ema-pruned-autoencoder.ckpt'
# consis_decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

def encode_images(images, unflatten=True):
    B = images.shape[0]
    image = rearrange(images, 'B T C H W  -> (B T) C H W')
    z = ae.encode(image.float()).sample() # (B*T, C, H, W)???
    if unflatten:
        z = rearrange(z, '(B T) C H W -> B T H W C', B=B)
    return z * scale_factor


def decode_images(z, H, W, unflatten=False):
    # print(H,W)
    # if unflatten:
    B = z.shape[0]
    z = rearrange(z, 'T B H W C->(T B) C H W ',H=H)
    # z = z.reshape(B, H, W,-1).permute(0,3,1,2)
    z = z * (1. / scale_factor)
    # images = ae.decode(z) # (B*T, C, H, W)???
    images = consis_decoder(z) # (B*T, C, H, W)???
    if not unflatten:
        images = rearrange(images, '(T B) C H W -> T B C H W', B=B)
    return images
# os.environ['WORLD_SIZE']='8'
# os.environ['LOCAL_RANK']='0'
# root = '/mnt/hdd1/caixiao/data/pv_views_64_1'
# dataset = load_text2obj_dataset(data_dir=root)
# dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=4)
# data = iter(dataloader)


# # info =next(data)
# # print(info)
#
counter = 0
max_samples = 8
while True:
    try:
        # info = next(data)
        # print(info['depth'].shape)
        # print(info['rgb'].shape)
        # path = '/home/caixiao/projects/nerf-pytorch/dsnerf/logs_latent_net/obj_cubes_rgb_test/000001.png'
        path = '/mnt/hdd1/caixiao/3dgen/nerf/logs/logs_mipfree/latent_64_n_plane/visual/1_1_lat_gt.png'
        
        # print(img_path)
        img = Image.open(path)
        # img = img.resize((512, 512))
        lat_img = img.resize((512, 512))
        lat_img = np.asarray(lat_img)
        lat_img = lat_img  / 255.
        tensor_image1 = Trans.ToTensor()(lat_img)
        tensor_image1 = torch.stack((tensor_image1,tensor_image1),0)
        print()
        
        path = '/mnt/hdd1/caixiao/3dgen/nerf/logs/logs_mipfree/latent_64_n_plane/visual/1_0_lat_gt.png'
        
        # print(img_path)
        img = Image.open(path)
        # img = img.resize((512, 512))
        lat_img = img.resize((512, 512))
        lat_img = np.asarray(lat_img)
        lat_img = lat_img  / 255.
        tensor_image2 = Trans.ToTensor()(lat_img)
        tensor_image2 = torch.stack((tensor_image2,tensor_image2),0)
        print()
        # tensor_image = tensor_image / 255.
        # 处理图像和标签
        # images = info['rgbd'].to(torch.float32).cuda()
        # images = tensor_image.cuda()
        # print(info['file'])
        # print(images.shape)
        
        tensor_image = torch.stack((tensor_image1,tensor_image2),1)
        images = tensor_image.to(torch.float16).cuda()
        print(images.shape)
        print(images.dtype)
        z = encode_images(images)
        # image = rearrange(images, 'C T H W -> (T) C H W')
        # z = ae.encode(image).sample()  # (N*T, C, H, W)
        # z=z[:,:,16:48,16:48]
        print(z.shape)
        # exit()
        # z = rearrange(z, '(T) C H W -> C T H W')
        out = decode_images(z,64,64)
        # out = out.cpu().numpy()  # (N*T, C, H, W)
        print(out.shape)
        # exit()
        # # print(out)
        # # out_unet = rearrange(out_unet, '(B T) C H W ->B T C H W', B=batch_size)
        # min_val = torch.min(out)
        # max_val = torch.max(out)

        # # 将像素值归一化到0-1的范围
        # normalized_image = (out - min_val) / (max_val - min_val)

        # 将像素值转换到0-255的范围
        # out = (normalized_image * 255).to(torch.uint8).cpu()
        out = ((out ) * 255).clamp(0, 255).to(torch.uint8).cpu()
        # print(out)
        
        # depth = out[0, 3, :, :]
        img = out[1, 0, :3, :, :]
        to_img = ToPILImage()
        img = to_img(img)
        # dep = to_img(depth)
        img.save(f"/home/caixiao/projects/3DGen/render_rgb.jpg")
        # dep.save(f"/home/caixiao/projects/3DGen/test/dep_{counter}.png")
        # print(img.shape)
        print(counter)
        exit()
        counter += 1  # 更新计数器
        if counter >= max_samples:  # 如果已经返回了足够的样本
            break  # 提前终止迭代
    except StopIteration:
        break  # 如果已经到达了数据集的末尾，终止迭代