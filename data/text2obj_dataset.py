import random
import numpy as np
import os
import csv
import math
import torch
import subprocess
# import pyvista as pv
import time
# import h5py

from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms as Trans
from torchvision.transforms import InterpolationMode
from torchvision import utils as vutils
# from data.process import look_at_view_transform, compute_projection_matrix, rotation_matrix
# from process import look_at_view_transform, compute_projection_matrix, rotation_matrix
# from data.process import look_at_view_transform, compute_projection_matrix
# from xvfbwrapper import Xvfb
import random
import json



def load_camera(path):
    with open(path, 'r') as f:
        camera = json.load(f)
    return camera


def get_trans_matrix(camera):
    elev = camera['elev']
    azimuth = camera['azimuth']
    R = rotation_matrix(azimuth=azimuth, elevation=elev, roll=0)
    R = torch.from_numpy(R)
    C = np.asarray(camera['camera_position'][0], dtype=np.float32)
    norm = np.linalg.norm(C)
    C = C / norm
    # print(norm)
    # exit()
    C[[0]] = -1 * C[[0]]
    # C[[1]] = -1 * C[[1]]
    C = torch.tensor(C).view((-1, 1))
    # print(C)
    T = -torch.matmul(R.to(torch.float32),C)
    # T[1]=0
    # T = T[0].view(-1, 1)
    RT = torch.cat((R, T), dim=1)
    d = 512.0
    f =  d / (2 * math.tan(math.radians(camera['fov'] / 2.0)))
    K = np.array([[f, 0, 256.0],
                [0, f, 256.0],
                [0, 0, 1]])
    K = torch.from_numpy(K)
    return RT, K, norm

def normalize_depth(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val) - 0.5
    return normalized_matrix

def read_obj(obj_path):
    info_id = os.path.splitext(os.path.basename(obj_path))[0]
    rgb_path = os.path.join(obj_path, 'rgb')
    # depth_path = os.path.join(obj_path, 'depth')
    # camera_path = f'/mnt/hdd1/caixiao/data/pv_cameras/{info_id}'
    

    # rgb and caption
    imgs = []
    deps = []
    # RTs = []
    # Ks = []
    caption = ""
    img_info = dict()
    # camera_info = []

#### caption
    # view_path = os.path.join('/mnt/hdd1/caixiao_hdd1/datasets/objaverse/views_release', info_id)
    # # view_path = os.path.join('/mnt/hdd1/caixiao_hdd1/datasets/objaverse/views_release', '000b958aaf8346baae84568d7918d060')
    # caption_path = os.path.join(view_path, 'caption.txt')
    # if os.path.exists(caption_path):
    #     with open(caption_path, 'r') as file:
    #             # 读取文件内容
    #         caption = file.read()
    # else:
    #     caption = None
    # t1 = time.time()
    # with open('/home/caixiao/projects/LLaVA/out/merged/caption.json', 'r') as f:
    #     data = json.load(f)
    # # t2 = time.time()
    # # print(t2-t1)
    # caption = data[obj_path]
    # # t3 = time.time()
    # # print(t3-t2)
    # # print(caption)
    # # print(len(caption))
    # # exit()
    # if len(caption) > 255 or len(caption) == 0:
    #     caption = None
    #     return None, None


#### rgb

    parts = obj_path.split('/')
    last_two_parts = '/'.join(parts[-2:])


    img_dir = os.path.join(obj_path,'rgb')
    # print(img_dir)
    img_lists = os.listdir(img_dir)
    # random_select = random.sample(img_lists, 3)
    necessary_lists = ['rgb_0.png', 'rgb_63.png', 'rgb_4.png' ]
    # idx1 = random.randint(0, 63)
    # idx2 = random.randint(0, 63)
    # idx3 = random.randint(0, 63)
    # print(idx)
    # for i in [idx+48, idx, idx+4]:
    t6 = time.time()
    # print(t6-t5)
    for i in necessary_lists:
    # for i in random_select:
        # print(i)
        # img_path = os.path.join(img_dir, f'rgb/rgb_{i}.png')
        img_path = os.path.join(img_dir, i)
        # print(img_path)
        # bad_case = is_black_and_white(img_path)
        # if bad_case:
        #     return None, None
        
        img = Image.open(img_path)
        img = img.resize((512, 512))
        img = np.asarray(img)[:,:,0:3]
        # print(img.shape)
        
        img = img * 1.0 / 127.5 - 1
        
        tensor_image = Trans.ToTensor()(img)
        # print(tensor_image[0,255,:])
        # exit()
        imgs.append(tensor_image[:3])

        # depth
        idx = i.split('_')[1].split('.')[0]
        # print(idx)

    # t7 = time.time()
    # print(t7-t6)
    # print(imgs)
    img_info['rgb'] = torch.stack(imgs)
    img_info['img'] = imgs[0]
    # img_info['depth'] = torch.stack(deps)
    # img_info['caption'] = caption
    # img_info['rgbd'] = torch.cat((img_info['rgb'], img_info['depth']), dim=1)
    # img_info['rgbd'] = img_info['rgbd'].permute(1, 0, 2, 3)
    #img_info['rgbd'] = img_info['rgb'].permute(1, 0, 2, 3)
    # img_info['camera'] = camera_info
    # img_info['RT'] = torch.stack(RTs)
    # img_info['K'] = torch.stack(Ks)
    # t8 = time.time()
    # print(t8-t7)
    # print(img_info['camera'])
    # print(t8-t1)
    # exit()
    # del depth
    return img_info['rgb'],img_info['img'] #img_info['caption'] #, img_info['RT'], img_info['K']


class Text2ObjDataset(IterableDataset):
    def __init__(
            self,
            data_dir,
            img_list,
            p_uncond=0.2,
            random_flip=False,
            center_crop=True,
    ):
        super().__init__()
        
        self.data_dir = data_dir
        # self.view_num = view_num
        self.img_list = img_list

        self.random_flip = random_flip
        self.center_crop = center_crop
        self.p_uncond = p_uncond

        # self.size = int(len(img_list))
        # os.environ['LOCAL_RANK']=torch.distributed.get_rank(group=None)
        # print(os.environ['LOCAL_RANK'])
        # exit()
        # NODE_RANK = torch.cuda.current_device()
        # local_rank = int(os.getenv('LOCAL_RANK'))
        # print(local_rank)
        # worker_info = torch.utils.data.get_worker_info()
        # print(worker_info)
        # time.sleep(5)
        # exit()
        # print(os.environ)
        # # time.sleep(300)
        # exit()
        # os.environ['WORLD_SIZE'] = "1"
        # os.environ['LOCAL_RANK'] = "0"
        self.size = int(len(img_list) / float(os.environ['WORLD_SIZE']))
        # self.size = int(len(img_list)) 
        self.shuffle_indices = list(range(len(img_list)))[int(os.environ['LOCAL_RANK'])*self.size:(int(os.environ['LOCAL_RANK'])+1)*self.size]
        random.shuffle(self.shuffle_indices)


        # self.size = len(video_list)
        # self.shuffle_indices = [i for i in list(range(self.size))]
        # random.shuffle(self.shuffle_indices, lambda : float(os.environ['NODE_RANK']) / float(os.environ['WORLD_SIZE']))


    def __len__(self):
        return self.size


    

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # print(worker_info.num_workers)
        # print(worker_info.id)
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        idx = worker_id
        while True:
            nidx = self.shuffle_indices[idx]
            # print(nidx)
            try:
                img_info = {'dataset': 'text2obj'}

                # print(self.img_list[2]['filename'])
                # exit()
                obj_path = self.img_list[nidx]['filename']
                img_info['caption'] = self.img_list[nidx]['caption']
                img_info['file'] = obj_path
                # if self.img_list[nidx][1] == -1:
                #     continue
                # print(obj_path)
                
                info_id = os.path.splitext(os.path.basename(obj_path))[0]
                img_info['id'] = info_id
                # print(info_id)
                def took_too_long(signum, frame):
                    raise TimeoutError('Load', obj_path, 'timeout')

                # img_info['rgbd'], img_info['caption'] = read_obj(obj_path)
                img_info['rgb'],img_info['img'] = read_obj(obj_path)
                # print(img_info['rgbd'].shape)
                # print(shape)
                # img_info['rgbd'], img_info['caption'], img_info['RT'], img_info['K']= read_obj2(obj_path)
                if img_info['rgb'] == None:
                    # os.system(f"rm -r '/mnt/nfs/caixiao/data/obj_views/{info_id}'")
                    # print(obj_path, 'has an corrupted obj')
                    raise ValueError(obj_path, 'has an corrupted obj')
               


                # check1: non-empty (non-corrupted) video
                # if img_info['RT'] == None:
                #     raise ValueError(obj_path, 'is an corrupted obj with no camera')

                # check2: non-empty (non-corrupted) video
                if img_info['caption'] == None:
                    raise ValueError(obj_path, 'is an corrupted obj with no caption')

                img_info['id'] = info_id
                # video = self.process_video(video)
                # video = video.float() / 127.5 - 1 # [-1, 1]
                # data_dict['video'] = video.permute(1, 0, 2, 3).contiguous() # c * t * h * w


                idx = (idx + num_workers) % self.size

                yield img_info

            except Exception as e:
                print(e)
                idx = (idx + num_workers) % self.size
                continue



# from torchvision.transforms import ToPILImage
# img = np.load('/home/caixiao/projects/objaverse-xl/tests/000.npy')
# img = img[:,:,2]
# print(img[220][26])

# img_p = '/home/caixiao/projects/objaverse-xl/scripts/rendering/21dd4d7b-b203-5d00-b325-0c041f43524e/000.png'
# img = Image.open(img_p)
# img = img.resize((512, 512))
# tensor_image = T.ToTensor()(img)
# to_img = ToPILImage()
# img = to_img(tensor_image[3])
#
# img.save('/home/caixiao/projects/3DGen/data/000.png')

# rgb= read_obj('/mnt/hdd1/caixiao/data/pv_views_v2/part6/fa720594a27b4752bb73a7804c383888')
# rgb= read_obj('/mnt/hdd1/caixiao/data/pv_views_v2/part6/fa519827e1f94b91b69945e75ccd57de')
# # rgb= read_obj('/mnt/hdd1/caixiao/data/pv_views_v2/part6/48bf5f8b56154a6289685911bb80fb20')
# print(rgb.shape)
# print(caption)
# print(RT)
# print(K)

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