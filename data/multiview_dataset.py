import random
# import signal
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
from utils import render_obj,render_gso
import numpy as np
import multiprocessing
from multiprocessing import TimeoutError as MPTimeoutError
from multiprocessing.pool import ThreadPool





def read_obj(obj_path):
    choice =np.random.randint(0,100)
    if choice>10:
        azimuth = np.random.uniform(0, 360)
        elevation = np.random.uniform(0, 360)
        roll = np.random.uniform(0, 360)
    else:
        azimuth = 0
        elevation = 0
        roll = 0
    if(obj_path.endswith('.glb')):
        imgs = render_obj.render_single_view(obj_path, azimuth, elevation, roll)
    else:
        imgs = render_gso.render_single_view(obj_path, azimuth, elevation, roll)
    return torch.stack(imgs),imgs[0]


class MultiviewDataset(IterableDataset):
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
        
        try:
            world_size = os.environ['WORLD_SIZE']
            local_rank = os.environ['LOCAL_RANK']
        except:
            os.environ['WORLD_SIZE'] = "1"
            os.environ['LOCAL_RANK'] = "0"
        self.size = int(len(img_list) / float(os.environ['WORLD_SIZE']))
        # self.size = int(len(img_list)) 
        self.shuffle_indices = list(range(len(img_list)))[int(os.environ['LOCAL_RANK'])*self.size:(int(os.environ['LOCAL_RANK'])+1)*self.size]
        random.shuffle(self.shuffle_indices)


        # self.size = len(video_list)
        # self.shuffle_indices = [i for i in list(range(self.size))]
        # random.shuffle(self.shuffle_indices, lambda : float(os.environ['NODE_RANK']) / float(os.environ['WORLD_SIZE']))


    def __len__(self):
        return self.size


    

    def _create_pool(self, worker_info):
        try:
            if worker_info is None:
                p = multiprocessing.Pool(processes=1, maxtasksperchild=256)
                self._pool_is_thread = False
            else:
                p = ThreadPool(processes=1)
                self._pool_is_thread = True
            return p
        except Exception:
            # Fallback
            self._pool_is_thread = True
            return ThreadPool(processes=1)

    def _ensure_pool(self, worker_info):
        if not hasattr(self, "_pool"):
            self._pool = self._create_pool(worker_info)
            return
        # multiprocessing.Pool: _state == 0 means RUN
        state = getattr(self._pool, "_state", 0)
        if state != 0:
            try:
                if self._pool_is_thread:
                    self._pool.close()
                    self._pool.join()
                else:
                    self._pool.terminate()
                    self._pool.join()
            except Exception:
                pass
            self._pool = self._create_pool(worker_info)

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

        # Ensure pool at start
        self._ensure_pool(worker_info)

        idx = worker_id
        while True:
            nidx = self.shuffle_indices[idx]
            try:
                img_info = {'dataset': 'text2obj'}
                obj_path = self.img_list[nidx]['filename']
                img_info['caption'] = self.img_list[nidx]['caption']
                img_info['file'] = obj_path
                info_id = os.path.splitext(os.path.basename(obj_path))[0]
                img_info['id'] = info_id

                # Re-validate pool each loop (guards after prior failures)
                self._ensure_pool(worker_info)

                try:
                    async_result = self._pool.apply_async(read_obj, (obj_path,))
                except (ValueError, OSError) as e:
                    # Pool unusable, recreate and retry next item
                    print(f"Pool submit failed ({e}); recreating pool.")
                    try:
                        if self._pool_is_thread:
                            self._pool.close()
                            self._pool.join()
                        else:
                            self._pool.terminate()
                            self._pool.join()
                    except Exception:
                        pass
                    del self._pool
                    self._ensure_pool(worker_info)
                    idx = (idx + num_workers) % self.size
                    continue

                try:
                    img_info['rgb'], img_info['img'] = async_result.get(timeout=20)
                except MPTimeoutError:
                    print(f"Timeout loading {obj_path}; recreating pool.")
                    try:
                        if self._pool_is_thread:
                            self._pool.close()
                            self._pool.join()
                        else:
                            self._pool.terminate()
                            self._pool.join()
                    except Exception:
                        pass
                    del self._pool
                    self._ensure_pool(worker_info)
                    idx = (idx + num_workers) % self.size
                    continue
                except (ValueError, OSError) as e:
                    # Pool became invalid between submit and get
                    print(f"Pool get failed ({e}); recreating pool.")
                    try:
                        if self._pool_is_thread:
                            self._pool.close()
                            self._pool.join()
                        else:
                            self._pool.terminate()
                            self._pool.join()
                    except Exception:
                        pass
                    del self._pool
                    self._ensure_pool(worker_info)
                    idx = (idx + num_workers) % self.size
                    continue

                if img_info['rgb'] is None:
                    raise ValueError(obj_path, 'has a corrupted obj')
                if img_info['caption'] is None:
                    raise ValueError(obj_path, 'is a corrupted obj with no caption')

                idx = (idx + num_workers) % self.size
                print(f'{obj_path} has been loaded')
                yield img_info

            except Exception as e:
                print(e)
                idx = (idx + num_workers) % self.size

def load_multiview_dataset(
    *,
    data_dir,
    center_crop=True,
    random_flip=False,
):
    img_infos = []
    with open(data_dir,'r') as f:
        data = json.load(f)
        print("data_dir: ",data_dir)
        for line in data:
            #print(line)
    
            img_info = dict()
            line = line.replace('/mnt/hdd1/caixiao/data/pv_views/','/mnt/hdd1/caixiao/data/objaverse_1.0/hf-objaverse-v1/glbs/')
            #line += '.glb'
            img_info['filename'] = line
            img_info['caption'] = ' '
            img_infos.append(img_info)

    # img_infos = img_infos[:6500]
    # img_infos = sorted(img_infos, key=lambda x: x['filename'])
    print(f"load {len(img_infos)} objs in Text2ObjDataset")
    # exit()

    return MultiviewDataset(
        data_dir=data_dir,
        # view_num=view_num,
        img_list = img_infos,
        random_flip = random_flip,
        center_crop = center_crop,
    )


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

