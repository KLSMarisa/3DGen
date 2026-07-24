
import os
import csv
from typing import Iterable

# from data.obj2render_dataset import Obj2Render_Dataset
from data.text2obj_dataset import Text2ObjDataset
from data.elevation_dataset import ElevationIterableDataset, ElevationInferenceDataset
from data.rotate_dataset import RotateDataset
from data.render_dataset import RenderDataset
from data.pair_dataset import PairDataset
from torch.utils.data import Dataset, IterableDataset, ChainDataset
from torch.utils.data import DataLoader
# import pyvista as pv
# import pandas as pd
from tqdm import tqdm
import json
import time

# pv.start_xvfb()
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
    
    #with open('/mnt/hdd1/caixiao/data/gt23d-bench/merge_caption3.json', 'r') as f:
    #    data2 = json.load(f)
    data2 = {}
        
        
    
    with open(data_dir,'r') as f:
        data = json.load(f)
        print("data_dir: ",data_dir)
        for line in data:
            #print(line)
    
            img_info = dict()
            if 'gso' in data_dir or 'multi_view' or 'multiview_300k' in data_dir:
                img_info['filename'] = line
                img_info['caption'] = ' '
            else:
                # img_info['filename'] = line.split('\n')[0]
                id = line.split('/')[-1]
                if not id in data3:
                    continue
                if not str(id) in data2:
                    continue
                caption = data2[id]
                if not ('select' in data_dir):
                    img_info['filename'] = line
                else:
                    img_info['filename'] = os.path.join("/mnt/hdd1/data", data3[id].split('pv_views_v2/')[1])
                img_info['caption'] = caption
                
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



def load_text2render_dataset(
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


    # path_lists = os.listdir(os.path.join(data_dir))


    # path_lists.sort()
    # Filter = []
    # with open('/home/caixiao/projects/3d_lib/caption/filter.txt','r') as f:
    #     for line in f:
    #         Filter.append(line.split('\n')[0])
    
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
    with open('/mnt/hdd1/caixiao/data/gt23d-bench/merge_caption3.json', 'r') as f:
        data2 = json.load(f)
    
    with open(data_dir,'r') as f:
        data = json.load(f)
        keys = data.keys()
        key_list = list(keys)
        for line in key_list:
            # print(line.split('\n'))
    
            img_info = dict()
            img_info['filename'] = os.path.join("/mnt/hdd1/caixiao/data/pv_views_v2",line)
            # print(img_info['filename'])
            # exit()
            # white test
            # file_path = img_info['filename'].replace("/3dgen/nerf/white_test2", "/data/pv_views_v2")
            # if not file_path in data2:
            #     continue
            # caption = data2[file_path]
            # if len(caption)>96:
            #     continue
            # img_info['caption'] = caption
            # # print(img_info['filename'])
            # # print(caption)
            # # exit()
            # img_infos.append(img_info)
            
            ## black
            # img_info['filename'] = img_info['filename'].replace("/3dgen/nerf/white_test2", "/data/pv_views_v2")
            # ## ori
            # print(data2.type())
            if not str(img_info['filename']) in data2:
                print(img_info['filename'])
                continue
            # caption = data2[img_info['filename']]
            # if len(caption)>96:
            #     continue
            # id = img_info['filename'].split('/')[-1]
            caption = data2[img_info['filename']]
            # plane_root = '/share/home/tj24011/home/datasets/3d_data/triplanes'
            # plane_root = '/mnt/hdd1/caixiao/test/3dgen/planes'
            # planes = os.listdir(plane_root)
            # img_info['plane'] = ''
            # info_id = img_info['filename'].split('/')[-1]
            # for plane in planes:
            #     # print(plane)
            #     if plane.split('.')[0] == info_id:
                    
            #         img_info['plane'] = os.path.join(plane_root, plane)
            #     else:
            #         continue
            if len(caption)>96:
                continue
            # if len(img_info['plane'])==0:
            #     continue
            img_info['filename'] = img_info['filename'].replace("mnt/hdd1/caixiao/data/pv_views_v2", "mnt/hdd1/data")
            img_info['caption'] = caption
            # print(img_info['filename'])
            # exit()
            img_infos.append(img_info)
    # img_infos = img_infos[:6500]
    # img_infos = sorted(img_infos, key=lambda x: x['filename'])
    print(f"load {len(img_infos)} objs in Text2RenderDataset")
    # exit()

    return Text2Render_Dataset(
        data_dir=data_dir,
        # view_num=view_num,
        img_list = img_infos,
        random_flip = random_flip,
        center_crop = center_crop,
    )



##TODO
def load_text2scene_dataset(
        *,
        data_dir,
        view_num,
        center_crop=True,
        random_flip=False,
):
    """
    """

    img_infos = []

    path_lists = os.listdir(data_dir)
    path_lists.sort()
    for path_list in path_lists:
        path = os.path.join(data_dir, path_list)
        img_info = dict()
        img_info['filename'] = path
        img_infos.append(img_info)
    # img_infos = img_infos[:50]
    img_infos = sorted(img_infos, key=lambda x: x['filename'])
    print(f"load {len(img_infos)} objs in Text2SceneDataset")
    # exit()

    return Text2SceneDataset(
        data_dir=data_dir,
        view_num=view_num,
        img_list=img_infos,
        random_flip=random_flip,
        center_crop=center_crop,
    )

def load_val_text2scene_dataset(
        *,
        data_dir,
        view_num,
        center_crop=True,
        random_flip=False,
):
    """
    """

    img_infos = []

    path_lists = os.listdir(data_dir)
    path_lists.sort()
    for path_list in path_lists:
        path = os.path.join(data_dir, path_list)
        img_info = dict()
        img_info['filename'] = path
        img_infos.append(img_info)
    # img_infos = img_infos[:50]
    img_infos = sorted(img_infos, key=lambda x: x['filename'])
    print(f"load {len(img_infos)} objs in Text2SceneDataset")
    # exit()

    return Text2SceneDataset(
        data_dir=data_dir,
        view_num=view_num,
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
# --- 新增的安全获取嵌套配置的辅助函数 ---
def safe_get(config, key_path: str, default=None):
    """
    安全地从嵌套的 config 对象（如 OmegaConf, dict, argparse）中获取值。
    例如: safe_get(config, 'dataset.rotator_val.root', None)
    """
    keys = key_path.split('.')
    curr = config
    try:
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            elif hasattr(curr, k):
                curr = getattr(curr, k)
            elif hasattr(curr, 'get'):
                curr = curr.get(k, default)
                if curr is default:
                    return default
            else:
                return default
        return curr if curr is not None else default
    except Exception:
        return default
# ----------------------------------------

# ... [保留你原有的 load_text2obj_dataset, load_text2render_dataset, load_text2scene_dataset, ConcatDataset 等函数不动] ...


def UnifiedDataset(config):
    _datasets = []

    datasets = config.datasets.split(',')
    assert len(datasets) > 0, "No dataset specified"
    for dataset in datasets:
        if dataset == 'text2obj':
            _datasets.append(PairDataset(
                json_path=config.dataset.text2obj.data_dir,
                image_size=config.image_resolution,
                # ✅ 修改点
                data_root=safe_get(config, 'dataset.text2obj.root', None)
            ))
        elif dataset == 'rotator':
            # print(safe_get(config, 'dataset.rotator_train.root')) # 如果需要打印，也可以用 safe_get
            _datasets.append(RotateDataset(
                json_path=config.dataset.rotator_train.data_dir,
                # ✅ 修改点
                data_root=safe_get(config, 'dataset.rotator_train.root', None),
                glb_db=config.dataset.rotator_train.glb_db,
                glb_root=safe_get(config, 'dataset.rotator_train.glb_root', '/mnt/hdd1/caixiao/data/objaverse_1.0/hf-objaverse-v1/glbs/'),
                image_size=config.image_resolution,
                center_crop=safe_get(config, 'center_crop', False),
                random_flip=safe_get(config, 'random_flip', False),
                is_train=True,
                val_views_per_obj=7,   # 训练无效，但传了也没事
            ))
        elif dataset == 'elevation':
            processor = config.dataset.elavation_train.processor
            if processor == 'ElevationIterableDataset':
                _datasets.append(ElevationIterableDataset(
                    json_path=config.dataset.elavation_train.data_dir,
                    image_size=config.image_resolution,
                    # ✅ 修改点
                    data_root=safe_get(config, 'dataset.elavation_train.root', None),
                    center_crop=safe_get(config, 'center_crop', False),
                    random_flip=safe_get(config, 'random_flip', True),
                    is_train=True,
                ))
            else:
                _datasets.append(PairDataset(
                    json_path=config.dataset.elavation_train.data_dir,
                    image_size=config.image_resolution,
                    # ✅ 修改点
                    data_root=safe_get(config, 'dataset.elavation_train.root', None),
                ))
        elif dataset == 'render_sparse':
            _datasets.append(RenderDataset(
                json_path=config.dataset.render_sparse.data_dir,
                data_root=safe_get(config, 'dataset.render_sparse.root', None),
                image_size=config.image_resolution,
            ))
        elif dataset == 'text2scene':
            _datasets.append(load_text2scene_dataset(
                data_dir=config.dataset.text2scene.data_dir,
                view_num=config.dataset.text2scene.view_num
            ))
        elif dataset == 'obj2render':
            _datasets.append(load_obj2render_dataset(
                data_dir=config.dataset.obj2render.data_dir,
            ))
        elif dataset == 'text2render':
            _datasets.append(load_text2render_dataset(
                data_dir=config.dataset.text2render.data_dir,
            ))
        else:
            raise NotImplementedError

    return ConcatDataset(_datasets)


def Unified_val_Dataset(config):
    _datasets = []

    datasets = config.datasets.split(',')
    assert len(datasets) > 0, "No dataset specified"
    for dataset in datasets:
        if dataset == 'text2obj':
            if 'gso' in config.dataset.text2obj_val.data_dir:
                _datasets.append(load_multiview_dataset(
                    data_dir=config.dataset.text2obj_val.data_dir,
                ))
            else:
                _datasets.append(PairDataset(
                json_path=config.dataset.text2obj.data_dir,
                image_size=config.image_resolution,
                # ✅ 修改点
                data_root=safe_get(config, 'dataset.text2obj.root', None)
                ))
        elif dataset == 'rotator':
            _datasets.append(RotateDataset(
                json_path=config.dataset.rotator_val.data_dir,
                # ✅ 修改点
                data_root=safe_get(config, 'dataset.rotator_val.root', None),
                glb_db=config.dataset.rotator_val.glb_db,
                glb_root=safe_get(config, 'dataset.rotator_val.glb_root', '/mnt/hdd1/caixiao/data/objaverse_1.0/hf-objaverse-v1/glbs/'),
                image_size=config.image_resolution,
                center_crop=safe_get(config, 'center_crop', False),
                random_flip=False,
                is_train=False,
                val_views_per_obj=10,
            ))
        elif dataset == 'elevation':
            processor = config.dataset.elavation_val.processor
            if processor == 'ElevationIterableDataset':
                _datasets.append(ElevationIterableDataset(
                    json_path=config.dataset.elavation_val.data_dir,
                    image_size=config.image_resolution,
                    # ✅ 修改点
                    data_root=safe_get(config, 'dataset.elavation_val.root', None),
                    center_crop=safe_get(config, 'center_crop', False),
                    random_flip=False,
                    is_train=False,
                    yaw_eval=0,
                ))
            else:
                _datasets.append(PairDataset(
                    json_path=config.dataset.elavation_val.data_dir,
                    image_size=config.image_resolution,
                    # ✅ 修改点
                    data_root=safe_get(config, 'dataset.elavation_val.root', None),
                ))
        elif dataset == 'text2scene':
            _datasets.append(load_text2scene_dataset(
                data_dir=config.dataset.text2scene.data_dir,
                view_num=config.dataset.text2scene.view_num
            ))
        elif dataset == 'obj2render':
            _datasets.append(load_val_obj2render_dataset(
                data_dir=config.dataset.obj2render.data_dir,
            ))
        else:
            raise NotImplementedError

    return ConcatDataset(_datasets)

def identity_collate_fn(x):
    '''Identity function to be passed as collate function to DataLoader'''
    return x


def create_dataloader(config):
    dataset = UnifiedDataset(config)
    dataloader = DataLoader(
        dataset          =   dataset,
        batch_size       =   config.batch_size,
        shuffle          =   False,
        num_workers      =   10,
        collate_fn       =   (lambda x: x) if config.no_data_stack else None,
        pin_memory       =   False,
        prefetch_factor  = 20,
        drop_last=True,
    )
    return dataloader



def create_val_dataloader(config):
    dataset = Unified_val_Dataset(config)
    dataloader = DataLoader(
        dataset          =   dataset,
        batch_size       =   1,
        shuffle          =   False,
        num_workers      =   10,
        collate_fn       =   (lambda x: x) if config.no_data_stack else None,
        pin_memory       =   False,
        drop_last=True,
    )
    return dataloader


def create_elevation_inference_dataloader(config):
    """elevation 推理专用 dataloader：从文件夹直接加载图片，无需 JSONL 列表"""
    elev_cfg = config.dataset.elevation
    image_dir = str(elev_cfg.inference_image_dir)

    dataset = ElevationInferenceDataset(
        image_dir=image_dir,
        image_size=config.image_resolution,
        center_crop=getattr(config, "center_crop", False),
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader


# os.environ['WORLD_SIZE']='8'
# os.environ['LOCAL_RANK']='2'
# # root = '/mnt/hdd1/caixiao/data/pv_views_v2/part1'
# root = '/mnt/hdd1/data/select_data1.json'
# # root = '/mnt/hdd1/caixiao/data/objaverse_1.0/utils/data_select/objaverse_qilian_WB.json'
# # dataset = load_obj2render_dataset(