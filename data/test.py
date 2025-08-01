# import os
# import csv
# from typing import Iterable
# # from data.text2obj_dataset import Text2ObjDataset
# # from data.text2scene_dataset import Text2SceneDataset
# from text2obj_dataset import Text2ObjDataset
# from text2scene_dataset import Text2SceneDataset
# # from data.text2obj_dataset_old import Text2ObjDataset
# from torch.utils.data import Dataset, IterableDataset, ChainDataset
# from torch.utils.data import DataLoader
# import pyvista as pv
# import multiprocessing



# def load_text2obj_dataset(
#     *,
#     data_dir,
#     view_num,
#     center_crop=True,
#     random_flip=False,
# ):

#     """
#     For a dataset, create a generator over text2obj (text-objaverse pair) data samples.
#     The format of data is a dict. The meaning of each element is as follows,
#     "video": the video, a NFCHW float tensor
#     "video_caption": tokenized text, a tensor
#     "video_text": the original text
#     "dataset": the dataset name
#     "audio": None (TODO)
#     "audio_caption": None (TODO)
#     "audio_text": None (TODO)

#     :param annotation_path: annotation file.
#     :param data_dir: video dir
#     :param video_size: [frane_num, channnel, height, width]
#     :param video_fps: loading fps of the video
#     :param frame_gap: after loading the video with video_fps, sample under this frame gap
#     :param center_crop/random_flip: data augmentation
#     :param p_uncond: the rate of "" for classifier free guidance

#     :param audio_size, audio_fps (TODO)
#     """

#     img_infos = []

#     # path_lists = os.listdir(os.path.join(data_dir, 'views_release'))
#     # path_lists.sort()
#     # for path_list in path_lists:
#     #     path = os.path.join(data_dir, 'views_release', path_list)
#     #     img_info = dict()
#     #     img_info['filename'] = path
#     #     img_infos.append(img_info)
#     with open('/mnt/nfs/caixiao/datasets/objaverse/hf-objaverse-v1/downloaded.txt','r') as f:
#         for line in f:
#             # print(line.split('\n'))
    
#             img_info = dict()
#             img_info['filename'] = line.split('\n')[0]
#             # print(img_info['filename'])
#             # exit()
#             img_infos.append(img_info)
#     # img_infos = img_infos[:1000]
#     img_infos = sorted(img_infos, key=lambda x: x['filename'])
#     print(f"load {len(img_infos)} objs in Text2ObjDataset")
#     # exit()

#     return Text2ObjDataset(
#         data_dir=data_dir,
#         view_num=view_num,
#         img_list = img_infos,
#         random_flip = random_flip,
#         center_crop = center_crop,
#     )


# def load_text2scene_dataset(
#         *,
#         data_dir,
#         view_num,
#         center_crop=True,
#         random_flip=False,
# ):
#     """
#     """

#     img_infos = []

#     path_lists = os.listdir(data_dir)
#     path_lists.sort()
#     for path_list in path_lists:
#         path = os.path.join(data_dir, path_list)
#         img_info = dict()
#         img_info['filename'] = path
#         img_infos.append(img_info)
#     # img_infos = img_infos[:40]
#     img_infos = sorted(img_infos, key=lambda x: x['filename'])
#     print(f"load {len(img_infos)} objs in Text2SceneDataset")
#     # exit()

#     return Text2SceneDataset(
#         data_dir=data_dir,
#         view_num=view_num,
#         img_list=img_infos,
#         random_flip=random_flip,
#         center_crop=center_crop,
#     )

# class ConcatDataset(IterableDataset):
#     r"""Dataset for concating multiple :class:`IterableDataset` s.

#     This class is useful to assemble different existing dataset streams. The
#     concating operation is done on-the-fly, so concatenating large-scale
#     datasets with this class will be efficient.

#     Args:
#         datasets (iterable of IterableDataset): datasets to be chained together
#     """
#     def __init__(self, datasets: Iterable[Dataset], length_cut='max') -> None:
#         super(ConcatDataset, self).__init__()
#         self.datasets = datasets
#         self.length_cut = length_cut
#         assert length_cut in ['max', 'min']
#         self.len = len(self)

#     def __iter__(self):
#         datasets_iter = [iter(ds) for ds in self.datasets]
#         for i in range(self.len):
#             idx = i % len(self.datasets)
#             try:
#                 x = next(datasets_iter[idx])
#             except:
#                 datasets_iter[idx] = iter(self.datasets[idx])
#                 x = next(datasets_iter[idx])
#             yield x

#     def __len__(self):
#         lengths = []
#         for d in self.datasets:
#             assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
#             lengths.append(len(d))
#         if self.length_cut == 'max':
#             total = max(lengths) * len(lengths)
#         elif self.length_cut == 'min':
#             total = min(lengths) * len(lengths)
#         return total


# def UnifiedDataset(config):
#     _datasets = []

#     datasets = config.datasets.split(',')
#     assert len(datasets) > 0, "No dataset specified"
#     for dataset in datasets:
#         if dataset == 'text2obj':
#             _datasets.append(load_text2obj_dataset(
#                 data_dir=config.dataset.text2obj.data_dir,
#                 view_num=config.dataset.text2obj.view_num
#             ))
#         elif dataset == 'text2scene':
#             _datasets.append(load_text2scene_dataset(
#                 data_dir=config.dataset.text2scene.data_dir,
#                 view_num=config.dataset.text2obj.view_num
#             ))
#         else:
#             '''
#             TODO
#             '''
#             raise NotImplementedError

#     # return ChainDataset(_datasets)
#     return ConcatDataset(_datasets)


# def identity_collate_fn(x):
#     '''Identity function to be passed as collate function to DataLoader'''
#     return x


# def create_dataloader(config):
#     dataset = UnifiedDataset(config)
#     dataloader = DataLoader(
#         dataset          =   dataset,
#         batch_size       =   config.batch_size,
#         shuffle          =   False,
#         num_workers      =   4,
#         collate_fn       =   None,
#         pin_memory       =   False,
#     )
#     return dataloader

# if __name__ == '__main__':
#     # multiprocessing.set_start_method('spawn', force=True)
#     pv.start_xvfb()
#     os.environ['WORLD_SIZE']='8'
#     os.environ['LOCAL_RANK']='3'
#     root = '/mnt/nfs/caixiao/datasets/blendedMVS/select'
#     dataset = load_text2obj_dataset(data_dir=root, view_num=16)
#     # dataset = load_text2scene_dataset(data_dir=root, view_num=16)
#     dataloader = DataLoader(dataset, shuffle=False, batch_size=4, num_workers=4, pin_memory=True)
#     data = iter(dataloader)
#     # info =next(data)
#     # print(info)
#     #
#     counter = 0
#     max_samples = 32
#     while True:
#         try:
#             info = next(data)
#             print(info['rgbd'].shape)
#             print(info['caption'])
#             # 处理图像和标签
#             print(counter)
#             counter += 1  # 更新计数器
#             if counter >= max_samples:  # 如果已经返回了足够的样本
#                 break  # 提前终止迭代
#         except StopIteration:
#             break  # 如果已经到达了数据集的末尾，终止迭代

import re
import random
import numpy as np
# 原始列表
# lst = ['rgb_0.png','rgb_63.png', 'rgb_4.png','rgb_15.png','rgb_28.png','rgb_39.png']

# # 提取数字并排序
# lst.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
# print(lst)
# # 按照要求选择元素
# result = []
# result.append(random.choice(lst[0:1]))  # 从前16个中选择1个
# result.extend(random.sample(lst[1:3], 2))  # 从16-48中选择2个
# result.append(random.choice(lst[3:5]))  # 从48-64中选择1个

# print(result)

depth_path = '/mnt/hdd1/caixiao/data/pv_views/000-005/66bb7ee29a644459b999e89fc719b567/depth/depth_18.npz'
depth = np.load(depth_path, mmap_mode='r')
print(depth['data.npy'])