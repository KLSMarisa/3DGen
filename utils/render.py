import json

# 定义变量
# range_data = (1.2499920813286667, 3.507105337564262)
# scale_data = [1.0, 1.0, 1.0]
# fov_data = 30.0

# # 将变量值赋给字典中的键
# data = {
#     'range': range_data,
#     'scale': scale_data,
#     'fov': fov_data
# }

# # 将数据写入JSON文件
# with open('data.json', 'w') as f:
#     json.dump(data, f)

# # 从JSON文件中读取数据
# with open('data.json', 'r') as f:
#     data_loaded = json.load(f)
#     print('fov: ', data_loaded['fov'])



#####  ------------------------- pyvista ----------------------------------
import pyvista as pv
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm
# from process import look_at_view_transform
import pandas as pd
import time
import csv
 
# filename = 'example.csv'
def update_csv(filename, id, idx, pyvista_open): 
    new_row = [id, idx, pyvista_open]
    print(new_row)
    line_to_append = ','.join(new_row)
    # file.write(line_to_append + '\n') 
    with open(filename, 'a') as file:
        file.write(line_to_append + '\n')
    # with open(filename, 'w+', newline='') as file:
    #     reader = csv.reader(file)
    #     writer = csv.writer(file)
    #     # 读取所有行，并在末尾添加新行
    #     rows = list(reader)
    #     rows.append(new_row)
    #     # 将文件指针移动到开始
    #     file.seek(0)
    #     # 写入所有行，包括新行
    #     writer.writerows(rows)
    # with open(filename, 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(['obj_path', 'pyvista_open'])
    #     csvwriter.writerow(['Data1', 'Data2', 'Data3'])

# t1 = time.time()
img_infos = []
# 读取csv文件
# df = pd.read_csv('/home/caixiao/projects/3d_lib/data/select_all.csv')
# base_path = '/mnt/hdd1/caixiao/data/pv_views_v2/part6'
# dep_path = '/mnt/hdd1/caixiao/data/pv_views_v2/part6'
# dep_path = '/mnt/nfs/caixiao/datasets/pv_views_v2/part6'
# base_path = '/mnt/nfs/caixiao/datasets/test'
# len(df)
# for i in range():#86000
# for i in range(0+1855+5509+1492+693+1915+1270+3638+1485+4570+22+48+558+1959+527+72+486+475+201+1173+559+246+552+282+441+537+905+101+294+48+213+1738+736+523+283+143+1659+146+2539+1112+1187+1768+328+1814+192+81+1015+1666+250+257+428+539+219+1211+1826+521+223+1921+1138+541+295+453+488+515+661+532+337+980+721+747+202+6467+850+403+855+1165+140+848+633+1440+673+1283+217+2922+717+390+1241+2086+286+1646+1534+30+95+150+89+834+91, 93000): 

# 选择你想要的行，例如第3行
#     row = df.iloc[i]

# # 将行数据转换为列表
#     data = row.tolist()
#     if data[1] != -1:
#         img_infos.append(data)
# print(len(img_infos))
# exit()
with open('/mnt/hdd1/caixiao/data/objaverse_1.0/path_index.json', 'r') as f:
    data = json.load(f)
    # for line in f:
        # print(line.split('\n'))
#
#         img_info = dict()
#         img_info['filename'] = line.split('\n')[0]
#         # print(img_info['filename'])
#         # exit()
#         img_infos.append(img_info)
img_infos = data[20:100000]
# img_infos = sorted(img_infos, key=lambda x: x['filename'])
print(f"load {len(img_infos)} objs in Text2ObjDataset")

from xvfbwrapper import Xvfb

vdisplay = Xvfb(width=1280, height=640)
vdisplay.start()

# 在此虚拟显示中启动你的程序



# t2 = time.time()
# pv.start_xvfb()
# 读取glb文件
# mesh = pv.read('/mnt/hdd1/caixiao/datasets/objaverse/hf-objaverse-v1/glbs/000-000/000074a334c541878360457c672b6c2e.glb')
# # 斜挎包 /mnt/hdd1/caixiao/datasets/objaverse/hf-objaverse-v1/glbs/000-000/000a3d9fa4ff4c888e71e698694eb0b0.glb
# # 楼梯 /home/caixiao/projects/3d_lib/obj/0a0f1b107acb4b94a8211e11ab69a67f.glb
# # 拖鞋 /mnt/hdd1/caixiao/datasets/objaverse/hf-objaverse-v1/glbs/000-000/000074a334c541878360457c672b6c2e.glb
# # room /mnt/hdd1/caixiao/datasets/objaverse/hf-objaverse-v1/glbs/000-001/000b958aaf8346baae84568d7918d060.glb
# # 大象 /home/caixiao/projects/3d_lib/obj/000a82b4e6bf4e909fbe5a3b0e6d67dc.glb
# # 马桶 /mnt/hdd1/caixiao/datasets/objaverse/hf-objaverse-v1/glbs/000-000/00b360de1846428eb5c23464824c5fc8.glb
# 创建一个plotter对象
# counter=0
# max_samples=16
# i=
for img_list in tqdm(img_infos):
    idx = img_list.split('/')[-2]
    base_path = f'/mnt/hdd1/caixiao/data/pv_views/{idx}'
    csv_path = f'/mnt/hdd1/caixiao/data/objaverse_1.0/utils/{idx}.txt'
    flag = 0
    # if img_list[1] == -1:
    #     continue
    obj_path = img_list
    # obj_path = '/mnt/nfs/caixiao/datasets/objaverse/hf-objaverse-v1/glbs/000-080/912b0308a23d4aeeba71c2762a32dc4c.glb'
    # print(obj_path)
    info_id = os.path.splitext(os.path.basename(obj_path))[0]
    # print(info_id)
    # exit()
    plotter = pv.Plotter(off_screen=True, window_size=(512, 512), image_scale=1)
    plotter.set_background('white')
    pyvista_open = '1'
    try:
        obj = plotter.import_gltf(obj_path)
        # plotter.set_environment_texture(obj_path)
    except Exception as e:
        # print(e)
        continue
    # print(info_id)
    update_csv(csv_path, info_id, idx, pyvista_open)
    # print(obj_path)
    # print(plotter.camera)
    # exit()
    #     print(plotter.camera.clipping_range)
    #     print(plotter.scale)
    #     print(plotter.camera.view_angle)
    # # distance = plotter.camera.distance
    
    
    # azimuth_angle = 0.0
    # elevation_angle = 0.0
    # plotter.show(auto_close=False)
    # num_images = 48
    # # t3 = time.time()
    # # camera_dist = 1
    # # 对于每个视角，生成图像和深度图
    # for i in range(num_images):
    #     # i = 7
    #     # print(i)
    #     camera_position = plotter.camera_position
    #     # print(camera_position[0:])
    #     clipping_range = plotter.camera.clipping_range
    #     distance = plotter.camera.distance
    #     data = {
    #             'camera_position': camera_position[0:],
    #             'dsitance': distance,
    #             'azimuth': azimuth_angle,
    #             'elev': elevation_angle,
    #             'clipping_range': clipping_range,
    #             'scale': scale,
    #             'fov': fov
    #         }
    #     # exit()
    #     # R, T = look_at_view_transform(elev=elevation_angle, azim=azimuth_angle, dist=distance, at=camera_position[1])
    #     # R = R[0]
    #     # # print(R)
    #     # T = T[0].view(-1, 1)
    #     # # print(T)
    #     # RT = torch.cat((R, T), dim=1).numpy()
    #     if i < 16:

    #         # 设置相机的视角
    #         azimuth_angle = 360.0 * ((i + 1) / 16)
    #         elevation_angle = 0.0
    #         if i == 15:
    #             elevation_angle = 30.0
    #     # print(azimuth_angle)
    #     # exit()
    #     elif i < 48:
    #         azimuth_angle = 360.0 * ((i - 16 + 1) / 32)
    #         elevation_angle = 30.0
    #         if i == 47:
    #             elevation_angle = 90.0

    #     # depth = plotter.get_image_depth(fill_value=0)
    #     # if (depth.max() - depth.min()) == 0:
    #     #     print(obj_path)
    #     #     flag = 1
    #     #     break
    #     # rgb = plotter.image
    #     # np.savetxt('/mnt/hdd1/caixiao/data/pv_views/4543efbf06cc4d899740b53d957bee7a/test2.txt', depth)
    #     # exit()
    #     # depth -= depth.min()

    #     # depth /= depth.max()
    #     # # 将浮点数数组转换为0-255的整数数组
    #     # depth = (depth * 255).astype(np.int8)

    #     path = f'/mnt/hdd1/caixiao/data/pv_cameras/{info_id}'
    #     os.makedirs(path, exist_ok=True)
    #     # print(path)
    #     # 将数据写入JSON文件
    #     with open(f'/mnt/hdd1/caixiao/data/pv_cameras/{info_id}/camera_{i}.json', 'w') as f:
    #         json.dump(data, f)
    #     # with open(f'/mnt/hdd1/caixiao/data/pv_cameras/{info_id}/camera_{i}.json', 'r') as f:
    #     #     data_loaded = json.load(f)
    #     #     print('camera: ', data_loaded['camera_position'][0])
    #     # exit()
    #     # path = f'/mnt/hdd1/caixiao/data/pv_views_64/{info_id}/rgb'
    #     # os.makedirs(path, exist_ok=True)
    #     # path = f'/mnt/hdd1/caixiao/data/pv_views_64/{info_id}/depth'
    #     # os.makedirs(path, exist_ok=True)

    #     # np.save(f'/mnt/hdd1/caixiao/data/pv_views_64/{info_id}/pose/pose_{i}.npy', RT)
    #     # plt.imsave(f'/mnt/hdd1/caixiao/data/pv_views_64/{info_id}/depth/depth_{i}.png', depth, cmap='gray')
    #     # plt.imsave(f'/mnt/hdd1/caixiao/data/pv_views_64/{info_id}/rgb/rgb_{i}.png', rgb)

    #     # plt.figure(figsize=(512, 512), dpi=1)
    #     # plt.axis('off')
    #     # plt.imshow(depth,cmap=plt.cm.gray_r)
    #     # plt.savefig(f"/mnt/hdd1/caixiao/data/pv_views/{info_id}/depth_{i}.png")
    #     # plt.close()
    #     # plt.figure(figsize=(512, 512), dpi=1)
    #     # plt.axis('off')
    #     # plt.imshow(rgb, cmap=plt.cm.gray_r)
    #     # plt.savefig(f"/mnt/hdd1/caixiao/data/pv_views/{info_id}/rgb_{i}.png")
    #     # plt.close()
    #     # dep = Image.fromarray(depth)
    #     # dep.save(f'/mnt/hdd1/caixiao/data/pv_views/{info_id}/depth_{i}.png')
    #     # img = Image.fromarray(rgb)
    #     # img.save(f'/mnt/hdd1/caixiao/data/pv_views/{info_id}/rgb_{i}.png')
    #     # print(azimuth_angle)
    #     plotter.camera.azimuth = azimuth_angle
    #     plotter.camera.elevation = elevation_angle
    # # t4 = time.time()
    # plotter.deep_clean()
    # plotter.close()
    # plotter = pv.Plotter(off_screen=True, window_size=(512, 512), image_scale=1)
    
    # plotter.set_background('black')
    # try:
    #     plotter.import_gltf(obj_path)
    # except Exception as e:
    #     # print(e)
    #     continue
    # plotter.show_axes()
    azimuth_angle = 0
    elevation_angle = 0
    plotter.camera.azimuth = azimuth_angle
    plotter.camera.elevation = elevation_angle
    plotter.show(auto_close=False)
    num_images = 64
    depths = []
    for i in range(num_images):
        # plotter.set_viewup([0, 1, 0], reset=False)
        # print(i)
        camera_position = plotter.camera_position
        clipping_range = plotter.camera.clipping_range
        distance = plotter.camera.distance
        scale = plotter.scale
        fov = plotter.camera.view_angle

        data = {
            'camera_position': camera_position[0:],
            'distance': distance,
            'azimuth': azimuth_angle,
            'elev': elevation_angle,
            'clipping_range': clipping_range,
            'scale': scale,
            'fov': fov
        }
        # R, T = look_at_view_transform(elev=elevation_angle, azim=azimuth_angle, dist=distance, at=camera_position[1])
        # R = R[0]
        # # print(R)
        # T = T[0].view(-1, 1)
        # # print(T)
        # RT = torch.cat((R, T), dim=1).numpy()
        if i < 16:
            # 设置相机的视角
            azimuth_angle = 360.0 * ((i + 1) / 16)
            elevation_angle = 0.0
            if i == 15:
                elevation_angle = 30.0
        # print(azimuth_angle)
        # exit()
        elif i < 48:
            azimuth_angle = 360.0 * ((i - 16 + 1) / 32)
            elevation_angle = 30.0
            if i == 47:
                azimuth_angle = 382.5
                elevation_angle = 61.875
        else:
            azimuth_angle = 360.0 * ((i- 48 + 2) / 16)
            elevation_angle = 61.875 + 30.0 * ((i - 48 + 1) / 16)

        depth = plotter.get_image_depth(fill_value=0)
        depths.append(depth)
        if (depth.max() - depth.min()) == 0:
            print(obj_path)
            break
        deps_path = os.path.join(base_path, info_id, 'depth')
        os.makedirs(deps_path, exist_ok=True)
        # print(deps_path)
        # depth_all = np.stack(depths)
        # print(depth_all.shape)
        # print(depth.shape)
        
        # np.save(os.path.join(deps_path, f'depth_{i}.npy'), depth)
        np.savez_compressed(os.path.join(deps_path, f'depth_{i}.npz'), data=depth)
        
        rgb = plotter.image
        print(rgb.shape)
        exit()
        # print(plotter.camera.azimuth)
        # print(plotter.camera.elevation)
        path = os.path.join(base_path, info_id)
        
        # if os.path.exists(path):
        cam_path = os.path.join(path, 'camera')
        os.makedirs(cam_path, exist_ok=True)
        with open(os.path.join(cam_path, f'camera_{i}.json'), 'w') as f:
            json.dump(data, f)
        # else:
        #     break
        
        # path0 = f'/mnt/hdd1/caixiao/data/pv_views_64/{info_id}'
        # path1 = f'/mnt/hdd1/caixiao/data/pv_views_64_1/{info_id}'
        # if os.path.exists(path1):
            # print(path1)
        
        # 
        rgb_path = os.path.join(path, 'rgb')
        os.makedirs(rgb_path, exist_ok=True)
        # print(depth[255])
        plt.imsave(os.path.join(rgb_path, f'rgb_{i}.png'), rgb)
        # dep = np.load(os.path.join(dep_path, f'depth_{i}.npy'))
        # plt.imsave(os.path.join(dep_path, f'depth_{i}.png'), dep, cmap='gray')
        # print(dep[255][255])
        # exit()
        # os.makedirs(f'/mnt/nfs/caixiao/caixiao_backup/pv_views_sup/{info_id}/rgb', exist_ok=True)
        # plt.imsave(f'/mnt/nfs/caixiao/caixiao_backup/pv_views_sup/{info_id}/depth/depth_{i+48}.png', depth, cmap='gray')
        # plt.imsave(f'/mnt/nfs/caixiao/caixiao_backup/pv_views_sup/{info_id}/rgb/rgb_{i+48}.png', rgb)
        # elif os.path.exists(path0):
        #     # print(path0)
        #     os.makedirs(f'/mnt/nfs/caixiao/caixiao_backup/pv_views_sup/{info_id}/depth', exist_ok=True)
        #     os.makedirs(f'/mnt/nfs/caixiao/caixiao_backup/pv_views_sup/{info_id}/rgb', exist_ok=True)
        #     plt.imsave(f'/mnt/nfs/caixiao/caixiao_backup/pv_views_sup/{info_id}/depth/depth_{i+48}.png', depth, cmap='gray')
        #     plt.imsave(f'/mnt/nfs/caixiao/caixiao_backup/pv_views_sup/{info_id}/rgb/rgb_{i+48}.png', rgb)
        # else:
        #     print(info_id)
        #     break
        # os.makedirs(path, exist_ok=True)
        # path = f'/home/caixiao/projects/3d_lib/camera/{info_id}/rgb'
        # os.makedirs(path, exist_ok=True)
        
        # np.save(f'/mnt/hdd1/caixiao/data/pv_views_64/{info_id}/pose/pose_{i+48}.npy', RT)
        
        plotter.camera.azimuth = azimuth_angle
        plotter.camera.elevation = elevation_angle
        
    # deps_path = os.path.join(base_path, info_id, 'depth')
    # os.makedirs(deps_path, exist_ok=True)
    # print(deps_path)
    # depth_all = np.stack(depths)
    # print(depth_all.shape)
    # print(depth.shape)
    
    # np.save(os.path.join(deps_path, 'depth_all.npy'), depth_all)
    # dep = np.load(os.path.join(dep_path, 'depth_all.npy'))
    # for i in range(64):
    #     plt.imsave(os.path.join(dep_path, f'depth_{i}.png'), dep[i], cmap='gray')
    plotter.deep_clean()
    plotter.close()
    exit()        
vdisplay.stop()