import json
import pyvista as pv
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import pyvista as pv
import glob

def update_csv(filename, id, idx, pyvista_open):
    new_row = [id, idx, pyvista_open]
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(new_row)

from xvfbwrapper import Xvfb
vdisplay = Xvfb(width=1280, height=640)
vdisplay.start()

# GSO数据集根目录
gso_base_path = '/mnt/hdd3/linzhuohang/3DGen/gso'
# 输出目录
output_dir = '/mnt/hdd3/linzhuohang/3DGen/gso_rendered'



# 使用glob获取所有子文件夹的路径，作为模型ID
gso_model_paths = glob.glob(os.path.join(gso_base_path, '*'))
gso_models = [os.path.basename(p) for p in gso_model_paths]

print(f"load {len(gso_models)} objs in GSO dataset")

for model_id in tqdm(gso_models):
    # 构建OBJ文件路径
    obj_path = os.path.join(gso_base_path, model_id, 'meshes', 'model.obj')
    # 构建纹理文件路径
    texture_path = os.path.join(gso_base_path, model_id, 'materials', 'textures', 'texture.png')

    # 检查OBJ文件是否存在
    if not os.path.exists(obj_path):
        print(f"OBJ file not found: {obj_path}")
        continue
    
    plotter = pv.Plotter(off_screen=True, window_size=(512, 512), image_scale=1)
    plotter.set_background('white')
    
    pyvista_open = '1'
    try:
        # 使用pv.read加载OBJ文件
        mesh = pv.read(obj_path)
        # 尝试加载纹理
        if os.path.exists(texture_path):
            tex = pv.read_texture(texture_path)
            plotter.add_mesh(mesh, texture=tex)
        else:
            plotter.add_mesh(mesh)
    except Exception as e:
        print(f"Error reading OBJ file {obj_path}: {e}")
        pyvista_open = '0'
        # 记录加载失败的模型
        idx = model_id[:3]
        csv_path = os.path.join(output_dir, 'utils', f'{idx}.txt')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        update_csv(csv_path, model_id, idx, pyvista_open)
        continue

    # 记录成功加载的模型ID
    idx = model_id[:3]
    csv_path = os.path.join(output_dir, 'utils', f'{idx}.txt')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    update_csv(csv_path, model_id, idx, pyvista_open)

    # 设置相机初始视角
    azimuth_angle = 0
    elevation_angle = 0
    plotter.camera.azimuth = azimuth_angle
    plotter.camera.elevation = elevation_angle
    plotter.show(auto_close=False)

    num_images = 64
    for i in range(num_images):
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
        
        # 视点设置
        if i < 16:
            azimuth_angle = 360.0 * ((i + 1) / 16)
            elevation_angle = 0.0
            if i == 15:
                elevation_angle = 30.0
        elif i < 48:
            azimuth_angle = 360.0 * ((i - 16 + 1) / 32)
            elevation_angle = 30.0
            if i == 47:
                azimuth_angle = 382.5
                elevation_angle = 61.875
        else:
            azimuth_angle = 360.0 * ((i - 48 + 2) / 16)
            elevation_angle = 61.875 + 30.0 * ((i - 48 + 1) / 16)

        # 获取深度图和RGB图像
        depth = plotter.get_image_depth(fill_value=0)
        if (depth.max() - depth.min()) == 0:
            print(f"Empty depth map for {model_id}, view {i}")
            break

        # 保存深度图
        deps_path = os.path.join(output_dir, model_id, 'depth')
        os.makedirs(deps_path, exist_ok=True)
        np.savez_compressed(os.path.join(deps_path, f'depth_{i}.npz'), data=depth)
        
        # 保存RGB图像
        rgb = plotter.image
        rgb_path = os.path.join(output_dir, model_id, 'rgb')
        os.makedirs(rgb_path, exist_ok=True)
        plt.imsave(os.path.join(rgb_path, f'rgb_{i}.png'), rgb)

        # 保存相机数据
        cam_path = os.path.join(output_dir, model_id, 'camera')
        os.makedirs(cam_path, exist_ok=True)
        with open(os.path.join(cam_path, f'camera_{i}.json'), 'w') as f:
            json.dump(data, f)
        
        # 更新相机视角
        plotter.camera.azimuth = azimuth_angle
        plotter.camera.elevation = elevation_angle
    
    plotter.deep_clean()
    plotter.close()
    
vdisplay.stop()