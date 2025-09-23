import json 
import os
import random
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import os


# ---------------- Camera Utils ---------------- #
def get_camera_extrinsic(azimuth, elevation, roll, distance=3.0, lookat=(0, 0, 0)):
    """生成相机外参矩阵 (4x4)"""
    az = np.deg2rad(azimuth)
    el = np.deg2rad(elevation)
    rl = np.deg2rad(roll)

    # 球坐标 -> 相机位置
    x = distance * np.cos(el) * np.cos(az)
    y = distance * np.cos(el) * np.sin(az)
    z = distance * np.sin(el)
    cam_pos = np.array([x, y, z])

    lookat = np.array(lookat)
    forward = lookat - cam_pos
    forward /= np.linalg.norm(forward)

    world_up = np.array([0, 0, 1], dtype=float)
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0, 1, 0], dtype=float)

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # 应用 roll
    if roll != 0:
        cosr, sinr = np.cos(rl), np.sin(rl)
        up = cosr * up + sinr * right
        right = np.cross(forward, up)

    R = np.stack([right, up, -forward], axis=1)
    t = -R.T @ cam_pos

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R.T
    extrinsic[:3, 3] = t
    return extrinsic, cam_pos.tolist(), lookat.tolist(), up.tolist()


def calculate_camera_views(azimuth, elevation, roll):
    """返回 front/left/up 三个视角"""
    return {
    'front':(azimuth, elevation, roll),
    'left':(azimuth - 90, -roll, elevation),
    'right':(azimuth+90,roll,-elevation),
    'up':(azimuth, elevation + 90, roll),
    }


# ---------------- Rendering ---------------- #
def render_model(obj_path, output_dir, num_views=10, image_size=(512, 512), fov=60.0):
    try:
        mesh = o3d.io.read_triangle_model(obj_path)
    except Exception as e:
        print(f"Failed to load {obj_path}: {e}")
        return

    w, h = image_size
    renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"

    for mi, geom in enumerate(mesh.meshes):
        name = f"mesh_{mi}"
        renderer.scene.add_geometry(name, geom.mesh, mat)

    bounds = renderer.scene.bounding_box
    center = bounds.get_center()
    extent = np.linalg.norm(bounds.get_extent())
    distance = extent * 2.5
    base = (165, 0, -20)
    for i in range(num_views):
        views = calculate_camera_views(base[0] + i * 1.0 / num_views * 360 * 3, base[1] + i * 3 // num_views * 20 + 1e-6, base[2])

        base_dir = os.path.join(output_dir, str(i))
        depth_dir = os.path.join(base_dir, "depth")
        rgb_dir = os.path.join(base_dir, "rgb")
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(rgb_dir, exist_ok=True)

        camera_info = {}

        for view_name, (azimuth, elevation, roll) in views.items():
            #print('Rendering view:', i, view_name)
            extrinsic, pos, focal, up = get_camera_extrinsic(
                azimuth, elevation, roll, distance, lookat=center
            )

            renderer.scene.camera.set_projection(
                fov, w / h, 0.1, 100.0, o3d.visualization.rendering.Camera.FovType.Vertical
            )
            #print('Camera position:', pos)
            renderer.scene.camera.look_at(
                np.array(center, dtype=np.float32),
                np.array(pos, dtype=np.float32),
                np.array(up, dtype=np.float32)
            )
            #print('camera rotated')


            # RGB
            img = renderer.render_to_image()
            #print('image rendered')
            rgb_path = os.path.join(rgb_dir, f"{view_name}.png")
            o3d.io.write_image(rgb_path, img)
            

            # Depth
            depth_img = renderer.render_to_depth_image(z_in_view_space=True)
            depth_np = np.asarray(depth_img)
            depth_path = os.path.join(depth_dir, f"{view_name}.npz")
            np.savez_compressed(depth_path, data=depth_np)

            camera_info[view_name] = {
                "azimuth": float(azimuth),
                "elevation": float(elevation),
                "roll": float(roll),
                "position": pos,
                "focal_point": focal,
                "up": up,
                "fov": float(fov),
                "center": center.tolist(),
                "distance": float(distance),
            }

        json_path = os.path.join(base_dir, "cameras.json")
        with open(json_path, "w") as jf:
            json.dump(camera_info, jf, indent=2)


def process_model(args):
    obj_path, output_dir, num_views = args
    render_model(obj_path, output_dir, num_views)


def set_worker_affinity(core_list):
    try:
        worker_idx = multiprocessing.current_process()._identity[0] - 1
    except Exception:
        worker_idx = 0
    if not core_list:
        return
    core = core_list[worker_idx % len(core_list)]
    try:
        os.sched_setaffinity(0, {core})
    except Exception:
        try:
            import psutil

            psutil.Process().cpu_affinity([core])
        except Exception:
            pass


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    with open('/home/linzhuohang/3DGen/utils/high_score.json', 'r') as f:
        obj_paths = json.load(f)

    num_views = 20
    tasks = []
    for obj_path in obj_paths:
        info_id = os.path.splitext(os.path.basename(obj_path))[0]
        if info_id !='0d94fa80e87e49e2b0747d1252b9e3bd':continue
        output_dir = os.path.join('/mnt/hdd3/linzhuohang/3DGen/data/high_score2/', info_id)
        #render_model(obj_path, output_dir, num_views)
        tasks.append((obj_path, output_dir, num_views))

    num_processes = min(multiprocessing.cpu_count(), len(tasks))

    try:
        available_cores = sorted(os.sched_getaffinity(0))
    except AttributeError:
        available_cores = list(range(os.cpu_count() or 1))
    core_list = list(available_cores)[:num_processes]

    with multiprocessing.Pool(
        processes=num_processes,
        initializer=set_worker_affinity,
        initargs=(core_list,)
    ) as pool:
        for _ in tqdm(pool.imap_unordered(process_model, tasks, chunksize=1),
                      total=len(tasks), dynamic_ncols=True):
            pass
