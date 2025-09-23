import json
import os

# NOTE: Do NOT set PyVista/VTK-related environment variables here at import time.
# For multi-GPU rendering we set them per-worker in the process initializer so each
# worker binds to a specific GPU before importing VTK/PyVista.

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
import numpy as np
import os
import matplotlib
# Use a headless backend for matplotlib image saving
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm
# from process import look_at_view_transform
import pandas as pd
import time
import csv
import multiprocessing
 
 
import numpy as np

# IMPORTANT: We import VTK/PyVista inside worker functions after environment
# variables are set to ensure EGL context binds to the correct GPU.


def sanitize_gltf_single_uv(in_path: str) -> str:
    """
    If the input is a GLTF/GLB, remove additional TEXCOORD_* attributes so only TEXCOORD_0 remains.
    Returns a path to a sanitized copy if successful, otherwise returns the original path.
    """
    ext = os.path.splitext(in_path)[1].lower()
    if ext not in {".gltf", ".glb"}:
        return in_path
    try:
        from pygltflib import GLTF2
    except Exception:
        # Library not available; skip sanitization
        return in_path

    try:
        gltf = GLTF2().load(in_path)
        changed = False
        for mesh in gltf.meshes or []:
            for prim in mesh.primitives or []:
                attrs = prim.attributes
                # pygltflib represents attributes with named fields if present
                # We keep TEXCOORD_0 and drop TEXCOORD_1..7 if they exist
                for i in range(1, 8):
                    name = f"TEXCOORD_{i}"
                    if hasattr(attrs, name):
                        if getattr(attrs, name) is not None:
                            setattr(attrs, name, None)
                            changed = True
        if not changed:
            return in_path

        base, ext = os.path.splitext(in_path)
        out_path = base + "_sanitized" + ext
        gltf.save(out_path)
        return out_path if os.path.isfile(out_path) else in_path
    except Exception:
        # Any parsing/saving issue -> fall back to original file
        return in_path
def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def sph_to_pos(az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return np.array([x, y, z])

def rotz_deg(angle_deg):
    a = np.deg2rad(angle_deg)
    ca = np.cos(a); sa = np.sin(a)
    return np.array([[ca, -sa, 0],[sa, ca, 0],[0,0,1]])

def rotation_matrix_from_axis_angle(axis, angle_rad):
    axis = normalize(axis)
    ux, uy, uz = axis
    c = np.cos(angle_rad); s = np.sin(angle_rad)
    R = np.array([
        [c + ux*ux*(1-c),    ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c),    uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)]
    ])
    return R

# ------------------ 从 (az,el,roll) 构造相机基座 ------------------
def build_camera_R_from_az_el_roll(az_deg, el_deg, roll_deg, world_up=np.array([0,0,1])):
    """
    返回 3x3 矩阵 R（camera->world），列分别是 [right, up, forward_zc]
    约定：
      - az,el 表示相机位置在单位球面的球坐标（和之前的约定一致）
      - forward zc = -pos（camera 朝向物体）
      - roll 表示绕 forward 轴的图像内旋转（度）
    """
    pos = sph_to_pos(az_deg, el_deg)
    zc = -normalize(pos)  # forward（在 world 坐标）
    # 处理 world_up 与 zc 共线的奇异情况
    if abs(np.dot(normalize(world_up), zc)) > 0.999999:
        # 换一个参考 up
        world_up2 = np.array([0.0, 1.0, 0.0])
    else:
        world_up2 = world_up
    right0 = normalize(np.cross(world_up2, zc))
    up0 = normalize(np.cross(zc, right0))
    rho = np.deg2rad(roll_deg)
    right = right0 * np.cos(rho) + up0 * np.sin(rho)
    up    = up0    * np.cos(rho) - right0 * np.sin(rho)
    R = np.column_stack([right, up, zc])
    return R

# ------------------ 从 R 回算 (az,el,roll) ------------------
def extract_az_el_roll_from_R(R, world_up=np.array([0,0,1])):
    """
    R: 3x3 矩阵，列为 [right, up, zc]
    返回 (az_deg, el_deg, roll_deg) 与构造时的约定对应。
    """
    right = R[:,0]; up = R[:,1]; zc = R[:,2]
    pos = -zc  # camera position on unit sphere
    x,y,z = pos
    az = np.degrees(np.arctan2(y, x)) % 360
    el = np.degrees(np.arcsin(np.clip(z / np.linalg.norm(pos), -1, 1)))
    # 复原 roll：基于 right0/up0（无 roll 情况）
    if abs(np.dot(normalize(world_up), zc)) > 0.999999:
        world_up2 = np.array([0.0, 1.0, 0.0])
    else:
        world_up2 = world_up
    right0 = normalize(np.cross(world_up2, zc))
    up0 = normalize(np.cross(zc, right0))
    rho = np.arctan2(-np.dot(up, right0), np.dot(up, up0))
    roll = np.degrees(rho)
    # 规范化到 [-180,180)
    if roll >= 180: roll -= 360
    if roll < -180: roll += 360
    return az, el, roll

# ------------------ 构造 left/top（严格的几何操作） ------------------
def make_left_from_R(R_front):
    """绕全局 world_up (z) 旋转 -90°：left = R_z(-90) @ R_front"""
    return rotz_deg(-90) @ R_front

def make_top_from_R(R_front):
    """把 front 的 forward (zc) 旋转到 (0,0,-1) 的旋转（轴角），然后作用到整个基座上"""
    zc = R_front[:,2]
    target = np.array([0.0, 0.0, -1.0])
    dot = np.dot(zc, target)
    if dot > 0.999999:
        return R_front.copy()   # 已经是 top
    if dot < -0.999999:
        axis = np.array([1.0, 0.0, 0.0])
        angle = np.pi
    else:
        axis = normalize(np.cross(zc, target))
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
    S = rotation_matrix_from_axis_angle(axis, angle)
    return S @ R_front
def calculate_camera_position(azimuth, elevation,roll):
    #az_f, el_f, roll_f = azimuth, elevation, 0.0
    #Rf = build_camera_R_from_az_el_roll(az_f, el_f, roll_f)
    #Rl = make_left_from_R(Rf)
    #Rt = make_top_from_R(Rf)
    #return {
    #   'front':extract_az_el_roll_from_R(Rf),
    #   'left':extract_az_el_roll_from_R(Rl),
    #   'up':extract_az_el_roll_from_R(Rt),
    #}
    return {
    'front':(azimuth, elevation, roll),
    #'left':(azimuth - 90, roll, -elevation),
    'right':(azimuth+90,-roll,elevation),
    'up':(azimuth, elevation + 90, roll),
    }




def render_model(obj_path, output_dir, num_views):
    """Render a model from multiple views and save the results."""
    # Import VTK/PyVista lazily inside the worker so env vars (GPU binding,
    # off-screen EGL) are already in place.
    # Silence noisy VTK warnings like GLTF multiple UV sets.
    try:
        import vtk  # type: ignore
        if hasattr(vtk, "vtkLogger"):
            vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_ERROR)
        if hasattr(vtk, "vtkObject") and hasattr(vtk.vtkObject, "GlobalWarningDisplayOff"):
            vtk.vtkObject.GlobalWarningDisplayOff()
    except Exception:
        pass

    import pyvista as pv
    
    #try:
    #    # Pre-sanitize GLTF/GLB to keep a single UV set to avoid VTK warnings and potential issues
    #    safe_path = sanitize_gltf_single_uv(obj_path)
    #    obj = plotter.import_gltf(safe_path)
    #except Exception as e:
    #    print(f"Failed to load {obj_path}: {e}")
    #    return

    # 20 pairs of angles in [0, 360)
    rng = np.random.default_rng(42)
    angle_pairs = rng.uniform(0.0, 360.0, size=(num_views, 2))
    
    #base = (165,0,-20)
    #base =  (75,20,0)
    #base=(-15,0,-20)
    base=(0,0,0)
    # Render and save images for each pair and each view
    for i, (azi, ele) in enumerate(angle_pairs):
        #views = calculate_camera_position(0, ele)
        
        views = calculate_camera_position(base[0]+i*1.0/num_views*360*4,base[1]+i*4//num_views*22.5+1e-6,base[2])
        
        # Create base directories: output_dir/i/{depth|rgb}
        base_dir = os.path.join(output_dir, str(i))
        depth_dir = os.path.join(base_dir, "depth")
        rgb_dir = os.path.join(base_dir, "rgb")
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(rgb_dir, exist_ok=True)

        for view_name, (azimuth, elevation, roll) in views.items():
            # Set camera
            rgb_path = os.path.join(rgb_dir, f"{view_name}.png")
            if os.path.exists(rgb_path):
                continue
            plotter = pv.Plotter(off_screen=True, window_size=(512, 512), image_scale=1)
            plotter.set_background('white')
            obj = plotter.import_gltf(obj_path)
            plotter.show(auto_close=False)
            #if view_name=='front': continue
            #print(f"Rendering view {i} {view_name} az={azimuth:.1f} el={elevation:.1f} roll={roll:.1f}")
            plotter.camera.azimuth = float(azimuth)
            plotter.camera.elevation = float(elevation)
            plotter.camera.SetRoll(float(roll))
            #plotter.render()

            # Save depth: output_dir/i/depth/{view_name}.npz
            depth = plotter.get_image_depth(fill_value=0)
            depth_path = os.path.join(depth_dir, f"{view_name}.npz")
            np.savez_compressed(depth_path, data=depth)

            # Save RGB: output_dir/i/rgb/{view_name}.png
            rgb = plotter.image
            
            plt.imsave(rgb_path, rgb)

            plotter.deep_clean()
            plotter.close()
    #print(f"Rendered {obj_path} to {output_dir}")
    

def process_model(args):
    """Wrapper function for multiprocessing."""
    obj_path, output_dir, num_views = args
    render_model(obj_path, output_dir, num_views)

def set_worker_affinity(core_list):
    """
    Pin this worker process to a specific CPU core based on worker identity.
    """
    try:
        worker_idx = multiprocessing.current_process()._identity[0] - 1  # 0-based
    except Exception:
        worker_idx = 0
    if not core_list:
        return
    core = core_list[worker_idx % len(core_list)]
    try:
        # Linux: pin current process to the selected core
        os.sched_setaffinity(0, {core})
    except Exception:
        # Fallback via psutil if available or on non-Linux
        try:
            import psutil
            psutil.Process().cpu_affinity([core])
        except Exception:
            pass

def set_worker_affinity_and_device(core_list, gpu_list, use_egl=True):
    """Pin worker to a CPU core and bind it to a specific GPU via env vars.

    This must run before importing VTK/PyVista in the worker process.
    """
    # CPU core pinning
    try:
        worker_idx = multiprocessing.current_process()._identity[0] - 1  # 0-based
    except Exception:
        worker_idx = 0
    if core_list:
        core = core_list[worker_idx % len(core_list)]
        try:
            os.sched_setaffinity(0, {core})
        except Exception:
            try:
                import psutil
                psutil.Process().cpu_affinity([core])
            except Exception:
                pass

    # GPU binding
    gpu_id = None
    if gpu_list:
        gpu_id = gpu_list[worker_idx % len(gpu_list)]
        # Restrict visibility to a single physical GPU for this worker
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Some stacks use this variable name as a hint
        os.environ["EGL_DEVICE_ID"] = "0"  # after masking, the single device is index 0

    # Configure PyVista for off-screen rendering
    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    if use_egl and (gpu_id is not None):
        os.environ["PYVISTA_USE_EGL"] = "true"
        # Help some OpenGL stacks pick EGL
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        # If a single GPU is visible, make sure index 0 is chosen by VTK
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            os.environ["VTK_DEFAULT_EGL_DEVICE"] = "0"
    else:
        # Fallback to software OSMesa if no GPU is available
        os.environ["PYVISTA_USE_OSMESA"] = "true"
        os.environ.pop("PYVISTA_USE_EGL", None)

if __name__ == "__main__":
    # Multi-GPU rendering setup: we will launch one worker per GPU
    # and bind the EGL context of that worker to its assigned GPU.
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    # Load JSON list of file paths
    with open('/home/linzhuohang/3DGen/utils/high_score2.json', 'r') as f:
        obj_paths = json.load(f)

    # Prepare arguments for multiprocessing
    num_views = 30
    tasks = []
    for obj_path in obj_paths:
        info_id = os.path.splitext(os.path.basename(obj_path))[0]
        #if info_id !='0d94fa80e87e49e2b0747d1252b9e3bd':continue
        output_dir = os.path.join('/mnt/hdd3/linzhuohang/3DGen/data/high_score/', info_id)
        #os.makedirs(output_dir, exist_ok=True)
        tasks.append((obj_path, output_dir, num_views))

    # Discover available GPUs (respecting any user-provided CUDA visibility)
    # Build GPU list: respect pre-set CUDA visibility if present
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    gpu_list: list[int]
    if cuda_visible:
        try:
            gpu_list = [int(x) for x in cuda_visible.split(",") if x != ""]
        except Exception:
            gpu_list = []
    else:
        try:
            visible_gpu_count = torch.cuda.device_count()
        except Exception:
            visible_gpu_count = 0
        gpu_list = list(range(visible_gpu_count))
    if visible_gpu_count == 0:
        print("[WARN] No CUDA GPUs detected; falling back to software OSMesa rendering.")

    # Use multiprocessing to render models: one worker per GPU when possible
    if visible_gpu_count > 0:
        num_processes = min(len(gpu_list), len(tasks))
    else:
        num_processes = min(multiprocessing.cpu_count(), len(tasks))

    # Select distinct cores for workers
    try:
        available_cores = sorted(os.sched_getaffinity(0))
    except AttributeError:
        available_cores = list(range(os.cpu_count() or 1))
    core_list = list(available_cores)[:num_processes]

    with multiprocessing.Pool(
        processes=num_processes,
        initializer=set_worker_affinity_and_device,
        initargs=(core_list, gpu_list, True)
    ) as pool:
        for _ in tqdm(pool.imap_unordered(process_model, tasks, chunksize=1),
                      total=len(tasks), dynamic_ncols=True):
            pass
    # Done

