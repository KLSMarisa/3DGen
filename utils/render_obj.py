import json
import os
import shutil

# Force PyVista to use OSMesa for off-screen rendering
#os.environ["PYVISTA_OFF_SCREEN"] = "true"
#os.environ["PYVISTA_USE_OSMESA"] = "true"
#os.environ["PYVISTA_PLOT_THEME"] = "document"
#os.environ["VTK_DEFAULT_EGL_DEVICE"] = "software"

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
import multiprocessing
from utils import calc_angle
 
import numpy as np
from torchvision import transforms as Trans
# Silence noisy VTK warnings like:
# "vtkGLTFImporter: Using multiple texture coordinates for the same model is not supported."
# We only need errors to surface; warnings from GLTF importer are harmless for our use-case.
try:
    import vtk  # type: ignore
    # Prefer logger verbosity control when available (VTK 9+)
    if hasattr(vtk, "vtkLogger"):
        vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_ERROR)
    # Also disable legacy global warning popups/prints just in case
    if hasattr(vtk, "vtkObject") and hasattr(vtk.vtkObject, "GlobalWarningDisplayOff"):
        vtk.vtkObject.GlobalWarningDisplayOff()
except Exception:
    # If VTK isn't available or the API differs, proceed without silencing
    pass


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

def compute_fitting_distance(plotter, margin: float = 1.1) -> float | None:
    bounds = plotter.bounds
    if bounds is None:
        return None
    x_len = bounds[1] - bounds[0]
    y_len = bounds[3] - bounds[2]
    z_len = bounds[5] - bounds[4]
    max_extent = max(x_len, y_len, z_len)
    if max_extent <= 0:
        return None
    view_angle = np.deg2rad(plotter.camera.view_angle or 30.0)
    return margin * (0.5 * max_extent) / np.sin(view_angle / 2.0)



from vtk.util.numpy_support import vtk_to_numpy
def render_single_view(obj_path, azimuth, elevation, roll, output_dir=None):
    t_total_start = time.time()
    t0 = time.time()
    views = calc_angle.views_from_front_360(azimuth, elevation, roll)
    t_gen = time.time() - t0
    imgs = []
    render_time = 0.0
    post_time = 0.0
    for view_name, (azimuth, elevation, roll) in views.items():
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{view_name}.png")
            #if os.path.exists(output_path):
            #    continue
        t_r0 = time.time()
        plotter = pv.Plotter(off_screen=True, window_size=(256, 256), image_scale=1)
        plotter.set_background('white')
        obj = plotter.import_gltf(obj_path)
        plotter.show(auto_close=False)
        plotter.reset_camera()
        focal_point = plotter.camera.focal_point
        fit_distance = compute_fitting_distance(plotter)
        if fit_distance:
            plotter.camera.SetDistance(fit_distance)
        plotter.camera.azimuth = float(azimuth)
        plotter.camera.elevation = float(elevation)
        plotter.camera.SetRoll(float(roll))
        if fit_distance:
            plotter.camera.SetFocalPoint(*focal_point)
            plotter.camera.SetDistance(fit_distance)
        plotter.render()
        if output_dir:
            output_path = os.path.join(output_dir, f"{view_name}.png")
            rgb = plotter.image
            plt.imsave(output_path, rgb)
        else:
            w2if = vtk.vtkWindowToImageFilter()
            w2if.SetInput(plotter.ren_win)
            w2if.ReadFrontBufferOff()
            w2if.Update()
            vtk_img = w2if.GetOutput()
            vtk_array = vtk_img.GetPointData().GetScalars()
            w, h = plotter.ren_win.GetSize()
            np_img = vtk_to_numpy(vtk_array).reshape(h, w, -1)
            np_img = np.flipud(np_img)[:, :, :3]
            t_r1 = time.time()
            # 后处理
            if (h, w) != (256, 256):
                np_img = np.asarray(Image.fromarray(np_img).resize((256, 256), Image.BILINEAR))
            img = np_img.astype(np.float32) / 127.5 - 1.0
            tensor_image = torch.from_numpy(img.transpose(2, 0, 1))
            imgs.append(tensor_image[:3])
            t_p1 = time.time()
            render_time += (t_r1 - t_r0)
            post_time += (t_p1 - t_r1)
        plotter.deep_clean()
        plotter.close()
        del plotter, obj
    if output_dir:
        return
    total = time.time() - t_total_start
    n = len(imgs)
    #print(f"render_single_view: {os.path.basename(obj_path)} views={n} "
    #      f"total={total:.3f}s gen={t_gen:.3f}s render={render_time:.3f}s "
    #      f"post={post_time:.3f}s avg_render={render_time/max(n,1):.3f}s "
    #      f"avg_post={post_time/max(n,1):.3f}s")
    return imgs

def render_model(obj_path, output_dir, num_views,cover_existing=False):
    """Render a model from multiple views and save the results."""
    
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
    
    for i in range(num_views):
        choice =np.random.randint(0,100)
        if choice>10:
            azimuth = np.random.uniform(0, 360)
            elevation = np.random.uniform(0, 360)
            roll = np.random.uniform(0, 360)
        else:
            azimuth = 0
            elevation = 0
            roll = 0
        
        render_single_view(obj_path, azimuth, elevation, roll, os.path.join(output_dir, str(i)))
        
        # Render and save images for each pair and each view
    return
    base=(0,0,30)
    for i, (azi, ele) in enumerate(angle_pairs):
        #views = calculate_camera_position(0, ele)
        
        views = calc_angle.views_from_front_360(base[0]+i*1.0/num_views*360*4,base[1]+i*4//num_views*22.5+1e-6,base[2])
        
        # Create base directories: output_dir/i/{depth|rgb}
        base_dir = os.path.join(output_dir, str(i))
        depth_dir = os.path.join(base_dir, "depth")
        rgb_dir = os.path.join(base_dir, "rgb")
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(rgb_dir, exist_ok=True)

        for view_name, (azimuth, elevation, roll) in views.items():
            # Set camera
            rgb_path = os.path.join(rgb_dir, f"{view_name}.png")
            if os.path.exists(rgb_path) and not cover_existing:
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
            depth = plotter.get_image_depth(fill_value=0)
            depth_path = os.path.join(depth_dir, f"{view_name}.npz")
            np.savez_compressed(depth_path, data=depth)
            #plotter.render()
            rgb = plotter.image 
            print(rgb.shape)
            print(rgb.min(),' ',rgb.max())
            plt.imsave(rgb_path, rgb)
            # Save depth: output_dir/i/depth/{view_name}.npz
            
            
            

            # Save RGB: output_dir/i/rgb/{view_name}.png
            
            
            

            plotter.deep_clean()
            plotter.close()
            del plotter,obj
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

TIMEOUT_SECONDS = 200  # 单个渲染进程最大允许时间
MAX_ATTEMPTS = 2

def run_render_tasks(tasks, num_processes, core_list, attempt):
    if not tasks:
        return []
    desc = f"Rendering (attempt {attempt})"
    active = []
    failed = []
    completed = 0
    total = len(tasks)
    pbar = tqdm(total=total, dynamic_ncols=True, desc=desc)

    def reap():
        nonlocal active, completed, failed
        now = time.time()
        still = []
        delta = 0
        for proc, start_t, task in active:
            proc.join(timeout=0)
            if proc.is_alive():
                if now - start_t > TIMEOUT_SECONDS:
                    print(f"[Timeout] Skip {task[0]}")
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    proc.join()
                    try:
                        out_dir = task[1]
                        if out_dir and os.path.isdir(out_dir):
                            shutil.rmtree(out_dir, ignore_errors=True)
                    except Exception as _ce:
                        print(f"Cleanup failed for {task[1]}: {_ce}")
                    failed.append(task)
                    completed += 1
                    delta += 1
                else:
                    still.append((proc, start_t, task))
            else:
                if proc.exitcode not in (0, None):
                    failed.append(task)
                completed += 1
                delta += 1
        if delta:
            pbar.update(delta)
        active = still

    for idx, task in enumerate(tasks):
        while len(active) >= num_processes:
            reap()
            time.sleep(0.05)
        core = core_list[idx % len(core_list)] if core_list else None
        p = multiprocessing.Process(target=worker_entry, args=(*task, core))
        p.start()
        active.append((p, time.time(), task))
        if (idx + 1) % 20 == 0:
            reap()

    while active:
        reap()
        time.sleep(0.1)

    pbar.close()
    return failed

def worker_entry(obj_path, output_dir, num_views, core=None):
    # 可选绑定 CPU
    if core is not None:
        try:
            os.sched_setaffinity(0, {core})
        except Exception:
            try:
                import psutil
                psutil.Process().cpu_affinity([core])
            except Exception:
                pass
    process_model((obj_path, output_dir, num_views))

if __name__ == "__main__":
    from xvfbwrapper import Xvfb
    vdisplay = Xvfb(width=1280, height=640)
    vdisplay.start()
    # Load JSON list of file paths
    with open('/home/linzhuohang/3DGen/configs/rgb_multiview.json', 'r') as f:
        obj_paths = json.load(f)

    # Prepare arguments for multiprocessing
    num_views = 1
    tasks = []
    for i,obj_path in enumerate(obj_paths):
        info_id = os.path.splitext(os.path.basename(obj_path))[0]
        obj_path = obj_path.replace('/mnt/hdd1/caixiao/data/pv_views/','/mnt/hdd1/caixiao/data/objaverse_1.0/hf-objaverse-v1/glbs/')
        obj_path += '.glb'
        #if info_id !='0d94fa80e87e49e2b0747d1252b9e3bd':continue
        output_dir = os.path.join('/mnt/hdd3/linzhuohang/3DGen/rgb_multiview', info_id)
        #obj_path = obj_path.replace('/mnt/hdd1/caixiao/data/pv_views/','/mnt/hdd1/caixiao/data/objaverse_1.0/hf-objaverse-v1/glbs/')
        #obj_path += '.glb'
        if not os.path.exists(obj_path):
            print(f"File not found: {obj_path}")
            continue
        #os.makedirs(output_dir, exist_ok=True)
        tasks.append((obj_path, output_dir, num_views))

    # 替换 Pool 为手工进程管理以支持严格超时
    num_processes = min(50, len(tasks))
    try:
        available_cores = sorted(os.sched_getaffinity(0))
    except AttributeError:
        available_cores = list(range(os.cpu_count() or 1))
    core_list = list(available_cores)[:num_processes]

    remaining = tasks
    attempt = 1
    final_failures = []
    while attempt <= MAX_ATTEMPTS and remaining:
        failed = run_render_tasks(remaining, num_processes, core_list, attempt)
        if not failed:
            break
        final_failures = failed
        remaining = failed
        attempt += 1

    completed = len(tasks) - len(final_failures)
    print(f"All done. Finished {completed} / {len(tasks)}")
    if final_failures:
        print("Failed after retries:")
        for obj_path, out_dir, _ in final_failures:
            print(f" - {obj_path}")
    vdisplay.stop()

