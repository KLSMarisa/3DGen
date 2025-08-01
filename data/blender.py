"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import math
import os
import random
import sys
import time
import urllib.request
import numpy as np
from typing import Tuple

import bpy
from mathutils import Matrix, Vector

parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--object_path",
#     type=str,
#     required=True,
#     help="Path to the object file",
# )
parser.add_argument("--output_dir", type=str, default="/mnt/nfs/caixiao/data/blender_views")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--num_images", type=int, default=16)
parser.add_argument("--camera_dist", type=int, default=1.5)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGB"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 16
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def add_lighting() -> None:
    # delete the default light
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    # print(bpy.data.lights[1])
    light2 = bpy.data.lights["Area"]
    light2.energy = 30000
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    # object_path=object_path.split(":")[1]
    # print(object_path)
    if object_path.endswith(".glb"):
        # print(object_path)
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
        # 获取刚刚导入的物体
        # imported_object = bpy.context.selected_objects[0] 
        # print(imported_object)
        # # 指定物体的位置
        # imported_object.location = (64.64999389648438, 71.64023150918175, -20.04620444040053)
        # exit()
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint


def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()
    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    # print(object_uid)
    # exit()
    normalize_scene()
    add_lighting()
    cam, cam_constraint = setup_camera()
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    for i in range(args.num_images):
        # set the camera position
        theta = (i / (args.num_images)) * math.pi * 2
        phi = math.radians(90)
        # phi = math.radians((-30 + 60 * (i * args.num_images)))
        point = (
            args.camera_dist * math.sin(phi) * math.cos(theta),
            args.camera_dist * math.sin(phi) * math.sin(theta),
            args.camera_dist * math.cos(phi),
        )
        cam.location = point
        # cam.location = (747.34772958076, 71.64023150918175, -20.04620444040036)
        # direction = -cam.location
        # rot_quat = direction.to_track_quat("-Z", "Y")
        # cam.rotation_euler = rot_quat.to_euler()

        # render the image
        # render_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.png")
        # scene.render.filepath = render_path
        # bpy.ops.render.render(write_still=True)

        scene.view_layers['ViewLayer'].use_pass_z = True
        # context.view_layer.use_pass_z = True
        render.use_compositing = True
        scene.use_nodes = True
        # context.use_nodes = True
        #

        tree = scene.node_tree
        links = tree.links

        # # Clear default nodes
        for n in tree.nodes:
            # print(n)

            tree.nodes.remove(n)
        # exit()

        base_path = os.path.join(args.output_dir, object_uid)
        render_path = os.path.join(base_path, f"{i:03d}.png")
        dep_path = os.path.join(base_path, f"depth_{i:03d}")
        # Create input render layer node.
        # bpy.ops.node.new_node_tree(type='CompositorNodeTree', name='NodeTree')
        render_layers = tree.nodes.new('CompositorNodeRLayers')
        # render_layers = tree.nodes.get('Render Layers')
        # compositor = tree.nodes.get('Composite')
        #
        normalize_node = tree.nodes.new(type='CompositorNodeNormalize')
        link = links.new(render_layers.outputs['Depth'], normalize_node.inputs[0])

        # image_out = bpy.ops.node.add_node(type="CompositorNodeViewer")
        image_out = tree.nodes.new(type="CompositorNodeOutputFile")
        # image_out.use_alpha = True
        image_out.base_path = dep_path
        link = links.new(normalize_node.outputs[0], image_out.inputs[0])
        # for n in tree.nodes:
        #     print(tree)
        # exit()
        # depth_out = tree.nodes.new(type="CompositorNodeViewer")
        # depth_out.label = 'Depth Output'

        # links.new(normalize_node.outputs[0], image_out.inputs[0])
        # links.new(render_layers.outputs['Depth'], image_out.inputs[0])

        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)

        rt_matrix = get_3x4_RT_matrix_from_blender(cam)
        rt_matrix_path = os.path.join(base_path, f"{i:03d}.npy")
        np.save(rt_matrix_path, rt_matrix)
        # exit()
        # depth = np.array(normalize_node.outputs[0])
        # np.save(f'/home/caixiao/projects/objaverse-xl/tests/depth_{i:03d}',depth)

        # pixels = np.array(bpy.data.images['Render Result'].pixels)
        # print(bpy.data.images['Viewer Node'].pixels)
        # print(pixels.shape)
        # exit()
        # pixels = np.array(bpy.data.images['Viewer Node'].pixels)
        # resolution = 512
        # reshaping into image array 4 channel (rgbz)
        # img_path = os.path.join(output_dir, f"rgbd_{i:03d}")
        # image_with_depth = pixels.reshape(resolution, resolution, -1)
        # dep_path = os.path.join(base_path,  f"{i:03d}.npy")
        # np.save(dep_path, image_with_depth)
    # exit()

def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object):
    """Returns the 3x4 RT matrix from the given camera.

    Taken from Zero123, which in turn was taken from
    https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Matrix: The 3x4 RT matrix from the given camera.
    """
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    # print(R_world2bcam)
    # print(R_world2bcam.transposed())
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location
    print(T_world2bcam)
    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )
    return RT

def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    try:
        # start_i = time.time()
        # if args.object_path.startswith("http"):
        #     local_path = download_object(args.object_path)
        # else:
        #     local_path = args.object_path
        # img_infos = []
        # with open('/mnt/nfs/caixiao/datasets/objaverse/hf-objaverse-v1/downloaded.txt', 'r') as f:
        #     for line in f:
        #         # print(line.split('\n'))

        #         img_info = dict()
        #         img_info['filename'] = line.split('\n')[0]
        #         # print(img_info['filename'])
        #         # exit()
        #         img_infos.append(img_info)
        # img_infos = img_infos[0:10]
        # for info in img_infos:
        #     path = info['filename']
        # local_path = '/mnt/nfs/caixiao/datasets/objaverse/hf-objaverse-v1/glbs/000-080/912b0308a23d4aeeba71c2762a32dc4c.glb'
        local_path = '/home/caixiao/projects/3d_lib/obj/000a82b4e6bf4e909fbe5a3b0e6d67dc.glb'
        save_images(local_path)
        # end_i = time.time()
        # print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        # if args.object_path.startswith("http"):
        #     os.remove(local_path)
    except Exception as e:
        print("Failed to render", local_path)
        print(e)