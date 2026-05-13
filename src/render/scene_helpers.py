from pathlib import Path

import bpy
from mathutils import Vector
from src.utils import paths
from src.utils.exceptions import ModelException

TARGET_SCALE = 2.0

def normalize_model() -> None:
    """
    Function to normalize the model to a target scale
    :return: None
    """
    meshes = [obj for obj in bpy.context.scene.objects if _is_real_mesh(obj)]
    all_corners = []

    for mesh in meshes:
        for corner in mesh.bound_box:
            world_corner = mesh.matrix_world @ Vector(corner)
            all_corners.append(world_corner)

    bbox_min, bbox_max = calculate_bboxes(all_corners)
    size = max(bbox_max - bbox_min)
    scale_factor = TARGET_SCALE / size

    roots = [obj for obj in bpy.context.scene.objects if obj.parent is None]
    for root in roots:
        root.scale.x *= scale_factor
        root.scale.y *= scale_factor
        root.scale.z *= scale_factor

    bpy.context.view_layer.update()

    all_corners = [m.matrix_world @ Vector(c) for m in meshes for c in m.bound_box]
    bbox_min, bbox_max = calculate_bboxes(all_corners)
    center = (bbox_min + bbox_max) / 2

    roots = [obj for obj in bpy.context.scene.objects if obj.parent is None]
    for root in roots:
        root.location -= center

    bpy.context.view_layer.update()


def calculate_bboxes(all_corners: list) -> tuple[Vector, Vector]:
    """
    Function to calculate the bounding boxes
    :param all_corners: Corners of the bounding boxes
    :return: The min and max bounds of the box
    """
    bbox_min = Vector((min(corner.x for corner in all_corners), min(corner.y for corner in all_corners), min(corner.z for corner in all_corners)))
    bbox_max = Vector((max(corner.x for corner in all_corners), max(corner.y for corner in all_corners), max(corner.z for corner in all_corners)))
    return bbox_min, bbox_max


def _is_real_mesh(obj) -> bool:
    """
    Checks if an object is a real mesh, and if it is checks if it is "not_expored"
    :param obj: The given object
    :return: bool
    """
    if obj.type != 'MESH':
        return False
    return not any("not_exported" in c.name.lower() for c in obj.users_collection)


def setup_scene_objects() -> tuple[bpy.types.Object, bpy.types.Object]:
    """
    Function to set up scene objects
    :return: tuple of scene objects [light obj]
    """
    light_data = bpy.data.lights.new(name='KEY', type='SUN')
    light_obj = bpy.data.objects.new(name='KEY', object_data=light_data)

    camera_data = bpy.data.cameras.new(name='CAMERA')
    camera_obj = bpy.data.objects.new(name='CAMERA', object_data=camera_data)

    return light_obj, camera_obj

def scene_setup() -> None:
    """
    Setup default scene settings
    :return: None
    """
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_x = 256
    bpy.context.scene.render.resolution_y = 256


def clear_scene() -> None:
    """
    Clear out the scene
    :return: None
    """
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def import_model(glb_path: Path) -> None:
    """Import a single GLB file into the current scene."""
    try:
        bpy.ops.import_scene.gltf(filepath=str(glb_path), loglevel=20)
    except Exception as e:
        with open(paths.DATA_DIR / "render_failure.log", "a") as f:
            f.write(f"{glb_path.name}: {type(e).__name__}: {e}\n")
        raise ModelException(f"{glb_path}: {type(e).__name__}: {e}")