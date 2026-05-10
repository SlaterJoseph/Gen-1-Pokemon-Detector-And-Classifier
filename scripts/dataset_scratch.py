import bpy
import sys
from mathutils import Vector
import datetime

sys.path.insert(0, "/Users/josphslater/Programming/Personal/Gen-1-Pokemon-Detector-And-Classifier")

from src.utils import paths
from src.render.scene_helpers import normalize_model, setup_scene_objects
from src.render.generation_modifications import random_lighting, adjust_camera



def render_dataset():
    for i in range(1, 151):
        pass



def render_one():
    """
    Function to render a single image hard coded to understand bpy
    :return: None
    """
    pokedex = 1
    current_glb = f'{pokedex}.glb'

    # Clear the scene
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

    # Import Pokemon
    bulbasaur = paths.GLB_DIR / current_glb
    bpy.ops.import_scene.gltf(filepath= str(bulbasaur), directory= str(paths.GLB_DIR), files= [{"name":current_glb, "name":current_glb}], loglevel=20)

    # Find the armature
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    current_armature = armatures[0]

    # Select the armature
    bpy.ops.object.select_all(action="DESELECT")
    current_armature.select_set(True)
    bpy.context.view_layer.objects.active = current_armature

    # Set up the scene
    normalize_model()
    light, camera = setup_scene_objects()

    bpy.context.scene.collection.objects.link(light)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    # Adjust
    random_lighting(light)
    adjust_camera(camera)
    bpy.context.view_layer.update()


    # === DIAGNOSTIC ===
    # Recompute bbox after normalize and list parentless objects.
    # Appends to data/normalize_diagnostic.log so you can compare across runs.
    all_c = [m.matrix_world @ Vector(c)
             for m in bpy.context.scene.objects if m.type == "MESH"
             for c in m.bound_box]
    mn = Vector((min(c.x for c in all_c), min(c.y for c in all_c), min(c.z for c in all_c)))
    mx = Vector((max(c.x for c in all_c), max(c.y for c in all_c), max(c.z for c in all_c)))
    center = (mn + mx) / 2
    size = max(mx - mn)

    log_path = paths.DATA_DIR / "normalize_diagnostic.log"
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"\n=== Pokemon #{pokedex} ({datetime.datetime.now():%H:%M:%S}) ===\n")
        f.write(f"AFTER normalize:\n")
        f.write(f"  size:     {size:.4f}\n")
        f.write(f"  center:   ({center.x:.3f}, {center.y:.3f}, {center.z:.3f})\n")
        f.write(f"  bbox_min: ({mn.x:.3f}, {mn.y:.3f}, {mn.z:.3f})\n")
        f.write(f"  bbox_max: ({mx.x:.3f}, {mx.y:.3f}, {mx.z:.3f})\n")
        f.write(f"Parentless objects:\n")
        for o in bpy.context.scene.objects:
            if o.parent is None:
                f.write(f"  {o.name:35s} type={o.type:10s} "
                        f"scale={tuple(round(s, 3) for s in o.scale)} "
                        f"loc={tuple(round(l, 3) for l in o.location)}\n")

    output_path = paths.DATA_DIR / "test_renders" / f"{pokedex}_test.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.context.scene.render.filepath = str(output_path)
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_x = 256
    bpy.context.scene.render.resolution_y = 256

    bpy.ops.render.render(write_still=True)
    print(f"Rendered to: {output_path}")

if __name__ == "__main__":
    render_one()
