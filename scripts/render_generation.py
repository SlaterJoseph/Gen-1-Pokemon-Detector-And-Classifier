import bpy
import sys

sys.path.insert(0, "/Users/josphslater/Programming/Personal/Gen-1-Pokemon-Detector-And-Classifier")

from src.utils import paths
from src.render.scene_helpers import normalize_model, setup_scene_objects
from src.render.generation_modifications import random_lighting, adjust_camera



def render_dataset():
    for pokedex in range(1, 151):
        current_glb = f'{pokedex}.glb'

        # Clear the scene
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

        # Import Pokemon
        pokemon = paths.GLB_DIR / current_glb
        bpy.ops.import_scene.gltf(filepath=str(pokemon), directory=str(paths.GLB_DIR),
                                  files=[{"name": current_glb, "name": current_glb}], loglevel=20)

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

        for render in range(10):
            # Adjust
            random_lighting(light)
            adjust_camera(camera)
            bpy.context.view_layer.update()

            output_path = paths.DATA_DIR / "test_renders" / f"{pokedex}_{render}_test.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            bpy.context.scene.render.filepath = str(output_path)
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            bpy.context.scene.render.image_settings.color_mode = 'RGBA'
            bpy.context.scene.render.film_transparent = True
            bpy.context.scene.render.resolution_x = 256
            bpy.context.scene.render.resolution_y = 256

            bpy.ops.render.render(write_still=True)