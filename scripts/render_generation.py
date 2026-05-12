import bpy
import sys
from tqdm import tqdm

sys.path.insert(0, "/Users/josphslater/Programming/Personal/Gen-1-Pokemon-Detector-And-Classifier")

from src.utils import paths
from src.render.scene_helpers import normalize_model, setup_scene_objects, scene_setup, clear_scene, import_model
from src.render.generation_modifications import random_lighting, adjust_camera
from src.utils.exceptions import ModelException


def render_dataset(samples_per_pokemon: int=10) -> None:
    """
    Render the snapshots of all original 151 pokemon
    :param samples_per_pokemon: The number of samples to render per pokemon
    :return: None
    """
    scene_setup()

    for pokedex in tqdm(range(1, 152), desc="Pokemon"):
        clear_scene()

        try:
            import_model(pokedex)
        except ModelException:
            continue
        render(pokedex, samples_per_pokemon)



def render_individuals(pokedex_entry: int, samples_per_pokemon: int=10) -> None:
    """
    Render the snapshots of a specific pokemon
    :param pokedex_entry: The pokemon index
    :param samples_per_pokemon: The number of samples to render
    :return: None
    """
    scene_setup()
    clear_scene()

    try:
        import_model(pokedex_entry)
    except ModelException:
        return

    render(pokedex_entry, samples_per_pokemon)


def render(pokedex: int, samples_per_pokemon: int=10) -> None:
    """
    Render the snapshots of a pokemon
    :param pokedex: The pokemon index
    :param samples_per_pokemon: The number of samples to render
    :return: None
    """
    # Set up the scene
    normalize_model()
    light, camera = setup_scene_objects()

    bpy.context.scene.collection.objects.link(light)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    for i in range(samples_per_pokemon):
        # Adjust Settings
        random_lighting(light)
        adjust_camera(camera)
        bpy.context.view_layer.update()

        output_path = paths.SYNTHETIC_DIR / f"{pokedex:03d}" / f"sample_{i:04d}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    render_dataset(samples_per_pokemon=200)