import re
from pathlib import Path
import bpy
import sys
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import paths
from src.render.scene_helpers import normalize_model, setup_scene_objects, scene_setup, clear_scene, import_model
from src.render.generation_modifications import random_lighting, adjust_camera
from src.utils.misc import NON_GEN_1_REFERENCE
from src.utils.exceptions import ModelException

SAMPLE_PATTERN = re.compile(r"sample_(\d+)\.png")


def render_dataset(samples_per_pokemon: int=10, start: int=1, end: int=152) -> None:
    """
    Render the snapshots of pokemon in the inclusive-exclusive [start, end) range.
    Defaults cover the full gen-1 dex (1..151).
    :param samples_per_pokemon: The number of samples to render per pokemon
    :param start: First Pokédex number to render (inclusive)
    :param end: One past the last Pokédex number to render (exclusive)
    :return: None
    """
    scene_setup()

    for pokedex in tqdm(range(start, end), desc="Pokemon"):
        current_glb = f"{pokedex}.glb"
        glbs = [
            variant_dir / current_glb
            for variant_dir in paths.VARIANT_DIRS
            if (variant_dir / current_glb).exists()
        ]

        if not glbs:
            continue

        samples_per_species = samples_per_pokemon // len(glbs)

        for glb in glbs:
            clear_scene()
            try:
                import_model(glb)
            except ModelException:
                continue
            render(pokedex, None, samples_per_species)



def render_individuals(pokedex_entry: int, folder: int=None, samples_per_pokemon: int=10) -> None:
    """
    Render the snapshots of a specific pokemon
    :param pokedex_entry: The pokemon index
    :param samples_per_pokemon: The number of samples to render
    :param folder: The folder to render to (default to 152 for not gen 1)
    :return: None
    """
    scene_setup()
    current_glb = f"{pokedex_entry}.glb"
    glbs = [
        variant_dir / current_glb
        for variant_dir in paths.VARIANT_DIRS
        if (variant_dir / current_glb).exists()
    ]

    if not glbs:
        return

    samples_per_species = samples_per_pokemon // len(glbs)

    for glb in glbs:
        clear_scene()
        try:
            import_model(glb)
        except ModelException:
            continue
        render(pokedex_entry, folder, samples_per_species)


def render_specifics(entries: list, folder: int = None, samples_per_pokemon:int = 10) -> None:
    """
    A function that generates renders of a list of specified pokemon
    :param entries: The list of pokemon to render
    :param folder: The folder to render to
    :param samples_per_pokemon: The number of renders to generate
    :return: None
    """
    for entry in entries:
        render_individuals(entry, folder, samples_per_pokemon)


def next_sample_index(output_dir: Path) -> int:
    """
    Continue the render from the next number
    """
    if not output_dir.exists():
        return 0

    indices = []
    for path in output_dir.glob("sample_*.png"):
        match = SAMPLE_PATTERN.match(path.name)
        if match:
            indices.append(int(match.group(1)))

    return max(indices) + 1 if indices else 0


def render(pokedex: int, folder: int = None, samples_per_pokemon: int=10) -> None:
    """
    Render the snapshots of a pokemon
    :param pokedex: The pokemon index
    :param folder: The folder to render to
    :param samples_per_pokemon: The number of samples to render
    :return: None
    """
    # Set up the scene
    normalize_model()
    light, camera = setup_scene_objects()

    bpy.context.scene.collection.objects.link(light)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    if folder is not None:
        output_dir = paths.SYNTHETIC_DIR / f"{folder:03d}"
    else:
        output_dir = paths.SYNTHETIC_DIR / f"{pokedex:03d}"

    output_dir.mkdir(parents=True, exist_ok=True)
    start_idx = next_sample_index(output_dir)

    for i in range(samples_per_pokemon):
        # Adjust Settings
        random_lighting(light)
        adjust_camera(camera)
        bpy.context.view_layer.update()

        sample_idx = start_idx + i
        output_path = output_dir / f"sample_{sample_idx:04d}.png"
        bpy.context.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    render_dataset(300)
    render_specifics(NON_GEN_1_REFERENCE, 152, 10)