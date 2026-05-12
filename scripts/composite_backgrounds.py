import PIL
from PIL import Image
from src.utils import paths
import random
from tqdm import tqdm

fg_models = paths.SYNTHETIC_DIR
bg_images = paths.BACKGROUNDS_DIR

def composite_backgrounds():
    bg_list = [bg for ext in ("*.jpg", "*.jpeg", "*.png") for bg in bg_images.rglob(ext)]

    for directory in tqdm(list(fg_models.iterdir()), desc="Pokemon"):
        if not directory.is_dir():
            continue

        for file in directory.iterdir():
            output_path = paths.TRAINING_DATA / f"{directory.name}" / f'{file.name}'
            if output_path.exists():
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)

            model = PIL.Image.open(file).convert("RGBA")
            bg = PIL.Image.open(random.choice(bg_list)).convert("RGB")

            crop_frame = min(bg.width, bg.height)
            left = random.randint(0, bg.width - crop_frame)
            top = random.randint(0, bg.height - crop_frame)

            bg = bg.crop((left, top, left + crop_frame, top + crop_frame))
            bg = bg.resize(model.size).convert("RGBA")

            result = Image.alpha_composite(bg, model)

            result.convert("RGB").save(output_path, "JPEG")



if __name__ == "__main__":
    composite_backgrounds()