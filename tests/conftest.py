from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tiny_image_folder(tmp_path: Path) -> Path:
    """
    Build a minimal ImageFolder-compatible directory in a temp dir:

        tmp_path/
          001/  sample_0.png ... sample_5.png
          002/  sample_0.png ... sample_5.png
          ...

    8 classes x 6 images each = 48 images total. Enough to satisfy stratified
    splitting (every class needs >=2 samples) without bloat.
    """
    num_classes = 8
    images_per_class = 6
    image_size = 32

    for class_idx in range(1, num_classes + 1):
        class_dir = tmp_path / f"{class_idx:03d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for img_idx in range(images_per_class):
            arr = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(class_dir / f"sample_{img_idx}.png")

    return tmp_path
