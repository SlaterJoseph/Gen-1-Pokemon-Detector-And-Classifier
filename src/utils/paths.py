import os
from pathlib import Path

def _detect_env():
    if Path("/kaggle/input").exists():
        return "kaggle"
    if "COLAB_GPU" in os.environ:
        return "colab"
    return "local"

ENV = _detect_env()

if ENV == "kaggle":
    PROJECT_ROOT = Path("/kaggle/working")
    TRAINING_DATA = Path("/kaggle/input/pokemon-gen1-synthetic/training_data")
    CONFIGS_DIR = PROJECT_ROOT / "Gen-1-Pokemon-Detector-And-Classifier" / "configs"
elif ENV == "colab":
    PROJECT_ROOT = Path("/content/Gen-1-Pokemon-Detector-And-Classifier")
    TRAINING_DATA = Path("/content/training_data")
    CONFIGS_DIR = PROJECT_ROOT / "configs"
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    TRAINING_DATA = PROJECT_ROOT / "data" / "training_data"
    CONFIGS_DIR = PROJECT_ROOT / "configs"


DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models" / "pokemon-3d-api"
GLB_REGULAR = MODELS_DIR / "models" / "opt" / "regular"
GLB_ALOLAN = MODELS_DIR / "models" / "opt" / "alolan"
GLB_GALARIAN = MODELS_DIR / "models" / "opt" / "galarian"
GLB_HISUIAN = MODELS_DIR / "models" / "opt" / "hisuian"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
BACKGROUNDS_DIR = DATA_DIR / "backgrounds"
REAL_DIR = DATA_DIR / "real"
RENDER_CONFIG = CONFIGS_DIR / "render_config.yaml"
POKEDEX_YAML  = CONFIGS_DIR / "gen1_pokedex.yaml"

VARIANT_DIRS = [GLB_REGULAR, GLB_ALOLAN, GLB_GALARIAN, GLB_HISUIAN]