from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models" / "pokemon-3d-api"
GLB_REGULAR = MODELS_DIR / "models" / "opt" / "regular"
GLB_ALOLAN = MODELS_DIR / "models" / "opt" / "alolan"
GLB_GALARIAN = MODELS_DIR / "models" / "opt" / "galarian"
GLB_HISUIAN = MODELS_DIR / "models" / "opt" / "hisuian"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
BACKGROUNDS_DIR = DATA_DIR / "backgrounds"
REAL_DIR = DATA_DIR / "real"

CONFIGS_DIR = PROJECT_ROOT / "configs"
RENDER_CONFIG = CONFIGS_DIR / "render_config.yaml"
POKEDEX_YAML  = CONFIGS_DIR / "gen1_pokedex.yaml"

TRAINING_DATA = DATA_DIR / "training_data"

VARIANT_DIRS = [GLB_REGULAR, GLB_ALOLAN, GLB_GALARIAN, GLB_HISUIAN]