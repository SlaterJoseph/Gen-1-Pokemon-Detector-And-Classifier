from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models" / "pokemon-3d-api"
GLB_DIR = MODELS_DIR / "models" / "opt" / "regular"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
BACKGROUNDS_DIR = DATA_DIR / "backgrounds"
REAL_DIR = DATA_DIR / "real"

CONFIGS_DIR = PROJECT_ROOT / "configs"
RENDER_CONFIG = CONFIGS_DIR / "render_config.yaml"
POKEDEX_YAML  = CONFIGS_DIR / "gen1_pokedex.yaml"

TRAINING_DATA = DATA_DIR / "training_data"