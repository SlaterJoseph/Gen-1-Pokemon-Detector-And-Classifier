import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.classifier.train import train
from src.utils import paths

if __name__ == "__main__":
    with open(paths.CONFIGS_DIR / "train_config.yaml") as f:
        config = yaml.safe_load(f)
    train(config)