import os
from pathlib import Path

MODEL_ARCHITECTURE = "srresnet"

BASE_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
