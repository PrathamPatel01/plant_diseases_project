from pathlib import Path

# Project root = folder that contains "src/"
ROOT_DIR = Path(__file__).resolve().parents[1]

# Dataset directory (we expect class folders directly inside this)
DATASET_DIR = ROOT_DIR / "data_raw" / "PlantVillage"

# Preprocessing
IMAGE_SIZE = (128, 128)  # (width, height)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Splitss
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
