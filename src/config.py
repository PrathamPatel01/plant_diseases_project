# from pathlib import Path

# # Project root = folder that contains "src/"
# ROOT_DIR = Path(__file__).resolve().parents[1]

# # Dataset directory (we expect class folders directly inside this)
# DATASET_DIR = ROOT_DIR / "data_raw" / "PlantVillage"

# # Preprocessing
# IMAGE_SIZE = (128, 128)  # (width, height)
# ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# # Splitss
# RANDOM_SEED = 42
# TRAIN_RATIO = 0.70
# VAL_RATIO = 0.15
# TEST_RATIO = 0.15
from pathlib import Path

# ============================================================
# Project Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT_DIR / "data_raw" / "PlantVillage"

MODEL_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"
LABELS_PATH = MODEL_DIR / "labels.json"
HISTORY_PATH = REPORTS_DIR / "history.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.json"


# ============================================================
# Image Settings
# ============================================================

# TensorFlow convention: (height, width)
IMAGE_SIZE = (224, 224)

CHANNELS = 3

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ============================================================
# Dataset Split Settings
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# Training Settings
# ============================================================

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-4


# ============================================================
# Model Settings
# ============================================================

MODEL_NAME = "MobileNetV2"
DROPOUT_RATE = 0.30


# ============================================================
# App Settings
# ============================================================

APP_TITLE = "Plant Disease Intelligence"
APP_SUBTITLE = "Enterprise AI dashboard for plant disease detection"