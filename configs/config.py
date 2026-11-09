"""
Global Configuration for Open-Set Recognition Using Underwater SONAR Images.
Supports both local environment and Kaggle runtime seamlessly.
"""

import os
from pathlib import Path
try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    DEVICE = "cpu (torch not installed in this environment)"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths: check local path first, fallback to Kaggle path
LOCAL_WATERTANK = PROJECT_ROOT / "md_fls_dataset" / "data" / "watertank-cropped"
LOCAL_TURNTABLE = PROJECT_ROOT / "md_fls_dataset" / "data" / "turntable-cropped"

KAGGLE_WATERTANK = Path(
    "/kaggle/input/datasets/mariyaabdul/marine-debris-fls/marine-debris-fls-datasets/md_fls_dataset/data/watertank-cropped"
)
KAGGLE_TURNTABLE = Path(
    "/kaggle/input/datasets/mariyaabdul/marine-debris-fls/marine-debris-fls-datasets/md_fls_dataset/data/turntable-cropped"
)

if LOCAL_WATERTANK.exists():
    WATERTANK_DIR = LOCAL_WATERTANK
    TURNTABLE_DIR = LOCAL_TURNTABLE
elif KAGGLE_WATERTANK.exists():
    WATERTANK_DIR = KAGGLE_WATERTANK
    TURNTABLE_DIR = KAGGLE_TURNTABLE
else:
    # Default fallback
    WATERTANK_DIR = LOCAL_WATERTANK
    TURNTABLE_DIR = LOCAL_TURNTABLE

# ── Output Directories ───────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

for d in [OUTPUT_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Class Partitions (Disjoint Open-Set Split) ───────────────────────────────
# Default benchmark split (as evaluated in notebook)
KNOWN_CLASSES = [
    "bottle",
    "can",
    "tire",
    "drink-carton",
    "hook",
    "propeller",
]

UNKNOWN_CLASSES = [
    "chain",
    "valve",
    "shampoo-bottle",
    "standing-bottle",
]

CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(KNOWN_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for cls_name, idx in CLASS_TO_IDX.items()}
NUM_CLASSES = len(KNOWN_CLASSES)

# Label for unknown classes in test loaders
UNKNOWN_LABEL = -1

# ── Preprocessing & Augmentation Settings ────────────────────────────────────
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
MEDIAN_BLUR_KSIZE = 3

# ── Training Hyperparameters ────────────────────────────────────────────────
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0  # 0 for safe multi-platform compatibility (Windows & Kaggle)

# Baseline ResNet-50
BASELINE_EPOCHS = 25
BASELINE_LR = 1e-4
BASELINE_PATIENCE = 5
STEP_SIZE = 7
GAMMA = 0.5

# Deep Metric Learning
METRIC_EPOCHS = 30
METRIC_LR = 1e-4
EMBEDDING_DIM = 128
TRIPLET_MARGIN = 0.5

# ── Device Configuration ────────────────────────────────────────────────────
# (Configured at top of file with fallback)

