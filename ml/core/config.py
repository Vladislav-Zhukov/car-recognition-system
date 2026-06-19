from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parents[2]


MODEL_NAME = "EfficientNet-B0"
MODEL_VERSION = "1.0.0"

DATASET_DIR = BASE_DIR / "dataset" / "car_data_subset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "test"

MODEL_DIR = BASE_DIR / "trained_models"
MODEL_PATH = MODEL_DIR / "car_model.pt"

FAIR_EPOCHS = 100

FAIR_EFFICIENTNET_MODEL_PATH = MODEL_DIR / "fair_efficientnet_100.pt"
FAIR_CUSTOM_CNN_MODEL_PATH = MODEL_DIR / "fair_custom_cnn_100.pt"

REPORTS_DIR = BASE_DIR / "reports"
UPLOAD_DIR = BASE_DIR / "uploads"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
