"""
Configuration file for Waste Classification & Adversarial Robustness project.
All hyperparameters and constants are defined here.
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "trashnet")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
SAMPLE_PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "sample_predictions")

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(SAMPLE_PREDICTIONS_DIR, exist_ok=True)

# Dataset configuration
DATASET_PATH = os.path.join(DATA_DIR, "Garbage classification", "Garbage classification")
NUM_CLASSES = 6
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Train/Val/Test splits
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Image preprocessing
IMAGE_SIZE = 224
RESIZE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 4

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Training hyperparameters
EPOCHS_CLEAN = 40
EPOCHS_ADV = 50
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
SCHEDULER_TYPE = "cosine"  # "cosine" or "step"
STEP_SIZE = 10
GAMMA = 0.1

# Adversarial attack parameters (FIXED - no user input)
EPS_FGSM = 8.0 / 255.0
EPS_PGD = 8.0 / 255.0
PGD_ALPHA = 2.0 / 255.0
PGD_STEPS_TRAIN = 7
PGD_STEPS_EVAL = 20
RANDOM_START = True

# Adversarial training
ADV_TRAIN_MIX_RATIO = 0.5  # 0.5 * clean_loss + 0.5 * adv_loss

# Model architecture
MODEL_NAME = "resnet50"  # "resnet50" or "resnet18"
PRETRAINED = True

# Evaluation
ACCURACY_VS_EPS_EPSILONS = [0, 2/255, 4/255, 6/255, 8/255, 10/255, 12/255]

# Random seed for reproducibility
SEED = 42

# Device
DEVICE = "cuda"  # Will be set to "cpu" if CUDA not available

# Model file names
CLEAN_MODEL_PATH = os.path.join(MODELS_DIR, "resnet_trashnet_clean.pth")
ADV_MODEL_PATH = os.path.join(MODELS_DIR, "resnet_trashnet_adv.pth")

# Metrics file
METRICS_FILE = os.path.join(OUTPUTS_DIR, "metrics.json")

