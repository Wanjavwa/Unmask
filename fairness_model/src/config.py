"""
Configuration for the fairness model training and inference pipeline.
All paths, hyperparameters, and toggles are centralized here.
"""

import os

# ---------------------------------------------------------------------------
# Paths (relative to fairness_model/ or project root)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DATA_DIR = os.path.join(BASE_DIR, "data", "real")
FAKE_DATA_DIR = os.path.join(BASE_DIR, "data", "fake")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "fairness_head.pt")
MODEL_BEST_PATH = os.path.join(MODEL_SAVE_DIR, "fairness_head_best.pt")
TRAINING_LOG_PATH = os.path.join(MODEL_SAVE_DIR, "training_log.txt")

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
# ImageNet normalization (used by pretrained ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Optional face detection & cropping (facenet-pytorch MTCNN)
USE_FACE_CROP = True  # Set to False to use center crop only

# ---------------------------------------------------------------------------
# Augmentations (training only)
# ---------------------------------------------------------------------------
AUGMENT_HORIZONTAL_FLIP = True
AUGMENT_COLOR_JITTER = True   # Slight brightness/contrast/saturation
AUGMENT_RANDOM_ROTATION = True  # Small rotation in degrees
ROTATION_DEGREES = 15
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.2

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% val
RANDOM_SEED = 42

# Scheduler: "step" or "plateau"
LR_SCHEDULER = "plateau"
# For StepLR
STEPLR_STEP_SIZE = 3
STEPLR_GAMMA = 0.1
# For ReduceLROnPlateau
PLATEAU_PATIENCE = 2
PLATEAU_FACTOR = 0.5

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
# Set to "cuda" or "cpu"; None = auto-detect
DEVICE = None  # Will be set to cuda if available, else cpu

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
# real = 0, fake = 1 (consistent with CrossEntropyLoss class indices)
LABEL_REAL = 0
LABEL_FAKE = 1
NUM_CLASSES = 2
