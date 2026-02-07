"""
Dataset loader for real (UTKFace Black) and fake (AI-generated Black) images.
Validates UTKFace filenames and skips corrupted or badly formatted files.
"""

import os
import re
import random
from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image

from .config import (
    REAL_DATA_DIR,
    FAKE_DATA_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    USE_FACE_CROP,
    AUGMENT_HORIZONTAL_FLIP,
    AUGMENT_COLOR_JITTER,
    AUGMENT_RANDOM_ROTATION,
    ROTATION_DEGREES,
    COLOR_JITTER_BRIGHTNESS,
    COLOR_JITTER_CONTRAST,
    COLOR_JITTER_SATURATION,
    LABEL_REAL,
    LABEL_FAKE,
)
from .utils import load_image_rgb, get_face_cropper, align_face_or_center_crop, normalize_tensor


# UTKFace filename format: age_gender_race_date.jpg 
# We only want race segment == "1" (Black). So we require _1_ as the race segment.
UTKFACE_PATTERN = re.compile(r"^(\d+)_(\d+)_1_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def is_valid_utkface_filename(filename):
    """
    Verify that a filename matches UTKFace format and has race segment _1_ (Black).
    Returns True only for valid, well-formatted filenames.
    """
    if not filename or not isinstance(filename, str):
        return False
    name = filename.strip()
    if not name:
        return False
    return UTKFACE_PATTERN.match(name) is not None


def collect_real_paths(real_dir):
    """
    Scan real_dir for UTKFace images. Only include files with _1_ as race segment.
    Skips corrupted or badly formatted filenames.
    Returns list of (absolute_path, label=LABEL_REAL).
    """
    real_dir = Path(real_dir)
    if not real_dir.is_dir():
        return []
    paths = []
    for f in real_dir.iterdir():
        if not f.is_file():
            continue
        name = f.name
        if not is_valid_utkface_filename(name):
            continue
        paths.append((str(f.resolve()), LABEL_REAL))
    return paths


def collect_fake_paths(fake_dir):
    """
    Scan fake_dir for images (jpg/png). All are treated as fake (label=1).
    Returns list of (absolute_path, label=LABEL_FAKE).
    """
    fake_dir = Path(fake_dir)
    if not fake_dir.is_dir():
        return []
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for f in fake_dir.glob(ext):
            if f.is_file():
                paths.append((str(f.resolve()), LABEL_FAKE))
    return paths


def verify_and_collect_dataset(real_dir=REAL_DATA_DIR, fake_dir=FAKE_DATA_DIR):
    """
    Verify UTKFace filenames and collect all valid real (Black) and fake paths.
    Skips corrupted or badly formatted UTKFace files.
    Returns (real_list, fake_list) where each list is [(path, label), ...].
    Also prints: number of real Black images, number of fake images, total dataset size
    """
    real_list = collect_real_paths(real_dir)
    fake_list = collect_fake_paths(fake_dir)
    n_real = len(real_list)
    n_fake = len(fake_list)
    total = n_real + n_fake
    print(f"Number of real Black images found: {n_real}")
    print(f"Number of fake images found: {n_fake}")
    print(f"Total dataset size: {total}")
    return real_list, fake_list


class FairnessDataset(Dataset):
    """
    Dataset of real (UTKFace Black) and fake (AI Black) images.
    Supports optional MTCNN face cropping and train-time augmentations.
    """

    def __init__(self, paths_and_labels, transform=None, use_face_crop=None, is_train=False):
        """
        paths_and_labels: list of (file_path, label).
        transform: optional torchvision transform (used when use_face_crop is False).
        use_face_crop: if True, use MTCNN for face crop; fallback to center crop.
        is_train: if True, apply optional augmentations after loading (flip, jitter, rotation).
        """
        self.samples = list(paths_and_labels)
        self.transform = transform
        if use_face_crop is None:
            use_face_crop = USE_FACE_CROP
        self.use_face_crop = use_face_crop
        self.mtcnn = get_face_cropper() if use_face_crop else None
        self.is_train = is_train
        # Augmentations applied on PIL (before tensor) when is_train and not using MTCNN path
        from torchvision import transforms as T
        self.augment = None
        if is_train and not use_face_crop and transform is None:
            augs = []
            if AUGMENT_HORIZONTAL_FLIP:
                augs.append(T.RandomHorizontalFlip())
            if AUGMENT_COLOR_JITTER:
                augs.append(T.ColorJitter(
                    brightness=COLOR_JITTER_BRIGHTNESS,
                    contrast=COLOR_JITTER_CONTRAST,
                    saturation=COLOR_JITTER_SATURATION,
                ))
            if AUGMENT_RANDOM_ROTATION:
                augs.append(T.RandomRotation(ROTATION_DEGREES))
            if augs:
                self.augment = T.Compose(augs)
        self.resize_crop = T.Resize(IMAGE_SIZE)
        self.center_crop = T.CenterCrop(IMAGE_SIZE)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = load_image_rgb(path)
        if img is None:
            # Corrupted image: return a black 224x224 image and label (avoid crash)
            img = Image.new("RGB", IMAGE_SIZE, (0, 0, 0))
        if self.use_face_crop and self.mtcnn is not None:
            try:
                tensor = self.mtcnn(img)
                if tensor is not None:
                    tensor = normalize_tensor(tensor)
                    if self.is_train and self.augment is None:
                        from torchvision import transforms as T
                        # Simple augment on tensor: flip only to avoid breaking alignment
                        if AUGMENT_HORIZONTAL_FLIP and random.random() > 0.5:
                            tensor = T.functional.hflip(tensor)
                    return tensor, label
            except Exception:
                pass
            # Fallback: resize and center crop
            img = self.resize_crop(img)
            img = self.center_crop(img)
            tensor = self.to_tensor(img)
        else:
            if self.augment is not None:
                img = self.augment(img)
            img = self.resize_crop(img)
            img = self.center_crop(img)
            tensor = self.to_tensor(img)
        tensor = normalize_tensor(tensor)
        return tensor, label


def build_dataset_splits(real_dir=REAL_DATA_DIR, fake_dir=FAKE_DATA_DIR, train_ratio=0.8, seed=42):
    """
    Collect real/fake paths, split into train/val, and return
    (train_dataset, val_dataset) using FairnessDataset.
    """
    real_list, fake_list = verify_and_collect_dataset(real_dir, fake_dir)
    all_samples = real_list + fake_list
    random.seed(seed)
    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * train_ratio)
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:]
    use_face = USE_FACE_CROP
    train_dataset = FairnessDataset(train_samples, transform=None, use_face_crop=use_face, is_train=True)
    val_dataset = FairnessDataset(val_samples, transform=None, use_face_crop=use_face, is_train=False)
    return train_dataset, val_dataset
