"""
Utility functions for preprocessing, augmentation, and device handling.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from .config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    AUGMENT_HORIZONTAL_FLIP,
    AUGMENT_COLOR_JITTER,
    AUGMENT_RANDOM_ROTATION,
    ROTATION_DEGREES,
    COLOR_JITTER_BRIGHTNESS,
    COLOR_JITTER_CONTRAST,
    COLOR_JITTER_SATURATION,
    USE_FACE_CROP,
    DEVICE,
)


def get_device():
    """
    Return the computation device (cuda if available, else cpu).
    Respects config.DEVICE if set to a string; otherwise auto-detects.
    """
    if DEVICE is not None:
        return torch.device(DEVICE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_rgb(path):
    """
    Load an image from path with PIL and convert to RGB.
    Returns PIL Image or None if loading fails.
    """
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        return img
    except Exception:
        return None


def get_inference_transform(use_face_crop=None):
    """
    Build the transform used at inference time (no augmentation).
    Optionally uses face crop if use_face_crop is True; otherwise center crop.
    If use_face_crop is None, uses config.USE_FACE_CROP.
    """
    if use_face_crop is None:
        use_face_crop = USE_FACE_CROP
    # Resize then center crop to 224x224 if no face crop (face crop is applied in dataset/infer)
    resize = transforms.Resize(IMAGE_SIZE)
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return transforms.Compose([
        resize,
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])


def get_train_transform(use_face_crop=None):
    """
    Build the transform for training (with optional augmentations).
    Same resize/crop logic as inference; augmentations applied before resize.
    """
    if use_face_crop is None:
        use_face_crop = USE_FACE_CROP
    augments = []
    if AUGMENT_HORIZONTAL_FLIP:
        augments.append(transforms.RandomHorizontalFlip())
    if AUGMENT_COLOR_JITTER:
        augments.append(
            transforms.ColorJitter(
                brightness=COLOR_JITTER_BRIGHTNESS,
                contrast=COLOR_JITTER_CONTRAST,
                saturation=COLOR_JITTER_SATURATION,
            )
        )
    if AUGMENT_RANDOM_ROTATION:
        augments.append(transforms.RandomRotation(ROTATION_DEGREES))
    augments.extend([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(augments)


def get_face_cropper():
    """
    Return an MTCNN face cropper from facenet-pytorch, or None if not used.
    Used to optionally align/crop faces before resizing to 224x224.
    """
    if not USE_FACE_CROP:
        return None
    try:
        from facenet_pytorch import MTCNN
        # Default MTCNN; image_size is the output size for the face crop
        mtcnn = MTCNN(image_size=IMAGE_SIZE[0], margin=20)
        return mtcnn
    except ImportError:
        return None


def align_face_or_center_crop(pil_image, mtcnn, target_size):
    """
    If mtcnn is not None and a face is detected, crop and align to target_size.
    Otherwise return center crop of resized image.
    target_size: (H, W) 
    """
    if mtcnn is not None:
        try:
            # MTCNN returns tensor (C, H, W) or None
            face_tensor = mtcnn(pil_image)
            if face_tensor is not None:
                return face_tensor  # Already 224x224 from MTCNN config
        except Exception:
            pass
    # Fallback: resize and center crop
    resize = transforms.Resize(max(target_size))
    crop = transforms.CenterCrop(target_size)
    to_tensor = transforms.ToTensor()
    img = resize(pil_image)
    img = crop(img)
    return to_tensor(img)


def normalize_tensor(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Apply ImageNet normalization to a tensor (C, H, W)."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (tensor - mean) / std
