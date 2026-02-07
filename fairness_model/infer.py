"""
Inference script: load saved fairness model and predict real vs fake for an image.
Usage: python infer.py --image path/to/image.jpg
"""

import os
import sys
import argparse

import torch
from PIL import Image

# Add project root so we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import (
    MODEL_BEST_PATH,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    USE_FACE_CROP,
    LABEL_REAL,
    LABEL_FAKE,
)
from src.model_builder import load_fairness_model
from src.utils import get_device, load_image_rgb, get_face_cropper, align_face_or_center_crop, normalize_tensor


def preprocess_for_inference(image_path, use_face_crop=None):
    """
    Load image, optionally run MTCNN face crop, then resize/normalize to model input.
    Returns tensor (1, 3, 224, 224) and device.
    """
    if use_face_crop is None:
        use_face_crop = USE_FACE_CROP
    img = load_image_rgb(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    device = get_device()
    mtcnn = get_face_cropper() if use_face_crop else None
    if mtcnn is not None:
        try:
            tensor = mtcnn(img)
            if tensor is not None:
                tensor = normalize_tensor(tensor)
                tensor = tensor.unsqueeze(0).to(device)
                return tensor, device
        except Exception:
            pass
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor, device


def infer(image_path, model_path=MODEL_BEST_PATH):
    """
    Load model and image, run forward pass, return label and probability for fake (class 1).
    """
    model = load_fairness_model(model_path)
    device = next(model.parameters()).device
    x, _ = preprocess_for_inference(image_path)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    # prob_fake = P(class=1)
    prob_fake = probs[0, 1].item()
    pred = 1 if prob_fake >= 0.5 else 0
    label_str = "fake" if pred == LABEL_FAKE else "real"
    return label_str, prob_fake


def main():
    parser = argparse.ArgumentParser(description="Run fairness model inference on an image")
    parser.add_argument("--image", required=True, help="Path to image file (e.g. .jpg, .png)")
    parser.add_argument("--model", default=MODEL_BEST_PATH, help="Path to saved model (default: fairness_head_best.pt)")
    args = parser.parse_args()
    if not os.path.isfile(args.image):
        print(f"Error: file not found: {args.image}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"Error: model not found: {args.model}")
        sys.exit(1)
    label, prob_fake = infer(args.image, model_path=args.model)
    print(f"Prediction: {label}")
    print(f"Probability (fake): {prob_fake:.4f}")


if __name__ == "__main__":
    main()
