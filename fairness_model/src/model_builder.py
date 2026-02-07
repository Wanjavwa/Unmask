"""
ResNet18-based binary classifier (real vs fake) for fairness evaluation.
Uses pretrained torchvision ResNet18 and replaces the final FC layer for 2 classes.
"""

import torch
import torch.nn as nn
from torchvision import models

from .config import NUM_CLASSES


def build_fairness_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    Build ResNet18 with pretrained ImageNet weights and replace the final
    fully connected layer to output num_classes (default 2: real=0, fake=1).
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_fairness_model(path, device=None, num_classes=NUM_CLASSES):
    """
    Load a saved fairness model from path. Expects state_dict or full model state.
    Returns model on the given device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_fairness_model(num_classes=num_classes, pretrained=False)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model
