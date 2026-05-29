"""
================================================================================
MODEL.PY — Neural inference + face preprocessing
================================================================================

ROLE IN THE SYSTEM
------------------
Loads three PyTorch classifiers once, runs them on a face crop, then hands
probabilities to evaluation.py for labels and explanations.

Models:
  - EfficientNet-B4 (DeepfakeBench weights) — 256×256, mean/std 0.5
  - XceptionNet (DeepfakeBench weights)       — same preprocess as EffB4
  - Fairness ResNet18                         — 224×224, ImageNet normalize

PIPELINE POSITION
-----------------
  app.py  →  predict_deepfake()  →  build_evaluation() in evaluation.py

Environment:
  UNMASK_DEBUG=1        → verbose pipeline logs via diagnostics.py
  UNMASK_DETERMINISTIC=1 → fixed seeds for reproducible tests
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Tuple

from diagnostics import debug_enabled, log_stage
from evaluation import (
    ModelOutputs,
    PreprocessInfo,
    THRESHOLD_FAKE,
    THRESHOLD_REAL,
    build_evaluation,
)

# ==============================================================================
# OPTIONAL DETERMINISTIC MODE — for regression tests
# ==============================================================================
_DETERMINISTIC = os.environ.get("UNMASK_DETERMINISTIC", "").strip() in ("1", "true", "yes")
if _DETERMINISTIC:
    try:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass

logging.basicConfig(level=logging.INFO if debug_enabled() else logging.WARNING)

# ==============================================================================
# HEAVY IMPORTS — torch, PIL, OpenCV (after logging setup)
# ==============================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    import cv2  
except Exception as e:  
    raise RuntimeError("OpenCV is required. Install opencv-python-headless.") from e

try:
    from efficientnet_pytorch import EfficientNet  # type: ignore
except Exception as e: 
    raise RuntimeError("efficientnet-pytorch is required. Install efficientnet-pytorch.") from e


# ==============================================================================
# PREPROCESSING CONSTANTS + WEIGHT FILE PATHS
# ==============================================================================
# EffB4/Xception: 256×256, mean/std 0.5 (DeepfakeBench convention).
# Fairness: 224×224, ImageNet mean/std. Paths searched in order until found.
# ==============================================================================
INPUT_RESOLUTION = 256
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)
_MEAN_T = torch.tensor(NORM_MEAN).view(3, 1, 1)
_STD_T = torch.tensor(NORM_STD).view(3, 1, 1)

# Model weights paths.
_HERE = os.path.dirname(__file__)
_EFFB4_WEIGHTS_PATHS = (
    os.path.join(_HERE, "DeepfakeBench", "weights", "effnb4_best.pth"),
    os.path.join(_HERE, "weights", "effnb4_best.pth"),
)
_XCEPTION_WEIGHTS_PATHS = (
    os.path.join(_HERE, "DeepfakeBench", "weights", "xception_best.pth"),
    os.path.join(_HERE, "weights", "xception_best.pth"),
)

# Fairness model (ResNet18 real vs fake): required. Search in these locations (in order):
_FAIRNESS_WEIGHTS_PATHS = (
    os.path.join(_HERE, "fairness_model", "models", "fairness_head_best.pt"),
    os.path.join(_HERE, "fairness_model", "models", "fairness_head.pt"),
    os.path.join(_HERE, "..", "fairness_model", "models", "fairness_head_best.pt"),
    os.path.join(_HERE, "..", "fairness_model", "models", "fairness_head.pt"),
)
# Preprocessing for fairness model: 224x224, ImageNet normalization
_FAIRNESS_SIZE = 224
_FAIRNESS_MEAN = (0.485, 0.456, 0.406)
_FAIRNESS_STD = (0.229, 0.224, 0.225)

# ==============================================================================
# NEURAL NETWORK ARCHITECTURES (checkpoint-compatible with DeepfakeBench / fairness)
# ==============================================================================
# EfficientNetB4Deepfake, XceptionNetDeepfake: main deepfake detectors.
# FairnessResNet18: auxiliary head for calibration / fairness signal in ensemble.
# Block/SeparableConv: Xception building blocks only — not used outside XceptionNet.
# ==============================================================================
class EfficientNetB4Deepfake(nn.Module):

    # This class serves as a minimal reproduction of DeepfakeBench EfficientNet-B4

    def __init__(self) -> None:
        super().__init__()
        self.efficientnet = EfficientNet.from_name("efficientnet-b4")
        # Match DeepfakeBench: remove original classifier.
        self.efficientnet._fc = nn.Identity()
        # Match DeepfakeBench: explicit conv stem for 3-channel input (same as original B4)
        self.efficientnet._conv_stem = nn.Conv2d(3, 48, kernel_size=3, stride=2, bias=False)
        # Match DeepfakeBench naming for checkpoint compatibility
        self.last_layer = nn.Linear(1792, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.efficientnet.extract_features(x)
        pooled = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        return self.last_layer(pooled)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super().__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class XceptionNetDeepfake(nn.Module):
    
    # This class serves as a minimal reproduction of DeepfakeBench XceptionNet
    

    def __init__(self) -> None:
        super().__init__()
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        # Middle flow
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        # Exit flow
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.last_linear = nn.Linear(2048, 2)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return self.last_linear(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.classifier(feats)


class FairnessResNet18(nn.Module):
# ResNet18 with 2-class head (real=0, fake=1) which has same structure as fairness_model (no wrapper)

    def __init__(self) -> None:
        super().__init__()
        from torchvision import models as tv_models
        self.backbone = tv_models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ==============================================================================
# SINGLETON MODEL CACHE — load once per process, thread-safe
# ==============================================================================
# _load_once(): loads EffB4, Xception, fairness, Haar cascade on first request.
# Subsequent calls reuse globals (important for FastAPI — avoid reload per image).
# ==============================================================================
_lock = threading.Lock()
_model_effb4: EfficientNetB4Deepfake | None = None
_model_xception: XceptionNetDeepfake | None = None
_model_fairness: FairnessResNet18 | None = None
_device: torch.device | None = None
_face_detector: "cv2.CascadeClassifier | None" = None


def _load_once() -> tuple[
    EfficientNetB4Deepfake, XceptionNetDeepfake, FairnessResNet18, torch.device, "cv2.CascadeClassifier"
]:
    global _model_effb4, _model_xception, _model_fairness, _device, _face_detector

    if _model_effb4 is not None and _model_xception is not None and _model_fairness is not None and _device is not None and _face_detector is not None:
        return _model_effb4, _model_xception, _model_fairness, _device, _face_detector

    with _lock:
        if _model_effb4 is not None and _model_xception is not None and _model_fairness is not None and _device is not None and _face_detector is not None:
            return _model_effb4, _model_xception, _model_fairness, _device, _face_detector

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Face detector (OpenCV), same crop used for all models
        cascade_path = os.path.join(getattr(cv2.data, "haarcascades", ""), "haarcascade_frontalface_default.xml")
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            raise RuntimeError("Failed to initialize OpenCV Haar cascade face detector.")

        # Load EfficientNet-B4.
        effb4_path = None
        for candidate in _EFFB4_WEIGHTS_PATHS:
            if os.path.exists(candidate):
                effb4_path = candidate
                break
        if not effb4_path:
            raise RuntimeError(
                "Missing EfficientNet-B4 weights. Expected at one of:\n"
                + "\n".join(f"  - {p}" for p in _EFFB4_WEIGHTS_PATHS)
            )
        model_effb4 = EfficientNetB4Deepfake()
        _load_checkpoint_into_model(model_effb4, effb4_path)
        model_effb4.to(device)
        model_effb4.eval()

        # Load XceptionNet.
        xception_path = None
        for candidate in _XCEPTION_WEIGHTS_PATHS:
            if os.path.exists(candidate):
                xception_path = candidate
                break
        if not xception_path:
            raise RuntimeError(
                "Missing XceptionNet weights. Expected at one of:\n"
                + "\n".join(f"  - {p}" for p in _XCEPTION_WEIGHTS_PATHS)
            )
        model_xception = XceptionNetDeepfake()
        _load_checkpoint_into_model(model_xception, xception_path)
        model_xception.to(device)
        model_xception.eval()

        # Fairness model (ResNet18 real vs fake)
        fairness_path = None
        for candidate in _FAIRNESS_WEIGHTS_PATHS:
            p = os.path.normpath(os.path.abspath(candidate))
            if os.path.isfile(p):
                fairness_path = p
                break
        if fairness_path is None:
            raise RuntimeError(
                "Missing fairness model weights. Expected at one of:\n  "
                + "\n  ".join(os.path.normpath(os.path.abspath(p)) for p in _FAIRNESS_WEIGHTS_PATHS)
            )
        model_fairness = FairnessResNet18()
        ckpt = torch.load(fairness_path, map_location=device)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        # Fairness checkpoint was saved from a plain ResNet18 (keys: conv1, fc, ...). Our wrapper uses .backbone, so remap.
        if any(k.startswith("backbone.") for k in state.keys()):
            pass  # already has prefix
        else:
            state = {f"backbone.{k}": v for k, v in state.items()}
        model_fairness.load_state_dict(state, strict=True)
        model_fairness.to(device)
        model_fairness.eval()

        _model_effb4, _model_xception, _model_fairness, _device, _face_detector = (
            model_effb4, model_xception, model_fairness, device, detector
        )
        return model_effb4, model_xception, model_fairness, device, detector


# ==============================================================================
# CHECKPOINT LOADING — DeepfakeBench .pth and fairness .pt formats
# ==============================================================================
# _load_checkpoint_into_model(): strips module./backbone. prefixes, strict=False.
# Fairness checkpoint remapped to FairnessResNet18.backbone.* in _load_once().
# ==============================================================================
def _load_checkpoint_into_model(model: nn.Module, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "net", "network"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break

    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported checkpoint format at '{path}'.")

    cleaned: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k
        for prefix in ("module.", "backbone."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    # If the checkpoint is very incompatible, fail with error
    if len(missing) > 20:
        raise RuntimeError(
            f"Checkpoint does not match model architecture. "
            f"Missing keys (sample): {missing[:10]}; unexpected keys (sample): {unexpected[:10]}"
        )


# ==============================================================================
# FACE DETECTION + CROPPING — OpenCV Haar, shared crop for all models
# ==============================================================================
# _detect_largest_face(): largest face bbox or None.
# _crop_with_margin(): expand bbox 25% for context.
# _center_square_crop(): fallback when no face (still analyzed, weaker reliability).
# ==============================================================================
def _detect_largest_face(image_rgb: np.ndarray, detector: "cv2.CascadeClassifier") -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if faces is None or len(faces) == 0:
        return None
    # Chooses the largest face.
    x, y, w, h = max(faces, key=lambda b: int(b[2]) * int(b[3]))
    return int(x), int(y), int(w), int(h)


def _crop_with_margin(img: Image.Image, bbox: tuple[int, int, int, int], margin: float = 0.25) -> Image.Image:
    x, y, w, h = bbox
    mx = int(w * margin)
    my = int(h * margin)
    left = max(0, x - mx)
    top = max(0, y - my)
    right = min(img.width, x + w + mx)
    bottom = min(img.height, y + h + my)
    return img.crop((left, top, right, bottom))


def _center_square_crop(img: Image.Image) -> Image.Image:
    side = min(img.width, img.height)
    left = (img.width - side) // 2
    top = (img.height - side) // 2
    return img.crop((left, top, left + side, top + side))


# ==============================================================================
# TENSOR PREPROCESSING — separate pipelines for main detectors vs fairness
# ==============================================================================
# _preprocess(): 256×256 tensor for EffB4 + Xception (one forward pass each).
# _preprocess_fairness(): 224×224 ImageNet-normalized tensor for fairness head.
# ==============================================================================
def _preprocess(face_img: Image.Image) -> torch.Tensor:
    if face_img.mode != "RGB":
        face_img = face_img.convert("RGB")
    face_img = face_img.resize((INPUT_RESOLUTION, INPUT_RESOLUTION), Image.BILINEAR)
    arr = np.asarray(face_img).astype(np.float32) / 255.0 
    tensor = torch.from_numpy(arr).permute(2, 0, 1) 
    tensor = (tensor - _MEAN_T) / _STD_T
    return tensor.unsqueeze(0)


def _preprocess_fairness(face_img: Image.Image) -> torch.Tensor:
    """Resize to 224x224 and ImageNet normalize for the fairness (ResNet18) model."""
    if face_img.mode != "RGB":
        face_img = face_img.convert("RGB")
    face_img = face_img.resize((_FAIRNESS_SIZE, _FAIRNESS_SIZE), Image.BILINEAR)
    arr = np.asarray(face_img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean_t = torch.tensor(_FAIRNESS_MEAN).view(3, 1, 1)
    std_t = torch.tensor(_FAIRNESS_STD).view(3, 1, 1)
    tensor = (tensor - mean_t) / std_t
    return tensor.unsqueeze(0)


# ==============================================================================
# MAIN ENTRY POINT — predict_deepfake(image)
# ==============================================================================
# Called by app.py on every /detect-image request.
#
# Steps:
#   1. _load_once() — models + face detector on GPU/CPU
#   2. Detect face → crop (or center crop)
#   3. Forward pass ×3 → softmax class-1 = P(fake)
#   4. build_evaluation(ModelOutputs, PreprocessInfo) in evaluation.py
#   5. Return label, prob_fake, confidence, texts, developer_report
#
# Does NOT apply policy thresholds here — that is entirely evaluation.py.
# ==============================================================================
@torch.inference_mode()
def predict_deepfake(image: Image.Image) -> Tuple[str, float, float, str, str, dict[str, Any]]:
    """
    Run ensemble inference.
    Returns: label, prob_fake, overall_confidence, user_explanation, disclaimer, developer_report.
    """
    if _DETERMINISTIC:
        torch.manual_seed(42)
        np.random.seed(42)

    model_effb4, model_xception, model_fairness, device, detector = _load_once()
    log_stage("device", {"device": str(device), "deterministic": _DETERMINISTIC})

    if hasattr(image, "load"):
        image.load()
    if image.mode != "RGB":
        image = image.convert("RGB")

    np_rgb = np.asarray(image)
    bbox = _detect_largest_face(np_rgb, detector)

    if bbox is not None:
        face = _crop_with_margin(image, bbox)
        face_bbox = {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]}
        face_detected = True
    else:
        face = _center_square_crop(image)
        face_bbox = None
        face_detected = False

    log_stage("preprocess", {
        "face_bbox": face_bbox,
        "face_size": [face.width, face.height],
        "effb4_xception": {"size": INPUT_RESOLUTION, "mean": NORM_MEAN, "std": NORM_STD},
        "fairness": {"size": _FAIRNESS_SIZE, "mean": _FAIRNESS_MEAN, "std": _FAIRNESS_STD},
    })

    # EffB4 and Xception: same 256x256 preprocessing
    x = _preprocess(face).to(device)
    log_stage("tensor_shapes", {"effb4_xception_input": list(x.shape)})
    logits_effb4 = model_effb4(x)[0]
    prob_effb4 = float(torch.softmax(logits_effb4, dim=0)[1].item())
    logits_xception = model_xception(x)[0]
    prob_xception = float(torch.softmax(logits_xception, dim=0)[1].item())

    # Fairness model: same face crop, 224x224 ImageNet normalization
    x_fairness = _preprocess_fairness(face).to(device)
    log_stage("tensor_shapes", {"fairness_input": list(x_fairness.shape)})
    logits_fairness = model_fairness(x_fairness)[0]
    prob_fairness = float(torch.softmax(logits_fairness, dim=0)[1].item())

    log_stage("raw_model_outputs", {
        "logits_effb4": logits_effb4.detach().cpu().tolist(),
        "logits_xception": logits_xception.detach().cpu().tolist(),
        "logits_fairness": logits_fairness.detach().cpu().tolist(),
        "prob_fake_class_index": 1,
        "prob_effb4": prob_effb4,
        "prob_xception": prob_xception,
        "prob_fairness": prob_fairness,
    })

    preprocess = PreprocessInfo(
        face_detected=face_detected,
        face_note=(
            "Detected a face and analyzed the cropped face region."
            if face_detected
            else "No clear face detected; analyzed a center crop of the image."
        ),
        face_bbox=face_bbox,
        face_size=(face.width, face.height),
    )
    models = ModelOutputs(
        prob_effb4=prob_effb4,
        prob_xception=prob_xception,
        prob_fairness=prob_fairness,
        logits_effb4=logits_effb4.detach().cpu().tolist(),
        logits_xception=logits_xception.detach().cpu().tolist(),
        logits_fairness=logits_fairness.detach().cpu().tolist(),
    )

    result = build_evaluation(models, preprocess)
    debug_scores = result.developer_report["debug_scores"]
    log_stage("evaluation", {
        "label": result.label,
        "prob_fake": result.prob_fake,
        "confidence_level": result.confidence_level,
        "fusion_mode": debug_scores.get("fusion_mode"),
        **{k: debug_scores[k] for k in ("effb4", "xception", "fairness", "ensemble")},
    })
    log_stage("final", {"label": result.label, **debug_scores})
    return (
        result.label,
        result.prob_fake,
        result.overall_confidence,
        result.user_explanation,
        result.disclaimer,
        result.developer_report,
    )
