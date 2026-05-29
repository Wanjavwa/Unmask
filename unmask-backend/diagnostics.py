"""
================================================================================
DIAGNOSTICS.PY — Optional pipeline logging (no effect on scores or labels)
================================================================================

ROLE IN THE SYSTEM
------------------
When UNMASK_DEBUG=1, model.py emits structured JSON log lines at each stage
(device, preprocess, raw logits, evaluation summary). This helps trace bugs
without changing inference results.

PIPELINE POSITION
-----------------
  model.predict_deepfake()  →  log_stage("preprocess", {...})
                             →  log_stage("raw_model_outputs", {...})
                             →  log_stage("evaluation", {...})

This module is intentionally tiny: enable flag + logger + JSON serializer.
It does NOT import torch or evaluation.py (avoids circular imports).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

# ==============================================================================
# ENABLE FLAG
# ==============================================================================
# Set UNMASK_DEBUG=1 in the environment before starting uvicorn.
# model.py also sets logging level to INFO when debug is on.
# ==============================================================================
_DEBUG = os.environ.get("UNMASK_DEBUG", "").strip() in ("1", "true", "yes")
_LOGGER = logging.getLogger("unmask.pipeline")


# ==============================================================================
# PUBLIC API
# ==============================================================================
def debug_enabled() -> bool:
    """Return True if UNMASK_DEBUG is enabled. Used by model.py for log level."""
    return _DEBUG


def log_stage(stage: str, payload: dict[str, Any]) -> None:
    """
    Log one pipeline stage as JSON when debugging is on.

    Parameters
    ----------
    stage : short name, e.g. "preprocess", "evaluation", "final"
    payload : dict of numbers/strings/lists safe for JSON (tensors converted)

    By itself: writes one INFO line to the "unmask.pipeline" logger.
    In the system: called from model.predict_deepfake at key milestones.
    """
    if not _DEBUG:
        return
    _LOGGER.info("[%s] %s", stage, json.dumps(payload, default=_json_default))


# ==============================================================================
# JSON SERIALIZATION HELPER
# ==============================================================================
def _json_default(obj: Any) -> Any:
    """Convert numpy/torch types to lists/strings so json.dumps does not fail."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
