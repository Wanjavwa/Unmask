"""
Structured diagnostic logging for the Unmask detection pipeline.
Enable with environment variable UNMASK_DEBUG=1
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

_DEBUG = os.environ.get("UNMASK_DEBUG", "").strip() in ("1", "true", "yes")
_LOGGER = logging.getLogger("unmask.pipeline")


def debug_enabled() -> bool:
    return _DEBUG


def log_stage(stage: str, payload: dict[str, Any]) -> None:
    if not _DEBUG:
        return
    _LOGGER.info("[%s] %s", stage, json.dumps(payload, default=_json_default))


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
