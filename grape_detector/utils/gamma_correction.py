from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

import cv2
import numpy as np


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null"}:
            return default
        return text in {"1", "true", "yes", "y", "on"}
    return bool(value)


@lru_cache(maxsize=32)
def _gamma_lut(gamma: float) -> np.ndarray:
    return np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")


def apply_gamma_correction(image: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError(f"gamma must be greater than 0: {gamma}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.LUT(image, _gamma_lut(gamma))


def resolve_gamma_correction_config(
    config: Dict[str, Any],
    *,
    prefix: str,
    default_gamma: float = 0.6,
) -> Dict[str, Any]:
    prefix = str(prefix).strip().upper()
    enabled_key = f"{prefix}_GAMMA_CORRECTION_ENABLED"
    value_key = f"{prefix}_GAMMA_CORRECTION_VALUE"

    enabled = as_bool(config.get(enabled_key, False), default=False)
    raw_gamma = config.get(value_key, default_gamma)
    if raw_gamma is None or (isinstance(raw_gamma, str) and raw_gamma.strip().lower() in {"", "none", "null"}):
        raw_gamma = default_gamma
    gamma = float(raw_gamma)
    if gamma <= 0:
        raise ValueError(f"{value_key} must be greater than 0: {gamma}")

    return {
        "enabled": enabled,
        "gamma": gamma,
        "enabled_key": enabled_key,
        "value_key": value_key,
    }
