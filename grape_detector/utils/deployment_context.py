"""Identify the model and data/settings behind a reusable deployment threshold."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def _model_state_digest(model: Any) -> str:
    # Include all parameters and buffers, including the actual adapted BN stats.
    # Copy one tensor at a time so the fingerprint does not duplicate the model.
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(model.model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        header = [name, str(tensor.dtype), list(tensor.shape)]
        digest.update(json.dumps(header, separators=(",", ":")).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _background_digest(image_dir: Path, image_paths: Sequence[Path]) -> str:
    # Content hashes detect in-place edits even if filenames/mtimes are retained.
    digest = hashlib.sha256()
    for path in sorted(image_paths):
        file_digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(chunk)
        record = [str(path.relative_to(image_dir)), file_digest.hexdigest()]
        digest.update(json.dumps(record, separators=(",", ":")).encode())
    return digest.hexdigest()


def build_deployment_reuse_context(
    *,
    model: Any,
    adabn_info: Mapping[str, Any],
    calibration_options: Mapping[str, Any],
    test_settings: Mapping[str, Any],
    image_paths: Sequence[Path],
) -> Dict[str, Any]:
    import torch
    import ultralytics

    image_dir = Path(calibration_options["image_dir"]).expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Deployment calibration directory does not exist: {image_dir}")
    if not image_paths:
        raise ValueError(f"Deployment calibration directory contains no images: {image_dir}")

    adabn_keys = (
        "enabled", "applied", "status", "image_dir", "imgsz", "batch_size",
        "max_images", "sample_seed", "sampling_method", "selected_image_names",
        "epochs", "max_stage", "reset_stats", "momentum_override",
        "fp16_requested", "fp16_applied", "compute_dtype", "torch_device",
        "num_images_used", "num_skipped_images", "num_batches", "selected_stage_indices",
    )
    calibration = {
        key: value for key, value in calibration_options.items()
        if key not in {"image_dir", "disable_tqdm"}
    }
    calibration.update(
        image_dir=str(image_dir),
        num_images=len(image_paths),
        images_sha256=_background_digest(image_dir, image_paths),
    )
    context = {
        "version": 1,
        "model_state_sha256": _model_state_digest(model),
        "end2end": bool(getattr(model.model, "end2end", False)),
        "adabn": {key: adabn_info.get(key) for key in adabn_keys},
        "calibration": calibration,
        "test_settings": dict(test_settings),
        "runtime": {"torch": str(torch.__version__), "ultralytics": str(ultralytics.__version__)},
    }
    # Normalize mapping keys and sequence types exactly as persisted JSON does.
    return json.loads(json.dumps(context))


def deployment_context_mismatches(saved: Any, current: Mapping[str, Any]) -> list[str]:
    """Return differing field names, without logging image lists or tensor data."""
    if not isinstance(saved, dict):
        return ["missing reuse context (legacy threshold)"]

    def compare(old: Any, new: Any, prefix: str) -> list[str]:
        if isinstance(old, dict) and isinstance(new, dict):
            differences = []
            for key in sorted(old.keys() | new.keys()):
                name = f"{prefix}.{key}" if prefix else key
                if key not in old or key not in new:
                    differences.append(name)
                else:
                    differences.extend(compare(old[key], new[key], name))
            return differences
        return [] if old == new else [prefix]

    return compare(saved, current, "")
