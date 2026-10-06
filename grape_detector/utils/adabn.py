from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import ultralytics
from tqdm import tqdm
from ultralytics import YOLO


IMAGE_GLOB_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")


def _resolve_torch_device(device: Any) -> torch.device:
    device_text = str(device).strip()
    if not device_text:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_text.lower() == "cpu":
        return torch.device("cpu")

    primary = device_text.split(",")[0].strip()
    # The CLI may expose a single physical GPU (e.g. CUDA_VISIBLE_DEVICES=1).
    # PyTorch addresses that GPU as cuda:0; Ultralytics still receives the
    # configured device string and handles its own selection as before.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1" and "," not in visible:
        if primary.isdigit() or primary == "cuda" or (
            primary.startswith("cuda:") and primary.split(":", 1)[1].isdigit()
        ):
            primary = "0"
    if primary.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{primary}")
        print(f"[AdaBN] CUDA is unavailable; using CPU instead of the requested device={device_text}.")
        return torch.device("cpu")

    try:
        resolved = torch.device(primary)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            print(f"[AdaBN] CUDA is unavailable; using CPU instead of the requested device={device_text}.")
            return torch.device("cpu")
        return resolved
    except (TypeError, RuntimeError):
        print(f"[AdaBN] Could not resolve device={device_text}; using CPU.")
        return torch.device("cpu")


def _set_current_cuda_device_if_needed(torch_device: torch.device) -> None:
    if torch_device.type != "cuda":
        return
    if torch_device.index is None:
        return
    try:
        torch.cuda.set_device(torch_device.index)
    except Exception as exc:
        print(f"[AdaBN][WARN] torch.cuda.set_device({torch_device.index}) failed: {exc}")


def _extract_top_level_stage_index(module_name: str) -> Optional[int]:
    parts = str(module_name).split(".")
    if len(parts) < 2 or parts[0] != "model":
        return None
    if parts[1].isdigit():
        return int(parts[1])
    return None


def list_depth1_images(
    image_dir: Path,
    max_images: Optional[int] = None,
    *,
    sample_seed: Optional[int] = None,
) -> List[Path]:
    """Select without replacement from all images; keep filename processing order.

    An omitted seed retains the historical filename-prefix selection. A local
    RNG makes seeded selection independent of model/global random state.
    """
    limit = None if max_images is None else max(0, int(max_images))
    if limit == 0:
        return []
    image_paths: List[Path] = []
    image_exts = {Path(pattern).suffix.lower() for pattern in IMAGE_GLOB_PATTERNS}
    for filename in sorted(os.listdir(image_dir)):
        if Path(filename).suffix.lower() not in image_exts:
            continue
        if not (image_dir / filename).is_file():
            continue
        image_paths.append(image_dir / filename)
        if sample_seed is None and limit is not None and len(image_paths) >= limit:
            break
    if sample_seed is not None and limit is not None and limit < len(image_paths):
        image_paths = sorted(random.Random(sample_seed).sample(image_paths, limit))
    return image_paths


def _tensor_summary(tensor: Optional[torch.Tensor]) -> Dict[str, Any]:
    if tensor is None:
        return {
            "numel": 0,
            "finite_count": 0,
            "nonfinite_count": 0,
            "finite_ratio": None,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "abs_max": None,
        }

    data = tensor.detach().float()
    numel = int(data.numel())
    finite_mask = torch.isfinite(data)
    finite_count = int(finite_mask.sum().item())
    nonfinite_count = numel - finite_count
    summary = {
        "numel": numel,
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "finite_ratio": (float(finite_count) / float(numel)) if numel > 0 else None,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "abs_max": None,
    }
    if finite_count <= 0:
        return summary

    finite_values = data[finite_mask]
    summary.update(
        {
            "min": float(finite_values.min().item()),
            "max": float(finite_values.max().item()),
            "mean": float(finite_values.mean().item()),
            "std": float(finite_values.std(unbiased=False).item()),
            "abs_max": float(finite_values.abs().max().item()),
        }
    )
    return summary


def _snapshot_bn_stats(bn_layers: Sequence[Tuple[str, nn.BatchNorm2d]]) -> List[Dict[str, Any]]:
    snapshot: List[Dict[str, Any]] = []
    for layer_name, bn_layer in bn_layers:
        running_mean = None if bn_layer.running_mean is None else bn_layer.running_mean.detach().float().cpu().clone()
        running_var = None if bn_layer.running_var is None else bn_layer.running_var.detach().float().cpu().clone()
        snapshot.append(
            {
                "name": layer_name,
                "running_mean_tensor": running_mean,
                "running_var_tensor": running_var,
                "running_mean_summary": _tensor_summary(running_mean),
                "running_var_summary": _tensor_summary(running_var),
            }
        )
    return snapshot


def _compare_bn_snapshots(
    before_snapshot: Sequence[Dict[str, Any]],
    after_snapshot: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    layer_reports: List[Dict[str, Any]] = []
    mean_delta_values: List[float] = []
    var_delta_values: List[float] = []
    nonfinite_after_layers = 0
    near_zero_var_layers = 0
    huge_var_layers = 0

    for before_item, after_item in zip(before_snapshot, after_snapshot):
        before_mean = before_item.get("running_mean_tensor")
        before_var = before_item.get("running_var_tensor")
        after_mean = after_item.get("running_mean_tensor")
        after_var = after_item.get("running_var_tensor")

        mean_abs_delta_mean = None
        mean_abs_delta_var = None
        max_abs_delta_mean = None
        max_abs_delta_var = None
        mean_delta_ratio = None
        var_delta_ratio = None

        if before_mean is not None and after_mean is not None:
            mean_delta = (after_mean - before_mean).abs()
            mean_abs_delta_mean = float(mean_delta.mean().item())
            max_abs_delta_mean = float(mean_delta.max().item())
            denom = float(before_mean.abs().mean().item()) + 1e-12
            mean_delta_ratio = float(mean_abs_delta_mean / denom)
            mean_delta_values.append(mean_abs_delta_mean)

        if before_var is not None and after_var is not None:
            var_delta = (after_var - before_var).abs()
            mean_abs_delta_var = float(var_delta.mean().item())
            max_abs_delta_var = float(var_delta.max().item())
            denom = float(before_var.abs().mean().item()) + 1e-12
            var_delta_ratio = float(mean_abs_delta_var / denom)
            var_delta_values.append(mean_abs_delta_var)

        after_var_summary = after_item["running_var_summary"]
        if (after_item["running_mean_summary"]["nonfinite_count"] or 0) > 0 or (
            after_var_summary["nonfinite_count"] or 0
        ) > 0:
            nonfinite_after_layers += 1
        if after_var_summary.get("abs_max") is not None and float(after_var_summary["abs_max"]) < 1e-6:
            near_zero_var_layers += 1
        if after_var_summary.get("abs_max") is not None and float(after_var_summary["abs_max"]) > 1e3:
            huge_var_layers += 1

        layer_reports.append(
            {
                "name": before_item["name"],
                "before_running_mean": before_item["running_mean_summary"],
                "after_running_mean": after_item["running_mean_summary"],
                "before_running_var": before_item["running_var_summary"],
                "after_running_var": after_item["running_var_summary"],
                "mean_abs_delta_running_mean": mean_abs_delta_mean,
                "max_abs_delta_running_mean": max_abs_delta_mean,
                "mean_abs_delta_running_var": mean_abs_delta_var,
                "max_abs_delta_running_var": max_abs_delta_var,
                "mean_delta_ratio_vs_before_mean_abs_mean": mean_delta_ratio,
                "var_delta_ratio_vs_before_var_abs_mean": var_delta_ratio,
            }
        )

    top_mean_delta = sorted(
        [item for item in layer_reports if item["mean_abs_delta_running_mean"] is not None],
        key=lambda item: float(item["mean_abs_delta_running_mean"]),
        reverse=True,
    )[:10]
    top_var_delta = sorted(
        [item for item in layer_reports if item["mean_abs_delta_running_var"] is not None],
        key=lambda item: float(item["mean_abs_delta_running_var"]),
        reverse=True,
    )[:10]

    return {
        "global_summary": {
            "num_layers": len(layer_reports),
            "mean_of_mean_abs_delta_running_mean": (
                float(sum(mean_delta_values) / len(mean_delta_values)) if mean_delta_values else None
            ),
            "max_of_mean_abs_delta_running_mean": (
                float(max(mean_delta_values)) if mean_delta_values else None
            ),
            "mean_of_mean_abs_delta_running_var": (
                float(sum(var_delta_values) / len(var_delta_values)) if var_delta_values else None
            ),
            "max_of_mean_abs_delta_running_var": (
                float(max(var_delta_values)) if var_delta_values else None
            ),
            "nonfinite_after_layers": nonfinite_after_layers,
            "near_zero_running_var_layers": near_zero_var_layers,
            "huge_running_var_layers": huge_var_layers,
        },
        "top_mean_delta_layers": top_mean_delta,
        "top_var_delta_layers": top_var_delta,
        "all_layers": layer_reports,
    }


def _collect_bn_activation_debug(
    base_model: nn.Module,
    bn_layers: Sequence[Tuple[str, nn.BatchNorm2d]],
    batch_tensor: torch.Tensor,
) -> Dict[str, Any]:
    activation_records: List[Dict[str, Any]] = []
    hooks = []

    for layer_name, bn_layer in bn_layers:
        def _hook(module, inputs, output, *, _layer_name=layer_name):
            activation_records.append(
                {
                    "name": _layer_name,
                    "output_summary": _tensor_summary(output),
                }
            )

        hooks.append(bn_layer.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            output = base_model(batch_tensor)
    finally:
        for hook in hooks:
            hook.remove()

    suspicious_layers = [
        item
        for item in activation_records
        if (item["output_summary"].get("nonfinite_count") or 0) > 0
        or (
            item["output_summary"].get("abs_max") is not None
            and float(item["output_summary"]["abs_max"]) > 1e3
        )
    ]

    return {
        "input_summary": _tensor_summary(batch_tensor),
        "model_output_summary": _tensor_summary(output if isinstance(output, torch.Tensor) else None),
        "num_bn_layers_recorded": len(activation_records),
        "suspicious_layer_count": len(suspicious_layers),
        "suspicious_layers": suspicious_layers[:20],
        "layer_preview": activation_records[:20],
    }


def _letterbox_bgr_image(image: np.ndarray, imgsz: int) -> np.ndarray:
    h0, w0 = image.shape[:2]
    if h0 <= 0 or w0 <= 0:
        raise ValueError(f"Invalid image shape: {(h0, w0)}")

    ratio = min(float(imgsz) / float(h0), float(imgsz) / float(w0))
    new_w = max(1, int(round(w0 * ratio)))
    new_h = max(1, int(round(h0 * ratio)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top = (imgsz - new_h) // 2
    left = (imgsz - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def _prepare_batch_tensor(
    image_paths: Sequence[Path],
    *,
    imgsz: int,
    torch_device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], int]:
    batch_images: List[torch.Tensor] = []
    skipped = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[AdaBN][WARN] Failed to load image: {image_path}")
            skipped += 1
            continue

        image = _letterbox_bgr_image(image, imgsz)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        batch_images.append(tensor)

    if not batch_images:
        return None, skipped

    batch_tensor = torch.stack(batch_images, dim=0).to(device=torch_device, dtype=dtype, non_blocking=True)
    return batch_tensor, skipped


def apply_adabn_to_yolo_model(
    model: YOLO,
    *,
    image_dir: Path,
    imgsz: int,
    batch_size: int,
    device: Any,
    max_images: Optional[int] = None,
    sample_seed: Optional[int] = None,
    epochs: int = 1,
    max_stage: Optional[int] = None,
    reset_stats: bool = True,
    momentum_override: Any = "use_original",
    use_fp16: bool = False,
    debug: bool = False,
    debug_max_batches: int = 1,
    disable_tqdm: Optional[bool] = None,
) -> Dict[str, Any]:
    image_dir = Path(image_dir).expanduser().resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"AdaBN image directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"AdaBN image path is not a directory: {image_dir}")
    if batch_size <= 0:
        raise ValueError("AdaBN batch_size must be at least 1.")
    if imgsz <= 0:
        raise ValueError("AdaBN imgsz must be at least 1.")
    if epochs <= 0:
        raise ValueError("AdaBN epochs must be at least 1.")
    if max_stage is not None and int(max_stage) < 0:
        raise ValueError("AdaBN max_stage must be non-negative.")

    base_model = model.model
    all_bn_layers = [(name, module) for name, module in base_model.named_modules() if isinstance(module, nn.BatchNorm2d)]
    if max_stage is None:
        bn_layers = all_bn_layers
    else:
        bn_layers = []
        for layer_name, module in all_bn_layers:
            stage_index = _extract_top_level_stage_index(layer_name)
            if stage_index is not None and stage_index <= int(max_stage):
                bn_layers.append((layer_name, module))
    bn_layer_names = [name for name, _ in bn_layers]
    all_bn_layer_names = [name for name, _ in all_bn_layers]
    selected_stage_indices = sorted(
        {
            stage_index
            for stage_index in (_extract_top_level_stage_index(name) for name, _ in bn_layers)
            if stage_index is not None
        }
    )

    metadata: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "image_dir": str(image_dir),
        "imgsz": int(imgsz),
        "batch_size": int(batch_size),
        "max_images": int(max_images) if max_images is not None else None,
        "sample_seed": sample_seed,
        "sampling_method": "random_without_replacement" if sample_seed is not None else "filename_prefix",
        "selected_image_names": [],
        "epochs": int(epochs),
        "max_stage": int(max_stage) if max_stage is not None else None,
        "reset_stats": bool(reset_stats),
        "momentum_override": momentum_override,
        "fp16_requested": bool(use_fp16),
        "bn_layer_count": len(bn_layers),
        "bn_layer_count_total": len(all_bn_layers),
        "bn_layer_name_preview": bn_layer_names[:10],
        "bn_layer_name_preview_total": all_bn_layer_names[:10],
        "selected_stage_indices": selected_stage_indices,
        "num_candidate_images": 0,
        "num_images_used": 0,
        "num_batches": 0,
        "num_skipped_images": 0,
        "torch_device": None,
        "compute_dtype": None,
        "status": "initialized",
        "debug_enabled": bool(debug),
        "debug_max_batches": int(debug_max_batches),
        "debug": None,
        "tqdm_disabled": bool(disable_tqdm) if disable_tqdm is not None else None,
    }

    print("\n[AdaBN] ========================================================")
    print(f"[AdaBN] image_dir              : {image_dir}")
    print(f"[AdaBN] imgsz / batch         : {imgsz} / {batch_size}")
    print(f"[AdaBN] epochs / max_stage    : {epochs} / {max_stage}")
    print(f"[AdaBN] max_images / reset    : {max_images} / {reset_stats}")
    print(f"[AdaBN] momentum override     : {momentum_override}")
    print(f"[AdaBN] fp16 requested        : {use_fp16}")
    print(f"[AdaBN] BN layer count        : {len(bn_layers)} / {len(all_bn_layers)} selected")
    if bn_layer_names:
        preview = ", ".join(bn_layer_names[:5])
        suffix = " ..." if len(bn_layer_names) > 5 else ""
        print(f"[AdaBN] BN layer preview      : {preview}{suffix}")
        print(f"[AdaBN] BN stage selection    : {selected_stage_indices if selected_stage_indices else 'stage parse unavailable'}")
    else:
        print("[AdaBN] BN layer preview      : none")

    if not all_bn_layers:
        module_preview = []
        for index, (module_name, module) in enumerate(base_model.named_modules()):
            if index >= 20:
                break
            module_preview.append(f"{module_name or '<root>'}:{type(module).__name__}")
        metadata["module_preview_when_no_bn"] = module_preview
        metadata["ultralytics_version"] = getattr(ultralytics, "__version__", None)
        metadata["model_class_name"] = type(base_model).__name__
        print(f"[AdaBN] ultralytics version   : {metadata['ultralytics_version']}")
        print(f"[AdaBN] model class           : {metadata['model_class_name']}")
        if module_preview:
            print("[AdaBN] module preview        :")
            for item in module_preview:
                print(f"[AdaBN]   {item}")
        print("[AdaBN] No BatchNorm2d layers found; skipping AdaBN.")
        print("[AdaBN] ========================================================\n")
        metadata["status"] = "skipped_no_batchnorm"
        return metadata
    if not bn_layers:
        print("[AdaBN] No BatchNorm2d layers found in the selected stage range; skipping AdaBN.")
        print("[AdaBN] ========================================================\n")
        metadata["status"] = "skipped_no_selected_batchnorm"
        return metadata

    image_paths = list_depth1_images(image_dir, max_images=max_images, sample_seed=sample_seed)
    metadata["selected_image_names"] = [path.name for path in image_paths]
    metadata["num_candidate_images"] = len(image_paths)
    candidate_suffix = " (limited)" if max_images is not None else ""
    print(f"[AdaBN] candidate images      : {len(image_paths)} (depth 1{candidate_suffix})")
    if max_images is not None:
        print(f"[AdaBN] truncated images      : {len(image_paths)}")

    if not image_paths:
        raise RuntimeError(f"No images found directly in the AdaBN directory: {image_dir}")

    torch_device = _resolve_torch_device(device)
    metadata["torch_device"] = str(torch_device)
    print(f"[AdaBN] torch device          : {torch_device}")
    _set_current_cuda_device_if_needed(torch_device)

    parameter = next(base_model.parameters(), None)
    original_model_dtype = parameter.dtype if parameter is not None else torch.float32
    effective_use_fp16 = bool(use_fp16) and torch_device.type == "cuda"
    if bool(use_fp16) and torch_device.type != "cuda":
        print("[AdaBN] FP16 requires CUDA; using FP32.")
    compute_dtype = torch.float16 if effective_use_fp16 else original_model_dtype
    metadata["compute_dtype"] = str(compute_dtype)
    metadata["fp16_applied"] = bool(effective_use_fp16)
    print(f"[AdaBN] model dtype           : {original_model_dtype}")
    print(f"[AdaBN] compute dtype         : {compute_dtype}")

    before_snapshot = _snapshot_bn_stats(bn_layers) if debug else None
    debug_batches: List[Dict[str, Any]] = []
    original_bn_momentums = {bn_layer: bn_layer.momentum for _, bn_layer in bn_layers}
    metadata["original_bn_momentum_preview"] = [bn_layer.momentum for _, bn_layer in bn_layers[:5]]
    if momentum_override == "use_original":
        metadata["effective_bn_momentum"] = "use_original"
    else:
        metadata["effective_bn_momentum"] = momentum_override

    module_training_states = {module: module.training for module in base_model.modules()}
    try:
        base_model.to(device=torch_device, dtype=compute_dtype)
        base_model.eval()
        for _, bn_layer in bn_layers:
            if reset_stats and getattr(bn_layer, "track_running_stats", True):
                bn_layer.reset_running_stats()
            if momentum_override != "use_original":
                bn_layer.momentum = momentum_override
            bn_layer.train()

        batches_per_epoch = math.ceil(len(image_paths) / batch_size)
        progress_bar = tqdm(
            total=batches_per_epoch * int(epochs),
            desc="AdaBN",
            disable=bool(disable_tqdm) if disable_tqdm is not None else False,
        )

        global_batch_index = 0
        for epoch_index in range(int(epochs)):
            for start_index in range(0, len(image_paths), batch_size):
                global_batch_index += 1
                batch_paths = image_paths[start_index:start_index + batch_size]
                batch_tensor, skipped = _prepare_batch_tensor(
                    batch_paths,
                    imgsz=imgsz,
                    torch_device=torch_device,
                    dtype=compute_dtype,
                )
                metadata["num_skipped_images"] += int(skipped)
                if batch_tensor is None:
                    print(
                        f"[AdaBN][WARN] Skipping epoch {epoch_index + 1}, batch {global_batch_index}: "
                        "all images failed to load."
                    )
                    progress_bar.update(1)
                    continue

                if debug and global_batch_index <= max(0, int(debug_max_batches)):
                    batch_debug = _collect_bn_activation_debug(base_model, bn_layers, batch_tensor)
                    batch_debug["batch_index"] = int(global_batch_index)
                    batch_debug["epoch_index"] = int(epoch_index + 1)
                    batch_debug["image_paths"] = [str(path) for path in batch_paths]
                    debug_batches.append(batch_debug)
                else:
                    with torch.no_grad():
                        base_model(batch_tensor)

                metadata["num_images_used"] += int(batch_tensor.shape[0])
                metadata["num_batches"] += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    epoch=epoch_index + 1,
                    used=metadata["num_images_used"],
                    skipped=metadata["num_skipped_images"],
                )

        progress_bar.close()
    finally:
        if compute_dtype != original_model_dtype:
            base_model.to(dtype=original_model_dtype)
        for bn_layer, original_momentum in original_bn_momentums.items():
            bn_layer.momentum = original_momentum
        for module, was_training in module_training_states.items():
            module.train(was_training)
        base_model.eval()

    if metadata["num_images_used"] <= 0:
        raise RuntimeError("No valid images were processed by AdaBN; BatchNorm statistics could not be updated.")

    metadata["applied"] = True
    metadata["status"] = "applied"
    if debug and before_snapshot is not None:
        after_snapshot = _snapshot_bn_stats(bn_layers)
        bn_delta_debug = _compare_bn_snapshots(before_snapshot, after_snapshot)
        metadata["debug"] = {
            "bn_stats": bn_delta_debug,
            "activation_batches": debug_batches,
        }
        global_summary = bn_delta_debug["global_summary"]
        print(
            "[AdaBN][DEBUG] BN delta summary : "
            f"mean_delta_mean={global_summary['mean_of_mean_abs_delta_running_mean']}, "
            f"mean_delta_var={global_summary['mean_of_mean_abs_delta_running_var']}, "
            f"nonfinite_after_layers={global_summary['nonfinite_after_layers']}, "
            f"near_zero_var_layers={global_summary['near_zero_running_var_layers']}"
        )
        if debug_batches:
            print(
                "[AdaBN][DEBUG] activation summary : "
                f"recorded_batches={len(debug_batches)}, "
                f"first_batch_suspicious_layers={debug_batches[0]['suspicious_layer_count']}"
            )
    print(f"[AdaBN] used / skipped        : {metadata['num_images_used']} / {metadata['num_skipped_images']}")
    print(f"[AdaBN] processed batches     : {metadata['num_batches']}")
    print("[AdaBN] Updated running_mean/var only in the model held in memory.")
    print("[AdaBN] ========================================================\n")
    return metadata
