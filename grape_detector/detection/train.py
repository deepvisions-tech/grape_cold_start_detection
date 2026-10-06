from __future__ import annotations

import argparse
import gc
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_TRAIN_CONFIG_PATH = "configs/detection/config_for_train.yaml"


def _resolve_bootstrap_config_path() -> str:
    if __name__ != "__main__":
        return DEFAULT_TRAIN_CONFIG_PATH
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=DEFAULT_TRAIN_CONFIG_PATH)
    args, _ = parser.parse_known_args()
    return str(args.config)


_conf_path = _resolve_bootstrap_config_path()
if os.path.exists(_conf_path):
    with open(_conf_path, "r", encoding="utf-8") as _f:
        _cfg = yaml.safe_load(_f)
        if _cfg and "DEVICE" in _cfg and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(_cfg["DEVICE"])

import torch

from grape_detector.utils.config_loader import (
    as_bool, load_config, normalize_class_names, parse_class_ids,
)
from grape_detector.utils.checkpoints import cleanup_run_weights_after_export
from grape_detector.utils.gamma_correction import resolve_gamma_correction_config
from grape_detector.data.dataset_splitter import has_existing_split, split_dataset, validate_split_labels
from grape_detector.data.yolo_yaml import write_yolo_yaml


def _cleanup_cuda_runtime(note: str) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    print(
        f"[CUDA cleanup] {note} | "
        f"allocated={torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB | "
        f"reserved={torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB"
    )


def _jsonify_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonify_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify_value(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _jsonify_value(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _extract_validation_metrics(val_results: Any) -> Dict[str, Any] | None:
    if val_results is None:
        return None

    results_dict = getattr(val_results, "results_dict", None)
    if callable(results_dict):
        results_dict = results_dict()

    box = getattr(val_results, "box", None)
    box_metrics = None
    if box is not None:
        box_metrics = {
            "precision": _jsonify_value(getattr(box, "mp", None)),
            "recall": _jsonify_value(getattr(box, "mr", None)),
            "f1": _jsonify_value(getattr(box, "f1", None)),
            "map50": _jsonify_value(getattr(box, "map50", None)),
            "map75": _jsonify_value(getattr(box, "map75", None)),
            "map50_95": _jsonify_value(getattr(box, "map", None)),
            "per_class_precision": _jsonify_value(getattr(box, "p", None)),
            "per_class_recall": _jsonify_value(getattr(box, "r", None)),
            "per_class_ap50_95": _jsonify_value(getattr(box, "maps", None)),
        }

    return {
        "output_dir": str(getattr(val_results, "save_dir", "")) if getattr(val_results, "save_dir", None) else None,
        "fitness": _jsonify_value(getattr(val_results, "fitness", None)),
        "speed": _jsonify_value(getattr(val_results, "speed", None)),
        "num_images": _jsonify_value(getattr(val_results, "seen", None)),
        "results_dict": _jsonify_value(results_dict) if results_dict is not None else None,
        "box": box_metrics,
    }


def _parse_dataset_roots(config: Dict[str, Any]) -> List[Path]:
    raw_roots = config.get("DATASET_ROOTS")
    if raw_roots is not None:
        if not isinstance(raw_roots, list) or not raw_roots:
            raise ValueError("DATASET_ROOTS must be a non-empty list.")
        dataset_roots = [Path(str(root)).expanduser() for root in raw_roots]
    else:
        raw_root = config.get("DATASET_ROOT", None)
        if not raw_root:
            raise ValueError("Either DATASET_ROOT or DATASET_ROOTS must be set.")
        dataset_roots = [Path(str(raw_root)).expanduser()]

    for dataset_root in dataset_roots:
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    return dataset_roots


def _parse_saved_split_dirs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_specs = config.get("SAVED_SPLIT_DIRS")
    if raw_specs is None:
        legacy_dir = config.get("SAVED_SPLIT_DIR", None)
        if legacy_dir and os.path.exists(legacy_dir):
            return [{"dir": legacy_dir, "prob": 0.0}]
        return []

    if isinstance(raw_specs, str) and raw_specs.strip().lower() in {"", "none", "null"}:
        return []

    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("SAVED_SPLIT_DIRS must be a non-empty list.")

    parsed: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_specs):
        if isinstance(item, str):
            split_dir = item
            prob = 0.0
        elif isinstance(item, dict):
            split_dir = item.get("dir") or item.get("path")
            if not split_dir:
                raise ValueError(f"SAVED_SPLIT_DIRS[{idx}] requires dir or path.")
            prob = 0.0 if idx == 0 else float(item.get("prob", 0.0))
        else:
            raise ValueError(f"Invalid SAVED_SPLIT_DIRS[{idx}] entry: {type(item)}")

        parsed.append({"dir": str(split_dir), "prob": float(prob)})

    return parsed


def _resolve_split_root(
    config: Dict[str, Any],
    split_parent: Path,
    run_name: str,
    split_specs: List[Dict[str, Any]],
) -> Path:
    if split_specs:
        split_root = Path(split_specs[0]["dir"])
        if len(split_specs) == 1:
            print(f"[INFO] Using the base split from SAVED_SPLIT_DIRS: {split_root}")
        else:
            print(
                f"[INFO] Using on-the-fly mixing across splits: base={split_root}, alternates={len(split_specs) - 1}"
            )
        return split_root

    saved_split_dir = config.get("SAVED_SPLIT_DIR", None)
    if saved_split_dir and os.path.exists(saved_split_dir):
        split_root = Path(saved_split_dir)
        print(f"[INFO] Using the existing split from SAVED_SPLIT_DIR: {split_root}")
        return split_root

    return split_parent / "splits" / run_name


def _get_optional_int(config: Dict[str, Any], key: str, default: Optional[int] = None) -> Optional[int]:
    value = config.get(key, default)
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return default
    return int(value)


def _get_optional_float(config: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    value = config.get(key, default)
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return default
    return float(value)


def _get_optional_str(config: Dict[str, Any], key: str, default: Optional[str] = None) -> Optional[str]:
    value = config.get(key, default)
    if value is None:
        return default
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return default
    return text


def _get_adabn_momentum_value(config: Dict[str, Any], key: str) -> Any:
    sentinel = "use_original"
    raw_value = config.get(key, sentinel)
    if raw_value == sentinel:
        return sentinel
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        text = raw_value.strip().lower()
        if text in {"none", "null"}:
            return None
        if text in {"", "default", "keep", "original", "use_original"}:
            return sentinel
        return float(raw_value)
    return float(raw_value)


def _resolve_f1_optimize_target_classes(config: Dict[str, Any]) -> Optional[List[int]]:
    names = normalize_class_names(config.get("names"))
    configured_classes = parse_class_ids(
        config.get("F1_OPTIMIZE_TARGET_CLASSES", config.get("F1_OPTIMIZE_CLASSES")),
        names,
    )
    if configured_classes is not None:
        return configured_classes
    if len(names) > 1:
        return sorted(names.keys())
    return [int(config.get("F1_OPTIMIZE_TARGET_GT_CLASS", 0))]


def _resolve_requested_device(config: Dict[str, Any]) -> str:
    requested_device = str(config.get("DEVICE", "0")).strip()
    return requested_device or "0"


def _should_use_single_visible_gpu_local_index(
    requested_device: str,
    limit_visible_device: bool,
) -> bool:
    if not requested_device or requested_device.lower() == "cpu" or "," in requested_device:
        return False
    if limit_visible_device:
        return True

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return False
    visible_devices = visible_devices.strip()
    if not visible_devices or "," in visible_devices:
        return False
    return True


def _resolve_runtime_device_from_config(config: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    if not torch.cuda.is_available():
        return "cpu", {
            "requested_device": "cpu",
            "effective_device": "cpu",
            "limit_visible_device": as_bool(config.get("LIMIT_VISIBLE_DEVICE", False), default=False),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "warning": None,
        }

    requested_device = _resolve_requested_device(config)
    limit_visible_device = as_bool(config.get("LIMIT_VISIBLE_DEVICE", False), default=False)
    effective_device = requested_device
    warning = None

    if requested_device.lower() == "cpu":
        effective_device = "cpu"
    elif _should_use_single_visible_gpu_local_index(requested_device, limit_visible_device):
        effective_device = "0"
    elif limit_visible_device and "," in requested_device:
        warning = f"Skipping device remapping: LIMIT_VISIBLE_DEVICE=true requires a single device, but multiple devices were specified: {requested_device}"

    return effective_device, {
        "requested_device": requested_device,
        "effective_device": effective_device,
        "limit_visible_device": limit_visible_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "warning": warning,
    }


def _resolve_runtime_context(config: Dict[str, Any], device: str) -> Dict[str, Any]:
    names = config["names"]
    dataset_roots = _parse_dataset_roots(config)
    dataset_root = dataset_roots[0]
    run_name = config.get("RUN_NAME", "grape_detect_exp1")
    split_parent = Path(config.get("SPLIT_PARENT_DIR", dataset_root))
    split_specs = _parse_saved_split_dirs(config)
    split_root = _resolve_split_root(config, split_parent, run_name, split_specs)

    run_project_dir = Path(config.get("RUN_PROJECT_DIR", split_parent / "yolo_runs"))
    run_project_dir.mkdir(parents=True, exist_ok=True)
    run_run_dir = run_project_dir / run_name
    run_run_dir.mkdir(parents=True, exist_ok=True)

    final_best_dir = Path(config.get("FINAL_BEST_DIR", split_parent / "models_final"))
    final_best_dir.mkdir(parents=True, exist_ok=True)

    generated_yaml_path = Path(config.get("GENERATED_YAML_PATH", split_root / "dataset_split.yaml"))
    return {
        "names": names,
        "dataset_roots": dataset_roots,
        "dataset_root": dataset_root,
        "run_name": run_name,
        "split_parent": split_parent,
        "split_specs": split_specs,
        "split_root": split_root,
        "run_project_dir": run_project_dir,
        "run_run_dir": run_run_dir,
        "final_best_dir": final_best_dir,
        "generated_yaml_path": generated_yaml_path,
        "device": device,
    }


def _prepare_training_model_with_optional_adabn(
    config: Dict[str, Any],
    *,
    device: str,
    train_batch: int,
    train_imgsz: int,
    disable_tqdm: bool,
) -> tuple[YOLO | str, Dict[str, Any]]:
    adabn_info: Dict[str, Any] = {
        "enabled": as_bool(config.get("TRAIN_ADABN_ENABLED", False), default=False),
        "applied": False,
        "status": "disabled",
    }
    if not adabn_info["enabled"]:
        return str(config["DETECTION_MODEL_PATH"]), adabn_info

    adabn_image_dir = config.get("TRAIN_ADABN_IMAGE_DIR")
    if not adabn_image_dir:
        raise ValueError("TRAIN_ADABN_IMAGE_DIR is required when TRAIN_ADABN_ENABLED=true.")

    from ultralytics import YOLO
    from grape_detector.utils.adabn import apply_adabn_to_yolo_model

    model = YOLO(str(config["DETECTION_MODEL_PATH"]))
    adabn_info = apply_adabn_to_yolo_model(
        model=model,
        image_dir=Path(str(adabn_image_dir)),
        imgsz=_get_optional_int(config, "TRAIN_ADABN_IMGSZ", train_imgsz),
        batch_size=_get_optional_int(config, "TRAIN_ADABN_BATCH", train_batch),
        device=device,
        max_images=_get_optional_int(config, "TRAIN_ADABN_MAX_IMAGES"),
        epochs=_get_optional_int(config, "TRAIN_ADABN_EPOCHS", 1),
        max_stage=_get_optional_int(config, "TRAIN_ADABN_MAX_STAGE"),
        reset_stats=as_bool(config.get("TRAIN_ADABN_RESET_STATS", True), default=True),
        momentum_override=_get_adabn_momentum_value(config, "TRAIN_ADABN_MOMENTUM"),
        use_fp16=as_bool(config.get("TRAIN_ADABN_FP16", False), default=False),
        debug=as_bool(config.get("TRAIN_ADABN_DEBUG", False), default=False),
        debug_max_batches=_get_optional_int(config, "TRAIN_ADABN_DEBUG_MAX_BATCHES", 1),
        disable_tqdm=disable_tqdm,
    )
    print(
        "[TRAIN_ADABN] "
        f"applied={adabn_info.get('applied', False)} | "
        f"images={adabn_info.get('num_images_used')} | "
        f"batches={adabn_info.get('num_batches')} | "
        f"bn_layers={adabn_info.get('bn_layer_count')}/{adabn_info.get('bn_layer_count_total')}"
    )
    return model, adabn_info


def _run_post_train_steps(
    config: Dict[str, Any],
    *,
    model: YOLO,
    run_dir: Path,
    run_name: str,
    split_root: Path,
    generated_yaml_path: Path,
    run_project_dir: Path,
    final_best_dir: Path,
    device: str,
) -> Dict[str, Any]:
    checkpoint_source = str(config.get("MODEL_EXPORT_WEIGHT_SOURCE", "best")).strip().lower()
    if checkpoint_source not in {"best", "last"}:
        raise ValueError("MODEL_EXPORT_WEIGHT_SOURCE must be 'best' or 'last'.")

    from grape_detector.utils.f1_conf_optimizer import optimize_conf_for_validation
    from grape_detector.detection.yolo_trainer import (
        export_checkpoint,
        validate_yolo,
    )

    disable_tqdm = as_bool(config.get("TQDM_DISABLE", False), default=False)
    gamma_config = resolve_gamma_correction_config(config, prefix="TRAIN")
    val_results = validate_yolo(
        model=model,
        data_yaml=str(generated_yaml_path),
        run_project_dir=run_project_dir,
        run_name=run_name,
        disable_tqdm=disable_tqdm,
        gamma_correction_enabled=gamma_config["enabled"],
        gamma_correction_value=gamma_config["gamma"],
    )
    validation_metrics = _extract_validation_metrics(val_results)

    exported_model_path = export_checkpoint(
        run_dir=run_dir,
        final_best_dir=final_best_dir,
        run_name=run_name,
        checkpoint_name=checkpoint_source,
    )
    tuned_conf = None
    if exported_model_path:
        print(f"[{checkpoint_source.upper()} saved] {exported_model_path}")
        should_optimize_conf = as_bool(
            config.get("F1_OPTIMIZE_CONF_AFTER_TRAIN", config.get("F1_OPTIMIZE_CONF_ON_VAL", True)),
            default=True,
        )
        if should_optimize_conf:
            optimize_split_name = str(config.get("F1_OPTIMIZE_SPLIT", "val")).strip()
            optimize_target_classes = _resolve_f1_optimize_target_classes(config)
            tuned_conf = optimize_conf_for_validation(
                model_path=exported_model_path,
                split_dir=split_root,
                split_name=optimize_split_name,
                imgsz=int(config.get("F1_OPTIMIZE_IMGSZ", config.get("IMGSZ", 960))),
                pred_iou=float(config.get("F1_OPTIMIZE_PRED_IOU", 0.7)),
                max_det=int(config.get("F1_OPTIMIZE_MAX_DET", 300)),
                device=str(config.get("F1_OPTIMIZE_DEVICE", device)),
                batch_size=int(config.get("F1_OPTIMIZE_BATCH", config.get("BATCH", 8))),
                eval_iou_thr=float(
                    config.get("F1_OPTIMIZE_EVAL_IOU_THRESHOLD", config.get("EVAL_IOU_THRESHOLD", 0.5))
                ),
                target_gt_class=int(config.get("F1_OPTIMIZE_TARGET_GT_CLASS", 0)),
                target_classes=optimize_target_classes,
                conf_min=float(config.get("F1_OPTIMIZE_CONF_MIN", 0.01)),
                conf_max=float(config.get("F1_OPTIMIZE_CONF_MAX", 0.99)),
                conf_step=float(config.get("F1_OPTIMIZE_CONF_STEP", 0.01)),
                save_metadata=True,
                gamma_correction_enabled=gamma_config["enabled"],
                gamma_correction_value=gamma_config["gamma"],
            )
            print(
                f"[F1-optimized confidence saved] split={optimize_split_name} | "
                f"conf={float(tuned_conf['best_conf']):.4f} | "
                f"path={tuned_conf.get('metadata_path', 'N/A')}"
            )
    else:
        print(f"[WARN] {checkpoint_source}.pt not found: {run_dir / 'weights' / f'{checkpoint_source}.pt'}")

    run_weights_cleanup = None
    if exported_model_path and as_bool(config.get("CLEAN_RUN_WEIGHTS_AFTER_EXPORT", False), default=False):
        run_weights_cleanup = cleanup_run_weights_after_export(
            run_dir=run_dir,
            final_model_path=exported_model_path,
            run_name=run_name,
        )

    return {
        "best_model_path": str(exported_model_path) if exported_model_path else None,
        "exported_model_path": str(exported_model_path) if exported_model_path else None,
        "model_weight_source": checkpoint_source,
        "val_results": str(val_results) if val_results is not None else None,
        "validation_metrics": validation_metrics,
        "tuned_conf": tuned_conf,
        "run_weights_cleanup": run_weights_cleanup,
    }


def run_train_with_config(config: Dict[str, Any]) -> Dict[str, Any]:
    from grape_detector.detection.yolo_trainer import train_yolo

    device, device_info = _resolve_runtime_device_from_config(config)
    print(
        "[TRAIN_DEVICE] "
        f"requested={device_info['requested_device']} | "
        f"effective={device_info['effective_device']} | "
        f"CUDA_VISIBLE_DEVICES={device_info['cuda_visible_devices']}"
    )
    if device_info.get("warning"):
        print(f"[TRAIN_DEVICE][WARN] {device_info['warning']}")
    ctx = _resolve_runtime_context(config, device)
    names = ctx["names"]
    dataset_roots = ctx["dataset_roots"]
    dataset_root = ctx["dataset_root"]
    run_name = ctx["run_name"]
    split_specs = ctx["split_specs"]
    split_root = ctx["split_root"]
    run_project_dir = ctx["run_project_dir"]
    final_best_dir = ctx["final_best_dir"]

    if not split_specs:
        if has_existing_split(split_root):
            print(f"[Split skipped] Split data already exists: {split_root}")
        else:
            if len(dataset_roots) == 1:
                print(f"[Split start] {dataset_root} -> {split_root}")
            else:
                print(f"[Split start] {len(dataset_roots)} dataset roots -> {split_root}")
            split_dataset(
                dataset_roots=dataset_roots,
                split_root=split_root,
                train_ratio=float(config.get("TRAIN_RATIO", 0.8)),
                val_ratio=float(config.get("VAL_RATIO", 0.1)),
                test_ratio=float(config.get("TEST_RATIO", 0.1)),
                seed=int(config.get("SPLIT_SEED", 42)),
                use_symlink=bool(config.get("USE_SYMLINK", True)),
            )

    validate_split_labels(split_root)
    generated_yaml_path = ctx["generated_yaml_path"]
    write_yolo_yaml(generated_yaml_path, split_root, names)

    train_seed = int(config.get("TRAIN_SEED", 0))
    train_deterministic = as_bool(config.get("TRAIN_DETERMINISTIC", True), default=True)
    disable_tqdm = as_bool(config.get("TQDM_DISABLE", False), default=False)
    gamma_config = resolve_gamma_correction_config(config, prefix="TRAIN")
    train_imgsz = int(config.get("IMGSZ", 640))
    train_batch = int(config.get("BATCH", 8))
    train_model, train_adabn_info = _prepare_training_model_with_optional_adabn(
        config,
        device=device,
        train_batch=train_batch,
        train_imgsz=train_imgsz,
        disable_tqdm=disable_tqdm,
    )

    model, run_dir, train_results = train_yolo(
        model_path=train_model,
        data_yaml=str(generated_yaml_path),
        device=device,
        epochs=int(config.get("EPOCHS", 75)),
        imgsz=train_imgsz,
        batch=train_batch,
        run_project_dir=run_project_dir,
        run_name=run_name,
        train_seed=train_seed,
        train_deterministic=train_deterministic,
        save_period=int(config.get("SAVE_PERIOD", 10)),
        amp=bool(config.get("AMP", True)),
        cache=str(config.get("CACHE", "disk")),
        close_mosaic=int(config.get("CLOSE_MOSAIC", 15)),
        rect=bool(config.get("RECT", True)),
        workers=int(config.get("WORKERS", 8)),
        split_mix_specs=split_specs if len(split_specs) > 1 else None,
        lr0=_get_optional_float(config, "LR0"),
        optimizer=_get_optional_str(config, "OPTIMIZER"),
        disable_tqdm=disable_tqdm,
        gamma_correction_enabled=gamma_config["enabled"],
        gamma_correction_value=gamma_config["gamma"],
    )

    post_train_result = _run_post_train_steps(
        config,
        model=model,
        run_dir=run_dir,
        run_name=run_name,
        split_root=split_root,
        generated_yaml_path=generated_yaml_path,
        run_project_dir=run_project_dir,
        final_best_dir=final_best_dir,
        device=device,
    )

    response = {
        "requested_device": device_info["requested_device"],
        "device": device,
        "device_info": device_info,
        "run_name": run_name,
        "split_root": str(split_root),
        "generated_yaml_path": str(generated_yaml_path),
        "run_project_dir": str(run_project_dir),
        "run_dir": str(run_dir),
        "final_best_dir": str(final_best_dir),
        "best_model_path": post_train_result["best_model_path"],
        "train_seed": train_seed,
        "train_deterministic": train_deterministic,
        "train_results": str(train_results) if train_results is not None else None,
        "gamma_correction": {
            "enabled": bool(gamma_config["enabled"]),
            "gamma": float(gamma_config["gamma"]),
        },
        "train_adabn": train_adabn_info,
        "val_results": post_train_result["val_results"],
        "validation_metrics": post_train_result["validation_metrics"],
        "tuned_conf": post_train_result["tuned_conf"],
        "run_weights_cleanup": post_train_result["run_weights_cleanup"],
    }
    del model
    del train_results
    del post_train_result
    _cleanup_cuda_runtime("run_train_with_config")
    return response


def run_post_train_only_with_config(config: Dict[str, Any]) -> Dict[str, Any]:
    device, device_info = _resolve_runtime_device_from_config(config)
    print(
        "[TRAIN_DEVICE] "
        f"requested={device_info['requested_device']} | "
        f"effective={device_info['effective_device']} | "
        f"CUDA_VISIBLE_DEVICES={device_info['cuda_visible_devices']}"
    )
    if device_info.get("warning"):
        print(f"[TRAIN_DEVICE][WARN] {device_info['warning']}")
    ctx = _resolve_runtime_context(config, device)
    names = ctx["names"]
    run_name = ctx["run_name"]
    split_root = ctx["split_root"]
    run_project_dir = ctx["run_project_dir"]
    final_best_dir = ctx["final_best_dir"]
    generated_yaml_path = ctx["generated_yaml_path"]
    run_dir = ctx["run_run_dir"]
    checkpoint_source = str(config.get("MODEL_EXPORT_WEIGHT_SOURCE", "best")).strip().lower()
    if checkpoint_source not in {"best", "last"}:
        raise ValueError("MODEL_EXPORT_WEIGHT_SOURCE must be 'best' or 'last'.")
    selected_weight_path = run_dir / "weights" / f"{checkpoint_source}.pt"

    if not selected_weight_path.exists():
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_source}.pt required for post_train_only was not found: {selected_weight_path}"
        )

    validate_split_labels(split_root)
    write_yolo_yaml(generated_yaml_path, split_root, names)
    from ultralytics import YOLO

    model = YOLO(str(selected_weight_path))
    gamma_config = resolve_gamma_correction_config(config, prefix="TRAIN")
    post_train_result = _run_post_train_steps(
        config,
        model=model,
        run_dir=run_dir,
        run_name=run_name,
        split_root=split_root,
        generated_yaml_path=generated_yaml_path,
        run_project_dir=run_project_dir,
        final_best_dir=final_best_dir,
        device=device,
    )

    response = {
        "requested_device": device_info["requested_device"],
        "device": device,
        "device_info": device_info,
        "run_name": run_name,
        "split_root": str(split_root),
        "generated_yaml_path": str(generated_yaml_path),
        "run_project_dir": str(run_project_dir),
        "run_dir": str(run_dir),
        "final_best_dir": str(final_best_dir),
        "best_model_path": post_train_result["best_model_path"] or str(selected_weight_path),
        "exported_model_path": post_train_result["exported_model_path"] or str(selected_weight_path),
        "model_weight_source": post_train_result["model_weight_source"],
        "train_seed": int(config.get("TRAIN_SEED", 0)),
        "train_deterministic": as_bool(config.get("TRAIN_DETERMINISTIC", True), default=True),
        "train_results": None,
        "gamma_correction": {
            "enabled": bool(gamma_config["enabled"]),
            "gamma": float(gamma_config["gamma"]),
        },
        "val_results": post_train_result["val_results"],
        "validation_metrics": post_train_result["validation_metrics"],
        "tuned_conf": post_train_result["tuned_conf"],
        "run_weights_cleanup": post_train_result["run_weights_cleanup"],
        "mode": f"reused_existing_run_weights_{checkpoint_source}",
    }
    del model
    del post_train_result
    _cleanup_cuda_runtime("run_post_train_only_with_config")
    return response


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grapevine detection train runner")
    parser.add_argument(
        "--config",
        default=DEFAULT_TRAIN_CONFIG_PATH,
        help="Path to a YAML config file.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Reserved Memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

    config = load_config(args.config)
    run_train_with_config(config)


if __name__ == "__main__":
    main()
