from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import os
import random
import re
import shutil
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

import yaml

from grape_detector.data.dataset_splitter import has_existing_split, split_dataset
from grape_detector.data.yolo_yaml import write_yolo_yaml
from grape_detector.utils.adabn_grid import apply_adabn_grid_variant, build_adabn_grid_variants
from grape_detector.utils.config_loader import as_bool, load_config
from grape_detector.utils.checkpoints import cleanup_run_weights_after_export
from grape_detector.utils.run_path_utils import ensure_name_suffix, normalize_optional_text


DEFAULT_SEED_SWEEP_CONFIG_PATH = "configs/detection/config_for_train_test_seed_sweep.yaml"


def _resolve_path(path_value: Any, base_dir: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()

    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (base_dir / path).resolve()


def _build_named_output_root(output_base: Path, run_name_prefix: str) -> Path:
    return output_base if output_base.name == run_name_prefix else output_base / run_name_prefix


def _preconfigure_seed_sweep_process_device(sweep_cfg: Dict[str, Any]) -> Dict[str, Any]:
    requested_device = normalize_optional_text(sweep_cfg.get("SEED_SWEEP_DEVICE"))
    info = {
        "requested_device": requested_device,
        "enabled": requested_device is not None,
        "applied": False,
        "logical_device": None,
        "cuda_visible_devices_before": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_visible_devices_after": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "warning": None,
    }

    if requested_device is None:
        return info

    if requested_device.lower() == "cpu":
        info["applied"] = True
        info["logical_device"] = "cpu"
        return info

    if "," in requested_device:
        info["warning"] = (
            "SEED_SWEEP_DEVICE supports only a single GPU index or cpu. "
            f"Current value={requested_device}"
        )
        return info

    os.environ["CUDA_VISIBLE_DEVICES"] = requested_device
    info["cuda_visible_devices_after"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    info["applied"] = True
    info["logical_device"] = "0"
    return info


def _apply_seed_sweep_device_override(
    train_cfg: Dict[str, Any],
    test_cfg: Dict[str, Any],
    process_device_info: Dict[str, Any],
) -> None:
    if not process_device_info.get("applied", False):
        return

    logical_device = process_device_info.get("logical_device")
    if logical_device is None:
        return

    train_cfg["DEVICE"] = logical_device
    test_cfg["TEST_DEVICE"] = logical_device
    test_cfg["TEST_LIMIT_VISIBLE_DEVICE"] = False
    test_cfg["_SEED_SWEEP_DEVICE_OVERRIDE"] = process_device_info.get("requested_device")
    if "TEST_TUNE_DEVICE" in test_cfg:
        test_cfg["TEST_TUNE_DEVICE"] = logical_device


def _preconfigure_test_only_visibility(test_cfg: Dict[str, Any], sweep_mode: str) -> Dict[str, Any]:
    info = {
        "enabled": False,
        "applied": False,
        "requested_device": str(test_cfg.get("TEST_DEVICE", "0")).strip(),
        "cuda_visible_devices_before": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_visible_devices_after": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "warning": None,
    }

    if sweep_mode != "test_only":
        return info

    if not as_bool(test_cfg.get("TEST_LIMIT_VISIBLE_DEVICE", False), default=False):
        return info

    requested_device = info["requested_device"]
    info["enabled"] = True

    if not requested_device or requested_device.lower() == "cpu":
        info["warning"] = "Skipping early GPU visibility configuration because TEST_DEVICE is cpu."
        return info

    if "," in requested_device:
        info["warning"] = (
            "TEST_LIMIT_VISIBLE_DEVICE requires a single GPU index. "
            f"Current TEST_DEVICE={requested_device}"
        )
        return info

    os.environ["CUDA_VISIBLE_DEVICES"] = requested_device
    info["cuda_visible_devices_after"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    info["applied"] = True
    return info


def _generate_train_seeds(cfg: Dict[str, Any]) -> List[int]:
    explicit = cfg.get("SEED_SWEEP_TRAIN_SEEDS")
    if explicit:
        if not isinstance(explicit, list):
            raise ValueError("SEED_SWEEP_TRAIN_SEEDS must be a list.")
        return [int(seed) for seed in explicit]

    num_runs = int(cfg.get("SEED_SWEEP_NUM_RUNS", 1))
    if num_runs <= 0:
        raise ValueError("SEED_SWEEP_NUM_RUNS must be at least 1.")

    master_seed = int(cfg.get("SEED_SWEEP_MASTER_SEED", 42))
    train_seed_min = int(cfg.get("SEED_SWEEP_TRAIN_SEED_MIN", 0))
    train_seed_max = int(cfg.get("SEED_SWEEP_TRAIN_SEED_MAX", 999999))
    if train_seed_max < train_seed_min:
        raise ValueError("SEED_SWEEP_TRAIN_SEED_MAX must be greater than or equal to SEED_SWEEP_TRAIN_SEED_MIN.")

    candidates = list(range(train_seed_min, train_seed_max + 1))
    if len(candidates) < num_runs:
        raise ValueError("SEED_SWEEP_NUM_RUNS exceeds the number of available seeds.")

    rng = random.Random(master_seed)
    return rng.sample(candidates, num_runs)


def _build_run_name(prefix: str, train_seed: int) -> str:
    prefix = str(prefix).strip()
    if prefix.lower().endswith("seed"):
        return f"{prefix}{train_seed}"
    return f"{prefix}_seed{train_seed}"


def _extract_seed_from_run_name(run_name: str, prefix: str) -> int | None:
    pattern = rf"^{re.escape(prefix)}(?:_seed)?(\d+)$"
    match = re.match(pattern, run_name)
    if not match:
        return None
    return int(match.group(1))


def _discover_existing_run_entries(
    *,
    output_root: Path,
    run_name_prefix: str,
    mode: str,
    explicit_seeds: List[int] | None,
    model_weight_source: str,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    if mode == "post_train_only":
        search_root = output_root / "train_runs"
        if not search_root.exists():
            raise FileNotFoundError(f"train_runs directory not found: {search_root}")
        for child in sorted(search_root.iterdir()):
            if not child.is_dir():
                continue
            if child.name.endswith("_val_aug_tmp"):
                continue
            train_seed = _extract_seed_from_run_name(child.name, run_name_prefix)
            if train_seed is None:
                continue
            selected_weight = child / "weights" / f"{model_weight_source}.pt"
            if not selected_weight.exists():
                continue
            entries.append({"run_name": child.name, "train_seed": train_seed})
    elif mode == "test_only":
        search_root = output_root / "models_final"
        if not search_root.exists():
            raise FileNotFoundError(f"models_final directory not found: {search_root}")
        for model_path in sorted(search_root.glob(f"*_{model_weight_source}.pt")):
            suffix = f"_{model_weight_source}"
            run_name = model_path.stem[: -len(suffix)] if model_path.stem.endswith(suffix) else model_path.stem
            train_seed = _extract_seed_from_run_name(run_name, run_name_prefix)
            if train_seed is None:
                continue
            entries.append({"run_name": run_name, "train_seed": train_seed})
    else:
        raise ValueError(f"Unsupported discovery mode: {mode}")

    if explicit_seeds is not None:
        seed_set = {int(seed) for seed in explicit_seeds}
        entries = [entry for entry in entries if entry["train_seed"] in seed_set]

    unique_entries: Dict[int, Dict[str, Any]] = {}
    for entry in entries:
        unique_entries.setdefault(entry["train_seed"], entry)

    resolved_entries = [unique_entries[seed] for seed in sorted(unique_entries)]
    if not resolved_entries:
        raise RuntimeError(
            f"No existing runs found for mode={mode}, prefix={run_name_prefix}: {output_root}"
        )
    return resolved_entries


def _discover_existing_summary_entries(
    *,
    output_root: Path,
    run_name_prefix: str,
    explicit_seeds: List[int] | None,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for seed_root in sorted(output_root.glob("seed_*")):
        if not seed_root.is_dir():
            continue
        seed_text = seed_root.name[len("seed_") :]
        if not seed_text.isdigit():
            continue
        train_seed = int(seed_text)
        if explicit_seeds is not None and train_seed not in explicit_seeds:
            continue

        train_result = _read_json(seed_root / "train_result.json") or {}
        run_name = train_result.get("run_name")
        if not run_name:
            effective_train_cfg_path = seed_root / "effective_train_config.yaml"
            if effective_train_cfg_path.exists():
                effective_train_cfg = load_config(str(effective_train_cfg_path))
                run_name = effective_train_cfg.get("RUN_NAME")
        if not run_name:
            run_name = _build_run_name(run_name_prefix, train_seed)
        if _extract_seed_from_run_name(str(run_name), run_name_prefix) is None:
            continue
        entries.append({"run_name": str(run_name), "train_seed": train_seed})

    if not entries:
        raise RuntimeError(
            f"No existing seed results found to summarize in summary_only mode: {output_root}"
        )
    return entries


def _resolve_run_entries(
    sweep_cfg: Dict[str, Any],
    output_root: Path,
    train_run_name_prefix: str,
    test_run_name_prefix: str,
    mode: str,
    model_weight_source: str,
) -> List[Dict[str, Any]]:
    explicit_seeds_raw = sweep_cfg.get("SEED_SWEEP_TRAIN_SEEDS")
    explicit_seeds = None
    if explicit_seeds_raw:
        if not isinstance(explicit_seeds_raw, list):
            raise ValueError("SEED_SWEEP_TRAIN_SEEDS must be a list.")
        explicit_seeds = [int(seed) for seed in explicit_seeds_raw]

    if mode in {"full", "finetune", "resume"}:
        train_seeds = explicit_seeds if explicit_seeds is not None else _generate_train_seeds(sweep_cfg)
        entries = [
            {
                "run_name": _build_run_name(train_run_name_prefix, train_seed),
                "train_seed": int(train_seed),
            }
            for train_seed in train_seeds
        ]
        return _attach_test_run_names(entries, test_run_name_prefix)

    if mode == "summary_only":
        entries = _discover_existing_summary_entries(
            output_root=output_root,
            run_name_prefix=train_run_name_prefix,
            explicit_seeds=explicit_seeds,
        )
        return _attach_test_run_names(entries, test_run_name_prefix)

    entries = _discover_existing_run_entries(
        output_root=output_root,
        run_name_prefix=train_run_name_prefix,
        mode=mode,
        explicit_seeds=explicit_seeds,
        model_weight_source=model_weight_source,
    )
    return _attach_test_run_names(entries, test_run_name_prefix)


def _resolve_finetune_source_run_dir(sweep_cfg: Dict[str, Any], config_dir: Path) -> Path:
    source_run_dir = normalize_optional_text(sweep_cfg.get("SEED_SWEEP_FINETUNE_RUN_DIR"))
    if source_run_dir is None:
        raise ValueError("SEED_SWEEP_FINETUNE_RUN_DIR is required in finetune mode.")

    resolved = _resolve_path(source_run_dir, config_dir)
    if not resolved.exists():
        raise FileNotFoundError(f"Fine-tuning source run directory not found: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Fine-tuning source run path is not a directory: {resolved}")
    return resolved


def _resolve_finetune_source_model_paths(
    *,
    source_run_dir: Path,
    run_entries: List[Dict[str, Any]],
) -> Dict[int, Path]:
    models_root = source_run_dir / "models_final"
    if not models_root.exists():
        raise FileNotFoundError(f"finetune source models_final directory not found: {models_root}")

    source_run_name_prefix = source_run_dir.name
    resolved_paths: Dict[int, Path] = {}
    missing_paths: List[Dict[str, Any]] = []
    for entry in run_entries:
        train_seed = int(entry["train_seed"])
        source_run_name = _build_run_name(source_run_name_prefix, train_seed)
        source_model_path = models_root / f"{source_run_name}_best.pt"
        if source_model_path.exists():
            resolved_paths[train_seed] = source_model_path.resolve()
        else:
            missing_paths.append(
                {
                    "train_seed": train_seed,
                    "expected_model_path": str(source_model_path),
                }
            )

    if missing_paths:
        missing_summary = "\n".join(
            f"  seed={item['train_seed']}: {item['expected_model_path']}"
            for item in missing_paths
        )
        raise FileNotFoundError(
            "Best checkpoints are missing from the fine-tuning source. "
            "The new seed sweep must use the same seeds as the source seed sweep.\n"
            f"{missing_summary}"
        )

    return resolved_paths


def _attach_test_run_names(
    entries: List[Dict[str, Any]],
    test_run_name_prefix: str,
) -> List[Dict[str, Any]]:
    resolved: List[Dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)
        train_seed = int(item["train_seed"])
        train_run_name = str(item["run_name"])
        test_run_name = normalize_optional_text(item.get("test_run_name"))
        if test_run_name is None:
            test_run_name = _build_run_name(test_run_name_prefix, train_seed)
        item["train_run_name"] = train_run_name
        item["test_run_name"] = test_run_name
        resolved.append(item)
    return resolved


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else None


def _resolve_output_roots(
    sweep_cfg: Dict[str, Any],
    *,
    config_dir: Path,
    train_run_name_prefix: str,
) -> Dict[str, Any]:
    final_output_base = _resolve_path(sweep_cfg["SEED_SWEEP_OUTPUT_DIR"], config_dir)
    final_output_root = _build_named_output_root(final_output_base, train_run_name_prefix)
    local_staging_enabled = as_bool(
        sweep_cfg.get("SEED_SWEEP_LOCAL_STAGING_ENABLED", False),
        default=False,
    )

    if not local_staging_enabled:
        return {
            "local_staging_enabled": False,
            "output_base": final_output_base,
            "output_root": final_output_root,
            "final_output_base": final_output_base,
            "final_output_root": final_output_root,
            "local_staging_base": None,
            "local_staging_root": None,
        }

    local_staging_dir = normalize_optional_text(sweep_cfg.get("SEED_SWEEP_LOCAL_STAGING_DIR"))
    if local_staging_dir is None:
        raise ValueError(
            "When SEED_SWEEP_LOCAL_STAGING_ENABLED=true, "
            "SEED_SWEEP_LOCAL_STAGING_DIR must be set."
        )

    local_staging_base = _resolve_path(local_staging_dir, config_dir)
    local_staging_root = _build_named_output_root(local_staging_base, train_run_name_prefix)
    final_root_resolved = final_output_root.resolve()
    local_root_resolved = local_staging_root.resolve()
    if final_root_resolved == local_root_resolved:
        raise ValueError(
            "SEED_SWEEP_LOCAL_STAGING_DIR and the final SEED_SWEEP_OUTPUT_DIR point to the same run root: "
            f"{local_staging_root}"
        )
    if (
        local_root_resolved.is_relative_to(final_root_resolved)
        or final_root_resolved.is_relative_to(local_root_resolved)
    ):
        raise ValueError(
            "The local staging run root and final output run root must not be nested inside one another. "
            f"local={local_staging_root}, final={final_output_root}"
        )

    return {
        "local_staging_enabled": True,
        "output_base": local_staging_base,
        "output_root": local_staging_root,
        "final_output_base": final_output_base,
        "final_output_root": final_output_root,
        "local_staging_base": local_staging_base,
        "local_staging_root": local_staging_root,
    }


def _remove_destination_for_copy(dst: Path) -> None:
    if dst.is_dir() and not dst.is_symlink():
        shutil.rmtree(dst)
    elif dst.exists() or dst.is_symlink():
        dst.unlink()


def _copy_path_for_staging_sync(
    src: Path,
    dst: Path,
    *,
    overwrite: bool,
    stats: Dict[str, int],
) -> None:
    if src.is_symlink():
        if dst.exists() or dst.is_symlink():
            if not overwrite:
                stats["skipped_existing"] += 1
                return
            _remove_destination_for_copy(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.readlink(src), dst)
        stats["symlinks"] += 1
        return

    if src.is_dir():
        if dst.exists() or dst.is_symlink():
            if not dst.is_dir() or dst.is_symlink():
                if not overwrite:
                    stats["skipped_existing"] += 1
                    return
                _remove_destination_for_copy(dst)
                dst.mkdir(parents=True, exist_ok=True)
                stats["dirs"] += 1
        else:
            dst.mkdir(parents=True, exist_ok=True)
            stats["dirs"] += 1

        for child in sorted(src.iterdir()):
            _copy_path_for_staging_sync(
                child,
                dst / child.name,
                overwrite=overwrite,
                stats=stats,
            )
        return

    if dst.exists() or dst.is_symlink():
        if not overwrite:
            stats["skipped_existing"] += 1
            return
        _remove_destination_for_copy(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    stats["files"] += 1


def _copy_tree_for_staging_sync(
    *,
    src_root: Path,
    dst_root: Path,
    overwrite: bool,
) -> Dict[str, int | str]:
    src_root = Path(src_root).expanduser()
    dst_root = Path(dst_root).expanduser()
    if not src_root.exists():
        raise FileNotFoundError(f"Staging source to copy was not found: {src_root}")

    src_resolved = src_root.resolve()
    dst_resolved = dst_root.resolve()
    if src_resolved == dst_resolved:
        return {
            "status": "skipped_same_path",
            "files": 0,
            "dirs": 0,
            "symlinks": 0,
            "skipped_existing": 0,
        }
    if dst_resolved.is_relative_to(src_resolved):
        raise ValueError(
            f"Staging sync destination is inside the source: src={src_root}, dst={dst_root}"
        )

    stats: Dict[str, int] = {
        "files": 0,
        "dirs": 0,
        "symlinks": 0,
        "skipped_existing": 0,
    }
    dst_root.mkdir(parents=True, exist_ok=True)
    for child in sorted(src_root.iterdir()):
        _copy_path_for_staging_sync(
            child,
            dst_root / child.name,
            overwrite=overwrite,
            stats=stats,
        )

    return {
        "status": "completed",
        **stats,
    }


def _cleanup_local_staging_root_after_sync(
    *,
    local_root: Path,
    final_root: Path,
) -> Dict[str, str]:
    local_root = Path(local_root).expanduser()
    final_root = Path(final_root).expanduser()
    if not local_root.exists() and not local_root.is_symlink():
        return {
            "status": "skipped_missing",
            "local_root": str(local_root),
            "final_root": str(final_root),
        }

    local_resolved = local_root.resolve()
    final_resolved = final_root.resolve()
    if local_resolved == final_resolved:
        raise ValueError(f"The local staging cleanup path is the same as the final output path: {local_root}")
    if (
        local_resolved.is_relative_to(final_resolved)
        or final_resolved.is_relative_to(local_resolved)
    ):
        raise ValueError(
            "The local staging cleanup path and final output path must not be nested inside one another. "
            f"local={local_root}, final={final_root}"
        )

    if local_root.is_dir() and not local_root.is_symlink():
        shutil.rmtree(local_root)
        status = "removed_directory"
    else:
        local_root.unlink()
        status = "removed_file_or_symlink"

    return {
        "status": status,
        "local_root": str(local_root),
        "final_root": str(final_root),
    }


def _as_float_scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and len(value) == 1:
        return _as_float_scalar(value[0])
    return None


def _safe_metric_str(value: Any) -> str:
    numeric = _as_float_scalar(value)
    if numeric is None:
        return "N/A"
    return f"{numeric:.4f}"


def _extract_validation_summary(validation_metrics: Dict[str, Any] | None) -> Dict[str, Any]:
    empty = {
        "validation_output_dir": None,
        "validation_num_images": None,
        "val_precision": None,
        "val_recall": None,
        "val_f1": None,
        "val_ap50": None,
        "val_mAP50": None,
        "val_mAP50_95": None,
    }
    if not validation_metrics:
        return empty

    box = validation_metrics.get("box", {}) if isinstance(validation_metrics.get("box"), dict) else {}
    results_dict = (
        validation_metrics.get("results_dict", {})
        if isinstance(validation_metrics.get("results_dict"), dict)
        else {}
    )
    precision = _as_float_scalar(box.get("precision"))
    recall = _as_float_scalar(box.get("recall"))
    f1 = _as_float_scalar(box.get("f1"))
    map50 = _as_float_scalar(box.get("map50"))
    map50_95 = _as_float_scalar(box.get("map50_95"))

    if precision is None:
        precision = _as_float_scalar(results_dict.get("metrics/precision(B)"))
    if recall is None:
        recall = _as_float_scalar(results_dict.get("metrics/recall(B)"))
    if map50 is None:
        map50 = _as_float_scalar(results_dict.get("metrics/mAP50(B)"))
    if map50_95 is None:
        map50_95 = _as_float_scalar(results_dict.get("metrics/mAP50-95(B)"))
    if f1 is None and precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "validation_output_dir": validation_metrics.get("output_dir"),
        "validation_num_images": _as_float_scalar(validation_metrics.get("num_images")),
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1,
        "val_ap50": map50,
        "val_mAP50": map50,
        "val_mAP50_95": map50_95,
    }


def _load_existing_validation_metrics(seed_root: Path) -> Dict[str, Any] | None:
    validation_metrics_payload = _read_json(seed_root / "validation_metrics.json")
    if validation_metrics_payload is not None:
        return validation_metrics_payload

    train_result_payload = _read_json(seed_root / "train_result.json")
    if train_result_payload is None:
        return None

    validation_metrics = train_result_payload.get("validation_metrics")
    return validation_metrics if isinstance(validation_metrics, dict) else None


def _parse_dataset_roots_for_split(train_cfg: Dict[str, Any]) -> List[Path]:
    raw_roots = train_cfg.get("DATASET_ROOTS")
    if raw_roots is not None:
        if not isinstance(raw_roots, list) or not raw_roots:
            raise ValueError("DATASET_ROOTS must be a non-empty list.")
        dataset_roots = [Path(str(root)).expanduser() for root in raw_roots]
    else:
        raw_root = train_cfg.get("DATASET_ROOT")
        if not raw_root:
            raise ValueError("Either DATASET_ROOT or DATASET_ROOTS must be set.")
        dataset_roots = [Path(str(raw_root)).expanduser()]

    for dataset_root in dataset_roots:
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    return dataset_roots


def _resolve_shared_split_root(
    sweep_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    *,
    output_root: Path,
    config_dir: Path,
    run_name_prefix: str,
) -> Path:
    explicit_shared_split_dir = sweep_cfg.get("SEED_SWEEP_SHARED_SPLIT_DIR")
    if explicit_shared_split_dir:
        return _resolve_path(explicit_shared_split_dir, config_dir)

    use_train_split_parent = as_bool(
        sweep_cfg.get("SEED_SWEEP_SHARED_SPLIT_USE_TRAIN_SPLIT_PARENT", False),
        default=False,
    )
    if use_train_split_parent:
        dataset_roots = _parse_dataset_roots_for_split(train_cfg)
        dataset_root = dataset_roots[0]
        split_parent = Path(str(train_cfg.get("SPLIT_PARENT_DIR", dataset_root))).expanduser().resolve()
        return split_parent / "splits" / f"{run_name_prefix}_shared"

    return output_root / "shared_split"


def _prepare_shared_split_if_needed(
    sweep_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    *,
    output_root: Path,
    config_dir: Path,
    mode: str,
    run_name_prefix: str,
) -> Dict[str, str] | None:
    reuse_shared_split = as_bool(sweep_cfg.get("SEED_SWEEP_REUSE_SHARED_SPLIT", True), default=True)
    if mode not in {"full", "finetune"} or not reuse_shared_split:
        return None

    if train_cfg.get("SAVED_SPLIT_DIRS") or train_cfg.get("SAVED_SPLIT_DIR"):
        print("[Seed Sweep shared split] Skipping shared split creation because the base training config defines SAVED_SPLIT_DIR(S).")
        return None

    shared_split_root = _resolve_shared_split_root(
        sweep_cfg,
        train_cfg,
        output_root=output_root,
        config_dir=config_dir,
        run_name_prefix=run_name_prefix,
    )
    shared_yaml_path = _resolve_path(
        sweep_cfg.get("SEED_SWEEP_SHARED_YAML_PATH", str(shared_split_root / "dataset_split.yaml")),
        config_dir,
    )

    dataset_roots = _parse_dataset_roots_for_split(train_cfg)
    if has_existing_split(shared_split_root):
        print(f"[Seed Sweep shared split reuse] {shared_split_root}")
    else:
        print(f"[Seed Sweep shared split creation] roots={dataset_roots} -> {shared_split_root}")
        split_dataset(
            dataset_roots=dataset_roots,
            split_root=shared_split_root,
            train_ratio=float(train_cfg.get("TRAIN_RATIO", 0.8)),
            val_ratio=float(train_cfg.get("VAL_RATIO", 0.1)),
            test_ratio=float(train_cfg.get("TEST_RATIO", 0.1)),
            seed=int(train_cfg.get("SPLIT_SEED", 42)),
            use_symlink=as_bool(train_cfg.get("USE_SYMLINK", True), default=True),
        )

    write_yolo_yaml(shared_yaml_path, shared_split_root, train_cfg["names"])
    print(f"[Seed Sweep shared YAML] {shared_yaml_path}")
    return {
        "split_root": str(shared_split_root),
        "yaml_path": str(shared_yaml_path),
    }


def _load_yaml_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    payload = load_config(str(path))
    return payload if isinstance(payload, dict) else None


def _resolve_resume_train_mode(sweep_cfg: Dict[str, Any], output_root: Path) -> str:
    requested_mode = normalize_optional_text(sweep_cfg.get("SEED_SWEEP_RESUME_TRAIN_MODE"))
    if requested_mode is not None:
        requested_mode = requested_mode.lower()
    if requested_mode in {"full", "finetune"}:
        return requested_mode
    if requested_mode not in {None, "auto"}:
        raise ValueError("SEED_SWEEP_RESUME_TRAIN_MODE must be 'auto', 'full', or 'finetune'.")

    snapshot_cfg = _load_yaml_if_exists(output_root / "sweep_config_snapshot.yaml") or {}
    snapshot_mode = str(snapshot_cfg.get("SEED_SWEEP_MODE", "")).strip().lower()
    if snapshot_mode in {"full", "finetune"}:
        return snapshot_mode

    if normalize_optional_text(sweep_cfg.get("SEED_SWEEP_FINETUNE_RUN_DIR")) is not None:
        return "finetune"
    return "full"


def _resume_best_model_path(models_root: Path, run_name: str) -> Path:
    return models_root / f"{run_name}_best.pt"


def _build_test_variants(base_test_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    grid_variants = build_adabn_grid_variants(base_test_cfg)
    variants: List[Dict[str, Any]] = []
    adabn_grid_keys = [
        "TEST_ADABN_EPOCHS",
        "TEST_ADABN_MAX_IMAGES",
        "TEST_ADABN_MAX_STAGE",
        "TEST_ADABN_MOMENTUM",
    ]
    has_explicit_list = any(isinstance(base_test_cfg.get(key), list) for key in adabn_grid_keys)
    include_baseline = (
        as_bool(base_test_cfg.get("TEST_ADABN_ENABLED", False), default=False)
        and has_explicit_list
    )

    if include_baseline:
        variants.append(
            {
                "variant_name": "baseline_no_adabn",
                "variant_label": "AdaBN off",
                "variant_type": "baseline",
                "is_baseline": True,
                "combo_fields": {
                    "adabn_epochs": None,
                    "adabn_max_images": None,
                    "adabn_max_stage": None,
                    "adabn_momentum": None,
                },
                "apply": lambda cfg: cfg.update({"TEST_ADABN_ENABLED": False}),
            }
        )

    if not grid_variants:
        return variants

    has_grid = any(variant.get("has_grid", False) for variant in grid_variants)
    for variant in grid_variants:
        combo_name = variant["combo_name"]
        if has_grid:
            variant_name = combo_name
            variant_label = combo_name
        else:
            variant_name = "adabn_default" if include_baseline else "default"
            variant_label = "AdaBN default" if include_baseline else "default"

        combo_fields = dict(variant.get("fields", {}))
        variants.append(
            {
                "variant_name": variant_name,
                "variant_label": variant_label,
                "variant_type": "adabn",
                "is_baseline": False,
                "combo_fields": combo_fields,
                "apply": lambda cfg, _variant=variant: apply_adabn_grid_variant(cfg, _variant),
            }
        )

    return variants


def _apply_adabn_sample_seed(test_cfg: Dict[str, Any], train_seed: int) -> None:
    """Resolve the requested per-model sample seed before persisting the config."""
    value = test_cfg.get("TEST_ADABN_SAMPLE_SEED")
    if isinstance(value, str) and value.strip().lower() == "train_seed":
        test_cfg["TEST_ADABN_SAMPLE_SEED"] = int(train_seed)


def _apply_test_variant(
    base_test_cfg: Dict[str, Any],
    variant: Dict[str, Any],
    *,
    test_output_dir: Path,
) -> Dict[str, Any]:
    cfg = deepcopy(base_test_cfg)
    cfg["TEST_OUTPUT_DIR"] = str(test_output_dir)
    cfg["_TEST_OUTPUT_DIR_RESOLVED"] = True
    cfg["TEST_RESET_OUTPUT_DIR"] = True
    cfg["_ADABN_GRID_COMBO_NAME"] = variant["variant_name"]
    cfg["_ADABN_GRID_VARIANT_LABEL"] = variant["variant_label"]
    cfg["_ADABN_GRID_VARIANT_TYPE"] = variant["variant_type"]
    cfg["_ADABN_GRID_IS_BASELINE"] = bool(variant["is_baseline"])
    for field_name, value in variant.get("combo_fields", {}).items():
        cfg[f"_{field_name.upper()}"] = value
    variant["apply"](cfg)
    return cfg


def _recover_variant_record(
    *,
    run_name: str,
    train_seed: int,
    train_result: Dict[str, Any],
    metrics_path: Path,
    config_path: Path | None,
) -> Dict[str, Any] | None:
    test_metrics = _read_json(metrics_path)
    if not isinstance(test_metrics, dict):
        return None

    cfg = _load_yaml_if_exists(config_path) if config_path is not None else None
    adabn_grid = test_metrics.get("adabn_grid", {}) if isinstance(test_metrics.get("adabn_grid"), dict) else {}
    variant_name = (
        adabn_grid.get("combo_name")
        or (cfg.get("_ADABN_GRID_COMBO_NAME") if isinstance(cfg, dict) else None)
        or metrics_path.parent.name
    )
    variant_label = (
        adabn_grid.get("variant_label")
        or (cfg.get("_ADABN_GRID_VARIANT_LABEL") if isinstance(cfg, dict) else None)
        or variant_name
    )
    is_baseline_variant = bool(
        adabn_grid.get("is_baseline_variant")
        if "is_baseline_variant" in adabn_grid
        else (cfg.get("_ADABN_GRID_IS_BASELINE") if isinstance(cfg, dict) else False)
    )

    def _cfg_or_none(field_key: str) -> Any:
        if isinstance(cfg, dict) and field_key in cfg:
            return cfg.get(field_key)
        return None

    adabn_epochs = _cfg_or_none("_ADABN_EPOCHS")
    adabn_max_images = _cfg_or_none("_ADABN_MAX_IMAGES")
    adabn_max_stage = _cfg_or_none("_ADABN_MAX_STAGE")
    adabn_momentum = _cfg_or_none("_ADABN_MOMENTUM")

    if adabn_epochs is None and isinstance(cfg, dict) and not is_baseline_variant:
        adabn_epochs = cfg.get("TEST_ADABN_EPOCHS")
        adabn_max_images = cfg.get("TEST_ADABN_MAX_IMAGES")
        adabn_max_stage = cfg.get("TEST_ADABN_MAX_STAGE")
        adabn_momentum = cfg.get("TEST_ADABN_MOMENTUM")

    configured_test_run_name = (
        (cfg.get("TEST_RUN_NAME") if isinstance(cfg, dict) else None)
        or test_metrics.get("test_run_name")
        or run_name
    )
    has_multiple_variants = metrics_path.parent.parent.name == "test_variants"
    test_run_name = (
        f"{configured_test_run_name}__{variant_name}"
        if has_multiple_variants
        else str(configured_test_run_name)
    )

    return {
        "run_name": run_name,
        "test_run_name": test_run_name,
        "test_variant_name": variant_name,
        "test_variant_label": variant_label,
        "is_baseline_variant": is_baseline_variant,
        "adabn_epochs": adabn_epochs,
        "adabn_max_images": adabn_max_images,
        "adabn_max_stage": adabn_max_stage,
        "adabn_momentum": adabn_momentum,
        "train_seed": int(train_seed),
        "train_result": train_result,
        "test_metrics": test_metrics,
    }


def _load_summary_only_results_for_seed(
    *,
    seed_root: Path,
    run_name: str,
    train_seed: int,
    expected_model_path: Path,
    model_weight_source: str,
) -> List[Dict[str, Any]]:
    existing_train_result = _read_json(seed_root / "train_result.json") or {}
    validation_metrics = _load_existing_validation_metrics(seed_root)
    train_result = {
        "run_name": run_name,
        "train_seed": int(train_seed),
        "best_model_path": str(existing_train_result.get("best_model_path") or expected_model_path),
        "model_weight_source": existing_train_result.get("model_weight_source", model_weight_source),
        "generated_yaml_path": existing_train_result.get("generated_yaml_path"),
        "split_root": existing_train_result.get("split_root"),
        "val_results": existing_train_result.get("val_results"),
        "validation_metrics": validation_metrics,
        "tuned_conf": existing_train_result.get("tuned_conf"),
        "mode": existing_train_result.get("mode", "summary_only_reused"),
    }

    recovered: List[Dict[str, Any]] = []
    variant_root = seed_root / "test_variants"
    if variant_root.exists():
        for variant_dir in sorted(variant_root.iterdir()):
            if not variant_dir.is_dir():
                continue
            record = _recover_variant_record(
                run_name=run_name,
                train_seed=train_seed,
                train_result=train_result,
                metrics_path=variant_dir / "test_metrics.json",
                config_path=variant_dir / "effective_test_config.yaml",
            )
            if record is not None:
                recovered.append(record)

    if recovered:
        return recovered

    single_record = _recover_variant_record(
        run_name=run_name,
        train_seed=train_seed,
        train_result=train_result,
        metrics_path=seed_root / "test_metrics.json",
        config_path=seed_root / "effective_test_config.yaml",
    )
    return [single_record] if single_record is not None else []


def _build_summary_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in results:
        test_metrics = item["test_metrics"]
        object_level = test_metrics.get("object_level", {})
        image_level = test_metrics.get("image_level", {})
        prediction_settings = test_metrics.get("prediction_settings", {})
        adabn_info = test_metrics.get("adabn", {}) if isinstance(test_metrics.get("adabn"), dict) else {}
        validation_summary = _extract_validation_summary(item["train_result"].get("validation_metrics"))
        rows.append(
            {
                "run_name": item["test_run_name"],
                "base_run_name": item["run_name"],
                "train_seed": item["train_seed"],
                "test_variant_name": item.get("test_variant_name"),
                "test_variant_label": item.get("test_variant_label"),
                "is_baseline_variant": item.get("is_baseline_variant"),
                "best_model_path": item["train_result"].get("best_model_path"),
                "model_weight_source": item["train_result"].get("model_weight_source", "best"),
                "test_output_dir": test_metrics.get("output_dir"),
                "validation_output_dir": validation_summary["validation_output_dir"],
                "conf": prediction_settings.get("conf"),
                "conf_source": prediction_settings.get("conf_source"),
                "adabn_epochs": item.get("adabn_epochs"),
                "adabn_max_images": item.get("adabn_max_images"),
                "adabn_max_stage": item.get("adabn_max_stage"),
                "adabn_momentum": item.get("adabn_momentum"),
                "adabn_enabled": adabn_info.get("enabled"),
                "adabn_applied": adabn_info.get("applied"),
                "adabn_image_dir": adabn_info.get("image_dir"),
                "adabn_num_images": adabn_info.get("num_images_used"),
                "adabn_num_batches": adabn_info.get("num_batches"),
                "adabn_bn_layers": adabn_info.get("bn_layer_count"),
                "precision": object_level.get("precision"),
                "recall": object_level.get("recall"),
                "f1": object_level.get("f1"),
                "ap50": object_level.get("ap50"),
                "mAP50": object_level.get("mAP50"),
                "mAP50_95": object_level.get("mAP50_95"),
                "image_accuracy": image_level.get("accuracy"),
                "num_images": test_metrics.get("num_images"),
                "validation_num_images": validation_summary["validation_num_images"],
                "val_precision": validation_summary["val_precision"],
                "val_recall": validation_summary["val_recall"],
                "val_f1": validation_summary["val_f1"],
                "val_ap50": validation_summary["val_ap50"],
                "val_mAP50": validation_summary["val_mAP50"],
                "val_mAP50_95": validation_summary["val_mAP50_95"],
            }
        )
    return rows


def _compute_aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_keys = [
        "precision",
        "recall",
        "f1",
        "ap50",
        "mAP50",
        "mAP50_95",
        "image_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_ap50",
        "val_mAP50",
        "val_mAP50_95",
    ]
    aggregate: Dict[str, Any] = {}
    for key in metric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            continue
        sorted_values = sorted(values)
        aggregate[key] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "gap": max(values) - min(values),
            "median": sorted_values[len(sorted_values) // 2] if len(sorted_values) % 2 == 1 else (
                sorted_values[len(sorted_values) // 2 - 1] + sorted_values[len(sorted_values) // 2]
            ) / 2.0,
        }

    if rows:
        best_f1_row = max(rows, key=lambda row: float(row.get("f1") or 0.0))
        worst_f1_row = min(rows, key=lambda row: float(row.get("f1") or 0.0))
        aggregate["best_run_by_f1"] = {
            "run_name": best_f1_row["run_name"],
            "train_seed": best_f1_row["train_seed"],
            "f1": best_f1_row.get("f1"),
            "mAP50": best_f1_row.get("mAP50"),
            "mAP50_95": best_f1_row.get("mAP50_95"),
            "test_output_dir": best_f1_row.get("test_output_dir"),
        }
        aggregate["worst_run_by_f1"] = {
            "run_name": worst_f1_row["run_name"],
            "train_seed": worst_f1_row["train_seed"],
            "f1": worst_f1_row.get("f1"),
            "mAP50": worst_f1_row.get("mAP50"),
            "mAP50_95": worst_f1_row.get("mAP50_95"),
            "test_output_dir": worst_f1_row.get("test_output_dir"),
        }
        val_f1_rows = [row for row in rows if row.get("val_f1") is not None]
        if val_f1_rows:
            best_val_f1_row = max(val_f1_rows, key=lambda row: float(row["val_f1"]))
            worst_val_f1_row = min(val_f1_rows, key=lambda row: float(row["val_f1"]))
            aggregate["best_run_by_val_f1"] = {
                "run_name": best_val_f1_row["run_name"],
                "train_seed": best_val_f1_row["train_seed"],
                "val_f1": best_val_f1_row.get("val_f1"),
                "val_mAP50": best_val_f1_row.get("val_mAP50"),
                "val_mAP50_95": best_val_f1_row.get("val_mAP50_95"),
                "validation_output_dir": best_val_f1_row.get("validation_output_dir"),
            }
            aggregate["worst_run_by_val_f1"] = {
                "run_name": worst_val_f1_row["run_name"],
                "train_seed": worst_val_f1_row["train_seed"],
                "val_f1": worst_val_f1_row.get("val_f1"),
                "val_mAP50": worst_val_f1_row.get("val_mAP50"),
                "val_mAP50_95": worst_val_f1_row.get("val_mAP50_95"),
                "validation_output_dir": worst_val_f1_row.get("validation_output_dir"),
            }
    return aggregate


def _compute_combo_comparison(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        combo_name = str(row.get("test_variant_name") or "default")
        grouped.setdefault(combo_name, []).append(row)

    combo_rows: List[Dict[str, Any]] = []
    baseline_row = None
    baseline_metrics = {}
    metric_keys = ["precision", "recall", "f1", "ap50", "mAP50", "mAP50_95", "image_accuracy"]

    for combo_name, combo_group_rows in grouped.items():
        example = combo_group_rows[0]
        row = {
            "test_variant_name": combo_name,
            "test_variant_label": example.get("test_variant_label"),
            "is_baseline_variant": bool(example.get("is_baseline_variant")),
            "num_runs": len(combo_group_rows),
            "num_seeds": len({item.get("train_seed") for item in combo_group_rows}),
            "adabn_epochs": example.get("adabn_epochs"),
            "adabn_max_images": example.get("adabn_max_images"),
            "adabn_max_stage": example.get("adabn_max_stage"),
            "adabn_momentum": example.get("adabn_momentum"),
        }
        for key in metric_keys:
            values = [float(item[key]) for item in combo_group_rows if item.get(key) is not None]
            row[f"{key}_mean"] = mean(values) if values else None
            row[f"{key}_std"] = pstdev(values) if len(values) > 1 else (0.0 if values else None)
            row[f"{key}_min"] = min(values) if values else None
            row[f"{key}_max"] = max(values) if values else None
        combo_rows.append(row)
        if row["is_baseline_variant"] and baseline_row is None:
            baseline_row = row
            baseline_metrics = {
                key: row.get(f"{key}_mean")
                for key in metric_keys
            }

    for row in combo_rows:
        for key in metric_keys:
            baseline_value = baseline_metrics.get(key)
            current_value = row.get(f"{key}_mean")
            row[f"{key}_delta_vs_baseline"] = (
                (float(current_value) - float(baseline_value))
                if current_value is not None and baseline_value is not None
                else None
            )

    best_combo = None
    non_baseline_rows = [row for row in combo_rows if not row["is_baseline_variant"]]
    if non_baseline_rows:
        best_combo = max(
            non_baseline_rows,
            key=lambda row: (
                float(row.get("mAP50_mean") or 0.0),
                float(row.get("f1_mean") or 0.0),
            ),
        )

    combo_rows.sort(
        key=lambda row: (
            1 if row["is_baseline_variant"] else 0,
            -(float(row.get("f1_mean") or 0.0)),
            str(row["test_variant_name"]),
        )
    )
    if baseline_row is not None:
        combo_rows.sort(
            key=lambda row: (0 if row["is_baseline_variant"] else 1, str(row["test_variant_name"]))
        )
        if best_combo is not None:
            combo_rows = [baseline_row] + sorted(
                [row for row in combo_rows if not row["is_baseline_variant"]],
                key=lambda row: (
                    -(float(row.get("mAP50_mean") or 0.0)),
                    -(float(row.get("f1_mean") or 0.0)),
                    str(row["test_variant_name"]),
                ),
            )

    return {
        "baseline": baseline_row,
        "best_combo_by_map50_mean": best_combo,
        "rows": combo_rows,
    }


def _should_emit_combo_summary(rows: List[Dict[str, Any]]) -> bool:
    variant_names = {
        str(row.get("test_variant_name"))
        for row in rows
        if row.get("test_variant_name") is not None
    }
    return len(variant_names) > 1


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "base_run_name",
        "train_seed",
        "test_variant_name",
        "test_variant_label",
        "is_baseline_variant",
        "best_model_path",
        "model_weight_source",
        "test_output_dir",
        "validation_output_dir",
        "conf",
        "conf_source",
        "adabn_epochs",
        "adabn_max_images",
        "adabn_max_stage",
        "adabn_momentum",
        "adabn_enabled",
        "adabn_applied",
        "adabn_image_dir",
        "adabn_num_images",
        "adabn_num_batches",
        "adabn_bn_layers",
        "precision",
        "recall",
        "f1",
        "ap50",
        "mAP50",
        "mAP50_95",
        "image_accuracy",
        "num_images",
        "validation_num_images",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_ap50",
        "val_mAP50",
        "val_mAP50_95",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_txt(path: Path, rows: List[Dict[str, Any]], aggregate: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("========== Train/Test Seed Sweep Summary ==========\n")
        f.write(f"Num runs: {len(rows)}\n")
        f.write("==================================================\n\n")

        for row in rows:
            f.write(f"[{row['run_name']}] seed={row['train_seed']}\n")
            f.write(f"  base_run_name   : {row['base_run_name']}\n")
            f.write(
                f"  test_variant    : {row['test_variant_name']} "
                f"(label={row['test_variant_label']}, baseline={row['is_baseline_variant']})\n"
            )
            f.write(f"  best_model_path : {row['best_model_path']}\n")
            f.write(f"  model_weight    : {row['model_weight_source']}\n")
            f.write(f"  val_output_dir  : {row['validation_output_dir']}\n")
            f.write(f"  test_output_dir : {row['test_output_dir']}\n")
            f.write(f"  conf            : {row['conf']} ({row['conf_source']})\n")
            f.write(
                f"  AdaBN           : epochs={row['adabn_epochs']}, max_images={row['adabn_max_images']}, "
                f"stage={row['adabn_max_stage']}, momentum={row['adabn_momentum']}, "
                f"enabled={row['adabn_enabled']}, applied={row['adabn_applied']}, "
                f"images={row['adabn_num_images']}, batches={row['adabn_num_batches']}, "
                f"bn_layers={row['adabn_bn_layers']}\n"
            )
            f.write(f"  precision       : {_safe_metric_str(row['precision'])}\n")
            f.write(f"  recall          : {_safe_metric_str(row['recall'])}\n")
            f.write(f"  f1              : {_safe_metric_str(row['f1'])}\n")
            f.write(f"  mAP50           : {_safe_metric_str(row['mAP50'])}\n")
            f.write(f"  mAP50_95        : {_safe_metric_str(row['mAP50_95'])}\n")
            f.write(f"  image_accuracy  : {_safe_metric_str(row['image_accuracy'])}\n")
            f.write(f"  val_precision   : {_safe_metric_str(row['val_precision'])}\n")
            f.write(f"  val_recall      : {_safe_metric_str(row['val_recall'])}\n")
            f.write(f"  val_f1          : {_safe_metric_str(row['val_f1'])}\n")
            f.write(f"  val_mAP50       : {_safe_metric_str(row['val_mAP50'])}\n")
            f.write(f"  val_mAP50_95    : {_safe_metric_str(row['val_mAP50_95'])}\n")
            f.write("\n")

        f.write("---------- Aggregate ----------\n")
        for key, stats in aggregate.items():
            if key in {"best_run_by_f1", "worst_run_by_f1"}:
                f.write(
                    f"[{key}] "
                    f"name={stats['run_name']}, seed={stats['train_seed']}, "
                    f"f1={_safe_metric_str(stats['f1'])}, mAP50={_safe_metric_str(stats['mAP50'])}, "
                    f"mAP50_95={_safe_metric_str(stats['mAP50_95'])}\n"
                )
                continue
            if key in {"best_run_by_val_f1", "worst_run_by_val_f1"}:
                f.write(
                    f"[{key}] "
                    f"name={stats['run_name']}, seed={stats['train_seed']}, "
                    f"val_f1={_safe_metric_str(stats['val_f1'])}, "
                    f"val_mAP50={_safe_metric_str(stats['val_mAP50'])}, "
                    f"val_mAP50_95={_safe_metric_str(stats['val_mAP50_95'])}\n"
                )
                continue
            f.write(
                f"[{key}] mean={float(stats['mean']):.4f}, std={float(stats['std']):.4f}, "
                f"median={float(stats['median']):.4f}, min={float(stats['min']):.4f}, "
                f"max={float(stats['max']):.4f}, gap={float(stats['gap']):.4f}\n"
            )


def _write_combo_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "test_variant_name",
        "test_variant_label",
        "is_baseline_variant",
        "num_runs",
        "num_seeds",
        "adabn_epochs",
        "adabn_max_images",
        "adabn_max_stage",
        "adabn_momentum",
        "precision_mean",
        "precision_std",
        "precision_min",
        "precision_max",
        "precision_delta_vs_baseline",
        "recall_mean",
        "recall_std",
        "recall_min",
        "recall_max",
        "recall_delta_vs_baseline",
        "f1_mean",
        "f1_std",
        "f1_min",
        "f1_max",
        "f1_delta_vs_baseline",
        "ap50_mean",
        "ap50_std",
        "ap50_min",
        "ap50_max",
        "ap50_delta_vs_baseline",
        "mAP50_mean",
        "mAP50_std",
        "mAP50_min",
        "mAP50_max",
        "mAP50_delta_vs_baseline",
        "mAP50_95_mean",
        "mAP50_95_std",
        "mAP50_95_min",
        "mAP50_95_max",
        "mAP50_95_delta_vs_baseline",
        "image_accuracy_mean",
        "image_accuracy_std",
        "image_accuracy_min",
        "image_accuracy_max",
        "image_accuracy_delta_vs_baseline",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_combo_summary_txt(path: Path, combo_summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = combo_summary.get("rows", [])
    baseline = combo_summary.get("baseline")
    best_combo = combo_summary.get("best_combo_by_map50_mean")
    with open(path, "w", encoding="utf-8") as f:
        f.write("========== AdaBN Combination Comparison ==========\n")
        f.write(f"Num combinations: {len(rows)}\n")
        if baseline is not None:
            f.write(
                f"Baseline: {baseline['test_variant_name']} "
                f"(f1_mean={_safe_metric_str(baseline.get('f1_mean'))}, "
                f"mAP50_mean={_safe_metric_str(baseline.get('mAP50_mean'))})\n"
            )
        if best_combo is not None:
            f.write(
                f"Best combo by mean mAP50: {best_combo['test_variant_name']} "
                f"(mAP50_mean={_safe_metric_str(best_combo.get('mAP50_mean'))}, "
                f"delta_vs_baseline={_safe_metric_str(best_combo.get('mAP50_delta_vs_baseline'))})\n"
            )
        f.write("==================================================\n\n")

        for row in rows:
            f.write(f"[{row['test_variant_name']}] baseline={row['is_baseline_variant']}\n")
            f.write(
                f"  params          : epochs={row['adabn_epochs']}, max_images={row['adabn_max_images']}, "
                f"stage={row['adabn_max_stage']}, momentum={row['adabn_momentum']}\n"
            )
            f.write(f"  num_runs/seeds  : {row['num_runs']} / {row['num_seeds']}\n")
            f.write(
                f"  f1_mean/std     : {_safe_metric_str(row.get('f1_mean'))} / "
                f"{_safe_metric_str(row.get('f1_std'))}\n"
            )
            f.write(
                f"  mAP50_mean/std  : {_safe_metric_str(row.get('mAP50_mean'))} / "
                f"{_safe_metric_str(row.get('mAP50_std'))}\n"
            )
            f.write(
                f"  mAP50_95_mean/std : {_safe_metric_str(row.get('mAP50_95_mean'))} / "
                f"{_safe_metric_str(row.get('mAP50_95_std'))}\n"
            )
            f.write(
                f"  delta_vs_base   : f1={_safe_metric_str(row.get('f1_delta_vs_baseline'))}, "
                f"mAP50={_safe_metric_str(row.get('mAP50_delta_vs_baseline'))}, "
                f"mAP50_95={_safe_metric_str(row.get('mAP50_95_delta_vs_baseline'))}, "
                f"acc={_safe_metric_str(row.get('image_accuracy_delta_vs_baseline'))}\n"
            )
            f.write(
                f"  mean metrics    : precision={_safe_metric_str(row.get('precision_mean'))}, "
                f"recall={_safe_metric_str(row.get('recall_mean'))}, "
                f"mAP50={_safe_metric_str(row.get('mAP50_mean'))}, "
                f"mAP50_95={_safe_metric_str(row.get('mAP50_95_mean'))}, "
                f"acc={_safe_metric_str(row.get('image_accuracy_mean'))}\n"
            )
            f.write("\n")


def _test_summary_prefix(test_run_name_prefix: str) -> str:
    return normalize_optional_text(test_run_name_prefix) or "seed_sweep"


def _print_run_config_preview(train_cfg: Dict[str, Any], test_cfg: Dict[str, Any]) -> None:
    dataset_roots = train_cfg.get("DATASET_ROOTS") or [train_cfg.get("DATASET_ROOT")]
    saved_split_dirs = train_cfg.get("SAVED_SPLIT_DIRS") or train_cfg.get("SAVED_SPLIT_DIR")

    print("[Config check][Train]")
    print(f"  RUN_NAME              : {train_cfg.get('RUN_NAME')}")
    print(f"  TRAIN_SEED            : {train_cfg.get('TRAIN_SEED')}")
    print(f"  TRAIN_DETERMINISTIC   : {train_cfg.get('TRAIN_DETERMINISTIC')}")
    print(f"  DATASET_ROOTS         : {dataset_roots}")
    print(f"  SAVED_SPLIT_DIRS      : {saved_split_dirs}")
    print(f"  SPLIT_SEED            : {train_cfg.get('SPLIT_SEED')}")
    print(
        f"  EPOCHS/IMGSZ/BATCH    : "
        f"{train_cfg.get('EPOCHS')} / {train_cfg.get('IMGSZ')} / {train_cfg.get('BATCH')}"
    )
    print(
        f"  AMP/CACHE/WORKERS     : "
        f"{train_cfg.get('AMP')} / {train_cfg.get('CACHE')} / {train_cfg.get('WORKERS')}"
    )
    print(f"  DETECTION_MODEL_PATH  : {train_cfg.get('DETECTION_MODEL_PATH')}")
    print(
        f"  CLOSE_MOSAIC/RECT     : "
        f"{train_cfg.get('CLOSE_MOSAIC')} / {train_cfg.get('RECT')}"
    )
    print(f"  GENERATED_YAML_PATH   : {train_cfg.get('GENERATED_YAML_PATH')}")
    print(f"  MODEL_WEIGHT_SOURCE   : {train_cfg.get('MODEL_EXPORT_WEIGHT_SOURCE', 'best')}")

    print("[Config check][Test]")
    print(f"  TEST_RUN_NAME         : {test_cfg.get('TEST_RUN_NAME')}")
    print(f"  TEST_IMAGE_DIR        : {test_cfg.get('TEST_IMAGE_DIR')}")
    print(f"  TEST_LABEL_DIR        : {test_cfg.get('TEST_LABEL_DIR')}")
    print(f"  TEST_OUTPUT_DIR       : {test_cfg.get('TEST_OUTPUT_DIR')}")
    print(f"  TEST_MODEL_PATH       : {test_cfg.get('TEST_MODEL_PATH')}")
    print(
        f"  TEST_IMGSZ/CONF/IOU   : "
        f"{test_cfg.get('TEST_IMGSZ')} / {test_cfg.get('TEST_CONF')} / {test_cfg.get('TEST_IOU')}"
    )
    print(
        f"  TEST_BATCH/DEVICE     : "
        f"{test_cfg.get('TEST_BATCH')} / {test_cfg.get('TEST_DEVICE')}"
    )
    print(f"  SEED_SWEEP_DEVICE     : {test_cfg.get('_SEED_SWEEP_DEVICE_OVERRIDE')}")
    print(f"  TEST_LIMIT_VISIBLE    : {test_cfg.get('TEST_LIMIT_VISIBLE_DEVICE', False)}")
    print(f"  TEST_FP16             : {test_cfg.get('TEST_FP16', False)}")
    print(f"  TEST_SAVE_VISUALS     : {test_cfg.get('TEST_SAVE_VISUALIZATIONS', True)}")
    print(
        f"  TEST_ADABN            : "
        f"{test_cfg.get('TEST_ADABN_ENABLED', False)} "
        f"(dir={test_cfg.get('TEST_ADABN_IMAGE_DIR')}, "
        f"batch={test_cfg.get('TEST_ADABN_BATCH')}, imgsz={test_cfg.get('TEST_ADABN_IMGSZ')}, "
        f"epochs={test_cfg.get('TEST_ADABN_EPOCHS', 1)}, stage={test_cfg.get('TEST_ADABN_MAX_STAGE')}, "
        f"momentum={test_cfg.get('TEST_ADABN_MOMENTUM', 'use_original')}, "
        f"fp16={test_cfg.get('TEST_ADABN_FP16', False)}, "
        f"debug={test_cfg.get('TEST_ADABN_DEBUG', False)})"
    )
    print(
        f"  TUNED_CONF            : "
        f"{test_cfg.get('TEST_USE_F1_OPTIMIZED_CONF')} "
        f"(offset={test_cfg.get('TEST_F1_OPTIMIZED_CONF_OFFSET', 0.0)}, "
        f"split={test_cfg.get('TEST_TUNE_SPLIT_DIR')}, name={test_cfg.get('TEST_TUNE_SPLIT')})"
    )
    print(
        f"  DEPLOYMENT_CONF       : use={test_cfg.get('TEST_USE_DEPLOYMENT_THRESHOLD', False)} "
        f"tune={test_cfg.get('TEST_TUNE_DEPLOYMENT_THRESHOLD', False)} "
        f"overwrite={test_cfg.get('TEST_OVERWRITE_DEPLOYMENT_THRESHOLD', False)} "
        f"(background={test_cfg.get('TEST_DEPLOYMENT_BACKGROUND_IMAGE_DIR')}, "
        f"max_fp={test_cfg.get('TEST_DEPLOYMENT_MAX_FP', 0)}, "
        f"by_image={test_cfg.get('TEST_DEPLOYMENT_COUNT_FP_BY_IMAGE', True)})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test seed sweep runner")
    parser.add_argument(
        "--config",
        default=DEFAULT_SEED_SWEEP_CONFIG_PATH,
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=None,
        help="Override CLOSE_MOSAIC for all train runs in this sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_config_path = Path(args.config)
    sweep_cfg = load_config(str(sweep_config_path))
    config_dir = sweep_config_path.parent.resolve()
    sweep_mode = str(sweep_cfg.get("SEED_SWEEP_MODE", "full")).strip().lower()
    if sweep_mode not in {"full", "finetune", "resume", "post_train_only", "test_only", "summary_only"}:
        raise ValueError(
            "SEED_SWEEP_MODE must be 'full', 'finetune', 'resume', 'post_train_only', "
            "'test_only', or 'summary_only'."
        )
    if args.close_mosaic is not None and args.close_mosaic < 0:
        raise ValueError("--close-mosaic must be a non-negative integer.")
    model_weight_source = str(sweep_cfg.get("SEED_SWEEP_MODEL_WEIGHT_SOURCE", "best")).strip().lower()
    if model_weight_source not in {"best", "last"}:
        raise ValueError("SEED_SWEEP_MODEL_WEIGHT_SOURCE must be 'best' or 'last'.")
    if sweep_mode == "resume" and model_weight_source != "best":
        print(
            "[Seed Sweep][resume][WARN] Resume mode uses models_final/*_best.pt; "
            "using best for SEED_SWEEP_MODEL_WEIGHT_SOURCE."
        )
        model_weight_source = "best"

    base_train_config_path = _resolve_path(sweep_cfg["BASE_TRAIN_CONFIG_PATH"], config_dir)
    base_test_config_path = _resolve_path(sweep_cfg["BASE_TEST_CONFIG_PATH"], config_dir)
    base_train_cfg = load_config(str(base_train_config_path))
    base_test_cfg = load_config(str(base_test_config_path))
    if "TEST_ADABN_SAMPLE_SEED" in sweep_cfg:
        base_test_cfg["TEST_ADABN_SAMPLE_SEED"] = sweep_cfg["TEST_ADABN_SAMPLE_SEED"]
    train_run_name_prefix = str(
        sweep_cfg.get(
            "SEED_SWEEP_TRAIN_RUN_NAME_PREFIX",
            sweep_cfg.get(
                "SEED_SWEEP_RUN_NAME_PREFIX",
                base_train_cfg.get("RUN_NAME", "seed_sweep"),
            ),
        )
    ).strip()
    train_run_name_prefix = ensure_name_suffix(train_run_name_prefix, "seed")
    test_run_name_prefix = str(
        sweep_cfg.get("SEED_SWEEP_TEST_RUN_NAME_PREFIX", train_run_name_prefix)
    ).strip()

    output_roots = _resolve_output_roots(
        sweep_cfg,
        config_dir=config_dir,
        train_run_name_prefix=train_run_name_prefix,
    )
    output_base = output_roots["output_base"]
    output_root = output_roots["output_root"]
    final_output_base = output_roots["final_output_base"]
    final_output_root = output_roots["final_output_root"]
    local_staging_enabled = bool(output_roots["local_staging_enabled"])
    local_staging_cleanup_after_sync = as_bool(
        sweep_cfg.get("SEED_SWEEP_LOCAL_STAGING_CLEANUP_AFTER_SYNC", True),
        default=True,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    local_staging_bootstrap_stats = None
    if (
        local_staging_enabled
        and sweep_mode in {"resume", "post_train_only", "test_only", "summary_only"}
        and final_output_root.exists()
    ):
        print(
            "[Seed Sweep][local staging] Copying existing final results into local staging "
            f"(preserving existing local files): {final_output_root} -> {output_root}"
        )
        local_staging_bootstrap_stats = _copy_tree_for_staging_sync(
            src_root=final_output_root,
            dst_root=output_root,
            overwrite=False,
        )
    train_runs_root = output_root / "train_runs"
    models_root = output_root / "models_final"
    tests_root = output_root / "test_runs"
    resume_train_mode = _resolve_resume_train_mode(sweep_cfg, output_root) if sweep_mode == "resume" else None
    process_device_info = (
        _preconfigure_seed_sweep_process_device(sweep_cfg)
        if sweep_mode != "summary_only"
        else {
            "requested_device": None,
            "enabled": False,
            "applied": False,
            "logical_device": None,
            "cuda_visible_devices_before": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_visible_devices_after": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "warning": None,
        }
    )

    test_variants = _build_test_variants(base_test_cfg)
    continue_on_error = as_bool(sweep_cfg.get("SEED_SWEEP_CONTINUE_ON_ERROR", False), default=False)
    clean_run_weights_after_export = as_bool(
        sweep_cfg.get("SEED_SWEEP_CLEAN_RUN_WEIGHTS_AFTER_EXPORT", False),
        default=False,
    )
    run_entries = _resolve_run_entries(
        sweep_cfg,
        output_root,
        train_run_name_prefix,
        test_run_name_prefix,
        sweep_mode,
        model_weight_source,
    )
    resume_completed_model_paths: Dict[int, Path] = {}
    resume_missing_entries: List[Dict[str, Any]] = []
    if sweep_mode == "resume":
        for entry in run_entries:
            train_seed = int(entry["train_seed"])
            completed_model_path = _resume_best_model_path(models_root, str(entry["train_run_name"]))
            if completed_model_path.exists():
                resume_completed_model_paths[train_seed] = completed_model_path.resolve()
            else:
                resume_missing_entries.append(entry)

    finetune_source_run_dir: Path | None = None
    finetune_source_model_paths: Dict[int, Path] = {}
    needs_finetune_source = sweep_mode == "finetune" or (
        sweep_mode == "resume"
        and resume_train_mode == "finetune"
        and bool(resume_missing_entries)
    )
    if needs_finetune_source:
        finetune_source_run_dir = _resolve_finetune_source_run_dir(sweep_cfg, config_dir)
        finetune_source_model_paths = _resolve_finetune_source_model_paths(
            source_run_dir=finetune_source_run_dir,
            run_entries=resume_missing_entries if sweep_mode == "resume" else run_entries,
        )
    shared_split_info = _prepare_shared_split_if_needed(
        sweep_cfg,
        base_train_cfg,
        output_root=output_root,
        config_dir=config_dir,
        mode=resume_train_mode if sweep_mode == "resume" and resume_missing_entries else sweep_mode,
        run_name_prefix=train_run_name_prefix,
    )

    _write_yaml(output_root / "sweep_config_snapshot.yaml", sweep_cfg)
    _write_yaml(output_root / "base_train_config_snapshot.yaml", base_train_cfg)
    _write_yaml(output_root / "base_test_config_snapshot.yaml", base_test_cfg)

    print(f"[Seed Sweep] mode={sweep_mode}")
    print(f"[Seed Sweep] model_weight_source={model_weight_source}")
    print(f"[Seed Sweep] train_run_name_prefix={train_run_name_prefix}")
    print(f"[Seed Sweep] test_run_name_prefix={test_run_name_prefix}")
    print(f"[Seed Sweep] output_base={output_base}")
    print(f"[Seed Sweep] output_root={output_root}")
    if local_staging_enabled:
        print("[Seed Sweep][local staging] enabled=True")
        print(f"[Seed Sweep][local staging] work_output_root={output_root}")
        print(f"[Seed Sweep][local staging] final_output_root={final_output_root}")
        print(f"[Seed Sweep][local staging] cleanup_after_sync={local_staging_cleanup_after_sync}")
        if local_staging_bootstrap_stats is not None:
            print(f"[Seed Sweep][local staging] bootstrap={local_staging_bootstrap_stats}")
    else:
        print("[Seed Sweep][local staging] enabled=False")
    print(f"[Seed Sweep] clean_run_weights_after_export={clean_run_weights_after_export}")
    if sweep_mode == "resume":
        print(f"[Seed Sweep][resume] train_mode_for_missing={resume_train_mode}")
        print(
            "[Seed Sweep][resume] "
            f"skip_train={len(resume_completed_model_paths)}, train_required={len(resume_missing_entries)}"
        )
    if finetune_source_run_dir is not None:
        print(f"[Seed Sweep] finetune_source_run_dir={finetune_source_run_dir}")
        print(f"[Seed Sweep] finetune_source_model_count={len(finetune_source_model_paths)}")
    if sweep_mode != "summary_only":
        print(
            f"[Seed Sweep] test_variants={[(variant['variant_name'], variant['variant_type']) for variant in test_variants]}"
        )
    else:
        print("[Seed Sweep] test_variants=recover_from_saved_results")
    if process_device_info.get("enabled", False):
        print(
            f"[Seed Sweep] process_device={process_device_info.get('requested_device')} "
            f"(logical={process_device_info.get('logical_device')}, "
            f"CVD={process_device_info.get('cuda_visible_devices_before')} -> "
            f"{process_device_info.get('cuda_visible_devices_after')})"
        )
        if process_device_info.get("warning"):
            print(f"[Seed Sweep][WARN] {process_device_info.get('warning')}")
    if args.close_mosaic is not None:
        print(f"[Seed Sweep] CLI override CLOSE_MOSAIC={args.close_mosaic}")
    if shared_split_info:
        print(f"[Seed Sweep] shared_split={shared_split_info['split_root']}")
    print(
        "[Seed Sweep] runs="
        f"{[(item['train_run_name'], item['test_run_name'], item['train_seed']) for item in run_entries]}"
    )

    run_results: List[Dict[str, Any]] = []
    failed_runs: List[Dict[str, Any]] = []

    for index, run_entry in enumerate(run_entries, start=1):
        train_seed = int(run_entry["train_seed"])
        run_name = str(run_entry["train_run_name"])
        test_run_name = str(run_entry["test_run_name"])
        seed_root = output_root / f"seed_{train_seed}"
        train_mode_for_seed = resume_train_mode if sweep_mode == "resume" else sweep_mode
        resume_existing_best_model_path = _resume_best_model_path(models_root, run_name)
        resume_skip_train = sweep_mode == "resume" and resume_existing_best_model_path.exists()
        effective_train_cfg_path = seed_root / "effective_train_config.yaml"
        if (
            sweep_mode in {"post_train_only", "test_only"}
            or (sweep_mode == "resume" and resume_skip_train)
        ) and effective_train_cfg_path.exists():
            train_cfg = load_config(str(effective_train_cfg_path))
        else:
            train_cfg = deepcopy(base_train_cfg)
        test_cfg = deepcopy(base_test_cfg)
        _apply_seed_sweep_device_override(train_cfg, test_cfg, process_device_info)

        train_cfg["TRAIN_SEED"] = int(train_seed)
        train_cfg["RUN_NAME"] = run_name
        train_cfg["MODEL_EXPORT_WEIGHT_SOURCE"] = model_weight_source
        if args.close_mosaic is not None:
            train_cfg["CLOSE_MOSAIC"] = int(args.close_mosaic)
        train_cfg["RUN_PROJECT_DIR"] = str(train_runs_root)
        train_cfg["FINAL_BEST_DIR"] = str(models_root)
        finetune_source_model_path = finetune_source_model_paths.get(train_seed)
        if train_mode_for_seed == "finetune" and not resume_skip_train:
            if finetune_source_model_path is None:
                raise RuntimeError(f"Fine-tuning source model path not found: train_seed={train_seed}")
            train_cfg["DETECTION_MODEL_PATH"] = str(finetune_source_model_path)

        if shared_split_info and train_mode_for_seed in {"full", "finetune"}:
            train_cfg["SAVED_SPLIT_DIR"] = shared_split_info["split_root"]
            train_cfg.pop("SAVED_SPLIT_DIRS", None)
            train_cfg["GENERATED_YAML_PATH"] = shared_split_info["yaml_path"]
        else:
            train_cfg["GENERATED_YAML_PATH"] = str(seed_root / "train_dataset_split.yaml")

        if (
            not train_cfg.get("SAVED_SPLIT_DIRS")
            and not train_cfg.get("SAVED_SPLIT_DIR")
            and not shared_split_info
        ):
            dataset_root = Path(str(train_cfg.get("DATASET_ROOT") or train_cfg["DATASET_ROOTS"][0])).expanduser()
            split_parent = Path(str(train_cfg.get("SPLIT_PARENT_DIR", dataset_root))).expanduser()
            train_cfg["SPLIT_PARENT_DIR"] = str(split_parent)

        expected_model_path = models_root / f"{run_name}_{model_weight_source}.pt"
        test_cfg["TEST_MODEL_PATH"] = str(expected_model_path)
        test_cfg["TEST_RUN_NAME"] = test_run_name
        _apply_adabn_sample_seed(test_cfg, train_seed)

        print("\n" + "=" * 100)
        print(
            f"[Seed Sweep {index}/{len(run_entries)}] "
            f"train_run_name={run_name} | test_run_name={test_run_name} | train_seed={train_seed}"
        )
        print("=" * 100)
        if sweep_mode == "summary_only":
            print(f"[Summary Only] seed_root           : {seed_root}")
            print(f"[Summary Only] expected_model_path: {expected_model_path}")
        else:
            _print_run_config_preview(train_cfg, test_cfg)

        try:
            if resume_skip_train:
                existing_train_result = _read_json(seed_root / "train_result.json") or {}
                validation_metrics = _load_existing_validation_metrics(seed_root)
                train_result = dict(existing_train_result)
                train_result.update(
                    {
                        "run_name": run_name,
                        "train_seed": int(train_seed),
                        "test_run_name": test_run_name,
                        "best_model_path": str(resume_existing_best_model_path),
                        "exported_model_path": str(resume_existing_best_model_path),
                        "model_weight_source": "best",
                        "validation_metrics": validation_metrics,
                        "mode": "resume_reused_existing_model",
                        "resume_original_mode": existing_train_result.get("mode"),
                        "resume_skipped_training": True,
                    }
                )
                test_cfg["TEST_MODEL_PATH"] = str(resume_existing_best_model_path)
                if not test_cfg.get("TEST_TUNE_SPLIT_DIR") and existing_train_result.get("split_root"):
                    test_cfg["TEST_TUNE_SPLIT_DIR"] = str(existing_train_result["split_root"])
                _write_json(seed_root / "train_result.json", train_result)
                print(
                    f"[Seed Sweep][resume] train skip: seed={train_seed} | "
                    f"model={resume_existing_best_model_path}"
                )
            elif sweep_mode in {"full", "finetune", "resume"}:
                from grape_detector.detection.train import run_train_with_config

                train_result = run_train_with_config(train_cfg)
                if train_mode_for_seed == "finetune":
                    train_result["mode"] = "finetune"
                    train_result["finetune_source_run_dir"] = (
                        str(finetune_source_run_dir) if finetune_source_run_dir is not None else None
                    )
                    train_result["finetune_source_model_path"] = str(finetune_source_model_path)
                    train_result["finetune_source_model_weight_source"] = "best"
                if sweep_mode == "resume":
                    train_result["mode"] = f"resume_trained_{train_mode_for_seed}"
                    train_result["resume_skipped_training"] = False
                best_model_path = train_result.get("best_model_path")
                if not best_model_path:
                    raise RuntimeError(f"Best model path is missing: run_name={run_name}")
                test_cfg["TEST_MODEL_PATH"] = str(best_model_path)
                if clean_run_weights_after_export and not train_result.get("run_weights_cleanup"):
                    train_result["run_weights_cleanup"] = cleanup_run_weights_after_export(
                        run_name=run_name,
                        run_dir=Path(str(train_result.get("run_dir") or train_runs_root / run_name)),
                        final_model_path=Path(str(best_model_path)),
                    )
                if not test_cfg.get("TEST_TUNE_SPLIT_DIR"):
                    test_cfg["TEST_TUNE_SPLIT_DIR"] = str(train_result["split_root"])
                _write_yaml(seed_root / "effective_train_config.yaml", train_cfg)
                _write_json(seed_root / "train_result.json", train_result)
            elif sweep_mode == "post_train_only":
                from grape_detector.detection.train import run_post_train_only_with_config

                train_result = run_post_train_only_with_config(train_cfg)
                best_model_path = train_result.get("best_model_path")
                if not best_model_path:
                    raise RuntimeError(f"Best model path is missing after post_train_only: run_name={run_name}")
                test_cfg["TEST_MODEL_PATH"] = str(best_model_path)
                if clean_run_weights_after_export and not train_result.get("run_weights_cleanup"):
                    train_result["run_weights_cleanup"] = cleanup_run_weights_after_export(
                        run_name=run_name,
                        run_dir=Path(str(train_result.get("run_dir") or train_runs_root / run_name)),
                        final_model_path=Path(str(best_model_path)),
                    )
                if not test_cfg.get("TEST_TUNE_SPLIT_DIR"):
                    test_cfg["TEST_TUNE_SPLIT_DIR"] = str(train_result["split_root"])
                _write_yaml(seed_root / "effective_train_config.yaml", train_cfg)
                _write_json(seed_root / "train_result.json", train_result)
            elif sweep_mode == "test_only":
                existing_train_result = _read_json(seed_root / "train_result.json") or {}
                validation_metrics = _load_existing_validation_metrics(seed_root)
                if not expected_model_path.exists():
                    raise FileNotFoundError(
                        f"Cannot run test_only: existing trained model not found: {expected_model_path}"
                    )
                train_result = {
                    "run_name": run_name,
                    "train_seed": int(train_seed),
                    "test_run_name": test_run_name,
                    "best_model_path": str(expected_model_path),
                    "model_weight_source": model_weight_source,
                    "generated_yaml_path": existing_train_result.get("generated_yaml_path"),
                    "split_root": existing_train_result.get("split_root"),
                    "val_results": existing_train_result.get("val_results"),
                    "validation_metrics": validation_metrics,
                    "tuned_conf": existing_train_result.get("tuned_conf"),
                    "mode": "reused_existing_model",
                }
                if not test_cfg.get("TEST_TUNE_SPLIT_DIR") and existing_train_result.get("split_root"):
                    test_cfg["TEST_TUNE_SPLIT_DIR"] = str(existing_train_result["split_root"])
                _write_json(seed_root / "train_result.json", train_result)
            else:
                recovered_results = _load_summary_only_results_for_seed(
                    seed_root=seed_root,
                    run_name=run_name,
                    train_seed=train_seed,
                    expected_model_path=expected_model_path,
                    model_weight_source=model_weight_source,
                )
                failed_payload = _read_json(seed_root / "failed_run.json")
                if isinstance(failed_payload, dict):
                    failed_runs.append(failed_payload)
                    print(
                        f"[Summary Only][Recorded failure] run_name={run_name} | "
                        f"error={failed_payload.get('error')}"
                    )
                if not recovered_results:
                    if isinstance(failed_payload, dict):
                        continue
                    raise FileNotFoundError(
                        f"No test_metrics found to read in summary_only mode: {seed_root}"
                    )
                run_results.extend(recovered_results)
                print(
                    f"[Summary Only] recovered_results={len(recovered_results)} "
                    f"for seed={train_seed}"
                )
                continue

            validation_metrics = train_result.get("validation_metrics")
            if validation_metrics is not None:
                _write_json(seed_root / "validation_metrics.json", validation_metrics)
            visibility_info = _preconfigure_test_only_visibility(test_cfg, sweep_mode)
            if visibility_info.get("enabled", False):
                print("[Seed Sweep][TEST_DEVICE_LOCK] ===========================================")
                print(
                    f"[Seed Sweep][TEST_DEVICE_LOCK] requested TEST_DEVICE : "
                    f"{visibility_info.get('requested_device')}"
                )
                print(
                    f"[Seed Sweep][TEST_DEVICE_LOCK] CUDA_VISIBLE_DEVICES : "
                    f"{visibility_info.get('cuda_visible_devices_before')} -> "
                    f"{visibility_info.get('cuda_visible_devices_after')}"
                )
                if visibility_info.get("warning"):
                    print(f"[Seed Sweep][TEST_DEVICE_LOCK][WARN] {visibility_info.get('warning')}")
                print("[Seed Sweep][TEST_DEVICE_LOCK] ===========================================")
            from grape_detector.detection.evaluate import run_test_with_config

            has_multiple_test_variants = len(test_variants) > 1
            if has_multiple_test_variants:
                _write_yaml(seed_root / "effective_test_config_base.yaml", test_cfg)

            for variant_index, variant in enumerate(test_variants, start=1):
                if has_multiple_test_variants:
                    variant_output_dir = tests_root / test_run_name / variant["variant_name"]
                    variant_seed_dir = seed_root / "test_variants" / variant["variant_name"]
                    variant_test_run_name = f"{test_run_name}__{variant['variant_name']}"
                    print(
                        f"[Seed Sweep][Test Variant {variant_index}/{len(test_variants)}] "
                        f"{variant_test_run_name}"
                    )
                else:
                    variant_output_dir = tests_root / test_run_name
                    variant_seed_dir = seed_root
                    variant_test_run_name = test_run_name

                variant_test_cfg = _apply_test_variant(
                    test_cfg,
                    variant,
                    test_output_dir=variant_output_dir,
                )
                variant_test_cfg["TEST_RUN_NAME"] = variant_test_run_name
                _write_yaml(variant_seed_dir / "effective_test_config.yaml", variant_test_cfg)

                test_metrics = run_test_with_config(variant_test_cfg)
                _write_json(variant_seed_dir / "test_metrics.json", test_metrics)

                run_results.append(
                    {
                        "run_name": run_name,
                        "test_run_name": variant_test_run_name,
                        "test_variant_name": variant["variant_name"],
                        "test_variant_label": variant["variant_label"],
                        "is_baseline_variant": bool(variant["is_baseline"]),
                        "adabn_epochs": variant["combo_fields"].get("adabn_epochs"),
                        "adabn_max_images": variant["combo_fields"].get("adabn_max_images"),
                        "adabn_max_stage": variant["combo_fields"].get("adabn_max_stage"),
                        "adabn_momentum": variant["combo_fields"].get("adabn_momentum"),
                        "train_seed": int(train_seed),
                        "train_result": train_result,
                        "test_metrics": test_metrics,
                    }
                )
        except Exception as exc:
            failed = {
                "run_name": run_name,
                "test_run_name": test_run_name,
                "train_seed": int(train_seed),
                "error": repr(exc),
            }
            failed_runs.append(failed)
            _write_json(seed_root / "failed_run.json", failed)
            print(f"[Seed Sweep failed] run_name={run_name} | error={exc}")
            if not continue_on_error:
                raise

    summary_rows = _build_summary_rows(run_results)
    aggregate = _compute_aggregate(summary_rows) if summary_rows else {}
    emit_combo_summary = _should_emit_combo_summary(summary_rows)
    combo_summary = _compute_combo_comparison(summary_rows) if emit_combo_summary else None
    effective_test_variant_count = (
        len(test_variants)
        if sweep_mode != "summary_only"
        else (
            len(combo_summary.get("rows", []))
            if combo_summary is not None
            else len({row.get("test_variant_name") for row in summary_rows if row.get("test_variant_name") is not None})
        )
    )
    summary_payload = {
        "mode": sweep_mode,
        "output_base": str(output_base),
        "output_root": str(output_root),
        "final_output_base": str(final_output_base),
        "final_output_root": str(final_output_root),
        "local_staging": {
            "enabled": local_staging_enabled,
            "work_output_base": str(output_base),
            "work_output_root": str(output_root),
            "final_output_base": str(final_output_base),
            "final_output_root": str(final_output_root),
            "cleanup_after_sync": local_staging_cleanup_after_sync,
            "bootstrap_stats": local_staging_bootstrap_stats,
        },
        "train_run_name_prefix": train_run_name_prefix,
        "test_run_name_prefix": test_run_name_prefix,
        "cli_overrides": {
            "CLOSE_MOSAIC": int(args.close_mosaic) if args.close_mosaic is not None else None,
        },
        "num_test_variants": effective_test_variant_count,
        "num_runs_requested": len(run_entries),
        "num_test_runs_requested": (
            len(run_entries) * len(test_variants)
            if sweep_mode != "summary_only"
            else len(run_results) + len(failed_runs)
        ),
        "num_runs_completed": len(run_results),
        "num_runs_failed": len(failed_runs),
        "train_seeds": [int(item["train_seed"]) for item in run_entries],
        "run_entries": run_entries,
        "resume_train_mode": resume_train_mode,
        "resume_num_skipped_training": len(resume_completed_model_paths),
        "resume_num_training_required": len(resume_missing_entries),
        "resume_completed_model_paths": {
            str(seed): str(path)
            for seed, path in sorted(resume_completed_model_paths.items())
        },
        "finetune_source_run_dir": str(finetune_source_run_dir) if finetune_source_run_dir is not None else None,
        "finetune_source_model_paths": {
            str(seed): str(path)
            for seed, path in sorted(finetune_source_model_paths.items())
        },
        "summary_rows": summary_rows,
        "aggregate": aggregate,
        "combo_summary": combo_summary,
        "failed_runs": failed_runs,
        "runs": run_results,
    }
    test_summary_prefix = _test_summary_prefix(test_run_name_prefix)
    test_summary_json_path = tests_root / f"{test_summary_prefix}_seed_sweep_summary.json"
    test_summary_csv_path = tests_root / f"{test_summary_prefix}_seed_sweep_summary.csv"
    test_summary_txt_path = tests_root / f"{test_summary_prefix}_seed_sweep_summary.txt"
    test_combo_summary_json_path = tests_root / f"{test_summary_prefix}_seed_sweep_adabn_combo_summary.json"
    test_combo_summary_csv_path = tests_root / f"{test_summary_prefix}_seed_sweep_adabn_combo_summary.csv"
    test_combo_summary_txt_path = tests_root / f"{test_summary_prefix}_seed_sweep_adabn_combo_summary.txt"

    _write_json(output_root / "seed_sweep_summary.json", summary_payload)
    _write_json(test_summary_json_path, summary_payload)
    _write_summary_csv(output_root / "seed_sweep_summary.csv", summary_rows)
    _write_summary_csv(test_summary_csv_path, summary_rows)
    _write_summary_txt(output_root / "seed_sweep_summary.txt", summary_rows, aggregate)
    _write_summary_txt(test_summary_txt_path, summary_rows, aggregate)
    if combo_summary is not None:
        _write_json(output_root / "seed_sweep_adabn_combo_summary.json", combo_summary)
        _write_json(test_combo_summary_json_path, combo_summary)
        _write_combo_summary_csv(output_root / "seed_sweep_adabn_combo_summary.csv", combo_summary.get("rows", []))
        _write_combo_summary_csv(test_combo_summary_csv_path, combo_summary.get("rows", []))
        _write_combo_summary_txt(output_root / "seed_sweep_adabn_combo_summary.txt", combo_summary)
        _write_combo_summary_txt(test_combo_summary_txt_path, combo_summary)

    local_staging_sync_stats = None
    local_staging_cleanup_stats = None
    if local_staging_enabled:
        print(
            "[Seed Sweep][local staging] Syncing results to the final output path: "
            f"{output_root} -> {final_output_root}"
        )
        local_staging_sync_stats = _copy_tree_for_staging_sync(
            src_root=output_root,
            dst_root=final_output_root,
            overwrite=True,
        )
        if local_staging_cleanup_after_sync:
            print(f"[Seed Sweep][local staging] Deleting local staging results: {output_root}")
            local_staging_cleanup_stats = _cleanup_local_staging_root_after_sync(
                local_root=output_root,
                final_root=final_output_root,
            )
        _write_json(
            final_output_root / "local_staging_sync.json",
            {
                "enabled": True,
                "work_output_root": str(output_root),
                "final_output_root": str(final_output_root),
                "stats": local_staging_sync_stats,
                "cleanup_after_sync": local_staging_cleanup_after_sync,
                "cleanup": local_staging_cleanup_stats,
            },
        )
        print(f"[Seed Sweep][local staging] sync={local_staging_sync_stats}")
        if local_staging_cleanup_stats is not None:
            print(f"[Seed Sweep][local staging] cleanup={local_staging_cleanup_stats}")

    print(f"[Seed Sweep complete] completed={len(run_results)}, failed={len(failed_runs)}")
    best_combo = combo_summary.get("best_combo_by_map50_mean") if combo_summary is not None else None
    if best_combo is not None:
        print(
            f"[Seed Sweep best combination] {best_combo['test_variant_name']} "
            f"(mAP50_mean={_safe_metric_str(best_combo.get('mAP50_mean'))}, "
            f"delta_vs_baseline={_safe_metric_str(best_combo.get('mAP50_delta_vs_baseline'))})"
        )
    reported_output_root = final_output_root if local_staging_enabled else output_root
    reported_tests_root = reported_output_root / "test_runs"
    reported_test_summary_json_path = reported_tests_root / f"{test_summary_prefix}_seed_sweep_summary.json"
    reported_test_summary_csv_path = reported_tests_root / f"{test_summary_prefix}_seed_sweep_summary.csv"
    reported_test_summary_txt_path = reported_tests_root / f"{test_summary_prefix}_seed_sweep_summary.txt"
    reported_test_combo_summary_json_path = (
        reported_tests_root / f"{test_summary_prefix}_seed_sweep_adabn_combo_summary.json"
    )
    reported_test_combo_summary_csv_path = (
        reported_tests_root / f"{test_summary_prefix}_seed_sweep_adabn_combo_summary.csv"
    )
    reported_test_combo_summary_txt_path = (
        reported_tests_root / f"{test_summary_prefix}_seed_sweep_adabn_combo_summary.txt"
    )

    print(f"[Seed Sweep summary] json={reported_output_root / 'seed_sweep_summary.json'}")
    print(f"[Seed Sweep summary] csv={reported_output_root / 'seed_sweep_summary.csv'}")
    print(f"[Seed Sweep summary] txt={reported_output_root / 'seed_sweep_summary.txt'}")
    if local_staging_enabled:
        print(f"[Seed Sweep final output] root={final_output_root}")
        print(f"[Seed Sweep final output] sync_json={final_output_root / 'local_staging_sync.json'}")
    print(f"[Seed Sweep test directory summary] json={reported_test_summary_json_path}")
    print(f"[Seed Sweep test directory summary] csv={reported_test_summary_csv_path}")
    print(f"[Seed Sweep test directory summary] txt={reported_test_summary_txt_path}")
    if combo_summary is not None:
        print(f"[Seed Sweep combination comparison] json={reported_output_root / 'seed_sweep_adabn_combo_summary.json'}")
        print(f"[Seed Sweep combination comparison] csv={reported_output_root / 'seed_sweep_adabn_combo_summary.csv'}")
        print(f"[Seed Sweep combination comparison] txt={reported_output_root / 'seed_sweep_adabn_combo_summary.txt'}")
        print(f"[Seed Sweep test directory combination comparison] json={reported_test_combo_summary_json_path}")
        print(f"[Seed Sweep test directory combination comparison] csv={reported_test_combo_summary_csv_path}")
        print(f"[Seed Sweep test directory combination comparison] txt={reported_test_combo_summary_txt_path}")


if __name__ == "__main__":
    main()
