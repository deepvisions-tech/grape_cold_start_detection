from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from grape_detector.utils.box_ops import compute_iou, xywhn_to_xyxy
from grape_detector.utils.config_loader import as_bool, normalize_class_names, parse_class_ids


def load_gt_box_records(
    label_path: Path,
    img_w: int,
    img_h: int,
) -> List[Dict[str, Any]]:
    """
    Read all GT labels from a TXT file and return absolute xyxy boxes with class IDs.
    """
    gt_records: List[Dict[str, Any]] = []

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        box = list(map(float, parts[1:5]))
        xyxy = xywhn_to_xyxy(box, img_w, img_h)
        gt_records.append(
            {
                "class_id": cls_id,
                "xyxy": xyxy,
            }
        )

    return gt_records


def validate_gt_label_coverage(image_paths: Sequence[Path], label_dir: Path) -> None:
    """Require explicit labels for evaluation; an empty file denotes a negative."""
    if not label_dir.is_dir():
        raise FileNotFoundError(f"TEST_LABEL_DIR does not exist or is not a directory: {label_dir}")
    missing = [
        label_dir / f"{path.stem}.txt"
        for path in image_paths
        if not (label_dir / f"{path.stem}.txt").is_file()
    ]
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"Missing GT label files for {len(missing)} evaluation images: {preview}. "
            "Every evaluation image needs a label file; use an empty .txt file "
            "only for a verified negative image."
        )


def get_class_name(class_id: int, class_names: Dict[int, str]) -> str:
    return class_names.get(int(class_id), f"class_{int(class_id)}")


def resolve_eval_class_ids(cfg: Dict[str, Any]) -> Tuple[List[int], Dict[int, str]]:
    class_names = normalize_class_names(cfg.get("names"))
    configured_classes = parse_class_ids(
        cfg.get("TEST_EVAL_CLASSES", cfg.get("TEST_TARGET_CLASSES", cfg.get("EVAL_CLASSES"))),
        class_names,
    )
    if configured_classes is not None:
        if not configured_classes:
            raise ValueError("TEST_EVAL_CLASSES must not be empty.")
        return configured_classes, class_names

    if len(class_names) > 1:
        return sorted(class_names.keys()), class_names

    return [int(cfg.get("TARGET_GT_CLASS_FOR_SINGLE_EVAL", 0))], class_names


def resolve_prediction_class_mapping(
    cfg: Dict[str, Any],
    eval_class_ids: List[int],
    class_names: Dict[int, str],
) -> Optional[Dict[int, int]]:
    raw_mapping = cfg.get("TEST_PRED_TO_EVAL_CLASS_MAP")
    if isinstance(raw_mapping, dict):
        return {
            int(pred_class): int(eval_class)
            for pred_class, eval_class in raw_mapping.items()
        }

    if len(eval_class_ids) == 1 and len(class_names) <= 1:
        pred_class = int(cfg.get("TEST_PRED_CLASS_FOR_SINGLE_EVAL", 0))
        return {pred_class: int(eval_class_ids[0])}

    return None


def map_prediction_class_id(
    pred_class_id: int,
    *,
    eval_class_set: Set[int],
    prediction_class_mapping: Optional[Dict[int, int]],
) -> Optional[int]:
    pred_class_id = int(pred_class_id)
    if prediction_class_mapping is not None:
        mapped_class_id = prediction_class_mapping.get(pred_class_id)
        if mapped_class_id is None:
            return None
        mapped_class_id = int(mapped_class_id)
        return mapped_class_id if mapped_class_id in eval_class_set else None
    return pred_class_id if pred_class_id in eval_class_set else None


def greedy_match_prediction_records_to_gt_records_by_class(
    pred_boxes: List[List[float]],
    pred_classes: List[int],
    gt_records: List[Dict[str, Any]],
    iou_thr: float = 0.5,
) -> List[Dict[str, Any]]:
    """Return class-aware 1:1 prediction/GT matches with their IoUs."""
    matched_gt: Set[int] = set()
    matched_pred: Set[int] = set()

    candidates = []
    for pred_index, pred_box in enumerate(pred_boxes):
        pred_class = int(pred_classes[pred_index])
        for gt_index, gt_record in enumerate(gt_records):
            if pred_class != int(gt_record["class_id"]):
                continue
            iou = compute_iou(pred_box, gt_record["xyxy"])
            if iou >= iou_thr:
                candidates.append((iou, pred_index, gt_index))

    candidates.sort(key=lambda item: item[0], reverse=True)

    matches: List[Dict[str, Any]] = []
    for iou, pred_index, gt_index in candidates:
        if pred_index in matched_pred or gt_index in matched_gt:
            continue
        matched_pred.add(pred_index)
        matched_gt.add(gt_index)
        matches.append(
            {
                "prediction_index": int(pred_index),
                "gt_index": int(gt_index),
                "iou": float(iou),
            }
        )

    return matches


def greedy_match_predictions_to_gt(
    pred_boxes: List[List[float]],
    gt_boxes: List[List[float]],
    iou_thr: float = 0.5,
) -> Tuple[int, int, int]:
    """
    1:1 greedy matching
    return: TP, FP, FN
    """
    matched_gt = set()
    matched_pred = set()

    candidates = []
    for pi, pbox in enumerate(pred_boxes):
        for gi, gtbox in enumerate(gt_boxes):
            iou = compute_iou(pbox, gtbox)
            if iou >= iou_thr:
                candidates.append((iou, pi, gi))

    candidates.sort(key=lambda x: x[0], reverse=True)

    for iou, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)

    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn


def get_optional_float(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null"}:
            return default
        return float(text)
    return float(value)


def apply_f1_optimized_conf_offset(base_conf, cfg, conf_source):
    offset = get_optional_float(cfg.get("TEST_F1_OPTIMIZED_CONF_OFFSET"), 0.0)
    adjusted_conf = min(1.0, max(0.0, float(base_conf) + offset))
    if offset != 0.0:
        print(
            "[TEST_CONF] Applying the F1-optimized confidence offset: "
            f"base={float(base_conf):.4f}, offset={offset:+.4f}, applied={adjusted_conf:.4f}"
        )
        conf_source = f"{conf_source}+offset({offset:+.4f})"
    return adjusted_conf, conf_source, offset


def set_current_cuda_device_if_needed(device):
    try:
        import torch
    except ImportError:
        return

    device_text = str(device).strip()
    if not device_text or device_text.lower() == "cpu":
        return
    primary = device_text.split(",")[0].strip()
    if primary.lower().startswith("cuda:"):
        primary = primary.split(":", 1)[1]
    if not primary.isdigit():
        return
    if not torch.cuda.is_available():
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1" and "," not in visible:
        primary = "0"
    torch.cuda.set_device(int(primary))


def configure_test_cuda_visibility(cfg):
    requested_device = str(cfg.get("TEST_DEVICE", "0")).strip()
    limit_visible_device = as_bool(cfg.get("TEST_LIMIT_VISIBLE_DEVICE", False), default=False)
    info = {
        "enabled": limit_visible_device,
        "requested_device": requested_device,
        "effective_test_device": requested_device,
        "cuda_visible_devices_before": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_visible_devices_after": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "applied": False,
        "warning": None,
    }

    if not limit_visible_device:
        return requested_device, info

    if not requested_device or requested_device.lower() == "cpu":
        info["warning"] = "Skipping GPU visibility limits because TEST_DEVICE is cpu."
        return requested_device, info

    if "," in requested_device:
        info["warning"] = (
            "TEST_LIMIT_VISIBLE_DEVICE requires a single GPU index. "
            f"Current TEST_DEVICE={requested_device}"
        )
        return requested_device, info

    os.environ["CUDA_VISIBLE_DEVICES"] = requested_device
    info["cuda_visible_devices_after"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    info["effective_test_device"] = "0"
    info["applied"] = True
    if "torch" in sys.modules or "ultralytics" in sys.modules:
        info["warning"] = (
            "CUDA visibility changes may not take full effect because torch or ultralytics has already been imported."
        )
    return "0", info


def remap_single_visible_gpu_device(device_value, visibility_info, fallback_device):
    if device_value is None or str(device_value).strip() == "":
        return fallback_device

    device_text = str(device_value).strip()
    if device_text.lower() == "cpu":
        return device_text

    if not visibility_info.get("applied", False):
        return device_text

    if "," in device_text:
        print(
            "[TEST_DEVICE_LOCK][WARN] Single-GPU visibility is enabled; "
            f"leaving the multi-device setting ({device_text}) unchanged may cause instability."
        )
        return device_text

    return "0"


def get_optional_int(cfg, key, default=None):
    value = cfg.get(key, default)
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return default
    return int(value)


def get_adabn_momentum_value(cfg, key):
    sentinel = "use_original"
    raw_value = cfg.get(key, sentinel)
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
