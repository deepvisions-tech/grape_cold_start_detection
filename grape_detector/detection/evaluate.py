import argparse
import gc
import math
from copy import deepcopy
from pathlib import Path
import json
import shutil
from typing import Any, Dict, List, Optional, Set
from tqdm import tqdm

import cv2
import numpy as np

from grape_detector.utils.adabn_grid import apply_adabn_grid_variant, build_adabn_grid_variants
from grape_detector.utils.coco_detection_metrics import compute_coco_detection_metrics
from grape_detector.utils.config_loader import as_bool, load_config, parse_class_ids, resolve_tqdm_disable
from grape_detector.utils.deployment_context import (
    build_deployment_reuse_context,
    deployment_context_mismatches,
)
from grape_detector.utils.gamma_correction import apply_gamma_correction, resolve_gamma_correction_config
from grape_detector.utils.run_path_utils import resolve_named_output_dir, validate_output_separation
from grape_detector.utils.single_class_detection_metrics import compute_single_class_ap
from grape_detector.utils.tiled_detection_inference import infer_image_with_tiling

from ._eval_helpers import (
    load_gt_box_records,
    validate_gt_label_coverage,
    get_class_name,
    resolve_eval_class_ids,
    resolve_prediction_class_mapping,
    map_prediction_class_id,
    greedy_match_prediction_records_to_gt_records_by_class,
    greedy_match_predictions_to_gt,
    get_optional_float,
    apply_f1_optimized_conf_offset,
    set_current_cuda_device_if_needed,
    configure_test_cuda_visibility,
    remap_single_visible_gpu_device,
    get_optional_int,
    get_adabn_momentum_value,
)
from .visualization_output import (
    prepare_eval_visualization_dirs,
    resolve_eval_visualization_path,
)
from .performance_output import (
    build_auto_labeling_performance_record,
    write_per_image_result_json,
)

def draw_eval_result(
    image,
    pred_boxes,
    pred_scores,
    gt_boxes,
    save_path: Path,
    pred_classes=None,
    gt_classes=None,
    class_names: Optional[Dict[int, str]] = None,
):
    """
    GT: green, predictions: red.
    """
    vis = image.copy()
    class_names = class_names or {}
    pred_classes = pred_classes if pred_classes is not None else [None] * len(pred_boxes)
    gt_classes = gt_classes if gt_classes is not None else [None] * len(gt_boxes)

    for box, class_id in zip(gt_boxes, gt_classes):
        x1, y1, x2, y2 = map(int, box)
        class_label = "GT" if class_id is None else f"GT_{get_class_name(int(class_id), class_names)}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis, class_label, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    for box, score, class_id in zip(pred_boxes, pred_scores, pred_classes):
        x1, y1, x2, y2 = map(int, box)
        class_label = "Pred" if class_id is None else f"Pred_{get_class_name(int(class_id), class_names)}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            vis, f"{class_label} {float(score):.2f}", (x1, min(vis.shape[0] - 5, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)



def collect_debug_prediction_probe(
    model,
    image_paths,
    *,
    imgsz,
    conf,
    iou,
    max_det,
    device,
    half=False,
    tiled_infer_enabled=False,
    tiled_merge_iou=None,
    tile_select_box_from_middle=None,
    gamma_correction_enabled=False,
    gamma_correction_value=0.6,
):
    if not image_paths:
        return []

    records = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        if gamma_correction_enabled:
            image = apply_gamma_correction(image, gamma=gamma_correction_value)

        prediction = infer_image_with_tiling(
            model,
            image,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            half=half,
            tile_enabled=tiled_infer_enabled,
            tile_merge_iou=tiled_merge_iou,
            tile_select_box_from_middle=tile_select_box_from_middle,
        )
        scores = [float(score) for score in prediction["scores"].tolist()]
        records.append(
            {
                "image_path": str(image_path),
                "num_predictions": len(scores),
                "max_conf": max(scores) if scores else None,
                "mean_conf": (sum(scores) / len(scores)) if scores else None,
                "top3_conf": sorted(scores, reverse=True)[:3],
            }
        )
    return records


def build_debug_probe_delta(before_records, after_records):
    before_by_path = {item["image_path"]: item for item in before_records}
    delta_records = []
    for after_item in after_records:
        before_item = before_by_path.get(after_item["image_path"])
        if before_item is None:
            continue
        delta_records.append(
            {
                "image_path": after_item["image_path"],
                "before_num_predictions": before_item["num_predictions"],
                "after_num_predictions": after_item["num_predictions"],
                "before_max_conf": before_item["max_conf"],
                "after_max_conf": after_item["max_conf"],
                "delta_num_predictions": after_item["num_predictions"] - before_item["num_predictions"],
                "delta_max_conf": (
                    (after_item["max_conf"] - before_item["max_conf"])
                    if after_item["max_conf"] is not None and before_item["max_conf"] is not None
                    else None
                ),
            }
        )
    return delta_records


def xyxy_to_coco_bbox(box: List[float], width: int, height: int) -> List[float]:
    x1 = min(max(float(box[0]), 0.0), float(width))
    y1 = min(max(float(box[1]), 0.0), float(height))
    x2 = min(max(float(box[2]), x1), float(width))
    y2 = min(max(float(box[3]), y1), float(height))
    return [x1, y1, x2 - x1, y2 - y1]


def normalize_coco_subset_image_dirs(raw_value: Any) -> Dict[str, Path]:
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise ValueError("TEST_COCO_SUBSET_IMAGE_DIRS must be a name-to-directory mapping")

    resolved: Dict[str, Path] = {}
    for raw_name, raw_path in raw_value.items():
        name = str(raw_name).strip()
        path_text = str(raw_path).strip()
        if not name or not path_text:
            raise ValueError("TEST_COCO_SUBSET_IMAGE_DIRS contains an empty name or path")
        path = Path(path_text).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"COCO subset image directory does not exist: {path}")
        resolved[name] = path
    return resolved


def _validate_test_inputs(cfg, output_dir):
    from grape_detector.utils.adabn import list_depth1_images

    protected = {key: cfg.get(key) for key in (
        "TEST_MODEL_PATH", "TEST_IMAGE_DIR", "TEST_LABEL_DIR",
        "TEST_ADABN_IMAGE_DIR", "TEST_TUNE_SPLIT_DIR",
        "TEST_DEPLOYMENT_BACKGROUND_IMAGE_DIR",
    )}
    if as_bool(cfg.get("TEST_COCO_EVAL_ENABLED", False), default=False):
        protected.update(normalize_coco_subset_image_dirs(cfg.get("TEST_COCO_SUBSET_IMAGE_DIRS")))
    validate_output_separation(output_dir, protected)
    image_paths = list_depth1_images(Path(cfg["TEST_IMAGE_DIR"]).expanduser().resolve())
    if not image_paths:
        raise RuntimeError(f"No test images found: {cfg['TEST_IMAGE_DIR']}")
    raw_label_dir = cfg.get("TEST_LABEL_DIR")
    label_dir = None
    if raw_label_dir is not None and str(raw_label_dir).strip():
        label_dir = Path(str(raw_label_dir)).expanduser().resolve()
        validate_gt_label_coverage(image_paths, label_dir)
    return image_paths, label_dir


def run_test_with_config(cfg):
    device, device_lock_info = configure_test_cuda_visibility(cfg)
    from grape_detector.utils.adabn import apply_adabn_to_yolo_model, list_depth1_images
    from grape_detector.utils.f1_conf_optimizer import (
        _list_image_paths,
        load_tuned_conf_metadata,
        optimize_conf_for_validation,
        optimize_deployment_threshold_for_background,
    )
    from ultralytics import YOLO

    model_path = Path(cfg["TEST_MODEL_PATH"]).expanduser().resolve()
    image_dir = Path(cfg["TEST_IMAGE_DIR"]).expanduser()
    output_dir = resolve_named_output_dir(
        cfg,
        "TEST_OUTPUT_DIR",
        "TEST_RUN_NAME",
        resolved_flag_key="_TEST_OUTPUT_DIR_RESOLVED",
    )

    # Check coverage before threshold tuning, model loading or output cleanup.
    image_paths, label_dir = _validate_test_inputs(cfg, output_dir)

    imgsz = int(cfg.get("TEST_IMGSZ", 960))
    default_conf_thr = float(cfg.get("TEST_CONF", 0.25))
    ap_eval_conf_thr = float(cfg.get("TEST_AP_EVAL_CONF", 0.001))
    pred_iou = float(cfg.get("TEST_IOU", 0.7))
    max_det = int(cfg.get("TEST_MAX_DET", 300))
    coco_eval_enabled = as_bool(cfg.get("TEST_COCO_EVAL_ENABLED", False), default=False)
    coco_max_dets = int(cfg.get("TEST_COCO_MAX_DETS", 100))
    coco_subset_image_dirs = normalize_coco_subset_image_dirs(
        cfg.get("TEST_COCO_SUBSET_IMAGE_DIRS")
    ) if coco_eval_enabled else {}
    coco_subset_strict = as_bool(cfg.get("TEST_COCO_SUBSET_STRICT", True), default=True)
    if coco_eval_enabled and ap_eval_conf_thr > 0.001:
        raise ValueError(
            "COCO AP evaluation requires TEST_AP_EVAL_CONF <= 0.001 so low-score predictions are retained"
        )
    if coco_eval_enabled and max_det < coco_max_dets:
        raise ValueError(
            "TEST_MAX_DET must be >= TEST_COCO_MAX_DETS for COCO AP evaluation"
        )
    test_fp16 = as_bool(cfg.get("TEST_FP16", False), default=False)
    gamma_config = resolve_gamma_correction_config(cfg, prefix="TEST")
    save_jsonl = as_bool(cfg.get("TEST_SAVE_JSONL", True), default=True)
    save_visualizations = as_bool(cfg.get("TEST_SAVE_VISUALIZATIONS", True), default=True)
    save_auto_labeling_compatible_results = as_bool(
        cfg.get("TEST_SAVE_AUTO_LABELING_COMPATIBLE_RESULTS", False),
        default=False,
    )
    if save_auto_labeling_compatible_results and label_dir is None:
        raise ValueError(
            "When TEST_SAVE_AUTO_LABELING_COMPATIBLE_RESULTS=true, "
            "a valid TEST_LABEL_DIR is required."
        )
    tiled_infer_enabled = as_bool(cfg.get("TEST_TILED_INFER_ENABLED", False), default=False)
    tiled_merge_iou = float(cfg.get("TEST_TILED_MERGE_IOU", pred_iou))
    tile_select_box_from_middle = get_optional_float(cfg.get("TEST_TILED_SELECT_BOX_FROM_MIDDLE"), None)
    if gamma_config["enabled"]:
        print(f"[TEST] gamma correction enabled: gamma={gamma_config['gamma']}")
    disable_tqdm = resolve_tqdm_disable(
        cfg,
        "TEST_DISABLE_TQDM",
        "TEST_TQDM_DISABLE",
        "DISABLE_TQDM",
        default=False,
    )
    set_current_cuda_device_if_needed(device)
    if device_lock_info.get("enabled", False):
        print("[TEST_DEVICE_LOCK] ========================================================")
        print(f"[TEST_DEVICE_LOCK] requested TEST_DEVICE : {device_lock_info.get('requested_device')}")
        print(f"[TEST_DEVICE_LOCK] effective TEST_DEVICE : {device}")
        print(
            f"[TEST_DEVICE_LOCK] CUDA_VISIBLE_DEVICES : "
            f"{device_lock_info.get('cuda_visible_devices_before')} -> "
            f"{device_lock_info.get('cuda_visible_devices_after')}"
        )
        if device_lock_info.get("warning"):
            print(f"[TEST_DEVICE_LOCK][WARN] {device_lock_info.get('warning')}")
        print("[TEST_DEVICE_LOCK] ========================================================")

    # Batch inference settings
    test_batch = int(cfg.get("TEST_BATCH", 1))

    # Evaluation IoU threshold for GT matching
    eval_iou_thr = float(cfg.get("EVAL_IOU_THRESHOLD", 0.5))

    eval_class_ids, class_names = resolve_eval_class_ids(cfg)
    eval_class_set = set(int(class_id) for class_id in eval_class_ids)
    prediction_class_mapping = resolve_prediction_class_mapping(cfg, eval_class_ids, class_names)
    target_gt_class = int(eval_class_ids[0])
    is_multiclass_eval = len(eval_class_ids) > 1

    conf_thr = default_conf_thr
    conf_source = "config:TEST_CONF"
    tuned_conf_metadata = None
    tuned_conf_base = None
    f1_resolved_conf = None
    tuned_conf_offset = get_optional_float(cfg.get("TEST_F1_OPTIMIZED_CONF_OFFSET"), 0.0)

    if as_bool(cfg.get("TEST_USE_F1_OPTIMIZED_CONF", False), default=False):
        tuned_conf_metadata = load_tuned_conf_metadata(model_path)
        saved_f1_conf = (
            get_optional_float(tuned_conf_metadata.get("best_conf"), None)
            if tuned_conf_metadata is not None
            else None
        )
        overwrite_tuned_conf = as_bool(cfg.get("TEST_OVERWRITE_F1_OPTIMIZED_CONF", False), default=False)
        tune_if_missing = as_bool(cfg.get("TEST_AUTO_TUNE_F1_CONF_IF_MISSING", False), default=False)
        split_dir_value = cfg.get("TEST_TUNE_SPLIT_DIR")
        tune_split_name = str(cfg.get("TEST_TUNE_SPLIT", "val")).strip()

        should_tune_now = False
        if overwrite_tuned_conf:
            should_tune_now = True
        elif saved_f1_conf is None and tune_if_missing:
            should_tune_now = True

        if should_tune_now:
            if not split_dir_value and as_bool(cfg.get("TEST_REQUIRE_TUNED_CONF", False), default=False):
                raise ValueError("The paper recipe requires TEST_TUNE_SPLIT_DIR for validation-F1 tuning.")
            if split_dir_value:
                tuned_conf_metadata = optimize_conf_for_validation(
                    model_path=model_path,
                    split_dir=Path(split_dir_value),
                    split_name=tune_split_name,
                    imgsz=int(cfg.get("TEST_TUNE_IMGSZ", imgsz)),
                    pred_iou=float(cfg.get("TEST_TUNE_PRED_IOU", pred_iou)),
                    max_det=int(cfg.get("TEST_TUNE_MAX_DET", max_det)),
                    device=remap_single_visible_gpu_device(
                        cfg.get("TEST_TUNE_DEVICE"),
                        device_lock_info,
                        device,
                    ),
                    batch_size=int(cfg.get("TEST_TUNE_BATCH", test_batch)),
                    eval_iou_thr=float(cfg.get("TEST_TUNE_EVAL_IOU_THRESHOLD", eval_iou_thr)),
                    target_gt_class=int(cfg.get("TEST_TUNE_TARGET_GT_CLASS", 0)),
                    target_classes=parse_class_ids(
                        cfg.get("TEST_TUNE_TARGET_CLASSES"),
                        class_names,
                    ) or eval_class_ids,
                    prediction_class_mapping=prediction_class_mapping,
                    conf_min=float(cfg.get("TEST_TUNE_CONF_MIN", 0.01)),
                    conf_max=float(cfg.get("TEST_TUNE_CONF_MAX", 0.99)),
                    conf_step=float(cfg.get("TEST_TUNE_CONF_STEP", 0.01)),
                    save_metadata=True,
                    gamma_correction_enabled=gamma_config["enabled"],
                    gamma_correction_value=gamma_config["gamma"],
                )
                tuned_conf_base = float(tuned_conf_metadata["best_conf"])
                conf_thr = tuned_conf_base
                conf_source = "overwritten_f1_tuned_conf" if overwrite_tuned_conf else "auto_tuned_from_split"
                print(
                    f"[TEST_CONF] {tune_split_name} split: saving and using F1-optimized confidence: "
                    f"{tuned_conf_base:.4f} ({tuned_conf_metadata.get('metadata_path', model_path)})"
                )
            else:
                print("[WARN] TEST_TUNE_SPLIT_DIR is not set; skipping confidence retuning and using the existing settings.")
                if saved_f1_conf is not None:
                    tuned_conf_base = float(saved_f1_conf)
                    conf_thr = tuned_conf_base
                    conf_source = "saved_f1_tuned_conf"
        elif saved_f1_conf is not None:
            tuned_conf_base = float(saved_f1_conf)
            conf_thr = tuned_conf_base
            conf_source = "saved_f1_tuned_conf"
            print(
                f"[TEST_CONF] Using saved F1-optimized confidence: {tuned_conf_base:.4f} "
                f"({tuned_conf_metadata.get('metadata_path', model_path)})"
            )
        else:
            print("[INFO] No saved tuned confidence found; using TEST_CONF.")

    if tuned_conf_base is not None:
        conf_thr, conf_source, tuned_conf_offset = apply_f1_optimized_conf_offset(
            tuned_conf_base,
            cfg,
            conf_source,
        )
        f1_resolved_conf = conf_thr

    deployment_tune_enabled = as_bool(
        cfg.get("TEST_TUNE_DEPLOYMENT_THRESHOLD", False),
        default=False,
    )
    deployment_use_enabled = as_bool(
        cfg.get("TEST_USE_DEPLOYMENT_THRESHOLD", False),
        default=False,
    )
    deployment_overwrite = as_bool(
        cfg.get("TEST_OVERWRITE_DEPLOYMENT_THRESHOLD", False),
        default=False,
    )
    deployment_count_fp_by_image = as_bool(
        cfg.get("TEST_DEPLOYMENT_COUNT_FP_BY_IMAGE", True),
        default=True,
    )
    deployment_fp_unit = "image" if deployment_count_fp_by_image else "object"
    deployment_threshold = None
    deployment_tuned_now = False
    deployment_reuse_mismatches = []

    # Curated validation tuning above uses the unadapted checkpoint.
    # Deployment tuning below reuses this model after AdaBN.
    model = YOLO(str(model_path))
    end2end_inference = bool(getattr(model.model, "end2end", False))
    adabn_debug_enabled = as_bool(cfg.get("TEST_ADABN_DEBUG", False), default=False)
    adabn_debug_probe = None
    if as_bool(cfg.get("TEST_ADABN_ENABLED", False), default=False) and adabn_debug_enabled:
        debug_sample_count = get_optional_int(cfg, "TEST_ADABN_DEBUG_SAMPLE_COUNT", 4)
        debug_probe_paths = list_depth1_images(image_dir.resolve())[:debug_sample_count]
        probe_model = YOLO(str(model_path))
        adabn_debug_probe = {
            "probe_image_paths": [str(path) for path in debug_probe_paths],
            "before_predictions": collect_debug_prediction_probe(
                probe_model,
                debug_probe_paths,
                imgsz=imgsz,
                conf=ap_eval_conf_thr,
                iou=pred_iou,
                max_det=max_det,
                device=device,
                half=test_fp16,
                tiled_infer_enabled=tiled_infer_enabled,
                tiled_merge_iou=tiled_merge_iou,
                tile_select_box_from_middle=tile_select_box_from_middle,
                gamma_correction_enabled=gamma_config["enabled"],
                gamma_correction_value=gamma_config["gamma"],
            ),
        }
        del probe_model
    adabn_info = {
        "enabled": as_bool(cfg.get("TEST_ADABN_ENABLED", False), default=False),
        "applied": False,
        "status": "disabled",
    }
    if adabn_info["enabled"]:
        adabn_image_dir = cfg.get("TEST_ADABN_IMAGE_DIR")
        if not adabn_image_dir:
            raise ValueError("TEST_ADABN_IMAGE_DIR is required when TEST_ADABN_ENABLED=true.")
        adabn_info = apply_adabn_to_yolo_model(
            model=model,
            image_dir=Path(str(adabn_image_dir)),
            imgsz=get_optional_int(cfg, "TEST_ADABN_IMGSZ", imgsz),
            batch_size=get_optional_int(cfg, "TEST_ADABN_BATCH", test_batch),
            device=device,
            max_images=get_optional_int(cfg, "TEST_ADABN_MAX_IMAGES"),
            sample_seed=get_optional_int(cfg, "TEST_ADABN_SAMPLE_SEED"),
            epochs=get_optional_int(cfg, "TEST_ADABN_EPOCHS", 1),
            max_stage=get_optional_int(cfg, "TEST_ADABN_MAX_STAGE"),
            reset_stats=as_bool(cfg.get("TEST_ADABN_RESET_STATS", True), default=True),
            momentum_override=get_adabn_momentum_value(cfg, "TEST_ADABN_MOMENTUM"),
            use_fp16=as_bool(cfg.get("TEST_ADABN_FP16", False), default=False),
            debug=adabn_debug_enabled,
            debug_max_batches=get_optional_int(cfg, "TEST_ADABN_DEBUG_MAX_BATCHES", 1),
            disable_tqdm=disable_tqdm,
        )
        if adabn_debug_probe is not None:
            adabn_debug_probe["after_predictions"] = collect_debug_prediction_probe(
                model,
                [Path(path) for path in adabn_debug_probe["probe_image_paths"]],
                imgsz=imgsz,
                conf=ap_eval_conf_thr,
                iou=pred_iou,
                max_det=max_det,
                device=device,
                half=test_fp16,
                tiled_infer_enabled=tiled_infer_enabled,
                tiled_merge_iou=tiled_merge_iou,
                tile_select_box_from_middle=tile_select_box_from_middle,
                gamma_correction_enabled=gamma_config["enabled"],
                gamma_correction_value=gamma_config["gamma"],
            )
            adabn_debug_probe["prediction_delta"] = build_debug_probe_delta(
                adabn_debug_probe["before_predictions"],
                adabn_debug_probe["after_predictions"],
            )
            adabn_info["debug_probe"] = adabn_debug_probe

    # Deployment tuning must observe the exact model state used by the test.
    # In particular, when TEST_ADABN_ENABLED=true, the same adapted model
    # instance is tuned here and then reused for the actual test inference.
    if deployment_tune_enabled or deployment_use_enabled:
        if tuned_conf_metadata is None:
            tuned_conf_metadata = load_tuned_conf_metadata(model_path)
        deployment_threshold = (
            get_optional_float(tuned_conf_metadata.get("deployment_threshold"), None)
            if tuned_conf_metadata is not None
            else None
        )
        background_image_dir = cfg.get("TEST_DEPLOYMENT_BACKGROUND_IMAGE_DIR")
        if background_image_dir is None or not str(background_image_dir).strip():
            raise ValueError(
                "TEST_DEPLOYMENT_BACKGROUND_IMAGE_DIR is required to calibrate or "
                "validate a saved deployment threshold."
            )
        background_image_dir = Path(str(background_image_dir)).expanduser().resolve()
        deployment_target_classes = sorted(set(parse_class_ids(
            cfg.get("TEST_DEPLOYMENT_TUNE_CLASSES"), class_names,
        ) or eval_class_ids))
        # One resolved option set drives both calibration and its reuse check.
        calibration_options = {
            "image_dir": background_image_dir,
            "max_fp": int(cfg.get("TEST_DEPLOYMENT_MAX_FP", 0)),
            "fp_unit": deployment_fp_unit,
            "imgsz": int(cfg.get("TEST_DEPLOYMENT_TUNE_IMGSZ", imgsz)),
            "pred_iou": float(cfg.get("TEST_DEPLOYMENT_TUNE_PRED_IOU", pred_iou)),
            "max_det": int(cfg.get("TEST_DEPLOYMENT_TUNE_MAX_DET", max_det)),
            "device": str(remap_single_visible_gpu_device(
                cfg.get("TEST_DEPLOYMENT_TUNE_DEVICE"), device_lock_info, device,
            )),
            "batch_size": max(1, int(cfg.get("TEST_DEPLOYMENT_TUNE_BATCH", test_batch))),
            "target_classes": deployment_target_classes,
            "prediction_class_mapping": prediction_class_mapping,
            "use_fp16": as_bool(cfg.get("TEST_DEPLOYMENT_TUNE_FP16", test_fp16), default=test_fp16),
            "tiled_infer_enabled": as_bool(
                cfg.get("TEST_DEPLOYMENT_TUNE_TILED_INFER_ENABLED", tiled_infer_enabled),
                default=tiled_infer_enabled,
            ),
            "tiled_merge_iou": get_optional_float(
                cfg.get("TEST_DEPLOYMENT_TUNE_TILED_MERGE_IOU", tiled_merge_iou), tiled_merge_iou,
            ),
            "tile_select_box_from_middle": get_optional_float(
                cfg.get("TEST_DEPLOYMENT_TUNE_TILED_SELECT_BOX_FROM_MIDDLE", tile_select_box_from_middle),
                tile_select_box_from_middle,
            ),
            "disable_tqdm": as_bool(
                cfg.get("TEST_DEPLOYMENT_TUNE_DISABLE_TQDM", disable_tqdm), default=disable_tqdm,
            ),
            "conf_min": float(cfg.get("TEST_DEPLOYMENT_TUNE_CONF_MIN", 0.01)),
            "conf_max": float(cfg.get("TEST_DEPLOYMENT_TUNE_CONF_MAX", 0.99)),
            "conf_step": float(cfg.get("TEST_DEPLOYMENT_TUNE_CONF_STEP", 0.01)),
            "gamma_correction_enabled": gamma_config["enabled"],
            "gamma_correction_value": gamma_config["gamma"],
        }
        reuse_context = build_deployment_reuse_context(
            model=model,
            adabn_info=adabn_info,
            calibration_options=calibration_options,
            test_settings={
                "imgsz": imgsz, "pred_iou": pred_iou, "max_det": max_det,
                "device": str(device), "batch_size": test_batch, "fp16": test_fp16,
                "tiled_infer_enabled": tiled_infer_enabled, "tiled_merge_iou": tiled_merge_iou,
                "tile_select_box_from_middle": tile_select_box_from_middle,
                "gamma_correction": gamma_config,
                "eval_class_ids": sorted(eval_class_ids),
                "prediction_class_mapping": prediction_class_mapping,
            },
            image_paths=_list_image_paths(background_image_dir),
        )
        saved_tuning = (tuned_conf_metadata or {}).get("deployment_tuning")
        saved_model_context = saved_tuning.get("model_context") if isinstance(saved_tuning, dict) else None
        saved_context = saved_model_context.get("reuse_context") if isinstance(saved_model_context, dict) else None
        deployment_reuse_mismatches = deployment_context_mismatches(saved_context, reuse_context)
        if deployment_threshold is None:
            deployment_reuse_mismatches.insert(0, "missing deployment threshold")
        elif not math.isfinite(deployment_threshold) or not 0 <= deployment_threshold <= 1:
            deployment_reuse_mismatches.insert(0, "invalid deployment threshold")

        should_tune_deployment = deployment_tune_enabled and (
            deployment_overwrite or bool(deployment_reuse_mismatches)
        )
        if should_tune_deployment:
            if deployment_reuse_mismatches:
                print("[DEPLOYMENT_CONF] Recalibrating: " + ", ".join(deployment_reuse_mismatches[:10]))
            print(
                "[DEPLOYMENT_CONF] Starting deployment threshold tuning: "
                f"AdaBN enabled={adabn_info.get('enabled', False)}, "
                f"applied={adabn_info.get('applied', False)}"
            )
            tuned_conf_metadata = optimize_deployment_threshold_for_background(
                model_path=model_path,
                **calibration_options,
                save_metadata=True,
                model=model,
                model_context={"adabn": reuse_context["adabn"], "reuse_context": reuse_context},
            )
            deployment_threshold = float(tuned_conf_metadata["deployment_threshold"])
            deployment_tuned_now = True
        elif deployment_reuse_mismatches:
            raise ValueError(
                "Cannot reuse deployment threshold: "
                + ", ".join(deployment_reuse_mismatches[:10])
                + ". Set TEST_TUNE_DEPLOYMENT_THRESHOLD=true to recalibrate for the current model and settings."
            )
        else:
            print("[DEPLOYMENT_CONF] Saved deployment threshold matches the current model, data and settings.")

        if deployment_use_enabled:
            conf_thr = float(deployment_threshold)
            conf_source = (
                "background_tuned_deployment_threshold"
                if deployment_tuned_now else "saved_deployment_threshold"
            )
            print(
                f"[DEPLOYMENT_CONF] Using deployment threshold: {conf_thr:.4f} "
                f"({tuned_conf_metadata.get('metadata_path', model_path)})"
            )

    # Invalid inputs/cache must not remove the previous evaluation outputs.
    if as_bool(cfg.get("TEST_RESET_OUTPUT_DIR", False), default=False) and output_dir.exists():
        print(f"[Output reset] Deleting the existing results directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dirs = prepare_eval_visualization_dirs(output_dir, enabled=save_visualizations)

    coco_category_id_by_class = {
        int(class_id): index + 1
        for index, class_id in enumerate(eval_class_ids)
    }
    coco_ground_truth: Dict[str, Any] = {
        "info": {
            "description": "Detection evaluation ground truth generated from YOLO labels",
            "metric_style": "COCOeval bbox",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": int(coco_category_id_by_class[int(class_id)]),
                "name": get_class_name(int(class_id), class_names),
                "supercategory": "symptom",
            }
            for class_id in eval_class_ids
        ],
    }
    coco_predictions: List[Dict[str, Any]] = []
    coco_image_id_by_name: Dict[str, int] = {}
    coco_annotation_id = 1
    total_tp = 0
    total_fp = 0
    total_fn = 0

    img_tp = 0
    img_tn = 0
    img_fp = 0
    img_fn = 0

    per_image_results = []
    ap_records_by_class: Dict[int, List[Dict[str, Any]]] = {
        int(class_id): []
        for class_id in eval_class_ids
    }
    per_class_totals: Dict[int, Dict[str, int]] = {
        int(class_id): {"tp": 0, "fp": 0, "fn": 0, "img_tp": 0, "img_tn": 0, "img_fp": 0, "img_fn": 0}
        for class_id in eval_class_ids
    }
    total_pred_boxes = 0
    total_pred_images = 0
    total_pred_boxes_before_merge = 0
    total_tile_windows = 0

    # Process images in batches
    pbar = tqdm(total=len(image_paths), desc="Running inference", disable=disable_tqdm)
    for batch_start in range(0, len(image_paths), test_batch):
        batch_paths = image_paths[batch_start:batch_start + test_batch]

        # Load and augment images in the batch
        batch_sources = []    # Sources passed to predict
        batch_meta = []       # (img_path, img, h, w, label_path, gt_box_records_all)

        for img_path in batch_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Failed to load image: {img_path}")
                if save_auto_labeling_compatible_results:
                    per_image_results.append(
                        {
                            "image_path": str(img_path),
                            "relative_path": img_path.name,
                            "image_name": img_path.name,
                            "output_visualization_path": None,
                            "num_boxes": 0,
                            "detections": [],
                            "performance": None,
                            "warning": "image_load_failed",
                        }
                    )
                continue

            predict_img = (
                apply_gamma_correction(img, gamma=gamma_config["gamma"])
                if gamma_config["enabled"]
                else img
            )
            h, w = img.shape[:2]
            label_path = None
            gt_box_records_all = []
            if label_dir is not None:
                label_path = label_dir / f"{img_path.stem}.txt"
                gt_box_records_all = load_gt_box_records(
                    label_path=label_path,
                    img_w=w,
                    img_h=h,
                )
            batch_sources.append(predict_img)
            batch_meta.append((img_path, img, h, w, label_path, gt_box_records_all))

        if not batch_sources:
            pbar.update(len(batch_paths))
            continue

        batch_predictions = []
        if tiled_infer_enabled:
            for predict_img in batch_sources:
                batch_predictions.append(
                    infer_image_with_tiling(
                        model,
                        predict_img,
                        imgsz=imgsz,
                        conf=ap_eval_conf_thr,
                        iou=pred_iou,
                        max_det=max_det,
                        device=device,
                        half=test_fp16,
                        tile_enabled=True,
                        tile_merge_iou=tiled_merge_iou,
                        tile_select_box_from_middle=tile_select_box_from_middle,
                    )
                )
        else:
            results = model.predict(
                source=batch_sources,
                imgsz=imgsz,
                conf=ap_eval_conf_thr,
                iou=pred_iou,
                max_det=max_det,
                device=device,
                half=test_fp16,
                verbose=False,
            )
            for result in results:
                batch_predictions.append(
                    {
                        "boxes_xyxy": (
                            result.boxes.xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
                            if result.boxes is not None and len(result.boxes) > 0
                            else np.zeros((0, 4), dtype=np.float32)
                        ),
                        "scores": (
                            result.boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
                            if result.boxes is not None and len(result.boxes) > 0
                            else np.zeros((0,), dtype=np.float32)
                        ),
                        "classes": (
                            result.boxes.cls.detach().cpu().numpy().astype(np.int32, copy=False)
                            if result.boxes is not None and len(result.boxes) > 0
                            else np.zeros((0,), dtype=np.int32)
                        ),
                        "num_predictions_before_merge": int(len(result.boxes)) if result.boxes is not None else 0,
                        "num_predictions_after_merge": int(len(result.boxes)) if result.boxes is not None else 0,
                        "tile_window_count": 0,
                    }
                )

        # Postprocess batch results
        for prediction, (img_path, img, h, w, label_path, gt_box_records_all) in zip(batch_predictions, batch_meta):
            pred_boxes = []
            pred_scores = []
            pred_classes = []
            all_pred_boxes = []
            all_pred_scores = []
            all_pred_classes = []
            xyxy = prediction["boxes_xyxy"]
            confs = prediction["scores"]
            classes = np.asarray(
                prediction.get("classes", np.zeros((len(xyxy),), dtype=np.int32)),
                dtype=np.int32,
            ).reshape(-1)
            target_gt_records = [
                record for record in gt_box_records_all
                if int(record["class_id"]) in eval_class_set
            ]
            target_gt_boxes = [record["xyxy"] for record in target_gt_records]
            target_gt_classes = [int(record["class_id"]) for record in target_gt_records]
            for box, score, class_id in zip(xyxy, confs, classes):
                score = float(score)
                mapped_class_id = map_prediction_class_id(
                    int(class_id),
                    eval_class_set=eval_class_set,
                    prediction_class_mapping=prediction_class_mapping,
                )
                if mapped_class_id is None:
                    continue
                box = box.tolist()
                all_pred_boxes.append(box)
                all_pred_scores.append(score)
                all_pred_classes.append(mapped_class_id)
                if score >= conf_thr:
                    pred_boxes.append(box)
                    pred_scores.append(score)
                    pred_classes.append(mapped_class_id)

            if coco_eval_enabled:
                if img_path.name in coco_image_id_by_name:
                    raise ValueError(
                        f"COCO evaluation requires unique image filenames, duplicate: {img_path.name}"
                    )
                coco_image_id = len(coco_ground_truth["images"]) + 1
                coco_image_id_by_name[img_path.name] = coco_image_id
                coco_ground_truth["images"].append(
                    {
                        "id": int(coco_image_id),
                        "file_name": img_path.name,
                        "width": int(w),
                        "height": int(h),
                    }
                )
                for gt_record in target_gt_records:
                    class_id = int(gt_record["class_id"])
                    bbox = xyxy_to_coco_bbox(gt_record["xyxy"], width=w, height=h)
                    coco_ground_truth["annotations"].append(
                        {
                            "id": int(coco_annotation_id),
                            "image_id": int(coco_image_id),
                            "category_id": int(coco_category_id_by_class[class_id]),
                            "bbox": bbox,
                            "area": float(bbox[2] * bbox[3]),
                            "iscrowd": 0,
                        }
                    )
                    coco_annotation_id += 1
                for box, score, class_id in zip(
                    all_pred_boxes,
                    all_pred_scores,
                    all_pred_classes,
                ):
                    coco_predictions.append(
                        {
                            "image_id": int(coco_image_id),
                            "category_id": int(coco_category_id_by_class[int(class_id)]),
                            "bbox": xyxy_to_coco_bbox(box, width=w, height=h),
                            "score": float(score),
                        }
                    )

            total_pred_boxes_before_merge += int(prediction.get("num_predictions_before_merge", len(all_pred_boxes)))
            total_tile_windows += int(prediction.get("tile_window_count", 0))
            total_pred_boxes += len(pred_boxes)
            if pred_boxes:
                total_pred_images += 1

            gt_available = label_dir is not None
            matched_pred_indices: Set[int] = set()
            matched_gt_indices: Set[int] = set()
            match_records: List[Dict[str, Any]] = []
            if gt_available:
                match_records = greedy_match_prediction_records_to_gt_records_by_class(
                    pred_boxes=pred_boxes,
                    pred_classes=pred_classes,
                    gt_records=target_gt_records,
                    iou_thr=eval_iou_thr,
                )
                matched_pred_indices = {
                    int(match["prediction_index"])
                    for match in match_records
                }
                matched_gt_indices = {
                    int(match["gt_index"])
                    for match in match_records
                }
                tp = len(matched_pred_indices)
                fp = len(pred_boxes) - tp
                fn = len(target_gt_records) - len(matched_gt_indices)
                for class_id in eval_class_ids:
                    class_gt_boxes = [
                        record["xyxy"] for record in target_gt_records
                        if int(record["class_id"]) == int(class_id)
                    ]
                    class_pred_boxes = [
                        box for box, pred_class in zip(pred_boxes, pred_classes)
                        if int(pred_class) == int(class_id)
                    ]
                    class_all_pred_boxes = [
                        box for box, pred_class in zip(all_pred_boxes, all_pred_classes)
                        if int(pred_class) == int(class_id)
                    ]
                    class_all_pred_scores = [
                        score for score, pred_class in zip(all_pred_scores, all_pred_classes)
                        if int(pred_class) == int(class_id)
                    ]
                    class_tp, class_fp, class_fn = greedy_match_predictions_to_gt(
                        pred_boxes=class_pred_boxes,
                        gt_boxes=class_gt_boxes,
                        iou_thr=eval_iou_thr,
                    )
                    class_totals = per_class_totals[int(class_id)]
                    class_totals["tp"] += class_tp
                    class_totals["fp"] += class_fp
                    class_totals["fn"] += class_fn
                    class_gt_has_positive = len(class_gt_boxes) > 0
                    class_pred_has_positive = len(class_pred_boxes) > 0
                    if class_gt_has_positive and class_pred_has_positive:
                        class_totals["img_tp"] += 1
                    elif (not class_gt_has_positive) and (not class_pred_has_positive):
                        class_totals["img_tn"] += 1
                    elif (not class_gt_has_positive) and class_pred_has_positive:
                        class_totals["img_fp"] += 1
                    elif class_gt_has_positive and (not class_pred_has_positive):
                        class_totals["img_fn"] += 1
                    ap_records_by_class[int(class_id)].append(
                        {
                            "gt_boxes": class_gt_boxes,
                            "pred_boxes": class_all_pred_boxes,
                            "pred_scores": class_all_pred_scores,
                        }
                    )
            else:
                tp = None
                fp = None
                fn = None

            if gt_available:
                total_tp += tp
                total_fp += fp
                total_fn += fn

            gt_has_positive = len(target_gt_boxes) > 0
            pred_has_positive = len(pred_boxes) > 0

            if gt_available:
                if gt_has_positive and pred_has_positive:
                    img_tp += 1
                elif (not gt_has_positive) and (not pred_has_positive):
                    img_tn += 1
                elif (not gt_has_positive) and pred_has_positive:
                    img_fp += 1
                elif gt_has_positive and (not pred_has_positive):
                    img_fn += 1

            save_path = None
            if gt_available:
                save_path = resolve_eval_visualization_path(
                    visualization_dirs,
                    img_path.name,
                    gt_has_positive=gt_has_positive,
                    pred_has_positive=pred_has_positive,
                )

            if save_path is not None:
                draw_eval_result(
                    image=img,
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    gt_boxes=target_gt_boxes,
                    save_path=save_path,
                    pred_classes=pred_classes,
                    gt_classes=target_gt_classes,
                    class_names=class_names,
                )

            gt_label_exists = bool(
                gt_available
                and label_path is not None
                and label_path.is_file()
            )
            performance_result = build_auto_labeling_performance_record(
                tp=tp,
                fp=fp,
                fn=fn,
                num_predictions=len(pred_boxes),
                num_gt_boxes=len(target_gt_records),
                matches=match_records,
                status="ok" if gt_label_exists else "missing_gt_label",
                gt_label_path=label_path if gt_label_exists else None,
                iou_threshold=eval_iou_thr,
                target_class_id=target_gt_class,
                eval_class_ids=eval_class_ids,
            ) if save_auto_labeling_compatible_results else None

            row = {
                "image_name": img_path.name,
                "label_path": str(label_path) if (gt_available and label_path is not None) else None,
                "gt_target_count": len(target_gt_boxes),
                "pred_target_count": len(pred_boxes),
                "gt_counts_by_class": {
                    str(class_id): sum(1 for gt_class in target_gt_classes if int(gt_class) == int(class_id))
                    for class_id in eval_class_ids
                },
                "pred_counts_by_class": {
                    str(class_id): sum(1 for pred_class in pred_classes if int(pred_class) == int(class_id))
                    for class_id in eval_class_ids
                },
                "gt_magnesium_count": len(target_gt_boxes),
                "pred_magnesium_count": len(pred_boxes),
                "pred_scores": pred_scores,
                "pred_classes": pred_classes,
                "num_predictions_before_merge": int(prediction.get("num_predictions_before_merge", len(all_pred_boxes))),
                "num_predictions_after_merge": int(prediction.get("num_predictions_after_merge", len(all_pred_boxes))),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "gt_has_positive": gt_has_positive if gt_available else None,
                "pred_has_positive": pred_has_positive,
                "gt_available": gt_available,
                "tile_window_count": int(prediction.get("tile_window_count", 0)),
            }
            if save_auto_labeling_compatible_results:
                row.update(
                    {
                        "image_path": str(img_path),
                        "relative_path": img_path.name,
                        "output_visualization_path": (
                            str(save_path) if save_path is not None else None
                        ),
                        "width": int(w),
                        "height": int(h),
                        "num_boxes": len(pred_boxes),
                        "detections": [
                            {
                                "box_xyxy": [float(value) for value in box],
                                "score": float(score),
                                "class_id": int(class_id),
                            }
                            for box, score, class_id in zip(
                                pred_boxes,
                                pred_scores,
                                pred_classes,
                            )
                        ],
                        "performance": performance_result,
                    }
                )
            per_image_results.append(row)

        pbar.update(len(batch_paths))
    pbar.close()

    coco_evaluation = None
    if coco_eval_enabled:
        if label_dir is None:
            raise ValueError("TEST_COCO_EVAL_ENABLED=true requires a valid TEST_LABEL_DIR")

        ground_truth_path = output_dir / "coco_ground_truth.json"
        predictions_path = output_dir / "coco_predictions.json"
        with ground_truth_path.open("w", encoding="utf-8") as handle:
            json.dump(coco_ground_truth, handle, ensure_ascii=False)
        with predictions_path.open("w", encoding="utf-8") as handle:
            json.dump(coco_predictions, handle, ensure_ascii=False)

        coco_category_ids = list(coco_category_id_by_class.values())
        primary_coco_metrics = compute_coco_detection_metrics(
            coco_ground_truth,
            coco_predictions,
            category_ids=coco_category_ids,
            max_dets=coco_max_dets,
        )
        subset_metrics: Dict[str, Dict[str, Any]] = {}
        all_evaluated_names = set(coco_image_id_by_name)
        for subset_name, subset_dir in coco_subset_image_dirs.items():
            requested_names = {path.name for path in list_depth1_images(subset_dir)}
            missing_names = sorted(requested_names - all_evaluated_names)
            if missing_names and coco_subset_strict:
                raise ValueError(
                    f"COCO subset {subset_name!r} contains {len(missing_names)} images "
                    f"not found in TEST_IMAGE_DIR; examples={missing_names[:5]}"
                )
            matched_names = sorted(requested_names & all_evaluated_names)
            if not matched_names:
                raise ValueError(f"COCO subset {subset_name!r} matched no test images")
            selected_image_ids = [
                coco_image_id_by_name[name]
                for name in matched_names
            ]
            metrics_for_subset = compute_coco_detection_metrics(
                coco_ground_truth,
                coco_predictions,
                image_ids=selected_image_ids,
                category_ids=coco_category_ids,
                max_dets=coco_max_dets,
            )
            metrics_for_subset.update(
                {
                    "image_dir": str(subset_dir),
                    "num_requested_images": len(requested_names),
                    "num_matched_images": len(matched_names),
                    "num_missing_images": len(missing_names),
                    "missing_image_examples": missing_names[:20],
                }
            )
            subset_metrics[subset_name] = metrics_for_subset

        coco_metrics_path = output_dir / "coco_metrics.json"
        coco_evaluation = {
            "enabled": True,
            "ground_truth_path": str(ground_truth_path),
            "predictions_path": str(predictions_path),
            "metrics_path": str(coco_metrics_path),
            "primary": primary_coco_metrics,
            "subsets": subset_metrics,
        }
        with coco_metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(coco_evaluation, handle, ensure_ascii=False, indent=2)

    gt_available = label_dir is not None
    if gt_available:
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class_metrics: Dict[str, Dict[str, Any]] = {}
        ap50_values = []
        map50_values = []
        map50_95_values = []
        for class_id in eval_class_ids:
            class_id = int(class_id)
            class_totals = per_class_totals[class_id]
            class_precision = (
                class_totals["tp"] / (class_totals["tp"] + class_totals["fp"])
                if (class_totals["tp"] + class_totals["fp"]) > 0
                else 0.0
            )
            class_recall = (
                class_totals["tp"] / (class_totals["tp"] + class_totals["fn"])
                if (class_totals["tp"] + class_totals["fn"]) > 0
                else 0.0
            )
            class_f1 = (
                2 * class_precision * class_recall / (class_precision + class_recall)
                if (class_precision + class_recall) > 0
                else 0.0
            )
            class_ap = compute_single_class_ap(ap_records_by_class[class_id], iou_thr=eval_iou_thr)
            class_total_images = (
                class_totals["img_tp"]
                + class_totals["img_tn"]
                + class_totals["img_fp"]
                + class_totals["img_fn"]
            )
            class_image_accuracy = (
                (class_totals["img_tp"] + class_totals["img_tn"]) / class_total_images
                if class_total_images > 0
                else 0.0
            )
            ap50_values.append(class_ap["ap50"])
            map50_values.append(class_ap["map50"])
            map50_95_values.append(class_ap["map50_95"])
            per_class_metrics[str(class_id)] = {
                "class_id": class_id,
                "class_name": get_class_name(class_id, class_names),
                "tp": class_totals["tp"],
                "fp": class_totals["fp"],
                "fn": class_totals["fn"],
                "precision": class_precision,
                "recall": class_recall,
                "f1": class_f1,
                "ap50": class_ap["ap50"],
                "mAP50": class_ap["map50"],
                "mAP50_95": class_ap["map50_95"],
                "num_gt": class_ap["num_gt"],
                "image_level": {
                    "tp": class_totals["img_tp"],
                    "tn": class_totals["img_tn"],
                    "fp": class_totals["img_fp"],
                    "fn": class_totals["img_fn"],
                    "accuracy": class_image_accuracy,
                },
            }
        ap_metrics = {
            "ap50": sum(ap50_values) / len(ap50_values) if ap50_values else 0.0,
            "map50": sum(map50_values) / len(map50_values) if map50_values else 0.0,
            "map50_95": sum(map50_95_values) / len(map50_95_values) if map50_95_values else 0.0,
            "per_class": per_class_metrics,
        }
        if coco_evaluation is not None:
            legacy_ap = {
                "ap50": ap_metrics["ap50"],
                "map50": ap_metrics["map50"],
                "map50_95": ap_metrics["map50_95"],
                "per_class": {
                    class_key: {
                        "ap50": values["ap50"],
                        "mAP50": values["mAP50"],
                        "mAP50_95": values["mAP50_95"],
                    }
                    for class_key, values in per_class_metrics.items()
                },
            }
            primary_coco = coco_evaluation["primary"]
            ap_metrics["ap50"] = primary_coco["ap50"]
            ap_metrics["map50"] = primary_coco["ap50"]
            ap_metrics["map50_95"] = primary_coco["ap50_95"]
            ap_metrics["legacy_non_coco"] = legacy_ap
            for class_id in eval_class_ids:
                class_key = str(int(class_id))
                category_id = coco_category_id_by_class[int(class_id)]
                class_coco = primary_coco["per_category"][str(category_id)]
                per_class_metrics[class_key]["ap50"] = class_coco["ap50"]
                per_class_metrics[class_key]["mAP50"] = class_coco["ap50"]
                per_class_metrics[class_key]["mAP50_95"] = class_coco["ap50_95"]
                per_class_metrics[class_key]["ap_metric_style"] = primary_coco["metric_style"]
        total_images = img_tp + img_tn + img_fp + img_fn
        image_accuracy = (img_tp + img_tn) / total_images if total_images > 0 else 0.0
    else:
        precision = None
        recall = None
        f1 = None
        ap_metrics = None
        per_class_metrics = None
        total_images = len(image_paths)
        image_accuracy = None

    metrics = {
        "target_gt_class": target_gt_class,
        "target_class_name": get_class_name(target_gt_class, class_names),
        "eval_class_ids": [int(class_id) for class_id in eval_class_ids],
        "eval_class_names": {
            str(class_id): get_class_name(int(class_id), class_names)
            for class_id in eval_class_ids
        },
        "prediction_class_mapping": (
            {str(key): int(value) for key, value in prediction_class_mapping.items()}
            if prediction_class_mapping is not None
            else None
        ),
        "eval_mode": "multi_class" if is_multiclass_eval else "single_class",
        "model_path": str(model_path),
        "image_dir": str(image_dir),
        "label_dir": str(label_dir) if label_dir is not None else None,
        "output_dir": str(output_dir),
        "test_run_name": cfg.get("TEST_RUN_NAME"),
        "gt_available": gt_available,
        "auto_labeling_compatible_output": {
            "enabled": save_auto_labeling_compatible_results,
            "summary_path": (
                str(output_dir / "summary.json")
                if save_auto_labeling_compatible_results
                else None
            ),
            "per_image_results_path": str(output_dir / "per_image_results.json"),
        },
        "visualizations": {
            "enabled": save_visualizations,
            "tp_dir": str(visualization_dirs["tp"]) if visualization_dirs["tp"] is not None else None,
            "fp_dir": str(visualization_dirs["fp"]) if visualization_dirs["fp"] is not None else None,
            "fn_dir": str(visualization_dirs["fn"]) if visualization_dirs["fn"] is not None else None,
        },
        "prediction_settings": {
            "imgsz": imgsz,
            "conf": conf_thr,
            "ap_eval_conf": ap_eval_conf_thr,
            "conf_source": conf_source,
            "iou": pred_iou,
            "postprocessing": "end2end_without_nms" if end2end_inference else "standard_nms",
            "nms_iou_applied": None if end2end_inference else pred_iou,
            "max_det": max_det,
            "device": device,
            "fp16": test_fp16,
            "limit_visible_device": device_lock_info.get("applied", False),
            "cuda_visible_devices": device_lock_info.get("cuda_visible_devices_after"),
            "tiled_infer_enabled": tiled_infer_enabled,
            "tiled_merge_iou": tiled_merge_iou if tiled_infer_enabled else None,
            "tile_select_box_from_middle": tile_select_box_from_middle if tiled_infer_enabled else None,
            "tiled_tile_size": imgsz if tiled_infer_enabled else None,
            "tiled_tile_stride": (imgsz // 2) if tiled_infer_enabled else None,
            "eval_class_ids": [int(class_id) for class_id in eval_class_ids],
            "gamma_correction": {
                "enabled": bool(gamma_config["enabled"]),
                "gamma": float(gamma_config["gamma"]),
            },
        },
        "evaluation_settings": {
            "eval_iou_threshold": eval_iou_thr,
            "eval_class_ids": [int(class_id) for class_id in eval_class_ids],
            "ap_metric_style": (
                "COCOeval_bbox_101_point"
                if coco_evaluation is not None
                else "legacy_all_points_interpolation"
            ),
            "coco_max_dets": coco_max_dets if coco_evaluation is not None else None,
        },
        "coco_evaluation": coco_evaluation or {"enabled": False},
        "f1_tuned_conf": {
            "enabled": as_bool(cfg.get("TEST_USE_F1_OPTIMIZED_CONF", False), default=False),
            "base_conf": tuned_conf_base,
            "offset": tuned_conf_offset,
            "resolved_conf": f1_resolved_conf,
            "applied_conf": (
                f1_resolved_conf
                if tuned_conf_base is not None
                and not (deployment_use_enabled and deployment_threshold is not None)
                else None
            ),
            "metadata_path": (
                tuned_conf_metadata.get("metadata_path")
                if tuned_conf_metadata is not None
                else None
            ),
            "split_dir": (
                tuned_conf_metadata.get("split_dir")
                if tuned_conf_metadata is not None
                else None
            ),
            "split_name": (
                tuned_conf_metadata.get("split_name")
                if tuned_conf_metadata is not None
                else None
            ),
        },
        "deployment_tuned_conf": {
            "use_enabled": deployment_use_enabled,
            "tune_enabled": deployment_tune_enabled,
            "overwrite_enabled": deployment_overwrite,
            "tuned_now": deployment_tuned_now,
            "reuse_mismatches": deployment_reuse_mismatches,
            "fp_unit": deployment_fp_unit,
            "count_fp_by_image": deployment_count_fp_by_image,
            "threshold": deployment_threshold,
            "applied_conf": (
                conf_thr
                if deployment_use_enabled and deployment_threshold is not None
                else None
            ),
            "metadata_path": (
                tuned_conf_metadata.get("metadata_path")
                if tuned_conf_metadata is not None
                else None
            ),
            "background_image_dir": (
                tuned_conf_metadata.get("deployment_tuning", {}).get("background_image_dir")
                if tuned_conf_metadata is not None
                else None
            ),
            "max_fp": (
                tuned_conf_metadata.get("deployment_tuning", {}).get("max_fp")
                if tuned_conf_metadata is not None
                else None
            ),
        },
        "adabn_grid": {
            "combo_name": cfg.get("_ADABN_GRID_COMBO_NAME"),
            "variant_label": cfg.get("_ADABN_GRID_VARIANT_LABEL"),
            "variant_type": cfg.get("_ADABN_GRID_VARIANT_TYPE"),
            "is_baseline_variant": cfg.get("_ADABN_GRID_IS_BASELINE"),
        },
        "adabn": adabn_info,
        "prediction_summary": {
            "num_images": len(image_paths),
            "num_pred_images": total_pred_images,
            "num_pred_boxes": total_pred_boxes,
            "num_pred_boxes_before_merge": total_pred_boxes_before_merge if tiled_infer_enabled else total_pred_boxes,
            "num_tile_windows": total_tile_windows if tiled_infer_enabled else 0,
        },
        "object_level": {
            "tp": total_tp if gt_available else None,
            "fp": total_fp if gt_available else None,
            "fn": total_fn if gt_available else None,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap50": ap_metrics["ap50"] if ap_metrics is not None else None,
            "mAP50": ap_metrics["map50"] if ap_metrics is not None else None,
            "mAP50_95": ap_metrics["map50_95"] if ap_metrics is not None else None,
            "per_class": per_class_metrics if gt_available else None,
            "legacy_non_coco_ap": (
                ap_metrics.get("legacy_non_coco")
                if ap_metrics is not None
                else None
            ),
        },
        "image_level": {
            "tp": img_tp if gt_available else None,
            "tn": img_tn if gt_available else None,
            "fp": img_fp if gt_available else None,
            "fn": img_fn if gt_available else None,
            "accuracy": image_accuracy,
        },
        "num_images": total_images,
    }

    if adabn_info.get("debug") is not None or adabn_info.get("debug_probe") is not None:
        adabn_debug_payload = {
            "model_path": str(model_path),
            "test_image_dir": str(image_dir),
            "adabn": {
                "config": {
                    "enabled": adabn_info.get("enabled"),
                    "image_dir": adabn_info.get("image_dir"),
                    "batch_size": adabn_info.get("batch_size"),
                    "imgsz": adabn_info.get("imgsz"),
                    "max_images": adabn_info.get("max_images"),
                    "reset_stats": adabn_info.get("reset_stats"),
                },
                "debug": adabn_info.get("debug"),
                "debug_probe": adabn_info.get("debug_probe"),
            },
        }
        debug_path = output_dir / "adabn_debug.json"
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(adabn_debug_payload, f, ensure_ascii=False, indent=2)
        metrics["adabn"]["debug_path"] = str(debug_path)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if save_jsonl:
        with open(output_dir / "per_image_results.jsonl", "w", encoding="utf-8") as f:
            for row in per_image_results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_per_image_result_json(
        output_dir,
        per_image_results,
        auto_labeling_compatible=save_auto_labeling_compatible_results,
        metrics=metrics,
    )

    print("\n========== Detection Evaluation ==========")
    print(f"Model path              : {model_path}")
    print(f"Test image dir          : {image_dir}")
    print(f"Test label dir          : {label_dir if label_dir is not None else 'N/A (GT skipped)'}")
    print(f"Output dir              : {output_dir}")
    print(f"Save visualizations     : {save_visualizations}")
    print(f"Auto-label output format: {save_auto_labeling_compatible_results}")
    print(
        "Eval classes            : "
        + ", ".join(
            f"{class_id}:{get_class_name(int(class_id), class_names)}"
            for class_id in eval_class_ids
        )
    )
    if prediction_class_mapping is not None:
        print(f"Prediction class mapping: {prediction_class_mapping}")
    print("--------------------------------------------------------")
    print("[Prediction settings]")
    print(f"imgsz                   : {imgsz}")
    print(f"conf                    : {conf_thr}")
    print(f"conf source             : {conf_source}")
    if deployment_threshold is not None:
        print(
            "deployment threshold    : "
            f"{deployment_threshold} (use={deployment_use_enabled}, tuned_now={deployment_tuned_now})"
        )
    print(f"postprocessing          : {'end2end without NMS' if end2end_inference else 'standard NMS'}")
    print(f"prediction IoU argument : {pred_iou} (inactive for end2end)")
    print(f"max_det                 : {max_det}")
    print(f"device                  : {device}")
    print(f"fp16                    : {test_fp16}")
    print(
        "gamma correction        : "
        f"{gamma_config['enabled']} (gamma={float(gamma_config['gamma'])})"
    )
    print(f"tiled infer enabled     : {tiled_infer_enabled}")
    if tiled_infer_enabled:
        print(f"tile size / stride      : {imgsz} / {imgsz // 2}")
        print(f"tile merge IoU          : {tiled_merge_iou}")
        print(f"tile middle select ratio: {tile_select_box_from_middle}")
    print(f"AdaBN enabled           : {adabn_info.get('enabled', False)}")
    print(f"AdaBN applied           : {adabn_info.get('applied', False)}")
    if adabn_info.get("enabled", False):
        print(f"AdaBN image dir         : {adabn_info.get('image_dir')}")
        print(f"AdaBN used images       : {adabn_info.get('num_images_used')}")
        print(
            f"AdaBN batch/imgsz/epoch : "
            f"{adabn_info.get('batch_size')} / {adabn_info.get('imgsz')} / {adabn_info.get('epochs')}"
        )
        print(f"AdaBN max stage         : {adabn_info.get('max_stage')}")
        print(f"AdaBN momentum          : {adabn_info.get('effective_bn_momentum')}")
        print(f"AdaBN fp16              : {adabn_info.get('fp16_applied')}")
        print(
            f"AdaBN BN layers         : "
            f"{adabn_info.get('bn_layer_count')} / {adabn_info.get('bn_layer_count_total')}"
        )
        if adabn_info.get("debug") is not None:
            debug_summary = adabn_info["debug"].get("bn_stats", {}).get("global_summary", {})
            print(
                f"AdaBN debug delta       : mean_var_delta={debug_summary.get('mean_of_mean_abs_delta_running_var')}, "
                f"nonfinite_layers={debug_summary.get('nonfinite_after_layers')}, "
                f"near_zero_var_layers={debug_summary.get('near_zero_running_var_layers')}"
            )
        if adabn_info.get("debug_probe") is not None:
            print(f"AdaBN debug probe       : {len(adabn_info['debug_probe'].get('probe_image_paths', []))} images")
    print("--------------------------------------------------------")
    if gt_available:
        print("[Evaluation settings]")
        print(f"IoU match threshold     : {eval_iou_thr}")
        print("--------------------------------------------------------")
        print("[Detection-only object-level]")
        print(f"TP={total_tp}, FP={total_fp}, FN={total_fn}")
        print(f"Precision={precision:.4f}")
        print(f"Recall   ={recall:.4f}")
        print(f"F1-score ={f1:.4f}")
        print(f"AP@0.5   ={ap_metrics['ap50']:.4f}")
        print(f"mAP@0.5  ={ap_metrics['map50']:.4f}")
        print(f"mAP@0.5:0.95={ap_metrics['map50_95']:.4f}")
        if is_multiclass_eval:
            print("[Detection-only per-class]")
            for class_id in eval_class_ids:
                class_metric = per_class_metrics[str(int(class_id))]
                print(
                    f"  {class_id}:{class_metric['class_name']} "
                    f"TP={class_metric['tp']} FP={class_metric['fp']} FN={class_metric['fn']} "
                    f"P={class_metric['precision']:.4f} R={class_metric['recall']:.4f} "
                    f"F1={class_metric['f1']:.4f} mAP50={class_metric['mAP50']:.4f}"
                )
        print("--------------------------------------------------------")
        print("[Detection-only image-level]")
        print(f"TP={img_tp}, TN={img_tn}, FP={img_fp}, FN={img_fn}")
        print(f"Accuracy ={image_accuracy:.4f}")
    else:
        print("[GT-less retest]")
        print("GT label dir            : not provided")
        print(f"Pred images / boxes     : {total_pred_images} / {total_pred_boxes}")
        print("GT-dependent metrics    : skipped")
    print("========================================================\n")
    del model
    _cleanup_cuda_runtime("run_test_with_config")
    return metrics


def _write_test_grid_summary_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_test_grid_summary_csv(path, rows):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "test_variant_name",
        "adabn_epochs",
        "adabn_max_images",
        "adabn_max_stage",
        "adabn_momentum",
        "output_dir",
        "precision",
        "recall",
        "f1",
        "ap50",
        "mAP50",
        "mAP50_95",
        "image_accuracy",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_test_grid_summary_txt(path, rows, best_row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("========== AdaBN Grid Test Summary ==========\n")
        f.write(f"Num combinations: {len(rows)}\n")
        if best_row is not None:
            f.write(
                f"Best combo by mAP50: {best_row['test_variant_name']} "
                f"(mAP50={_safe_metric(best_row.get('mAP50'))}, f1={_safe_metric(best_row.get('f1'))})\n"
            )
        f.write("============================================\n\n")
        for row in rows:
            f.write(f"[{row['test_variant_name']}]\n")
            f.write(
                f"  params      : epochs={row['adabn_epochs']}, max_images={row['adabn_max_images']}, "
                f"stage={row['adabn_max_stage']}, momentum={row['adabn_momentum']}\n"
            )
            f.write(f"  output_dir   : {row['output_dir']}\n")
            f.write(
                f"  metrics      : precision={_safe_metric(row.get('precision'))}, "
                f"recall={_safe_metric(row.get('recall'))}, "
                f"f1={_safe_metric(row.get('f1'))}, "
                f"mAP50={_safe_metric(row.get('mAP50'))}, "
                f"mAP50_95={_safe_metric(row.get('mAP50_95'))}, "
                f"acc={_safe_metric(row.get('image_accuracy'))}\n"
            )
            f.write("\n")


def _safe_metric(value):
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def _cleanup_cuda_runtime(note: str) -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
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


def run_single_test_with_config(cfg):
    variants = build_adabn_grid_variants(cfg)
    has_grid = any(variant.get("has_grid", False) for variant in variants)
    if not has_grid:
        return run_test_with_config(cfg)

    base_output_dir = resolve_named_output_dir(cfg, "TEST_OUTPUT_DIR", "TEST_RUN_NAME")
    _validate_test_inputs(cfg, base_output_dir)
    if as_bool(cfg.get("TEST_RESET_OUTPUT_DIR", False), default=False) and base_output_dir.exists():
        print(f"[TEST_GRID] Deleting the existing grid results directory: {base_output_dir}")
        shutil.rmtree(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for index, variant in enumerate(variants, start=1):
        variant_cfg = deepcopy(cfg)
        apply_adabn_grid_variant(variant_cfg, variant)
        variant_cfg["TEST_OUTPUT_DIR"] = str(base_output_dir / variant["combo_name"])
        variant_cfg["_TEST_OUTPUT_DIR_RESOLVED"] = True
        variant_cfg["TEST_RESET_OUTPUT_DIR"] = True
        print(
            f"[TEST_GRID] variant {index}/{len(variants)} "
            f"name={variant['combo_name']} overrides={variant['overrides']}"
        )
        metrics = run_test_with_config(variant_cfg)
        object_level = metrics.get("object_level", {})
        image_level = metrics.get("image_level", {})
        results.append(
            {
                "test_variant_name": variant["combo_name"],
                "adabn_epochs": variant["fields"].get("adabn_epochs"),
                "adabn_max_images": variant["fields"].get("adabn_max_images"),
                "adabn_max_stage": variant["fields"].get("adabn_max_stage"),
                "adabn_momentum": variant["fields"].get("adabn_momentum"),
                "output_dir": metrics.get("output_dir"),
                "precision": object_level.get("precision"),
                "recall": object_level.get("recall"),
                "f1": object_level.get("f1"),
                "ap50": object_level.get("ap50"),
                "mAP50": object_level.get("mAP50"),
                "mAP50_95": object_level.get("mAP50_95"),
                "image_accuracy": image_level.get("accuracy"),
            }
        )

    best_row = max(
        results,
        key=lambda row: (
            float(row.get("mAP50") or 0.0),
            float(row.get("f1") or 0.0),
        ),
    ) if results else None
    payload = {
        "num_combinations": len(results),
        "best_combo_by_map50": best_row,
        "rows": results,
    }
    _write_test_grid_summary_json(base_output_dir / "adabn_grid_summary.json", payload)
    _write_test_grid_summary_csv(base_output_dir / "adabn_grid_summary.csv", results)
    _write_test_grid_summary_txt(base_output_dir / "adabn_grid_summary.txt", results, best_row)
    print(f"[TEST_GRID] summary json={base_output_dir / 'adabn_grid_summary.json'}")
    print(f"[TEST_GRID] summary csv={base_output_dir / 'adabn_grid_summary.csv'}")
    print(f"[TEST_GRID] summary txt={base_output_dir / 'adabn_grid_summary.txt'}")
    return payload


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grapevine detection test runner")
    parser.add_argument(
        "--config",
        default="configs/detection/config_for_test.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    return run_single_test_with_config(cfg)


if __name__ == "__main__":
    main()
