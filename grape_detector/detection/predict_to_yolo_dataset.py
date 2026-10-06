import argparse
from pathlib import Path
import json
import shutil
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from grape_detector.utils.adabn import apply_adabn_to_yolo_model
from grape_detector.utils.config_loader import as_bool, load_config, resolve_tqdm_disable
from grape_detector.utils.run_path_utils import validate_output_separation
from grape_detector.utils.f1_conf_optimizer import load_tuned_conf_metadata, optimize_conf_for_validation
from grape_detector.utils.tiled_detection_inference import extract_prediction_arrays, infer_image_with_tiling


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CLASS_NAMES = [
    "etc",
    "gray mold",
    "magnesium deficiency",
    "colomerus vitis",
]


def list_images(image_dir: Path) -> List[Path]:
    image_paths = []
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            image_paths.append(path)
    image_paths.sort()
    return image_paths


def resolve_cfg_value(cfg, primary_key: str, fallback_key: Optional[str] = None, default=None):
    if primary_key in cfg and cfg[primary_key] is not None:
        return cfg[primary_key]
    if fallback_key and fallback_key in cfg and cfg[fallback_key] is not None:
        return cfg[fallback_key]
    return default


def get_optional_int(value, default=None):
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return default
    return int(value)


def get_adabn_momentum_value(value):
    sentinel = "use_original"
    if value == sentinel:
        return sentinel
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"none", "null"}:
            return None
        if text in {"", "default", "keep", "original", "use_original"}:
            return sentinel
        return float(value)
    return float(value)


def boxes_xyxy_to_xywhn(boxes_xyxy, img_w: int, img_h: int):
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = boxes.reshape(-1, 4).copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, float(img_w))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, float(img_h))

    xywhn = np.zeros_like(boxes, dtype=np.float32)
    xywhn[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2.0) / float(img_w)
    xywhn[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2.0) / float(img_h)
    xywhn[:, 2] = (boxes[:, 2] - boxes[:, 0]) / float(img_w)
    xywhn[:, 3] = (boxes[:, 3] - boxes[:, 1]) / float(img_h)
    return np.clip(xywhn, 0.0, 1.0)


def draw_prediction_visualization(
    image,
    boxes_xyxy,
    scores,
    classes,
    save_path: Path,
    kept_indices=None,
):
    vis = image.copy()
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes = np.asarray(classes, dtype=np.int32).reshape(-1)
    kept_set = set(range(len(boxes))) if kept_indices is None else {int(index) for index in kept_indices}

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        is_kept = index in kept_set
        color = (0, 180, 0) if is_kept else (0, 0, 255)
        thickness = 2 if is_kept else 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        label_parts = [f"cls{int(classes[index])}", f"det {float(scores[index]):.2f}"]
        label = " ".join(label_parts)

        text_y = max(15, y1 - 5)
        cv2.putText(
            vis,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)
    return save_path


def calculate_intersection_ratio_pair(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    ratio_a = inter_area / area_a if area_a > 0 else 0.0
    ratio_b = inter_area / area_b if area_b > 0 else 0.0
    return ratio_a, ratio_b


def suppress_boxes_by_max_overlap_ratio(boxes_xyxy, scores, max_overlap_ratio: Optional[float]):
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if max_overlap_ratio is None or len(boxes) <= 1:
        return list(range(len(boxes))), []

    order = np.argsort(scores)[::-1]
    keep_indices = []
    removed_records = []
    threshold = float(max_overlap_ratio)

    for index in order:
        index = int(index)
        remove = False
        remove_reason = None
        for kept_index in keep_indices:
            ratio_current, ratio_kept = calculate_intersection_ratio_pair(boxes[index], boxes[kept_index])
            if ratio_current > threshold or ratio_kept > threshold:
                remove = True
                remove_reason = {
                    "removed_index": index,
                    "kept_index": int(kept_index),
                    "removed_score": float(scores[index]),
                    "kept_score": float(scores[kept_index]),
                    "overlap_ratio_removed": float(ratio_current),
                    "overlap_ratio_kept": float(ratio_kept),
                    "threshold": threshold,
                }
                break
        if remove:
            removed_records.append(remove_reason)
        else:
            keep_indices.append(index)

    keep_indices.sort()
    return keep_indices, removed_records


def resolve_prediction_conf(cfg, model_path: Path, imgsz: int, pred_iou: float, max_det: int, device: str, batch_size: int):
    default_conf_thr = float(resolve_cfg_value(cfg, "PREDICT_CONF", "TEST_CONF", 0.25))
    conf_thr = default_conf_thr
    conf_source = "config"
    tuned_conf_metadata = None

    use_f1_optimized_conf = as_bool(
        resolve_cfg_value(cfg, "PREDICT_USE_F1_OPTIMIZED_CONF", "TEST_USE_F1_OPTIMIZED_CONF", False),
        default=False,
    )
    overwrite_tuned_conf = as_bool(
        resolve_cfg_value(cfg, "PREDICT_OVERWRITE_F1_OPTIMIZED_CONF", "TEST_OVERWRITE_F1_OPTIMIZED_CONF", False),
        default=False,
    )
    tune_if_missing = as_bool(
        resolve_cfg_value(cfg, "PREDICT_AUTO_TUNE_F1_CONF_IF_MISSING", "TEST_AUTO_TUNE_F1_CONF_IF_MISSING", False),
        default=False,
    )

    if not use_f1_optimized_conf:
        return conf_thr, conf_source, tuned_conf_metadata

    tuned_conf_metadata = load_tuned_conf_metadata(model_path)
    split_dir_value = resolve_cfg_value(cfg, "PREDICT_TUNE_SPLIT_DIR", "TEST_TUNE_SPLIT_DIR")
    tune_split_name = str(resolve_cfg_value(cfg, "PREDICT_TUNE_SPLIT", "TEST_TUNE_SPLIT", "val")).strip()

    should_tune_now = False
    tuned_now = False
    if overwrite_tuned_conf:
        should_tune_now = True
    elif tuned_conf_metadata is None and tune_if_missing:
        should_tune_now = True

    if should_tune_now:
        if split_dir_value:
            tuned_conf_metadata = optimize_conf_for_validation(
                model_path=model_path,
                split_dir=Path(split_dir_value),
                split_name=tune_split_name,
                imgsz=int(resolve_cfg_value(cfg, "PREDICT_TUNE_IMGSZ", "TEST_TUNE_IMGSZ", imgsz)),
                pred_iou=float(resolve_cfg_value(cfg, "PREDICT_TUNE_PRED_IOU", "TEST_TUNE_PRED_IOU", pred_iou)),
                max_det=int(resolve_cfg_value(cfg, "PREDICT_TUNE_MAX_DET", "TEST_TUNE_MAX_DET", max_det)),
                device=str(resolve_cfg_value(cfg, "PREDICT_TUNE_DEVICE", "TEST_TUNE_DEVICE", device)),
                batch_size=int(resolve_cfg_value(cfg, "PREDICT_TUNE_BATCH", "TEST_TUNE_BATCH", batch_size)),
                eval_iou_thr=float(
                    resolve_cfg_value(
                        cfg,
                        "PREDICT_TUNE_EVAL_IOU_THRESHOLD",
                        "TEST_TUNE_EVAL_IOU_THRESHOLD",
                        resolve_cfg_value(cfg, "EVAL_IOU_THRESHOLD", None, 0.5),
                    )
                ),
                target_gt_class=int(resolve_cfg_value(cfg, "PREDICT_TUNE_TARGET_GT_CLASS", "TEST_TUNE_TARGET_GT_CLASS", 0)),
                conf_min=float(resolve_cfg_value(cfg, "PREDICT_TUNE_CONF_MIN", "TEST_TUNE_CONF_MIN", 0.01)),
                conf_max=float(resolve_cfg_value(cfg, "PREDICT_TUNE_CONF_MAX", "TEST_TUNE_CONF_MAX", 0.99)),
                conf_step=float(resolve_cfg_value(cfg, "PREDICT_TUNE_CONF_STEP", "TEST_TUNE_CONF_STEP", 0.01)),
                save_metadata=True,
            )
            conf_thr = float(tuned_conf_metadata["best_conf"])
            tuned_now = True
            conf_source = "overwritten_f1_tuned_conf" if overwrite_tuned_conf else "auto_tuned_from_split"
            print(
                f"[PREDICT_CONF] {tune_split_name} split: using F1-optimized confidence: "
                f"{conf_thr:.4f} ({tuned_conf_metadata.get('metadata_path', model_path)})"
            )
        else:
            print("[WARN] TUNE_SPLIT_DIR is not set; skipping confidence retuning.")

    if tuned_conf_metadata is not None:
        conf_thr = float(tuned_conf_metadata["best_conf"])
        if not tuned_now:
            conf_source = "saved_f1_tuned_conf"
            print(
                f"[PREDICT_CONF] Using saved F1-optimized confidence: "
                f"{conf_thr:.4f} ({tuned_conf_metadata.get('metadata_path', model_path)})"
            )
    else:
        print(f"[PREDICT_CONF] Using configured confidence: {conf_thr:.4f}")

    return conf_thr, conf_source, tuned_conf_metadata


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def save_classes_txt(output_dir: Path, class_names: List[str]):
    classes_txt_path = output_dir / "labels/classes.txt"
    classes_txt_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    return classes_txt_path


def save_yolo_label(
    label_path: Path,
    boxes_xywhn,
    classes,
    confs,
    save_conf: bool,
):
    ensure_parent(label_path)

    if boxes_xywhn is None or len(boxes_xywhn) == 0:
        label_path.write_text("", encoding="utf-8")
        return 0

    lines = []
    for box, cls_id, conf in zip(boxes_xywhn, classes, confs):
        xc, yc, w, h = [float(x) for x in box.tolist()]
        if save_conf:
            lines.append(f"{int(cls_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {float(conf):.6f}")
        else:
            lines.append(f"{int(cls_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def run_prediction_to_yolo_dataset(cfg):
    model_path = Path(resolve_cfg_value(cfg, "PREDICT_MODEL_PATH", "TEST_MODEL_PATH")).expanduser().resolve()
    image_dir = Path(resolve_cfg_value(cfg, "PREDICT_IMAGE_DIR", "TEST_IMAGE_DIR")).expanduser().resolve()
    output_dir = Path(resolve_cfg_value(cfg, "PREDICT_OUTPUT_DIR", "TEST_OUTPUT_DIR")).expanduser()

    validate_output_separation(output_dir, {
        "model": model_path,
        "images": image_dir,
        "AdaBN images": resolve_cfg_value(cfg, "PREDICT_ADABN_IMAGE_DIR", "TEST_ADABN_IMAGE_DIR"),
        "validation split": resolve_cfg_value(cfg, "PREDICT_TUNE_SPLIT_DIR", "TEST_TUNE_SPLIT_DIR"),
    })

    imgsz = int(resolve_cfg_value(cfg, "PREDICT_IMGSZ", "TEST_IMGSZ", 960))
    pred_iou = float(resolve_cfg_value(cfg, "PREDICT_IOU", "TEST_IOU", 0.7))
    max_det = int(resolve_cfg_value(cfg, "PREDICT_MAX_DET", "TEST_MAX_DET", 300))
    device = str(resolve_cfg_value(cfg, "PREDICT_DEVICE", "TEST_DEVICE", "0"))
    batch_size = int(resolve_cfg_value(cfg, "PREDICT_BATCH", "TEST_BATCH", 1))
    test_fp16 = as_bool(resolve_cfg_value(cfg, "PREDICT_FP16", "TEST_FP16", False), default=False)
    tiled_infer_enabled = as_bool(
        resolve_cfg_value(cfg, "PREDICT_TILED_INFER_ENABLED", "TEST_TILED_INFER_ENABLED", False),
        default=False,
    )
    tiled_merge_iou = float(resolve_cfg_value(cfg, "PREDICT_TILED_MERGE_IOU", "TEST_TILED_MERGE_IOU", pred_iou))
    tile_select_box_from_middle = resolve_cfg_value(
        cfg,
        "PREDICT_TILED_SELECT_BOX_FROM_MIDDLE",
        "TEST_TILED_SELECT_BOX_FROM_MIDDLE",
        None,
    )
    max_overlap_ratio = resolve_cfg_value(
        cfg,
        "PREDICT_MAX_OVERLAP_RATIO",
        "TEST_MAX_OVERLAP_RATIO",
        None,
    )
    max_overlap_ratio = (
        None
        if max_overlap_ratio is None
        or (isinstance(max_overlap_ratio, str) and max_overlap_ratio.strip().lower() in {"", "none", "null"})
        else float(max_overlap_ratio)
    )
    save_conf = as_bool(resolve_cfg_value(cfg, "PREDICT_SAVE_CONF", None, False), default=False)
    reset_output_dir = as_bool(
        resolve_cfg_value(cfg, "PREDICT_RESET_OUTPUT_DIR", "TEST_RESET_OUTPUT_DIR", False),
        default=False,
    )
    disable_tqdm = resolve_tqdm_disable(
        cfg,
        "PREDICT_DISABLE_TQDM",
        "PREDICT_TQDM_DISABLE",
        "TEST_DISABLE_TQDM",
        "TEST_TQDM_DISABLE",
        "DISABLE_TQDM",
        default=False,
    )

    classes = resolve_cfg_value(cfg, "PREDICT_CLASSES", "INFER_CLASSES")
    if classes is not None:
        classes = [int(cls_id) for cls_id in classes]

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Input image directory not found: {image_dir}")

    image_paths = list_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"No input images found: {image_dir}")

    conf_thr, conf_source, tuned_conf_metadata = resolve_prediction_conf(
        cfg=cfg,
        model_path=model_path,
        imgsz=imgsz,
        pred_iou=pred_iou,
        max_det=max_det,
        device=device,
        batch_size=batch_size,
    )

    if reset_output_dir and output_dir.exists():
        print(f"[Output reset] Deleting the existing results directory: {output_dir}")
        shutil.rmtree(output_dir)

    images_out_dir = output_dir / "images"
    labels_out_dir = output_dir / "labels"
    vis_out_dir = output_dir / "visualizations"
    images_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    vis_out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    adabn_info = {
        "enabled": as_bool(resolve_cfg_value(cfg, "PREDICT_ADABN_ENABLED", "TEST_ADABN_ENABLED", False), default=False),
        "applied": False,
        "status": "disabled",
    }
    if adabn_info["enabled"]:
        adabn_image_dir = resolve_cfg_value(cfg, "PREDICT_ADABN_IMAGE_DIR", "TEST_ADABN_IMAGE_DIR")
        if not adabn_image_dir:
            raise ValueError("An AdaBN image directory is required when PREDICT_ADABN_ENABLED or TEST_ADABN_ENABLED is true.")
        adabn_info = apply_adabn_to_yolo_model(
            model=model,
            image_dir=Path(str(adabn_image_dir)),
            imgsz=get_optional_int(resolve_cfg_value(cfg, "PREDICT_ADABN_IMGSZ", "TEST_ADABN_IMGSZ"), imgsz),
            batch_size=get_optional_int(resolve_cfg_value(cfg, "PREDICT_ADABN_BATCH", "TEST_ADABN_BATCH"), batch_size),
            device=device,
            max_images=get_optional_int(resolve_cfg_value(cfg, "PREDICT_ADABN_MAX_IMAGES", "TEST_ADABN_MAX_IMAGES")),
            epochs=get_optional_int(resolve_cfg_value(cfg, "PREDICT_ADABN_EPOCHS", "TEST_ADABN_EPOCHS"), 1),
            max_stage=get_optional_int(resolve_cfg_value(cfg, "PREDICT_ADABN_MAX_STAGE", "TEST_ADABN_MAX_STAGE")),
            reset_stats=as_bool(
                resolve_cfg_value(cfg, "PREDICT_ADABN_RESET_STATS", "TEST_ADABN_RESET_STATS", True),
                default=True,
            ),
            momentum_override=get_adabn_momentum_value(
                resolve_cfg_value(cfg, "PREDICT_ADABN_MOMENTUM", "TEST_ADABN_MOMENTUM", "use_original")
            ),
            use_fp16=as_bool(resolve_cfg_value(cfg, "PREDICT_ADABN_FP16", "TEST_ADABN_FP16", False), default=False),
            debug=as_bool(resolve_cfg_value(cfg, "PREDICT_ADABN_DEBUG", "TEST_ADABN_DEBUG", False), default=False),
            debug_max_batches=get_optional_int(
                resolve_cfg_value(cfg, "PREDICT_ADABN_DEBUG_MAX_BATCHES", "TEST_ADABN_DEBUG_MAX_BATCHES"),
                1,
            ),
            disable_tqdm=disable_tqdm,
        )

    print("\n========== Predict Unlabeled Folder To YOLO Dataset ==========")
    print(f"Model path        : {model_path}")
    print(f"Input image dir   : {image_dir}")
    print(f"Output dir        : {output_dir}")
    print(f"Num images        : {len(image_paths)}")
    print(f"imgsz             : {imgsz}")
    print(f"conf              : {conf_thr:.4f} ({conf_source})")
    print(f"iou               : {pred_iou}")
    print(f"max_det           : {max_det}")
    print(f"device            : {device}")
    print(f"fp16              : {test_fp16}")
    print(f"batch             : {batch_size}")
    print(f"save_conf_in_txt  : {save_conf}")
    print(f"classes filter    : {classes if classes is not None else 'all'}")
    print(f"max overlap ratio : {max_overlap_ratio}")
    print(f"tiled infer       : {tiled_infer_enabled}")
    if tiled_infer_enabled:
        print(f"tile size/stride  : {imgsz} / {imgsz // 2}")
        print(f"tile merge iou    : {tiled_merge_iou}")
    print(f"AdaBN enabled     : {adabn_info.get('enabled', False)}")
    print(f"AdaBN applied     : {adabn_info.get('applied', False)}")
    if adabn_info.get("enabled", False):
        print(f"AdaBN image dir   : {adabn_info.get('image_dir')}")
        print(
            f"AdaBN batch/imgsz : "
            f"{adabn_info.get('batch_size')} / {adabn_info.get('imgsz')}"
        )
        print(
            f"AdaBN epochs/stage: "
            f"{adabn_info.get('epochs')} / {adabn_info.get('max_stage')}"
        )
        print(
            f"AdaBN momentum    : {adabn_info.get('effective_bn_momentum')} "
            f"(fp16={adabn_info.get('fp16_applied')})"
        )
    print("==============================================================\n")

    total_boxes = 0
    detected_images = 0
    total_boxes_before_overlap = 0
    detected_images_before_overlap = 0
    total_boxes_filtered_by_overlap = 0
    total_tile_windows = 0
    per_image_results = []

    pbar = tqdm(total=len(image_paths), desc="Predicting and saving labels", disable=disable_tqdm)
    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start:batch_start + batch_size]
        batch_items = []
        for img_path in batch_paths:
            image = cv2.imread(str(img_path))
            rel_path = img_path.relative_to(image_dir)
            out_img_path = images_out_dir / rel_path
            out_label_path = (labels_out_dir / rel_path).with_suffix(".txt")
            out_vis_path = (vis_out_dir / rel_path).with_suffix(".jpg")

            ensure_parent(out_img_path)
            shutil.copy2(img_path, out_img_path)
            if image is None:
                print(f"[WARN] Failed to load image; saving an empty label: {img_path}")
                save_yolo_label(
                    label_path=out_label_path,
                    boxes_xywhn=None,
                    classes=None,
                    confs=None,
                    save_conf=save_conf,
                )
                per_image_results.append(
                    {
                        "image_path": str(img_path),
                        "relative_path": str(rel_path),
                        "output_image_path": str(out_img_path),
                        "output_label_path": str(out_label_path),
                        "output_visualization_path": None,
                        "num_boxes_before_overlap": 0,
                        "num_boxes": 0,
                        "overlap_filtered_boxes": None,
                        "tile_window_count": 0,
                        "warning": "image_load_failed",
                    }
                )
                continue

            batch_items.append((img_path, image))

        if not batch_items:
            pbar.update(len(batch_paths))
            continue

        batch_predictions = []
        if tiled_infer_enabled:
            for _, image in batch_items:
                batch_predictions.append(
                    infer_image_with_tiling(
                        model,
                        image,
                        imgsz=imgsz,
                        conf=conf_thr,
                        iou=pred_iou,
                        max_det=max_det,
                        device=device,
                        half=test_fp16,
                        classes=classes,
                        tile_enabled=True,
                        tile_merge_iou=tiled_merge_iou,
                        tile_select_box_from_middle=tile_select_box_from_middle,
                    )
                )
        else:
            predict_kwargs = {
                "source": [image for _, image in batch_items],
                "imgsz": imgsz,
                "conf": conf_thr,
                "iou": pred_iou,
                "max_det": max_det,
                "device": device,
                "half": test_fp16,
                "verbose": False,
            }
            if classes is not None:
                predict_kwargs["classes"] = classes
            results = model.predict(**predict_kwargs)
            for result in results:
                boxes_xyxy, scores, pred_classes = extract_prediction_arrays(result)
                batch_predictions.append(
                    {
                        "boxes_xyxy": boxes_xyxy,
                        "scores": scores,
                        "classes": pred_classes,
                        "num_predictions_before_merge": int(len(boxes_xyxy)),
                        "num_predictions_after_merge": int(len(boxes_xyxy)),
                        "tile_window_count": 0,
                    }
                )

        for (img_path, image), prediction in zip(batch_items, batch_predictions):
            rel_path = img_path.relative_to(image_dir)
            out_img_path = images_out_dir / rel_path
            out_label_path = (labels_out_dir / rel_path).with_suffix(".txt")
            out_vis_path = (vis_out_dir / rel_path).with_suffix(".jpg")

            h, w = image.shape[:2]
            boxes_xyxy = np.asarray(prediction["boxes_xyxy"], dtype=np.float32).reshape(-1, 4)
            confs = np.asarray(prediction["scores"], dtype=np.float32).reshape(-1)
            clss = np.asarray(prediction.get("classes", np.zeros((len(boxes_xyxy),), dtype=np.int32)), dtype=np.int32).reshape(-1)
            boxes_before_overlap = boxes_xyxy.copy()
            confs_before_overlap = confs.copy()
            clss_before_overlap = clss.copy()
            num_boxes_before_overlap = int(len(boxes_xyxy))
            total_boxes_before_overlap += num_boxes_before_overlap
            total_tile_windows += int(prediction.get("tile_window_count", 0))
            if num_boxes_before_overlap > 0:
                detected_images_before_overlap += 1

            overlap_filtered_boxes = None
            overlap_removed_records = []
            kept_indices = list(range(num_boxes_before_overlap))

            if max_overlap_ratio is not None:
                overlap_keep_positions, overlap_removed_records = suppress_boxes_by_max_overlap_ratio(
                    boxes_xyxy,
                    confs,
                    max_overlap_ratio,
                )
                overlap_filtered_boxes = len(boxes_xyxy) - len(overlap_keep_positions)
                total_boxes_filtered_by_overlap += overlap_filtered_boxes
                kept_indices = [kept_indices[position] for position in overlap_keep_positions]
                overlap_keep = np.asarray(overlap_keep_positions, dtype=np.int64)
                boxes_xyxy = (
                    boxes_xyxy[overlap_keep]
                    if overlap_keep.size > 0
                    else np.zeros((0, 4), dtype=np.float32)
                )
                confs = (
                    confs[overlap_keep]
                    if overlap_keep.size > 0
                    else np.zeros((0,), dtype=np.float32)
                )
                clss = (
                    clss[overlap_keep]
                    if overlap_keep.size > 0
                    else np.zeros((0,), dtype=np.int32)
                )
                if num_boxes_before_overlap > 0:
                    print(
                        "[OVERLAP_FILTER] "
                        f"{rel_path}: filtered={overlap_filtered_boxes} "
                        f"max_overlap_ratio={max_overlap_ratio:.3f}"
                    )

            num_boxes = 0
            if len(boxes_xyxy) > 0:
                xywhn = boxes_xyxy_to_xywhn(boxes_xyxy, img_w=w, img_h=h)
                num_boxes = save_yolo_label(
                    label_path=out_label_path,
                    boxes_xywhn=xywhn,
                    classes=clss,
                    confs=confs,
                    save_conf=save_conf,
                )
            else:
                num_boxes = save_yolo_label(
                    label_path=out_label_path,
                    boxes_xywhn=None,
                    classes=None,
                    confs=None,
                    save_conf=save_conf,
                )

            total_boxes += num_boxes
            if num_boxes > 0:
                detected_images += 1

            draw_prediction_visualization(
                image=image,
                boxes_xyxy=boxes_before_overlap,
                scores=confs_before_overlap,
                classes=clss_before_overlap,
                save_path=out_vis_path,
                kept_indices=kept_indices,
            )

            per_image_results.append(
                {
                    "image_path": str(img_path),
                    "relative_path": str(rel_path),
                    "output_image_path": str(out_img_path),
                    "output_label_path": str(out_label_path),
                    "output_visualization_path": str(out_vis_path),
                    "num_boxes_before_overlap": num_boxes_before_overlap,
                    "num_boxes": num_boxes,
                    "max_overlap_ratio": max_overlap_ratio,
                    "overlap_filtered_boxes": overlap_filtered_boxes,
                    "overlap_removed_records": overlap_removed_records,
                    "num_predictions_before_merge": int(
                        prediction.get("num_predictions_before_merge", num_boxes_before_overlap)
                    ),
                    "num_predictions_after_merge": int(
                        prediction.get("num_predictions_after_merge", num_boxes_before_overlap)
                    ),
                    "tile_window_count": int(prediction.get("tile_window_count", 0)),
                }
            )

        pbar.update(len(batch_paths))
    pbar.close()

    classes_txt_path = save_classes_txt(output_dir=output_dir, class_names=DEFAULT_CLASS_NAMES)

    summary = {
        "model_path": str(model_path),
        "input_image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "output_images_dir": str(images_out_dir),
        "output_labels_dir": str(labels_out_dir),
        "output_visualizations_dir": str(vis_out_dir),
        "classes_txt_path": str(classes_txt_path),
        "num_images": len(image_paths),
        "detected_images_before_overlap": detected_images_before_overlap,
        "detected_images": detected_images,
        "images_without_boxes": len(image_paths) - detected_images,
        "total_boxes_before_overlap": total_boxes_before_overlap,
        "total_boxes": total_boxes,
        "total_boxes_filtered_by_overlap": total_boxes_filtered_by_overlap,
        "prediction_settings": {
            "imgsz": imgsz,
            "conf": conf_thr,
            "conf_source": conf_source,
            "iou": pred_iou,
            "max_det": max_det,
            "device": device,
            "fp16": test_fp16,
            "batch": batch_size,
            "classes": classes,
            "save_conf_in_txt": save_conf,
            "max_overlap_ratio": max_overlap_ratio,
            "tiled_infer_enabled": tiled_infer_enabled,
            "tiled_merge_iou": tiled_merge_iou if tiled_infer_enabled else None,
            "tiled_tile_size": imgsz if tiled_infer_enabled else None,
            "tiled_tile_stride": (imgsz // 2) if tiled_infer_enabled else None,
            "num_tile_windows": total_tile_windows if tiled_infer_enabled else 0,
        },
        "f1_tuned_conf": {
            "enabled": as_bool(
                resolve_cfg_value(cfg, "PREDICT_USE_F1_OPTIMIZED_CONF", "TEST_USE_F1_OPTIMIZED_CONF", False),
                default=False,
            ),
            "metadata_path": tuned_conf_metadata.get("metadata_path") if tuned_conf_metadata else None,
            "split_dir": tuned_conf_metadata.get("split_dir") if tuned_conf_metadata else None,
            "split_name": tuned_conf_metadata.get("split_name") if tuned_conf_metadata else None,
        },
        "adabn": adabn_info,
        "overlap_filtering": {
            "enabled": max_overlap_ratio is not None,
            "max_overlap_ratio": max_overlap_ratio,
            "num_filtered_boxes": total_boxes_filtered_by_overlap,
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(output_dir / "per_image_results.json", "w", encoding="utf-8") as f:
        json.dump(per_image_results, f, ensure_ascii=False, indent=2)

    print("[Done] Prediction labels saved")
    print(f"- images dir         : {images_out_dir}")
    print(f"- labels dir         : {labels_out_dir}")
    print(f"- classes txt        : {classes_txt_path}")
    print(f"- total images       : {len(image_paths)}")
    print(f"- boxes before overlap: {total_boxes_before_overlap}")
    print(f"- filtered by overlap: {total_boxes_filtered_by_overlap}")
    print(f"- visualizations dir : {vis_out_dir}")
    print(f"- detected images    : {detected_images}")
    print(f"- images w/o boxes   : {len(image_paths) - detected_images}")
    print(f"- total boxes        : {total_boxes}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict unlabeled images to a YOLO-format dataset")
    parser.add_argument(
        "--config",
        default="configs/detection/config_for_predict.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    run_prediction_to_yolo_dataset(cfg)


if __name__ == "__main__":
    main()
