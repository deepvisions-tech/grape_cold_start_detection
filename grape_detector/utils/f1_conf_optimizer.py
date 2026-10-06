from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from grape_detector.utils.box_ops import compute_iou, xywhn_to_xyxy


IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def get_tuned_conf_metadata_path(model_path: Path) -> Path:
    model_path = Path(model_path).expanduser().resolve()
    return model_path.parent / f"{model_path.stem}.f1_tuned_conf.json"


def load_tuned_conf_metadata(model_path: Path) -> Optional[Dict]:
    metadata_path = get_tuned_conf_metadata_path(model_path)
    if not metadata_path.exists():
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Tuned confidence metadata must be a JSON object: {metadata_path}")
    data["metadata_path"] = str(metadata_path)
    return data


def _merge_tuned_conf_metadata(model_path: Path, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge metadata without dropping another tuning strategy's fields."""
    metadata_path = get_tuned_conf_metadata_path(model_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = metadata_path.with_suffix(metadata_path.suffix + ".lock")

    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            fcntl = None

        existing: Dict[str, Any] = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError(f"Tuned confidence metadata must be a JSON object: {metadata_path}")
            existing = loaded

        merged = dict(existing)
        merged.update(updates)
        merged["metadata_path"] = str(metadata_path)

        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(metadata_path.parent),
                prefix=f".{metadata_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                json.dump(merged, temp_file, ensure_ascii=False, indent=2)
                temp_file.write("\n")
                temp_path = Path(temp_file.name)
            os.replace(temp_path, metadata_path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return merged


def resolve_split_dirs(split_dir: Path, split_name: str = "val") -> Tuple[Path, Path]:
    split_dir = Path(split_dir).expanduser().resolve()
    split_name = str(split_name).strip()
    images_dir = split_dir / "images" / split_name
    labels_dir = split_dir / "labels" / split_name
    if not images_dir.exists():
        raise FileNotFoundError(f"{split_name} images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"{split_name} labels directory not found: {labels_dir}")
    return images_dir, labels_dir


def load_gt_box_records(
    label_path: Path,
    img_w: int,
    img_h: int,
) -> List[Dict[str, Any]]:
    gt_records: List[Dict[str, Any]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        box = [float(x) for x in parts[1:5]]
        gt_records.append(
            {
                "class_id": cls_id,
                "box": xywhn_to_xyxy(box, img_w, img_h),
            }
        )

    return gt_records


def greedy_match_predictions_to_gt(
    pred_boxes: List[Sequence[float]],
    gt_boxes: List[Sequence[float]],
    iou_thr: float = 0.5,
) -> Tuple[int, int, int]:
    matched_gt = set()
    matched_pred = set()

    candidates = []
    for pi, pbox in enumerate(pred_boxes):
        for gi, gtbox in enumerate(gt_boxes):
            iou = compute_iou(pbox, gtbox)
            if iou >= iou_thr:
                candidates.append((iou, pi, gi))

    candidates.sort(key=lambda x: x[0], reverse=True)

    for _, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)

    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def greedy_match_predictions_to_gt_by_class(
    predictions: List[Dict[str, Any]],
    gt_records: List[Dict[str, Any]],
    target_classes: Set[int],
    iou_thr: float = 0.5,
) -> Tuple[int, int, int]:
    matched_gt = set()
    matched_pred = set()

    candidates = []
    for pred_index, pred in enumerate(predictions):
        pred_class = int(pred["class_id"])
        if pred_class not in target_classes:
            continue
        for gt_index, gt in enumerate(gt_records):
            gt_class = int(gt["class_id"])
            if gt_class not in target_classes or pred_class != gt_class:
                continue
            iou = compute_iou(pred["box"], gt["box"])
            if iou >= iou_thr:
                candidates.append((iou, pred_index, gt_index))

    candidates.sort(key=lambda item: item[0], reverse=True)

    for _, pred_index, gt_index in candidates:
        if pred_index in matched_pred or gt_index in matched_gt:
            continue
        matched_pred.add(pred_index)
        matched_gt.add(gt_index)

    num_target_predictions = sum(
        1 for pred in predictions if int(pred["class_id"]) in target_classes
    )
    num_target_gt = sum(
        1 for gt in gt_records if int(gt["class_id"]) in target_classes
    )
    tp = len(matched_pred)
    fp = num_target_predictions - tp
    fn = num_target_gt - len(matched_gt)
    return tp, fp, fn


def normalize_target_classes(
    target_gt_class: int = 0,
    target_classes: Optional[Sequence[int]] = None,
) -> List[int]:
    if target_classes is None:
        return [int(target_gt_class)]
    normalized = sorted({int(class_id) for class_id in target_classes})
    if not normalized:
        raise ValueError("target_classes must not be empty.")
    return normalized


def _list_image_paths(image_dir: Path) -> List[Path]:
    image_paths: List[Path] = []
    for pattern in IMAGE_PATTERNS:
        image_paths.extend(image_dir.glob(pattern))
    image_paths = sorted(set(image_paths))
    return image_paths


def _build_conf_candidates(conf_min: float, conf_max: float, conf_step: float) -> List[float]:
    if conf_step <= 0:
        raise ValueError("conf_step must be greater than 0.")
    if conf_min < 0 or conf_max > 1 or conf_min > conf_max:
        raise ValueError("conf_min and conf_max must be between 0 and 1, with conf_min <= conf_max.")

    values: List[float] = []
    cur = conf_min
    while cur <= conf_max + 1e-9:
        values.append(round(cur, 6))
        cur += conf_step

    conf_max = round(conf_max, 6)
    if not values or abs(values[-1] - conf_max) > 1e-9:
        values.append(conf_max)

    deduped = sorted(set(values))
    return deduped


def _collect_predictions(
    model_path: Path,
    image_dir: Path,
    label_dir: Path,
    imgsz: int,
    min_conf: float,
    pred_iou: float,
    max_det: int,
    device: str,
    batch_size: int,
    target_gt_class: int,
    target_classes: Optional[Sequence[int]] = None,
    prediction_class_mapping: Optional[Dict[int, int]] = None,
    gamma_correction_enabled: bool = False,
    gamma_correction_value: float = 0.6,
) -> List[Dict]:
    import cv2
    from ultralytics import YOLO
    from grape_detector.utils.gamma_correction import apply_gamma_correction

    image_paths = _list_image_paths(image_dir)
    if not image_paths:
        raise RuntimeError(f"No validation images found: {image_dir}")

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(
                f"Missing validation label: {label_path}. "
                "Use an empty .txt file for a verified negative image."
            )

    model = YOLO(str(model_path))
    records: List[Dict] = []
    gamma_correction_enabled = bool(gamma_correction_enabled)
    gamma_correction_value = float(gamma_correction_value)
    normalized_target_classes = normalize_target_classes(
        target_gt_class=target_gt_class,
        target_classes=target_classes,
    )
    target_class_set = set(normalized_target_classes)

    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start:batch_start + batch_size]
        batch_sources = []
        batch_meta = []

        for img_path in batch_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Skipping image during confidence tuning because it could not be loaded: {img_path}")
                continue

            if gamma_correction_enabled:
                img = apply_gamma_correction(img, gamma=gamma_correction_value)

            h, w = img.shape[:2]
            label_path = label_dir / f"{img_path.stem}.txt"
            gt_records = [
                record
                for record in load_gt_box_records(label_path=label_path, img_w=w, img_h=h)
                if int(record["class_id"]) in target_class_set
            ]

            batch_sources.append(img)
            batch_meta.append((img_path, gt_records))

        if not batch_sources:
            continue

        results = model.predict(
            source=batch_sources,
            imgsz=imgsz,
            conf=min_conf,
            iou=pred_iou,
            max_det=max_det,
            device=device,
            verbose=False,
        )

        for result, (img_path, gt_records) in zip(results, batch_meta):
            predictions = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                for box, score, class_id in zip(xyxy, confs, classes):
                    raw_class_id = int(class_id)
                    class_id = (
                        int(prediction_class_mapping[raw_class_id])
                        if prediction_class_mapping is not None and raw_class_id in prediction_class_mapping
                        else raw_class_id
                    )
                    if class_id not in target_class_set:
                        continue
                    predictions.append({
                        "box": [float(x) for x in box.tolist()],
                        "score": float(score),
                        "class_id": class_id,
                    })

            records.append({
                "image_path": str(img_path),
                "gt_boxes": [record["box"] for record in gt_records],
                "gt_records": gt_records,
                "predictions": predictions,
            })

    if not records:
        raise RuntimeError("Validation predictions for confidence tuning are empty.")
    return records


def _summarize_background_scores(
    score_records: Sequence[Sequence[float]],
    conf_thr: float,
    fp_unit: str = "image",
) -> Dict[str, Any]:
    normalized_fp_unit = normalize_deployment_fp_unit(fp_unit)
    per_image_fp = [
        sum(1 for score in scores if float(score) >= float(conf_thr))
        for scores in score_records
    ]
    object_fp = int(sum(per_image_fp))
    image_fp = int(sum(1 for count in per_image_fp if count > 0))
    return {
        "conf": round(float(conf_thr), 6),
        "fp": image_fp if normalized_fp_unit == "image" else object_fp,
        "fp_unit": normalized_fp_unit,
        "image_fp": image_fp,
        "object_fp": object_fp,
        "images_with_fp": image_fp,
        "max_fp_per_image": int(max(per_image_fp, default=0)),
    }


def normalize_deployment_fp_unit(fp_unit: Any) -> str:
    text = str(fp_unit).strip().lower()
    aliases = {
        "image": "image",
        "images": "image",
        "image_level": "image",
        "object": "object",
        "objects": "object",
        "detection": "object",
        "detections": "object",
        "object_level": "object",
    }
    if text not in aliases:
        raise ValueError("fp_unit must be 'image' or 'object'.")
    return aliases[text]


def select_deployment_threshold(
    score_records: Sequence[Sequence[float]],
    conf_candidates: Sequence[float],
    max_fp: int,
    fp_unit: str = "image",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Select the lowest candidate whose background FP count is <= max_fp."""
    if int(max_fp) < 0:
        raise ValueError("max_fp must be non-negative.")
    if not conf_candidates:
        raise ValueError("Deployment threshold candidates are empty.")
    normalized_fp_unit = normalize_deployment_fp_unit(fp_unit)

    sweep_results = [
        _summarize_background_scores(
            score_records,
            conf_thr=float(conf_thr),
            fp_unit=normalized_fp_unit,
        )
        for conf_thr in sorted({float(value) for value in conf_candidates})
    ]
    selected = next((item for item in sweep_results if item["fp"] <= int(max_fp)), None)
    if selected is None:
        last = sweep_results[-1]
        raise RuntimeError(
            "No confidence threshold in the configured range satisfies the background FP limit: "
            f"fp_unit={normalized_fp_unit}, max_fp={int(max_fp)}, "
            f"conf_max={last['conf']}, fp_at_conf_max={last['fp']}"
        )
    return selected, sweep_results


def _collect_background_prediction_scores(
    model_path: Path,
    image_dir: Path,
    imgsz: int,
    min_conf: float,
    pred_iou: float,
    max_det: int,
    device: str,
    batch_size: int,
    target_classes: Optional[Sequence[int]] = None,
    prediction_class_mapping: Optional[Dict[int, int]] = None,
    use_fp16: bool = False,
    tiled_infer_enabled: bool = False,
    tiled_merge_iou: Optional[float] = None,
    tile_select_box_from_middle: Optional[float] = None,
    disable_tqdm: bool = False,
    gamma_correction_enabled: bool = False,
    gamma_correction_value: float = 0.6,
    model: Any = None,
) -> Tuple[List[List[float]], List[str]]:
    import cv2
    from grape_detector.utils.gamma_correction import apply_gamma_correction
    from grape_detector.utils.tiled_detection_inference import infer_image_with_tiling
    from tqdm import tqdm

    image_paths = _list_image_paths(image_dir)
    if not image_paths:
        raise RuntimeError(f"No background images found: {image_dir}")

    target_class_set = (
        {int(class_id) for class_id in target_classes}
        if target_classes is not None
        else None
    )
    if target_class_set is not None and not target_class_set:
        raise ValueError("target_classes must not be empty.")

    if model is None:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
    score_records: List[List[float]] = []
    used_image_paths: List[str] = []
    with tqdm(
        total=len(image_paths),
        desc="Deployment background inference",
        unit="image",
        disable=bool(disable_tqdm),
    ) as progress:
        for batch_start in range(0, len(image_paths), max(1, int(batch_size))):
            batch_paths = image_paths[batch_start:batch_start + max(1, int(batch_size))]
            batch_sources = []
            valid_paths = []
            for image_path in batch_paths:
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"[WARN] Skipping image during deployment threshold tuning because it could not be loaded: {image_path}")
                    continue
                if gamma_correction_enabled:
                    image = apply_gamma_correction(image, gamma=float(gamma_correction_value))
                batch_sources.append(image)
                valid_paths.append(image_path)

            if batch_sources:
                prediction_arrays = []
                if tiled_infer_enabled:
                    for image in batch_sources:
                        prediction = infer_image_with_tiling(
                            model,
                            image,
                            imgsz=int(imgsz),
                            conf=float(min_conf),
                            iou=float(pred_iou),
                            max_det=int(max_det),
                            device=str(device),
                            half=bool(use_fp16),
                            tile_enabled=True,
                            tile_merge_iou=tiled_merge_iou,
                            tile_select_box_from_middle=tile_select_box_from_middle,
                        )
                        prediction_arrays.append((prediction["scores"], prediction["classes"]))
                else:
                    results = model.predict(
                        source=batch_sources,
                        imgsz=int(imgsz),
                        conf=float(min_conf),
                        iou=float(pred_iou),
                        max_det=int(max_det),
                        device=str(device),
                        half=bool(use_fp16),
                        verbose=False,
                    )
                    for result in results:
                        if result.boxes is not None and len(result.boxes) > 0:
                            prediction_arrays.append(
                                (
                                    result.boxes.conf.detach().cpu().numpy(),
                                    result.boxes.cls.detach().cpu().numpy().astype(int),
                                )
                            )
                        else:
                            prediction_arrays.append(([], []))

                for (confidences, classes), image_path in zip(prediction_arrays, valid_paths):
                    scores: List[float] = []
                    for score, raw_class_id in zip(confidences, classes):
                        raw_class_id = int(raw_class_id)
                        class_id = (
                            int(prediction_class_mapping[raw_class_id])
                            if prediction_class_mapping is not None and raw_class_id in prediction_class_mapping
                            else raw_class_id
                        )
                        if target_class_set is None or class_id in target_class_set:
                            scores.append(float(score))
                    score_records.append(scores)
                    used_image_paths.append(str(image_path))
            progress.update(len(batch_paths))

    if not score_records:
        raise RuntimeError("Background predictions for deployment threshold tuning are empty.")
    return score_records, used_image_paths


def optimize_deployment_threshold_for_background(
    model_path: Path,
    image_dir: Path,
    *,
    max_fp: int = 0,
    fp_unit: str = "image",
    imgsz: int = 960,
    pred_iou: float = 0.7,
    max_det: int = 300,
    device: str = "0",
    batch_size: int = 8,
    target_classes: Optional[Sequence[int]] = None,
    prediction_class_mapping: Optional[Dict[int, int]] = None,
    use_fp16: bool = False,
    tiled_infer_enabled: bool = False,
    tiled_merge_iou: Optional[float] = None,
    tile_select_box_from_middle: Optional[float] = None,
    disable_tqdm: bool = False,
    conf_min: float = 0.01,
    conf_max: float = 0.99,
    conf_step: float = 0.01,
    save_metadata: bool = True,
    gamma_correction_enabled: bool = False,
    gamma_correction_value: float = 0.6,
    model: Any = None,
    model_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model_path = Path(model_path).expanduser().resolve()
    image_dir = Path(image_dir).expanduser().resolve()
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Background image directory not found: {image_dir}")

    conf_candidates = _build_conf_candidates(conf_min, conf_max, conf_step)
    normalized_target_classes = (
        sorted({int(class_id) for class_id in target_classes})
        if target_classes is not None
        else None
    )
    if normalized_target_classes is not None and not normalized_target_classes:
        raise ValueError("target_classes must not be empty.")
    normalized_fp_unit = normalize_deployment_fp_unit(fp_unit)

    print(
        f"[Deployment tuning] model={model_path} | background={image_dir} | "
        f"fp_unit={normalized_fp_unit} | max_fp={int(max_fp)} | "
        f"conf_range={conf_candidates[0]:.2f}~{conf_candidates[-1]:.2f} | "
        f"target_classes={normalized_target_classes or 'all'}"
    )
    score_records, used_image_paths = _collect_background_prediction_scores(
        model_path=model_path,
        image_dir=image_dir,
        imgsz=int(imgsz),
        min_conf=float(conf_candidates[0]),
        pred_iou=float(pred_iou),
        max_det=int(max_det),
        device=str(device),
        batch_size=max(1, int(batch_size)),
        target_classes=normalized_target_classes,
        prediction_class_mapping=prediction_class_mapping,
        use_fp16=bool(use_fp16),
        tiled_infer_enabled=bool(tiled_infer_enabled),
        tiled_merge_iou=tiled_merge_iou,
        tile_select_box_from_middle=tile_select_box_from_middle,
        disable_tqdm=bool(disable_tqdm),
        gamma_correction_enabled=bool(gamma_correction_enabled),
        gamma_correction_value=float(gamma_correction_value),
        model=model,
    )
    selected, sweep_results = select_deployment_threshold(
        score_records=score_records,
        conf_candidates=conf_candidates,
        max_fp=int(max_fp),
        fp_unit=normalized_fp_unit,
    )

    tuning_details = {
        "optimized_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(model_path),
        "model_context": dict(model_context or {}),
        "background_image_dir": str(image_dir),
        "max_fp": int(max_fp),
        "fp_unit": normalized_fp_unit,
        "selected_result": selected,
        "search_space": {
            "conf_min": round(float(conf_min), 6),
            "conf_max": round(float(conf_max), 6),
            "conf_step": round(float(conf_step), 6),
        },
        "predict_settings": {
            "imgsz": int(imgsz),
            "pred_iou": round(float(pred_iou), 6),
            "max_det": int(max_det),
            "device": str(device),
            "batch_size": int(batch_size),
            "fp16": bool(use_fp16),
            "tiled_infer_enabled": bool(tiled_infer_enabled),
            "tiled_merge_iou": (
                float(tiled_merge_iou)
                if tiled_infer_enabled and tiled_merge_iou is not None
                else None
            ),
            "tile_select_box_from_middle": (
                float(tile_select_box_from_middle)
                if tiled_infer_enabled and tile_select_box_from_middle is not None
                else None
            ),
            "disable_tqdm": bool(disable_tqdm),
            "gamma_correction": {
                "enabled": bool(gamma_correction_enabled),
                "gamma": float(gamma_correction_value),
            },
        },
        "evaluation_settings": {
            "target_classes": normalized_target_classes,
            "prediction_class_mapping": (
                {str(key): int(value) for key, value in prediction_class_mapping.items()}
                if prediction_class_mapping is not None
                else None
            ),
            "num_images": len(score_records),
            "num_discovered_images": len(_list_image_paths(image_dir)),
            "num_skipped_images": len(_list_image_paths(image_dir)) - len(used_image_paths),
            "fp_unit": normalized_fp_unit,
        },
        "sweep_results": sweep_results,
    }
    updates = {
        "deployment_threshold": selected["conf"],
        "deployment_tuning": tuning_details,
    }
    if save_metadata:
        metadata = _merge_tuned_conf_metadata(model_path, updates)
        print(
            f"[Deployment tuning complete] deployment_threshold={selected['conf']:.4f}, "
            f"{normalized_fp_unit}_fp={selected['fp']} <= {int(max_fp)}, "
            f"saved={metadata['metadata_path']}"
        )
        return metadata

    metadata = dict(updates)
    metadata["metadata_path"] = str(get_tuned_conf_metadata_path(model_path))
    print(
        f"[Deployment tuning complete] deployment_threshold={selected['conf']:.4f}, "
        f"{normalized_fp_unit}_fp={selected['fp']} <= {int(max_fp)}"
    )
    return metadata


def _summarize_records(
    records: List[Dict],
    conf_thr: float,
    eval_iou_thr: float,
    target_classes: Sequence[int],
) -> Dict:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    target_class_set = set(int(class_id) for class_id in target_classes)
    per_class = {
        int(class_id): {"tp": 0, "fp": 0, "fn": 0}
        for class_id in target_classes
    }

    for record in records:
        selected_predictions = [
            pred
            for pred in record["predictions"]
            if float(pred["score"]) >= conf_thr and int(pred["class_id"]) in target_class_set
        ]
        gt_records = [
            gt
            for gt in record.get("gt_records", [])
            if int(gt["class_id"]) in target_class_set
        ]
        tp, fp, fn = greedy_match_predictions_to_gt_by_class(
            predictions=selected_predictions,
            gt_records=gt_records,
            target_classes=target_class_set,
            iou_thr=eval_iou_thr,
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        for class_id in target_classes:
            class_set = {int(class_id)}
            class_tp, class_fp, class_fn = greedy_match_predictions_to_gt_by_class(
                predictions=selected_predictions,
                gt_records=gt_records,
                target_classes=class_set,
                iou_thr=eval_iou_thr,
            )
            per_class[int(class_id)]["tp"] += class_tp
            per_class[int(class_id)]["fp"] += class_fp
            per_class[int(class_id)]["fn"] += class_fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    per_class_metrics = {}
    for class_id, counts in per_class.items():
        class_precision = (
            counts["tp"] / (counts["tp"] + counts["fp"])
            if (counts["tp"] + counts["fp"]) > 0
            else 0.0
        )
        class_recall = (
            counts["tp"] / (counts["tp"] + counts["fn"])
            if (counts["tp"] + counts["fn"]) > 0
            else 0.0
        )
        class_f1 = (
            2 * class_precision * class_recall / (class_precision + class_recall)
            if (class_precision + class_recall) > 0
            else 0.0
        )
        per_class_metrics[str(class_id)] = {
            "tp": int(counts["tp"]),
            "fp": int(counts["fp"]),
            "fn": int(counts["fn"]),
            "precision": round(float(class_precision), 6),
            "recall": round(float(class_recall), 6),
            "f1": round(float(class_f1), 6),
        }

    return {
        "conf": round(float(conf_thr), 6),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "per_class": per_class_metrics,
    }


def optimize_conf_for_validation(
    model_path: Path,
    split_dir: Path,
    split_name: str = "val",
    imgsz: int = 960,
    pred_iou: float = 0.7,
    max_det: int = 300,
    device: str = "0",
    batch_size: int = 8,
    eval_iou_thr: float = 0.5,
    target_gt_class: int = 0,
    target_classes: Optional[Sequence[int]] = None,
    prediction_class_mapping: Optional[Dict[int, int]] = None,
    conf_min: float = 0.01,
    conf_max: float = 0.99,
    conf_step: float = 0.01,
    save_metadata: bool = True,
    gamma_correction_enabled: bool = False,
    gamma_correction_value: float = 0.6,
) -> Dict:
    model_path = Path(model_path).expanduser().resolve()
    split_dir = Path(split_dir).expanduser().resolve()
    split_name = str(split_name).strip()
    image_dir, label_dir = resolve_split_dirs(split_dir, split_name=split_name)
    conf_candidates = _build_conf_candidates(conf_min, conf_max, conf_step)
    normalized_target_classes = normalize_target_classes(
        target_gt_class=target_gt_class,
        target_classes=target_classes,
    )
    gamma_correction_enabled = bool(gamma_correction_enabled)
    gamma_correction_value = float(gamma_correction_value)

    print(
        f"[F1 tuning] model={model_path} | split={split_dir} | "
        f"split_name={split_name} | images={image_dir} | "
        f"conf_range={conf_candidates[0]:.2f}~{conf_candidates[-1]:.2f} | "
        f"target_classes={normalized_target_classes} | "
        f"gamma_enabled={gamma_correction_enabled} | gamma={gamma_correction_value}"
    )

    records = _collect_predictions(
        model_path=model_path,
        image_dir=image_dir,
        label_dir=label_dir,
        imgsz=int(imgsz),
        min_conf=float(conf_candidates[0]),
        pred_iou=float(pred_iou),
        max_det=int(max_det),
        device=str(device),
        batch_size=max(1, int(batch_size)),
        target_gt_class=int(target_gt_class),
        target_classes=normalized_target_classes,
        prediction_class_mapping=prediction_class_mapping,
        gamma_correction_enabled=gamma_correction_enabled,
        gamma_correction_value=gamma_correction_value,
    )

    sweep_results = [
        _summarize_records(
            records,
            conf_thr=conf_thr,
            eval_iou_thr=float(eval_iou_thr),
            target_classes=normalized_target_classes,
        )
        for conf_thr in conf_candidates
    ]
    best_result = max(
        sweep_results,
        key=lambda item: (item["f1"], item["precision"], item["recall"], -item["conf"]),
    )

    metadata = {
        "optimized_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(model_path),
        "split_dir": str(split_dir),
        "split_name": split_name,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "selection_criterion": "validation_f1",
        "model_state": "checkpoint_before_test_adabn",
        "best_conf": best_result["conf"],
        "best_metrics": best_result,
        "search_space": {
            "conf_min": round(float(conf_min), 6),
            "conf_max": round(float(conf_max), 6),
            "conf_step": round(float(conf_step), 6),
        },
        "predict_settings": {
            "imgsz": int(imgsz),
            "pred_iou": round(float(pred_iou), 6),
            "max_det": int(max_det),
            "device": str(device),
            "batch_size": int(batch_size),
            "gamma_correction": {
                "enabled": bool(gamma_correction_enabled),
                "gamma": float(gamma_correction_value),
            },
        },
        "evaluation_settings": {
            "eval_iou_threshold": round(float(eval_iou_thr), 6),
            "target_gt_class": int(target_gt_class),
            "target_classes": [int(class_id) for class_id in normalized_target_classes],
            "prediction_class_mapping": (
                {str(key): int(value) for key, value in prediction_class_mapping.items()}
                if prediction_class_mapping is not None
                else None
            ),
            "num_images": len(records),
        },
        "sweep_results": sweep_results,
    }

    if save_metadata:
        metadata = _merge_tuned_conf_metadata(model_path, metadata)
        print(
            f"[F1 tuning complete] best_conf={best_result['conf']:.4f}, "
            f"f1={best_result['f1']:.4f}, saved={metadata['metadata_path']}"
        )
    else:
        print(
            f"[F1 tuning complete] best_conf={best_result['conf']:.4f}, "
            f"f1={best_result['f1']:.4f}"
        )

    return metadata
