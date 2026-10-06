from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _build_axis_starts(length: int, tile_size: int, stride: int) -> List[int]:
    length = int(length)
    tile_size = max(1, int(tile_size))
    stride = max(1, int(stride))

    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    last_start = length - tile_size
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def build_sliding_tile_windows(
    image_shape: Tuple[int, int, int],
    *,
    tile_size: int,
    stride: int,
) -> List[Tuple[int, int, int, int]]:
    height, width = image_shape[:2]
    x_starts = _build_axis_starts(width, tile_size, stride)
    y_starts = _build_axis_starts(height, tile_size, stride)

    windows: List[Tuple[int, int, int, int]] = []
    for y0 in y_starts:
        y1 = min(y0 + tile_size, height)
        for x0 in x_starts:
            x1 = min(x0 + tile_size, width)
            windows.append((x0, y0, x1, y1))
    return windows


def extract_prediction_arrays(result: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
    scores = boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
    classes = boxes.cls.detach().cpu().numpy().astype(np.int32, copy=False)
    return xyxy, scores, classes


def _compute_iou_vector(box: np.ndarray, other_boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], other_boxes[:, 0])
    y1 = np.maximum(box[1], other_boxes[:, 1])
    x2 = np.minimum(box[2], other_boxes[:, 2])
    y2 = np.minimum(box[3], other_boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    box_area = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    other_areas = np.maximum(0.0, other_boxes[:, 2] - other_boxes[:, 0]) * np.maximum(0.0, other_boxes[:, 3] - other_boxes[:, 1])

    union = box_area + other_areas - inter_area
    ious = np.zeros_like(union, dtype=np.float32)
    valid = union > 0
    ious[valid] = inter_area[valid] / union[valid]
    return ious


def class_aware_nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_thr: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, scores, classes
    if iou_thr is None:
        return boxes_xyxy, scores, classes

    keep_indices: List[int] = []
    unique_classes = np.unique(classes.astype(np.int32, copy=False))
    for cls_id in unique_classes:
        cls_indices = np.where(classes == cls_id)[0]
        if cls_indices.size == 0:
            continue

        order = cls_indices[np.argsort(scores[cls_indices])[::-1]]
        while order.size > 0:
            current = int(order[0])
            keep_indices.append(current)
            if order.size == 1:
                break

            remaining = order[1:]
            ious = _compute_iou_vector(boxes_xyxy[current], boxes_xyxy[remaining])
            order = remaining[ious <= float(iou_thr)]

    if not keep_indices:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    keep = np.array(keep_indices, dtype=np.int32)
    keep = keep[np.argsort(scores[keep])[::-1]]
    return boxes_xyxy[keep], scores[keep], classes[keep]


def _normalize_middle_select_ratio(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null"}:
            return None
        value = text
    ratio = float(value)
    if ratio <= 0.0 or ratio >= 1.0:
        return None
    return ratio


def _build_tile_middle_select_region(
    *,
    window: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    ratio: float,
) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = window
    tile_w = max(1.0, float(x1 - x0))
    tile_h = max(1.0, float(y1 - y0))
    margin_x = tile_w * (1.0 - ratio) / 2.0
    margin_y = tile_h * (1.0 - ratio) / 2.0

    select_x0 = float(x0) if x0 <= 0 else float(x0) + margin_x
    select_y0 = float(y0) if y0 <= 0 else float(y0) + margin_y
    select_x1 = float(x1) if x1 >= image_width else float(x1) - margin_x
    select_y1 = float(y1) if y1 >= image_height else float(y1) - margin_y
    return select_x0, select_y0, select_x1, select_y1


def _filter_boxes_intersecting_region(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    region_xyxy: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, scores, classes

    x0, y0, x1, y1 = region_xyxy
    inter_w = np.maximum(0.0, np.minimum(boxes_xyxy[:, 2], x1) - np.maximum(boxes_xyxy[:, 0], x0))
    inter_h = np.maximum(0.0, np.minimum(boxes_xyxy[:, 3], y1) - np.maximum(boxes_xyxy[:, 1], y0))
    keep = (inter_w * inter_h) > 0.0
    return boxes_xyxy[keep], scores[keep], classes[keep]


def infer_image_with_tiling(
    model: Any,
    image_bgr: np.ndarray,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: Any,
    half: bool = False,
    classes: Optional[List[int]] = None,
    tile_enabled: bool = False,
    tile_merge_iou: Optional[float] = None,
    tile_select_box_from_middle: Optional[float] = None,
) -> Dict[str, Any]:
    tile_size = int(imgsz)
    tile_stride = max(1, tile_size // 2)
    middle_select_ratio = _normalize_middle_select_ratio(tile_select_box_from_middle)

    predict_kwargs: Dict[str, Any] = {
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "device": device,
        "half": half,
        "verbose": False,
    }
    if classes is not None:
        predict_kwargs["classes"] = classes

    predictions_boxes: List[np.ndarray] = []
    predictions_scores: List[np.ndarray] = []
    predictions_classes: List[np.ndarray] = []

    full_result = model.predict(source=image_bgr, **predict_kwargs)[0]
    full_boxes, full_scores, full_classes = extract_prediction_arrays(full_result)
    if len(full_boxes) > 0:
        predictions_boxes.append(full_boxes)
        predictions_scores.append(full_scores)
        predictions_classes.append(full_classes)

    tile_windows = 0
    tile_predictions_removed_by_middle_select = 0
    if tile_enabled:
        windows = build_sliding_tile_windows(
            image_bgr.shape,
            tile_size=tile_size,
            stride=tile_stride,
        )
        tile_windows = len(windows)
        image_height, image_width = image_bgr.shape[:2]
        for x0, y0, x1, y1 in windows:
            tile = image_bgr[y0:y1, x0:x1]
            tile_result = model.predict(source=tile, **predict_kwargs)[0]
            tile_boxes, tile_scores, tile_classes = extract_prediction_arrays(tile_result)
            if len(tile_boxes) == 0:
                continue

            tile_boxes = tile_boxes.copy()
            tile_boxes[:, [0, 2]] += float(x0)
            tile_boxes[:, [1, 3]] += float(y0)
            if middle_select_ratio is not None:
                before_filter_count = len(tile_boxes)
                select_region = _build_tile_middle_select_region(
                    window=(x0, y0, x1, y1),
                    image_width=image_width,
                    image_height=image_height,
                    ratio=middle_select_ratio,
                )
                tile_boxes, tile_scores, tile_classes = _filter_boxes_intersecting_region(
                    tile_boxes,
                    tile_scores,
                    tile_classes,
                    select_region,
                )
                tile_predictions_removed_by_middle_select += before_filter_count - len(tile_boxes)
                if len(tile_boxes) == 0:
                    continue

            predictions_boxes.append(tile_boxes)
            predictions_scores.append(tile_scores)
            predictions_classes.append(tile_classes)

    if predictions_boxes:
        merged_boxes = np.concatenate(predictions_boxes, axis=0)
        merged_scores = np.concatenate(predictions_scores, axis=0)
        merged_classes = np.concatenate(predictions_classes, axis=0)
    else:
        merged_boxes = np.zeros((0, 4), dtype=np.float32)
        merged_scores = np.zeros((0,), dtype=np.float32)
        merged_classes = np.zeros((0,), dtype=np.int32)

    num_predictions_before_merge = int(len(merged_boxes))
    merged_boxes, merged_scores, merged_classes = class_aware_nms(
        merged_boxes,
        merged_scores,
        merged_classes,
        tile_merge_iou,
    )

    return {
        "boxes_xyxy": merged_boxes,
        "scores": merged_scores,
        "classes": merged_classes,
        "num_predictions_before_merge": num_predictions_before_merge,
        "num_predictions_after_merge": int(len(merged_boxes)),
        "tile_enabled": bool(tile_enabled),
        "tile_size": tile_size if tile_enabled else None,
        "tile_stride": tile_stride if tile_enabled else None,
        "tile_window_count": tile_windows,
        "tile_select_box_from_middle": middle_select_ratio if tile_enabled else None,
        "tile_predictions_removed_by_middle_select": int(tile_predictions_removed_by_middle_select),
        "includes_full_image_pass": True,
    }
