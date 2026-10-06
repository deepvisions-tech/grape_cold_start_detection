from __future__ import annotations

import io
from contextlib import redirect_stdout
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


def _build_coco_api(dataset: Dict[str, Any]):
    from pycocotools.coco import COCO

    coco = COCO()
    coco.dataset = deepcopy(dataset)
    with redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def _build_detection_api(
    ground_truth: Dict[str, Any],
    detections: Sequence[Dict[str, Any]],
):
    annotations: List[Dict[str, Any]] = []
    for annotation_id, detection in enumerate(detections, start=1):
        bbox = [float(value) for value in detection["bbox"]]
        annotations.append(
            {
                "id": int(annotation_id),
                "image_id": int(detection["image_id"]),
                "category_id": int(detection["category_id"]),
                "bbox": bbox,
                "score": float(detection["score"]),
                "area": float(max(0.0, bbox[2]) * max(0.0, bbox[3])),
                "iscrowd": 0,
            }
        )
    return _build_coco_api(
        {
            "info": deepcopy(ground_truth.get("info", {})),
            "licenses": deepcopy(ground_truth.get("licenses", [])),
            "images": deepcopy(ground_truth.get("images", [])),
            "categories": deepcopy(ground_truth.get("categories", [])),
            "annotations": annotations,
        }
    )


def _mean_valid(values: np.ndarray) -> Optional[float]:
    valid = values[values > -1]
    if valid.size == 0:
        return None
    return float(np.mean(valid))


def compute_coco_detection_metrics(
    ground_truth: Dict[str, Any],
    detections: Sequence[Dict[str, Any]],
    *,
    image_ids: Optional[Iterable[int]] = None,
    category_ids: Optional[Iterable[int]] = None,
    max_dets: int = 100,
) -> Dict[str, Any]:
    """Compute canonical COCO bbox AP using COCOeval's 101 recall points.

    The returned primary AP is area=all and maxDets=max_dets. Predictions may
    contain more than ``max_dets`` detections per image; COCOeval performs the
    standard per-image truncation after sorting by score.
    """

    from pycocotools.cocoeval import COCOeval

    resolved_max_dets = int(max_dets)
    if resolved_max_dets < 1:
        raise ValueError("max_dets must be >= 1")

    all_image_ids = [int(item["id"]) for item in ground_truth.get("images", [])]
    all_category_ids = [int(item["id"]) for item in ground_truth.get("categories", [])]
    selected_image_ids = sorted({int(value) for value in (image_ids or all_image_ids)})
    selected_category_ids = sorted({int(value) for value in (category_ids or all_category_ids)})
    if not selected_image_ids:
        raise ValueError("COCO evaluation requires at least one image id")
    if not selected_category_ids:
        raise ValueError("COCO evaluation requires at least one category id")

    selected_image_set = set(selected_image_ids)
    selected_category_set = set(selected_category_ids)
    selected_detections = [
        detection
        for detection in detections
        if int(detection["image_id"]) in selected_image_set
        and int(detection["category_id"]) in selected_category_set
    ]

    coco_gt = _build_coco_api(ground_truth)
    coco_dt = _build_detection_api(ground_truth, selected_detections)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = selected_image_ids
    evaluator.params.catIds = selected_category_ids
    evaluator.params.maxDets = [1, min(10, resolved_max_dets), resolved_max_dets]

    captured = io.StringIO()
    with redirect_stdout(captured):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    precision = evaluator.eval["precision"]
    area_all_index = evaluator.params.areaRngLbl.index("all")
    max_det_index = len(evaluator.params.maxDets) - 1
    ap_by_iou: Dict[str, Optional[float]] = {}
    for threshold_index, threshold in enumerate(evaluator.params.iouThrs):
        ap_by_iou[f"{float(threshold):.2f}"] = _mean_valid(
            precision[
                threshold_index,
                :,
                :,
                area_all_index,
                max_det_index,
            ]
        )

    per_category: Dict[str, Dict[str, Any]] = {}
    for category_index, category_id in enumerate(evaluator.params.catIds):
        category_precision = precision[
            :,
            :,
            category_index,
            area_all_index,
            max_det_index,
        ]
        per_category[str(int(category_id))] = {
            "category_id": int(category_id),
            "ap50_95": _mean_valid(category_precision),
            "ap50": _mean_valid(category_precision[0, :]),
        }

    stats = [float(value) if float(value) >= 0 else None for value in evaluator.stats]
    return {
        "metric_style": "COCOeval_bbox_101_point",
        "iou_thresholds": [float(value) for value in evaluator.params.iouThrs],
        "recall_thresholds": [float(value) for value in evaluator.params.recThrs],
        "area_range": "all",
        "max_dets": resolved_max_dets,
        "num_images": len(selected_image_ids),
        "num_categories": len(selected_category_ids),
        "num_detections_before_coco_max_dets": len(selected_detections),
        "ap50_95": stats[0],
        "ap50": stats[1],
        "ap75": stats[2],
        "ar_max_dets": stats[8],
        "ap_by_iou": ap_by_iou,
        "per_category": per_category,
        "summary": captured.getvalue(),
    }
