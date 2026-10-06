from __future__ import annotations

from typing import Dict, List

from grape_detector.utils.box_ops import compute_iou


def compute_ap_from_pr(recalls: List[float], precisions: List[float]) -> float:
    if not recalls:
        return 0.0

    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def _compute_single_class_ap_at_iou(
    records: List[Dict[str, List[List[float]]]],
    iou_thr: float,
) -> float:
    num_gt = sum(len(record["gt_boxes"]) for record in records)
    detections = []

    gt_by_image = {}
    for image_index, record in enumerate(records):
        gt_by_image[image_index] = [{"box": gt_box, "matched": False} for gt_box in record["gt_boxes"]]
        for pred_box, pred_score in zip(record["pred_boxes"], record["pred_scores"]):
            detections.append((float(pred_score), image_index, pred_box))

    detections.sort(key=lambda item: item[0], reverse=True)

    tp_flags: List[int] = []
    fp_flags: List[int] = []
    for _, image_index, pred_box in detections:
        best_iou = 0.0
        best_gt_index = -1
        for gt_index, gt in enumerate(gt_by_image[image_index]):
            if gt["matched"]:
                continue
            iou = compute_iou(pred_box, gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index >= 0 and best_iou >= iou_thr:
            gt_by_image[image_index][best_gt_index]["matched"] = True
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    tp_cum = []
    fp_cum = []
    running_tp = 0
    running_fp = 0
    for tp_flag, fp_flag in zip(tp_flags, fp_flags):
        running_tp += tp_flag
        running_fp += fp_flag
        tp_cum.append(running_tp)
        fp_cum.append(running_fp)

    precisions = []
    recalls = []
    for tp_value, fp_value in zip(tp_cum, fp_cum):
        precision = tp_value / (tp_value + fp_value) if (tp_value + fp_value) > 0 else 0.0
        recall = tp_value / num_gt if num_gt > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

    return compute_ap_from_pr(recalls, precisions) if num_gt > 0 else 0.0


def compute_single_class_ap(
    records: List[Dict[str, List[List[float]]]],
    iou_thr: float = 0.5,
) -> Dict[str, float]:
    num_gt = sum(len(record["gt_boxes"]) for record in records)
    ap50 = _compute_single_class_ap_at_iou(records, iou_thr=iou_thr)

    ap_by_iou: Dict[str, float] = {}
    map_iou_values: List[float] = []
    for step in range(10):
        eval_iou = round(0.5 + 0.05 * step, 2)
        ap_at_iou = _compute_single_class_ap_at_iou(records, iou_thr=eval_iou)
        ap_by_iou[f"{eval_iou:.2f}"] = float(ap_at_iou)
        map_iou_values.append(float(ap_at_iou))

    map50_95 = sum(map_iou_values) / len(map_iou_values) if map_iou_values else 0.0
    return {
        "num_gt": int(num_gt),
        "ap50": float(ap50),
        "map50": float(ap50),
        "map50_95": float(map50_95),
        "ap_by_iou": ap_by_iou,
    }
