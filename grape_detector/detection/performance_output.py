import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def build_auto_labeling_performance_record(
    *,
    tp: Optional[int],
    fp: Optional[int],
    fn: Optional[int],
    num_predictions: int,
    num_gt_boxes: int,
    matches: Sequence[Dict[str, Any]],
    status: str,
    gt_label_path: Optional[Path],
    iou_threshold: float,
    target_class_id: int,
    eval_class_ids: Sequence[int],
) -> Optional[Dict[str, Any]]:
    """Build the per-image performance block used by auto-labeling outputs."""
    if tp is None or fp is None or fn is None:
        return None

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "num_predictions": int(num_predictions),
        "num_gt_boxes": int(num_gt_boxes),
        "matches": [dict(match) for match in matches],
        "status": str(status),
        "gt_label_path": str(gt_label_path) if gt_label_path is not None else None,
        "candidate_gt_label_paths": None,
        "iou_threshold": float(iou_threshold),
        "target_class_id": int(target_class_id),
        "eval_class_ids": [int(class_id) for class_id in eval_class_ids],
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_auto_labeling_compatible_summary(
    metrics: Dict[str, Any],
    per_image_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a summary.json payload compatible with auto-labeling performance output."""
    object_level = metrics.get("object_level", {})
    prediction_summary = metrics.get("prediction_summary", {})
    evaluation_settings = metrics.get("evaluation_settings", {})
    performance_rows = [
        row["performance"]
        for row in per_image_results
        if isinstance(row.get("performance"), dict)
    ]
    missing_gt_count = sum(
        1
        for performance in performance_rows
        if performance.get("status") == "missing_gt_label"
    )
    num_gt_boxes = sum(
        int(performance.get("num_gt_boxes", 0))
        for performance in performance_rows
    )
    num_images = int(prediction_summary.get("num_images", metrics.get("num_images", 0)))
    detected_images = int(prediction_summary.get("num_pred_images", 0))

    return {
        "input_path": metrics.get("image_dir"),
        "input_root": metrics.get("image_dir"),
        "output_dir": metrics.get("output_dir"),
        "model_path": metrics.get("model_path"),
        "num_images": num_images,
        "detected_images": detected_images,
        "images_without_boxes": max(0, num_images - detected_images),
        "total_boxes": int(prediction_summary.get("num_pred_boxes", 0)),
        "settings": {
            "performance_test_enabled": bool(metrics.get("gt_available", False)),
            "save_visual_outputs": bool(
                metrics.get("visualizations", {}).get("enabled", False)
            ),
            "confidence_threshold": metrics.get("prediction_settings", {}).get("conf"),
        },
        "performance": {
            "enabled": bool(metrics.get("gt_available", False)),
            "gt_label_dir": metrics.get("label_dir"),
            "iou_threshold": evaluation_settings.get("eval_iou_threshold"),
            "target_class_id": metrics.get("target_gt_class"),
            "eval_class_ids": metrics.get("eval_class_ids"),
            "tp": object_level.get("tp"),
            "fp": object_level.get("fp"),
            "fn": object_level.get("fn"),
            "num_predictions": int(prediction_summary.get("num_pred_boxes", 0)),
            "num_gt_boxes": num_gt_boxes,
            "images_missing_gt_label": missing_gt_count,
            "images_duplicate_gt_label": 0,
            "images_evaluated": len(performance_rows),
            "precision": object_level.get("precision"),
            "recall": object_level.get("recall"),
            "f1": object_level.get("f1"),
        },
    }


def write_per_image_result_json(
    output_dir: Path,
    per_image_results: Sequence[Dict[str, Any]],
    *,
    auto_labeling_compatible: bool,
    metrics: Dict[str, Any],
) -> None:
    """Write legacy rows or the opt-in auto-labeling-compatible result files."""
    per_image_payload: Any = (
        {"rows": list(per_image_results)}
        if auto_labeling_compatible
        else list(per_image_results)
    )
    with open(output_dir / "per_image_results.json", "w", encoding="utf-8") as f:
        json.dump(per_image_payload, f, ensure_ascii=False, indent=2)

    if auto_labeling_compatible:
        summary = build_auto_labeling_compatible_summary(metrics, per_image_results)
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
