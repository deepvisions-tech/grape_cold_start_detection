from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union


VisualizationDirectories = Dict[str, Optional[Path]]


def prepare_eval_visualization_dirs(
    output_dir: Path,
    *,
    enabled: bool,
) -> VisualizationDirectories:
    """Create TP/FP/FN output directories only when visualization is enabled."""
    directories: VisualizationDirectories = {
        "tp": output_dir / "TP" if enabled else None,
        "fp": output_dir / "FP" if enabled else None,
        "fn": output_dir / "FN" if enabled else None,
    }
    for directory in directories.values():
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
    return directories


def resolve_eval_visualization_path(
    directories: VisualizationDirectories,
    image_name: Union[str, Path],
    *,
    gt_has_positive: bool,
    pred_has_positive: bool,
) -> Optional[Path]:
    """Return the TP/FP/FN image path, or None when no image should be saved."""
    category = None
    if gt_has_positive and pred_has_positive:
        category = "tp"
    elif (not gt_has_positive) and pred_has_positive:
        category = "fp"
    elif gt_has_positive and (not pred_has_positive):
        category = "fn"

    directory = directories.get(category) if category is not None else None
    if directory is None:
        return None
    return directory / Path(image_name).name
