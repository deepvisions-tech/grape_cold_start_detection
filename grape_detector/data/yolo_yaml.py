from pathlib import Path
from typing import Dict, Any
import yaml

from .dataset_splitter import ensure_dir

def write_yolo_yaml(yaml_out: Path, split_root: Path, names: Dict[int, str]) -> None:
    """Write an Ultralytics dataset YAML with paths and class names."""
    data: Dict[str, Any] = {
        "path": str(split_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(k): v for k, v in names.items()},
    }
    ensure_dir(yaml_out.parent)
    with open(yaml_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
