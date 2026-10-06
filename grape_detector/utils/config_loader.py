"""YAML configuration loading and shared value parsing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import yaml


def load_config(config_path='config_for_api.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def resolve_tqdm_disable(cfg, *keys, default=False):
    for key in keys:
        if key in cfg:
            return as_bool(cfg.get(key), default=default)
    return default


def normalize_class_names(raw_names: Any) -> Dict[int, str]:
    if raw_names is None:
        return {}
    if isinstance(raw_names, dict):
        return {
            int(class_id): str(class_name)
            for class_id, class_name in raw_names.items()
        }
    if isinstance(raw_names, (list, tuple)):
        return {
            int(class_id): str(class_name)
            for class_id, class_name in enumerate(raw_names)
        }
    return {}


def parse_class_ids(value: Any, class_names: Dict[int, str]) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "none", "null"}:
            return None
        if text.lower() == "all":
            return sorted(class_names.keys()) if class_names else None
        raw_items: Sequence[Any] = [item.strip() for item in text.split(",") if item.strip()]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]

    name_to_id = {
        str(name).strip().lower(): int(class_id)
        for class_id, name in class_names.items()
    }
    class_ids: List[int] = []
    for item in raw_items:
        if isinstance(item, str):
            item_text = item.strip()
            if item_text.lower() == "all":
                class_ids.extend(sorted(class_names.keys()))
                continue
            if item_text.lower() in name_to_id:
                class_ids.append(name_to_id[item_text.lower()])
                continue
            class_ids.append(int(item_text))
        else:
            class_ids.append(int(item))
    return sorted(set(class_ids))
