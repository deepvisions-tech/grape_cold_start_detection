from __future__ import annotations

from itertools import product
import re
from typing import Any, Dict, List


ADABN_GRID_SPECS = [
    {
        "key": "TEST_ADABN_EPOCHS",
        "slug": "e",
        "field": "adabn_epochs",
        "none_label": "default",
    },
    {
        "key": "TEST_ADABN_MAX_IMAGES",
        "slug": "mi",
        "field": "adabn_max_images",
        "none_label": "all",
    },
    {
        "key": "TEST_ADABN_MAX_STAGE",
        "slug": "st",
        "field": "adabn_max_stage",
        "none_label": "all",
    },
    {
        "key": "TEST_ADABN_MOMENTUM",
        "slug": "mom",
        "field": "adabn_momentum",
        "none_label": "none",
    },
]


def _stringify_grid_value(value: Any, *, none_label: str) -> str:
    if value is None:
        return none_label
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.12g}"
        return text.replace("-", "m").replace(".", "p")

    text = str(value).strip()
    if not text:
        return none_label
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return none_label
    sanitized = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")
    return sanitized or none_label


def _get_grid_options(cfg: Dict[str, Any], key: str) -> List[Any]:
    raw_value = cfg.get(key)
    if isinstance(raw_value, list):
        return list(raw_value)
    return [raw_value]


def build_adabn_grid_variants(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    option_lists = [_get_grid_options(cfg, spec["key"]) for spec in ADABN_GRID_SPECS]
    has_grid = any(len(options) > 1 for options in option_lists)

    variants: List[Dict[str, Any]] = []
    for combo_index, combo_values in enumerate(product(*option_lists), start=1):
        overrides = {
            spec["key"]: value
            for spec, value in zip(ADABN_GRID_SPECS, combo_values)
        }
        fields = {
            spec["field"]: value
            for spec, value in zip(ADABN_GRID_SPECS, combo_values)
        }
        slug_parts = [
            f"{spec['slug']}{_stringify_grid_value(value, none_label=spec['none_label'])}"
            for spec, value in zip(ADABN_GRID_SPECS, combo_values)
        ]
        combo_name = "default" if not has_grid else "__".join(slug_parts)
        variants.append(
            {
                "combo_index": combo_index,
                "combo_name": combo_name,
                "has_grid": has_grid,
                "overrides": overrides,
                "fields": fields,
            }
        )
    return variants


def apply_adabn_grid_variant(cfg: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in variant.get("overrides", {}).items():
        cfg[key] = value
    cfg["_ADABN_GRID_COMBO_NAME"] = variant.get("combo_name")
    cfg["_ADABN_GRID_HAS_MULTIPLE"] = bool(variant.get("has_grid", False))
    for field_name, value in variant.get("fields", {}).items():
        cfg[f"_{field_name.upper()}"] = value
    return cfg
