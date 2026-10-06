from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Optional


def validate_output_separation(output_dir: Path, inputs: Mapping[str, Any]) -> None:
    """Reject overlapping input/output trees, including symlink aliases."""
    output = output_dir.expanduser().resolve()
    for name, value in inputs.items():
        if value is None or not str(value).strip():
            continue
        source = Path(str(value)).expanduser().resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(
                f"Output directory overlaps {name}: output={output}, input={source}. "
                "Choose a separate output directory."
            )


def normalize_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def ensure_name_suffix(name: str, suffix: str) -> str:
    normalized_name = str(name).strip()
    normalized_suffix = str(suffix).strip()
    if not normalized_name or not normalized_suffix:
        return normalized_name

    suffix_token = normalized_suffix.lstrip("_")
    lower_name = normalized_name.lower()
    lower_token = suffix_token.lower()
    name_tokens = [token for token in re.split(r"[^a-z0-9]+", lower_name) if token]
    if (
        lower_name == lower_token
        or lower_name.endswith(f"_{lower_token}")
        or lower_name.endswith(lower_token)
        or lower_token in name_tokens
    ):
        return normalized_name
    return f"{normalized_name}_{suffix_token}"


def resolve_named_output_dir(
    config: Mapping[str, Any],
    output_dir_key: str,
    run_name_key: str,
    *,
    default_run_name: Any = None,
    resolved_flag_key: Optional[str] = None,
) -> Path:
    output_dir_value = config.get(output_dir_key)
    if output_dir_value is None:
        raise KeyError(f"{output_dir_key} must be set.")

    output_dir = Path(str(output_dir_value)).expanduser()
    if resolved_flag_key and bool(config.get(resolved_flag_key, False)):
        return output_dir

    run_name = normalize_optional_text(config.get(run_name_key))
    if run_name is None:
        run_name = normalize_optional_text(default_run_name)
    if run_name is None:
        return output_dir

    # Backward-compatible guard for configs that already point at the named run dir.
    if output_dir.name == run_name:
        return output_dir
    return output_dir / run_name
