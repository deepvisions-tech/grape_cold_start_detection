"""Remove intermediate weights after a checkpoint has been exported."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def cleanup_run_weights_after_export(
    *,
    run_dir: Path,
    final_model_path: Path,
    run_name: str = "",
) -> Dict[str, Any]:
    weights_dir = Path(run_dir).expanduser() / "weights"
    final_model_path = Path(final_model_path).expanduser()
    label = run_name or Path(run_dir).name
    info: Dict[str, Any] = {
        "enabled": True,
        "run_name": label,
        "weights_dir": str(weights_dir),
        "final_model_path": str(final_model_path),
        "deleted_files": [],
        "deleted_bytes": 0,
        "skipped_files": [],
        "errors": [],
        "status": "pending",
    }

    if not final_model_path.exists():
        info["status"] = "skipped_final_model_missing"
        print(f"[weights cleanup][SKIP] Final model not found: {final_model_path}", flush=True)
        return info

    if not weights_dir.exists():
        info["status"] = "skipped_weights_dir_missing"
        print(f"[weights cleanup][SKIP] Weights directory not found: {weights_dir}", flush=True)
        return info

    weights_dir_resolved = weights_dir.resolve()
    final_model_resolved = final_model_path.resolve()
    if final_model_resolved.is_relative_to(weights_dir_resolved):
        info["status"] = "skipped_final_model_inside_weights_dir"
        print(
            f"[weights cleanup][SKIP] Skipping cleanup because the final model is inside the weights directory: {final_model_path}",
            flush=True,
        )
        return info

    for weight_path in sorted(weights_dir.glob("*.pt")):
        try:
            weight_resolved = weight_path.resolve()
            if weight_resolved == final_model_resolved:
                info["skipped_files"].append(str(weight_path))
                continue
            size_bytes = weight_path.stat().st_size
            weight_path.unlink()
            info["deleted_files"].append(str(weight_path))
            info["deleted_bytes"] += int(size_bytes)
        except FileNotFoundError:
            continue
        except OSError as exc:
            info["errors"].append({"path": str(weight_path), "error": repr(exc)})

    info["deleted_count"] = len(info["deleted_files"])
    info["skipped_count"] = len(info["skipped_files"])
    info["error_count"] = len(info["errors"])
    info["status"] = "completed" if not info["errors"] else "completed_with_errors"
    print(
        f"[weights cleanup] run_name={label} | "
        f"deleted={info['deleted_count']} files | "
        f"freed={info['deleted_bytes'] / 1024 ** 2:.2f} MB | "
        f"weights_dir={weights_dir}",
        flush=True,
    )
    if info["errors"]:
        print(f"[weights cleanup][WARN] Files that could not be deleted: {info['error_count']}", flush=True)
    return info
