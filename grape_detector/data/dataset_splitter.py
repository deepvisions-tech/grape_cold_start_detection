import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

IMG_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TILE_COORD_SUFFIX_RE = re.compile(r"^(?P<prefix>.+)_x\d+_y\d+$", re.IGNORECASE)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_images(images_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)


def has_existing_split(split_root: Path) -> bool:
    checks = [
        split_root / "images" / "train",
        split_root / "images" / "val",
        split_root / "images" / "test",
    ]
    for d in checks:
        if d.exists() and any(d.rglob("*")):
            return True
    return False


def copy_or_symlink(src: Path, dst: Path, use_symlink: bool) -> None:
    ensure_dir(dst.parent)
    if use_symlink:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _json_to_yolo_lines(label_data: Dict[str, Any]) -> List[str]:
    annotations = label_data.get("annotations", [])
    lines: List[str] = []

    for annotation in annotations:
        class_id = annotation.get("class_id")
        if class_id is None:
            raise ValueError(f"JSON annotation is missing class_id: {annotation}")

        x1 = float(annotation["x1"])
        y1 = float(annotation["y1"])
        x2 = float(annotation["x2"])
        y2 = float(annotation["y2"])

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        x1 = _clamp01(x1)
        y1 = _clamp01(y1)
        x2 = _clamp01(x2)
        y2 = _clamp01(y2)

        w = x2 - x1
        h = y2 - y1
        if w <= 0.0 or h <= 0.0:
            continue

        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        lines.append(f"{int(class_id)} {xc:.10f} {yc:.10f} {w:.10f} {h:.10f}")

    return lines


def _write_text_file(dst: Path, content: str) -> None:
    ensure_dir(dst.parent)
    dst.write_text(content, encoding="utf-8")


def _find_label_path(labels_dir: Path, rel: Path) -> Optional[Path]:
    candidates = [
        labels_dir / rel.with_suffix(".txt"),
        labels_dir / rel.with_suffix(".json"),
    ]
    if rel.parent != Path("."):
        candidates.extend(
            [
                labels_dir / f"{rel.stem}.txt",
                labels_dir / f"{rel.stem}.json",
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _require_label_path(labels_dir: Path, rel: Path) -> Path:
    label_path = _find_label_path(labels_dir, rel)
    if label_path is None:
        raise FileNotFoundError(
            f"Missing label for {rel}: {labels_dir}. "
            "Use an empty .txt file for a verified negative image."
        )
    return label_path


def validate_split_labels(split_root: Path) -> None:
    """Check the TXT files consumed by training/validation without changing the split."""
    for split in ("train", "val"):
        image_dir = split_root / "images" / split
        label_dir = split_root / "labels" / split
        for image in list_images(image_dir):
            label = label_dir / image.relative_to(image_dir).with_suffix(".txt")
            if not label.is_file():
                raise FileNotFoundError(
                    f"Missing label: {label}. "
                    "Use an empty .txt file for a verified negative image."
                )


def _place_label_file(src: Path, dst: Path, use_symlink: bool) -> None:
    if src.suffix.lower() == ".txt":
        copy_or_symlink(src, dst, use_symlink)
        return

    if src.suffix.lower() != ".json":
        raise ValueError(f"Unsupported label format: {src}")

    try:
        label_data = json.loads(src.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        label_data = json.loads(src.read_text(encoding="cp949"))

    lines = _json_to_yolo_lines(label_data)
    _write_text_file(dst, "\n".join(lines) + ("\n" if lines else ""))


def _normalize_dataset_roots(dataset_roots: Union[Path, List[Path]]) -> List[Path]:
    if isinstance(dataset_roots, Path):
        roots = [dataset_roots]
    else:
        roots = list(dataset_roots)

    if not roots:
        raise ValueError("dataset_roots must contain at least one path.")

    resolved_roots: List[Path] = []
    for root in roots:
        resolved = Path(root).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset root not found: {resolved}")
        resolved_roots.append(resolved)
    return resolved_roots


def _build_unique_output_rel_paths(
    sample_entries: List[Dict[str, Path]],
) -> List[Dict[str, Path]]:
    grouped: Dict[Path, List[Dict[str, Path]]] = {}
    for sample in sample_entries:
        grouped.setdefault(sample["source_rel"], []).append(sample)

    used_output_rels: Set[Path] = set()
    renamed_count = 0

    for source_rel, grouped_samples in grouped.items():
        if len(grouped_samples) == 1:
            sample = grouped_samples[0]
            output_rel = source_rel
            if output_rel in used_output_rels:
                raise RuntimeError(f"Duplicate relative path in split output: {output_rel}")
            sample["rel"] = output_rel
            used_output_rels.add(output_rel)
            continue

        for index, sample in enumerate(grouped_samples, start=1):
            dataset_root = sample["dataset_root"]
            prefix = dataset_root.name or f"dataset_{index}"
            candidate_name = f"{prefix}_{source_rel.name}"
            output_rel = source_rel.with_name(candidate_name)

            dedupe_index = 2
            while output_rel in used_output_rels:
                output_rel = source_rel.with_name(f"{prefix}_{dedupe_index}_{source_rel.name}")
                dedupe_index += 1

            sample["rel"] = output_rel
            used_output_rels.add(output_rel)
            renamed_count += 1

    if renamed_count > 0:
        print(
            f"[Duplicate filenames] Added dataset root prefixes to {renamed_count} files with duplicate names.",
            flush=True,
        )

    return sample_entries


def _tile_group_key(source_rel: Path) -> Optional[str]:
    match = TILE_COORD_SUFFIX_RE.match(source_rel.stem)
    if match is None:
        return None
    return match.group("prefix")


def _split_samples(
    all_samples: List[Dict[str, Path]],
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[Dict[str, Path]], List[Dict[str, Path]], List[Dict[str, Path]]]:
    tile_key_count = sum(1 for sample in all_samples if _tile_group_key(sample["source_rel"]) is not None)
    if tile_key_count == 0:
        n = len(all_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            all_samples[:n_train],
            all_samples[n_train:n_train + n_val],
            all_samples[n_train + n_val:],
        )

    grouped_samples: Dict[str, List[Dict[str, Path]]] = {}
    for sample in all_samples:
        tile_key = _tile_group_key(sample["source_rel"])
        if tile_key is None:
            tile_key = f"__single__:{sample['dataset_root']}:{sample['source_rel'].as_posix()}"
        grouped_samples.setdefault(tile_key, []).append(sample)

    def should_add_group(current_count: int, group_size: int, target_count: int) -> bool:
        if current_count >= target_count:
            return False
        if current_count == 0:
            return True
        return abs((current_count + group_size) - target_count) <= abs(current_count - target_count)

    groups = list(grouped_samples.values())
    n_groups = len(groups)
    n = len(all_samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_list: List[Dict[str, Path]] = []
    val_list: List[Dict[str, Path]] = []
    test_list: List[Dict[str, Path]] = []

    for group in groups:
        if should_add_group(len(train_list), len(group), n_train):
            train_list.extend(group)
        elif should_add_group(len(val_list), len(group), n_val):
            val_list.extend(group)
        else:
            test_list.extend(group)

    grouped_tile_keys = sum(
        1
        for key, samples in grouped_samples.items()
        if not key.startswith("__single__:") and len(samples) > 1
    )
    print(
        "[Split grouping] Detected filenames with x/y coordinates: "
        f"coord_files={tile_key_count}, groups={n_groups}, grouped_tile_keys={grouped_tile_keys}. "
        "Tile images with the same prefix are assigned to the same split.",
        flush=True,
    )
    return train_list, val_list, test_list


def split_dataset(
    dataset_roots: Union[Path, List[Path]],
    split_root: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    use_symlink: bool = True,
) -> None:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")

    resolved_roots = _normalize_dataset_roots(dataset_roots)
    root_infos: List[Dict[str, Path]] = []
    all_samples: List[Dict[str, Path]] = []

    for dataset_root in resolved_roots:
        images_dir = dataset_root / "images"
        labels_dir = dataset_root / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        images = list_images(images_dir)
        if not images:
            raise RuntimeError(f"No images found in the images directory: {images_dir}")

        root_infos.append(
            {
                "root": dataset_root,
                "images_dir": images_dir,
                "labels_dir": labels_dir,
            }
        )

        for img_path in images:
            source_rel = img_path.relative_to(images_dir)
            label_path = _require_label_path(labels_dir, source_rel)
            all_samples.append(
                {
                    "img_path": img_path,
                    "source_rel": source_rel,
                    "labels_dir": labels_dir,
                    "label_path": label_path,
                    "dataset_root": dataset_root,
                }
            )

    all_samples = _build_unique_output_rel_paths(all_samples)

    print(
        "[Split start] roots={roots}, total={total}, "
        "train={train_ratio}, val={val_ratio}, test={test_ratio}, "
        "seed={seed}, symlink={symlink}".format(
            roots=[str(info["root"]) for info in root_infos],
            total=len(all_samples),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            symlink=use_symlink,
        ),
        flush=True,
    )

    random.seed(seed)
    random.shuffle(all_samples)

    n = len(all_samples)
    train_list, val_list, test_list = _split_samples(all_samples, train_ratio, val_ratio)

    out_img_train = split_root / "images" / "train"
    out_img_val = split_root / "images" / "val"
    out_img_test = split_root / "images" / "test"

    out_lbl_train = split_root / "labels" / "train"
    out_lbl_val = split_root / "labels" / "val"
    out_lbl_test = split_root / "labels" / "test"

    for d in [out_img_train, out_img_val, out_img_test, out_lbl_train, out_lbl_val, out_lbl_test]:
        ensure_dir(d)

    def place(items: List[Dict[str, Path]], out_img_dir: Path, out_lbl_dir: Path, split_name: str) -> None:
        total_items = len(items)
        print(f"[Split {split_name}] start: {total_items} files -> {out_img_dir}", flush=True)
        for idx, item in enumerate(items, start=1):
            img_path = item["img_path"]
            output_rel = item["rel"]
            dst_img = out_img_dir / output_rel
            copy_or_symlink(img_path, dst_img, use_symlink)

            dst_lbl = out_lbl_dir / output_rel.with_suffix(".txt")
            _place_label_file(item["label_path"], dst_lbl, use_symlink)
            if idx == 1 or idx % 500 == 0 or idx == total_items:
                print(f"[Split {split_name}] progress: {idx}/{total_items}", flush=True)

    place(train_list, out_img_train, out_lbl_train, "train")
    place(val_list, out_img_val, out_lbl_val, "val")
    place(test_list, out_img_test, out_lbl_test, "test")

    print(
        f"[Split complete] total={n}, train={len(train_list)}, val={len(val_list)}, test={len(test_list)}",
        flush=True,
    )
    print(f"[Split path] {split_root}", flush=True)


