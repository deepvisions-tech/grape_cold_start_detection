import importlib
import math
import random
import shutil
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
from ultralytics import YOLO
from ultralytics.utils.patches import imread

from grape_detector.utils.gamma_correction import apply_gamma_correction


@contextmanager
def _temporary_disable_ultralytics_tqdm(disable: bool):
    if not disable:
        yield
        return

    patched = []
    try:
        def _disabled_tqdm(*args, **kwargs):
            from tqdm import tqdm
            kwargs["disable"] = True
            return tqdm(*args, **kwargs)

        for module_name in (
            "ultralytics.engine.trainer",
            "ultralytics.engine.validator",
            "ultralytics.data.base",
            "ultralytics.data.dataset",
        ):
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue

            for attr_name in ("TQDM", "tqdm"):
                original = getattr(module, attr_name, None)
                if original is None:
                    continue
                patched.append((module, attr_name, original))
                setattr(module, attr_name, _disabled_tqdm)
        yield
    finally:
        for module, attr_name, original in reversed(patched):
            setattr(module, attr_name, original)


def _resolve_train_split_mixing(split_mix_specs: Optional[List[Dict[str, Union[str, float]]]]):
    if not split_mix_specs:
        return None

    resolved_specs: List[Dict[str, Union[Path, float]]] = []
    alt_prob_sum = 0.0
    for idx, spec in enumerate(split_mix_specs):
        split_dir = Path(spec["dir"]).expanduser().resolve()
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        train_img_dir = split_dir / "images" / "train"
        label_dir = split_dir / "labels" / "train"
        if not train_img_dir.exists():
            raise FileNotFoundError(f"Invalid split directory structure: {train_img_dir}")

        prob = 0.0 if idx == 0 else float(spec.get("prob", 0.0))
        if prob < 0.0 or prob > 1.0:
            raise ValueError(f"Split probability must be between 0 and 1: dir={split_dir}, prob={prob}")
        if idx > 0:
            alt_prob_sum += prob

        resolved_specs.append({
            "dir": split_dir,
            "prob": prob,
            "train_img_dir": train_img_dir,
            "train_label_dir": label_dir,
        })

    if alt_prob_sum > 1.0 + 1e-6:
        raise ValueError(f"Alternate split probabilities must sum to at most 1.0; got {alt_prob_sum:.6f}")

    base_train_dir = resolved_specs[0]["train_img_dir"]
    base_label_dir = resolved_specs[0]["train_label_dir"]
    alternate_cutoffs = []
    cumulative = 0.0
    for spec in resolved_specs[1:]:
        cumulative += float(spec["prob"])
        alternate_cutoffs.append((cumulative, spec["train_img_dir"]))

    def sample_train_path(rel_path: Path) -> Path:
        rel_path = Path(rel_path)
        pick = random.random()
        for cutoff, alt_train_dir in alternate_cutoffs:
            if pick < cutoff:
                alt_path = alt_train_dir / rel_path
                if not alt_path.exists():
                    raise FileNotFoundError(
                        f"Image with matching relative path not found in alternate split: {alt_path} (base rel={rel_path})"
                    )
                return alt_path
        return base_train_dir / rel_path

    return {
        "sample_train_path": sample_train_path,
        "base_train_dir": base_train_dir,
        "base_label_dir": base_label_dir,
    }


def _load_image_uncached(dataset, image_path: Path, rect_mode: bool = True):
    im = imread(str(image_path), flags=dataset.cv2_flag)
    if im is None:
        raise FileNotFoundError(f"Image Not Found {image_path}")

    h0, w0 = im.shape[:2]
    if rect_mode:
        r = dataset.imgsz / max(h0, w0)
        if r != 1:
            w = min(math.ceil(w0 * r), dataset.imgsz)
            h = min(math.ceil(h0 * r), dataset.imgsz)
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    elif not (h0 == w0 == dataset.imgsz):
        im = cv2.resize(im, (dataset.imgsz, dataset.imgsz), interpolation=cv2.INTER_LINEAR)

    if im.ndim == 2:
        im = im[..., None]

    return im, (h0, w0), im.shape[:2]


def train_yolo(
    model_path: Union[str, YOLO],
    data_yaml: str,
    device: str,
    epochs: int,
    imgsz: Optional[int],
    batch: Optional[int],
    run_project_dir: Path,
    run_name: str,
    train_seed: int = 0,
    train_deterministic: bool = True,
    save_period: int = 10,
    amp: bool = True,
    cache: Union[bool, str] = "disk",
    close_mosaic: int = 15,
    rect: bool = True,
    workers: int = 8,
    split_mix_specs: Optional[List[Dict[str, Union[str, float]]]] = None,
    lr0: Optional[float] = None,
    optimizer: Optional[str] = None,
    disable_tqdm: bool = False,
    gamma_correction_enabled: bool = False,
    gamma_correction_value: float = 0.6,
):
    imgsz = int(imgsz) if imgsz is not None else 960
    batch = int(batch) if batch is not None else 2

    run_project_dir.mkdir(parents=True, exist_ok=True)
    (run_project_dir / run_name).mkdir(parents=True, exist_ok=True)

    dynamic_train_sampler = _resolve_train_split_mixing(split_mix_specs)
    effective_cache = cache
    if dynamic_train_sampler is not None and cache not in (False, None, "false", "False", 0, "0"):
        print("[INFO] On-the-fly split mixing enabled; disabling the training image cache.")
        effective_cache = False
    gamma_correction_enabled = bool(gamma_correction_enabled)
    gamma_correction_value = float(gamma_correction_value)
    if gamma_correction_enabled:
        print(f"[Train] gamma correction enabled: gamma={gamma_correction_value}", flush=True)

    import ultralytics.data.dataset as ultralytics_dataset
    original_dataset_init = ultralytics_dataset.YOLODataset.__init__
    original_get_image_and_label = ultralytics_dataset.YOLODataset.get_image_and_label

    def custom_dataset_init(self, *args, **kwargs):
        original_dataset_init(self, *args, **kwargs)
        if dynamic_train_sampler is None or not self.augment:
            self._dynamic_train_split_sampler = None
            self._dynamic_train_rel_paths = None
            return

        self._dynamic_train_split_sampler = dynamic_train_sampler["sample_train_path"]
        base_train_dir = dynamic_train_sampler["base_train_dir"]
        base_label_dir = dynamic_train_sampler["base_label_dir"]
        rel_paths = []
        for index, im_file in enumerate(self.im_files):
            im_path = Path(im_file)
            if len(getattr(self, "label_files", [])) > index:
                label_path = Path(self.label_files[index])
                try:
                    rel_path = label_path.relative_to(base_label_dir).with_suffix(im_path.suffix)
                    rel_paths.append(rel_path)
                    continue
                except ValueError:
                    pass
            try:
                rel_path = im_path.relative_to(base_train_dir)
            except ValueError:
                rel_path = Path(im_path.name)
            rel_paths.append(rel_path)
        self._dynamic_train_rel_paths = rel_paths

    def custom_get_image_and_label(self, index: int):
        sampler = getattr(self, "_dynamic_train_split_sampler", None)
        rel_paths = getattr(self, "_dynamic_train_rel_paths", None)
        if sampler is None or rel_paths is None:
            label = original_get_image_and_label(self, index)
        else:
            label = deepcopy(self.labels[index])
            label.pop("shape", None)
            sampled_im_file = sampler(rel_paths[index])
            label["im_file"] = str(sampled_im_file)
            label["img"], label["ori_shape"], label["resized_shape"] = _load_image_uncached(self, sampled_im_file)
            label["ratio_pad"] = (
                label["resized_shape"][0] / label["ori_shape"][0],
                label["resized_shape"][1] / label["ori_shape"][1],
            )
            if self.rect:
                label["rect_shape"] = self.batch_shapes[self.batch[index]]
            label = self.update_labels_info(label)

        if gamma_correction_enabled:
            label["img"] = apply_gamma_correction(label["img"], gamma=gamma_correction_value)

        return label

    ultralytics_dataset.YOLODataset.__init__ = custom_dataset_init
    ultralytics_dataset.YOLODataset.get_image_and_label = custom_get_image_and_label

    try:
        if isinstance(model_path, YOLO):
            model = model_path
            requested_model_path = getattr(model, "ckpt_path", None) or "<preloaded YOLO model>"
            print(f"[Train] requested YOLO weights: {requested_model_path}", flush=True)
        else:
            resolved_model_path = str(Path(model_path).expanduser().resolve())
            print(f"[Train] requested YOLO weights: {resolved_model_path}", flush=True)
            model = YOLO(model_path)
        actual_ckpt_path = getattr(model, "ckpt_path", None)
        if actual_ckpt_path is not None:
            print(f"[Train] loaded YOLO checkpoint: {actual_ckpt_path}", flush=True)
        else:
            print("[Train] loaded YOLO checkpoint: <unavailable>", flush=True)

        train_overrides: Dict[str, Any] = {}
        if lr0 is not None:
            train_overrides["lr0"] = float(lr0)
            print(f"[Train] requested lr0 override: {float(lr0)}", flush=True)
        if optimizer is not None and str(optimizer).strip():
            train_overrides["optimizer"] = str(optimizer).strip()
            print(f"[Train] requested optimizer override: {train_overrides['optimizer']}", flush=True)

        def _print_effective_lr0(trainer):
            args_lr0 = getattr(getattr(trainer, "args", None), "lr0", None)
            args_optimizer = getattr(getattr(trainer, "args", None), "optimizer", None)
            optimizer_obj = getattr(trainer, "optimizer", None)
            param_group_lrs = []
            if optimizer_obj is not None:
                param_group_lrs = [group.get("lr") for group in optimizer_obj.param_groups]
            unique_lrs = sorted({float(lr) for lr in param_group_lrs if lr is not None})
            print(
                "[Train] effective lr0 check: "
                f"args.lr0={args_lr0}, args.optimizer={args_optimizer}, "
                f"optimizer_lrs={unique_lrs}",
                flush=True,
            )
            if lr0 is not None and unique_lrs and float(lr0) not in unique_lrs:
                print(
                    "[Train][WARN] The optimizer learning rate differs from the requested lr0. "
                    "Ultralytics optimizer=auto may override lr0. "
                    "To use the requested lr0, set OPTIMIZER to an explicit value such as SGD or AdamW.",
                    flush=True,
                )

        model.add_callback("on_pretrain_routine_end", _print_effective_lr0)
        with _temporary_disable_ultralytics_tqdm(disable_tqdm):
            results = model.train(
                data=data_yaml,
                epochs=int(epochs),
                imgsz=imgsz,
                batch=batch,
                device=device,
                seed=int(train_seed),
                deterministic=bool(train_deterministic),
                project=str(run_project_dir),
                name=run_name,
                exist_ok=True,
                save=True,
                save_period=int(save_period),
                val=True,
                plots=True,
                amp=amp,
                cache=effective_cache,
                close_mosaic=close_mosaic,
                rect=rect,
                workers=int(workers),
                **train_overrides,
            )
    finally:
        ultralytics_dataset.YOLODataset.__init__ = original_dataset_init
        ultralytics_dataset.YOLODataset.get_image_and_label = original_get_image_and_label

    run_dir = Path(model.trainer.save_dir)
    return model, run_dir, results


def validate_yolo(
    model: YOLO,
    data_yaml: str,
    run_project_dir: Path,
    run_name: str,
    plots: bool = True,
    disable_tqdm: bool = False,
    gamma_correction_enabled: bool = False,
    gamma_correction_value: float = 0.6,
):
    gamma_correction_enabled = bool(gamma_correction_enabled)
    gamma_correction_value = float(gamma_correction_value)
    if not gamma_correction_enabled:
        with _temporary_disable_ultralytics_tqdm(disable_tqdm):
            return model.val(
                data=data_yaml,
                split="val",
                project=str(run_project_dir),
                name=f"{run_name}_val",
                exist_ok=True,
                plots=plots,
            )

    import ultralytics.data.dataset as ultralytics_dataset

    original_get_image_and_label = ultralytics_dataset.YOLODataset.get_image_and_label

    def custom_get_image_and_label(self, index: int):
        label = original_get_image_and_label(self, index)
        label["img"] = apply_gamma_correction(label["img"], gamma=gamma_correction_value)
        return label

    print(f"[Val] gamma correction enabled: gamma={gamma_correction_value}", flush=True)
    ultralytics_dataset.YOLODataset.get_image_and_label = custom_get_image_and_label
    try:
        with _temporary_disable_ultralytics_tqdm(disable_tqdm):
            return model.val(
                data=data_yaml,
                split="val",
                project=str(run_project_dir),
                name=f"{run_name}_val",
                exist_ok=True,
                plots=plots,
            )
    finally:
        ultralytics_dataset.YOLODataset.get_image_and_label = original_get_image_and_label


def export_checkpoint(
    run_dir: Path,
    final_best_dir: Path,
    run_name: str,
    checkpoint_name: str = "best",
) -> Optional[Path]:
    checkpoint_name = str(checkpoint_name).strip().lower()
    if checkpoint_name not in {"best", "last"}:
        raise ValueError(f"checkpoint_name must be 'best' or 'last': {checkpoint_name}")

    checkpoint_pt = run_dir / "weights" / f"{checkpoint_name}.pt"
    final_best_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_pt.exists():
        dst = final_best_dir / f"{run_name}_{checkpoint_name}.pt"
        shutil.copy2(checkpoint_pt, dst)
        return dst
    return None


