"""Console entry points with configuration-aware CUDA bootstrapping."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Iterable

import yaml


def _config_from_argv(default: str) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=default)
    args, _ = parser.parse_known_args()
    return str(args.config)


def _load_mapping(path: str) -> dict:
    config_path = Path(path)
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config if isinstance(config, dict) else {}


def _bootstrap_visible_device(config_path: str, keys: Iterable[str]) -> None:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    config = _load_mapping(config_path)
    for key in keys:
        value = config.get(key)
        if value is None or not str(value).strip():
            continue
        text = str(value).strip().lower()
        if text.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = text
        elif text.startswith("cuda:") and text.split(":", 1)[1].isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = text.split(":", 1)[1]
        return


def _run(module_name: str):
    return importlib.import_module(module_name).main()


def train():
    config = _config_from_argv("configs/detection/config_for_train.yaml")
    _bootstrap_visible_device(config, ("DEVICE",))
    return _run("grape_detector.detection.train")


def evaluate() -> int:
    config = _config_from_argv("configs/detection/config_for_test.yaml")
    _bootstrap_visible_device(config, ("TEST_DEVICE", "DEVICE"))
    _run("grape_detector.detection.evaluate")
    return 0


def predict():
    config = _config_from_argv("configs/detection/config_for_predict.yaml")
    _bootstrap_visible_device(config, ("PREDICT_DEVICE", "TEST_DEVICE", "DEVICE"))
    return _run("grape_detector.detection.predict_to_yolo_dataset")


def seed_sweep():
    config = _config_from_argv("configs/detection/config_for_train_test_seed_sweep.yaml")
    _bootstrap_visible_device(config, ("SEED_SWEEP_DEVICE", "DEVICE"))
    return _run("grape_detector.detection.seed_sweep")
