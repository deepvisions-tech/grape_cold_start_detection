# Grape Cold-Start Detection with YOLO

Detector training and evaluation code for:

> **A cold-start synthetic-to-real framework for grapevine visual symptom
> detection using background-preserving generative editing**  
> Taewoo Kim, Gyuwon Lee, Wonhee Cho, Bongsu Kang, and Junsang Cho.

This repository implements the YOLO detection experiments for symptomatic
grapevine leaves. It provides:

- Dataset splitting, detector training, and low-label fine-tuning.
- Evaluation with COCO bounding-box AP, precision, recall, and F1.
- Prediction export as a YOLO-format dataset for human review.
- Multi-seed experiments, AdaBN, confidence-threshold tuning, and tiled inference.

## Installation

Use Python 3.11 or newer and install PyTorch and torchvision for your platform.
Then, from this repository:

```bash
python -m pip install -e .
```

## Usage

Copy a configuration template and edit its model, data, and output paths:

```bash
cp configs/detection/config_for_train.example.yaml train.local.yaml
```

After editing, run:

```bash
grape-detector-train --config train.local.yaml
```

The same pattern applies to each command:

| Task | Command | Configuration template |
| --- | --- | --- |
| Training | `grape-detector-train --config <config.yaml>` | [Train](configs/detection/config_for_train.example.yaml) |
| Evaluation | `grape-detector-evaluate --config <config.yaml>` | [Evaluate](configs/detection/config_for_test.example.yaml) |
| Relabeling | `grape-detector-predict --config <config.yaml>` | [Predict](configs/detection/config_for_predict.example.yaml) |
| Seed sweeps | `grape-detector-seed-sweep --config <config.yaml>` | [Seed sweep](configs/detection/config_for_train_test_seed_sweep.example.yaml) |

Use separate input and output directories. Training and labeled evaluation
require a matching label for every image; verified negative images use empty
TXT files. Prediction can run on unlabeled images.

## Data format

```text
dataset/
  images/
  labels/
```

YOLO labels contain one box per line: `class cx cy width height`, with normalized
coordinates. Prediction exports images, labels, optional visualizations, and
per-image records. Review predicted labels before using them for training.

## Paper configurations

[Reproduction configurations](configs/reproduction/README.md) cover the
relabeler, synthetic-only training, real-only training, low-label fine-tuning,
and evaluation settings used in the paper.

## Paper results

The companion repository, `grape_cold_start_detection`, provides SAM3–DINOv2
label recovery, anonymous numerical inputs, result tables, and statistical
reproduction code. Its `reproduction/README.md` provides CPU commands for
regenerating the result tables.

## License

This repository is licensed under [AGPL-3.0-only](LICENSE). Third-party terms
are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
