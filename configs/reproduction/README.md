# Reproduction configurations

These templates describe the final reported runtime recipes. Supply the local
data and checkpoint paths, the scenario's saved split, and the checkpoint's
training seed. Original images and trained checkpoints are not distributed.
The templates under `configs/detection/` remain generic starting points.
To reproduce numerical tables without images or models, use the automatic-labeling
repository's [anonymous result workflow](../../README.md#paper-results).

Copy a template before running, and keep the `.example.yaml` files unchanged:

```bash
cp configs/reproduction/03_synthetic_only_train.example.yaml \
   configs/reproduction/03_synthetic_only_train.yaml
grape-detector-train --config configs/reproduction/03_synthetic_only_train.yaml
```

Local non-example YAML files in this directory are ignored by Git.

## Arm-to-file map

| Manuscript arm | Section | Config |
| --- | --- | --- |
| YOLO relabeler $r_\phi$ training | 4.3 | `01_relabeler_train.example.yaml` |
| YOLO relabeler proposals on batch 2 | 4.3 | `02_relabeler_predict.example.yaml` |
| Cold-start (synthetic-only) training | 5.2 | `03_synthetic_only_train.example.yaml` |
| Synthetic pretraining stage | 5.2 | `03_synthetic_only_train.example.yaml` |
| Generation ablation, prompt-only arm | 5.3 | `03_synthetic_only_train.example.yaml` (dataset swapped) |
| Real-only($k$) training | 5.2 | `04_real_label_train.example.yaml` |
| Low-label fine-tuning stage | 5.2 | `04_real_label_train.example.yaml` |
| Curated-mix evaluation, with AdaBN | 5.1, 5.3 | `05_evaluate_curated_mix_adabn.example.yaml` |
| Curated-mix evaluation, no AdaBN | 5.1, 5.3 | `06_evaluate_curated_mix_no_adabn.example.yaml` |
| Deployment-prevalence threshold and evaluation | 5.1 | `07_evaluate_deployment_threshold.example.yaml` |
| 15-seed repetition, cold-start synthetic-only + AdaBN | 5.3 | `08_seed_sweep_15_seeds.example.yaml` |
| 15-seed repetition, pretrain+fine-tune | 5.3 | `09_seed_sweep_pretrain_finetune.example.yaml` |
| 15-seed repetition, real-only($k$) | 5.3 | `10_seed_sweep_real_only.example.yaml` |
| 15-seed repetition, generation-source comparison without AdaBN | 5.3 | `11_seed_sweep_generation_ablation.example.yaml` |

The generation ablation's prompt-only arm uses `03_synthetic_only_train`
unchanged except for `DATASET_ROOTS` / `SAVED_SPLIT_DIRS`, and is evaluated
with `06_evaluate_curated_mix_no_adabn` because the manuscript specifies that
comparison without AdaBN.

Choose the sweep for the scenario; changing only the training file in the
cold-start sweep would leave AdaBN enabled. Copy the referenced base train and
evaluation examples to local `.yaml` files and set their paths before running:

| Sweep | Base training file | Base evaluation file | `TEST_TUNE_SPLIT_DIR` |
| --- | --- | --- | --- |
| `08` cold start | `03_synthetic_only_train.yaml` | `05_evaluate_curated_mix_adabn.yaml` | the saved synthetic split |
| `09` pretrain + fine-tune | `04_real_label_train.yaml` | `06_evaluate_curated_mix_no_adabn.yaml` | the fixed real validation split |
| `10` real-only | `04_real_label_train.yaml` | `06_evaluate_curated_mix_no_adabn.yaml` | the fixed real validation split |
| `11` generation comparison, each source separately | `03_synthetic_only_train.yaml` | `06_evaluate_curated_mix_no_adabn.yaml` | that source's saved synthetic split |

For `11`, run the background-edited and prompt-only datasets separately, with
their corresponding training and validation paths. Use distinct sweep prefixes
(`grape_generation_editing_15_seed` and `grape_generation_prompt_only_15_seed`)
to keep their outputs separate. For `09` and `10`, set the same real budget and
validation split; `09` initializes from the matching seed's synthetic checkpoint.

## Settings shared by every reported detector

| Item | Value |
| --- | --- |
| Initial weights | `yolo26x.pt` |
| Classes | 1 (`magnesium deficiency`) |
| Input resolution / epochs / batch | 960 / 50 / 8 |
| Precision, determinism | `AMP: true`, `TRAIN_DETERMINISTIC: true` |
| Mosaic | disabled throughout (`CLOSE_MOSAIC: 50`) |
| Split seed | 42 |
| Postprocessing | Default YOLO26 end-to-end inference without NMS; the configured IoU 0.7 argument is inactive |
| Validation-threshold search | F1 sweep on `val` before test-time AdaBN; max det 300; conf 0.01–0.99 step 0.01 |
| Reported AP | COCOeval bbox, 101 recall points, IoU 0.50:0.95, area=all, maxDets=100, export conf 0.001 |

Scenario-specific differences:

| Scenario | Optimizer | Learning rate |
| --- | --- | --- |
| Synthetic-only, synthetic pretraining, relabeler | Ultralytics automatic (`OPTIMIZER:` and `LR0:` left empty) | automatic |
| Real-only($k$), low-label fine-tuning | `adamW` | `0.0001` |

## Seeds

The 15 matched training seeds used for every ordinary repetition are:

```
111, 222, 333, 444, 555, 666, 777, 888, 999, 1111, 1424, 1679, 2289, 8935, 9674
```

The low-label subset-resampling robustness experiment (Section 5.2) instead
crosses five data-subset draws with three training seeds:

- data-subset seeds: `409, 1824, 3657, 4012, 4506`
- training seeds: `1679, 2289, 8935`

The relabeler uses training seed `11`. Split seed `42` fixes only the data
split and is never used as a training seed.

## Provenance of these values

Values were transcribed from the archived run records rather than retyped from
the manuscript, except where noted below:

- Training settings come from the per-seed `effective_train_config.yaml`
  records. The final AdaBN recipe comes from the canonical evaluation metrics:
  128 images from the 1,750-image second background batch, stage 1 (stages 0–1,
  two BN layers), and random sampling without replacement keyed to training seed.
  Earlier `effective_test_config.yaml` snapshots can contain superseded settings.
- The relabeler proposal threshold (`PREDICT_CONF: 0.40`) comes from the final
  `labeling_test_yolo26x2` run record.
- The real-label optimizer settings (`adamW`, `lr0=0.0001`, `batch=8`) come from
  the fine-tuning run log.
- The 15 training seeds come from the canonical statistics script's
  `EXPECTED_SEEDS`.
- Deployment settings are checked against the retained final operating-point
  records: at most 25 of 5,000 background tiles may contain a detection. For
  cold start, calibration receives the model after AdaBN. AP and archived
  operating-point results are preserved as separate sources in the numeric inputs.

The curated templates recompute validation F1 thresholds before AdaBN, and the
deployment template recalibrates after any AdaBN. This avoids treating mutable
checkpoint-side threshold files as authoritative under a different model state.
For a standalone adapted evaluation, set `TEST_ADABN_SAMPLE_SEED` to the training
seed. The 15-seed sweep resolves its `train_seed` setting for each model. Selection
uses the full source pool, then processes the selected images in filename order.

The training splitter remains a general-purpose utility. If `SAVED_SPLIT_DIRS`
is empty it may create a new random split; that is a new training experiment,
not a reconstruction of the paper's fixed dataset. This does not affect the
public numeric-table reproduction command.

## Effective values after configuration fallback

Two AdaBN keys are left empty in the reported evaluation config and fall back
to the test-time values, which is what the runs actually used:

- `TEST_ADABN_BATCH:` empty → `TEST_BATCH` (32)
- `TEST_ADABN_IMGSZ:` empty → `TEST_IMGSZ` (960)

`TEST_ADABN_MOMENTUM: null` is not the same as omitting the key. The loader
maps `null` to `None`, which makes PyTorch use a cumulative moving average for
the batch-normalization running statistics; omitting the key instead keeps each
layer's original momentum. The reported runs used `null`.

Likewise, `TEST_ADABN_RESET_STATS` defaults to `true` in code but the reported
runs set it to `false`. Both keys are written explicitly in the templates here
so the reported behavior does not depend on a default.

## Environment used for the reported runs

| Component | Version |
| --- | --- |
| Python | 3.13.13 |
| PyTorch / torchvision | 2.10.0+cu128 / 0.25.0+cu128 |
| PyTorch CUDA build / cuDNN package | 12.8 / 9.10.2.21 |
| Ultralytics | 8.4.60 |
| NumPy / OpenCV | 1.26.4 / 4.10.0.82 |
| SciPy / Pillow | 1.17.1 / 12.2.0 |
| pycocotools | 2.0.11 |
| PyYAML / tqdm | 6.0.3 / 4.67.3 |

Hardware: a Naver Cloud `gp2l16-g3` instance with two NVIDIA L4 GPUs, 16 vCPUs,
and 96 GB of system memory.

All reported YOLO training and inference and the companion SAM 3--DINOv2
automatic-labeling runs used this same Conda environment. PyTorch and
torchvision are recorded here as execution provenance but are not version-pinned
by the public package because users need platform-appropriate builds.

## Scope

This repository covers detector training, evaluation, relabeling and seed
sweeps. Anonymous numeric inputs, source-frame presence aggregation, paired
statistical analyses and result-table generation are maintained in the main
automatic-labeling repository, including the 180 saved low-label
subset-resampling evaluations. Repeating the underlying training and
image-level inference requires private inputs.
