# Third-party software

This project uses the `ultralytics` Python package and Ultralytics YOLO model
artifacts. The upstream open-source distribution is licensed under
AGPL-3.0; Ultralytics also offers separate commercial terms.

- Source license: <https://github.com/ultralytics/ultralytics/blob/main/LICENSE>
- Licensing overview: <https://www.ultralytics.com/license>

Canonical COCO bounding-box AP is computed with the `pycocotools` package,
distributed under the FreeBSD license.

- Package source: <https://github.com/ppwwyyxx/cocoapi>

No third-party source code, pretrained weights, datasets, or model checkpoints
are vendored in this repository.

This repository's own source is licensed under AGPL-3.0-only (see `LICENSE`),
chosen for compatibility with the upstream Ultralytics distribution. Detector
checkpoints trained with `ultralytics` are derived works of an AGPL-3.0
package; distribute them only under terms consistent with the Ultralytics
license, or under separate commercial terms obtained from Ultralytics.
