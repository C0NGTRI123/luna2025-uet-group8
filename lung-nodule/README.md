# lung-nodule

Installable Python library containing all AI logic for the LUNA25 lung nodule malignancy estimation system. Provides nodule detection and malignancy classification.

## Installation

```bash
# Full install (detection + training deps)
pip install -e "lung-nodule[all]"

# Classification only (inference + training)
pip install -e "lung-nodule"

# With detection support (MONAI)
pip install -e "lung-nodule[detection]"
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

## Modules

### `lung_nodule.classification` — Malignancy Classification

#### High-level inference

```python
from lung_nodule.classification import NoduleProcessor

processor = NoduleProcessor(
    image_path="scan.mha",
    nodule_locations_path="nodules.json",
    mode="3D-PULSE",
    model_root="results/",
)
processor.load_inputs()
output = processor.predict()
# output: Grand-Challenge-compatible JSON dict
```

`NoduleProcessor` loads a `.mha` CT file, reads nodule coordinates from JSON, flips axes `[x,y,z]` → `[z,y,x]`, and delegates to `MalignancyProcessor`.

#### Low-level inference

```python
from lung_nodule.classification import MalignancyProcessor
import numpy as np, SimpleITK as sitk

processor = MalignancyProcessor(mode="3D-PULSE", model_root="results/")
image, header = itk_image_to_numpy(sitk.ReadImage("scan.mha"))
coords = np.array([[z, y, x]])  # world-space, already flipped

processor.define_inputs(image, header, coords)
probabilities, logits = processor.predict()
```

#### Prediction modes

| Mode | Model | Description |
|------|-------|-------------|
| `"2D"` | `ResNet18` | 2D ResNet18 baseline |
| `"3D"` | `I3D` | Inflated 3D ConvNet baseline |
| `"3D-PULSE"` | `Pulse3D_v2` | Team model (recommended) |

#### Patch extraction

Patches of **64×64×64 px** (50 mm physical size) are extracted around each nodule centroid using `extract_patch()` from `lung_nodule.classification.data`. Voxel intensities are clipped and scaled with `clip_and_scale()`.

---

### `lung_nodule.classification.models` — Model Architectures

#### `Pulse3D_v2` (team model)

```python
from lung_nodule.classification.models import Pulse3D_v2

model = Pulse3D_v2()
# Input:  (B, 1, 64, 64, 64)
# Output: (B, 1)  — raw logit; apply sigmoid for probability
```

Architecture:
- **Backbone**: `r3d_18` (ResNet3D-18, pretrained) with first conv adapted to 1-channel input
- **Attention**: `SEBlock3D` (Squeeze-and-Excitation, reduction=16) after the backbone
- **Transformer**: 4-layer `ImprovedTransformerBlock`
  - 8-head multi-head attention
  - GEGLU feed-forward network
  - DropPath (stochastic depth, rate=0.2)
  - LayerScale
- **Tokens**: CLS token + 3D sinusoidal positional embeddings
- **Head**: LayerNorm → Linear(512, 512) → GELU → Dropout → Linear(512, 1)

#### `ResNet18` (2D baseline)

```python
from lung_nodule.classification.models import ResNet18
model = ResNet18()
# Input: (B, 1, H, W) — single 2D slice
```

#### `I3D` (3D baseline)

```python
from lung_nodule.classification.models import I3D
model = I3D()
# Input: (B, 1, D, H, W)
```

---

### `lung_nodule.classification.training` — Training

```bash
# Train 2D or 3D baseline
python -m lung_nodule.classification.training.train

# Train Pulse 3D v2
python -m lung_nodule.classification.training.train_pulse_v2
```

Key hyperparameters (from `training/config.py`):

| Parameter | Value |
|-----------|-------|
| Patch size | 64 px / 50 mm |
| Batch size | 32 |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Epochs | 100 |
| Early stopping patience | 10 |
| Rotation augmentation | ±20° per axis |
| Seed | 2025 |

---

### `lung_nodule.detection` — Nodule Detection

MONAI RetinaNet 3D pipeline for detecting nodule candidates in CT scans (inference only).

```python
from lung_nodule.detection import (
    DetectionConfig,
    build_detector,
    build_preprocess,
    build_postprocess,
)

cfg = DetectionConfig()                     # loads detection_config.json
preprocess = build_preprocess(cfg)
detector   = build_detector(cfg)
postprocess = build_postprocess(cfg)
```

Default detection config:

| Parameter | Value |
|-----------|-------|
| `score_thresh` | 0.02 |
| `nms_thresh` | 0.22 |
| `detections_per_img` | 300 |
| `infer_patch_size` | [512, 512, 192] |
| `pixdim` | [0.703125, 0.703125, 1.25] |

---

### `lung_nodule._io` — Shared I/O

```python
from lung_nodule import itk_image_to_numpy
import SimpleITK as sitk

image, header = itk_image_to_numpy(sitk.ReadImage("scan.mha"))
# image:  numpy array (Z, Y, X)
# header: {"origin": ..., "spacing": ..., "transform": ...}
```

This is the single source of truth for SimpleITK → NumPy conversion used across all modules.

## Model Weights

Weight files are **not** included in the package. Each model expects a file at:

```
{model_root}/{model_name}/best_metric_model.pth
```

Default paths (configurable via env vars in the backend):

| Mode | Default path |
|------|-------------|
| `2D` | `results/LUNA25-baseline-2D-20250225/` |
| `3D` | `results/LUNA25-baseline-3D-20250225/` |
| `3D-PULSE` | `results/UET-G8-LUNA25-baseline/` |

## Package Structure

```
lung-nodule/
├── pyproject.toml
└── src/lung_nodule/
    ├── __init__.py                    # Exports itk_image_to_numpy
    ├── _io.py                         # SimpleITK → numpy conversion
    ├── classification/
    │   ├── __init__.py                # Exports MalignancyProcessor, NoduleProcessor
    │   ├── processor.py               # MalignancyProcessor
    │   ├── nodule_processor.py        # NoduleProcessor (high-level)
    │   ├── models/
    │   │   ├── model_2d.py            # ResNet18
    │   │   ├── model_3d.py            # I3D
    │   │   └── pulse_3d_v2.py         # Pulse3D_v2 (team model)
    │   ├── data/
    │   │   ├── dataloader.py          # CTCaseDataset, get_data_loader
    │   │   └── transforms.py          # volumeTransform
    │   └── training/
    │       ├── config.py              # Training hyperparameters
    │       ├── train.py               # Train ResNet18 / I3D
    │       └── train_pulse_v2.py      # Train Pulse3D_v2
    └── detection/
        ├── __init__.py                # Exports builders + DetectionConfig
        ├── config.py                  # DetectionConfig dataclass
        ├── detector.py                # MONAI RetinaNet pipeline builders
        └── detection_config.json      # Bundled default detection config
```
