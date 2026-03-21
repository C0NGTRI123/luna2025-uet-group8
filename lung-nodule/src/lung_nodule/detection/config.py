"""Detection configuration loaded from JSON."""

import json
import pathlib
from dataclasses import dataclass, field
from typing import List


@dataclass
class DetectionConfig:
    """All hyperparameters for MONAI RetinaNet detection."""

    score_thresh: float = 0.02
    score_keep: float = 0.3
    topk_candidates_per_level: int = 1000
    nms_thresh: float = 0.22
    detections_per_img: int = 300
    infer_patch_size: List[int] = field(default_factory=lambda: [512, 512, 192])
    overlap: float = 0.25
    sw_batch_size: int = 1
    mode: str = "constant"
    pixdim: List[float] = field(default_factory=lambda: [0.703125, 0.703125, 1.25])
    a_min: float = -1024.0
    a_max: float = 300.0
    b_min: float = 0.0
    b_max: float = 1.0
    clip: bool = True

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "DetectionConfig":
        """Load configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if not k.startswith("_")})

    @classmethod
    def default(cls) -> "DetectionConfig":
        """Load from the bundled detection_config.json."""
        default_path = pathlib.Path(__file__).parent / "detection_config.json"
        return cls.from_json(default_path)
