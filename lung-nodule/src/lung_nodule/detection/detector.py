"""MONAI RetinaNet nodule detector builder functions."""

import torch

from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
    DeleteItemsd,
)
from monai.apps.detection.transforms.dictionary import (
    ClipBoxToImaged,
    AffineBoxToWorldCoordinated,
    ConvertBoxModed,
)

from lung_nodule.detection.config import DetectionConfig

# Detector architecture constants
SPATIAL_DIMS = 3
NUM_CLASSES = 1
SIZE_DIVISIBLE = [16, 16, 8]
FEATURE_MAP_SCALES = [1, 2, 4]
BASE_ANCHOR_SHAPES = [
    [6, 8, 4],
    [8, 6, 5],
    [10, 10, 6],
]


def build_preprocess(config: DetectionConfig = None, image_key: str = "image"):
    """Build MONAI preprocessing pipeline for detection.

    Parameters
    ----------
    config : DetectionConfig, optional
        Detection configuration. Defaults to bundled config.
    image_key : str
        Key for the image in the data dictionary.
    """
    if config is None:
        config = DetectionConfig.default()

    keys = [image_key]
    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        Spacingd(keys=keys, pixdim=config.pixdim, mode="bilinear", padding_mode="border"),
        ScaleIntensityRanged(
            keys=keys,
            a_min=config.a_min, a_max=config.a_max,
            b_min=config.b_min, b_max=config.b_max,
            clip=config.clip,
        ),
        EnsureTyped(keys=keys),
    ]
    return Compose(transforms)


def build_postprocess(
    image_key: str = "image",
    affine_lps_to_ras: bool = True,
    config: DetectionConfig = None,
):
    """Build MONAI postprocessing pipeline for detection.

    Parameters
    ----------
    image_key : str
        Key for the image in the data dictionary.
    affine_lps_to_ras : bool
        Whether to convert affine from LPS to RAS.
    config : DetectionConfig, optional
        Detection configuration (unused currently, reserved for future use).
    """
    return Compose(
        [
            ClipBoxToImaged(
                box_keys="box",
                label_keys="label",
                box_ref_image_keys=image_key,
                remove_empty=True,
            ),
            AffineBoxToWorldCoordinated(
                box_keys="box",
                box_ref_image_keys=image_key,
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            ConvertBoxModed(
                box_keys="box",
                src_mode="xyzxyz",
                dst_mode="cccwhd",
            ),
            DeleteItemsd(keys=[image_key]),
        ]
    )


def build_detector(model_path: str, device: str, config: DetectionConfig = None):
    """Build a RetinaNet detector.

    Parameters
    ----------
    model_path : str
        Path to the TorchScript model file.
    device : str
        Compute device ('cpu' or 'cuda').
    config : DetectionConfig, optional
        Detection configuration. Defaults to bundled config.
    """
    if config is None:
        config = DetectionConfig.default()

    network = torch.jit.load(model_path, map_location=device)

    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=FEATURE_MAP_SCALES,
        base_anchor_shapes=BASE_ANCHOR_SHAPES,
    )
    detector = RetinaNetDetector(
        network=network,
        anchor_generator=anchor_generator,
        spatial_dims=SPATIAL_DIMS,
        num_classes=NUM_CLASSES,
        size_divisible=SIZE_DIVISIBLE,
    )
    detector.set_target_keys(box_key="box", label_key="label")
    detector.set_box_selector_parameters(
        score_thresh=config.score_thresh,
        topk_candidates_per_level=config.topk_candidates_per_level,
        nms_thresh=config.nms_thresh,
        detections_per_img=config.detections_per_img,
    )
    detector.set_sliding_window_inferer(
        roi_size=config.infer_patch_size,
        overlap=config.overlap,
        sw_batch_size=config.sw_batch_size,
        mode=config.mode,
        device=device,
    )
    detector.eval()

    return detector
