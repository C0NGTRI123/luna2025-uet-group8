#!/usr/bin/env python3
"""
run_pipeline.py - End-to-end lung nodule detection + malignancy inference pipeline.

Usage:
    python run_pipeline.py <path/to/scan.zip> [--device cpu|cuda]

Pipeline:
    1. Extract .zip       ->  tmp/dicom/{zip_stem}/
    2. DICOM -> .mha          tmp/mha/{zip_stem}.mha          (malignancy model input)
    3. DICOM -> .nii.gz       tmp/nifti/{zip_stem}/*.nii.gz   (detector input)
    4. Run MONAI RetinaNet 3D detector on .nii.gz
       (build_preprocess: LoadImaged + Orientationd(RAS) + Spacingd + normalize)
    5. Format detection box centers as nodule_locations dict
    6. Run NoduleProcessor (Pulse3D_v2) on .mha using coordinates from step 4
    7. Save output/{zip_stem}_{timestamp}/lung-nodule-malginancy-likelihoods.json

Hyperparameters (patch size, score threshold, NMS, etc.) are in detection_config.json.
"""

import argparse
import json
import os
import sys
import zipfile
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK
import torch
import torchvision  # must be imported before MONAI to avoid partial-init circular import  # noqa: F401

# ── Project root (same dir as this script, works regardless of CWD) ──────────
ROOT = Path(__file__).resolve().parent

# ── Model paths ───────────────────────────────────────────────────────────────
DETECTION_MODEL_PATH = ROOT / "results" / "weights" / "dt_model.ts"
MALIGNANCY_MODEL_NAME = str(ROOT / "results" / "UET-G8-LUNA25-baseline")

# ── Working directories ───────────────────────────────────────────────────────
TMP_DIR = ROOT / "tmp"
OUTPUT_BASE_DIR = ROOT / "output"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Extract ZIP
# ─────────────────────────────────────────────────────────────────────────────

def extract_zip(zip_path: Path, tmp_dir: Path):
    """Extract .zip to tmp/dicom/{zip_stem}/ and return (extraction_root, zip_stem)."""
    zip_stem = zip_path.stem
    extraction_root = tmp_dir / "dicom" / zip_stem
    extraction_root.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Extracting {zip_path.name} -> {extraction_root}")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(extraction_root))

    return extraction_root, zip_stem


def find_dicom_leaf_dir(extraction_root: Path) -> Path:
    """
    Walk the extraction tree to find the first directory that directly contains
    .dcm (or .ima) files.

    Required because SimpleITK.GetGDCMSeriesFileNames() does NOT recurse into
    subdirectories — it only reads files that are directly inside the given path.
    """
    for dirpath, _dirnames, filenames in os.walk(str(extraction_root)):
        dcm_files = [
            f for f in filenames
            if f.lower().endswith(".dcm") or f.lower().endswith(".ima")
        ]
        if dcm_files:
            return Path(dirpath)

    raise FileNotFoundError(
        f"No .dcm/.ima files found anywhere under {extraction_root}. "
        "Verify the .zip contains a valid DICOM series."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Steps 2+3: DICOM → MHA (malignancy) and NIfTI (detection) in one read
# ─────────────────────────────────────────────────────────────────────────────

def convert_dicom_to_outputs(
    dcm_leaf_dir: Path, tmp_dir: Path, zip_stem: str
) -> "tuple[Path, Path]":
    """
    Read the DICOM series once and write two output files:
      - .mha  → used by the malignancy model (NoduleProcessor / Pulse3D_v2)
      - .nii.gz → used by the MONAI RetinaNet detector via build_preprocess()

    If the directory holds multiple DICOM series, the one with the most files
    (most slices) is selected. SimpleITK.WriteImage infers format from extension.

    Returns (mha_path, nifti_path).
    """
    reader = SimpleITK.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dcm_leaf_dir))
    if not series_ids:
        raise FileNotFoundError(
            f"SimpleITK found no DICOM series in {dcm_leaf_dir}."
        )

    # Pick the series with the most slices when there are multiple
    best_series_id = max(
        series_ids,
        key=lambda sid: len(reader.GetGDCMSeriesFileNames(str(dcm_leaf_dir), sid)),
    )
    dicom_names = reader.GetGDCMSeriesFileNames(str(dcm_leaf_dir), best_series_id)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    mha_path = tmp_dir / "mha" / f"{zip_stem}.mha"
    nifti_dir = tmp_dir / "nifti" / zip_stem
    nifti_path = nifti_dir / f"{zip_stem}.nii.gz"

    mha_path.parent.mkdir(parents=True, exist_ok=True)
    nifti_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2/7] Writing MHA  : {mha_path.name}  size={image.GetSize()} spacing={image.GetSpacing()}")
    SimpleITK.WriteImage(image, str(mha_path))

    print(f"[3/7] Writing NIfTI: {nifti_path.name}")
    SimpleITK.WriteImage(image, str(nifti_path))

    return mha_path, nifti_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Run nodule detection on NIfTI using full MONAI pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(nifti_path: Path, device: str) -> list:
    """
    Run the MONAI RetinaNet 3D detector on the CT .nii.gz file.

    Uses the same MONAI preprocessing pipeline as training:
      build_preprocess(): LoadImaged(itkreader) + Orientationd(RAS) + Spacingd + normalize
      build_postprocess(): ClipBoxToImaged + AffineBoxToWorldCoordinated + ConvertBoxModed

    All hyperparameters are read from detection_config.json via lung_node_detection
    module-level variables (score_keep, infer_patch_size, etc.).

    Returns list of {"box": [cx, cy, cz, w, h, d], "score": float}
    Coordinates are world-space mm matching the trained model's output convention.
    """
    from lung_nodule.detection import (
        build_detector, build_preprocess, build_postprocess,
        DetectionConfig,
    )
    from monai.data import Dataset, DataLoader
    from monai.data.utils import no_collation

    det_cfg = DetectionConfig.default()

    print(f"[4/7] Loading detection model from {DETECTION_MODEL_PATH}")
    detector = build_detector(model_path=str(DETECTION_MODEL_PATH), device=device, config=det_cfg)
    print(f"    Patch size: {det_cfg.infer_patch_size}")

    preprocess = build_preprocess(config=det_cfg)
    # NibabelReader stores affine in RAS natively — no extra LPS→RAS flip needed
    postprocess = build_postprocess(affine_lps_to_ras=False)

    ds = Dataset(data=[{"image": str(nifti_path)}], transform=preprocess)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=no_collation)

    # torch.amp.autocast(device_type="cuda") raises RuntimeError on CPU-only machines
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.float16)
        if device == "cuda"
        else nullcontext()
    )

    def to_np(t):
        return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)

    all_detections = []

    print("    Running sliding window inference...")
    for item in dl:
        item = item[0]
        image_4d = item["image"].to(device)           # (1, D, H, W) MetaTensor
        image_for_detector = image_4d.unsqueeze(0)    # (B=1, C=1, D, H, W)

        with torch.no_grad(), amp_ctx:
            out = detector(image_for_detector, use_inferer=True)

        out0 = out[0]
        boxes  = to_np(out0["box"]          if "box"          in out0 else out0.get("boxes"))
        labels = to_np(out0["label"]        if "label"        in out0 else out0.get("labels"))
        scores = to_np(out0["label_scores"] if "label_scores" in out0 else out0.get("scores"))

        # Postprocess: clip boxes to image bounds, voxel->world coords, cccwhd format.
        # MUST pass image_4d (4D), not image_for_detector (5D):
        #   ClipBoxToImaged expects (C, D, H, W) for spatial dimension checks.
        post_out = postprocess({
            "box":          boxes,
            "label":        labels,
            "label_scores": scores,
            "image":        image_4d,  # deleted by DeleteItemsd inside postprocess
        })

        final_boxes  = np.asarray(post_out["box"])
        final_scores = np.asarray(post_out["label_scores"])

        if len(final_scores) > 0:
            print(f"    Candidates: {len(final_scores)}, "
                  f"min={final_scores.min():.3f}, max={final_scores.max():.3f}")
            keep = final_scores >= det_cfg.score_keep
            for box, score in zip(final_boxes[keep].tolist(), final_scores[keep].tolist()):
                all_detections.append({"box": box, "score": float(score)})

    print(f"    Detected {len(all_detections)} nodule(s) with score >= {det_cfg.score_keep}.")
    return all_detections


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Format detections as nodule_locations dict
# ─────────────────────────────────────────────────────────────────────────────

def format_nodule_locations(detections: list) -> dict:
    """
    Convert detection boxes (cccwhd world coords) to the nodule_locations
    dict format consumed by NoduleProcessor.

    Box format from postprocess: [cx, cy, cz, w, h, d]
    NoduleProcessor.load_inputs() (inference.py:131-133) expects:
        nodule_locations["points"] = [{"name": str, "point": [x, y, z]}, ...]
    where [x, y, z] = [cx, cy, cz].

    Do NOT pre-flip coordinates here. NoduleProcessor.load_inputs() applies
    np.flip(coords, axis=1) internally to convert [x,y,z] -> [z,y,x].
    """
    points = []
    for i, det in enumerate(detections):
        cx, cy, cz = det["box"][0], det["box"][1], det["box"][2]
        points.append({
            "name": f"auto_nodule_{i + 1:03d}",
            "point": [-cx, -cy, cz],
        })

    nodule_locations = {
        "name": "Points of interest",
        "type": "Multiple points",
        "points": points,
        "version": {"major": 1, "minor": 0},
    }
    print(f"[5/7] Formatted {len(points)} nodule location(s).")
    return nodule_locations


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Run malignancy inference on MHA
# ─────────────────────────────────────────────────────────────────────────────

def run_malignancy_inference(
    mha_path: Path,
    nodule_locations: dict,
    model_name: str,
) -> dict:
    """
    Run NoduleProcessor (Pulse3D_v2) on the .mha CT file using nodule coordinates
    produced by the detector in step 4.

    model_name is an absolute path. Python's os.path.join drops all previous
    components when it hits an absolute path, so inside processor.py:
        os.path.join("/opt/app/resources/", model_name, "best_metric_model.pth")
    resolves correctly to model_name/best_metric_model.pth (model_root ignored).
    """
    from lung_nodule.classification import NoduleProcessor

    print(f"[6/7] Running malignancy inference (mode=3D-PULSE) on {mha_path.name}")

    processor = NoduleProcessor(
        ct_image_file=str(mha_path),
        nodule_locations=nodule_locations,
        clinical_information={"age": None, "gender": None},
        mode="3D-PULSE",
        model_name=model_name,
    )
    results = processor.process()

    n = len(results.get("points", []))
    print(f"    Predictions computed for {n} nodule(s).")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Save results in Grand-Challenge final test format
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, zip_stem: str, output_base_dir: Path) -> Path:
    """
    Save to output/{zip_stem}_{timestamp}/lung-nodule-malginancy-likelihoods.json

    The filename 'lung-nodule-malginancy-likelihoods.json' matches the
    Grand-Challenge convention exactly (including the 'malginancy' typo).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_base_dir / f"{zip_stem}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / "lung-nodule-malginancy-likelihoods.json"
    with open(str(output_path), "w") as f:
        json.dump(results, f, indent=4)

    print(f"[7/7] Results saved -> {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lung nodule detection + malignancy pipeline from a DICOM .zip file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "zip_path",
        type=str,
        help="Path to the .zip file containing the DICOM series (e.g. data/scan.zip)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Compute device. Auto-detects CUDA if not specified.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip_path).resolve()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not zip_path.exists():
        print(f"ERROR: ZIP file not found: {zip_path}", file=sys.stderr)
        return 1
    if zip_path.suffix.lower() != ".zip":
        print(f"ERROR: Expected a .zip file, got suffix '{zip_path.suffix}'", file=sys.stderr)
        return 1
    if not DETECTION_MODEL_PATH.exists():
        print(f"ERROR: Detection model not found: {DETECTION_MODEL_PATH}", file=sys.stderr)
        return 1
    malignancy_weight = Path(MALIGNANCY_MODEL_NAME) / "best_metric_model.pth"
    if not malignancy_weight.exists():
        print(f"ERROR: Malignancy model weights not found: {malignancy_weight}", file=sys.stderr)
        return 1

    # ── Device selection ──────────────────────────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"Device:    {device}")
    print(f"Input ZIP: {zip_path}")
    print(f"Config:    detection_config.json")
    print("=" * 60)

    # ── Step 1: Extract ZIP ───────────────────────────────────────────────────
    extraction_root, zip_stem = extract_zip(zip_path, TMP_DIR)
    dcm_leaf_dir = find_dicom_leaf_dir(extraction_root)
    print(f"    DICOM leaf dir: {dcm_leaf_dir}")

    # ── Steps 2+3: DICOM → MHA (malignancy) + NIfTI (detection) ─────────────
    mha_path, nifti_path = convert_dicom_to_outputs(dcm_leaf_dir, TMP_DIR, zip_stem)

    # ── Step 4: Detect nodules on NIfTI ──────────────────────────────────────
    detections = run_detection(nifti_path=nifti_path, device=device)

    # ── Handle zero detections ────────────────────────────────────────────────
    if not detections:
        print("WARNING: No nodules detected above score threshold.")
        print("         Saving empty result.")
        results = {
            "name": "Points of interest",
            "type": "Multiple points",
            "points": [],
            "version": {"major": 1, "minor": 0},
        }
    else:
        # ── Step 5: Format detections ─────────────────────────────────────────
        nodule_locations = format_nodule_locations(detections)

        # ── Step 6: Malignancy inference on MHA ───────────────────────────────
        results = run_malignancy_inference(
            mha_path=mha_path,
            nodule_locations=nodule_locations,
            model_name=MALIGNANCY_MODEL_NAME,
        )

    # ── Step 7: Save results ──────────────────────────────────────────────────
    output_path = save_results(results, zip_stem, OUTPUT_BASE_DIR)

    print("=" * 60)
    print(f"Pipeline complete. Output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
