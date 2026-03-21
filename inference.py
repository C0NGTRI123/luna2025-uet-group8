from typing import Dict, Tuple

from pathlib import Path
import json
from glob import glob
import numpy as np
from lung_nodule.classification import NoduleProcessor


INPUT_PATH = Path("test/input")
OUTPUT_PATH = Path("test/output")
RESOURCE_PATH = Path("results/LUNA25-baseline-2D-20250225")


def run(mode="2D", model_name="LUNA25-baseline-2D"):
    # Read the inputs
    input_nodule_locations = load_json_file(
        location=INPUT_PATH / "nodule-locations.json",
    )
    input_clinical_information = load_json_file(
        location=INPUT_PATH / "clinical-information-lung-ct.json",
    )
    input_chest_ct = load_image_path(
        location=INPUT_PATH / "images/chest-ct",
    )

    # Validate access to GPU
    _show_torch_cuda_info()

    # Run your algorithm here
    processor = NoduleProcessor(ct_image_file=input_chest_ct,
                                nodule_locations=input_nodule_locations,
                                clinical_information=input_clinical_information,
                                mode=mode,
                                model_name=model_name)
    malignancy_risks = processor.process()

    # Save your output
    write_json_file(
        location=OUTPUT_PATH / "lung-nodule-malginancy-likelihoods.json",
        content=malignancy_risks,
    )
    print(f"Completed writing output to {OUTPUT_PATH}")
    print(f"Output: {malignancy_risks}")
    return 0


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_image_path(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )

    assert (
                len(input_files) == 1
            ), "Please upload only one .mha file per job for grand-challenge.org"

    result = input_files[0]

    return result


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch version: {torch.version.cuda}")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    mode = "3D-PULSE"
    model_name = "/home/congtri/project/luna2025-uet-group8/results/UET-G8-LUNA25-baseline"
    raise SystemExit(run(mode=mode,
                         model_name=model_name))
