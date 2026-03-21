"""High-level nodule processing: loads CT, coordinates, runs predictions."""

from typing import Dict

import numpy as np
import SimpleITK

from lung_nodule._io import itk_image_to_numpy
from lung_nodule.classification.processor import MalignancyProcessor


class NoduleProcessor:
    def __init__(self, ct_image_file, nodule_locations, clinical_information, mode="2D", model_name="LUNA25-baseline-2D"):
        """
        Parameters
        ----------
        ct_image_file: Path to the CT image file
        nodule_locations: Dictionary containing nodule coordinates and annotationIDs
        clinical_information: Dictionary containing clinical information (Age and Gender)
        mode: 2D or 3D
        model_name: Name of the model to be used for prediction
        """
        self._image_file = ct_image_file
        self.nodule_locations = nodule_locations
        self.clinical_information = clinical_information
        self.mode = mode
        self.model_name = model_name

        self.processor = MalignancyProcessor(mode=mode, suppress_logs=True, model_name=model_name)


    def predict(self, input_image: SimpleITK.Image, coords: np.array) -> Dict:
        """
        Parameters
        ----------
        input_image: SimpleITK Image
        coords: numpy array with list of nodule coordinates

        Returns
        -------
        malignancy risk of the nodules
        """

        numpyImage, header = itk_image_to_numpy(input_image)

        malignancy_risks = []
        for i in range(len(coords)):
            self.processor.define_inputs(numpyImage, header, [coords[i]])
            malignancy_risk, logits = self.processor.predict()
            malignancy_risk = np.array(malignancy_risk).reshape(-1)[0]
            malignancy_risks.append(malignancy_risk)

        malignancy_risks = np.array(malignancy_risks)
        malignancy_risks = list(malignancy_risks)

        return malignancy_risks

    def load_inputs(self):
        # load image
        print(f"Reading {self._image_file}")
        image = SimpleITK.ReadImage(str(self._image_file))

        self.annotationIDs = [p["name"] for p in self.nodule_locations["points"]]
        self.coords = np.array([p["point"] for p in self.nodule_locations["points"]])
        self.coords = np.flip(self.coords, axis=1)  # reverse to [z, y, x] format

        return image, self.coords, self.annotationIDs

    def process(self):
        """
        Load CT scan(s) and nodule coordinates, predict malignancy risk and write the outputs
        """
        image, coords, annotationIDs = self.load_inputs()
        output = self.predict(image, coords)

        assert len(output) == len(annotationIDs), "Number of outputs should match number of inputs"
        results = {
            "name": "Points of interest",
            "type": "Multiple points",
            "points": [],
            "version": {
                "major": 1,
                "minor": 0
            }
        }

        # Populate the "points" section dynamically
        coords = np.flip(coords, axis=1)
        for i in range(len(annotationIDs)):
            results["points"].append(
                    {
                    "name": annotationIDs[i],
                    "point": coords[i].tolist(),
                    "probability": float(output[i])
                    }
                )
        return results
