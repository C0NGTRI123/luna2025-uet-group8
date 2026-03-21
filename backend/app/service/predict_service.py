from typing import List, Dict
import numpy as np

from lung_nodule.classification import MalignancyProcessor
from lung_nodule import itk_image_to_numpy
from app.repository.predict_repository import PredictRepository
from app.schemas.predict import NodulePrediction, PredictionResult


class PredictService:
    """Service for prediction operations."""

    def __init__(self, repository: PredictRepository):
        self._repository = repository
        self._processor = None

    def _initialize_processor(self, mode: str = "2D") -> MalignancyProcessor:
        """
        Initialize the malignancy processor.

        Args:
            mode: Prediction mode (2D or 3D)

        Returns:
            MalignancyProcessor instance
        """
        model_path = self._repository.get_model_path()
        return MalignancyProcessor(
            mode=mode,
            suppress_logs=True,
            model_name=model_path
        )

    def predict(
        self,
        image_path: str,
        coords: List[List[float]],
        annotation_ids: List[str],
        clinical_info: Dict = None,
        mode: str = "2D"
    ) -> PredictionResult:
        """
        Predict malignancy risk for nodules.

        Args:
            image_path: Path to CT image file
            coords: List of nodule coordinates [x, y, z]
            annotation_ids: List of annotation IDs corresponding to coordinates
            clinical_info: Clinical information dictionary (optional)
            mode: Prediction mode (2D or 3D)

        Returns:
            PredictionResult with predictions for each nodule
        """
        # Load image
        image = self._repository.load_image(image_path)

        # Convert coordinates to numpy array
        coords_array = np.array(coords)

        # Transform coordinates to [z, y, x] format
        coords_transformed = self._repository.transform_coordinates(coords_array)

        # Validate inputs
        if not self._repository.validate_inputs(coords_transformed.tolist(), image):
            raise ValueError("Invalid input data for prediction")

        # Convert image to numpy format
        numpy_image, header = itk_image_to_numpy(image)

        # Initialize processor
        if self._processor is None:
            self._processor = self._initialize_processor(mode)

        # Predict for each nodule
        malignancy_risks = []
        for i in range(len(coords_transformed)):
            self._processor.define_inputs(numpy_image, header, [coords_transformed[i]])
            malignancy_risk, logits = self._processor.predict()
            malignancy_risk = np.array(malignancy_risk).reshape(-1)[0]
            malignancy_risks.append(float(malignancy_risk))

        # Create predictions list
        predictions = []
        for i, (annotation_id, coord, risk) in enumerate(zip(annotation_ids, coords, malignancy_risks)):
            predictions.append(
                NodulePrediction(
                    name=annotation_id,
                    point=coord,
                    probability=risk
                )
            )

        return PredictionResult(
            predictions=predictions,
            model_name=self._repository.get_model_path(),
            mode=mode
        )
