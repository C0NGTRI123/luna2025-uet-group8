from typing import List, Dict
import numpy as np
import SimpleITK
import sys
from pathlib import Path

# Add the parent directory to the path to import processor
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from processor import MalignancyProcessor
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

    def _itk_image_to_numpy(self, input_image: SimpleITK.Image) -> tuple:
        """
        Convert SimpleITK image to numpy array with metadata.
        
        Args:
            input_image: SimpleITK image
            
        Returns:
            Tuple of (numpy_image, header_dict)
        """
        def transform(img, point):
            return np.array(
                list(
                    reversed(
                        img.TransformContinuousIndexToPhysicalPoint(
                            list(reversed(point))
                        )
                    )
                )
            )

        numpy_image = SimpleITK.GetArrayFromImage(input_image)
        numpy_origin = np.array(list(reversed(input_image.GetOrigin())))
        numpy_spacing = np.array(list(reversed(input_image.GetSpacing())))

        # Get numpy transform
        t_numpy_origin = transform(input_image, np.zeros((numpy_image.ndim,)))
        t_numpy_matrix_components = [None] * numpy_image.ndim
        for i in range(numpy_image.ndim):
            v = [0] * numpy_image.ndim
            v[i] = 1
            t_numpy_matrix_components[i] = transform(input_image, v) - t_numpy_origin
        numpy_transform = np.vstack(t_numpy_matrix_components).dot(np.diag(1 / numpy_spacing))

        header = {
            "origin": numpy_origin,
            "spacing": numpy_spacing,
            "transform": numpy_transform,
        }

        return numpy_image, header

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
        numpy_image, header = self._itk_image_to_numpy(image)
        
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
