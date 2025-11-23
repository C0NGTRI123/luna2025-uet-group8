from typing import Dict, List, Tuple
import numpy as np
import SimpleITK
from pathlib import Path

from app.core.config import config


class PredictRepository:
    """Repository for prediction operations."""

    def __init__(self, model_path: str = None):
        """
        Initialize predict repository.
        
        Args:
            model_path: Path to the model weights. If None, uses default from config.
        """
        self._model_path = model_path or self._get_default_model_path()

    def _get_default_model_path(self) -> str:
        """Get default model path from configuration or environment."""
        # Default to the 2D baseline model
        return str(Path(__file__).parent.parent.parent.parent / "results" / "LUNA25-baseline-2D-20250225")

    def get_model_path(self) -> str:
        """Get the current model path."""
        return self._model_path

    def load_image(self, image_path: str) -> SimpleITK.Image:
        """
        Load CT image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            SimpleITK.Image: Loaded image
        """
        return SimpleITK.ReadImage(str(image_path))

    def transform_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """
        Transform coordinates to the format expected by the model.
        
        Args:
            coords: Nodule coordinates in [x, y, z] format
            
        Returns:
            Transformed coordinates in [z, y, x] format
        """
        return np.flip(coords, axis=1)

    def validate_inputs(self, coords: List, image: SimpleITK.Image) -> bool:
        """
        Validate input data before prediction.
        
        Args:
            coords: List of nodule coordinates
            image: CT image
            
        Returns:
            True if inputs are valid
        """
        if not coords or len(coords) == 0:
            return False
        
        if image is None:
            return False
            
        return True
