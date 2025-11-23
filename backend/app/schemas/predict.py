from typing import List, Optional
from pydantic import BaseModel, Field


class NodulePoint(BaseModel):
    """Schema for a single nodule point."""
    
    name: str = Field(..., description="Annotation ID of the nodule")
    point: List[float] = Field(..., description="Coordinates [x, y, z] of the nodule")


class ClinicalInformation(BaseModel):
    """Schema for clinical information."""
    
    age: Optional[int] = Field(None, description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")


class PredictRequest(BaseModel):
    """Schema for prediction request."""
    
    nodule_locations: List[NodulePoint] = Field(..., description="List of nodule locations")
    clinical_information: Optional[ClinicalInformation] = Field(None, description="Clinical information")
    image_data: str = Field(..., description="Base64 encoded CT image data or image path")
    mode: str = Field(default="2D", description="Prediction mode: 2D or 3D")


class NodulePrediction(BaseModel):
    """Schema for a single nodule prediction result."""
    
    name: str = Field(..., description="Annotation ID of the nodule")
    point: List[float] = Field(..., description="Coordinates [x, y, z] of the nodule")
    probability: float = Field(..., description="Malignancy probability (0-1)")


class PredictResponse(BaseModel):
    """Schema for prediction response."""
    
    name: str = Field(default="Points of interest", description="Response name")
    type: str = Field(default="Multiple points", description="Response type")
    points: List[NodulePrediction] = Field(..., description="List of predictions")
    version: dict = Field(default={"major": 1, "minor": 0}, description="API version")


class PredictionResult(BaseModel):
    """Schema for internal prediction result."""
    
    predictions: List[NodulePrediction]
    model_name: str
    mode: str


class LesionPredictionRequest(BaseModel):
    """Schema for lesion prediction request."""
    
    seriesInstanceUID: str = Field(..., description="Series instance UID to map image")
    patientID: Optional[str] = Field(None, description="Patient ID")
    studyDate: Optional[str] = Field(None, description="Study date in YYYYMMDD format")
    lesionID: int = Field(..., description="Lesion ID (1, 2, 3...)")
    coordX: float = Field(..., description="World Coordinate X (mm)")
    coordY: float = Field(..., description="World Coordinate Y (mm)")
    coordZ: float = Field(..., description="World Coordinate Z (mm)")
    ageAtStudyDate: Optional[int] = Field(None, description="Patient age at study date")
    gender: Optional[str] = Field(None, description="Patient gender (Male or Female)")


class LesionPredictionData(BaseModel):
    """Schema for lesion prediction data in response."""
    
    seriesInstanceUID: str = Field(..., description="Series instance UID")
    lesionID: int = Field(..., description="Lesion ID")
    probability: float = Field(..., description="Malignancy probability (0.0 - 1.0)")
    predictionLabel: int = Field(..., description="Prediction label: 1 (Malignant) or 0 (Benign)")
    processingTimeMs: int = Field(..., description="Processing time in milliseconds")


class LesionPredictionResponse(BaseModel):
    """Schema for lesion prediction response."""
    
    status: str = Field(default="success", description="Response status")
    data: LesionPredictionData = Field(..., description="Prediction data")
