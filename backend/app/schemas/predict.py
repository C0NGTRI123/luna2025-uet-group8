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
