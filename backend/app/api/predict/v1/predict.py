from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import tempfile
import json
from pathlib import Path

from app.schemas.predict import (
    PredictResponse,
    NodulePoint
)
from app.service.predict_service import PredictService
from app.repository.predict_repository import PredictRepository


predict_router = APIRouter()


@predict_router.post(
    "/",
    response_model=PredictResponse,
    summary="Predict Nodule Malignancy",
    description="Predict malignancy risk for lung nodules from CT images",
)
async def predict_malignancy(
    image: UploadFile = File(..., description="CT image file (.mha format)"),
    nodule_locations: UploadFile = File(..., description="JSON file with nodule locations"),
    clinical_information: Optional[UploadFile] = File(None, description="JSON file with clinical information (optional)"),
    mode: str = Form(default="2D", description="Prediction mode: 2D or 3D"),
) -> PredictResponse:
    """
    Predict malignancy risk for lung nodules.
    
    Args:
        image: Uploaded CT image file (.mha format)
        nodule_locations: JSON file containing nodule locations
        clinical_information: JSON file containing clinical information (optional)
        mode: Prediction mode (2D or 3D)
        
    Returns:
        PredictResponse: Predictions for each nodule
    """
    tmp_image_path = None
    
    try:
        # Read and parse nodule locations JSON file
        nodule_content = await nodule_locations.read()
        nodule_data = json.loads(nodule_content.decode('utf-8'))
        points = [NodulePoint(**point) for point in nodule_data["points"]]
        
        # Read and parse clinical information JSON file if provided
        clinical_data = None
        if clinical_information:
            clinical_content = await clinical_information.read()
            clinical_data = json.loads(clinical_content.decode('utf-8'))
        
        # Extract coordinates and annotation IDs
        coords = [point.point for point in points]
        annotation_ids = [point.name for point in points]
        
        # Save uploaded image file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mha") as tmp_file:
            image_content = await image.read()
            tmp_file.write(image_content)
            tmp_image_path = tmp_file.name
        
        # Initialize repository and service
        repository = PredictRepository()
        service = PredictService(repository=repository)
        
        # Make prediction
        result = service.predict(
            image_path=tmp_image_path,
            coords=coords,
            annotation_ids=annotation_ids,
            clinical_info=clinical_data,
            mode=mode
        )
        
        # Format response
        response = PredictResponse(
            name="Points of interest",
            type="Multiple points",
            points=result.predictions,
            version={"major": 1, "minor": 0}
        )
        
        return response
            
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Clean up temporary image file
        if tmp_image_path:
            Path(tmp_image_path).unlink(missing_ok=True)


@predict_router.get(
    "/status",
    summary="Prediction Service Status",
    description="Check prediction service status and model information"
)
async def get_prediction_status() -> dict:
    """
    Get prediction service status.
    
    Returns:
        dict: Service status information
    """
    try:
        repository = PredictRepository()
        model_path = repository.get_model_path()
        
        return {
            "status": "ready",
            "model_path": model_path,
            "supported_modes": ["2D", "3D"],
            "message": "Prediction service is operational"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Service error: {str(e)}"
        }
