from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import tempfile
import json
from pathlib import Path
import time

from app.schemas.predict import (
    PredictResponse,
    NodulePoint,
    LesionPredictionResponse,
    LesionPredictionData
)
from app.service.predict_service import PredictService
from app.repository.predict_repository import PredictRepository
from app.core.exceptions import (
    InvalidFileFormatException,
    ProcessingErrorException,
    InternalServerErrorException
)


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
) -> PredictResponse:
    """
    Predict malignancy risk for lung nodules using Pulse 3D v2 model.
    
    Args:
        image: Uploaded CT image file (.mha format)
        nodule_locations: JSON file containing nodule locations
        clinical_information: JSON file containing clinical information (optional)
        
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
            mode="3D-PULSE"
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


@predict_router.post(
    "/lesion",
    response_model=LesionPredictionResponse,
    summary="Predict Lesion Malignancy",
    description="Predict malignancy for a single lung lesion from CT image and metadata",
)
async def predict_lesion(
    file: UploadFile = File(..., description="CT image file (.mha or .mhd format)"),
    seriesInstanceUID: str = Form(..., description="Series instance UID to map image"),
    lesionID: int = Form(..., description="Lesion ID (1, 2, 3...)"),
    coordX: float = Form(..., description="World Coordinate X (mm)"),
    coordY: float = Form(..., description="World Coordinate Y (mm)"),
    coordZ: float = Form(..., description="World Coordinate Z (mm)"),
    patientID: Optional[str] = Form(None, description="Patient ID"),
    studyDate: Optional[str] = Form(None, description="Study date in YYYYMMDD format"),
    ageAtStudyDate: Optional[int] = Form(None, description="Patient age at study date"),
    gender: Optional[str] = Form(None, description="Patient gender (Male or Female)"),
) -> LesionPredictionResponse:
    """
    Predict malignancy for a single lung lesion.
    
    Args:
        file: Uploaded CT image file (.mha or .mhd format)
        seriesInstanceUID: Series instance UID
        lesionID: Lesion ID
        coordX: World coordinate X in mm
        coordY: World coordinate Y in mm
        coordZ: World coordinate Z in mm
        patientID: Patient ID (optional)
        studyDate: Study date in YYYYMMDD format (optional)
        ageAtStudyDate: Patient age at study date (optional)
        gender: Patient gender - Male or Female (optional)
        mode: Prediction mode (2D or 3D)
        
    Returns:
        LesionPredictionResponse: Prediction result for the lesion
    """
    tmp_image_path = None
    mode = "3D-PULSE"
    start_time = time.time()
    
    try:
        # Validate file format
        if not file.filename:
            raise InvalidFileFormatException("File name is missing")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ['.mha', '.mhd']:
            raise InvalidFileFormatException(
                f"Invalid file format: {file_extension}. Only .mha and .mhd formats are supported"
            )
        
        # Validate seriesInstanceUID
        if not seriesInstanceUID or not seriesInstanceUID.strip():
            raise InvalidFileFormatException("seriesInstanceUID is required and cannot be empty")
        
        # Save uploaded image file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            image_content = await file.read()
            tmp_file.write(image_content)
            tmp_image_path = tmp_file.name
        
        # Prepare coordinates and clinical information
        coords = [[coordX, coordY, coordZ]]
        annotation_ids = [str(lesionID)]
        
        clinical_info = {}
        if ageAtStudyDate is not None:
            clinical_info['age'] = ageAtStudyDate
        if gender:
            clinical_info['gender'] = gender
        if patientID:
            clinical_info['patientID'] = patientID
        if studyDate:
            clinical_info['studyDate'] = studyDate
        
        # Initialize repository and service
        repository = PredictRepository()
        service = PredictService(repository=repository)
        
        # Make prediction
        result = service.predict(
            image_path=tmp_image_path,
            coords=coords,
            annotation_ids=annotation_ids,
            clinical_info=clinical_info if clinical_info else None,
            mode=mode
        )
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Get prediction result
        if not result.predictions or len(result.predictions) == 0:
            raise ProcessingErrorException("No prediction result returned from model")
        
        prediction = result.predictions[0]
        probability = prediction.probability
        prediction_label = 1 if probability >= 0.5 else 0
        
        # Format response
        response_data = LesionPredictionData(
            seriesInstanceUID=seriesInstanceUID,
            lesionID=lesionID,
            probability=round(probability, 6),
            predictionLabel=prediction_label,
            processingTimeMs=processing_time_ms
        )
        
        response = LesionPredictionResponse(
            status="success",
            data=response_data
        )
        
        return response
    
    except InvalidFileFormatException:
        raise
    except ProcessingErrorException:
        raise
    except ValueError as e:
        raise ProcessingErrorException(f"Processing error: {str(e)}")
    except Exception as e:
        raise InternalServerErrorException(f"Internal server error: {str(e)}")
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
