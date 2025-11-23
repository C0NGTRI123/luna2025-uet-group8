from fastapi import APIRouter

from .predict import predict_router as predict_v1_router

predict_router = APIRouter()
predict_router.include_router(predict_v1_router, prefix="/api/v1/predict", tags=["Predict"])

__all__ = ["predict_router"]

