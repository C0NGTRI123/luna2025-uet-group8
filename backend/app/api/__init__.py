from fastapi import APIRouter

from app.api.health.v1 import health_router
from app.api.predict.v1 import predict_router

router = APIRouter()
router.include_router(health_router)
router.include_router(predict_router)

__all__ = ["router"]