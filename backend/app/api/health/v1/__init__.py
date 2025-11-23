from fastapi import APIRouter

from .health import health_router as health_v1_router

health_router = APIRouter()
health_router.include_router(health_v1_router, prefix="/api/v1/health", tags=["Health"])

__all__ = ["health_router"]