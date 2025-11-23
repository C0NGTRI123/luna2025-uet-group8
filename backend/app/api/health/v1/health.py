from fastapi import APIRouter

from app.schemas.health import HealthCheckResponse
from app.service.health_service import HealthService
from app.repository.health_repository import HealthRepository


health_router = APIRouter()


@health_router.get(
    "/",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check the service status",
)
async def health_check() -> HealthCheckResponse:
    """
    Service health check endpoint

    Returns:
        HealthCheckResponse: Health check result
    """
    repository = HealthRepository()
    service = HealthService(repository=repository)
    health = service.get_health()
    return HealthCheckResponse(**health.model_dump())


@health_router.get("/ping", summary="Ping", description="Simple connectivity check endpoint")
async def ping() -> dict:
    """
    Simple connectivity check endpoint

    Returns:
        dict: pong response
    """
    return {"message": "pong"}