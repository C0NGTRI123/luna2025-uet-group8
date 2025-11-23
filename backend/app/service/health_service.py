from datetime import datetime

from pydantic import BaseModel

from app.repository.health_repository import HealthRepository
from app.schemas.health import HealthStatus


class HealthService:
    """Service for health check operations."""

    def __init__(self, repository: HealthRepository):
        self._repository = repository

    def get_health(self) -> HealthStatus:
        status = self._repository.get_status()
        timestamp = self._repository.get_current_timestamp()
        service, version = self._repository.get_service_info()
        return HealthStatus(
            status=status,
            timestamp=timestamp,
            service=service,
            version=version,
        )