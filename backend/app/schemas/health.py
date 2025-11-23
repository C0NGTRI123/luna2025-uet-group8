from datetime import datetime

from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    """Schema for health check response."""

    status: str
    timestamp: datetime
    service: str
    version: str


class HealthStatus(BaseModel):
    """Schema for health status information."""


    status: str
    timestamp: datetime
    service: str
    version: str