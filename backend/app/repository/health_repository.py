from datetime import datetime

from app.core.config import config


class HealthRepository:
    """Repository for health check data retrieval."""

    def get_status(self) -> str:
        return "healthy"

    def get_current_timestamp(self) -> datetime:
        return datetime.now()

    def get_service_info(self) -> tuple[str, str]:
        return config.APP_NAME, config.APP_VERSION