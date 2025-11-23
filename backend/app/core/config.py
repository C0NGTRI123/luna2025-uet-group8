import os


class Configuration:
    """Application configuration"""

    def __init__(self) -> None:
        # Environment settings
        self.ENV: str = os.environ.get("ENV", "development")
        self.DEBUG: bool = os.environ.get("DEBUG", "true").lower() == "true"

        # Application settings
        self.APP_NAME: str = os.environ.get("APP_NAME", "LUNA25 Predict API")
        self.APP_VERSION: str = os.environ.get("APP_VERSION", "1.0.0")

        # API settings
        self.API_PREFIX: str = os.environ.get("API_PREFIX", "/api")

        # Server settings
        self.HOST: str = os.environ.get("HOST", "0.0.0.0")
        self.PORT: int = int(os.environ.get("PORT", "8000"))

        # CORS settings
        self.CORS_ORIGINS: str = os.environ.get("CORS_ORIGINS", "*")

        # Log level
        self.LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

        # Model settings
        self.MODEL_PATH_2D: str = os.environ.get(
            "MODEL_PATH_2D", 
            "results/LUNA25-baseline-2D-20250225"
        )
        self.MODEL_PATH_3D: str = os.environ.get(
            "MODEL_PATH_3D", 
            "results/LUNA25-baseline-3D-20250225"
        )
        self.DEFAULT_PREDICTION_MODE: str = os.environ.get(
            "DEFAULT_PREDICTION_MODE", "2D"
        )


config = Configuration()