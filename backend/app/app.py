from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import router
from app.core.config import config
from app.core.exceptions import CustomException


def init_cors(app: FastAPI) -> None:
    """Initialize CORS settings"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600,
    )


def init_routers(app: FastAPI) -> None:
    """Initialize routers"""
    app.include_router(router)


def init_listeners(app: FastAPI) -> None:
    """Initialize exception handlers"""

    @app.exception_handler(CustomException)
    async def custom_exception_handler(request: Request, exc: CustomException):
        return JSONResponse(
            status_code=exc.code,
            content={"error_code": exc.error_code, "message": exc.message},
        )


def create_app() -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title="LUNA25 Predict API",
        description="API for LUNA25 Predict Service",
        version="1.0.0",
        docs_url=None if config.ENV == "production" else "/docs",
        redoc_url=None if config.ENV == "production" else "/redoc",
    )
    # CORS needs to be set first
    init_cors(app=app)
    # Add custom middleware
    init_routers(app=app)
    init_listeners(app=app)

    return app


app = create_app()