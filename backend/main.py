import uvicorn

from app.core.config import config

if __name__ == "__main__":
    uvicorn.run(
        "app.app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower(),
    )