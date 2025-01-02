from fastapi.responses import JSONResponse
import uvicorn
from fastapi import FastAPI, Request

from config import get_config
import fastapi_logging

from api.v1.data import routes as data_routes
from api.v1.models import routes as models_routes
from api import root as root_routes

logger = fastapi_logging.get_logger(__name__)

app = FastAPI(
    title=get_config().fastapi_config.title,
    version=get_config().fastapi_config.version,
    description=get_config().fastapi_config.description,
    docs_url=get_config().fastapi_config.docs_url,
    openapi_url=get_config().fastapi_config.openapi_url,
)

app.include_router(root_routes.router)
app.include_router(models_routes.router, prefix="/api/v1/models", tags=["models"])
app.include_router(data_routes.router, prefix="/api/v1/data", tags=["data"])


@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    logger.error(f"Error occurred: {exc}")
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )


if __name__ == "__main__":
    logger.info("Starting FastAPI Server...")
    uvicorn.run(
        "app:app",
        host=get_config().fastapi_config.host,
        port=get_config().fastapi_config.port,
        reload=True,
    )
    logger.info("FastAPI Server stopped.")
