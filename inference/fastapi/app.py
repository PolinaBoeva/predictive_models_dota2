import logging

import uvicorn
from fastapi import FastAPI

from config import get_config
from api.v1.data import routes as data_routes
from api.v1.models import routes as models_routes
from api import root as root_routes


logging.basicConfig(
    level=get_config().log_config.log_level,
    format=get_config().log_config.log_format,
    datefmt=get_config().log_config.log_datefmt,
)

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

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=get_config().fastapi_config.host,
        port=get_config().fastapi_config.port,
        reload=True,
    )
