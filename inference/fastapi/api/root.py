from fastapi import APIRouter

from models.responses import ServiceStatusResponse

router = APIRouter()


@router.get("/", response_model=ServiceStatusResponse)
async def root():
    return ServiceStatusResponse(status="App healthy")
