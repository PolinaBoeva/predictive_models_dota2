from typing import List

from pydantic import ConfigDict, BaseModel, Field

from models.base import (
    FitStatus,
    ModelId,
    ModelInfo,
    SinglePredictResult,
    PredictCsvResult,
)


class FitStatusResponse(BaseModel):
    status: FitStatus
    error: str | None = None


class ModelsListResponse(BaseModel):
    models: List[ModelId]


class SinglePredictResponse(BaseModel):
    prediction: SinglePredictResult


class PredictCsvResponse(BaseModel):
    predictions: PredictCsvResult


class ModelInfoResponse(BaseModel):
    modelInfo: ModelInfo = Field(alias="model_info")


class AccountIdsListResponse(BaseModel):
    account_ids: List[int]


class ServiceStatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App healthy"}]}
    )
