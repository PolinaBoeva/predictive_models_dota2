from typing import Any, Dict, List

from pydantic import ConfigDict, Field, BaseModel

from models.base import FitStatus, ModelId, ModelType, Prediction, PredictionProba


class FitResponse(BaseModel):
    task_id: str


class FitStatusResponse(BaseModel):
    status: FitStatus
    error: str | None = None


class ModelsListResponse(BaseModel):
    models: List[ModelId]


class SinglePredictResponse(BaseModel):
    modelId: ModelId = Field(alias="model_id")
    prediction: Prediction
    prediction_proba: PredictionProba


class PredictCsvResponse(BaseModel):
    modelId: ModelId = Field(alias="model_id")
    predictions: List[Prediction]
    prediction_probas: List[PredictionProba]


class ModelInfoResponse(BaseModel):
    modelId: ModelId = Field(alias="model_id")
    modelType: ModelType = Field(alias="model_type")
    feature_importances: List[Dict[str, Any]] | None
    fit_time: float
    metrics: Dict[str, float] | None


class AccountIdsListResponse(BaseModel):
    account_ids: List[int]


class ServiceStatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App healthy"}]}
    )
