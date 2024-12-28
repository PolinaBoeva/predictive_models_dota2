from typing import Any, Dict, List

from pydantic import Field, BaseModel

from models.base import ModelId, ModelType, Prediction, PredictionProba


class PredictResponse(BaseModel):
    prediction: Prediction
    prediction_proba: PredictionProba


class PredictCsvResponse(BaseModel):
    predictions: List[Prediction]
    prediction_probas: List[PredictionProba]


class FitResponse(BaseModel):
    success: bool
    message: str


class ModelInfoResponse(BaseModel):
    modelId: ModelId = Field(alias="model_id")
    modelType: ModelType = Field(alias="model_type")
    feature_importances: List[Dict[str, Any]]
    fit_time: float
    metrics: Dict[str, float]
