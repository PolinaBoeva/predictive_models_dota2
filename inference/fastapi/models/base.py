from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field

AccountId = int
HeroName = str
ModelId = str
Hyperparameters = Dict[str, Any]
PredictionProba = float
ErrorMessage = str


class Player(BaseModel):
    account_id: AccountId
    hero_name: HeroName | None


Team = List[Player]


class Match(BaseModel):
    radiant: Team
    dire: Team


class ModelType(str, Enum):
    CAT_BOOST = "CatBoost"
    RIDGE_CLASSIFIER = "RidgeClassifier"


class Prediction(str, Enum):
    RADIANT = "Radiant"
    DIRE = "Dire"


class FitStatus(str, Enum):
    SUCCESS = "Success"
    FAILED = "Failed"
    RUNNING = "Running"


class SinglePredictResult(BaseModel):
    modelId: ModelId = Field(alias="model_id")
    prediction: Prediction
    prediction_proba: PredictionProba


class PredictCsvResult(BaseModel):
    modelId: ModelId = Field(alias="model_id")
    predictions: List[Prediction]
    prediction_probas: List[PredictionProba]


class ModelInfo(BaseModel):
    modelId: ModelId = Field(alias="model_id")
    modelType: ModelType = Field(alias="model_type")
    feature_importances: List[Dict[str, Any]] | None
    fit_time: float
    metrics: Dict[str, float] | None
