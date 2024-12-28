from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel


class ModelType(str, Enum):
    CAT_BOOST = "CatBoost"
    RIDGE_CLASSIFIER = "RidgeClassifier"


class Player(BaseModel):
    account_id: int
    hero_name: str


class Prediction(str, Enum):
    RADIANT = "Radiant"
    DIRE = "Dire"


Team = List[Player]
ModelId = str
Hyperparameters = Dict[str, Any]
PredictionProba = float
