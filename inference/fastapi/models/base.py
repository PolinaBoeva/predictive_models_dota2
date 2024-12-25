from enum import Enum
from pydantic import BaseModel, conlist
from typing import Any, Dict, List

class ModelType(str, Enum):
    CatBoost = "CatBoost"
    RidgeClassifier = "RidgeClassifier"
    
class Player(BaseModel):
    account_id: int
    hero_name: str

class Prediction(str, Enum):
    Radiant = "Radiant"
    Dire = "Dire"

Team = List[Player]
ModelId = str
Hyperparameters = Dict[str, Any]
PredictionProba = float
