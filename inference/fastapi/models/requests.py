from pydantic import BaseModel, conlist
from typing import Any, Dict, List

from models.base import Team, ModelId, ModelType, Hyperparameters


class SinglePredictRequest(BaseModel):
    radiant_team: Team
    dire_team: Team

class CSVPredictRequest(BaseModel):
    # Храним «сырые» байты CSV, полученные из UploadFile
    csv_data: bytes

class FitRequest(BaseModel):
    model_type: ModelType
    model_id: ModelId
    hyperparameters: Hyperparameters
