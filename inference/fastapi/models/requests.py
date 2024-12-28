from pydantic import Field, BaseModel

from models.base import Team, ModelId, ModelType, Hyperparameters


class SinglePredictRequest(BaseModel):
    radiant_team: Team
    dire_team: Team


class CSVPredictRequest(BaseModel):
    # Храним «сырые» байты CSV, полученные из UploadFile
    csv_data: bytes


class FitRequest(BaseModel):
    modelType: ModelType = Field(alias="model_type")
    modelId: ModelId = Field(alias="model_id")
    hyperparameters: Hyperparameters
