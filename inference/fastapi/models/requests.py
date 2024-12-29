from pydantic import Field, BaseModel

from models.base import TaskId, Team, ModelId, ModelType, Hyperparameters


class FitRequest(BaseModel):
    modelType: ModelType = Field(alias="model_type")
    modelId: ModelId = Field(alias="model_id")
    hyperparameters: Hyperparameters


class FitStatusRequest(BaseModel):
    task_id: TaskId


class ActivateModelRequest(BaseModel):
    modelId: ModelId = Field(alias="model_id")


class SinglePredictRequest(BaseModel):
    radiant_team: Team
    dire_team: Team


class PredictCsvRequest(BaseModel):
    # Храним «сырые» байты CSV, полученные из UploadFile
    csv_data: bytes


class ModelInfoRequest(BaseModel):
    modelId: ModelId = Field(alias="model_id")
