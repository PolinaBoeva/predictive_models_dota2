from fastapi import UploadFile
from pydantic import Field, BaseModel

from models.base import Team, ModelId, ModelType, Hyperparameters


class FitRequest(BaseModel):
    modelType: ModelType = Field(alias="model_type")
    modelId: ModelId = Field(alias="model_id")
    hyperparameters: Hyperparameters


class FitStatusRequest(BaseModel):
    modelId: ModelId = Field(alias="model_id")


class ActivateModelRequest(BaseModel):
    modelId: ModelId = Field(alias="model_id")


class SinglePredictRequest(BaseModel):
    radiant_team: Team
    dire_team: Team


PredictCsvRequest = UploadFile


class ModelInfoRequest(BaseModel):
    modelId: ModelId = Field(alias="model_id")
