from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, Future
import threading

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LinearRegression

from models.base import (
    ModelId,
    ErrorMessage,
    ModelInfo,
    SinglePredictResult,
    PredictCsvResult,
)
from models.requests import (
    FitRequest,
    SinglePredictRequest,
    PredictCsvRequest,
)
from predictive_models_dota2.clients.model_predictor import ModelsPredictor
from predictive_models_dota2.clients.model_trainer import ModelTrainer
from predictive_models_dota2.clients.models_database import ModelsDatabase
from predictive_models_dota2.data.extract_features import DataPreprocessor


class ModelsClient:
    def __init__(self, train_data_path: str = "data/prepared/dota_23_24.csv"):
        self._models_database = ModelsDatabase()
        data_preprocessor = DataPreprocessor()
        self._model_trainer = ModelTrainer(
            models_database=self._models_database,
            train_data_path=train_data_path,
            data_preprocessor=data_preprocessor,
        )
        self._model_predictor = ModelsPredictor(
            models_database=self._models_database, data_preprocessor=data_preprocessor
        )

    def fit_model(self, request: FitRequest) -> ModelId:
        return self._model_trainer.start_fit(request)

    def get_fit_status(self, task_id: ModelId) -> Tuple[str, ErrorMessage | None]:
        return self._model_trainer.get_fit_status(task_id)

    def get_models_list(self) -> List[ModelId]:
        return self._models_database.get_models_list()

    def activate_model(self, request: ModelId) -> None:
        self._models_database.activate_model(request)

    def single_predict(self, request: SinglePredictRequest) -> SinglePredictResult:
        return self._model_predictor.single_predict(request)

    def predict_csv(self, request: PredictCsvRequest) -> PredictCsvResult:
        return self._model_predictor.predict_csv(request)

    def get_model_info(self, model_id: ModelId) -> ModelInfo:
        return self._models_database.get_model_info(model_id)
