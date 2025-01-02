from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, Future
import threading

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.base import BaseEstimator

from models.base import (
    ModelId,
    FitStatus,
    ErrorMessage,
    SinglePredictResult,
    PredictCsvResult,
    Hyperparameters,
)
from models.requests import (
    FitRequest,
    SinglePredictRequest,
    PredictCsvRequest,
)
from predictive_models_dota2.internal.model import Model, ModelsFactory
from predictive_models_dota2.internal.model_predictor import ModelsPredictor
from predictive_models_dota2.internal.models_database import ModelsDatabase
from predictive_models_dota2.data.datasets import TrainDataset
from predictive_models_dota2.data.extract_features import (
    DataPreprocessor,
    PredictionDataFetcher,
)


class ModelTrainer:
    def __init__(
        self,
        models_database: ModelsDatabase,
        data_preprocessor: DataPreprocessor,
        train_dataset: TrainDataset,
    ):
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._tasks: Dict[str, Future] = {}

        self._models_database = models_database
        self._train_dataset = train_dataset

    def start_fit(self, request: FitRequest) -> ModelId:
        model_id = request.modelId

        future = self._executor.submit(self._train_model, request)

        self._tasks[model_id] = future
        return model_id

    def _train_model(self, request: FitRequest):
        import time

        start_time = time.time()

        model_id = request.modelId
        model_type = request.modelType
        hyperparameters = request.hyperparameters

        model = ModelsFactory.create(
            model_id=model_id, model_type=model_type, hyperparameters=hyperparameters
        )

        X_train = self._train_dataset.X_train
        y_train = self._train_dataset.y_train

        model.fit(X_train, y_train)

        fit_time = time.time() - start_time
        model.fit_time = fit_time
        return model

    def get_fit_status(self, model_id: ModelId) -> Tuple[str, ErrorMessage | None]:
        future = self._tasks.get(model_id)

        if not future:
            raise ValueError("Model not found") # TODO: сделать кастомную ошибку

        if future.running():
            return FitStatus.RUNNING, None

        if future.done():
            try:
                model = future.result()
                self._models_database.add_model(model)
                return FitStatus.SUCCESS, None
            except Exception as e:
                return FitStatus.FAILED, str(e)

        return FitStatus.FAILED, "Unknown status"
