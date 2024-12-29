from typing import List, Tuple

import pandas as pd

from models.base import (
    ModelId,
    ModelType,
    Prediction,
    PredictionProba,
    FitStatus,
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


class ModelsClient:
    def fit_model(self, request: FitRequest):
        # TODO: implement
        pass

    def get_fit_status(self, task_id: ModelId) -> Tuple[FitStatus, ErrorMessage | None]:
        # TODO: implement
        return FitStatus.SUCCESS, None

    def get_models_list(self) -> List[ModelId]:
        # TODO: implement
        return [ModelId("123")]

    def activate_model(self, request: ModelId) -> None:
        # TODO: implement
        return

    def single_predict(self, request: SinglePredictRequest) -> SinglePredictResult:
        # TODO: implement
        return SinglePredictResult(
            model_id=ModelId("123"),
            prediction=Prediction.RADIANT,
            prediction_proba=PredictionProba(0.5),
        )

    def predict_csv(self, request: PredictCsvRequest) -> PredictCsvResult:
        # TODO: implement
        data = pd.read_csv(request.csv_data.decode("utf-8"))
        return PredictCsvResult(
            model_id=ModelId("123"),
            predictions=[Prediction.RADIANT] * len(data),
            prediction_probas=[PredictionProba(0.5)] * len(data),
        )

    def get_model_info(self, model_id: ModelId) -> ModelInfo:
        # TODO: implement
        return ModelInfo(
            model_id=ModelId(model_id),
            model_type=ModelType.CAT_BOOST,
            feature_importances=None,
            fit_time=10.0,
            metrics=None,
        )
