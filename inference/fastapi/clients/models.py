from typing import List

from models.base import (
    ModelId,
    TaskId,
    FitStatus,
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
    def fit_model(self, request: FitRequest) -> TaskId:
        pass

    def get_fit_status(self, task_id: TaskId) -> FitStatus:
        pass

    def get_models_list(self) -> List[ModelId]:
        pass

    def activate_model(self, request: ModelId) -> None:
        pass

    def single_predict(self, request: SinglePredictRequest) -> SinglePredictResult:
        pass

    def predict_csv(self, request: PredictCsvRequest) -> PredictCsvResult:
        pass

    def get_model_info(self, model_id: ModelId) -> ModelInfo:
        pass
