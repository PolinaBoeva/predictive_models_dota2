import pandas as pd

from models.base import (
    PredictCsvResult,
    Prediction,
    SinglePredictResult,
    Match,
    PredictionProba,
)
from models.requests import PredictCsvRequest, SinglePredictRequest
from predictive_models_dota2.internal.models_database import ModelsDatabase
from predictive_models_dota2.data.extract_features import (
    DataPreprocessor,
    PredictionDataFetcher,
)


class ModelsPredictor:
    def __init__(
        self, models_database: ModelsDatabase, data_preprocessor: DataPreprocessor
    ):
        self._models_database = models_database
        self._prediction_data_fetcher = PredictionDataFetcher(data_preprocessor)

    def single_predict(self, request: SinglePredictRequest) -> SinglePredictResult:
        model = self._models_database.get_active_model()

        match_data = Match(radiant=request.radiant_team, dire=request.dire_team)
        X = self._prediction_data_fetcher.get_team_info_from_dataclass(match_data)
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]

        return SinglePredictResult(
            model_id=model.model_id,
            prediction=prediction,
            prediction_proba=prediction_proba,
        )

    def predict_csv(self, request: PredictCsvRequest) -> PredictCsvResult:
        model = self._models_database.get_active_model()

        data = pd.read_csv(request.file)
        X = self._prediction_data_fetcher.get_team_info_from_dataframe(data)
        predictions = model.predict(X)
        prediction_probas = model.predict_proba(X)

        return PredictCsvResult(
            model_id=model.model_id,
            predictions=predictions,
            prediction_probas=prediction_probas,
        )
