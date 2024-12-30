import pandas as pd

from models.base import (
    PredictCsvResult,
    Prediction,
    SinglePredictResult,
    Match,
    PredictionProba,
)
from models.requests import PredictCsvRequest, SinglePredictRequest
from predictive_models_dota2.clients.models_database import ModelsDatabase
from predictive_models_dota2.data.extract_features import (
    DataPreprocessor,
    PredictionDataFetcher,
)


class ModelsPredictor:
    def __init__(
        self, models_database: ModelsDatabase, data_preprocessor: DataPreprocessor
    ):
        self._models_database = models_database
        self._data_preprocessor = data_preprocessor
        self._prediction_data_fetcher = PredictionDataFetcher(data_preprocessor)

    def single_predict(self, request: SinglePredictRequest) -> SinglePredictResult:
        model = self._models_database.get_active_model()

        match_data = Match(radiant=request.radiant_team, dire=request.dire_team)
        X = self._prediction_data_fetcher.get_team_info_from_dataclass(match_data)
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0, 1]

        return SinglePredictResult(
            model_id=model.model_id,
            prediction=self._get_win_team(prediction),
            prediction_proba=PredictionProba(prediction_proba),
        )

    def predict_csv(self, request: PredictCsvRequest) -> PredictCsvResult:
        model = self._models_database.get_active_model()

        data = pd.read_csv(request.csv_data.decode("utf-8"))
        X = self._prediction_data_fetcher.get_team_info_from_dataframe(data)
        predictions = model.predict(X)
        prediction_probas = model.predict_proba(X)

        return PredictCsvResult(
            model_id=model.model_id,
            predictions=[self._get_win_team(prediction) for prediction in predictions],
            prediction_probas=[PredictionProba(proba)[:, 1] for proba in prediction_probas],
        )

    def _get_win_team(self, prediction: int) -> str:
        return Prediction.RADIANT if prediction == 1 else Prediction.DIRE
