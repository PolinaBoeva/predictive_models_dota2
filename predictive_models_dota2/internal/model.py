from typing import List
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.base import BaseEstimator

from models.base import ModelInfo, ModelType, ModelId, Hyperparameters, Prediction, PredictionProba


class Model:
    def __init__(
        self,
        model: BaseEstimator,
        model_id: ModelId,
        model_type: ModelType,
        hyperparameters: Hyperparameters,
    ):
        self.model_id = model_id
        self.model_type = model_type
        self.hyperparameters = hyperparameters
        self.model = model
        self.fit_time = None

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X) -> List[Prediction]:
        predictions = self.model.predict(X)
        return [self._get_win_team(prediction) for prediction in predictions]

    def predict_proba(self, X) -> List:
        prediction_probas = self.model.predict_proba(X)
        result = [PredictionProba(proba[1]) for proba in prediction_probas]
        return result

    def get_info(self):
        # TODO: add feature importance
        return ModelInfo(
            model_id=self.model_id,
            model_type=self.model_type,
            feature_importances=None,
            fit_time=self.fit_time,
            metrics=None,
        )
    
    def _get_win_team(self, prediction: int) -> Prediction:
        return Prediction.RADIANT if prediction == 1 else Prediction.DIRE


class ModelsFactory:
    MODELS = {
        ModelType.CAT_BOOST: CatBoostClassifier,
        ModelType.RIDGE_CLASSIFIER: RidgeClassifier,
    }

    @staticmethod
    def create(
        model_id: ModelId, model_type: ModelType, hyperparameters: Hyperparameters
    ) -> Model:
        if model_type not in ModelsFactory.MODELS:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = ModelsFactory.MODELS[model_type]
        classifier = model_class(**hyperparameters)
        model = Model(
            model=classifier,
            model_id=model_id,
            model_type=model_type,
            hyperparameters=hyperparameters,
        )

        return model
