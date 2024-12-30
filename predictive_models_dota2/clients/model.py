from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.base import BaseEstimator

from models.base import ModelInfo, ModelType, ModelId, Hyperparameters


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
        return self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_info(self):
        # TODO: add feature importance
        return ModelInfo(
            model_id=self.model_id,
            model_type=self.model_type,
            feature_importances=None,
            fit_time=self.fit_time,
            metrics=None,
        )


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
