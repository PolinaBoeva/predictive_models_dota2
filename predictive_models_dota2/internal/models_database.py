from typing import Dict, List

from models.base import ModelId, ModelInfo
from predictive_models_dota2.internal.model import Model


class ModelsDatabase:
    def __init__(self):
        self._models: Dict[str, Model] = {}
        self.active_model_id = None

    def add_model(self, model: Model):
        self._models[model.model_id] = model

    def get_model(self, model_id: ModelId) -> Model:
        if model_id not in self._models:
            raise ValueError(f"Model with id {model_id} not found") # TODO: сделать кастомную ошибку
        return self._models[model_id]

    def get_model_info(self, model_id: ModelId) -> ModelInfo:
        model = self.get_model(model_id)
        return model.get_info()

    def get_models_list(self) -> List[ModelId]:
        return [ModelId(model_id) for model_id in self._models]

    def activate_model(self, model_id: ModelId):
        # TODO: честная актиавция модели
        if model_id not in self._models:
            raise ValueError(f"Model with id {model_id} not found") # TODO: сделать кастомную ошибку
        self.active_model_id = model_id

    def get_active_model(self) -> Model:
        if not self.active_model_id:
            raise ValueError("No active model") # TODO: сделать кастомную ошибку
        return self._models[self.active_model_id]
