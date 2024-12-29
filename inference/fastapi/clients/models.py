from models.requests import SinglePredictRequest, PredictCsvRequest, FitRequest


class ModelsClient:

    def predict_single(self, request: SinglePredictRequest):
        pass

    def predict_csv(self, request: PredictCsvRequest):
        pass

    def fit_model(self, request: FitRequest):
        pass

    def get_model_info(self, model_id: str):
        pass
