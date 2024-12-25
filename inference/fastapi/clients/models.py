from models.requests import SinglePredictRequest, CSVPredictRequest, FitRequest


class ModelsClient:

    def predict_single(self, request: SinglePredictRequest):
        pass

    def predict_csv(self, request: CSVPredictRequest):
        pass

    def fit_model(self, request: FitRequest):
        pass

    def get_model_info(self, model_id: str):
        pass