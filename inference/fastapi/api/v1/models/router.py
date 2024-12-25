from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from models.requests import SinglePredictRequest, CSVPredictRequest, FitRequest
from models.responses import PredictResponse, PredictCsvResponse, FitResponse, ModelInfoResponse
from clients.models import ModelsClient

router = APIRouter()
models_service = ModelsClient()

@router.post("/predict", response_model=PredictResponse, summary="Прогноз исхода на основе выбора героев")
def predict(request: SinglePredictRequest):
    try:
        prediction, probability = models_service.predict_single(request)
        return PredictResponse(prediction=prediction, prediction_proba=probability)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict_csv", response_model=PredictCsvResponse, summary="Прогноз исхода по загруженному CSV")
def predict_csv(file: UploadFile = File(...)):
    try:
        csv_data = CSVPredictRequest(file=file.file.read())
        predictions, probas = models_service.predict_csv(csv_data)
        return PredictCsvResponse(
            success=True,
            predictions=predictions,
            prediction_probas=probas
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/fit", response_model=FitResponse, summary="Обучение модели")
def fit_model(request: FitRequest):
    try:
        result_message = models_service.fit_model(request)
        return FitResponse(success=True, message=result_message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/model_info", response_model=ModelInfoResponse, summary="Получить информацию о модели")
def get_model_info(model_id: str):
    try:
        model_info = models_service.get_model_info(model_id)
        return model_info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
