from typing import Annotated

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Body

from models.base import FitStatus, TaskId, ModelId
from models.requests import SinglePredictRequest, PredictCsvRequest, FitRequest
from models.responses import (
    ModelsListResponse,
    SinglePredictResponse,
    PredictCsvResponse,
    FitResponse,
    ModelInfoResponse,
    FitStatusResponse,
)
from clients.models import ModelsClient

router = APIRouter()
models_service = ModelsClient()


@router.post(
    "/fit",
    response_model=FitResponse,
    summary="Запуск асинхронного обучения модели",
    status_code=202,
)
async def post_fit(request: Annotated[FitRequest, Body()]):
    try:
        task_id = models_service.fit_model(request)
        return FitResponse(task_id=task_id)
    except Exception as e:  # TODO: уточнить тип ошибки
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/fit/status",
    response_model=FitStatusResponse,
    summary="Получение статуса асинхронной задачи обучения",
)
async def get_fit_status(task_id: Annotated[TaskId, Query(min_length=1)]):
    try:
        status = models_service.get_fit_status(task_id)
        return FitStatusResponse(status=status)
    except Exception as e:  # TODO: уточнить тип ошибки
        return FitStatusResponse(status=FitStatus.FAILED, error=str(e))


@router.get(
    "/list",
    response_model=ModelsListResponse,
    summary="Список всех обученных моделей",
)
async def get_models_list():
    models = models_service.get_models_list()
    return ModelsListResponse(models=models)


@router.put(
    "/activate",
    summary="Установка активной модели для прогноза",
)
async def activate_model(model_id: Annotated[ModelId, Query(min_length=1)]):
    models_service.activate_model(request)


@router.post(
    "/predict",
    response_model=SinglePredictResponse,
    summary="Прогноз исхода на основе выбора героев",
)
def predict(request: Annotated[SinglePredictRequest, Body()]):
    try:
        model_id, prediction, probability = models_service.predict_single(request)
        return SinglePredictResponse(
            model_id=model_id, prediction=prediction, prediction_proba=probability
        )
    except Exception as e:  # TODO: уточнить тип ошибки
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/predict_csv",
    response_model=PredictCsvResponse,
    summary="Прогноз исхода на основе CSV-файла",
)
async def predict_csv(file: Annotated[UploadFile, File()]):
    try:
        model_id, predictions, probabilities = models_service.predict_csv(file)
        return PredictCsvResponse(
            model_id=model_id, predictions=predictions, prediction_probas=probabilities
        )
    except Exception as e:  # TODO: уточнить тип ошибки
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/model_info",
    response_model=ModelInfoResponse,
    summary="Получение информации о модели",
)
async def get_model_info(model_id: Annotated[ModelId, Query(min_length=1)]):
    try:
        model_info = models_service.get_model_info(model_id)
        return model_info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
