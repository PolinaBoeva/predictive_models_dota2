from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Body

from models.base import ModelId
from models.requests import SinglePredictRequest, PredictCsvRequest, FitRequest
from models.responses import (
    ModelsListResponse,
    SinglePredictResponse,
    PredictCsvResponse,
    ModelInfoResponse,
    FitStatusResponse,
)
from clients.models import ModelsClient


router = APIRouter()
models_service = ModelsClient()


@router.post(
    "/fit",
    summary="Запуск асинхронного обучения модели",
    status_code=202,
)
async def post_fit(request: Annotated[FitRequest, Body()]):
    models_service.fit_model(request)


@router.get(
    "/fit/status",
    response_model=FitStatusResponse,
    summary="Получение статуса асинхронной задачи обучения",
)
async def get_fit_status(model_id: Annotated[ModelId, Query(min_length=1)]):
    status, error = models_service.get_fit_status(model_id)
    return FitStatusResponse(status=status, error=error)


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
    models_service.activate_model(model_id)


@router.post(
    "/predict",
    response_model=SinglePredictResponse,
    summary="Прогноз исхода на основе выбора героев",
)
async def predict(request: Annotated[SinglePredictRequest, Body()]):
    try:
        predict_result = models_service.single_predict(request)
        return SinglePredictResponse(
            prediction=predict_result,
        )
    except Exception as e:  # TODO: уточнить тип ошибки
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/predict_csv",
    response_model=PredictCsvResponse,
    summary="Прогноз исхода на основе CSV-файла",
)
async def predict_csv(request: Annotated[PredictCsvRequest, File()]):
    try:
        predict_result = models_service.predict_csv(request)
        return PredictCsvResponse(predictions=predict_result)
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
        return ModelInfoResponse(model_info=model_info)
    except Exception as e:  # TODO: уточнить тип ошибки
        raise HTTPException(status_code=400, detail=str(e))
