from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Body

import fastapi_logging
from models.base import ModelId
from models.requests import SinglePredictRequest, PredictCsvRequest, FitRequest
from models.responses import (
    ModelsListResponse,
    SinglePredictResponse,
    PredictCsvResponse,
    ModelInfoResponse,
    FitStatusResponse,
)
from services.models import ModelsService


logger = fastapi_logging.get_logger(__name__)


router = APIRouter()
models_service = ModelsService()


@router.post(
    "/fit",
    summary="Запуск асинхронного обучения модели",
    status_code=202,
)
async def post_fit(request: Annotated[FitRequest, Body()]):
    logger.info(f"Запуск асинхронного обучения модели.: {request.model_dump_json()}")
    models_service.fit_model(request)


@router.get(
    "/fit/status",
    response_model=FitStatusResponse,
    summary="Получение статуса асинхронной задачи обучения",
)
async def get_fit_status(model_id: Annotated[ModelId, Query(min_length=1)]):
    logger.info(f"Получение статуса асинхронной задачи обучения: {model_id}")
    status, error = models_service.get_fit_status(model_id)
    logger.info(f"Статус: {status}, Ошибка: {error}")
    return FitStatusResponse(status=status, error=error)


@router.get(
    "/list",
    response_model=ModelsListResponse,
    summary="Список всех обученных моделей",
)
async def get_models_list():
    logger.info("Запрос списка всех обученных моделей.")
    models = models_service.get_models_list()
    logger.info(f"Список моделей: {models}")
    return ModelsListResponse(models=models)


@router.put(
    "/activate",
    summary="Установка активной модели для прогноза",
)
async def activate_model(model_id: Annotated[ModelId, Query(min_length=1)]):
    logger.info(f"Активация модели для прогноза: {model_id}")
    models_service.activate_model(model_id)
    logger.info(f"Модель активирована: {model_id}")


@router.post(
    "/predict",
    response_model=SinglePredictResponse,
    summary="Прогноз исхода на основе выбора героев",
)
async def predict(request: Annotated[SinglePredictRequest, Body()]):
    logger.info(f"Прогноз исхода на основе выбора героев: {request.model_dump_json()}")
    predict_result = models_service.single_predict(request)
    logger.info(f"Результат прогноза: {predict_result.model_dump_json()}")
    return SinglePredictResponse(
        prediction=predict_result,
    )


@router.post(
    "/predict_csv",
    response_model=PredictCsvResponse,
    summary="Прогноз исхода на основе CSV-файла",
)
async def predict_csv(request: Annotated[PredictCsvRequest, File()]):
    predict_result = models_service.predict_csv(request)
    return PredictCsvResponse(predictions=predict_result)
    


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
