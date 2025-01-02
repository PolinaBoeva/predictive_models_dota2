from typing import Annotated

from fastapi import APIRouter, File, Query, Body

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
    logger.info(f"POST /api/v1/models/fit: {request.model_dump_json()}")
    models_service.fit_model(request)


@router.get(
    "/fit/status",
    response_model=FitStatusResponse,
    summary="Получение статуса асинхронной задачи обучения",
)
async def get_fit_status(model_id: Annotated[ModelId, Query(min_length=1)]):
    logger.info(f"GET /api/v1/models/fit/status: {model_id}")
    status, error = models_service.get_fit_status(model_id)
    logger.info(f"Status: {status}, Error: {error}")
    return FitStatusResponse(status=status, error=error)


@router.get(
    "/list",
    response_model=ModelsListResponse,
    summary="Список всех обученных моделей",
)
async def get_models_list():
    logger.info("GET /api/v1/models/list")
    models = models_service.get_models_list()
    logger.info(f"Models: {models}")
    return ModelsListResponse(models=models)


@router.put(
    "/activate",
    summary="Установка активной модели для прогноза",
)
async def activate_model(model_id: Annotated[ModelId, Query(min_length=1)]):
    logger.info(f"PUT /api/v1/models/activate: {model_id}")
    models_service.activate_model(model_id)
    logger.info(f"Model {model_id} activated.")


@router.post(
    "/predict",
    response_model=SinglePredictResponse,
    summary="Прогноз исхода на основе выбора героев",
)
async def predict(request: Annotated[SinglePredictRequest, Body()]):
    logger.info(f"POST /api/v1/models/predict: {request.model_dump_json()}")
    predict_result = models_service.single_predict(request)
    logger.info(f"Predict result: {predict_result}")
    return SinglePredictResponse(
        prediction=predict_result,
    )


@router.post(
    "/predict_csv",
    response_model=PredictCsvResponse,
    summary="Прогноз исхода на основе CSV-файла",
)
async def predict_csv(request: Annotated[PredictCsvRequest, File()]):
    logger.info(f"POST /api/v1/models/predict_csv: {request.filename}")
    predict_result = models_service.predict_csv(request)
    logger.info(f"Predict result: {predict_result}")
    return PredictCsvResponse(predictions=predict_result)


@router.get(
    "/model_info",
    response_model=ModelInfoResponse,
    summary="Получение информации о модели",
)
async def get_model_info(model_id: Annotated[ModelId, Query(min_length=1)]):
    logger.info(f"GET /api/v1/models/model_info: {model_id}")
    model_info = models_service.get_model_info(model_id)
    logger.info(f"Model info: {model_info}")
    return ModelInfoResponse(model_info=model_info)
