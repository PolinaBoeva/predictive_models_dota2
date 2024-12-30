import streamlit as st
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelEncoder

from client import ModelsAPIClient, DataAPIClient
from utils import get_ridge_params, get_catboost_params
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fit_model(api_client):
    """Функция для обучения модели."""
    st.header("Обучение модели")
    logger.info("Инициализация процесса обучения модели.")

    type_of_model = st.selectbox("Выберите модель", ["⚖️ Ridge Classifier", "🧠 CatBoost Classifier"])
    params = {"type_of_model": type_of_model}
    logger.info(f"Выбрана модель: {type_of_model}")

    st.subheader("Гиперпараметры модели")

    # Выбор гиперпараметров в зависимости от типа модели
    if type_of_model == "⚖️ Ridge Classifier":
        get_ridge_params(params)
        logger.info(f"Параметры Ridge Classifier: {params}")
    elif type_of_model == "🧠 CatBoost Classifier":
        get_catboost_params(params)
        logger.info(f"Параметры CatBoost Classifier: {params}")

    params["model_id"] = st.text_input("Введите ID модели", value="model")

    if st.button("🚀 Обучить модель"):
        # Запуск обучения модели
        logger.info("Запуск обучения модели.")
        start_time = time.time()
        fit_response = api_client.fit_model(params)
        st.success("Обучение модели начато!")

        # Проверка статуса обучения
        while True:
            status_response = api_client.get_fit_status()
            st.write(f"Статус обучения: {status_response['status']}")
            logger.info(f"Текущий статус обучения: {status_response['status']}")
            if status_response['status'] in ["completed", "failed"]:
                break
            time.sleep(2)  # Задержка перед следующей проверкой

        end_time = time.time()
        st.write(f"⏳ Время обучения составило: {end_time - start_time:.2f} сек")
        logger.info(f"Обучение модели завершено. Время обучения: {end_time - start_time:.2f} секунд.")

