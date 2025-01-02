import streamlit as st
import pandas as pd
from client import ModelsAPIClient, DataAPIClient
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_model(models_api_client, data_api_client):
    """Функция для получения предсказаний."""
    st.header("Предсказания на основе обученной модели")
    logger.info("Инициализация процесса получения предсказаний.")

    # Получение account IDs
    try:
        account_ids_response = (
            data_api_client.get_account_ids()
        )  # Обратите внимание на использование data_api_client
        if account_ids_response:
            account_ids = account_ids_response["account_ids"]
            logger.info(f"Account IDs успешно получены: {account_ids}.")
        else:
            st.error("Не удалось загрузить account IDs.")
            logger.error("Ответ пустой при получении Account IDs.")
            st.stop()
    except Exception as e:
        st.error(f"Возникла ошибка при получении Account IDs: {str(e)}.")
        logger.error(f"Ошибка при получении Account IDs: {str(e)}.")
        st.stop()

    # Выбор для команды Radiant
    st.subheader("Выбор аккаунтов для команды Radiant")
    radiant_selected_ids = st.multiselect(
        "Выберите до 5 Account IDs для Radiant", account_ids, max_selections=5
    )

    # Выбор для команды Dire
    st.subheader("Выбор аккаунтов для команды Dire")
    dire_selected_ids = st.multiselect(
        "Выберите до 5 Account IDs для Dire", account_ids, max_selections=5
    )

    if st.button("🔮 Получить предсказания"):
        if len(radiant_selected_ids) != 5 or len(dire_selected_ids) != 5:
            st.error("Пожалуйста, выберите ровно 5 Account IDs для каждой команды.")
            logger.warning(
                "Необходимо выбрать ровно 5 Account IDs для одной или обеих команд."
            )
            return

        data = {
            "radiant_team": [
                {"account_id": int(account_id), "hero_name": "Pudge"}
                for account_id in radiant_selected_ids
            ],
            "dire_team": [
                {"account_id": int(account_id), "hero_name": "Pudge"}
                for account_id in dire_selected_ids
            ],
        }

        try:
            prediction_response = models_api_client.predict(data)
            st.json(prediction_response)
            logger.info("Получено предсказание для команд Radiant и Dire.")
        except Exception as e:
            st.error(f"Ошибка при получении предсказания: {str(e)}.")
            logger.error(f"Ошибка при получении предсказания: {str(e)}.")
