import requests
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelsAPIClient:
    def __init__(self, host: str, port: int):
        self.base_url = f"{host}:{port}/api/v1/models"

    def fit_model(self, params: dict):
        """Отправка параметров для обучения модели."""
        logger.info("Отправка параметров для обучения модели.")
        response = requests.post(f"{self.base_url}/fit", json=params)
        logger.info(f"Ответ от сервера: {response.status_code}, {response.json()}")
        return response.json()

    def get_fit_status(self):
        """Получение статуса асинхронной задачи обучения."""
        logger.info("Получение статуса обучения модели.")
        response = requests.get(f"{self.base_url}/fit/status")
        logger.info(f"Ответ от сервера: {response.status_code}, {response.json()}")
        return response.json()

    def get_model_list(self):
        """Получение списка всех обученных моделей."""
        logger.info("Запрос списка всех обученных моделей.")
        response = requests.get(f"{self.base_url}/list")
        logger.info(f"Ответ от сервера: {response.status_code}, {response.json()}")
        return response.json()

    def activate_model(self, model_id: str):
        """Установка активной модели для прогноза."""
        logger.info(f"Активация модели с ID: {model_id}.")
        response = requests.put(f"{self.base_url}/activate", json={"model_id": model_id})
        logger.info(f"Ответ от сервера: {response.status_code}, {response.json()}")
        return response.json()

    def predict(self, data: dict):
        """Прогноз исхода на основе выбранных данных. Используется активированная модель."""
        logger.info("Отправка данных для прогноза.")
        response = requests.post(f"{self.base_url}/predict", json=data)
        logger.info(f"Ответ от сервера: {response.status_code}, {response.json()}")
        return response.json()

    def predict_csv(self, csv_data):
        """Прогноз исхода на основе CSV-файла."""
        logger.info("Отправка CSV-файла для прогноза.")
        response = requests.post(f"{self.base_url}/predict_csv", files={"file": csv_data})
        logger.info(f"Ответ от сервера: {response.status_code}, {response.json()}")
        return response.json()

    def get_model_info(self, model_id: str):
        """Получение информации об обученной модели."""
        logger.info(f"Запрос информации о модели с ID: {model_id}.")
        response = requests.get(f"{self.base_url}/model_info", params={"model_id": model_id})
        logger.info(f"Ответ от сервера: {response.status_code}, {response.json()}")
        return response.json()


class DataAPIClient:
    def __init__(self, host: str, port: int):
        self.base_url = f"{host}:{port}/api/v1/data"

    def get_account_ids(self):
        """Получение уникальных account_id из API."""
        logger.info("Запрос уникальных account_id из API.")
        response = requests.get(f"{self.base_url}/account_ids")
        if response.status_code == 200:
            logger.info("Account IDs успешно получены.")
            return response.json()  # Предполагается, что API возвращает список account_ids
        else:
            logger.error("Не удалось получить Account IDs из API.")
            raise Exception("Не удалось получить Account IDs из API.")

