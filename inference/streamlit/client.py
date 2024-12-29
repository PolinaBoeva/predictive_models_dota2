import requests

class ModelsAPIClient:
    def __init__(self, host: str, port: int):
        self.base_url = f"{host}:{port}/api/v1/models"

    def fit_model(self, params: dict):
        """Отправка параметров для обучения модели."""
        response = requests.post(f"{self.base_url}/fit", json=params)
        return response.json()

    def get_fit_status(self):
        """Получение статуса асинхронной задачи обучения."""
        response = requests.get(f"{self.base_url}/fit/status")
        return response.json()

    def get_model_list(self):
        """Получение списка всех обученных моделей."""
        response = requests.get(f"{self.base_url}/list")
        return response.json()

    def activate_model(self, model_id: str):
        """Установка активной модели для прогноза."""
        response = requests.put(f"{self.base_url}/activate", json={"model_id": model_id})
        return response.json()

    def predict(self, data: dict):
        """Прогноз исхода на основе выбранных данных. Используется активированная модель."""
        response = requests.post(f"{self.base_url}/predict", json=data)
        return response.json()

    def predict_csv(self, csv_data):
        """Прогноз исхода на основе CSV-файла."""
        response = requests.post(f"{self.base_url}/predict_csv", files={"file": csv_data})
        return response.json()

    def get_model_info(self, model_id: str):
        """Получение информации об обученной модели."""
        response = requests.get(f"{self.base_url}/model_info", params={"model_id": model_id})
        return response.json()


class DataAPIClient:
    def __init__(self, host: str, port: int):
        self.base_url = f"{host}:{port}/api/v1/data"

    def get_account_ids(self):
        """Получение уникальных account_id из API."""
        response = requests.get(f"{self.base_url}/account_ids")
        if response.status_code == 200:
            return response.json()  # Предполагается, что API возвращает список account_ids
        else:
            raise Exception("Не удалось получить Account IDs из API.")

