from functools import lru_cache
from typing import Tuple

import pandas as pd

import fastapi_logging
from predictive_models_dota2.data.extract_features import DataPreprocessor


logger = fastapi_logging.get_logger(__name__)


class PreparedDataset:
    def __init__(self, data_path: str):
        logger.info(f"Loading data from {data_path}")
        self.X, self.y = self._load_data(data_path)
        logger.info("Data loaded")

    def _load_data(self, path: str) -> Tuple[pd.DataFrame, pd.Series]:
        data = pd.read_csv(path)
        X = data.drop(columns=["radiant_win"])
        y = data["radiant_win"]
        return X, y

    def get_account_ids(self):
        return sorted(self.X["account_id"].unique().tolist())


class TrainDataset:
    def __init__(self, data_preprocessor: DataPreprocessor, train_data_path: str):
        """
        Инициализация объекта класса TrainDataset.

        :param path: Путь до CSV-файла с тренировочными данными.
        :param preprocessor: Объект класса DataPreprocessor для обработки данных.
        """
        self.prepared_dataset = get_prepared_dataset(train_data_path)
        self.X_train, self.y_train = self.prepared_dataset.X, self.prepared_dataset.y
        self.preprocessor = data_preprocessor
        self._apply_preprocessing()
        logger.info("Data loaded and preprocessed")

    def _apply_preprocessing(self):
        """
        Применение предобработки данных с использованием объекта DataPreprocessor.
        """
        logger.info("Applying preprocessing")
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.y_train = self.preprocessor.transform_target_train(self.y_train)
        logger.info("Preprocessing applied")

    def get_account_ids(self):
        """
        Получение списка account_id игроков.

        :return: Список account_id игроков.
        """
        return self.prepared_dataset.X["account_id"].tolist()


@lru_cache
def get_prepared_dataset(data_path: str = "data/prepared/train.csv"):
    return PreparedDataset(data_path=data_path)


@lru_cache
def get_train_dataset(train_data_path: str = "data/prepared/train.csv") -> TrainDataset:
    """
    Получение объекта класса TrainDataset.

    :return: Объект класса TrainDataset.
    """
    data_preprocessor = DataPreprocessor()
    return data_preprocessor, TrainDataset(
        data_preprocessor=data_preprocessor, train_data_path=train_data_path
    )
