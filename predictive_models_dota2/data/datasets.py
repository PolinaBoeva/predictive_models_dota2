from typing import Tuple
import logging

import pandas as pd

from predictive_models_dota2.data.extract_features import DataPreprocessor

logger = logging.getLogger(__name__)


class TrainDataset:
    def __init__(self, data_preprocessor: DataPreprocessor, train_data_path: str):
        """
        Инициализация объекта класса TrainDataset.

        :param path: Путь до CSV-файла с тренировочными данными.
        :param preprocessor: Объект класса DataPreprocessor для обработки данных.
        """
        logger.info(f"Loading data from {train_data_path}")
        self.X_train, self.y_train = self._load_data(train_data_path)
        self.preprocessor = data_preprocessor
        self._apply_preprocessing()
        logger.info("Data loaded and preprocessed")

    def _load_data(self, path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Загрузка данных из CSV-файла и разделение на признаки и целевую переменную.

        :param path: Путь до CSV-файла.
        :return: Кортеж с X_train (признаки) и y_train (целевая переменная).
        """
        data = pd.read_csv(path)
        X = data.drop(columns=["radiant_win"])
        y = data["radiant_win"]
        return X, y

    def _apply_preprocessing(self):
        """
        Применение предобработки данных с использованием объекта DataPreprocessor.
        """
        logger.info("Applying preprocessing")
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.y_train = self.preprocessor.transform_target_train(self.y_train)
        logger.info("Preprocessing applied")
