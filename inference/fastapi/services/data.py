from typing import List
from predictive_models_dota2.data.datasets import get_prepared_dataset


class DataService:
    def __init__(self, train_data_path: str = "data/prepared/train.csv"):
        self.dataset = get_prepared_dataset(train_data_path)

    def get_account_ids(self) -> List[int]:
        return self.dataset.get_account_ids()
