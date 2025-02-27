import pandas as pd
from typing import Optional, Tuple, List, Iterable

from models.base import Match


@dataclass
class Player:
    account_id: int
    hero_name: str = None


@dataclass
class Match:
    dire: List[Player]
    radiant: List[Player]


class DataCleaner:

    def fill_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Метод для очистки данных в DataFrame.

        Args:
            df (pd.DataFrame): Исходный DataFrame с данными матчей.

        Returns:
            pd.DataFrame: Очищенный DataFrame.
        """

        df = self._remove_invalid_rows(df)
        df = self._fill_zero_values(df)
        df["actions_per_min"] = pd.to_numeric(df["actions_per_min"],
                                              errors="coerce")
        df["total_xp"] = pd.to_numeric(df["total_xp"], errors="coerce")
        df = self._remove_invalid_matches(df)
        df = df[df["game_mode"] == 2]
        return df

    def _remove_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Удаление строк с пустыми значениями в 'account_id' и 'win'.

        Args:
            df (pd.DataFrame): DataFrame, из которого будут удалены строки с пустыми значениями.

        Returns:
            pd.DataFrame: Очищенный DataFrame.
        """

        invalid_match_ids = set(df[df["account_id"].isna()].index.unique())
        df = df.loc[~df.index.isin(invalid_match_ids)]
        df = df[df["win"].notna()]
        return df

    def _fill_zero_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Заполнение NaN значений нулями в специфичных столбцах.

        Args:
            df (pd.DataFrame): DataFrame, в котором будут заполнены NaN значения.

        Returns:
            pd.DataFrame: DataFrame с заполненными значениями.
        """

        cols_to_fill_zero = [
            "hero_kills", "courier_kills", "observer_kills", "rune_pickups",
            "teamfight_participation", "camps_stacked", "creeps_stacked",
            "stuns", "sentry_uses", "roshan_kills", "tower_kills",
            "kills_per_min"
        ]
        df[cols_to_fill_zero] = df[cols_to_fill_zero].fillna(0)
        return df

    def _get_invalid_match_ids(self, df: pd.DataFrame, column: str):
        """
        Получение match_id с невалидными значениями по переменной (в командах Radiant или Dire более 2 игроков с invalid значениями).

        Args:
            df (pd.DataFrame): DataFrame с данными матчей.
            column (str): Столбец, по которому нужно проверять невалидные значения.

        Returns:
            Iterable: Индексы строк, где в Radiant или Dire более 2 игроков с невалидными значениями.
        """

        mask_invalid = df[column].isna() | (df[column] == 0)

        radiant_mask = df["player_slot"] < 5
        dire_mask = df["player_slot"] >= 128

        invalid_radiant_counts = df.loc[radiant_mask & mask_invalid].groupby(
            level=0)[column].count()
        invalid_dire_counts = df.loc[dire_mask & mask_invalid].groupby(
            level=0)[column].count()

        invalid_radiant_ids = set(
            invalid_radiant_counts[invalid_radiant_counts >= 2].index)
        invalid_dire_ids = set(
            invalid_dire_counts[invalid_dire_counts >= 2].index)

        return invalid_radiant_ids.union(invalid_dire_ids)

    def _remove_invalid_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Удаление матчей с некорректными действиями игроков.

        Args:
            df (pd.DataFrame): DataFrame с данными матчей.

        Returns:
            pd.DataFrame: DataFrame с удаленными некорректными матчами.
        """

        invalid_match_ids_actions = self._get_invalid_match_ids(
            df, "actions_per_min")
        invalid_match_ids_xp = self._get_invalid_match_ids(df, "total_xp")
        invalid_match_ids = invalid_match_ids_actions.union(
            invalid_match_ids_xp)

        df = df[~df.index.isin(invalid_match_ids)]
        return df


class DataPreprocessor:

    PLAYER_STATS_COLUMNS = [
        "previous_kills_avr", "previous_hero_kills_avr",
        "previous_courier_kills_avr", "previous_observer_kills_avr",
        "previous_kills_per_min_avr", "previous_kda_avr",
        "previous_denies_avr", "previous_hero_healing_avr",
        "previous_assists_avr", "previous_hero_damage_avr",
        "previous_deaths_avr", "previous_gold_per_min_avr",
        "previous_total_gold_avr", "previous_gold_spent_avr",
        "previous_level_avr", "previous_rune_pickups_avr",
        "previous_xp_per_min_avr", "previous_total_xp_avr",
        "previous_actions_per_min_avr", "previous_net_worth_avr",
        "previous_teamfight_participation_avr", "previous_camps_stacked_avr",
        "previous_creeps_stacked_avr", "previous_stuns_avr",
        "previous_sentry_uses_avr", "previous_roshan_kills_avr",
        "previous_tower_kills_avr", "previous_win_avr",
        "previous_duration_avr", "previous_first_blood_time_avr"
    ]

    def __init__(self) -> None:
        """
        Инициализация объекта класса DataPreprocessor.
        Инициализирует пустые атрибуты для хранения данных тренировочной и тестовой выборки.
        """
        self.df_train_aggregated: Optional[pd.DataFrame] = None
        self.df_train_team: Optional[pd.DataFrame] = None
        self.df_train: Optional[pd.DataFrame] = None
        self.df_test_team: Optional[pd.DataFrame] = None

    def fit(self, df_train: pd.DataFrame) -> "DataPreprocessor":
        """
        Вычисление агрегированных статистик на тренировочных данных.

        Args:
        df_train (pd.DataFrame): Тренировочные данные, содержащие информацию о матчах и игроках.

        Returns:
        DataPreprocessor: Объект класса.
        """
        self.df_train = df_train.copy()
        df_players_agg = self.aggregate_player_previous_stats(self.df_train)
        self.df_train_aggregated = df_players_agg
        df_team = self._aggregate_team_stats(df_players_agg)
        self.df_train_team = df_team.dropna(
            how="any")  # Удаление строк с NaN значениями
        return self

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """
        Вычисление агрегированных статистик и возврат преобразованных тренировочных данных.

        Args:
        df_train (pd.DataFrame): Тренировочные данные, содержащие информацию о матчах и игроках.

        Returns:
        pd.DataFrame: Преобразованные данные с агрегированной командной статистикой.
        """
        self.fit(df_train)
        return self.df_train_team

    def transform(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразование тестовых данных с использованием статистик, рассчитанных на train.

        Args:
        df_test (pd.DataFrame): Тестовые данные, содержащие информацию о матчах и игроках.

        Returns:
        pd.DataFrame: Преобразованные тестовые данные с агрегированной командной статистикой.
        """
        df_test = df_test.copy()
        df_test_players_agg = self._get_last_seen_player_stats(df_test)
        df_test_team = self._aggregate_team_stats(df_test_players_agg)
        self.df_test_team = df_test_team
        return self.df_test_team

    def transform_target_train(self, target: pd.Series) -> pd.Series:
        target = target.loc[self.df_train_team.index]
        target_aggregated = target.groupby(target.index).agg('mean')
        return target_aggregated

    def transform_target_test(self, target: pd.Series) -> pd.Series:
        """
        Преобразование целевой переменной для тренировочного набора данных.

        Args:
        target (pd.Series): Целевая переменная для тренировочных данных.

        Returns:
        pd.Series: Преобразованная целевая переменная для тренировочных данных.
        """
        target = target.loc[self.df_test_team.index]
        target_aggregated = target.groupby(target.index).agg('mean')
        return target_aggregated

    def aggregate_player_previous_stats(
            self, df_train: pd.DataFrame) -> pd.DataFrame:
        """
        Агрегация статистики игроков за предыдущие матчи.

        Args:
        df_train (pd.DataFrame): Данные о матчах и игроках, для которых необходимо вычислить статистику.

        Returns:
        pd.DataFrame: Данные с агрегированной статистикой для каждого игрока.
        """
        df_players_agg = df_train.copy()
        self._calculate_expanding_average(df_players_agg)
        return df_players_agg.sort_values(by=["start_date_time"])

    def get_player_previous_last_stats(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Получение статистических данных за последний матч для каждого игрока.

        Returns:
        tuple: Кортеж из двух элементов:
            - pd.DataFrame: Последняя статистика для каждого игрока.
            - pd.Series: Медианные значения для отсутствующих данных игроков.
        """
        # Создание копии, чтобы избежать "SettingWithCopyWarning"
        df_last_player_stats = self.df_train_aggregated.drop_duplicates(
            subset="account_id", keep="last").copy()
        df_last_player_stats["account_id"] = df_last_player_stats[
            "account_id"].astype(float)

        numeric_columns = self.df_train_aggregated.select_dtypes(
            include=["number"])
        missing_player_data = numeric_columns.median()
        return df_last_player_stats, missing_player_data

    def _get_last_seen_player_stats(self,
                                    df_test: pd.DataFrame) -> pd.DataFrame:
        """
        Получение последних статистических данных игроков для тестового набора.

        Args:
        df_test (pd.DataFrame): Тестовые данные, содержащие информацию о матчах и игроках.

        Returns:
        pd.DataFrame: Статистические данные за последний матч для каждого игрока из df_test.
        """
        df_test_agg = df_test.copy()[["account_id", "isRadiant"]]
        columns_to_keep = list(df_test_agg.columns) + self.PLAYER_STATS_COLUMNS
        columns_to_stats = self.PLAYER_STATS_COLUMNS + ["account_id"]
        df_test_agg['original_index'] = df_test_agg.index

        # Получение последних статистические данные игроков и данных о недостающих значениях.
        df_last_player_stats, missing_player_data = self.get_player_previous_last_stats(
        )

        df_test_agg = pd.merge(df_test_agg,
                               df_last_player_stats[columns_to_stats],
                               on="account_id",
                               how="left")

        # Заполнение пропусков значениями медиан из тренировочных данных.
        for column in self.PLAYER_STATS_COLUMNS:
            df_test_agg[column] = df_test_agg[column].fillna(
                missing_player_data[column])

        df_test_agg = df_test_agg.set_index('original_index')
        return df_test_agg[columns_to_keep]

    def _calculate_expanding_average(self,
                                     df_players_agg: pd.DataFrame) -> None:
        """
        Вычисление скользящего среднего для статистики игроков (получение данных за предыдущие матчи).

        Args:
        df_players_agg (pd.DataFrame): Данные с агрегированной статистикой для каждого игрока.

        Returns:
        None.
        """
        groupby_cols = "account_id"
        group_cols = [
            "kills", "hero_kills", "courier_kills", "observer_kills",
            "kills_per_min", "kda", "denies", "hero_healing", "assists",
            "hero_damage", "deaths", "gold_per_min", "total_gold",
            "gold_spent", "level", "rune_pickups", "xp_per_min", "total_xp",
            "actions_per_min", "net_worth", "teamfight_participation",
            "camps_stacked", "creeps_stacked", "stuns", "sentry_uses",
            "roshan_kills", "tower_kills", "win", "duration",
            "first_blood_time"
        ]

        # Расчет экспоненциального среднего
        df_players_agg[[
            f"previous_{col}_avr" for col in group_cols
        ]] = df_players_agg.groupby(groupby_cols)[group_cols].transform(
            lambda x: x.shift().expanding().mean())

    def _aggregate_team_stats(self,
                              df_players_agg: pd.DataFrame) -> pd.DataFrame:
        """
        Агрегация командной статистики.

        Args:
        df_players_agg (pd.DataFrame): Данные с агрегированной статистикой для каждого игрока.

        Returns:
        pd.DataFrame: Данные с агрегированной статистикой для команды.
        """
        aggregate_functions = ["mean", "max", "min"]
        radiant_df = df_players_agg[df_players_agg["isRadiant"] == 1]
        dire_df = df_players_agg[df_players_agg["isRadiant"] == 0]

        radiant_stats = self._calculate_team_stats(radiant_df, "team_1",
                                                   self.PLAYER_STATS_COLUMNS,
                                                   aggregate_functions)
        dire_stats = self._calculate_team_stats(dire_df, "team_2",
                                                self.PLAYER_STATS_COLUMNS,
                                                aggregate_functions)
        df_team = pd.merge(radiant_stats,
                           dire_stats,
                           left_index=True,
                           right_index=True)

        return df_team

    def _calculate_team_stats(self, team_df: pd.DataFrame, team_name: str,
                              columns_to_aggregate: list[str],
                              aggregate_functions: list[str]) -> pd.DataFrame:
        """
        Расчет командной статистики, основанной на данных игроков.

        Args:
        team_df (pd.DataFrame): Данные команды (Radiant или Dire), для которой необходимо выполнить агрегацию.
        team_name (str): Название команды (например, 'team_1' или 'team_2').
        columns_to_aggregate (list[str]): Список колонок, которые необходимо агрегировать.
        aggregate_functions (list[str]): Список статистических показателей для агрегации (например, 'mean', 'max', 'min').

        Returns:
        pd.DataFrame: Статистические показатели команды.
        """
        team_df = team_df.copy()
        aggregation_dict = {
            col: aggregate_functions
            for col in columns_to_aggregate
        }
        aggregated = team_df.groupby(team_df.index).agg(aggregation_dict)
        aggregated.columns = [
            f"{col}_{team_name}_{agg_func}"
            for col, agg_func in aggregated.columns.to_flat_index()
        ]
        aggregated.index.name = "match_id"
        return aggregated


class PredictionDataFetcher:

    def __init__(self, data_preprocessor: DataPreprocessor) -> None:
        """
        Инициализация объекта класса PredictionDataFetcher.

        Args:
            data_preprocessor (DataPreprocessor): Объект класса DataPreprocessor для предобработки данных.
        """
        self.data_preprocessing = data_preprocessor

    def get_team_info_from_dataclass(self, match: Match) -> pd.DataFrame:
        """
        Получение статистических показателей агрегированных по команде.

        Args:
            match (Match): Объект класса Match, содержащий информацию об игроках команд Radiant и Dire.

        Returns:
            pd.DataFrame: Агрегированная статистика по команде.
        """
        df = self._data_preprocessing_from_dataclass(match)
        df_team = self.data_preprocessing.transform(df)
        return df_team

    def get_team_info_from_dataframe(self,
                                     df_upload: pd.DataFrame) -> pd.DataFrame:
        """
        Получение статистических показателей агрегированных по команде.

        Args:
            df_upload (pd.DataFrame): Данные о матче и игроках в виде DataFrame.

        Returns:
            pd.DataFrame: Агрегированная статистика по команде.
        """
        df_upload = self._data_preprocessing_from_dataframe(df_upload)
        df_team = self.data_preprocessing.transform(df_upload)
        return df_team

    def _data_preprocessing_from_dataframe(
            self, df_upload: pd.DataFrame) -> pd.DataFrame:
        """
        Предобработка данных из DataFrame: преобразует данные о матче и игроках в формат необходимый для агрегации.

        Args:
            df_upload (pd.DataFrame): Данные о матчах и игроках в виде DataFrame.

        Returns:
            pd.DataFrame: Преобразованный DataFrame.
        """
        # Преобразование DataFrame из широкого формата в длинный, где 'slot' — это идентификаторы игроков в матче,
        # а 'account_id' — идентификаторы аккаунтов, привязанных к слотам.
        df_upload = df_upload.copy().melt(id_vars=["match_id"],
                                          var_name="slot",
                                          value_name="account_id")
        df_upload["slot"] = df_upload["slot"].str.extract("(\d+)")
        df_upload["isRadiant"] = df_upload["slot"].apply(
            lambda x: 1 if 0 <= int(x) <= 4 else 0)
        df_upload["account_id"] = df_upload["account_id"].astype(float)
        df_upload = df_upload.set_index("match_id")
        return df_upload

    def _data_preprocessing_from_dataclass(self, match: Match) -> pd.DataFrame:
        """
        Предобработка данных из объекта Match: преобразует информацию об игроках Radiant и Dire в формат DataFrame.

        Args:
            match (Match): Объект класса Match, содержащий информацию об игроках команд Radiant и Dire.

        Returns:
            pd.DataFrame: Преобразованный DataFrame с данными о матчах и игроках.
        """
        data = []

        # Добавление информации об игроках команд
        for player in match.radiant:
            data.append({
                "match_id": 1,
                "account_id": player.account_id,
                "isRadiant": 1
            })

        for player in match.dire:
            data.append({
                "match_id": 1,
                "account_id": player.account_id,
                "isRadiant": 0
            })

        df = pd.DataFrame(data)
        df = pd.DataFrame(data).set_index("match_id")
        return df
