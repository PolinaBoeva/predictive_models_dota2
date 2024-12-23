class DataCleaning:
    def __init__(self):
        pass  

    def fill_nan(self, df):
        """Метод для очистки данных в DataFrame"""
        # Удаление ненужных столбцов
        df = self._drop_unnecessary_columns(df)

        # Удаление строк с пустыми значениями в 'account_id' и 'win'
        df = self._remove_invalid_rows(df)

        # Заполнение NaN значений в числовых столбцах
        df = self._fill_zero_values(df)

        # Обработка столбца 'kills_per_min'
        df['kills_per_min'] = df.apply(self._fix_kills_per_min, axis=1)

        # Преобразование столбца 'actions_per_min' в числовой формат
        df['actions_per_min'] = pd.to_numeric(df['actions_per_min'], errors='coerce')

        # Удаление матчей с некорректными действиями игроков
        df = self._remove_invalid_matches(df)

        # Фильтрация по game_mode
        df = df[df['game_mode'] == 2]

        return df  # Возвращаем очищенный DataFrame

    def _drop_unnecessary_columns(self, df):
        """Удаление ненужных столбцов."""
        cols_to_drop = ['pings', 'throw', 'loss', 'comeback', 'rank_tier', 'moonshard', 'aghanims_scepter']
        return df.drop(cols_to_drop, axis=1)

    def _remove_invalid_rows(self, df):
        """Удаление строк с пустыми значениями в 'account_id' и 'win'."""
        # Удаление строк с отсутствующими значениями в 'account_id'
        invalid_match_ids = df[df['account_id'].isna()]['match_id'].unique()
        df = df[~df['match_id'].isin(invalid_match_ids)]

        # Удаление строк, где значение в 'win' отсутствует
        df = df[df['win'].notna()]
        return df

    def _fill_zero_values(self, df):
        """Заполнение NaN значений нулями в специфичных столбцах."""
        cols_to_fill_zero = [
            'hero_kills', 'courier_kills', 'observer_kills', 'rune_pickups',
            'teamfight_participation', 'camps_stacked', 'creeps_stacked',
            'stuns', 'sentry_uses', 'roshan_kills', 'tower_kills'
        ]
        df[cols_to_fill_zero] = df[cols_to_fill_zero].fillna(0)
        return df

    def _fix_kills_per_min(self, row):
        """Заполнение NaN в 'kills_per_min' на 0, если 'kills' равен 0."""
        return 0 if pd.isna(row['kills_per_min']) and row['kills'] == 0 else row['kills_per_min']

    def _remove_invalid_matches(self, df):
        """Удаление матчей с некорректными действиями игроков."""
        # Маска для игроков с actions_per_min = 0 или NaN
        mask_invalid = df['actions_per_min'].isna() | (df['actions_per_min'] == 0)

        # Разделение игроков на Radiant и Dire
        radiant_players = df[df['player_slot'] < 5]
        dire_players = df[df['player_slot'] >= 128]

        # Подсчёт числа игроков с invalid actions_per_min для каждой команды
        invalid_radiant_counts = radiant_players.groupby('match_id')['actions_per_min'].apply(
            lambda x: (mask_invalid.loc[x.index]).sum()
        )
        invalid_dire_counts = dire_players.groupby('match_id')['actions_per_min'].apply(
            lambda x: (mask_invalid.loc[x.index]).sum()
        )

        # Получение match_id, где в Radiant или Dire больше или равно 2 игроков с invalid actions_per_min
        invalid_match_ids_radiant = invalid_radiant_counts[invalid_radiant_counts >= 2].index
        invalid_match_ids_dire = invalid_dire_counts[invalid_dire_counts >= 2].index

        # Объединение match_id для обеих команд
        invalid_match_ids = invalid_match_ids_radiant.union(invalid_match_ids_dire)

        # Удаление строк с такими match_id
        df = df[~df['match_id'].isin(invalid_match_ids)]
        return df
