class DataCleaner:
    def __init__(self):
        pass  

    def fill_nan(self, df):
        """Метод для очистки данных в DataFrame"""
        df = self._drop_unnecessary_columns(df)
        df = self._remove_invalid_rows(df)
        df = self._fill_zero_values(df)
        df['kills_per_min'] = df.apply(self._fix_kills_per_min, axis=1)
        df['actions_per_min'] = pd.to_numeric(df['actions_per_min'], errors='coerce')
        df = self._remove_invalid_matches(df)
        df = df[df['game_mode'] == 2]
        return df  

    def _drop_unnecessary_columns(self, df):
        """Удаление ненужных столбцов."""
        cols_to_drop = ['pings', 'throw', 'loss', 'comeback', 'rank_tier', 'moonshard', 'aghanims_scepter']
        return df.drop(cols_to_drop, axis=1)

    def _remove_invalid_rows(self, df):
        """Удаление строк с пустыми значениями в 'account_id' и 'win'."""
        invalid_match_ids = df[df['account_id'].isna()]['match_id'].unique()
        df = df[~df['match_id'].isin(invalid_match_ids)]
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
        mask_invalid = df['actions_per_min'].isna() | (df['actions_per_min'] == 0)
        
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

        invalid_match_ids = invalid_match_ids_radiant.union(invalid_match_ids_dire)
        df = df[~df['match_id'].isin(invalid_match_ids)]
        return df

class DataPreprocessor:
    def __init__(self):
        self.df_train_aggregated = None
        self.df_train_team = None
        self.df_train = None

    def fit(self, df_train):
        self.df_train = df_train.copy()
        df_players_agg = self.aggregate_player_previous_stats(self.df_train)
        self.df_train_aggregated = df_players_agg
        df_team = self.aggregate_team_stats(df_players_agg)
        df_team = df_team.dropna(how='any')
        self.df_train_team = df_team

    def fit_transform(self, df_train):
        self.fit(df_train)
        return self.df_train_team

    def transform(self, df_test):
        df_test = df_test.copy()
        df_test_players_agg = self._get_last_seen_player_stats(df_test)
        df_test_team = self.aggregate_team_stats(df_test_players_agg)
        return df_test_team

    def aggregate_player_previous_stats(self, df_train):
        df_players_agg = df_train.copy()
        self._calculate_expanding_average(df_players_agg)
        return df_players_agg.sort_values(by=['start_date_time'])

    def get_player_previous_last_stats(self):
        df_last_player_stats = self.df_train_aggregated.copy().drop_duplicates(subset='account_id', keep='last')
        numeric_columns = self.df_train_aggregated.select_dtypes(include=['number'])
        missing_player_data = numeric_columns.median()
        return df_last_player_stats, missing_player_data

    def _get_last_seen_player_stats(self, df_test):
        df_test_agg = df_test.copy()
        player_columns = [
            'previous_kills_avr', 'previous_hero_kills_avr', 'previous_courier_kills_avr',
            'previous_observer_kills_avr', 'previous_kills_per_min_avr', 'previous_kda_avr',
            'previous_denies_avr', 'previous_hero_healing_avr', 'previous_assists_avr',
            'previous_hero_damage_avr', 'previous_deaths_avr', 'previous_gold_per_min_avr',
            'previous_total_gold_avr', 'previous_gold_spent_avr', 'previous_level_avr',
            'previous_rune_pickups_avr', 'previous_xp_per_min_avr', 'previous_total_xp_avr',
            'previous_actions_per_min_avr', 'previous_net_worth_avr', 'previous_teamfight_participation_avr',
            'previous_camps_stacked_avr', 'previous_creeps_stacked_avr', 'previous_stuns_avr',
            'previous_sentry_uses_avr', 'previous_roshan_kills_avr', 'previous_tower_kills_avr',
            'previous_win_avr', 'previous_duration_avr', 'previous_first_blood_time_avr'
        ]
        columns_to_keep = ['match_id', 'account_id', 'isRadiant', 'radiant_win'] + player_columns

        df_last_player_stats, missing_player_data = self.player_previous_last_stats

        df_test_agg = pd.merge(df_test_agg, df_last_player_stats, on='account_id', how='left')

        for column in player_columns:
            df_test_agg[column] = df_test_agg[column].fillna(missing_player_data[column])

        df_test_agg = df_test_agg[columns_to_keep]

        return df_test_agg

    def _calculate_expanding_average(self, df_players_agg):
        groupby_cols = 'account_id'
        group_cols = ['kills', 'hero_kills', 'courier_kills', 'observer_kills', 'kills_per_min', 'kda', 'denies',
                      'hero_healing', 'assists', 'hero_damage', 'deaths', 'gold_per_min', 'total_gold', 'gold_spent',
                      'level', 'rune_pickups', 'xp_per_min', 'total_xp', 'actions_per_min', 'net_worth', 'teamfight_participation',
                      'camps_stacked', 'creeps_stacked', 'stuns', 'sentry_uses', 'roshan_kills', 'tower_kills', 'win',
                      'duration', 'first_blood_time']

        for col in group_cols:
            df_players_agg[f'previous_{col}_avr'] = df_players_agg.groupby(groupby_cols)[col].transform(lambda x: x.shift().expanding().mean())

    def aggregate_team_stats(self, df_players_agg):
        stats = ['mean', 'max', 'min']
        columns_to_aggregate = ['previous_kills_avr', 'previous_hero_kills_avr', 'previous_courier_kills_avr',
                                'previous_observer_kills_avr', 'previous_kills_per_min_avr', 'previous_kda_avr',
                                'previous_denies_avr', 'previous_hero_healing_avr', 'previous_assists_avr',
                                'previous_hero_damage_avr', 'previous_deaths_avr', 'previous_gold_per_min_avr',
                                'previous_total_gold_avr', 'previous_gold_spent_avr', 'previous_level_avr',
                                'previous_rune_pickups_avr', 'previous_xp_per_min_avr', 'previous_total_xp_avr',
                                'previous_actions_per_min_avr', 'previous_net_worth_avr', 'previous_teamfight_participation_avr',
                                'previous_camps_stacked_avr', 'previous_creeps_stacked_avr', 'previous_stuns_avr',
                                'previous_sentry_uses_avr', 'previous_roshan_kills_avr', 'previous_tower_kills_avr',
                                'previous_win_avr', 'previous_duration_avr', 'previous_first_blood_time_avr']

        radiant_df = df_players_agg[df_players_agg['isRadiant'] == 1]
        dire_df = df_players_agg[df_players_agg['isRadiant'] == 0]

        radiant_stats = self._calculate_team_stats(radiant_df, 'team_1', columns_to_aggregate, stats)
        dire_stats = self._calculate_team_stats(dire_df, 'team_2', columns_to_aggregate, stats)

        df_team = pd.merge(radiant_stats, dire_stats, on='match_id')
        radiant_win = df_players_agg[['radiant_win', 'match_id']].groupby('match_id').mean().reset_index()
        df_team = df_team.merge(radiant_win, on='match_id')

        return df_team

    def _calculate_team_stats(self, team_df, team_name, columns_to_aggregate, stats):
        aggregation_dict = {col: stats for col in columns_to_aggregate}
        aggregated = team_df.groupby('match_id').agg(aggregation_dict)

        new_columns = []
        for col in aggregated.columns:
            col_name, stat = col
            new_columns.append(f'{col_name}_{team_name}_{stat}')

        aggregated.columns = new_columns
        return aggregated

@dataclass
class Player:
    account_id: int
    hero_name: str = None

@dataclass
class Match:
    dire: List[Player]
    radiant: List[Player]

class PredictionDataFetcher():
    def __init__(self):
        self.data_preprocessing = DataPreprocessor()

    def get_team_info_from_dataclass(self, match: Match, df_players_agg: pd.DataFrame) -> pd.DataFrame:
        """Метод для работы с dataclass"""
        return self._calculate_team_info_from_dataclass(match, df_players_agg)

    def get_team_info_from_dataframe(self, df_upload: pd.DataFrame, df_players_agg: pd.DataFrame, missing_player_data: pd.DataFrame) -> pd.DataFrame:
        df_upload_agg = self._calculate_player_info_from_dataframe(df_upload, df_players_agg, missing_player_data)
        df_team = self._calculate_team_info_from_dataframe(df_upload_agg)
        return df_team

    def _calculate_team_info_from_dataclass(self, match: Match, df_players_agg: pd.DataFrame) -> pd.DataFrame:
        """Агрегирует статистику по команде, когда данные подаются в виде dataclass."""
        stats = ['mean', 'max', 'min']
        columns_to_aggregate = [
            'previous_kills_avr', 'previous_hero_kills_avr', 'previous_courier_kills_avr',
            'previous_observer_kills_avr', 'previous_kills_per_min_avr', 'previous_kda_avr',
            'previous_denies_avr', 'previous_hero_healing_avr', 'previous_assists_avr',
            'previous_hero_damage_avr', 'previous_deaths_avr', 'previous_gold_per_min_avr',
            'previous_total_gold_avr', 'previous_gold_spent_avr', 'previous_level_avr',
            'previous_rune_pickups_avr', 'previous_xp_per_min_avr', 'previous_total_xp_avr',
            'previous_actions_per_min_avr', 'previous_net_worth_avr', 'previous_teamfight_participation_avr',
            'previous_camps_stacked_avr', 'previous_creeps_stacked_avr', 'previous_stuns_avr',
            'previous_sentry_uses_avr', 'previous_roshan_kills_avr', 'previous_tower_kills_avr',
            'previous_win_avr', 'previous_duration_avr', 'previous_first_blood_time_avr'
        ]
        radiant_stats = pd.DataFrame([{
            **{col: df_players_agg[df_players_agg['account_id'] == float(player.account_id)].iloc[-1][col]
              for col in columns_to_aggregate},
            'match_id': 1
        } for player in match.radiant])

        dire_stats = pd.DataFrame([{
            **{col: df_players_agg[df_players_agg['account_id'] == float(player.account_id)].iloc[-1][col]
              for col in columns_to_aggregate},
            'match_id': 1
        } for player in match.dire])

        radiant_stats = self.data_preprocessing._calculate_team_stats(radiant_stats, 'team_1', columns_to_aggregate, stats)
        dire_stats = self.data_preprocessing._calculate_team_stats(dire_stats, 'team_2', columns_to_aggregate, stats)

        df_team = pd.merge(radiant_stats, dire_stats, on='match_id')

        return df_team

    def _calculate_team_info_from_dataframe(self, df_upload_agg: pd.DataFrame) -> pd.DataFrame:
        """Расчет агрегированной статистики по командам на основе агрегированной статистики по действиям игроков за предыдущие матчи"""
        stats = ['mean', 'max', 'min']
        columns_to_aggregate = ['previous_kills_avr', 'previous_hero_kills_avr', 'previous_courier_kills_avr',
                                'previous_observer_kills_avr', 'previous_kills_per_min_avr', 'previous_kda_avr',
                                'previous_denies_avr', 'previous_hero_healing_avr', 'previous_assists_avr',
                                'previous_hero_damage_avr', 'previous_deaths_avr', 'previous_gold_per_min_avr',
                                'previous_total_gold_avr', 'previous_gold_spent_avr', 'previous_level_avr',
                                'previous_rune_pickups_avr', 'previous_xp_per_min_avr', 'previous_total_xp_avr',
                                'previous_actions_per_min_avr', 'previous_net_worth_avr', 'previous_teamfight_participation_avr',
                                'previous_camps_stacked_avr', 'previous_creeps_stacked_avr', 'previous_stuns_avr',
                                'previous_sentry_uses_avr', 'previous_roshan_kills_avr', 'previous_tower_kills_avr',
                                'previous_win_avr', 'previous_duration_avr', 'previous_first_blood_time_avr']

        radiant_df = df_upload_agg[df_upload_agg['slot'].isin([0, 1, 2, 3, 4])]
        dire_df = df_upload_agg[~df_upload_agg['slot'].isin([0, 1, 2, 3, 4])]

        radiant_stats = self.data_preprocessing._calculate_team_stats(radiant_df, 'team_1', columns_to_aggregate, stats)
        dire_stats = self.data_preprocessing._calculate_team_stats(dire_df, 'team_2', columns_to_aggregate, stats)

        df_team = pd.merge(radiant_stats, dire_stats, on='match_id')

        return df_team

    def _calculate_player_info_from_dataframe(self, df_upload: pd.DataFrame, df_last_player_stats: pd.DataFrame, missing_player_data: pd.DataFrame) -> pd.DataFrame:
          """Метод для получения значений из df_players_agg"""
          df_players_agg['account_id'] = df_players_agg['account_id'].astype(float)
          df_upload_agg = df_upload.copy().melt(id_vars=['match_id'], var_name='slot', value_name='account_id')
          df_upload_agg['slot'] = df_upload_agg['slot'].str.extract('(\d+)')
          df_upload_agg['slot'] = pd.to_numeric(df_upload_agg['slot'], errors='coerce')
          df_upload_agg['account_id'] = df_upload_agg['account_id'].astype(float)
          player_columns = [
                        'previous_kills_avr', 'previous_hero_kills_avr', 'previous_courier_kills_avr',
                        'previous_observer_kills_avr', 'previous_kills_per_min_avr', 'previous_kda_avr',
                        'previous_denies_avr', 'previous_hero_healing_avr', 'previous_assists_avr',
                        'previous_hero_damage_avr', 'previous_deaths_avr', 'previous_gold_per_min_avr',
                        'previous_total_gold_avr', 'previous_gold_spent_avr', 'previous_level_avr',
                        'previous_rune_pickups_avr', 'previous_xp_per_min_avr', 'previous_total_xp_avr',
                        'previous_actions_per_min_avr', 'previous_net_worth_avr', 'previous_teamfight_participation_avr',
                        'previous_camps_stacked_avr', 'previous_creeps_stacked_avr', 'previous_stuns_avr',
                        'previous_sentry_uses_avr', 'previous_roshan_kills_avr', 'previous_tower_kills_avr',
                        'previous_win_avr', 'previous_duration_avr', 'previous_first_blood_time_avr'
                    ]

          columns_to_keep = list(df_upload_agg.columns) + player_columns
          columns_to_stats = player_columns + ['account_id']
                    
          df_players_agg['account_id'] = df_players_agg['account_id'].astype(float)
          df_upload_agg['account_id'] = df_upload_agg['account_id'].astype(float)
          df_upload_agg = pd.merge(df_upload_agg, df_players_agg[columns_to_stats], on='account_id', how='left')

          for column in player_columns:
              df_upload_agg[column] = df_upload_agg[column].fillna(missing_player_data[column])

          df_upload_agg = df_upload_agg[columns_to_keep]

          return df_upload_agg
