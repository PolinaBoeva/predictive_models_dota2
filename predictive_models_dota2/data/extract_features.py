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
        missing_player_data = self.df_train_aggregated[player_columns].median()

        for account_id in df_test_agg['account_id'].unique():
            last_seen_stats = self.df_train_aggregated[self.df_train_aggregated['account_id'] == account_id]

            if not last_seen_stats.empty:
                last_seen_stats = last_seen_stats.iloc[-1]
                if last_seen_stats.isna().any():
                    for col in player_columns:
                        df_test_agg.loc[df_test_agg['account_id'] == account_id, col] = missing_player_data[col]
                else:
                    for col in player_columns:
                        if col in last_seen_stats.index:
                            df_test_agg.loc[df_test_agg['account_id'] == account_id, col] = last_seen_stats[col]
            else:
                for col in player_columns:
                    df_test_agg.loc[df_test_agg['account_id'] == account_id, col] = missing_player_data[col]

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

class PredictionDataFetcher():
    def __init__(self):
        self.data_preprocessing = DataPreprocessor()

    def get_team_info(self, data: Union['Match', pd.DataFrame], df_players_agg: pd.DataFrame) -> pd.DataFrame:
        """Метод для работы с dataclass или DataFrame, возвращающий DataFrame"""
        if isinstance(data, Match):
            return self._calculate_team_info_from_dataclass(data, df_players_agg)
        elif isinstance(data, pd.DataFrame):
            df_upload_agg = self._calculate_player_info_from_dataframe(df_upload, df_players_agg)
            df_team = self._calculate_team_info_from_dataframe(df_upload_agg)
            return df_team

    def _calculate_team_info_from_dataclass(self, match: 'Match', df_players_agg: pd.DataFrame) -> pd.DataFrame:
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

        self._assign_player_stats(match.radiant, df_players_agg, columns_to_aggregate)
        self._assign_player_stats(match.dire, df_players_agg, columns_to_aggregate)

        radiant_stats = pd.DataFrame([player.stats for player in match.radiant])
        dire_stats = pd.DataFrame([player.stats for player in match.dire])
        
        # Добавляем match_id как фиктивное значение для объединения
        radiant_stats['match_id'] = 1
        dire_stats['match_id'] = 1

        radiant_stats = self.data_preprocessing._calculate_team_stats(radiant_stats, 'team_1', columns_to_aggregate, stats)
        dire_stats = self.data_preprocessing._calculate_team_stats(dire_stats, 'team_2', columns_to_aggregate, stats)

        df_team = pd.merge(radiant_stats, dire_stats, on='match_id')

        return df_team

    def _assign_player_stats(self, players: List['Player'], df_players_agg: pd.DataFrame, columns_to_aggregate: List[str]):
        """Метод для подтягивания статистики из df_players_agg в Player"""
        for player in players:
            player_stats = df_players_agg[df_players_agg['account_id'] == str(player.account_id)].iloc[-1]
            player.stats = {col: player_stats[col] for col in columns_to_aggregate}  

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

        # Делим на команды Radiant и Dire
        radiant_df = df_upload_agg[df_upload_agg['slot'].isin([0, 1, 2, 3, 4])]
        dire_df = df_upload_agg[~df_upload_agg['slot'].isin([0, 1, 2, 3, 4])]

        # Агрегируем статистику для каждой команды
        radiant_stats = self.data_preprocessing._calculate_team_stats(radiant_df, 'team_1', columns_to_aggregate, stats)
        dire_stats = self.data_preprocessing._calculate_team_stats(dire_df, 'team_2', columns_to_aggregate, stats)

        # Объединяем результаты по match_id
        df_team = pd.merge(radiant_stats, dire_stats, on='match_id')

        return df_team

    def _calculate_player_info_from_dataframe(self, df_upload: pd.DataFrame, df_players_agg: pd.DataFrame) -> pd.DataFrame:
          """Метод для получения значений из df_players_agg"""
          df_players_agg['account_id'] = df_players_agg['account_id'].astype(str)
          df_upload_agg = df_upload.copy().melt(id_vars=['match_id'], var_name='slot', value_name='account_id')
          df_upload_agg['slot'] = df_upload_agg['slot'].str.extract('(\d+)')
          df_upload_agg['slot'] = pd.to_numeric(df_upload_agg['slot'], errors='coerce') 
          df_upload_agg['account_id'] = df_upload_agg['account_id'].astype(str)

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

          missing_player_data = df_players_agg[player_columns].median()

          for account_id in df_upload_agg['account_id'].unique():
              last_seen_stats = df_players_agg[df_players_agg['account_id'] == account_id]

              if not last_seen_stats.empty:
                  last_seen_stats = last_seen_stats.iloc[-1]  

                  if last_seen_stats.isna().any():
                      for col in player_columns:
                          df_upload_agg.loc[df_upload_agg['account_id'] == account_id, col] = missing_player_data[col]

                  else:
                      for col in player_columns:
                          if col in last_seen_stats.index:
                              df_upload_agg.loc[df_upload_agg['account_id'] == account_id, col] = last_seen_stats[col]
              else:
                  for col in player_columns:
                      df_upload_agg.loc[df_upload_agg['account_id'] == account_id, col] = missing_player_data[col]

          df_upload_agg = df_upload_agg[columns_to_keep]

          return df_upload_agg
