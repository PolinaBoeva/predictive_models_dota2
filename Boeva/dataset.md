Данные до обработки: https://www.kaggle.com/datasets/bwandowando/dota-2-pro-league-matches-2023/code

~~Данные после обработки (Dota2_EDA.ipynb): https://drive.google.com/file/d/1Wv4VnTA5Z_o-i7cA2nNH0saYmaMZXFUv/view?usp=drive_link~~

Данные за 10 месяцев 2024 (Dota2_EDA_new.ipynb): https://drive.google.com/file/d/16HEgmUnfbfw3Q_LSM419w1YSIfYP7UZR/view?usp=drive_link


Описание данных (табличные данные): 
  
| Переменная                | Тип данных               | Описание переменной                                          |
|---------------------------|-------------------------|-------------------------------------------------------------|
| match_id                  | int64                   | уникальный идентификатор матча                               |
| player_slot               | int64                   | слот игрока в команде                                       |
| account_id                | float64                 | уникальный идентификатор игрока                              |
| hero_id                   | float64                 | идентификатор героя                                         |
| kills                     | float64                 | количество убийств, совершенных игроком                     |
| hero_kills                | float64                 | количество убийств игроков, совершенных игроком            |
| courier_kills             | float64                 | количество убитых курьеров игроком                          |
| observer_kills            | float64                 | количество уничтоженных Observer Ward                        |
| kills_per_min             | float64                 | количество убийств в минуту                                  |
| kda                       | float64                 | соотношение между сумой убийств и ассистов на количество смертей |
| denies                    | float64                 | количество убийств союзников                                 |
| hero_healing              | float64                 | лечение союзных героев игроком                              |
| item_0 - item_5                 | object                  | предмет, купленный командой                                  |
| assists                   | float64                 | помощь союзному герою в убийстве вражеского героя          |
| hero_damage               | float64                 | общий урон героя вражеским героям                           |
| deaths                    | float64                 | количество смертей игрока                                    |
| gold_per_min              | float64                 | общее количество золота игрока в минуту                     |
| total_gold                | float64                 | общее количество золота за игру                              |
| level                     | float64                 | уровень игрока на конец игры                                 |
| rune_pickups              | float64                 | количество подобранных рун                                   |
| xp_per_min                | float64                 | количество опыта в минуту                                    |
| total_xp                  | float64                 | всего опыта                                                 |
| actions_per_min           | float64                 | количество действий игрока за минуту                        |
| net_worth                 | float64                 | общая ценность персонажа (количество кэша и инвентарь)    |
| teamfight_participation    | float64                 | участие в командных боях                                     |
| camps_stacked             | float64                 | количество stacked лагерей                                   |
| creeps_stacked            | float64                 | количество stacked крипов из их лагеря                      |
| stuns                     | float64                 | количество использованных оглушений                          |
| sentry_uses               | float64                 | количество использований Sentry Ward                         |
| lane_efficiency_pct       | float64                 | процент эффективности на линии                               |
| lane                      | float64                 | линия                                                       |
| lane_role                 | float64                 | роль на линии                                              |
| roshan_kills              | float64                 | общее количество убийств рошана, совершенных игроком       |
| tower_kills               | float64                 | количество уничтоженных башен противника                    |
| game_mode                 | int64                   | набор ограничений, в рамках которых можно играть           |
| rank_tier                 | float64                 | ранг игрока                                                |
| aghanims_scepter          | float64                 | наличие Scepter (0, 1, Nan)                                 |
| moonshard                 | float64                 | наличие moonshard (0, 1, Nan)                               |
| isRadiant                 | object                  | флаг указатель команды                                      |
| win                       | float64                 | флаг указатель победы                                      |
| start_date_time           | datetime64[ns]         | дата и время начала матча                                   |
| duration                  | float64                 | длительность матча в секундах                               |
| first_blood_time          | float64                 | время первого убийства в секундах                           |
| hero_name                 | object             | имя героя                                                  |
| primary_attr              | object             | основной атрибут героя                                     |
| attack_type               | object            | тип атаки героя                                            |
| hero_roles                | object            | основные роли героев       


Переменные с префиксами 'previous_{}_avr' - числовые переменные, рассчитанные для каждого account_id на основе предыдущих матчей, в случае, если предыдущих данных нет - Nan. 

Переменные с префиксами 'min/max/mean_previous_{}_avr' - числовые переменные, рассчитанные для каждого match_id на основе данных о предыдущих матчах accout_id, участников матча. В случае, если предыдущих данных нет ни для одного из участников команды -  Nan.
