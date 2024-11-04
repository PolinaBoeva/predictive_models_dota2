# Описание набора данных

### Ссылка на набор данных: https://disk.yandex.ru/d/SPUZTmuKDwVu1A

## Описание признаков

| Поле                                      | Тип       | Описание                                                                                         |
|-------------------------------------------|-----------|--------------------------------------------------------------------------------------------------|
| `match_id`                                | integer   | Уникальный идентификатор матча.                                                                        |
| `first_blood_time`                        | float64   | Время, когда произошло первое убийство (в секундах).                                                   |
| `buyback_log_{time}_{player_slot}`        | float64   | Количество выкупов на момент времени {time} у игрока {player_slot}                                     |
| `dn_t_{time}_{player_slot}`               | float64   | Количество добиваний союзных крипов на момент времени {time} у игрока {player_slot}                    |
| `lh_t_{time}_{player_slot}`               | float64   | Количество добиваний на момент времени {time} у игрока {player_slot}                                   |
| `kills_log_{time}_{player_slot}`          | float64   | Количество убийств на момент времени {time} у игрока {player_slot}                                     |
| `gold_t_{time}_{player_slot}`             | float64   | Количество золота на момент времени {time} у игрока {player_slot}                                      |
| `exp_t_{time}_{player_slot}`              | float64   | Количество опыта на момент времени {time} у игрока {player_slot}                                       |
| `party_size_{player_slot}`                | float64   | Размер команды игрока {player_slot}                                                                    |
| `purchase_log_{item}_{time}_{player_slot}`| float64   | Количество предметов {item}, купленных игроком {player_slot}, на момент времени {time}                 |
| `runes_log_{type}_{time}_{player_slot}`   | float64   | Количество рун типа {type}, подобранных игроком {player_slot}, на момент времени {time}                |
| `obs_log_{time}_{player_slot}`            | float64   | Количество Observer вардов, поставленных игроком {player_slot}, на момент времени {time}               |
| `sen_log_{time}_{player_slot}`            | float64   | Количество Sentry вардов, поставленных игроком {player_slot}, на момент времени {time}                 |
| `obs_left_log_{time}_{player_slot}`       | float64   | Количество Observer вардов, сломанных у игрока {player_slot}, на момент времени {time}                 |
| `sen_left_log_{time}_{player_slot}`       | float64   | Количество Sentry вардов, сломанных у игрока {player_slot}, на момент времени {time}                   |
| `max_gold_player_radiant_slot_{time}`     | float64   | Номер слота игрока в команде radiant с максимальным количеством золота на момент времени {time}        |
| `max_gold_player_dire_slot_{time}`        | float64   | Номер слота игрока в команде dire с максимальным количеством золота на момент времени {time}           |
| `max_gold_player_radiant_{time}`          | float64   | Количество золота игрока в команде radiant с максимальным количеством золота на момент времени {time}  |
| `max_gold_player_dire_{time}`             | float64   | Количество золота игрока в команде dire с максимальным количеством золота на момент времени {time}     |
| `max_gold_player_diff_{time}`             | float64   | Разница золота игроков с максимальным количеством золота на момент времени {time} в двух командах      |
| `sum_{feature}_radiant_{time}`            | float64   | Сумма значений признака {feature} у команды radiant на момент времени {time}                           |
| `sum_{feature}_dire_{time}`               | float64   | Сумма значений признака {feature} у команды dire на момент времени {time}                              |
| `sum_{feature}_diff_{time}`               | float64   | Разница сумм значений признака {feature} у команд radiant и dire на момент времени {time}              |
| `firstblood_claimed_{player_slot}`        | bool      | Совершил ли игрок {player_slot} первое убийство в матче                                                |

