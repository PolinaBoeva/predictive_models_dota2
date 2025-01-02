import pandas as pd
import plotly.express as px
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Гистограммы распределения значений признаков по матчам
def plot_metric_histogram(data, columns_to_plot, selected_metric):
    """Создание гистограммы для выбранной метрики."""
    logger.info(f"Построение гистограммы для метрики: {selected_metric}.")

    selected_column = columns_to_plot[selected_metric]

    fig = px.histogram(
        data,
        x=selected_column,
        labels={selected_column: selected_metric},
        nbins=30,
    )
    return fig

    fig.update_layout(
        bargap=0,
        xaxis_title='Номер матча',
        yaxis_title=selected_metric
    )
    logger.info("Гистограмма успешно построена.")
    return fig


# Распределение исходов матчей для данного игрока
def create_result_pie_chart(player_data):
    """Создание круговой диаграммы, показывающей распределение побед и поражений игрока."""
    logger.info("Создание круговой диаграммы для распределения исходов матчей.")

    result_table = pd.DataFrame({
        'Результат': ['Победы', 'Поражения'],
        'Количество': [player_data['win'].sum(), len(player_data) - player_data['win'].sum()]})
    fig = px.pie(result_table, names='Результат', values='Количество',
                 color_discrete_sequence=['#FFB6C1', '#A3D9FF'],
                 hole=0)

    logger.info("Круговая диаграмма успешно построена.")
    return fig


# Распределние переменной для выбраного игрока
def create_distribution_plot(df_player, variable):
    """Создание гистограммы распределения для выбранной переменной."""
    logger.info(f"Создание гистограммы распределения для переменной: {variable}.")

    if variable in df_player.columns:
        fig = px.histogram(df_player, x=variable, title=f'Распределение {variable} по игроку')
        logger.info("Гистограмма распределения успешно построена.")
        return fig
    else:
        logger.error(f"Переменная {variable} не найдена в данных.")
        raise ValueError(f"Переменная {variable} не найдена в данных.")


# топ-10 популярных героев
def create_top_10_heroes_plot(df, top_10_heroes):
    """Создание графика, отображающего топ-10 популярных героев."""
    logger.info("Создание графика топ-10 популярных героев.")

    hero_counts = df["hero_name"].value_counts()
    df_top_10 = hero_counts[top_10_heroes].reset_index()
    df_top_10.columns = ['hero_name', 'count']
    df_top_10 = df_top_10.sort_values(by='count', ascending=False)

    fig = px.bar(df_top_10, x='hero_name', y='count', color='hero_name',
                 labels={'hero_name': 'Имя героя', 'count': 'Число вхождений'})

    logger.info("График топ-10 популярных героев успешно построен.")
    return fig


# Распределение убийств по героям
def create_selected_heroes_plot(filtered_df, attribute):
    """Создание гистограммы, показывающей распределение заданного атрибута среди выбранных героев."""
    logger.info(f"Создание гистограммы для атрибута: {attribute}.")

    fig = px.histogram(
        filtered_df,
        x="hero_name",
        y=attribute,
        color="hero_name",
        labels={"hero_name": "Имя героя", attribute: attribute.capitalize()},
        height=600
    )
    fig.update_layout(xaxis_title="Имя героя", yaxis_title=attribute.capitalize())
    logger.info("Гистограмма по выбранным героям успешно построена.")
    return fig


# Box-plot распределения убийств по выбранным героям
def create_box_plot(filtered_df, attribute):
    """Создание box-plot для распределения заданного атрибута среди выбранных героев."""
    logger.info(f"Создание box-plot для атрибута: {attribute}.")

    fig = px.box(
        filtered_df,
        x="hero_name",
        y=attribute,
        color="hero_name",
        labels={"hero_name": "Имя героя", attribute: attribute.capitalize()},
        height=600
    )
    fig.update_layout(xaxis_title="Имя героя", yaxis_title=attribute.capitalize())
    logger.info("Box-plot успешно построен.")
    return fig


# Победы команд Radiant vs Dire
def create_winrate_pie_chart(df):
    """Создание круговой диаграммы, показывающей соотношение побед команд Radiant и Dire."""
    logger.info("Создание круговой диаграммы побед команд Radiant и Dire.")

    df['winrate'] = df['win'].mean()
    radiant_wins = df[df['isRadiant'] == 1]['win'].sum()
    dire_wins = df[df['isRadiant'] == 0]['win'].sum()

    win_data = pd.DataFrame({
        'Команда': ['Radiant', 'Dire'],
        'Победы': [radiant_wins, dire_wins]
    })
    fig = px.pie(win_data, names='Команда', values='Победы')
    logger.info("Круговая диаграмма побед команд успешно построена.")
    return fig


# Зависимость выбранной переменной на результат матча
def create_histogram_for_variable(df, variable):
    """Создание гистограммы, показывающей зависимость выбранной переменной на результат матча."""
    logger.info(f"Создание гистограммы для переменной: {variable}.")

    mapping = {0: 'Поражение', 1: 'Победа'}
    df['win'] = df['win'].replace(mapping)

    fig = px.histogram(
        df,
        x=variable,
        color='win',
        nbins=50,
        labels={variable: variable.capitalize(), 'win': 'Результат'},
        barmode='overlay',
        color_discrete_map={'Поражение': 'lightcoral', 'Победа': 'lightblue'},
        category_orders={'win': ['Поражение', 'Победа']}
    )
    fig.update_xaxes(title_text=variable.capitalize())
    fig.update_yaxes(title_text='Количество матчей')
    logger.info("Гистограмма зависимости успешно построена.")
    return fig


# Соотношение KDA и убийств в минуту
def create_kda_scatter_plot(df):
    """Создание диаграммы рассеяния, показывающей соотношение KDA и убийств в минуту."""
    logger.info("Создание диаграммы рассеяния KDA и убийств в минуту.")

    fig = px.scatter(
        df,
        x='kda',
        y='kills_per_min',
        color='win',
        labels={'kda': 'KDA', 'kills_per_min': 'Убийств в минуту', 'win': 'Результат'}
    )

    fig.update_layout(legend_title='Результат')
    logger.info("Диаграмма рассеяния успешно построена.")
    return fig
