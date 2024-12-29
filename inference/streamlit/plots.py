import pandas as pd
import plotly.express as px

# Гистограммы распределения значений признаков по матчам
def plot_metric_histogram(data, columns_to_plot, selected_metric):
    selected_column = columns_to_plot[selected_metric]
    sorted_data = data.sort_values(by='match_id')

    fig = px.histogram(
        sorted_data,
        x='match_id',
        y=selected_column,
        labels={'match_id': 'Номер матча',selected_column: selected_metric},
        nbins=len(data['match_id'].unique()),
    )

    fig.update_layout(
        bargap=0,
        xaxis_title='Номер матча',
        yaxis_title=selected_metric
    )
    return fig


# Распределение исходов матчей для данного игрока
def create_result_pie_chart(player_data):
    result_table = pd.DataFrame({
           'Результат': ['Победы', 'Поражения'],
           'Количество': [player_data['win'].sum(), len(player_data) - player_data['win'].sum()]})
    fig = px.pie(result_table, names='Результат', values='Количество',
                    color_discrete_sequence=['#FFB6C1', '#A3D9FF'],
                    hole=0)
    return fig


# Персонажи, за которых играет выбранный игрок
def create_hero_name_distribution_plot(df_player):
    fig = px.histogram(df_player, x='hero_name', title='Частота выбора героев игроком')
    return fig

# Распределние переменной для выбраного игрока
def create_distribution_plot(df_player, variable):
    if variable in df_player.columns:
        fig = px.histogram(df_player, x=variable, title=f'Распределение {variable} по игроку')
        return fig
    else:
        raise ValueError(f"Переменная {variable} не найдена в данных.")


# топ-10 популярных героев
def create_top_10_heroes_plot(df, top_10_heroes):
    hero_counts = df["hero_name"].value_counts()
    df_top_10 = hero_counts[top_10_heroes].reset_index()
    df_top_10.columns = ['hero_name', 'count']
    df_top_10 = df_top_10.sort_values(by='count', ascending=False)

    fig = px.bar(df_top_10, x='hero_name', y='count', color='hero_name',
                 labels={'hero_name': 'Имя героя', 'count': 'Число вхождений'})
    return fig


# Распределение убийств по героям
def create_selected_heroes_plot(filtered_df, attribute):
    fig = px.histogram(
        filtered_df,
        x="hero_name",
        y=attribute,
        color="hero_name",
        labels={"hero_name": "Имя героя", attribute: attribute.capitalize()},
        height=600
    )
    fig.update_layout(xaxis_title="Имя героя", yaxis_title=attribute.capitalize())
    return fig


# Box-plot распределения убийств по выбранным героям
def create_box_plot(filtered_df, attribute):
    fig = px.box(
        filtered_df,
        x="hero_name",
        y=attribute,
        color="hero_name",
        labels={"hero_name": "Имя героя", attribute: attribute.capitalize()},
        height=600
    )
    fig.update_layout(xaxis_title="Имя героя", yaxis_title=attribute.capitalize())
    return fig


# Победы команд Radiant vs Dire
def create_winrate_pie_chart(df):
    df['winrate'] = df['win'].mean()
    radiant_wins = df[df['isRadiant'] == 1]['win'].sum()
    dire_wins = df[df['isRadiant'] == 0]['win'].sum()

    win_data = pd.DataFrame({
        'Команда': ['Radiant', 'Dire'],
        'Победы': [radiant_wins, dire_wins]
    })
    fig = px.pie(win_data, names='Команда', values='Победы')
    return fig

# Зависимость выбранной переменной на результат матча
def create_histogram_for_variable(df, variable):
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
    return fig


# Соотношение KDA и убийств в минуту
def create_kda_scatter_plot(df):
    fig = px.scatter(
        df,
        x='kda',
        y='kills_per_min',
        color='win',
        labels={'kda': 'KDA', 'kills_per_min': 'Убийств в минуту', 'win': 'Результат'}
    )

    fig.update_layout(legend_title='Результат')
    return fig