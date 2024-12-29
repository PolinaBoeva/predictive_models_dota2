import streamlit as st
import pandas as pd

from utils import get_top_10_heroes
from plots import (plot_metric_histogram,
                   create_result_pie_chart,
                   create_distribution_plot,
                   create_top_10_heroes_plot,
                   create_selected_heroes_plot,
                   create_box_plot,
                   create_winrate_pie_chart,
                   create_histogram_for_variable,
                   create_kda_scatter_plot)

# Основная информация о датасете
def display_dataset_info(df):
    st.write("### Основная информация о данных")
    with st.expander("### Посмотреть основную информацию"):
        st.dataframe(df.head(2000))
        st.write(f"**Размер датасета:** {df.shape[0]} строк, {df.shape[1]} столбцов")

        st.write("#### Типы данных:")
        st.write(df.dtypes)

        st.write("#### Описательная статистика:")
        st.write("Числовые значения:")
        st.table(df.describe())

        st.write("Категориальные значения:")
        st.table(df.describe(include='object'))

# Статистика по выбранному игроку
def display_selected_player_info(df):
    st.write("### Статистика по выбранному игроку")
    with st.expander("### Посмотреть статистику по игроку"):
        match_count = df.groupby('account_id')['match_id'].count()
        most_matches_account_id = match_count.idxmax()

        account_ids = sorted(df['account_id'].unique())
        default_index = account_ids.index(most_matches_account_id)

        selected_account_id = st.selectbox(
            "Пожалуйста, выберите account_id игрока, по которому хотите посмотреть статистику:",
            account_ids,
            index=default_index  # игрок с наибольшим количеством матчей по умолчанию
        )
        player_data = df[df['account_id'] == selected_account_id]

        st.write("##### Герои игрока и статистика по ним:")
        if player_data.empty:
            st.write(
                f"Данные для игрока с account_id {selected_account_id} отсутствуют. Отображение медианных значений.")
            median_stats = df.median()
            st.write(median_stats)
        else:
            hero_stats = player_data.groupby('hero_name').agg({
                'kills': 'sum',
                'assists': 'sum',
                'deaths': 'sum',
                'gold_per_min': 'mean',
                'xp_per_min': 'mean',
                'hero_damage': 'sum',
                'win': 'mean'
            })
            st.write(hero_stats.T)

        st.write("#### Гистограммы распределения значений признаков по матчам")
        columns_to_plot = {
            'Количество убийств': 'kills',
            'Количество смертей': 'deaths',
            'Количество ассистов': 'assists',
            'Золото в минуту': 'gold_per_min',
            'Опыт в минуту': 'xp_per_min'
        }
        selected_metric = st.selectbox("Выберите метрику для отображения", options=list(columns_to_plot.keys()))
        fig = plot_metric_histogram(player_data, columns_to_plot, selected_metric)
        st.plotly_chart(fig)

        st.write("#### Распределение исходов матчей для данного игрока")
        fig = create_result_pie_chart(player_data)
        st.plotly_chart(fig)

# Статистика по выбранному матчу
def display_selected_player_match(df):
    st.write("### Статистика по выбранному матчу")
    with st.expander("### Посмотреть статистику по матчу"):
        max_players_match_id = str(df['match_id'].value_counts().idxmax())

        match_ids = sorted([str(match_id) for match_id in df['match_id'].unique()])

        default_index = match_ids.index(max_players_match_id)

        selected_match_id = st.selectbox(
            "Пожалуйста, выберите match_id, по которому хотите посмотреть статистику:",
            match_ids,
            index=default_index)

        match_data = df[df['match_id'] == int(selected_match_id)]

        # Какая команда победила
        radiant_win = match_data['isRadiant'].iloc[0] == 1 and match_data['win'].iloc[0] == 1
        winning_team = "Radiant" if radiant_win else "Dire"
        st.write(f"### Победившая команда: *{winning_team}*")

        st.write("#### Игроки, участвовавшие в матче:")
        players_info = match_data[['account_id', 'isRadiant', 'hero_name']]
        players_info['Команда'] = players_info['isRadiant'].apply(lambda x: 'Radiant' if x == 1 else 'Dire')
        st.write(players_info[['Команда', 'account_id', 'hero_name']].sort_values(by='Команда').reset_index(
            drop=True))

        st.write("#### Статистика по каждому игроку из матча:")
        stats_to_show = [
            'win', 'kills', 'deaths', 'assists', 'hero_damage', 'hero_healing',
            'gold_per_min', 'net_worth', 'xp_per_min'
        ]
        df_players = df[df['account_id'].isin(players_info['account_id'])]
        detailed_stats = df_players.groupby('account_id')[stats_to_show].mean().reset_index()
        st.write(detailed_stats)

        st.write("#### Поробная статистика по игроку из матча:")
        account_id = st.selectbox("Выберите account_id игрока:", players_info['account_id'])
        df_player = df_players[df_players['account_id'] == account_id]

        variable = st.selectbox("Выберите переменную для графика:",
                                ['hero_name', 'win', 'kills', 'deaths', 'assists',
                                 'hero_damage', 'hero_healing', 'gold_per_min',
                                 'net_worth', 'xp_per_min'])

        fig = create_distribution_plot(df_player, variable)
        st.plotly_chart(fig)

# Построение графиков по историческим данным
def display_graphics(df):
    st.write("### Общая аналитика исторических данных")
    with st.expander("### Посмотреть информацию"):
        top_10_heroes = get_top_10_heroes(df)

        st.write("#### ТОП-10 самых популярных героев")
        fig = create_top_10_heroes_plot(df, top_10_heroes)
        st.plotly_chart(fig)

        st.write("#### Статистика по топ-10 популярным героям")
        df_top_10_stats = df[df['hero_name'].isin(top_10_heroes)]

        important_stats = df_top_10_stats.groupby('hero_name').agg({
            'kills': 'mean',
            'deaths': 'mean',
            'assists': 'mean',
            'hero_damage': 'mean',
            'hero_healing': 'mean',
            'gold_per_min': 'mean',
            'net_worth': 'mean'
        }).reset_index()
        st.write(important_stats)

        st.write("#### Распределение выбранной переменной по героям")
        selected_heroes = st.multiselect("Выберите героев для отображения (по умолчанию - 10 самых популярных)",
                                         options=df['hero_name'].tolist(),
                                         default=top_10_heroes)

        filtered_df = df[df["hero_name"].isin(selected_heroes)]

        attribute = st.selectbox(
            "Выберите переменную для отображения:",
            options=['kills', 'deaths', 'assists', 'hero_damage', 'hero_healing', 'gold_per_min', 'net_worth',
                     'xp_per_min'])

        fig = create_selected_heroes_plot(filtered_df, attribute)
        st.plotly_chart(fig)

        st.write("#### Box-plot распределения переменной по выбранным героям")
        fig = create_box_plot(filtered_df, attribute)
        st.plotly_chart(fig)


        st.write("#### Победы команд Radiant vs Dire")
        fig = create_winrate_pie_chart(df)
        st.plotly_chart(fig)

        st.write("#### Зависимость различных переменных на результат матча")
        variables = [
            'kills',
            'deaths',
            'gold_per_min',
            'xp_per_min',
            'level',
            'duration'
        ]
        selected_variable = st.selectbox("Выберите переменную для анализа:", variables)
        st.write("Зависимость {} на результат".format(selected_variable))
        fig = create_histogram_for_variable(df, selected_variable)
        st.plotly_chart(fig)

        st.write("#### Соотношение KDA и убийств в минуту")
        fig = create_kda_scatter_plot(df)
        st.plotly_chart(fig)

def run_eda_streamlit():
    st.title("Предсказательные модели для Dota 2")
    st.write("EDA по историческим данным матчей Dota 2")
    image_url = "https://i.ibb.co/sjX0ntw/19459.webp"
    st.image(image_url)
    st.write('---')

    st.header("EDA")
    st.subheader("Загрузка данных")

    upload_file = st.file_uploader("Загрузите CSV-файл", type=['csv'])

    if upload_file is not None:

        df = pd.read_csv(upload_file)

        display_dataset_info(df)
        st.write('---')
        display_selected_player_info(df)
        st.write('---')
        display_selected_player_match(df)
        st.write('---')
        display_graphics(df)

    else:
        st.write("Пожалуйста, загрузите файл с историческими данными")

if __name__ == "__main__":
    run_eda_streamlit()