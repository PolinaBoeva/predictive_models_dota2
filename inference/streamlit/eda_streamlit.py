import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
        #df['match_id'] = df['match_id'].astype('object')
        #df['account_id'] = df['account_id'].astype('object')

        # Основная информация о датасете
        def display_dataset_info(df):
            st.write("### Основная информация о данных")
            with st.expander("### Посмотреть основную информацию"):
                st.dataframe(df)
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
                selected_column = columns_to_plot[selected_metric]

                sorted_data = player_data.sort_values(by='match_id')

                fig12 = px.histogram(
                    sorted_data,
                    x='match_id',
                    y=selected_column,
                    labels={
                        'match_id': 'Номер матча',
                        selected_column: selected_metric
                    },
                    nbins=len(player_data['match_id'].unique()),
                )

                fig12.update_layout(
                    bargap=0,
                    xaxis_title='Номер матча',
                    yaxis_title=selected_metric
                )
                st.plotly_chart(fig12)



                st.write("#### Распределение исходов матчей для данного игрока")
                result_table = pd.DataFrame({
                    'Результат': ['Победы', 'Поражения'],
                    'Количество': [player_data['win'].sum(), len(player_data) - player_data['win'].sum()]
                })

                fig13 = px.pie(result_table, names='Результат', values='Количество',
                             color_discrete_sequence=['#FFB6C1', '#A3D9FF'],
                             hole=0)
                st.plotly_chart(fig13)

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

                st.write("#### Количество убийств по игрокам (в разрезе героев):")
                fig_kills = px.bar(
                    df_players,
                    x='account_id',
                    y='kills',
                    color='hero_name',
                    labels={'account_id': 'ID игрока', 'kills': 'Убийства', 'hero_name': 'Имя героя'}
                )
                fig_kills.update_layout(
                    xaxis={'categoryorder': 'category ascending'},
                    bargap=0,
                    bargroupgap=0
                )
                st.plotly_chart(fig_kills)

                st.write("#### Смерти по игрокам (в разрезе героев):")
                fig_deaths = px.bar(df_players, x='account_id', y='deaths', color='hero_name',
                                    labels={'account_id': 'ID игрока', 'deaths': 'Смерти', 'hero_name': 'Имя героя'})
                st.plotly_chart(fig_deaths)

        # Построение графиков по историческим данным
        def display_graphics(df):
            st.write("### Общая аналитика исторических данных")
            with st.expander("### Посмотреть информацию"):
                hero_counts = df["hero_name"].value_counts()
                top_10_heroes = hero_counts.head(10).index.tolist()

                st.write("#### Самые популярные герои")
                df_top_10 = hero_counts[top_10_heroes].reset_index()
                df_top_10.columns = ['hero_name', 'count']
                df_top_10 = df_top_10.sort_values(by='count', ascending=False)
                fig10 = px.bar(df_top_10, x='hero_name', y='count', color='hero_name',
                               labels={'hero_name': 'Имя героя', 'count': 'Количество'})
                st.plotly_chart(fig10)

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

                st.write("#### Распределение убийств по героям")
                selected_heroes = st.multiselect("Выберите героев для отображения (по умолчанию - 10 самых популярных)",
                                                 options=hero_counts.index.tolist(),
                                                 default=top_10_heroes)

                filtered_df = df[df["hero_name"].isin(selected_heroes)]

                fig1 = px.histogram(filtered_df, x="hero_name", y="kills", color="hero_name",
                                    labels={"hero_name": "Имя героя", "kills": "Количество убийств"},
                                    height=600)

                fig1.update_layout(xaxis_title="Имя героя", yaxis_title="Количество убийств")
                st.plotly_chart(fig1)

                st.write("#### Box-plot распределения убийств по выбранным героям")
                fig2 = px.box(filtered_df, x="hero_name", y="kills", color="hero_name",
                              labels={"hero_name": "Имя героя", "kills": "Количество убийств"},
                              height=600)
                fig2.update_layout(xaxis_title="Имя героя", yaxis_title="Количество убийств")
                st.plotly_chart(fig2)

                st.write("#### Топ-10 признаков сильнее всего коррелирующие с результатом матча")
                numeric_df = df.select_dtypes(include=[np.number])
                correlation_matrix = numeric_df.corr()
                correlation_with_win = correlation_matrix['win']

                sorted_correlations = correlation_with_win.abs().sort_values(ascending=False)
                top_10_features = sorted_correlations.index[:10]

                top_10_corr_matrix = correlation_matrix.loc[top_10_features, top_10_features]

                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(
                    top_10_corr_matrix,
                    annot=True,
                    fmt=".2f",
                    cmap="viridis",
                    cbar=True,
                    ax=ax
                )
                st.pyplot(fig)

                st.write("#### Победы команд Radiant vs Dire")
                df['winrate'] = df['win'].mean()
                radiant_wins = df[df['isRadiant'] == 1]['win'].sum()
                dire_wins = df[df['isRadiant'] == 0]['win'].sum()

                win_data = pd.DataFrame({
                    'Команда': ['Radiant', 'Dire'],
                    'Победы': [radiant_wins, dire_wins]
                })
                fig_pie = px.pie(win_data, names='Команда', values='Победы')
                st.plotly_chart(fig_pie)

                st.write("#### Зависимость длительности матча на результат")
                mapping = {0: 'Поражение', 1: 'Победа'}
                df['win'] = df['win'].replace(mapping)

                fig6 = px.histogram(
                    df,
                    x='duration',
                    color='win',
                    nbins=50,
                    labels={'duration': 'Длительность матча', 'win': 'Результат'},
                    barmode='overlay',
                    color_discrete_map={'Поражение': 'lightcoral', 'Победа': 'lightblue'},
                    category_orders={'win': ['Поражение', 'Победа']}
                )
                fig6.update_xaxes(title_text='Длительность матча')
                fig6.update_yaxes(title_text='Количество матчей')
                st.plotly_chart(fig6)

                st.write("#### Соотношение KDA и убийств в минуту")
                fig3 = px.scatter(df, x='kda', y='kills_per_min', color='win',
                                  labels={'kda': 'KDA', 'kills_per_min': 'Убийств в минуту', 'win': 'Результат'})
                fig3.update_layout(legend_title='Результат')
                st.plotly_chart(fig3)

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