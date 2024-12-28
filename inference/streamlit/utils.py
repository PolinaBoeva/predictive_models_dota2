# Функция для получения 10 самых популярных героев
def get_top_10_heroes(df):

    hero_counts = df["hero_name"].value_counts()
    top_10_heroes = hero_counts.head(10).index.tolist()

    return top_10_heroes
