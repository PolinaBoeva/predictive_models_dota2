import streamlit as st


# Функция для получения 10 самых популярных героев
def get_top_10_heroes(df):

    hero_counts = df["hero_name"].value_counts()
    top_10_heroes = hero_counts.head(10).index.tolist()

    return top_10_heroes


def get_ridge_params():
    """Функция для ввода гиперпараметров Ridge Classifier."""
    hyperparameters = {}
    hyperparameters["alpha"] = st.number_input("Alpha", value=1.0, min_value=0.0)
    hyperparameters["fit_intercept"] = st.checkbox("Fit Intercept", value=True)
    return hyperparameters


def get_catboost_params():
    """Функция для ввода гиперпараметров CatBoost Classifier."""
    hyperparameters = {}
    hyperparameters["learning_rate"] = st.number_input(
        "Learning Rate", value=0.1, min_value=0.01, max_value=1.0
    )
    hyperparameters["depth"] = st.slider("Depth", min_value=1, max_value=16, value=6)
    hyperparameters["iterations"] = st.number_input(
        "Iterations", value=100, min_value=1
    )
    hyperparameters["l2_leaf_reg"] = st.number_input(
        "L2 Leaf Regularization", value=3, min_value=1, max_value=10
    )
    return hyperparameters


def clean_columns(df, columns):
    """Функция для преобразования типов данных для отображения в Streamlit."""
    for column in columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.replace(",", "")
            .str.replace(".0", "", regex=False)
        )
    return df
