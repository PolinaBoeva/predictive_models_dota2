import streamlit as st

def get_ridge_params(params):
    """Функция для ввода гиперпараметров Ridge Classifier."""
    params["alpha"] = st.number_input("Alpha", value=1.0, min_value=0.0)
    params["fit_intercept"] = st.checkbox("Fit Intercept", value=True)

def get_catboost_params(params):
    """Функция для ввода гиперпараметров CatBoost Classifier."""
    params["learning_rate"] = st.number_input("Learning Rate", value=0.1, min_value=0.01, max_value=1.0)
    params["depth"] = st.slider("Depth", min_value=1, max_value=16, value=6)
    params["iterations"] = st.number_input("Iterations", value=100, min_value=1)
    params["l2_leaf_reg"] = st.number_input("L2 Leaf Regularization", value=3, min_value=1, max_value=10)