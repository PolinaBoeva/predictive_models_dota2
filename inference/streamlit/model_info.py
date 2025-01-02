import streamlit as st
from client import ModelsAPIClient, DataAPIClient


def display_model_info(api_client):
    """Функция для отображения информации о модели."""
    st.header("Информация о модели")

    if st.button("Загрузить список обученных моделей"):
        model_list_response = api_client.get_model_list()
        if model_list_response:
            st.session_state.models = model_list_response[
                "models"
            ]  # Сохраняем список моделей
            st.write("Обученные модели:")
            st.write(st.session_state.models)

    model_id_input = st.selectbox("Выберите ID модели", st.session_state.models)

    if st.button("📖 Получить информацию о модели"):
        model_info = api_client.get_model_info(model_id_input)
        if model_info:
            st.write("📝 Информация о модели:")
            st.json(model_info)
        else:
            st.error("❌ Эта модель не существует.")

    if st.button("Активировать выбранную модель"):
        activate_response = api_client.activate_model(model_id_input)
        st.write("Модель активирована")
