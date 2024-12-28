import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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