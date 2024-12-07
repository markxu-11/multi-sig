import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from typing import Tuple


def create_plot(data: np.ndarray, 
                y_range: Tuple[float, float]=None, 
                title=None, x_title=None, y_title=None) -> go.Figure:
    """Creates a multichannel signal plot using Plotly

    Args:
        data (np.ndarray): Multichannel signal data
        y_range (Tuple[float, float], optional): Range of y axis. Defaults to None.
        title (optional): Title of the plot. Defaults to None.
        x_title (optional): x axis title. Defaults to None.
        y_title (optional): y axis title. Defaults to None.

    Raises:
        ValueError: Error when dimension of data is not 1 or 2

    Returns:
        go.Figure: Plotly Figure object
    """
    if data.ndim == 2:
        nrows = len(data)
        fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True)
        for i, row in enumerate(data, start=1):
            fig.add_trace(go.Scatter(y=row, name=f"Ch{i-1}"), row=i, col=1)
    elif data.ndim == 1:
        fig = go.Figure().add_trace(go.Scatter(y=data, name=f"Ch{1}"))
    else:
        raise ValueError("Data dimension is not correct")
    
    if y_range:
        fig.update_yaxes(range=y_range)
    fig.update_layout(template="simple_white", title=title,
                      xaxis_title=x_title, yaxis_title=y_title)

    return fig