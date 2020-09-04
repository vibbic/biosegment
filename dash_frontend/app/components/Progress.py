import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

progress = html.Div(
    [
        # dcc.Interval(id="progress-interval", n_intervals=0, interval=500),
        dbc.Progress(id="progress", value=100, striped=True, animated=False),
    ]
)