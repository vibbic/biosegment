import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app.app import app

title="API Token"
path="/token"

layout = html.Div([
    html.P(
        children="Token",
    ),
    html.P(
        id="token",
        # children=str(tokens),
    )
])

