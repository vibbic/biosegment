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
        id="text-token",
    )
])

@app.callback(
    Output('text-token', 'children'),
    [
        Input('token', 'data')
]
)
def display_page(token):
    return [token]  