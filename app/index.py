import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from app import app
from pages import (
    Viewer2D
)

pages = [
    Viewer2D
]

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "backgroundColor": "#F8F9FA",
}

CONTENT_STYLE = {
    "marginLeft": "18rem",
    "marginRight": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("BioSegment", className="display-5"),
        html.Hr(),
        html.P(
            "A segmentation viewer", className="lead"
        ),
        dbc.Nav(
            [dbc.NavLink(p.title, href=p.path, id=f"{p.title}-link") for p in pages],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    sidebar,
    html.Div(id='page-content', style=CONTENT_STYLE)
])

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    current_page = None
    for page in pages:
        if pathname == page.path:
            current_page = page
    if current_page is None:
        return "404"
    return current_page.layout    

if __name__ == '__main__':
    app.run_server(debug=True)