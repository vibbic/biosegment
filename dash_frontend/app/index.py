import logging

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from app.app import app
from app.env import DEV_MODE
from app.pages import Dataset, Datasets, Project, Projects, Viewer2DPage
from dash.dependencies import Input, Output

logging.basicConfig(
    filename="all.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)

# all pages for in navbar (no details)
navigation_pages = [
    Projects,
    Datasets,
    Viewer2DPage,
]

# details
details = [
    Project,
    Dataset,
]

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink(p.title, href=app.get_relative_path(p.path)))
        for p in navigation_pages
    ],
    brand="BioSegment",
    brand_href="#",
    color="primary",
    dark=True,
)

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        navbar,
        dbc.Container(html.Div(id="page-content"), fluid=True),
    ]
)


@app.callback(Output("page-content", "children"), [Input("url", "pathname"),])
def display_page(pathname):
    current_page = None
    for page in navigation_pages:
        if app.get_relative_path(page.path) == pathname:
            current_page = page
    if current_page is None:
        for page in details:
            if app.get_relative_path(page.path) in pathname:
                current_page = page
    if current_page is None:
        return "404"
    return current_page.layout


app.enable_dev_tools(debug=DEV_MODE)

server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)
