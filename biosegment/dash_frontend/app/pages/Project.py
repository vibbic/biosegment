import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from app.app import app
from app import api

title="Project"
path="/project"

layout = html.Div([
    html.P(
        children="Project",
    ),
    html.Button('Update', id='update-button-project'),
    html.Div(
        id="project",
    )
])

def get_card_for_project(p):
    # print(p)
    card = dbc.Card(
    dbc.CardBody(
        [
            html.H4(p['title'], className="card-title"),
            # html.H6("Card subtitle", className="card-subtitle"),
            html.P(
                p['description'],
                className="card-text",
            ),
            dbc.CardLink("Card link", href="#"),
            dbc.CardLink("External link", href="https://google.com"),
        ]
    ),
    # style={"width": "18rem"},
    )
    return card

@app.callback([
    Output('project', 'children'),
], [
    Input('url', 'pathname'),
    Input('token', 'data'),
    Input('update-button-project', 'n_clicks'),
])
def update_project(pathname, token, clicks):
    project_id = pathname.split('/')[-1]
    project = api.project.get(project_id, token=token)
    return [get_card_for_project(project)]

if __name__ == '__main__':
    app.layout = layout
    app.run_server(debug=True)