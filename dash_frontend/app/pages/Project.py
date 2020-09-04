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

def get_comp_for_dataset(d):
    card = dbc.Card(
    dbc.CardBody(
        [
            html.H4(d['title'], className="card-title"),
            # html.H6("Card subtitle", className="card-subtitle"),
            html.P(
                d['description'],
                className="card-text",
            ),
            html.P(
                className="card-text",
            ),
            dbc.CardLink("Go to Dataset", href=f"http://localhost/dash/dataset/{d['id']}"),
        ]
    ),
    # style={"width": "18rem"},
    )
    return card

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
            dbc.CardLink("Datasets", href=f"http://localhost/dash/project/{p['id']}/datasets"),
            dbc.Row(
                [get_comp_for_dataset(d) for d in p['datasets']]
            )
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
    datasets = api.dataset.get_multi_for_project(project_id, token=token)
    project['datasets'] = datasets
    return [get_card_for_project(project)]

if __name__ == '__main__':
    app.layout = layout
    app.run_server(debug=True)