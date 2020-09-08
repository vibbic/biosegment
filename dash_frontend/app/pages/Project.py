import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app.app import app
from app import api

title="Project"
path="/projects"

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
            dbc.CardLink("Go to Dataset", href=f"http://localhost/dash/datasets/{d['id']}"),
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
            html.H3(p['title'], className="card-title"),
            # html.H6("Card subtitle", className="card-subtitle"),
            html.P(
                p['description'],
                className="card-text",
            ),
            html.H4("Add dataset"),
            dcc.Input(
                id="new-dataset-name",
                type="text",
                value="New dataset name"
            ),
            dcc.Input(
                id="new-dataset-location",
                type="text",
                value="New dataset location"
            ),
            dbc.Button(
                "Add new dataset",
                id="add-new-dataset",
            ),
            html.H3(
                dbc.CardLink("Datasets", href=f"http://localhost/dash/projects/{p['id']}/datasets"),
            ),
            dbc.Row(
                [get_comp_for_dataset(d) for d in p['datasets']]
            )
        ]
    ),
    # style={"width": "18rem"},
    )
    return card

@app.callback(
    # dummy output, not needed
    Output('new-dataset-name', 'value'),
[
    Input('add-new-dataset', 'n_clicks'),
], [
    State('url', 'pathname'),
    State('new-dataset-name', 'value'),
    State('new-dataset-location', 'value'),
])
def add_new_dataset(n_clicks, pathname, new_dataset_name, new_dataset_location):
    if n_clicks:
        project_id = pathname.split('/')[-1]
        project = api.project.get(project_id)
        dataset_in = {
            "title": new_dataset_name,
            "location": new_dataset_location
        }
        api.dataset.create_for_project(project_id, json=dataset_in)
        return new_dataset_name
    raise PreventUpdate

@app.callback([
    Output('project', 'children'),
], [
    Input('update-button-project', 'n_clicks'),
], [
    State('url', 'pathname'),
])
def update_project(clicks, pathname):
    project_id = pathname.split('/')[-1]
    project = api.project.get(project_id)
    datasets = api.dataset.get_multi_for_project(project_id)
    project['datasets'] = datasets
    return [get_card_for_project(project)]

if __name__ == '__main__':
    app.layout = layout
    app.run_server(debug=True)