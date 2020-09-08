import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from app.app import app
from app import api

title="Dataset"
path="/datasets"

layout = html.Div([
    html.P(
        children="Dataset",
    ),
    html.Button('Update', id='update-dataset-button'),
    html.Div(
        id="dataset",
    )
])

def get_card_for_dataset(p):
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
    Output('dataset', 'children'),
], [
    Input('url', 'pathname'),
    Input('update-dataset-button', 'n_clicks'),
])
def update_datasets(pathname, clicks):
    dataset_id = pathname.split('/')[-1]
    try: 
        # prevent GET /api/v1/datasets/viewer when browsing
        int(dataset_id)
    except ValueError:
        raise PreventUpdate
    dataset = api.dataset.get(id=dataset_id)
    return [get_card_for_dataset(dataset)]

if __name__ == '__main__':
    app.layout = layout
    app.run_server(debug=True)