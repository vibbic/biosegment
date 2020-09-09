import logging

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app.app import app
from app.DatasetStore import DatasetStore
from app.components.Viewer2D import Viewer2D
from app.components.SegmentationRunner import segmentation_runner_layout
from app.components.DatasetSelector import dataset_selector_layout
from app.components.Interests import interests_layout
from app import api

WORKER_ROOT_DATA_FOLDER="/home/brombaut/workspace/biosegment/data/"

# import app.api.base

title="Viewer 2D"
path="/viewer"

DEFAULT_DATASET="EMBL"
VIEWER_ID = "viewer-1"

viewer = Viewer2D(main_id="viewer-1", default_dataset=DEFAULT_DATASET)

layout = dbc.Container(
    children=[
        html.H1("Viewer 2D"),
        html.Hr(),
        dbc.Row(
            dbc.Col(dataset_selector_layout),
        ),
        dbc.Row([
            dbc.Col([
                dbc.Row(
                    [
                        dbc.Col(segmentation_runner_layout),
                        dbc.Col(interests_layout),
                    ]
            )]),
            dbc.Col(
                dbc.Card(
                [
                    html.H3(
                        "Viewer"
                    ),
                    html.P(
                        "Selected segmentation"
                    ),
                    html.Button('Update segmentation options', id='update-button-segmentations-options'),
                    dcc.Dropdown(
                        id="selected-segmentation-name",
                    ),
                    viewer.layout(),
                ], body=True),
                md=12,
                lg=6,
            )
        ]),
    # Store for user created masks
    # data is a list of dicts describing shapes
    dcc.Store(id="masks", data={"shapes": []}),
])

@app.callback(
    Output("selected-dataset-name", "options"),
    [
        Input('url', 'pathname')
    ]
)
def get_dataset_options(pathname):
    if pathname:
        try:
            datasets = DatasetStore.get_names_available()
            if not datasets:
                raise PreventUpdate
            options = [{
                "label": d["title"],
                "value": d["id"],
            } for d in datasets]
            logging.debug(f"Datasets options: {options}")
            return options
        except:
            raise PreventUpdate
    raise PreventUpdate

@app.callback(
    [
        Output("task-id", "data"),
    ],
    [
        Input("start-new-segmentation", "n_clicks"),
    ],
    [
        State("new-segmentation-name", "value"),
        State("selected-model-name", "value")
    ]
)
def start_segmentation(n, new_segmentation_name, selected_model):
    if n:
        assert selected_model
        body = {
            "title": new_segmentation_name,
            "model": selected_model,
            "model_id": 1,
            "dataset_id": 1,
            "data_dir": f"EM/EMBL/raw",
            "labels_dir": f"EM/EMBL/labels",
            "write_dir": f"segmentations/EMBL/{new_segmentation_name}",
            "input_size": [256,256],
            "classes_of_interest": [0, 1, 2]
        }
        logging.debug(f"Start segmentation body: {body}")
        # task_id = api.utils.infer(json=body)
        task_id = api.segmentation.create(json=body)
        # task_id = test_celery(json={"timeout": 10})
        return [{"task_id": task_id}]
    raise PreventUpdate

@app.callback(
    Output(f'{VIEWER_ID}-current-segmentation-name', 'data'),[
        Input('selected-segmentation-name', 'value')
])
def select_segmentation(value):
    return value

@app.callback([
    Output(f'selected-model-name', 'options'),
], [
    Input('selected-dataset-name', 'value'),
])
def change_model_options(name):
    if name:
        options = DatasetStore.get_dataset(name).get_models_available()
        return [options]
    raise PreventUpdate

@app.callback([
    Output(f'selected-segmentation-name', 'options'),
], [
    Input('selected-dataset-name', 'value'),
    Input('update-button-segmentations-options', 'n_clicks')
]
)
def change_segmentation_options(name, n_clicks):
    if name:
        options = DatasetStore.get_dataset(name).get_segmentations_available()
        return [options]
    raise PreventUpdate

@app.callback([
    Output(f'{VIEWER_ID}-current-dataset-name', 'data'),
    Output(f'{VIEWER_ID}-dataset-dimensions', 'data'),
], [
    Input('selected-dataset-name', 'value')
])
def change_dataset(name):
    if name:
        dataset = DatasetStore.get_dataset(name)
        return name, dataset.get_dimensions()
    raise PreventUpdate

if __name__ == '__main__':
    app.layout = layout
    app.run_server(debug=True)