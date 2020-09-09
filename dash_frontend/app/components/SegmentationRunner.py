import logging

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from app.app import app
from dash.exceptions import PreventUpdate
from app.components.TaskProgress import task_progress
from app.DatasetStore import DatasetStore
from app import api
from app.shape_utils import annotations_to_png

segmentation_runner_layout = dbc.Card([
    html.H4(
        "Run segmentation",
        className="card-title"
    ),
    dbc.FormGroup(
        [
            dbc.Label("Selected model"),
            dcc.Dropdown(
                id="selected-model-name",
            ),
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("New segmentation name"),
            dcc.Input(
                id="new-segmentation-name",
                value="New segmentation 1",
                type="text",
            ),
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Button(
                "Start new segmentation",
                id="start-new-segmentation",
                color="primary", className="mr-1"
            ),
            task_progress,
        ]
    ),    
], body=True)

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

@app.callback(
    [
        Output("task-id", "data"),
    ],
    [
        Input("start-new-segmentation", "n_clicks"),
    ],
    [
        State("new-segmentation-name", "value"),
        State("selected-model-name", "value"),
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
            "input_size": [512,512],
            "classes_of_interest": [0, 1, 2]
        }
        logging.debug(f"Start segmentation body: {body}")
        task_id = api.utils.infer(json=body)
        task_id = api.segmentation.create(json=body)
        return [{"task_id": task_id}]
    raise PreventUpdate