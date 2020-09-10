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

PREFIX = "retrainer"

model_retrainer_layout = dbc.Card([
    html.H4(
        "Model retrainer",
        className="card-title"
    ),
    dbc.FormGroup(
        [
            dbc.Label("Selected model"),
            dcc.Dropdown(
                id=f"{PREFIX}-selected-model-name",
            ),
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("Selected annotation"),
            dcc.Dropdown(
                id=f"{PREFIX}-selected-annotation-name",
            ),
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("New model name"),
            dcc.Input(
                id=f"{PREFIX}-new-model-name",
                value="New model 1",
                type="text",
            ),
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Button(
                "Retrain model",
                id=f"{PREFIX}-start-retraining",
                color="primary", className="mr-1"
            ),
            task_progress,
        ]
    ),    
], body=True)

@app.callback([
    Output(f"{PREFIX}-selected-annotation-name", 'options'),
], [
    Input("selected-dataset-name", 'value'),
])
def change_annotation_options(name):
    if name:
        options = DatasetStore.get_dataset(name).get_annotations_available()
        return [options]
    raise PreventUpdate

@app.callback([
    Output(f'{PREFIX}-selected-model-name', 'options'),
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
        Output(f'{PREFIX}-task-id', "data"),
    ],
    [
        Input(f'{PREFIX}-start-retraining', "n_clicks"),
    ],
    [
        State(f'{PREFIX}-new-model-name', "value"),
        State(f'{PREFIX}-model-name', "value"),
        State(f'{PREFIX}-annotation-name', "value"),
    ]
)
def start_model_retraining(n, new_model_name, selected_model, selected_annotation):
    if n:
        assert selected_model
        body = {
            "title": new_model_name,
            "retrain_model": selected_model,
            "model_id": 1,
            "dataset_id": 1,
            "labels_dir": selected_annotation,
            "log_dir": f"segmentations/EMBL/{new_model_name}",
            "input_size": [512,512],
            "classes_of_interest": [0, 1, 2]
        }
        logging.debug(f"Start segmentation body: {body}")
        task_id = api.utils.train(json=body)
        return [{"task_id": task_id}]
    raise PreventUpdate