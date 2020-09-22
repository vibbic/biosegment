import logging

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from app import api
from app.app import app
from app.components.TaskProgress import create_callbacks, create_layout
from app.DatasetStore import DatasetStore
from app.layout_utils import dropdown_with_button
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

PREFIX = "retrainer"

create_callbacks(PREFIX)

model_retrainer_layout = dbc.Card(
    [
        html.H4("Model retrainer", className="card-title"),
        dbc.FormGroup(
            [
                dbc.Label("Selected model"),
                dropdown_with_button(
                    dropdown_id=f"{PREFIX}-selected-model-name",
                    button_id=f"{PREFIX}-refresh-selected-model-name",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Selected annotation"),
                dropdown_with_button(
                    dropdown_id=f"{PREFIX}-selected-annotation-name",
                    button_id=f"{PREFIX}-refresh-selected-annotation-name",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("New model name"),
                dcc.Input(
                    id=f"{PREFIX}-new-model-name", value="New model 1", type="text",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Button(
                    "Retrain model",
                    id=f"{PREFIX}-start-retraining",
                    color="primary",
                    className="mr-1",
                ),
                create_layout(PREFIX),
            ]
        ),
    ],
    body=True,
)


@app.callback(
    Output(f"{PREFIX}-start-new-segmentation", "disabled"),
    [Input(f"{PREFIX}-progress", "animated"),],
)
def change_state_button(animated):
    return animated


@app.callback(
    [Output(f"{PREFIX}-selected-annotation-name", "options"),],
    [
        Input("selected-dataset-name", "value"),
        Input(f"{PREFIX}-refresh-selected-annotation-name", "n_clicks"),
    ],
)
def change_annotation_options(name, n_clicks):
    if name:
        options = DatasetStore.get_dataset(name).get_annotations_available()
        return [options]
    raise PreventUpdate


@app.callback(
    [Output(f"{PREFIX}-selected-model-name", "options"),],
    [
        Input("selected-dataset-name", "value"),
        Input(f"{PREFIX}-refresh-selected-model-name", "n_clicks"),
    ],
)
def change_model_options(name, n_clicks):
    if name:
        options = DatasetStore.get_dataset(name).get_models_available()
        return [options]
    raise PreventUpdate


@app.callback(
    [Output(f"{PREFIX}-task-id", "data"),],
    [Input(f"{PREFIX}-start-retraining", "n_clicks"),],
    [
        State(f"{PREFIX}-new-model-name", "value"),
        State(f"{PREFIX}-selected-model-name", "value"),
        State(f"{PREFIX}-selected-annotation-name", "value"),
        State("selected-dataset-name", "value"),
    ],
)
def start_model_retraining(n, new_model_name, selected_model_id, selected_annotation, selected_dataset_id):
    if n:
        assert selected_model_id
        body = {
            "title": new_model_name,
            "from_model_id": selected_model_id,
            "dataset_id": selected_dataset_id,
            # TODO use annotation_id from database
            "annotation": selected_annotation,
            "log_dir": f"segmentations/EMBL/{new_model_name}",
            "input_size": [512, 512],
            "classes_of_interest": [0, 1, 2],
        }
        logging.debug(f"Start segmentation body: {body}")
        task_id = api.utils.train(json=body)
        return [{"task_id": task_id}]
    raise PreventUpdate
