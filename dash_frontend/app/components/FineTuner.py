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

PREFIX = "finetuner"

create_callbacks(PREFIX)

finetuner_layout = dbc.Card(
    [
        html.H4("Fine-tune model", className="card-title"),
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
                dbc.Label("Epochs"),
                dcc.Input(
                    id=f"{PREFIX}-epochs",
                    value=10,
                    type="number",
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
        State(f"{PREFIX}-epochs", "value"),
        State("selected-dataset-name", "value"),
    ],
)
def start_model_retraining(n, new_model_name, selected_model_id, selected_annotation, epochs, selected_dataset_id):
    if n:
        body = {
            "title": new_model_name,
            # TODO create location on backend based on new name
            "location": f"biosegment/models/{new_model_name}/best_checkpoint.pytorch",
            # TODO use annotation_id from database
            "annotation_id": selected_annotation,
            "classes_of_interest": [0, 1, 2],
            "epochs": epochs,
        }
        # allow for training from scratch
        if selected_model_id:
            body["from_model_id"] = selected_model_id
        logging.debug(f"Start segmentation body: {body}")
        task_id = api.utils.train(json=body)
        return [{"task_id": task_id}]
    raise PreventUpdate
