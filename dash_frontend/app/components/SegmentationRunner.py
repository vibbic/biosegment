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

PREFIX = "segmentation"

create_callbacks(PREFIX)

segmentation_runner_layout = dbc.Card(
    [
        html.H4("Run segmentation", className="card-title"),
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
                dbc.Label("New segmentation name"),
                dcc.Input(
                    id=f"{PREFIX}-new-segmentation-name",
                    value="New segmentation 1",
                    type="text",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Button(
                    "Start new segmentation",
                    id=f"{PREFIX}-start-new-segmentation",
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
    [Input(f"{PREFIX}-start-new-segmentation", "n_clicks"),],
    [
        State(f"{PREFIX}-new-segmentation-name", "value"),
        State(f"{PREFIX}-selected-model-name", "value"),
        State("selected-dataset-name", "value"),
    ],
)
def start_segmentation(n, new_segmentation_name, selected_model, selected_dataset):
    if n:
        assert selected_model
        dataset_title = (
            DatasetStore.getInstance().get_dataset(selected_dataset).get_title()
        )
        body = {
            "title": new_segmentation_name,
            "location": f"segmentations/{dataset_title}/{new_segmentation_name}",
            "model_id": selected_model,
            "dataset_id": selected_dataset,
        }
        logging.debug(f"Start segmentation body: {body}")
        task_id = api.utils.infer(json=body)
        return [{"task_id": task_id}]
    raise PreventUpdate
