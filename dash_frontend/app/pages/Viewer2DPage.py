import logging

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app.app import app
from app.DatasetStore import DatasetStore
from app.components.Viewer2D import Viewer2D
from app.components.TaskProgress import task_progress
from app.api.utils import infer, poll_task, test_celery
from app.api.dataset import get_multi

WORKER_ROOT_DATA_FOLDER="/home/brombaut/workspace/biosegment/data/"

# import app.api.base

title="Viewer 2D"
path="/viewer"

DEFAULT_DATASET="EMBL"
VIEWER_ID = "viewer-1"

viewer = Viewer2D(main_id="viewer-1", default_dataset=DEFAULT_DATASET)

layout = html.Div([
    html.P(
        "Selected dataset"
    ),
    dcc.Dropdown(
        id="selected-dataset-name",
    ),
    html.H3(
        "Run segmentation"
    ),
    html.P(
        "Selected model"
    ),
    dcc.Dropdown(
        id="selected-model-name",
    ),
    html.P(
        "New segmentation name"
    ),
    dcc.Input(
        id="new-segmentation-name",
        value="New segmentation 1",
        type="text",
    ),
    dbc.Button(
        "Start new segmentation",
        id="start-new-segmentation",
        color="primary", className="mr-1"
    ),
    task_progress,
    html.H3(
        "Viewer"
    ),
    html.P(
        "Selected segmentation"
    ),
    dcc.Dropdown(
        id="selected-segmentation-name",
    ),
    viewer.layout(),
])

@app.callback(
    Output("selected-dataset-name", "options"),
    [
        Input('token', 'data')
    ]
)
def get_dataset_options(token):
    if token:
        try:
            datasets = DatasetStore.get_names_available(token=token)
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
        State("selected-model-name", "value"),
        State('token', 'data')
    ]
)
def start_segmentation(n, new_segmentation_name, selected_model, token):
    if n:
        assert selected_model
        body = {
            "model": selected_model,
            "data_dir": f"EM/EMBL/raw",
            "labels_dir": f"EM/EMBL/labels",
            "write_dir": f"segmentations/EMBL/{new_segmentation_name}",
            "input_size": [256,256],
            "classes_of_interest": [0, 1, 2]
        }
        logging.debug(f"Start segmentation body: {body}")
        task_id = infer(token=token, json=body)
        # task_id = test_celery(token=token, json={"timeout": 10})
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
    Input('selected-dataset-name', 'value')
], [
    State('token', 'data')
]
)
def change_model_options(name, token):
    if name:
        options = DatasetStore.get_dataset(name, token).get_models_available(token)
        return [options]
    raise PreventUpdate

@app.callback([
    Output(f'selected-segmentation-name', 'options'),
], [
    Input('selected-dataset-name', 'value')
], [
    State('token', 'data')
]
)
def change_segmentation_options(name, token):
    if name:
        options = DatasetStore.get_dataset(name, token).get_segmentations_available(token)
        return [options]
    raise PreventUpdate

@app.callback([
    Output(f'{VIEWER_ID}-current-dataset-name', 'data'),
    Output(f'{VIEWER_ID}-dataset-dimensions', 'data'),
], [
    Input('selected-dataset-name', 'value')
], [State('token', 'data')]
)
def change_dataset(name, token):
    if name:
        dataset = DatasetStore.get_dataset(name, token)
        return name, dataset.get_dimensions()
    raise PreventUpdate

if __name__ == '__main__':
    app.layout = layout
    app.run_server(debug=True)