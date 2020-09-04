import logging

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app.app import app
from app.DatasetStore import DatasetStore
from app.components.Viewer2D import Viewer2D
from app.components.Progress import progress
from app.api.utils import infer, poll_task, test_celery

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
        options=[
            {"label": name, "value": name} for name in DatasetStore.get_names_available()
        ],
        value=DEFAULT_DATASET,
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
    progress,
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
    dcc.Store(id="task-id"),
    dcc.Interval(id="task-polling", disabled=True)
])

@app.callback(
    [
        Output("progress", "animated"),
    ],
    [
        Input("task-polling", "n_intervals"),
    ],
    [
        State("task-id", "data"),
        State('token', 'data'),
    ]
)
def dash_poll_task(n_intervals, task_id_data, token):
    try:
        task_id = task_id_data["task_id"]["task_id"]
    except:
        logging.debug("Not task id")
        raise PreventUpdate
    if not task_id:
        logging.debug("Not task id2")
        raise PreventUpdate
    logging.debug(f"Task id {task_id}")
    try:
        response = poll_task(token=token, json={"task_id": task_id})
    except:
        logging.debug(f"Polling failed")
        raise PreventUpdate
    logging.debug(f"Response {response}")
    state = response["state"]
    if state == "PROGRESS":
        return [True]
    elif state == "PENDING":
        return [False]
    raise PreventUpdate

@app.callback(
    [
        Output("task-id", "data"),
        Output("task-polling", "disabled"),
    ],
    [
        Input("start-new-segmentation", "n_clicks"),
    ],
    [
        State("new-segmentation-name", "value"),
        State("selected-model-name", "value"),
        State('token', 'data'),
        State("task-polling", "disabled"),
    ]
)
def start_segmentation(n, new_segmentation_name, selected_model, token, no_polling):
    if not no_polling:
        raise PreventUpdate
    if n:
        logging.debug
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
        # task_id = infer(token=token, json=body)
        task_id = test_celery(token=token, json={"timeout": 10})
        return {"task_id": task_id}, False
    raise PreventUpdate

# @app.callback(
#     [Output("progress", "value"), Output("progress", "children")],
#     [Input("progress-interval", "n_intervals")],
# )
# def update_progress(n):
#     # check progress of some background process, in this example we'll just
#     # use n_intervals constrained to be in 0-100
#     progress = min(n % 110, 100)
#     # only add text after 5% progress to ensure text isn't squashed too much
#     return progress, f"{progress} %" if progress >= 5 else ""

@app.callback(
    Output(f'{VIEWER_ID}-current-segmentation-name', 'data'),[
        Input('selected-segmentation-name', 'value')
])
def select_segmentation(value):
    return value

@app.callback([
    Output(f'selected-model-name', 'options'),
], [Input('selected-dataset-name', 'value')])
def change_model_options(name):
    options = DatasetStore.get_dataset(name).get_models_available()
    return [options]

@app.callback([
    Output(f'selected-segmentation-name', 'options'),
], [Input('selected-dataset-name', 'value')])
def change_segmentation_options(name):
    options = DatasetStore.get_dataset(name).get_segmentations_available()
    return [options]

@app.callback([
    Output(f'{VIEWER_ID}-current-dataset-name', 'data'),
    Output(f'{VIEWER_ID}-dataset-dimensions', 'data'),
], [Input('selected-dataset-name', 'value')])
def change_dataset(name):
    dataset = DatasetStore.get_dataset(name)
    return name, dataset.get_dimensions()

if __name__ == '__main__':
    app.layout = layout
    app.run_server(debug=True)