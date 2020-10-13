import logging

import dash_bootstrap_components as dbc
import dash_core_components as dcc
from app.app import app
from app.DatasetStore import DatasetStore
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from app.layout_utils import ANNOTATION_MODE

dataset_selector_layout = dbc.FormGroup(
    [dbc.Label("Selected dataset"), dcc.Dropdown(id="selected-dataset-name",),]
)


@app.callback(
    [
        Output("selected-dataset-name", "options"),
        Output("selected-dataset-name", "value"),
        Output("selected-dataset-name", "disabled"),
    ],
    [
        Input("url", "pathname"),
        Input("annotation-mode", "data"),
    ]
)
def get_dataset_options(pathname, annotation_mode):
    if pathname:
        try:
            datasets = DatasetStore.get_names_available()
            if not datasets:
                raise PreventUpdate
            options = [{"label": d["title"], "value": d["id"],} for d in datasets]
            value = datasets[0]["id"]
            logging.debug(f"Datasets options: {options}")
            return [options, value, annotation_mode == ANNOTATION_MODE.EDITING]
        except:
            raise PreventUpdate
    raise PreventUpdate
