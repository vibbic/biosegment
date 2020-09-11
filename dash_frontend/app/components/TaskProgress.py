import logging

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app.app import app
from app import api


def create_layout(prefix):
    return html.Div(
    [
        dbc.Progress(id=f'{prefix}-progress', value=0, striped=True, animated=False),
        dcc.Store(id=f'{prefix}-task-id'),
        dcc.Interval(id=f'{prefix}-task-polling', disabled=False),
    ]
)

def create_callbacks(prefix):
    @app.callback(
        [
            Output(f"{prefix}-progress", "value"),
            Output(f"{prefix}-progress", "max"),
            Output(f"{prefix}-progress", "animated"),
            Output(f"{prefix}-progress", "children"),
            Output(f"{prefix}-progress", "color"),
            Output(f"{prefix}-task-polling", "disabled"),
        ],
        [
            Input(f"{prefix}-task-polling", "n_intervals"),
            Input(f"{prefix}-task-id", "data"),
        ]
    )
    def dash_poll_task(n_intervals, task_id_data):
        try:
            task_id = task_id_data["task_id"]["task_id"]
        except:
            logging.debug("Not task id")
            return [0, 100, True, "0%", "primary", True]
        if not task_id:
            logging.debug("Not task id2")
            return [0, 100, True, "0%", "primary", True]
        logging.debug(f"Task id {task_id}")
        try:
            response = api.utils.poll_task(json={"task_id": task_id})
        except:
            logging.debug(f"Polling failed")
            raise PreventUpdate
        logging.debug(f"Response {response}")
        state = response["state"]
        current = response["current"]
        total = response["total"]
        logging.debug(f"Task {task_id} state is {state}")
        if state == "PROGRESS":
            return [current, total, True, f"{min(int(current / total * 100), 100)}%", "primary", False]
        elif state == "PENDING":
            return [0, 100, True, "PENDING", "primary", False]
        elif state == "SUCCESS":
            return [100, 100, False, "SUCCESS", "success", True]
        elif state == "FAILURE":
            return [100, 100, False, "ERROR", "danger", True]
        raise PreventUpdate