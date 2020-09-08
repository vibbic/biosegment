import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from app.components.TaskProgress import task_progress

segmentation_runner_layout = html.Div([
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
])