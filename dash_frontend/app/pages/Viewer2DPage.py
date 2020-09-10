import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from app.components.Viewer2D import viewer_layout
from app.components.SegmentationRunner import segmentation_runner_layout
from app.components.DatasetSelector import dataset_selector_layout
from app.components.AnnotationTools import annotation_tools_layout
from app.components.ModelRetrainer import model_retrainer_layout

WORKER_ROOT_DATA_FOLDER="/home/brombaut/workspace/biosegment/data/"

title="Viewer 2D"
path="/viewer"

layout = dbc.Container(
    children=[
        html.H1("Viewer 2D"),
        html.Hr(),
        dbc.Row(
            dbc.Col(dataset_selector_layout),
        ),
        dbc.Row([
            dbc.Col([
                dbc.Row(
                    [
                        dbc.Col(segmentation_runner_layout),
                        dbc.Col(model_retrainer_layout),
                    ]
            ),
                dbc.Row(
                    [
                        dbc.Col(annotation_tools_layout),
                    ]
                )
            ]),
            dbc.Col(
                dbc.Card(
                [
                    html.H3(
                        "Viewer"
                    ),
                    html.P(
                        "Selected segmentation"
                    ),
                    html.Button('Update segmentation options', id='update-button-segmentations-options'),
                    dcc.Dropdown(
                        id="selected-segmentation-name",
                    ),
                    viewer_layout,
                ], body=True),
                md=12,
                lg=6,
            )
        ]),
])