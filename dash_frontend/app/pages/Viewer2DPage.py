import dash_bootstrap_components as dbc
import dash_html_components as html
from app.components.AnnotationTools import annotation_tools_layout
from app.components.ModelRetrainer import model_retrainer_layout
from app.components.SegmentationRunner import segmentation_runner_layout
from app.components.Viewer2D import viewer_layout

title = "Viewer 2D"
path = "/viewer"

layout = dbc.Container(
    children=[
        html.H1("Viewer 2D"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(viewer_layout, md=6, lg=6),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [dbc.Row([dbc.Col(annotation_tools_layout),])],
                                    md=12,
                                    lg=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Row([dbc.Col(segmentation_runner_layout),]),
                                        dbc.Row([dbc.Col(model_retrainer_layout),]),
                                    ],
                                    md=12,
                                    lg=6,
                                ),
                            ]
                        )
                    ],
                    md=6,
                    lg=6,
                ),
            ]
        ),
    ],
    fluid=True,
)
