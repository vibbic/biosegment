import logging

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app.app import app
from app import api

import plotly.express as px

DEFAULT_STROKE_WIDTH = 3  # gives line width of 2^3 = 8
# the number of different classes for labels
NUM_LABEL_CLASSES = 3
class_label_colormap = px.colors.qualitative.Light24
class_labels = list(range(NUM_LABEL_CLASSES))
# we can't have less colors than classes
assert NUM_LABEL_CLASSES <= len(class_label_colormap)

def class_to_color(n):
    return class_label_colormap[n]

def color_to_class(c):
    return class_label_colormap.index(c)

interests_layout = dbc.Card(
    dbc.CardBody([
        html.H5("Annotation tools", className="card-title"),
        # Label class chosen with buttons
        dbc.FormGroup([
            dbc.Label("Classes of interest"),
            html.Div(
                        id="label-class-buttons",
                        children=[
                            html.Button(
                                "%2d" % (n,),
                                id={"type": "label-class-button", "index": n},
                                style={"background-color": class_to_color(c)},
                            )
                            for n, c in enumerate(class_labels)
                        ],
                    ),
        ]),
        dbc.FormGroup([
            dbc.Label("Stroke width"),
            dcc.Slider(
                id="stroke-width",
                min=0,
                max=6,
                step=0.1,
                value=DEFAULT_STROKE_WIDTH,
            ),
            html.H6(id="stroke-width-display"),
        ]),
        dbc.Label("Annotations"),
        html.P(id="masks-display"),     
    ]),
    id="right-column"
)

@app.callback(
    
        Output("masks-display", "children")
    ,
    [
        Input("annotations", "data"),
    ]
)
def show_annotations(annotations_data):
    logging.debug(f"annotations: {annotations_data}")
    try:
        annotations = annotations_data
        return html.Ul(
            [
                html.Div([
                    html.P(f"Slice {slice_id}"),
                    html.Ul(
                        [
                            html.Li(f"Color {annotation['line']['color']}") for annotation in annotations
                        ]
                    )
                ]) for slice_id, annotations in annotations_data.items()
            ])
    except Exception as e:
        logging.debug(f"Error displaying annotations: {e}")
        raise PreventUpdate

