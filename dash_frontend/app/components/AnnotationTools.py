import logging
import os 

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app.app import app
from app import api
from app.DatasetStore import DatasetStore
from app.shape_utils import annotations_to_png, class_to_color, color_to_class, DEFAULT_STROKE_WIDTH, NUM_LABEL_CLASSES, class_labels

annotation_tools_layout = dbc.Card(
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
        dbc.FormGroup([
            dbc.Label("New annotations name"),
            dcc.Input(
                id="new-annotation-name",
                type="text",
                value="New annotation 1",
            ),
            dbc.Button(
                "Save new annotation",
                id="save-new-annotation",
            )
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
    # logging.debug(f"annotations: {annotations_data}")
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

@app.callback(    
    # not needed
    Output("save-new-annotation", "value"),
[
    Input("save-new-annotation", "n_clicks"),
], [
    State("annotations", "data"),
    State("selected-dataset-name", "value"),
    State("new-annotation-name", "value"),
]
)
def save_annotations(n_clicks, annotations_data, dataset_id, new_annotation_name):
    if not n_clicks:
        raise PreventUpdate
    annotation_folder = f"/data/annotations/{DatasetStore.get_dataset(dataset_id).get_title()}/{new_annotation_name}"
    try:
        for slice_id, annotations in annotations_data.items():
            try:
                os.makedirs(annotation_folder)
            except OSError as error:
                logging.debug(error)
            annotations_to_png(width=512, height=512, annotations=annotations, write_to=f"{annotation_folder}/{int(slice_id):04d}.png")
    except Exception as e:
        logging.debug(f"Error saving annotations: {e}")
        raise PreventUpdate
