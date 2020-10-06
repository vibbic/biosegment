import logging
from pathlib import Path

import json
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from app.app import app
from app import api
from app.DatasetStore import DatasetStore
from app.layout_utils import dropdown_with_button, ANNOTATION_MODE
from app.shape_utils import (
    DEFAULT_STROKE_WIDTH,
    annotations_to_png,
    class_labels,
    class_to_color,
)
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


PREFIX = "annotation-tools"

annotation_tools_layout = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Annotation tools", className="card-title"),
            # Label class chosen with buttons
            dbc.FormGroup(
                [
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
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Stroke width"),
                    dcc.Slider(
                        id="stroke-width",
                        min=0,
                        max=6,
                        step=0.1,
                        value=DEFAULT_STROKE_WIDTH,
                    ),
                    html.H6(id="stroke-width-display"),
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Selected annotation"),
                    dropdown_with_button(
                        dropdown_id=f"{PREFIX}-selected-annotation-name",
                        button_id=f"{PREFIX}-refresh-selected-annotation-name",
                    ),
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Create new annotation from selected"),
                    dcc.Input(
                        id="new-annotation-name", type="text", value="New annotation 1",
                    ),
                    dbc.Button("Create new annotation", id="create-new-annotation"),
                ]
            ),
            dbc.FormGroup(
                [
                    # disabled by default because no annotation is selected by default
                    dbc.Button("Start editing current annotation", id="annotation-mode-btn", color="success", active=False, disabled=True),
                ]
            ),
            dbc.Label("Annotations"),
            html.P(id="masks-display"),
            dcc.Store(id="annotation-mode", data=ANNOTATION_MODE.NOT_EDITING)
        ]
    ),
    id="right-column",
)

@app.callback(
    [
        Output(f"{PREFIX}-selected-annotation-name", "options"),
        Output(f"{PREFIX}-selected-annotation-name", "disabled"),
    ],
    [
        Input("selected-dataset-name", "value"),
        Input(f"{PREFIX}-refresh-selected-annotation-name", "n_clicks"),
        Input("annotation-mode", "data")
    ],
    [
        State("selected-dataset-name", "value")
    ]
)
def change_annotation_dropdown(selected_dataset, n_clicks, annotation_mode, dataset_name):
    options = DatasetStore.get_dataset(dataset_name).get_annotations_available()
    return [options, annotation_mode == ANNOTATION_MODE.EDITING]

@app.callback(
    [Output("annotations", "data"),],
    [
        Input("viewer-graph", "relayoutData"),
        Input("annotation-mode", "data"),
    ],
    [
        State("annotations", "data"),
        State(f"viewer-slice-id", "value"),
    ]
)
def change_annotations(graph_relayoutData, annotation_mode, annotations_data, slice_id):
    if graph_relayoutData:
        if "shapes" in graph_relayoutData.keys():
            # TODO support for editing annotations
            # TODO use set
            new_shapes = graph_relayoutData["shapes"]
            try:
                old_shapes = annotations_data[slice_id]
            except:
                logging.debug(
                    f"No annotation entry for slice {slice_id} and keys {annotations_data.keys()}, creating one"
                )
                annotations_data[slice_id] = []
                old_shapes = []
            annotations_data[slice_id] += [s for s in new_shapes if s not in old_shapes]
    if annotation_mode:
        if annotation_mode == ANNOTATION_MODE.EDITING:
            # TODO unselect current selected shape
            editable = True
        else:
            editable = False
        # set annotations to correct mode on change
        for annotations in annotations_data.values():
            for annotation in annotations:
                annotation["editable"] = editable
    return [annotations_data]

@app.callback(
    [
        Output("annotation-mode-btn", "children"),
        Output("annotation-mode-btn", "color"),
        Output("annotation-mode-btn", "active"),
        Output("annotation-mode-btn", "disabled"),
    ],
    [
        Input("annotation-mode", "data"),
        Input(f"{PREFIX}-selected-annotation-name", "value")
    ],
)
def change_annotation_mode_btn(annotation_mode, selected_annotation):
    if annotation_mode == ANNOTATION_MODE.NOT_EDITING:
        return "Start editing current annotation", "success", False, selected_annotation is None
    else:
        return "Stop editing current annotation", "danger", True, False

@app.callback(
    [
        Output("annotation-mode", "data"),
    ],
    [
        Input("annotation-mode-btn", "n_clicks"),
    ],
    [
        State("annotation-mode", "data"),
        State("annotations", "data"),
        State(f"{PREFIX}-selected-annotation-name", "value"),
        State("selected-dataset-name", "value"),
    ]
)
def change_annotation_mode(n_clicks, old_mode, annotations_data, selected_annotation_id, dataset_id):
    if n_clicks:
        if old_mode == ANNOTATION_MODE.EDITING:
            # logging.debug(f"Annotations data: {annotations_data}")
            # TODO difference between POST and UPDATE
            api.annotation.update(id=selected_annotation_id, json={
                "shapes": annotations_data 
            })
            return [ANNOTATION_MODE.NOT_EDITING]
        else:
            # annotation data should already be loaded by annotation selector
            return [ANNOTATION_MODE.EDITING]
    raise PreventUpdate


@app.callback(Output("masks-display", "children"), [Input("annotations", "data"),])
def show_annotations(annotations_data):
    # logging.debug(f"annotations: {annotations_data}")
    try:
        return html.Ul(
            [
                html.Div(
                    [
                        html.P(f"Slice {slice_id}"),
                        html.Ul(
                            [
                                html.Li(f"Color {annotation['line']['color']}")
                                for annotation in annotations
                            ]
                        ),
                    ]
                )
                for slice_id, annotations in annotations_data.items()
            ]
        )
    except Exception as e:
        logging.debug(f"Error displaying annotations: {e}")
        raise PreventUpdate

@app.callback(
    # not needed
    Output("create-new-annotation", "children"),
    [
        Input("create-new-annotation", "n_clicks"),
    ],
    [
        State("selected-dataset-name", "value"),
        State("new-annotation-name", "value"),
        State(f"{PREFIX}-selected-annotation-name", "value")
    ],
)
def create_annotation(n_clicks, dataset_id, new_annotation_name, selected_annotation_name):
    if n_clicks:
        logging.debug(f"Creating annotation")
        # TODO support create from other annotation
        api.annotation.create_for_dataset(
            dataset_id=dataset_id,
            json={
                "title": new_annotation_name,
                "location": f"annotations/{DatasetStore.get_dataset(dataset_id).get_title()}/{new_annotation_name}/annotations.json"
            }
        )
        return dash.no_update
    logging.debug(f"Not creating annotation")
    raise PreventUpdate

# @app.callback(
#     # not needed
#     Output("save-new-annotation", "value"),
#     [Input("save-new-annotation", "n_clicks"),],
#     [
#         State("annotations", "data"),
#         State("selected-dataset-name", "value"),
#         State("new-annotation-name", "value"),
#         State("viewer-slice-id", "max"),
#     ],
# )
# def annotations_to_png(n_clicks, annotations_data, dataset_id, new_annotation_name, max_slice):
#     if not n_clicks:
#         raise PreventUpdate
#     annotation_folder = Path(f"/data/annotations/{DatasetStore.get_dataset(dataset_id).get_title()}/{new_annotation_name}")
#     try:
#         # TODO define behaviour if folder exists
#         annotation_folder.mkdir(parents=True, exist_ok=True)
#         logging.debug(annotations_data)
#         for slice_id in range(max_slice):
#             slice_id = str(slice_id)
#             if slice_id in annotations_data:
#                 annotations = annotations_data[slice_id]
#             else:
#                 annotations = None
#             annotations_to_png(
#                 width=512,
#                 height=512,
#                 annotations=annotations,
#                 write_to=f"{annotation_folder}/{int(slice_id):04d}.png",
#             )
#     except Exception as e:
#         logging.debug(f"Error saving annotations: {e}")
#         raise PreventUpdate
