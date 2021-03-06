import logging

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from app.app import app
from app.components.DatasetSelector import dataset_selector_layout
from app.DatasetStore import DatasetStore
from app.layout_utils import dropdown_with_button, ANNOTATION_MODE
from app.shape_utils import class_to_color
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

DEFAULT_LABEL_CLASS = 0


def add_layout_images_to_fig(fig, segmentations, dataset, slice_id, update_ranges=True):
    """ images is a sequence of PIL Image objects """
    images = [("slice", dataset.get_slice(slice_id))]
    if segmentations:
        if not isinstance(segmentations, list):
            segmentations = [segmentations]
        for s in segmentations:
            images.append(("label", dataset.get_label(s, slice_id)))
    for t_im in images:
        # if image is a path to an image, load the image to get its size
        is_slice = t_im[0] == "slice"
        im = t_im[1]

        width, height = im.size
        # Add images
        fig.add_layout_image(
            dict(
                source=im,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=width,
                sizey=height,
                sizing="contain",
                layer="below" if is_slice else "above",
            )
        )
    if update_ranges:
        width, height = [max([t_im[1].size[i] for t_im in images]) for i in range(2)]
        # TODO showgrid,showticklabels,zeroline should be passable to this
        # function
        fig.update_xaxes(
            showgrid=False, range=(0, width), showticklabels=False, zeroline=False
        )
        fig.update_yaxes(
            showgrid=False,
            scaleanchor="x",
            range=(height, 0),
            showticklabels=False,
            zeroline=False,
        )
    return fig


def make_default_figure(
    annotation_mode=ANNOTATION_MODE.NOT_EDITING, stroke_color=None, stroke_width=None, shapes=[]
):
    def dummy_fig():
        fig = go.Figure(go.Scatter(x=[], y=[]))
        fig.update_layout(template=None)
        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
        fig.update_yaxes(
            showgrid=False, scaleanchor="x", showticklabels=False, zeroline=False
        )
        return fig

    fig = dummy_fig()
    if (annotation_mode == ANNOTATION_MODE.EDITING):
        fig.update_layout(
            {
                "dragmode": "drawopenpath",
                "shapes": shapes,
                "newshape.line.color": stroke_color,
                "newshape.line.width": stroke_width,
                "margin": dict(l=0, r=0, b=0, t=0, pad=4),
            }
        )
    else:
        fig.update_layout(
            {
                "shapes": shapes,
                "margin": dict(l=0, r=0, b=0, t=0, pad=4),
            }
        )
    return fig


viewer_layout = dbc.Card(
    [
        html.H3("Viewer"),
        dataset_selector_layout,
        html.P("Selected segmentation"),
        dropdown_with_button(
            dropdown_id="selected-segmentation-name",
            button_id="update-button-segmentations-options",
            multi=True
        ),
        html.Div(id=f"viewer-slice-info"),
        dcc.Slider(id=f"viewer-slice-id"),
        html.Div(id=f"viewer-slider-output-container"),
        dcc.Graph(
            id=f"viewer-graph",
            figure=make_default_figure(),
        ),
        dcc.Store(id=f"viewer-dataset-dimensions", data={}),
        dcc.Store(id=f"viewer-slice-number-top", data=0),
        dcc.Store(id=f"viewer-slice-number-side", data=0),
        dcc.Store(id=f"viewer-annotations", data=None),
        # Store for annotations
        # data is a dict with slice_id as keys and list of shapes as values
        # shapes are dicts given by the Plotly graph
        # the line color is the interest class
        dcc.Store(id="annotations", data={}),
        html.Div(id="test"),
    ],
    body=True,
)


@app.callback(
    [
        Output(f"selected-segmentation-name", "options"),
        Output(f"selected-segmentation-name", "value"),
        Output(f"selected-segmentation-name", "disabled"),
    ],
    [
        Input("selected-dataset-name", "value"),
        Input("update-button-segmentations-options", "n_clicks"),
        Input("annotation-mode", "data"),
    ],
    [State(f"selected-segmentation-name", "value")],
)
def change_segmentation_options(name, n_clicks, annotation_mode, current_segmentation):
    if name or annotation_mode:
        cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
        options = DatasetStore.get_dataset(name).get_segmentations_available()
        # open first segmentation 1. as default or 2. on dataset change
        if (current_segmentation is None or cbcontext == "selected-dataset-name.value"):
            if options:
                current_segmentation = options[0]["value"]
            else:
                current_segmentation = None
        return [options, current_segmentation, annotation_mode == ANNOTATION_MODE.EDITING]
    raise PreventUpdate


@app.callback(
    Output(f"viewer-slice-info", "children"), [Input(f"viewer-slice-id", "value")]
)
def update_slice_info(value):
    return "Slice {}".format(value)


@app.callback(
    [
        Output(f"viewer-slice-id", "min"),
        Output(f"viewer-slice-id", "max"),
        Output(f"viewer-slice-id", "marks"),
        Output(f"viewer-slice-id", "value"),
    ],
    [
        Input(f"viewer-dataset-dimensions", "data")
    ],
)
def update_slider(dimensions):
    if dimensions:
        logging.info("Update slider")
        min_slice = dimensions["min"]
        max_slice = dimensions["max"]
        marks = {i: str(i) for i in range(0, max_slice, 10)}
        return min_slice, max_slice, marks, 10
    raise PreventUpdate


@app.callback(
    [Output(f"viewer-dataset-dimensions", "data"),],
    [Input("selected-dataset-name", "value")],
    [
        State(f"viewer-dataset-dimensions", "data")
    ],
)
def change_dataset_dimensions(name, old_dimensions):
    if name:
        dataset = DatasetStore.get_dataset(name)
        dimensions = dataset.get_dimensions()
        if dimensions == old_dimensions:
            raise PreventUpdate
        return [dataset.get_dimensions()]
    raise PreventUpdate

def create_annotations(shape, slice_id):
    try:
        interest = color_to_class(shape["line"]["color"])
    except:
        interest = None
    return {
        "slice_id": slice_id,
        "shape": shape,
        "interest": interest,
    }

@app.callback(
    [
        Output(f"viewer-graph", "config"),
        Output(f"viewer-graph", "figure"),
        Output("stroke-width-display", "children"),
    ],
    [
        Input(f"selected-segmentation-name", "value"),
        # TODO
        Input("selected-dataset-name", "value"),
        Input(f"viewer-slice-id", "value"),
        # Input("viewer-graph", "relayoutData"),
        Input(
            {"type": "label-class-button", "index": dash.dependencies.ALL},
            "n_clicks_timestamp",
        ),
        Input("stroke-width", "value"),
        Input("annotations", "data"),
        Input("annotation-mode", "data"),
    ],
    [
        State(f"viewer-slice-id", "value"),
        # State(f'viewer-graph', 'figure'),
        # State("annotations", "data"),
    ],
)
def update_fig(
    # inputs
    current_segmentations,
    current_dataset,
    slice_id_update,
    any_label_class_button_value,
    stroke_width_value,
    annotations_data,
    annotation_mode,
    # states
    # fig,
    slice_id,
):
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    config = {}
    # if fig is None:
    #     fig = make_default_figure()
    if not current_dataset or not slice_id:
        fig = make_default_figure(annotation_mode=annotation_mode,)
        return [config, fig, None]

    # TODO fix confusion
    # keys are strings, not ints
    slice_id = str(slice_id)
    
    if annotation_mode:
        edit_buttons = ["drawrect", "drawopenpath", "eraseshape",]
        if annotation_mode == ANNOTATION_MODE.EDITING:
            # TODO unselect current selected shape
            config =  {"modeBarButtonsToAdd": edit_buttons}
        else:
            config = {"modeBarButtonsToRemove": edit_buttons}

    # if cbcontext == "viewer-graph.relayoutData":
    #     if "shapes" in graph_relayoutData.keys():
    #         # TODO support for editing annotations
    #         # TODO use set
    #         new_shapes = graph_relayoutData["shapes"]
    #         try:
    #             old_shapes = annotations_data[slice_id]
    #         except:
    #             logging.debug(
    #                 f"No annotation entry for slice {slice_id} and keys {annotations_data.keys()}, creating one"
    #             )
    #             annotations_data[slice_id] = []
    #             old_shapes = []
    #         annotations_data[slice_id] += [s for s in new_shapes if s not in old_shapes]
        # else:
        #     return dash.no_update

    stroke_width = int(round(2 ** (stroke_width_value)))
    # find label class value by finding button with the most recent click
    if any_label_class_button_value is None:
        label_class_value = DEFAULT_LABEL_CLASS
    else:
        label_class_value = max(
            enumerate(any_label_class_button_value),
            key=lambda t: 0 if t[1] is None else t[1],
        )[0]

    try:
        current_annotations = annotations_data[slice_id]
    except:
        logging.debug(f"No annotations to draw, keys are: {annotations_data.keys()}")
        current_annotations = []

    fig = make_default_figure(
        stroke_color=class_to_color(label_class_value),
        stroke_width=stroke_width,
        shapes=current_annotations,
        annotation_mode=annotation_mode,
    )
    # logging.debug(f"Fig: {fig}")
    add_layout_images_to_fig(
        fig=fig,
        segmentations=current_segmentations,
        dataset=DatasetStore.get_dataset(current_dataset),
        slice_id=int(slice_id),
    )
    fig.update_layout(uirevision="segmentation")
    return (config, fig, "Stroke width: %d" % (stroke_width,))


if __name__ == "__main__":
    app.layout = viewer_layout
    app.run_server(debug=True)
