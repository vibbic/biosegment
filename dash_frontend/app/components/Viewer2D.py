import logging

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from app.app import app
from app.Dataset import Dataset
from app.DatasetStore import DatasetStore

from app.components.BasicComponent import BasicComponent
from app.shape_utils import class_to_color


DEFAULT_LABEL_CLASS = 0

def add_layout_images_to_fig(fig, segmentation, dataset, slice_id, update_ranges=True):
    """ images is a sequence of PIL Image objects """
    images = [("slice", dataset.get_slice(slice_id))]
    if segmentation:
        images.append(("label", dataset.get_label(segmentation, slice_id)))
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
                layer="below" if is_slice else "above"
            )
        )
    if update_ranges:
        width, height = [
            max([t_im[1].size[i] for t_im in images]) for i in range(2)
        ]
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
    stroke_color=None,
    stroke_width=None,
    shapes=[],
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
    fig.update_layout(
        {
            "dragmode": "drawopenpath",
            "shapes": shapes,
            "newshape.line.color": stroke_color,
            "newshape.line.width": stroke_width,
            "margin": dict(l=0, r=0, b=0, t=0, pad=4),
        }
    )
    return fig


viewer_layout = dbc.Card(
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
    html.Div(
        id=f'viewer-slice-info'
    ),
    dcc.Slider(
        id=f'viewer-slice-id'
    ),
    html.Div(id=f'viewer-slider-output-container'),
    dcc.Graph(
        id=f'viewer-graph',
        figure=make_default_figure(),
        config={
            "modeBarButtonsToAdd": [
                "drawrect",
                "drawopenpath",
                "eraseshape",
            ]
        },
    ),
    dcc.Store(id=f'viewer-dataset-dimensions', data={"min": 0, "max": 63}),
    dcc.Store(id=f'viewer-slice-number-top', data=0),
    dcc.Store(id=f'viewer-slice-number-side', data=0),
    dcc.Store(id=f'viewer-annotations', data=None),
    # Store for annotations
    # data is a dict with slice_id as keys and list of shapes as values
    # shapes are dicts given by the Plotly graph
    # the line color is the interest class
    dcc.Store(id="annotations", data={}),
    ], body=True
)

@app.callback([
    Output(f'selected-segmentation-name', 'options'),
    Output(f'selected-segmentation-name', 'value'),
], [
    Input('selected-dataset-name', 'value'),
    Input('update-button-segmentations-options', 'n_clicks')
], [
    State(f'selected-segmentation-name', 'value')
]
)
def change_segmentation_options(name, n_clicks, current_segmentation):
    if name:
        cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
        options = DatasetStore.get_dataset(name).get_segmentations_available()
        # open first segmentation 1. as default or 2. on dataset change
        if (current_segmentation is None and options) or cbcontext == "selected-dataset-name.value":
            current_segmentation = options[0]["value"]
        return [options, current_segmentation]
    raise PreventUpdate

@app.callback(
            Output(f'viewer-slice-info', 'children'),
            [Input(f'viewer-slice-id', 'value')]
        )
def update_slice_info(value):
    return 'Slice {}'.format(value)

@app.callback([
            Output(f'viewer-slice-id', 'min'),
            Output(f'viewer-slice-id', 'max'),
            Output(f'viewer-slice-id', 'marks'),
            Output(f'viewer-slice-id', 'value')
        ], [Input(f'viewer-dataset-dimensions', 'data')]
        )
def update_slider(dimensions):
    min_slice = dimensions["min"]
    max_slice = dimensions["max"]
    marks = { i: str(i) for i in range(0, max_slice, 10)}
    return min_slice, max_slice, marks, 10
    
@app.callback([
    Output(f'viewer-dataset-dimensions', 'data'),
], [
    Input('selected-dataset-name', 'value')
])
def change_dataset_dimensions(name):
    if name:
        dataset = DatasetStore.get_dataset(name)
        return [dataset.get_dimensions()]
    raise PreventUpdate

def create_annotation(shape, slice_id):
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
        Output(f'viewer-graph', 'figure'),
        Output("annotations", "data"),
        Output("stroke-width-display", "children"),
    ],
    [
        Input(f'selected-segmentation-name', 'value'),
        # TODO
        Input('selected-dataset-name', 'value'),
        Input(f'viewer-slice-id', 'value'),
        Input("viewer-graph", "relayoutData"),
        Input(
            {"type": "label-class-button", "index": dash.dependencies.ALL},
            "n_clicks_timestamp",
        ),
        Input("stroke-width", "value"),
    ], [
        State(f'viewer-slice-id', 'value'),
        # State(f'viewer-graph', 'figure'),
        State("annotations", "data"),
    ]
)
def update_fig(
    # inputs
    current_segmentation, 
    current_dataset, 
    slice_id_update,
    graph_relayoutData,
    any_label_class_button_value,
    stroke_width_value,
    # states
    # fig, 
    slice_id,
    annotations_data,
):
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    # if fig is None:
    #     fig = make_default_figure()
    if not current_dataset or not slice_id:
        fig = make_default_figure()
        return [fig, annotations_data, None]

    # TODO fix confusion
    # keys are strings, not ints
    slice_id = str(slice_id)

    if cbcontext == "viewer-graph.relayoutData":
        if "shapes" in graph_relayoutData.keys():
            # TODO support for editing annotations
            # TODO use set
            new_shapes = graph_relayoutData["shapes"]
            try:
                old_shapes = annotations_data[slice_id]
            except:
                logging.debug(f"No annotation entry for slice {slice_id} and keys {annotations_data.keys()}, creating one")
                annotations_data[slice_id] = []
                old_shapes = []
            annotations_data[slice_id] += [s for s in new_shapes if s not in old_shapes] 
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
    )
    # logging.debug(f"Fig: {fig}")
    add_layout_images_to_fig(fig=fig, segmentation=current_segmentation, dataset=DatasetStore.get_dataset(current_dataset), slice_id=int(slice_id))
    fig.update_layout(uirevision="segmentation")
    return (
        fig,
        annotations_data,
        "Stroke width: %d" % (stroke_width,)
    )

if __name__ == '__main__':
    app.layout = viewer_layout
    app.run_server(debug=True)