import logging

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from app.app import app
from app.Dataset import Dataset
from app.DatasetStore import DatasetStore

from app.components.BasicComponent import BasicComponent
from app.components.Interests import class_to_color

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


viewer_layout = html.Div([
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
    # Store for user created masks
    # data is a list of dicts describing shapes
    dcc.Store(id="masks", data={"shapes": []}),
])

@app.callback([
    Output(f'selected-segmentation-name', 'options'),
], [
    Input('selected-dataset-name', 'value'),
    Input('update-button-segmentations-options', 'n_clicks')
]
)
def change_segmentation_options(name, n_clicks):
    if name:
        options = DatasetStore.get_dataset(name).get_segmentations_available()
        return [options]
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
    
@app.callback(
Output(f'viewer-annotations', 'data'),
[
    Input(f'viewer-graph', 'relayoutData'),
])
def update_annotations(graph_relayoutData):
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    logging.debug(f"Update: {cbcontext}")
    # TODO remove hardcoded id
    if cbcontext == "viewer-1-graph.relayoutData":
        if "shapes" in graph_relayoutData.keys():
            shapes = graph_relayoutData["shapes"]
            logging.debug(f"Shapes: {shapes}")
    raise PreventUpdate

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

@app.callback(
    [
        Output(f'viewer-graph', 'figure'),
        Output("masks", "data"),
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
        State(f'viewer-annotations', 'data'),
        State(f'viewer-graph', 'figure'),
        State("masks", "data"),
    ]
)
def update_fig(
    # inputs
    current_segmentation, 
    current_dataset, 
    slice_id,
    graph_relayoutData,
    any_label_class_button_value,
    stroke_width_value,
    # states
    annotations, 
    fig, 
    masks_data,
):
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    if fig is None:
        fig = make_default_figure()
    if not current_dataset or not slice_id:
        fig = make_default_figure()
        return [fig, masks_data, None]

    if cbcontext == "graph.relayoutData":
        if "shapes" in graph_relayoutData.keys():
            masks_data["shapes"] = graph_relayoutData["shapes"]
        else:
            return dash.no_update

    stroke_width = int(round(2 ** (stroke_width_value)))
    # find label class value by finding button with the most recent click
    if any_label_class_button_value is None:
        label_class_value = DEFAULT_LABEL_CLASS
    else:
        label_class_value = max(
            enumerate(any_label_class_button_value),
            key=lambda t: 0 if t[1] is None else t[1],
        )[0]

    fig = make_default_figure(
        stroke_color=class_to_color(label_class_value),
        stroke_width=stroke_width,
        shapes=masks_data["shapes"],
    )
    logging.debug(f"Fig: {fig}")
    add_layout_images_to_fig(fig=fig, segmentation=current_segmentation, dataset=DatasetStore.get_dataset(current_dataset), slice_id=slice_id)
    fig.update_layout(uirevision="segmentation")
    return (
        fig,
        masks_data,
        "Stroke width: %d" % (stroke_width,)
    )


# @app.callback(
#     [
#         Output("viewer-graph", "figure"),
#         Output("masks", "data"),
#         Output("stroke-width-display", "children"),
#     ],
#     [
#         Input("graph", "relayoutData"),
#         Input(
#             {"type": "label-class-button", "index": dash.dependencies.ALL},
#             "n_clicks_timestamp",
#         ),
#         Input("stroke-width", "value"),
#         Input("sigma-range-slider", "value"),
#     ],
#     [State("masks", "data"),],
# )
# def annotation_react(
#     graph_relayoutData,
#     any_label_class_button_value,
#     stroke_width_value,
#     show_segmentation_value,
#     download_button_n_clicks,
#     download_image_button_n_clicks,
#     segmentation_features_value,
#     sigma_range_slider_value,
#     masks_data,
# ):
#     classified_image_store_data = dash.no_update
#     classifier_store_data = dash.no_update
#     cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
#     if cbcontext in ["segmentation-features.value", "sigma-range-slider.value"] or (
#         ("Show segmentation" in show_segmentation_value)
#         and (len(masks_data["shapes"]) > 0)
#     ):
#         segmentation_features_dict = {
#             "intensity": False,
#             "edges": False,
#             "texture": False,
#         }
#         for feat in segmentation_features_value:
#             segmentation_features_dict[feat] = True
#         t1 = time()
#         features = compute_features(
#             img,
#             **segmentation_features_dict,
#             sigma_min=sigma_range_slider_value[0],
#             sigma_max=sigma_range_slider_value[1],
#         )
#         t2 = time()
#         print(t2 - t1)
#     if cbcontext == "graph.relayoutData":
#         if "shapes" in graph_relayoutData.keys():
#             masks_data["shapes"] = graph_relayoutData["shapes"]
#         else:
#             return dash.no_update
#     stroke_width = int(round(2 ** (stroke_width_value)))
#     # find label class value by finding button with the most recent click
#     if any_label_class_button_value is None:
#         label_class_value = DEFAULT_LABEL_CLASS
#     else:
#         label_class_value = max(
#             enumerate(any_label_class_button_value),
#             key=lambda t: 0 if t[1] is None else t[1],
#         )[0]

#     fig = make_default_figure(
#         stroke_color=class_to_color(label_class_value),
#         stroke_width=stroke_width,
#         shapes=masks_data["shapes"],
#     )
#     # We want the segmentation to be computed
#     if ("Show segmentation" in show_segmentation_value) and (
#         len(masks_data["shapes"]) > 0
#     ):
#         segimgpng = None
#         try:
#             feature_opts = dict(segmentation_features_dict=segmentation_features_dict)
#             feature_opts["sigma_min"] = sigma_range_slider_value[0]
#             feature_opts["sigma_max"] = sigma_range_slider_value[1]
#             segimgpng, clf = show_segmentation(
#                 DEFAULT_IMAGE_PATH, masks_data["shapes"], features, feature_opts
#             )
#             if cbcontext == "download-button.n_clicks":
#                 classifier_store_data = clf
#             if cbcontext == "download-image-button.n_clicks":
#                 classified_image_store_data = plot_common.pil_image_to_uri(
#                     blend_image_and_classified_regions_pil(
#                         PIL.Image.open(DEFAULT_IMAGE_PATH), segimgpng
#                     )
#                 )
#         except ValueError:
#             # if segmentation fails, draw nothing
#             pass
#         images_to_draw = []
#         if segimgpng is not None:
#             images_to_draw = [segimgpng]
#         fig = plot_common.add_layout_images_to_fig(fig, images_to_draw)
#     fig.update_layout(uirevision="segmentation")
#     return (
#         fig,
#         masks_data,
#         "Stroke width: %d" % (stroke_width,),
#         classifier_store_data,
#         classified_image_store_data,
#     )

if __name__ == '__main__':
    app.layout = viewer_layout
    app.run_server(debug=True)