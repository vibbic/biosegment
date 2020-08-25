import json

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image
import io

import pandas as pd
import image_utils
from utils import get_folder_list
from skimage import data, img_as_ubyte, segmentation, measure
from skimage import io as skio
from app import app

def add_layout_images_to_fig(fig, images, update_ranges=True):
    """ images is a sequence of PIL Image objects """
    if len(images) <= 0:
        return fig
    for im in images:
        # if image is a path to an image, load the image to get its size
        width, height = Image.open(im).size

        if "labels" in im:
            source = skio.imread(im)
            source = image_utils.label_to_colors(source, **{
                "alpha":[128, 128], 
                "color_class_offset":0,
                "no_map_zero": True
            })
            source = Image.fromarray(source)
            layer = "above"
        else:
            source = Image.open(im)
            layer = "below"
        # Add images
        fig.add_layout_image(
            dict(
                source=source,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=width,
                sizey=height,
                sizing="contain",
                layer=layer,
            )
        )
    if update_ranges:
        width, height = [
            max([Image.open(im).size[i] for im in images]) for i in range(2)
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

app.layout = html.Div([
    html.P(
        children="Image folder path"
    ),
    dcc.Input(
        id="image-folder-path",
        type="text",
        value="data/EM/EMBL/raw/"
    ),
    html.P(
        id="image-folder-list"
    ),
    dcc.Input(
        id="annotation-folder-path",
        type="text",
        value="data/EM/EMBL/labels/"
    ),
    dcc.Slider(
        id="slice-id",
        min=0,
        max=63,
        step=1,
        value=10,
        marks={
            0: '0',
            30: '30',
            63: '63'
        },
    ),
    html.Div(id='slider-output-container'),
    dcc.Graph(
        id="graph",
        figure=make_default_figure(),
        config={
            "modeBarButtonsToAdd": [
                "drawrect",
                "drawopenpath",
                "eraseshape",
            ]
        }
    ),
    dcc.Store(id="image-slices"),
    dcc.Store(id="slice-number-top", data=0),
    dcc.Store(id="slice-number-side", data=0)
])

@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    [dash.dependencies.Input('slice-id', 'value')])
def update_output(value):
    return 'Slice {}'.format(value)

@app.callback(
    Output('graph', 'figure'),
    [
        Input('image-folder-path', 'value'), 
        Input('annotation-folder-path', 'value'), 
        Input('slice-id', 'value')
    ])
def update_fig(imageFolder, annotationFolder, sliceId):
    raw = [x for x in get_folder_list(imageFolder) if str(sliceId).zfill(4) in x][0]
    ann = [x for x in get_folder_list(annotationFolder) if str(sliceId).zfill(4) in x][0]
    fig = make_default_figure()
    add_layout_images_to_fig(fig, [raw, ann])
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)