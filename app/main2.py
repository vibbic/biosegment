import json
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image

import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def make_default_figure(
    images=["data/EM/EMBL/raw/data_0000.png"],
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
    
    
    def add_layout_images_to_fig(fig, images, update_ranges=True):
        """ images is a sequence of PIL Image objects """
        if len(images) <= 0:
            return fig
        for im in images:
            # if image is a path to an image, load the image to get its size
            width, height = Image.open(im).size
            # Add images
            fig.add_layout_image(
                dict(
                    source=Image.open(im),
                    xref="x",
                    yref="y",
                    x=0,
                    y=0,
                    sizex=width,
                    sizey=height,
                    sizing="contain",
                    layer="above",
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

    fig = dummy_fig()
    add_layout_images_to_fig(fig, images)
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
    html.P(
        children="Annotation folder path"
    ),
    dcc.Input(
        id="annotation-folder-path",
        type="text",
        value="data/EM/EMBL/labels/"
    ),
    html.P(
        id="annotation-folder-list"
    ),
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
    Output('image-folder-list', 'children'),
    [Input('image-folder-path', 'value')])
def get_image_list(folderName):
    return get_folder_list(folderName)

@app.callback(
    Output('annotation-folder-list', 'children'),
    [Input('annotation-folder-path', 'value')])
def get_annotation_list(folderName):
    return get_folder_list(folderName)

def get_folder_list(folderName):
    files = []
    # r=root, d=directories, f = files
    for r, _, f in os.walk(folderName):
        for file in f:
            if '.png' in file:
                files.append(os.path.join(r, file))
    return files

if __name__ == '__main__':
    app.run_server(debug=True)