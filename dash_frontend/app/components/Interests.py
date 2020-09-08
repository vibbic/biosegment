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
NUM_LABEL_CLASSES = 2
DEFAULT_LABEL_CLASS = 0
class_label_colormap = px.colors.qualitative.Light24
class_labels = list(range(NUM_LABEL_CLASSES))
# we can't have less colors than classes
assert NUM_LABEL_CLASSES <= len(class_label_colormap)

def class_to_color(n):
    return class_label_colormap[n]

def color_to_class(c):
    return class_label_colormap.index(c)

interests_layout = dbc.Card([
        html.H6("Label class"),
        # Label class chosen with buttons
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
        html.H6(id="stroke-width-display"),
        # Slider for specifying stroke width
        dcc.Slider(
            id="stroke-width",
            min=0,
            max=6,
            step=0.1,
            value=DEFAULT_STROKE_WIDTH,
        ),
    ],
    body=True,
    id="right-column",
)

