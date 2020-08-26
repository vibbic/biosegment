import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

from app import app
from Dataset import Dataset
from DatasetStore import DatasetStore

title="Viewer 2D"
path="/"

DEFAULT_DATASET="EMBL Raw"

store = DatasetStore()
dataset = store.get_dataset(DEFAULT_DATASET)

def add_layout_images_to_fig(fig, store, slice_id, update_ranges=True):
    """ images is a sequence of PIL Image objects """
    images = [("slice", store.get_slice(slice_id)), ("label", store.get_label(slice_id))]
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

layout = html.Div([
    html.P(
        children="Selected dataset"
    ),
    dcc.Dropdown(
        id="selected-dataset-name",
        options=[
            {"label": name, "value": name} for name in store.get_names_available()
        ],
        value=DEFAULT_DATASET
    ),
    html.Div(
        id="slice-info"
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
        },
    ),
    dcc.Store(id="current-dataset-name", data=DEFAULT_DATASET),
    dcc.Store(id="dataset-dimensions", data={"min": 0, "max": 63}),
    dcc.Store(id="slice-number-top", data=0),
    dcc.Store(id="slice-number-side", data=0),
])

@app.callback(
    Output('slice-info', 'children'),
    [Input('slice-id', 'value')])
def update_slice_info(value):
    return 'Slice {}'.format(value)

@app.callback([
    Output('slice-id', 'min'),
    Output('slice-id', 'max'),
    Output('slice-id', 'marks'),
    Output('slice-id', 'value')
], [Input('dataset-dimensions', 'data')])
def update_slider(dimensions):
    min_slice = dimensions["min"]
    max_slice = dimensions["max"]
    marks = { i: str(i) for i in range(0, max_slice, 10)}
    return min_slice, max_slice, marks, 10

@app.callback([
    Output('current-dataset-name', 'data'),
    Output('dataset-dimensions', 'data'),
], [Input('selected-dataset-name', 'value')])
def change_dataset(name):
    dataset = store.get_dataset(name)
    return name, dataset.get_dimensions()

@app.callback(
    Output('graph', 'figure'),
    [
        Input('current-dataset-name', 'data'),
        Input('slice-id', 'value')
    ]
    )
def update_fig(current_dataset, slice_id):
    fig = make_default_figure()
    add_layout_images_to_fig(fig, store.get_dataset(current_dataset), slice_id)
    return fig

if __name__ == '__main__':
    app.layout = layout
    app.run_server(debug=True)