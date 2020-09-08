import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from app.app import app
from app.Dataset import Dataset
from app.DatasetStore import DatasetStore

from app.components.BasicComponent import BasicComponent

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


class Viewer2D(BasicComponent):
    def __init__(self, main_id, default_dataset):
        self.main_id = main_id
        self.default_dataset = default_dataset
        app.callback(
            Output(f'{main_id}-slice-info', 'children'),
            [Input(f'{main_id}-slice-id', 'value')]
        )(Viewer2D.update_slice_info)
        app.callback([
            Output(f'{main_id}-slice-id', 'min'),
            Output(f'{main_id}-slice-id', 'max'),
            Output(f'{main_id}-slice-id', 'marks'),
            Output(f'{main_id}-slice-id', 'value')
        ], [Input(f'{main_id}-dataset-dimensions', 'data')]
        )(Viewer2D.update_slider)
        app.callback(
            Output(f'{main_id}-graph', 'figure'),
            [
                Input(f'{main_id}-current-segmentation-name', 'data'),
                # TODO
                Input('selected-dataset-name', 'value'),
                Input(f'{main_id}-slice-id', 'value')
            ]
        )(Viewer2D.update_fig)


    def layout(self):
        return html.Div([
            html.Div(
                id=f'{self.main_id}-slice-info'
            ),
            dcc.Slider(
                id=f'{self.main_id}-slice-id'
            ),
            html.Div(id=f'{self.main_id}-slider-output-container'),
            dcc.Graph(
                id=f'{self.main_id}-graph',
                figure=make_default_figure(),
                config={
                    "modeBarButtonsToAdd": [
                        "drawrect",
                        "drawopenpath",
                        "eraseshape",
                    ]
                },
            ),
            dcc.Store(id=f'{self.main_id}-current-segmentation-name'),
            dcc.Store(id=f'{self.main_id}-current-dataset-name', data=self.default_dataset),
            dcc.Store(id=f'{self.main_id}-dataset-dimensions', data={"min": 0, "max": 63}),
            dcc.Store(id=f'{self.main_id}-slice-number-top', data=0),
            dcc.Store(id=f'{self.main_id}-slice-number-side', data=0),
        ])

    @staticmethod
    def update_slice_info(value):
        return 'Slice {}'.format(value)

    @staticmethod
    def update_slider(dimensions):
        min_slice = dimensions["min"]
        max_slice = dimensions["max"]
        marks = { i: str(i) for i in range(0, max_slice, 10)}
        return min_slice, max_slice, marks, 10
    
    @staticmethod
    def update_fig(current_segmentation, current_dataset, slice_id):
        if not current_dataset or not slice_id:
            raise PreventUpdate
        fig = make_default_figure()
        add_layout_images_to_fig(fig=fig, segmentation=current_segmentation, dataset=DatasetStore.get_dataset(current_dataset), slice_id=slice_id)
        return fig

