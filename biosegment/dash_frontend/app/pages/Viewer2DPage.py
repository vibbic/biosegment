import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app.app import app
from app.DatasetStore import DatasetStore
from app.components.Viewer2D import Viewer2D

# import app.api.base

title="Viewer 2D"
path="/"

DEFAULT_DATASET="EMBL Raw"
VIEWER_ID = "viewer-1"

viewer = Viewer2D(main_id="viewer-1", default_dataset=DEFAULT_DATASET)

layout = html.Div([
    html.P(
        children="Selected dataset"
    ),
    dcc.Dropdown(
        id="selected-dataset-name",
        options=[
            {"label": name, "value": name} for name in DatasetStore.get_names_available()
        ],
        value=DEFAULT_DATASET
    ),
    viewer.layout(),
])

@app.callback([
    Output(f'{VIEWER_ID}-current-dataset-name', 'data'),
    Output(f'{VIEWER_ID}-dataset-dimensions', 'data'),
], [Input('selected-dataset-name', 'value')])
def change_dataset(name):
    dataset = DatasetStore.get_dataset(name)
    return name, dataset.get_dimensions()

if __name__ == '__main__':
    app.layout = layout
    app.run_server(debug=True)