import dash_html_components as html
import dash_core_components as dcc

dataset_selector_layout = html.Div(
[
    html.H1("Viewer 2D"),
    html.P(
        "Selected dataset"
    ),
    dcc.Dropdown(
        id="selected-dataset-name",
    ),
]
)