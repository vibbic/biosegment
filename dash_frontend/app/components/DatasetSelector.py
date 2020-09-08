import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

dataset_selector_layout = dbc.Card([
    dbc.FormGroup(
        [
            dbc.Label("Selected dataset"),
            dcc.Dropdown(
                id="selected-dataset-name",
            ),
        ]
    )
], body=True)