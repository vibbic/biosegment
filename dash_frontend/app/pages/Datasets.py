import dash_bootstrap_components as dbc
import dash_html_components as html
from app import api
from app.app import app
from dash.dependencies import Input, Output

title = "Datasets"
path = "/datasets"

layout = html.Div(
    [
        html.P(children="Datasets",),
        html.Button("Update", id="update-button"),
        html.Div(id="datasets",),
    ]
)


def get_card_for_dataset(p):
    # print(p)
    card = dbc.Card(
        dbc.CardBody(
            [
                html.H4(p["title"], className="card-title"),
                # html.H6("Card subtitle", className="card-subtitle"),
                html.P(p["description"], className="card-text",),
                dbc.CardLink("Card link", href="#"),
                dbc.CardLink("External link", href="https://google.com"),
            ]
        ),
        # style={"width": "18rem"},
    )
    return card


@app.callback([Output("datasets", "children"),], [Input("update-button", "n_clicks")])
def update_datasets(clicks):
    datasets = api.dataset.get_multi()
    return [[get_card_for_dataset(p) for p in datasets]]


if __name__ == "__main__":
    app.layout = layout
    app.run_server(debug=True)
