import dash
import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    url_base_pathname="/dash/",
    # remove Updating... message
    update_title=None,
)
server = app.server
app.config.suppress_callback_exceptions = True
