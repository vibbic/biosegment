import dash_bootstrap_components as dbc
import dash_core_components as dcc
from enum import IntEnum, auto
# Use IntEnum so ANNOTATION_MODE can be saved in Dash Store as int and still be compared
class ANNOTATION_MODE(IntEnum):
     NOT_EDITING = auto()
     EDITING = auto()

def dropdown_with_button(
    dropdown_id,
    button_id,
    dropdown_width=9,
    button_text="Refresh",
    button_size="sm",
    button_width=3,
    multi=False,
):
    return dbc.Row(
        [
            dbc.Col(dcc.Dropdown(id=dropdown_id, multi=multi), width=dropdown_width),
            dbc.Col(
                dbc.Button(button_text, id=button_id, size=button_size),
                width=button_width,
            ),
        ]
    )
