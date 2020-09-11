import dash_core_components as dcc
import dash_bootstrap_components as dbc


def dropdown_with_button(dropdown_id, button_id, dropdown_width=9, button_text="Refresh", button_size="sm", button_width=3):
    return dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id=dropdown_id
            ), 
            width=dropdown_width,
        ),
        dbc.Col(
            dbc.Button(button_text, id=button_id, size=button_size), 
            width=button_width,
        )
    ])
