import dash
import dash_bootstrap_components as dbc

external_stylesheets =  ['/assets/my-styling.css'] #['https://codepen.io/chriddyp/pen/bWLwgP.css'] #

app = dash.Dash(__name__,external_stylesheets = external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True

