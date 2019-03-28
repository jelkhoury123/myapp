import dash
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,external_stylesheets = external_stylesheets)

app.layout = html.Div(id ='page',style={
  'background-image': 'url("/assets/diginex2.png")',
  'background-repeat': 'no-repeat',
  'background-position': 'right top',
  'background-size': '300px 30px',
  'overflow':'hidden'},
  children = [
    html.H1('Hello World'),
    html.P('This image has an image in the background')
])

app.run_server(port = 8059,debug = True)

'''

'''