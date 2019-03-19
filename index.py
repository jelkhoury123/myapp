import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import hvg, skew, hiv, pricer, order_book, futures


app.layout = html.Div([
    dcc.Location(id='url',refresh=False),
    html.Div(id = 'page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/hvg':
         return hvg.layout
    elif pathname == '/apps/skew':
         return skew.layout
    elif pathname == '/apps/hiv':
         return hiv.layout
    elif pathname == '/apps/pricer':
         return pricer.layout
    elif pathname == '/apps/order_book':
         return order_book.layout
    elif pathname == '/apps/futures':
         return futures.layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)