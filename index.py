import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import hvg, skew, pricer, order_book, futures ,hiv, spreads

style={'color':'#808285','padding':'10px 15px','font-size':'20',
                                'text-align':'center','display':'inline-block','text-decoration':'None','marginRight':'7px'}
nav_menu = html.Div(style={'margin-bottom':'1px'},children=[
                    html.A('HVG', href='/apps/hvg',style=style,id='hvg-link'),
                    html.A('Skew', href='/apps/skew',style= style,id='skew-link'),
                    html.A('HIV', href='/apps/hiv',style=style,id='hiv-link'),
                    html.A('Pricer', href='/apps/pricer',style=style,id='pricer-link'),
                    html.A('Order Book', href='/apps/order_book',style=style,id='ob-link'),
                    html.A('Futures', href='/apps/futures',style=style,id='fut-link'),
                    html.A('Spreads', href='/apps/spreads',style=style,id='spr-link')
                    ],
                    )
app.title = 'Home'
app.layout = html.Div(style={'margin-bottom':'2px'},children=[
    dcc.Location(id='url',refresh=False),
    html.Div(className='row',children=[nav_menu, html.Hr(style={'border-color':'#cb1828'})],
    style={'background-image':'url("/assets/diginex_inline_logo.svg")',
    'background-repeat': 'no-repeat','background-position': '98% 18%','background-size': '300px 30px',}),
    html.Div(id = 'page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/hvg':
        app.title = hvg.title
        return hvg.layout
    elif pathname == '/apps/skew':
        app.title = skew.title
        return skew.layout
    elif pathname == '/apps/hiv':
        app.title = hiv.title
        return hiv.layout
    elif pathname == '/apps/pricer':
        app.title = pricer.title
        return pricer.layout
    elif pathname == '/apps/order_book':
        app.title = order_book.title
        return order_book.layout
    elif pathname == '/apps/futures':
        app.title = futures.title
        return futures.layout
    elif pathname == '/apps/spreads':
        app.title = spreads.title
        return spreads.layout
    else:
        return html.Div(html.Img(src='/assets/diginex_chain_logo.svg',
                        style={'opacity':0.4,'display':'block','margin-left':'auto','margin-right':'auto',
                        'margin-top':'13%','margin-bottom':'auto','width':'60%','height':'60%'}),
                        style={'height':'100%','width':'100%'})

if __name__ == '__main__':
    app.run_server(debug=True,threaded = True)

'''
'background-color':'#cdccce',
'''