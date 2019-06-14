import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import spot, futures  ,d1_board ,spreads, d1_history

style={'color':'#808285','padding':'10px 15px','font-size':'20px',
                                'text-align':'center','display':'inline-block','text-decoration':'None','marginRight':'7px'}

nav_menu = html.Div(style={'margin-bottom':'1px'},children=[
                    html.A('Spot', href='/apps/spot',style=style,id='spot-link'),
                    html.A('Futures', href='/apps/futures',style=style,id='fut-link'),
                    html.A('Spreads', href='/apps/spreads',style=style,id='spr-link'),
                    html.A('Delta One', href='/apps/d1_board',style=style,id='d1-link'),
                    html.A('D1 History', href='/apps/d1_history',style=style,id='d1h-link'),

                    ],
                    )
app.title = 'Home'
app.layout = html.Div(style={'margin-bottom':'2px'},children=[
    dcc.Location(id='url',refresh=False),
    html.Div(className='row',children=[nav_menu],
    style={'background-image':'url("/assets/diginex_inline_logo.svg")',
    'background-repeat': 'no-repeat','background-position': '98% 35%','background-size': '300px 30px',
    'height':'5%','top':'0','width':'100%','position':'fixed','zIndex':9999,'opacity':0.9,
    'background-color':'#FFFFFF','border-bottom':'1px solid #cb1828'}),
    html.Div(id = 'page-content',style={'margin-top':'5%','zIndex':-1})
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/spot':
        app.title = spot.title
        return spot.layout
    elif pathname == '/apps/futures':
        app.title = futures.title
        return futures.layout
    elif pathname == '/apps/spreads':
        app.title = spreads.title
        return spreads.layout
    elif pathname == '/apps/d1_board':
        app.title = d1_board.title
        return d1_board.layout
    elif pathname == '/apps/d1_history':
        app.title = d1_history.title
        return d1_history.layout
    else:
        return html.Div(html.Img(src='/assets/diginex_chain_logo.svg',
                        style={'opacity':0.4,'display':'block','margin-left':'auto','margin-right':'auto',
                        'margin-top':'13%','margin-bottom':'auto','width':'60%','height':'60%'}),
                        style={'height':'100%','width':'100%'})

if __name__ == '__main__':
    app.run_server(threaded = True, debug = True)
