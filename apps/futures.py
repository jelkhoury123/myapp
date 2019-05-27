import dash
import dash_core_components as dcc 
import dash_html_components as html 
import dash_table 
from dash_table.Format import Format, Scheme, Sign, Symbol
import dash_table.FormatTemplate as FormatTemplate
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import itertools
import json
import datetime as dt 
import os
import sys
sys.path.append('..') # add parent directory to the path to import app
import deribit_api3 as my_deribit
from requests.auth import HTTPBasicAuth
import fob 
from app import app  # app is the main app which will be run on the server in index.py

# If websocket use diginex.ccxt library and reduce update interval from 7 to 5 secs

ENABLE_WEBSOCKET_SUPPORT = False
refresh_rate = 3 if ENABLE_WEBSOCKET_SUPPORT else 4
if ENABLE_WEBSOCKET_SUPPORT:
    import diginex.ccxt.websocket_support as ccxt
else:
    import ccxt
    
# Define my world
deriv_exchanges =['deribit','bitmex']

exchanges =  deriv_exchanges 

api_keys = json.loads(os.environ.get('api_keys'))
api_secrets = json.loads(os.environ.get('api_secrets'))

auth = HTTPBasicAuth(api_keys['deribit'], api_secrets['deribit'])

exch_dict={}
for x in exchanges:
    exec('exch_dict[x]=ccxt.{}({{"apiKey": "{}", "secret": "{}"}})'.format(x, api_keys[x], api_secrets[x]))
    #exec('exch_dict[x]=ccxt.{}()'.format(x))
for xccxt in exch_dict.values():
    xccxt.load_markets()

deribit = exch_dict['deribit']
bitmex = exch_dict['bitmex']

Xpto_main = ['BTC','ETH']
Xpto_alt = ['ADA','BCH','EOS','LTC','TRX','XRP']
Xpto_sym = {i:i for i in Xpto_alt}
Xpto_sym.update({'BTC':'฿','ETH':'⧫'})
            
instruments, inversed, ticks = fob.get_d1_instruments()
deribit_d1_ins , bitmex_d1_ins = instruments['deribit'], instruments['bitmex']

title = 'Futures'

layout =  html.Div(style={'marginLeft':35,'marginRight':35},
                    children=[ 
                        html.Div(className='row',style={'margin-top':'2px'},children=[
                                        html.Div(className='three columns',
                                        children =[html.H6('Choose Base'),
                                            dcc.RadioItems(id='choose-base',
                                                        options = [{'label':base,'value':base} for base in Xpto_main+['ALT']],
                                                        value = Xpto_main[0],
                                                        labelStyle={'display':'inline-block','margin-top':'10px','margin-bottom':'10px'}),
                                            # Testing exchange dropdown
                                            dcc.Dropdown(id='fut-sup-exchanges',
                                                         style={'border-color':'#cb1828'},
                                                         multi = True),
                                            # Until Here
                                            html.Hr(style={'border-color':'#cb1828'}),
                                            html.H6('Choose Instrument'),
                                            dcc.Dropdown(id='fut-ins',
                                                        style={'border-color':'#cb1828'}),
                                            #html.H6('Book Params:'),
                                            html.Hr(style={'border-color':'#cb1828'}),
                                            html.Div(className='row',children=[
                                            html.Div(className='two columns',children = [html.Label( 'X :')]),
                                            html.Div(className='four columns',children=[
                                                        dcc.RadioItems(id='fut-x-scale',
                                                        options=[{'label':scale,'value':scale} for scale in ['Rel','Abs']], 
                                                        value ='Rel',
                                                        labelStyle={'display':'inline-block'})]),
                                            html.Div(className='two columns',children = [html.Label('Y :')]),
                                            html.Div(className='four columns',children=[
                                                        dcc.RadioItems(id='fut-y-scale',
                                                        options=[{'label':j,'value':i} for i,j in {'Ccy':'Quote','Coin':'Base'}.items()], 
                                                        value ='Ccy',
                                                        labelStyle={'display':'inline-block'})])
                                            ]),
                                            html.Div(className='row',children=[ html.Br(),
                                            html.Div(className='three columns',children =[html.Label('Cutoff % :')]),
                                            html.Div(className='three columns',style={'width' :'50%','align':'right'},children =[
                                                dcc.Slider(id='fut-cutoff',
                                                    min=.05,max=.3,step=.05,value=.1,
                                                    marks={round(j,2): str(round(j,2)) for j in list(np.arange(.05,.35,.05))})]),
                                            ]),
                                            html.Div(className='row',children=[html.Br(),
                                            html.Div(className='three columns',children =[html.Label('Price Agg (bps):')]),
                                            html.Div(className='three columns',style={'width' :'50%','align':'right'},children =[
                                                dcc.Slider(id='fut-agg-level',
                                                    marks = {i:str(10**(i-2) if i != 0 else 0) for i in range(0,5)},
                                                                                max = 4,
                                                                                value = 1,
                                                                                step = 1)]),
                                            ]),
                                            html.Hr(style={'border-color':'#cb1828'}),        
                                            #html.H6('Book Charts'),
                                            dcc.Graph(id='fut-order-book-chart'),
                                            html.Hr(style={'border-color':'#cb1828'}),
                                            dcc.Graph(id='fut-market-depth'),
                                            html.H6(id='fut-time')

                                        ]),
                                html.Div(className='five columns',
                                    children =[html.Div(className='row',
                                                        children=[
                                                            html.H6(id='fut-ins-name',children=[], style={'display':'inline-block'}),html.H6(children=['('], style={'display':'inline-block','margin-left':'5px'}),
                                                            html.H6(id='fut-exchanges',children=[i for i in fob.get_exchanges_for_ins('BTC/USD').keys()],style={'display':'inline-block'}),html.H6(children=[') -'], style={'display':'inline-block','margin-right':'5px'}),
                                                            html.H6(id='fut-ins-inv',children=[],)]),
                                                html.Hr(style={'border-color':'#cb1828'}),
                                                html.Div(children=[dash_table.DataTable(id='fut-order-table',
                                                    style_table={'border': '1px solid lightgrey','border-collapse':'collapse'},
                                                    style_cell={'textAlign':'center','width':'12%'},
                                                    style_data_conditional=[{
                                                        'if' : {'filter':  'side eq "bid"' },
                                                        'color':'blue'
                                                                }
                                                        ]+[
                                                        {
                                                        'if' : {'filter': 'side eq "ask"' },
                                                        'color':'rgb(203,24,40)'
                                                    }]+[
                                                        { 'if': {'row_index':'odd'},
                                                        'backgroundColor':'rgb(242,242,242)'}
                                                    ]+[
                                                        {'if':{'column_id':'price'},
                                                        'fontWeight':'bold',
                                                        'border': 'thin lightgrey solid'}
                                                    ]+[{'if':{'column_id':'from_mid'},
                                                        'fontWeight':'bold'}
                                                    ]+[
                                                        {'if' : {'column_id': c },
                                                        'textAlign': 'right',
                                                        'padding-right' : '2%'
                                                        } for c in ['size','cum_size','size_$','cum_size_$']
                                                    ],
                                                    style_header={'textAlign':'center','backgroundColor':'#DCDCDC','fontWeight':'bold'},
                                                    #style_as_list_view=True
                                                )
                                                ]),
                                                html.Div( className = 'row', children = [
                                                    html.P(id = 'fut-ob-last-timestamp', style = {'display':'inline-block','font-size':'1.2rem'}),
                                                    html.P(children=' / ', style = {'display':'inline-block', 'margin':'0px 5px','font-size':'1.2rem'}),
                                                    html.P(id = 'fut-ob-new-timestamp', children = dt.datetime.now().strftime('%X'), style = {'display':'inline-block','font-size':'1.2rem'}),
                                                ]),
                                                html.Hr(style={'border-color':'#cb1828'}),
                                                html.H6('Liquidity Metrics'),
                                                html.P(id='fut-liquidity-table'),
                                                html.H6('Slippage %'),
                                                html.P(id='fut-depth-table'),
                                                html.H6('Coin Stats'),
                                                html.P(id='fut-stat-table')]),

                                html.Div(className='four columns',
                                    children =[html.H6('Manage Orders'),
                                    html.Br(),
                                    html.Hr(style={'border-color':'#cb1828'}),
                                    html.Label('Order'),
                                    html.Hr(style={'border-color':'#cb1828'}),
                                    html.P(id = 'fut-diplay-tick', style={'font-size':'1.2rem', 'font-weight':'bold'}),
                                    html.Div(style={'height':'100px'},children = [dash_table.DataTable(
                                        id ='fut-order-to-send',
                                        columns = [{'id':'B/S','name':'B/S','presentation':'dropdown'}]+[{'id':c,'name':c} for c in ['Ins','Qty','Limit price','exc']],
                                        data = [{'B/S':'B',**{p:0 for p in ['Ins','Qty','Limit price','exc']}}],
                                        style_table={'border': '1.5px solid','border-color':'#cb1828','border-collapse':'collapse'},
                                        style_header={'fontWeight':'bold'},
                                        style_cell={'textAlign':'center','width':'12%'},
                                        style_as_list_view=True,
                                        editable=True,
                                        column_static_dropdown=[
                                            {
                                                'id': 'B/S',
                                                'dropdown': [
                                                    {'label': i, 'value': i}
                                                    for i in ['B','S']
                                                ]
                                            }]
                                        )]),
                                        html.Div(id='active'),
                                        html.Button(id='send-order',style={'display':'none'}),
                                        dcc.ConfirmDialog(id='confirm',message = 'Submit Order ? '),
                                        html.Div(id='output-confirm',style={'margin-top':'30px'},children=['Here']),
                                        html.Div([dcc.Tabs(id='orders',className = 'custom-tabs-container',parent_className='custom-tabs',
                                         children=[
                                            dcc.Tab(label='Open Orders',className='custom-tab', selected_className='custom-tab--selected',
                                            style = {'overflow':'hidden'},
                                            children = [
                                                html.Hr(style={'border-color':'#cb1828'}),
                                                html.Div(style = {'overflow':'hidden'},
                                                children = [dash_table.DataTable(
                                                id ='open-orders', 
                                                style_table={'border': '0.5px solid','border-color':'#cb1828','margin' :'2px',
                                                'border-collapse':'collapse'},
                                                style_header={'fontWeight':'bold'},
                                                style_cell={'textAlign':'center','width':'12%'},
                                                style_as_list_view=True,
                                                style_data_conditional=[
                                                                { 'if': {'row_index':'odd'},
                                                                'backgroundColor':'rgb(242,242,242)'}
                                                            ]
                                                )]),
                                                html.Div(id='cancel-confirm',style={'margin-top':'30px'},children=['Here'])
                                            ]),
                                            dcc.Tab(label='Closed Orders',className='custom-tab', selected_className='custom-tab--selected',
                                            children = [
                                                dcc.DatePickerSingle(
                                                    id='fut-go-back-date',
                                                    max_date_allowed=dt.datetime(pd.to_datetime('today').year, pd.to_datetime('today').month, pd.to_datetime('today').day),
                                                    date=dt.datetime(pd.to_datetime('today').year, pd.to_datetime('today').month, pd.to_datetime('today').day),
                                                    display_format='D-M-Y',
                                                    style = {'margin-top':'10px'},
                                                ),
                                                html.Hr(style={'border-color':'#cb1828', 'margin-top':'10px'}),
                                                html.Div(style = {'overflow':'hidden'},
                                                children = [dash_table.DataTable(
                                                id ='closed-orders',
                                                style_table={'border': '0.5px solid','border-color':'#cb1828','margin' :'2px',
                                                'border-collapse':'collapse'},
                                                style_header={'fontWeight':'bold'},
                                                style_cell={'textAlign':'center','width':'12%'},
                                                style_as_list_view=True,
                                                style_data_conditional=[
                                                                { 'if': {'row_index':'odd'},
                                                                'backgroundColor':'rgb(242,242,242)'}
                                                            ]
                                                )]),
                                            ])
                                        ])]),
                                        html.Hr(style={'border-color':'#cb1828'}),
                                        html.H6('Balances'),
                                        html.Div(id='balances')
                                    ]),
                                html.Div(id='the-fut-data',style={'display':'none'}),
                                dcc.Interval(
                                    id='fut-interval-component',
                                    interval = refresh_rate * 1000, # in milliseconds= 7 or 10 seconds
                                    n_intervals=0
                                    ),
                                dcc.Interval(
                                    id='fut-second',
                                    interval = 1000,
                                    n_intervals=0
                                )     
                        ]),
                        ])

@app.callback([Output('fut-sup-exchanges','options'),Output('fut-sup-exchanges','value')],
            [Input('choose-base','value')])
def update_sup_exchs(radio_button_value):
    sup_exchanges = [i for i in instruments if radio_button_value in instruments[i].keys()]
    options_sup_exchanges=[{'label':exch,'value':exch} for exch in sup_exchanges]
    value_sup_exchanges=sup_exchanges
    return options_sup_exchanges, value_sup_exchanges

@app.callback([Output('fut-ins','options'),Output('fut-ins','value')],
            [Input('choose-base','value'),Input('fut-sup-exchanges','value')])
def update_ins(radio_button_value, exch_dropdown_value):
    ins = []
    for exch in exch_dropdown_value:
        ins.extend(instruments[exch][radio_button_value])
    options=[{'label':ins,'value':ins} for ins in ins]
    value=ins[0]
    return options, value

@app.callback([Output('fut-exchanges','children'), Output('fut-ins-name','children'), Output('fut-ins-inv','children'), Output('fut-ins-inv','style')],
            [Input('fut-ins','value')])
def update_exchanges_options(ins):
    inv = 'Inverse' if inversed[ins] else 'Non-Inverse'
    style = {'color': 'red', 'display':'inline-block','font-weight':'bold'} if inversed[ins] else {'color': 'green', 'display':'inline-block','font-weight':'bold'}
    return [exch for exch in fob.get_exchanges_for_ins(ins).keys()],ins,inv, style

@app.callback(Output('the-fut-data','children'),
            [Input('fut-ins','value'),Input('fut-exchanges','children'),Input('fut-interval-component','n_intervals')])
def update_data(ins,ex,n):
    now = dt.datetime.now()
    ex = {x:exch_dict[x] for x in ex}
    order_books = fob.get_order_books(ins,ex,2000)
    save_this = (order_books,now.strftime("%Y-%m-%d  %H:%M:%S"))
    return json.dumps(save_this)

@app.callback(Output('fut-time','children'),
            [Input('fut-second','n_intervals'),Input('the-fut-data','children')])
def update_time(n,order_books):
    time_snap = json.loads(order_books)[1]
    return (dt.datetime.now()-dt.datetime.strptime(time_snap,"%Y-%m-%d  %H:%M:%S")).seconds

@app.callback([Output('fut-order-book-chart','figure'),Output('fut-market-depth','figure'),
            Output('fut-order-table','data'),Output('fut-order-table','columns'),
            Output('fut-liquidity-table','children'),
            Output('fut-depth-table','children'),Output('fut-stat-table','children'),
            Output('fut-ob-last-timestamp', 'children'), Output('fut-ob-new-timestamp', 'children')],
            [Input('the-fut-data','children'),Input('choose-base','value'),
            Input('fut-ins','value'),Input('fut-exchanges','children'),
            Input('fut-x-scale','value'),Input('fut-y-scale','value'),
            Input('fut-cutoff','value'),Input('fut-agg-level','value')],
            [State('fut-ob-new-timestamp', 'children')],)
def update_page(order_books,base,ins,exchanges,x_scale,y_scale,cutoff,step, last_update):

    #load data
    step = 10**(step-2)/10000 if step !=0 else step
    relative = x_scale == 'Rel'
    currency = y_scale == 'Ccy'
    order_books = json.loads(order_books)[0]
    order_books= {key:order_books[key] for key in order_books if key in exchanges}

    # 1. Plot Book and Depth
    book_plot = fob.plot_book(order_books,ins,exchanges,relative,currency,cutoff)
    depth_plot = fob.plot_depth(order_books,ins,exchanges,relative,currency,cutoff)

    # order table data
    df =  fob.build_book(order_books,ins,exchanges,cutoff,step)

    df_bids = df[[i for i in df.columns if 'bid' in i]].dropna().iloc[:13]
    df_bids['side'] = 'bid'
    df_bids.columns=['price','size','cum_size','size_$','cum_size_$','average_fill','exc','side']

    df_asks = df[[i for i in df.columns if 'ask' in i]].dropna().iloc[:13]
    df_asks['side'] = 'ask'
    df_asks.columns = ['price','size','cum_size','size_$','cum_size_$','average_fill','exc','side']
    
    df_all=pd.concat([df_asks.sort_values(by='price',ascending=False),df_bids]).rename_axis('from_mid')
    df_all=df_all.reset_index()
    df_all['from_mid'] = (df_all['from_mid']-1)

    data_ob = df_all.to_dict('rows')
    ob_new_time = dt.datetime.now().strftime('%X')

    # Get rounidng digits for the table
    precision = len(str(ticks[ins]).split('.')[1]) if '.' in str(ticks[ins]) else int(str(ticks[ins]).split('e-')[1])
    mid=(df_bids['price'].max() + df_asks['price'].min())/2
    rounding = max(min(int(np.ceil(-np.log(mid*step)/np.log(10))), precision),0) if step !=0 else precision
    r = int(np.ceil(-np.log(step)/np.log(10)))-2 if step !=0 else int(np.ceil(-np.log(10**-precision/mid)/np.log(10))-2)
    
    base_sym = Xpto_sym[base] if base!='ALT' else Xpto_sym[ins[:3]]
    quote_sym = '$' if inversed[ins] else '฿'
    columns_ob=[{'id':'from_mid','name':'From Mid','type':'numeric','format':FormatTemplate.percentage(r).sign(Sign.positive)},
                {'id':'price','name':'Price','type':'numeric','format':Format(precision=rounding,scheme=Scheme.fixed)},
                {'id':'size','name':'Size ({})'.format(base_sym),'type':'numeric','format':Format(precision=2,scheme=Scheme.fixed)},
                {'id':'cum_size','name': 'Total ({})'.format(base_sym),'type':'numeric','format':Format(precision=2,scheme=Scheme.fixed)},
                {'id':'size_$','name':'Size ({})'.format(quote_sym),'type':'numeric',
                'format':FormatTemplate.money(0) if inversed[ins] else Format(precision=2,scheme=Scheme.fixed,symbol=Symbol.yes,symbol_prefix=u'฿')},
                {'id':'cum_size_$','name':'Total ({})'.format(quote_sym),'type':'numeric',
                'format':FormatTemplate.money(0) if inversed[ins] else Format(precision=2,scheme=Scheme.fixed,symbol=Symbol.yes,symbol_prefix=u'฿')},
                {'id':'average_fill','name':'Averge Fill','type':'numeric',
                'format':Format(precision=rounding,scheme=Scheme.fixed)},
                {'id':'exc','name':'Exchange','hidden':True},
                {'id':'side','name':'side','hidden':True}]
    pair = base+'/USD' if inversed[ins] else base+'/BTC' if base!='ALT' else ins[:3]+'/BTC'
    try:
        
        liq_dfs = [df for df in fob.get_liq_params(df,pair,step)]
        col_format = {'bid':Format(precision=rounding,scheme=Scheme.fixed),
                      'ask':Format(precision=rounding,scheme=Scheme.fixed),
                      'mid':Format(precision=rounding,scheme=Scheme.fixed),
                      'spread':Format(precision=rounding,scheme=Scheme.fixed),
                      'spread%':FormatTemplate.percentage(2).sign(Sign.positive),
                      'cross%':FormatTemplate.percentage(2).sign(Sign.positive),
                      'arb%':FormatTemplate.percentage(2).sign(Sign.positive),
                      '24H % Volume':FormatTemplate.percentage(2).sign(Sign.positive),
                      'Market Cap M$':FormatTemplate.money(0),
                      '% 1h':FormatTemplate.percentage(2).sign(Sign.positive),
                      '% 24h':FormatTemplate.percentage(2).sign(Sign.positive),
                      '% 7d':FormatTemplate.percentage(2).sign(Sign.positive),}
        liq_tables = [dash_table.DataTable(
                data=liq_df.to_dict('rows'),
                columns=[{'id': c,'name':c,'type':'numeric','format':col_format[c] if c in col_format.keys() else None} for c in liq_df.columns],
                style_header={'backgroundColor':'#DCDCDC','fontWeight':'bold'},
                style_cell={'textAlign':'center','width':'10%'},
                style_table={'border': '1px solid lightgrey','border-collapse':'collapse'},
                ) for liq_df in liq_dfs]
    except:
        liq_tables=[0]*3
 
    return (book_plot,depth_plot,data_ob,columns_ob) + tuple(liq_tables) + (last_update, ob_new_time)

@app.callback(Output('fut-diplay-tick','children'),
                [Input('fut-ins','value')],
)
def display_tick(ins):
    return 'Tick Size: {}'.format(ticks[ins])

@app.callback(Output('fut-order-to-send','data'),
            [Input('fut-order-table','active_cell'),Input('fut-ins','value'), Input('fut-agg-level','value')],
            [State('fut-order-table','data')])
def update_order(active_cell,ins,step,data):
    step = 10**(step-2)/10000 if step !=0 else step
    data_df=pd.DataFrame(data)
    row = data_df.iloc[active_cell[0]]
    columns = ['B/S','Ins','Qty','Limit price','exc']
    order_df=pd.DataFrame(columns=columns)
    size = (row['cum_size_$'] if active_cell[1] == 5 else row['size_$']) if inversed[ins] else (row['cum_size'] if active_cell[1] == 3 else row['size'])
    order_df.loc[0]=['B' if row['side']=='bid' else 'S',ins,size,row['price'],row['exc']]
    precision = len(str(ticks[ins]).split('.')[1]) if '.' in str(ticks[ins]) else int(str(ticks[ins]).split('e-')[1])
    rounding = max(min(int(np.ceil(-np.log(row['price']*step)/np.log(10))), precision),0) if step !=0 else precision
    order_df['Limit price'] = round(round(order_df['Limit price']/ticks[ins]) * ticks[ins],rounding)
    return order_df.to_dict('row')

@app.callback([Output('send-order','children'),Output('send-order','style')],
                [Input('fut-order-to-send','data')])
def update_button(order):
    side = pd.DataFrame(order).iloc[0]['B/S']
    side_text = 'Buy' if side == 'B' else 'Sell'
    text = 'Place {} Order'.format(side_text)
    invisible = any (i == 0 for i in pd.DataFrame(order).iloc[0])
    if not invisible:
        style = {'margin-top' :'10px','background-color':'{}'.format('blue' if side =='B' else '#cb1828'),'color':'#FFF'}
    else:
        style = {'display':'none'}
    return (text,style)


@app.callback(Output('confirm','displayed'),
            [Input('send-order','n_clicks')])
def display_confirm(n_clicks):
    return True

@app.callback(Output('output-confirm','children'),
[Input('confirm','submit_n_clicks')],
[State('fut-order-to-send','data')])
def update_output(submit_n_clicks,order):
    if submit_n_clicks:
        order = pd.DataFrame(order).iloc[0]
        side ='buy' if order['B/S'] =='B' else 'sell'
        exc = order['exc']
        exch_dict[exc].create_limit_order(order['Ins'],side, str(order['Qty']), str(order['Limit price']))
        return 'Order sent'

@app.callback([Output('open-orders','data'),Output('open-orders','columns')],
            [Input('fut-interval-component','n_intervals')])
def update_open(interval):
    oo = fob.get_open_orders()
    oo['Cancel']='X'
    data = oo.to_dict('rows')
    names=['Id','Type','Ins','B/S','Qty','Price','State','Filled','Exc','Cancel']
    columns =[{'id':id,'name':name} for id,name in zip(oo.columns,names)]
    columns[0]['hidden'] = True
    return data,columns

@app.callback(Output('cancel-confirm','children'),
            [Input('open-orders','active_cell')],
            [State ('open-orders','data')])
def cancel_order(active_cell,open_orders):
    oo=pd.DataFrame(open_orders)
    if active_cell[1] == 8:
        exc = oo.iloc[active_cell[0]]['ex']
        order_id = oo.iloc[active_cell[0]]['id']
        exch_dict[exc].cancel_order(order_id)
        return 'Canceled {}  {}'.format(order_id ,exc)
    else:
        return active_cell

@app.callback([Output('closed-orders','data'),Output('closed-orders','columns')],
            [Input('fut-interval-component','n_intervals'),Input('fut-go-back-date','date')])
def update_closed(interval,go_back_date):
    #year = pd.to_datetime('today').year
    #month = pd.to_datetime('today').month
    #day = pd.to_datetime('today').day
    #midnight = pd.to_datetime(str(year)+'-'+str(month)+'-'+str(day)).timestamp()*10**3
    go_back_date = go_back_date.split(' ')[0]
    from_this_date = dt.datetime.strptime(go_back_date, '%Y-%m-%d').timestamp()*10**3
    co = fob.get_closed_orders(from_this_date)
    #co['timestamp']=pd.to_datetime((co['timestamp']/10**3).round(0), unit='s')
    co['timestamp']=pd.to_datetime(co['timestamp']/1000, unit='s')
    co['timestamp']=co['timestamp'].dt.strftime('%d %b %H:%M:%S')
    data = co.to_dict('rows')
    names=['Id','Type','Ins','B/S','Qty','Price','Average','Filled','Time','Exc']
    columns =[{'id':id,'name':name} for id,name in zip(co.columns,names)]
    columns[0]['hidden'] = True
    return data,columns

@app.callback(Output('balances','children'),
            [Input('fut-interval-component','n_intervals')])
def update_balance(interval):
    b = fob.get_balances().reset_index().round(4)
    b.columns=['Ccy',*b.columns[1:]]
    return dash_table.DataTable(
                data=b.to_dict('rows'),
                columns=[{'id': c,'name':c} for c in b.columns],
                style_header={'backgroundColor':'#DCDCDC','fontWeight':'bold'},
                style_cell={'textAlign':'center','width':'10%'},
                style_table={'border': '1px solid lightgrey','border-collapse':'collapse'},
                )
