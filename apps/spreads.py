import dash
import dash_core_components as dcc 
import dash_html_components as html 
import dash_table 
from dash_table.Format import Format, Scheme, Sign, Symbol
import dash_table.FormatTemplate as FormatTemplate
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go 
import plotly.figure_factory as ff
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import json
import datetime as dt 
import os 
import sys
#sys.path.append('../dashapp') # add parent directory to the path to import app
import fob
from app import app

# If websocket use diginex.ccxt library and reduce update interval from 10 to 5 secs

ENABLE_WEBSOCKET_SUPPORT = False
refresh_rate = 5 if ENABLE_WEBSOCKET_SUPPORT else 5
if ENABLE_WEBSOCKET_SUPPORT:
    import diginex.ccxt.websocket_support as ccxt
else:
    import ccxt

# Define my world
deriv_exchanges =['bitmex','deribit']
spot_exchanges = ['coinbasepro','kraken','liquid','binance',
                    'bitbank','bittrex','kucoin2','poloniex','bitfinex']

exchanges =  deriv_exchanges + spot_exchanges

try:
    api_keys = json.loads(os.environ.get('api_keys'))
    api_secrets = json.loads(os.environ.get('api_secrets'))
except:
    api_keys = [None]
    api_secrets = [None]

exch_dict={}
for x in exchanges:
    if x in api_keys:
        exec('exch_dict[x]=ccxt.{}({{"apiKey": "{}", "secret": "{}"}})'.format(x, api_keys[x], api_secrets[x]))
    else:
        exec('exch_dict[x]=ccxt.{}()'.format(x))

for xccxt in exch_dict.values():
    xccxt.load_markets()

deriv_exch_dict ={key:value for key,value in exch_dict.items() if key in deriv_exchanges}
spot_exch_dict ={key:value for key,value in exch_dict.items() if key in spot_exchanges}

deribit = deriv_exch_dict['deribit']
bitmex = deriv_exch_dict['bitmex']

Xpto_main = ['BTC','ETH']
Xpto_alt = ['ADA','BCH','EOS','LTC','TRX','XRP']
Xpto_sym = {i:i for i in Xpto_alt}
Xpto_sym.update({'BTC':'฿','ETH':'⧫'})

instruments,inversed, ticks = fob.get_d1_instruments()
deribit_d1_ins , bitmex_d1_ins = instruments['deribit'], instruments['bitmex']


# HAVE TO INTEGRATE WITH FOB --------------------------------------------------------------------------
Sym = {i:i for i in Xpto_alt}
Sym.update({'BTC':'฿','ETH':'⧫'})

Fiat = ['USD','EUR','GBP','CHF','HKD','JPY','CNH']
Fiat_sym ={'USD':'$','EUR':'€','GBP':'£','CHF':'CHF','HKD':'HKD','JPY':'JP ¥','CNH':'CN ¥'}
Sym.update(Fiat_sym)

underlying = {}
for ex, ex_obj in deriv_exch_dict.items():
    for ins_set in instruments[ex].values():
        for ins in ins_set:
            market = getattr(ex_obj,'markets')[ins]
            underlying[ins]=market['base']+'/'+market['quote']

ticks_spot = {}
for ex, ex_obj in spot_exch_dict.items():
    ticks_spot[ex] = {}
    market = getattr(ex_obj,'markets')
    for pair in underlying:
        if pair in market.keys():
            base_precision = market[pair]['precision']['amount'] if not 'none' else 10
            quote_precision = market[pair]['precision']['price']
            ticks_spot[ex].update({pair:[base_precision,quote_precision]})
print(ticks_spot)

def build_stack_book(normalized_order_book,step=0.001):
    ''' gets order books aggreagtes them then normalizes
        returns a vertically stacked dataframe e.g. bids below asks
    '''
    df = normalized_order_book
    df_bids = df[[i for i in df.columns if 'bid' in i]].dropna().iloc[:13]
    df_bids['side'] = 'bid'
    df_bids.columns=['price','size','cum_size','size_$','cum_size_$','average_fill','exc','side']

    df_asks = df[[i for i in df.columns if 'ask' in i]].dropna().iloc[:13]
    df_asks['side'] = 'ask'
    df_asks.columns = ['price','size','cum_size','size_$','cum_size_$','average_fill','exc','side']

    mid=(df_bids['price'].max() + df_asks['price'].min())/2
    df_all=pd.concat([df_asks.sort_values(by='price',ascending=False),df_bids]).rename_axis('from_mid')
    df_all=df_all.reset_index()
    df_all['from_mid'] = (df_all['from_mid']-1)

    return df_all,mid

def process_ob_for_dashtable(base, ins, normalized_order_book, precision, step=.001):
    # order table data
    
    df_all, mid  = build_stack_book(normalized_order_book,step=step)
    data_ob = df_all.to_dict('rows')

    # order table columns
    rounding = max(min(int(np.ceil(-np.log(mid*step)/np.log(10))), precision),0) if step !=0 else precision
    r = int(np.ceil(-np.log(step)/np.log(10)))-2 if step !=0 else int(np.ceil(-np.log(10**-precision/mid)/np.log(10))-2)

    base_sym = Sym[base] if base!='ALT' else Sym[ins[:3]]
    quote_sym = Sym['USD'] if inversed[ins] else '฿'

    columns_ob=[{'id':'from_mid','name':'From Mid','type':'numeric','format':FormatTemplate.percentage(r).sign(Sign.positive)},
                {'id':'price','name':'Price','type':'numeric','format':Format(precision=rounding,scheme=Scheme.fixed)},
                {'id':'size','name':'Size ({})'.format(base_sym),'type':'numeric','format':Format(precision=2,scheme=Scheme.fixed)},
                {'id':'cum_size','name': 'Size Total ({})'.format(base_sym),'type':'numeric','format':Format(precision=2,scheme=Scheme.fixed)},
                {'id':'size_$','name':'Size ({})'.format(quote_sym),'type':'numeric',
                'format':FormatTemplate.money(0) if inversed[ins] else Format(precision=2,scheme=Scheme.fixed,symbol=Symbol.yes,symbol_prefix=u'฿')},
                {'id':'cum_size_$','name':'Size Total ({})'.format(quote_sym),'type':'numeric',
                'format':FormatTemplate.money(0) if inversed[ins] else Format(precision=2,scheme=Scheme.fixed,symbol=Symbol.yes,symbol_prefix=u'฿')},
                {'id':'average_fill','name':'Averge Fill','type':'numeric',
                'format':Format(precision=rounding,scheme=Scheme.fixed)},
                {'id':'exc','name':'Exchange','hidden':True},
                {'id':'side','name':'side','hidden':True}]
    return data_ob, columns_ob

# UNTIL HERE --------------------------------------------------------------------------

title = 'Spreads'

layout =  html.Div( className = 'row', style = {'marginLeft':35,'marginRight':35,'margin-top':'20px'}, children = [
    
    # Page Params
    html.Div( className = 'row', style = {'border-bottom':'1px solid #cb1828','padding-bottom':'30px','margin-bottom':'30px'}, children = [
        
        # Choose Base
        html.Div( className = 'two columns', children = [
            html.Label('Choose Base', style = {'margin-bottom':'10px','font-size':'1.4rem','font-weight':'bold'}),
            dcc.RadioItems( id = 'spread-base',
                            options = [{'label':base,'value':base} for base in Xpto_main+['ALT']],
                            value = Xpto_main[0],
                            labelStyle = {'display':'inline-block','margin-right':'10px','font-size':'1.2rem'}
            ),
        ]),

        # Price Aggregation
        html.Div( className = 'three columns', children = [
            html.Label('Price Agg (bps):', style = {'margin-bottom':'10px','font-size':'1.4rem','font-weight':'bold'}),
            html.Div( className = 'row', style = {'width':'75%','font-size':'1.2rem'}, children = [
                dcc.Slider( id='spread-agg-level',
                            max = 4, step = 1, value = 0,
                            marks = {i:str(10**(i-2) if i != 0 else 0) for i in range(0,5)}
                )
            ]),
        ]),

        # Cutoff
        html.Div( className = 'three columns', children = [
            html.Label('Cutoff % :', style = {'margin-bottom':'10px','font-size':'1.4rem','font-weight':'bold'}),
            html.Div( className = 'row', style = {'width':'75%','font-size':'1.2rem'}, children = [
                dcc.Slider( id = 'spread-cutoff',
                            min = .05, max = .3, step = .05, value = .1,
                            marks = {round(j,2): str(round(j,2)) for j in list(np.arange(.05,.35,.05))})
            ]),
        ]),
    ]),

    # Instrument Tables
    html.Div(className = 'row', children = [

        # Instrument (+)
        html.Div(className = 'four columns', children = [
            # +Params
            html.Div( className = 'row', children = [
                # Label
                html.Label('Instrument [+]', style = {'margin-bottom':'10px','font-size':'1.4rem'}),
                # Dropdown Exchanges
                html.Div( children = [
                    dcc.Dropdown(id = 'spread-insp-ex',
                                 style = {'border-color':'#cb1828'},
                                 options = [{'label':i, 'value':i} for i in instruments.keys()],
                                 value = list(instruments.keys()),
                                 multi = True,
                    )
                ]),
                # Dropdown Instruments
                html.Div( children = [
                    dcc.Dropdown(id = 'spread-insp',
                                 style = {'border-color':'#cb1828', 'margin-top':'15px'},
                                 options = [{'label':i, 'value':i} for i in instruments['deribit']['BTC']],
                                 value = instruments['deribit']['BTC'][0],
                    )
                ]),
            ]),
            #  +Tables
            html.Div( children=[
                html.Br(),
                dash_table.DataTable(id = 'spread-order-tablep',
                                     style_table = {'border': '1px solid lightgrey','border-collapse':'collapse'},
                                     style_cell = {'textAlign':'center','width':'12%'},
                                     style_header={'textAlign':'center','backgroundColor':'#DCDCDC','fontWeight':'bold'},
                                     style_data_conditional = [{'if' : {'filter':  '{side} eq "bid"' }, 'color':'blue' }] +
                                                              [{'if' : {'filter': '{side} eq "ask"' }, 'color':'rgb(203,24,40)' }] +
                                                              [{'if': {'row_index':'odd'}, 'backgroundColor':'rgb(242,242,242)'}] +
                                                              [{'if':{'column_id':'price'}, 'fontWeight':'bold','border': 'thin lightgrey solid'}] +
                                                              [{'if':{'column_id':'from_mid'}, 'fontWeight':'bold'}] +
                                                              [{'if' : {'column_id': c }, 'textAlign': 'right','padding-right' : '2%'} for c in ['size','cum_size','size_$','cum_size_$'] ],
                ),
                html.P(id = 'spread-obp-last-timestamp', style = {'display':'inline-block','font-size':'1.2rem'}),
                html.P(children=' / ', style = {'display':'inline-block', 'margin':'0px 5px','font-size':'1.2rem'}),
                html.P(id = 'spread-obp-new-timestamp', children = dt.datetime.now().strftime('%X'), style = {'display':'inline-block','font-size':'1.2rem'}),
            ]),
        ]),
        
        # Instrument (-)
        html.Div(className = 'four columns', children = [
            # -Params
            html.Div( className = 'row', children = [
                html.Div( className = 'row', children = [
                    # Label
                    html.Div( className = 'two-thirds column', children = [
                        html.Label('Instrument [-]', style = {'margin-bottom':'10px','font-size':'1.4rem'}),
                    ]),
                    # Radio Buttons
                    html.Div( className = 'one-third column', children = [
                        dcc.RadioItems( id = 'spread-negative-type',
                                        options = [{'label':i, 'value':i} for i in ('Spots', 'Futures')],
                                        value = 'Futures',
                                        labelStyle={'display': 'inline-block', 'font-size':'1.4rem', 'margin-left':'10px'},
                        ),
                    ]),
                ]),
                # Dropdown Exchanges
                html.Div( children = [
                    dcc.Dropdown(id = 'spread-insn-ex',
                                 style = {'border-color':'#cb1828'},
                                 options = [{'label':i, 'value':i} for i in instruments.keys()],
                                 value = list(instruments.keys()),
                                 multi = True
                    )
                ]),
                # Dropdown Instrument
                html.Div( children = [
                    dcc.Dropdown(id = 'spread-insn',
                                 style = {'border-color':'#cb1828', 'margin-top':'15px'},
                                 options = [{'label':i, 'value':i} for i in instruments['deribit']['BTC']],
                                 value = instruments['deribit']['BTC'][1],
                    )
                ]),
            ]),
            # -Tables
            html.Div( children=[
                html.Br(),
                dash_table.DataTable(id = 'spread-order-tablen',
                                     style_table = {'border': '1px solid lightgrey','border-collapse':'collapse'},
                                     style_cell = {'textAlign':'center','width':'12%'},
                                     style_header={'textAlign':'center','backgroundColor':'#DCDCDC','fontWeight':'bold'},
                                     style_data_conditional = [{'if' : {'filter':  '{side} eq "bid"' }, 'color':'blue' }] +
                                                              [{'if' : {'filter': '{side} eq "ask"' }, 'color':'rgb(203,24,40)' }] +
                                                              [{'if': {'row_index':'odd'}, 'backgroundColor':'rgb(242,242,242)'}] +
                                                              [{'if':{'column_id':'price'}, 'fontWeight':'bold','border': 'thin lightgrey solid'}] +
                                                              [{'if':{'column_id':'from_mid'}, 'fontWeight':'bold'}] +
                                                              [{'if' : {'column_id': c }, 'textAlign': 'right','padding-right' : '2%'} for c in ['size','cum_size','size_$','cum_size_$']],
                ),
                html.P(id = 'spread-obn-last-timestamp', style = {'display':'inline-block','font-size':'1.2rem'}),
                html.P(children=' / ', style = {'display':'inline-block', 'margin':'0px 5px','font-size':'1.2rem'}),
                html.P(id = 'spread-obn-new-timestamp', children = dt.datetime.now().strftime('%X'), style = {'display':'inline-block','font-size':'1.2rem'}),
            ]),
        ]),

        # Spreads Table
        html.Div(className = 'four columns', children = [
            # Label
            html.Div( className = 'row', children = [
                html.Label( id = 'spread-spreaded-pair', style = {'margin-bottom':'5px','font-size':'1.4rem'})
            ]),
            html.Div( style = {'margin-bottom':'40px'}, className = 'row', children = [
                html.H6( id = 'spread-spreaded-insp'),
                html.H6( id = 'spread-spreaded-insn')
            ]),
            # Table
            html.Div( children=[
                html.Br(),
                dash_table.DataTable(id = 'spread-book',
                                     style_table = {'border': '1px solid lightgrey','border-collapse':'collapse'},
                                     style_cell = {'textAlign':'center','width':'12%'},
                                     style_header={'textAlign':'center','backgroundColor':'#DCDCDC','fontWeight':'bold'},
                                     style_data_conditional = [{'if' : {'filter':  '{side} eq "bid"' }, 'color':'blue' }] +
                                                              [{'if' : {'filter': '{side} eq "ask"' }, 'color':'rgb(203,24,40)' }] +
                                                              [{'if': {'row_index':'odd'}, 'backgroundColor':'rgb(242,242,242)'}] +
                                                              [{'if':{'column_id':'price'}, 'fontWeight':'bold','border': 'thin lightgrey solid'}] +
                                                              [{'if':{'column_id':'from_mid'}, 'fontWeight':'bold'}] +
                                                              [{'if' : {'column_id': c }, 'textAlign': 'right','padding-right' : '2%'} for c in ['size','cum_size','size_$','cum_size_$'] ],),
                html.P(id = 'spread-sb-last-timestamp', style = {'display': 'inline-block','font-size':'1.2rem'}),
                html.P(children = '/', style = {'display': 'inline-block', 'margin':'0px 5px','font-size':'1.2rem'}),
                html.P(id = 'spread-sb-new-timestamp', children = dt.datetime.now().strftime('%X'), style = {'display': 'inline-block','font-size':'1.2rem'}),]),]),]),
    html.Div( children = [
        html.Div( id = 'spread-data', style = {'display': 'none'}),
        dcc.Interval( id = 'spread-interval-component', interval = refresh_rate*1000, n_intervals = 0)
    ])
])

#----------------------------------------------------------------------------------------------#
# Callback
#----------------------------------------------------------------------------------------------#
# 1. Update positive instrument exchanges
@app.callback(
    [Output('spread-insp-ex','options'), Output('spread-insp-ex','value')],
    [Input('spread-base','value')]
)
def update_ins_p_exchanges(base):
    sup_exchanges = [i for i in instruments if base in instruments[i].keys()]
    options = [{'label':exch,'value':exch} for exch in sup_exchanges]
    value = sup_exchanges
    return options, value


# 2. Update positive instruments
@app.callback(
    [Output('spread-insp','options'), Output('spread-insp','value')],
    [Input('spread-base','value'), Input('spread-insp-ex','value')]
)
def update_ins_p(base, ins_p_ex):
    ins = []
    for exch in ins_p_ex:
        ins.extend(instruments[exch][base])
    options = [{'label':i,'value':i} for i in ins]
    value = ins[0]
    return options, value


# 3. Update negative instrument exchanges
@app.callback(
    [Output('spread-insn-ex','options'), Output('spread-insn-ex','value')],
    [Input('spread-base','value'), Input('spread-negative-type','value'), Input('spread-insp','value')]
)
def update_ins_n_exchanges(base, neg_type, ins_p):
    pair = underlying[ins_p]
    sup_exchanges = [i for i in instruments if base in instruments[i].keys()] if neg_type=='Futures' else [i for i in fob.get_exchanges_for_ins(pair,spot_exch_dict).keys()]
    options = [{'label':exch,'value':exch} for exch in sup_exchanges]
    value = sup_exchanges
    return options, value


# 4. Update negative instruments
@app.callback(
    [Output('spread-insn','options'), Output('spread-insn','value')],
    [Input('spread-base','value'), Input('spread-insp','value'), Input('spread-negative-type','value'), Input('spread-insn-ex','value')]
)
def update_ins_n(base, ins_p, neg_type, ins_n_ex):
    pair = underlying[ins_p]
    ins = []
    if neg_type=='Futures':
        for exch in ins_n_ex:
            ins.extend(instruments[exch][base])
    else:
        ins = [pair]
    options = [{'label':i,'value':i} for i in ins]
    value = ins[1] if neg_type=='Futures' else ins[0]
    return options, value


# 5. Update spread labels
@app.callback(
    [Output('spread-spreaded-pair','children'),
     Output('spread-spreaded-insp','children'), Output('spread-spreaded-insp','style'),
     Output('spread-spreaded-insn','children'), Output('spread-spreaded-insn','style'),],
    [Input('spread-base','value'), Input('spread-negative-type','value'), Input('spread-insp','value'), Input('spread-insn','value')]
)
def update_spread_labels(base, neg_type, ins_p, ins_n):
    spreaded = 'Spread: Futures [+] / '+neg_type+' [-]'
    style_p = {'color': 'red', 'width':'50%', 'display':'inline-block'} if inversed[ins_p] else {'color': 'green', 'width':'50%', 'display':'inline-block'}
    style_n = ({'color': 'red', 'width':'50%', 'display':'inline-block', 'text-align':'right'} if inversed[ins_n] else {'color': 'green', 'width':'50%', 'display':'inline-block', 'text-align':'right'}) if neg_type=='Futures' else {'color': 'blue', 'width':'50%', 'display':'inline-block', 'text-align':'right'}
    ins_p_ex = [exch for exch in instruments if ins_p in instruments[exch][base]]
    ins_n_ex = [exch for exch in instruments if ins_n in instruments[exch][base]] if neg_type=='Futures' else ['Spot Exch.']
    return spreaded, '[+] {} ({})'.format(ins_p, ins_p_ex[0]), style_p, '[-] {} ({})'.format(ins_n, ins_n_ex[0]), style_n


# 6. Get and store orderbooks
@app.callback(
    Output('spread-data','children'),
    [Input('spread-interval-component','n_intervals'), Input('spread-cutoff','value'), Input('spread-base','value'),
     Input('spread-insp','value'),
     Input('spread-insn','value'), Input('spread-insn-ex','value'), Input('spread-negative-type','value')]
)
def update_data(n_intervals, cutoff, base, ins_p, ins_n, ins_n_ex, neg_type):
    now = dt.datetime.now()
    # 1. Find exchanges for instruments
    ins_p_ex = [exch for exch in instruments if ins_p in instruments[exch][base]]
    ins_n_ex = [exch for exch in instruments if ins_n in instruments[exch][base]] if neg_type=='Futures' else ins_n_ex

    # 2. Get exchange objects from (1)
    p_ex = {x:deriv_exch_dict[x] for x in ins_p_ex}
    n_ex = {x:deriv_exch_dict[x] for x in ins_n_ex} if neg_type == 'Futures' else {x:spot_exch_dict[x] for x in ins_n_ex}

    # 3. Get Orderbooks
    ob_p, ob_n = fob.get_order_books(ins_p, p_ex, size=25, cutoff = cutoff), fob.get_order_books(ins_n, n_ex, size=25, cutoff = cutoff)
    save_this = (ob_p, ob_n, now.strftime("%Y-%m-%d  %H:%M:%S"))

    return json.dumps(save_this)


# 7. Updage Tables
@app.callback(
    [Output('spread-order-tablep','data'), Output('spread-order-tablep','columns'),
     Output('spread-order-tablen','data'), Output('spread-order-tablen','columns'),
     Output('spread-book','data'), Output('spread-book','columns'),
     Output('spread-obp-last-timestamp','children'), Output('spread-obn-last-timestamp','children'), Output('spread-sb-last-timestamp','children'),
     Output('spread-obp-new-timestamp','children'), Output('spread-obn-new-timestamp','children'), Output('spread-sb-new-timestamp','children')],
    [Input('spread-data','children'), Input('spread-base','value'),
     Input('spread-insn','value'), Input('spread-insn-ex','value'),
     Input('spread-insp','value'), Input('spread-insp-ex','value'), Input('spread-negative-type','value'),
     Input('spread-cutoff','value'), Input('spread-agg-level','value')],
    [State('spread-obp-new-timestamp', 'children'), State('spread-obn-new-timestamp', 'children'), State('spread-sb-new-timestamp', 'children')]
)
def update_tables(order_books, base, ins_n, ins_n_ex, ins_p, ins_p_ex, neg_type, cutoff, step, ob_p_last_update, ob_n_last_update, sb_last_update):
    
    # 1. Convert step to bps and load orderbooks
    step = 10**(step-2)/10000 if step !=0 else step
    order_books = json.loads(order_books)

    # 2. Find exchanges for the instruments
    now0 = dt.datetime.now()
    ins_p_ex = [exch for exch in instruments if ins_p in instruments[exch][base]]
    ins_n_ex = [exch for exch in instruments if ins_n in instruments[exch][base]] if neg_type=='Futures' else ins_n_ex
    print('===========================================================')
    print('Current :', dt.datetime.now())
    print('Find exs:', dt.datetime.now() - now0)

    # 3. Normalize Orderbook
    now1 = dt.datetime.now()
    ob_p = {key:order_books[0][key] for key in order_books[0] if key in ins_p_ex}
    ob_n = {key:order_books[1][key] for key in order_books[1] if key in ins_n_ex}
    
    nob_p = fob.build_book(ob_p, ins_p_ex, cutoff, step, inversed[ins_p])
    nob_n = fob.build_book(ob_n, ins_n_ex, cutoff, step, inversed[ins_n]) if neg_type == 'Futures' else fob.build_book(ob_n, ins_n_ex, cutoff, step)
    print('Norm obs:', dt.datetime.now() - now1)

    # 4. Orderbooks Formatting for Dash Table
    now2 = dt.datetime.now()
    precision_p = len(str(ticks[ins_p]).split('.')[1]) if '.' in str(ticks[ins_p]) else int(str(ticks[ins_p]).split('e-')[1])
    precision_n = (len(str(ticks[ins_n]).split('.')[1]) if '.' in str(ticks[ins_n]) else int(str(ticks[ins_n]).split('e-')[1])) if neg_type == 'Futures' else max([ticks_spot[ex][ins_n] for ex in ins_n_ex])[1]

    print(precision_p,precision_n)

    data_p, columns_p = process_ob_for_dashtable(base, ins_p, nob_p, precision_p, step)
    obp_new_time = dt.datetime.now().strftime('%X')
    data_n, columns_n = process_ob_for_dashtable(base, ins_n, nob_n, precision_n, step)
    obn_new_time = dt.datetime.now().strftime('%X')
    print('Format obs:', dt.datetime.now() - now2)

    # 5. Build and format spreadbook (fob.build_spread_book(nob1,nob2))
    now3 = dt.datetime.now()
    df = fob.build_spread_book(nob_n,nob_p)
    print('Build sb:', dt.datetime.now() - now3)
    
    now4 = dt.datetime.now()
    df_bids = df[[i for i in df.columns if 'bid' in i ]].dropna().iloc[:13]
    df_bids['side'] = 'bid'
    df_bids.columns=['price','size','cum_size','size_$','cum_size_$','average_fill','exc','side']

    df_asks = df[[i for i in df.columns if 'ask' in i ]].dropna().iloc[:13]
    df_asks['side'] = 'ask'
    df_asks.columns = ['price','size','cum_size','size_$','cum_size_$','average_fill','exc','side']
    mid = (df_bids['price'].max() + df_asks['price'].min())/2
    df_bids['from_mid'] = (df_bids['price']-mid)/np.abs(mid)
    df_asks['from_mid'] = (df_asks['price']-mid)/np.abs(mid)
    df_all = pd.concat([df_asks.sort_values(by='price',ascending=False), df_bids])

    data_sb = df_all.to_dict('rows')
    sb_new_time = dt.datetime.now().strftime('%X')
    print('Format sb:', dt.datetime.now() - now4)

    columns_sb=[{'id':'from_mid','name':'From Mid','type':'numeric','format':FormatTemplate.percentage(2).sign(Sign.positive)},
                {'id':'price','name':'Price','type':'numeric','format':Format(precision=2,scheme=Scheme.fixed)},
                {'id':'size','name':'Size','type':'numeric','format':Format(precision=2,scheme=Scheme.fixed)},
                {'id':'cum_size','name': 'Size Total','type':'numeric','format':Format(precision=2,scheme=Scheme.fixed)},
                {'id':'size_$','name':'Size $','type':'numeric','format':FormatTemplate.money(0)},
                {'id':'cum_size_$','name':'Size Total $','type':'numeric','format':FormatTemplate.money(0)},
                {'id':'average_fill','name':'Averge Fill','type':'numeric','format':Format(precision=2,scheme=Scheme.fixed)},
                {'id':'exc','name':'Exchange', 'hidden': True},
                {'id':'side','name':'side','hidden':True}]

    print('-----------------------------------------------------------')
    print('total   :', dt.datetime.now() - now0)
    print('===========================================================')

    return data_p, columns_p, data_n, columns_n, data_sb, columns_sb, ob_p_last_update, ob_n_last_update, sb_last_update, obp_new_time, obn_new_time, sb_new_time