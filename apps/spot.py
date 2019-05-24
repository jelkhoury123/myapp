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
import itertools
import json
import datetime as dt 

import sys
sys.path.append('..')

from app import app

ENABLE_WEBSOCKET_SUPPORT = False
refresh_rate = 3 if ENABLE_WEBSOCKET_SUPPORT else 6
if ENABLE_WEBSOCKET_SUPPORT:
    import diginex.ccxt.websocket_support as ccxt
else:
    import ccxt
    
# Define my world
spot_exchanges = ['coinbasepro','bitstamp','kraken','liquid','binance',
                    'bitbank','bittrex','kucoin2','poloniex']

exchanges = spot_exchanges 

exch_dict={}
for x in exchanges:
    if x == "binance":
        exch_dict[x] = ccxt.binance({
            "apiKey": "NBVxbseHg7qnbwJxCsye6ywL4Ut9IwdBL3sSknsYXu4MMDzGzcMTcUrzQ0HY89r7",
            "secret": "IvYRol5M6J9wsVNCibUSPoaLlJyP70KATDhuelmWh3w9yJxyDx5RD2SvbMTw3Ibb",
        })
    else:
        exec('exch_dict[x]=ccxt.{}()'.format(x))

#Xpto = ['ADA','BTC','BCH','EOS','ETH','LTC','TRX','XRP','USDC']
Xpto_main = ['BTC','ETH']
Xpto_alt = ['ADA','BCH','EOS','LTC','TRX','XRP', 'USDC']
Fiat = ['USD','EUR','GBP','CHF','HKD','JPY','CNH']
Xpto_sym = {'BTC':'฿','ETH':'⧫','ADA':'ADA','BCH':'BCH','EOS':'EOS','LTC':'LTC','TRX':'TRX','XRP':'XRP', 'USDC':'USDC','USD':'$','EUR':'€','GBP':'£','CHF':'CHF','HKD':'HKD','JPY':'JP ¥','CNH':'CN ¥'}
xpto_fiat = [xpto+'/'+ fiat for xpto in Xpto_main+Xpto_alt for fiat in Fiat]
xpto_xpto = [p[0]+'/'+p[1] for p in itertools.permutations(Xpto_main+Xpto_alt,2)]

all_pairs = set(sum(itertools.chain([*exch_dict[x].load_markets()] for x in exch_dict),[])) 
pairs = list(set(xpto_fiat + xpto_xpto) & set(all_pairs))
pairs.sort()

def load_exchanges():
    ticks = {}
    for ex, ex_obj in exch_dict.items():
        ticks[ex] = {}
        market = getattr(ex_obj,'markets')
        for pair in pairs:
            if pair in market.keys():
                base_precision = market[pair]['precision']['amount']
                quote_precision = market[pair]['precision']['price']
                ticks[ex].update({pair:[base_precision,quote_precision]})
    return ticks

ticks = load_exchanges()

def get_exchanges_for_pair(pair):
    '''input: a pair
    output: a dictionary of ccxt exchange objects of the exchanges listing the pair
    '''
    return {x:exch_dict[x] for x in exch_dict if pair in list(exch_dict[x].load_markets().keys())}
def get_pairs_for_exchange(ex):
    '''input: an exchange
    output: a list of pairs '''
    d={}
    exec('d[ex]=ccxt.{}()'.format(ex))
    d[ex].load_markets()
    return d[ex].symbols

def get_order_books(pair,ex):
    '''pair is the pair string ,'BTC/USD'...
        returns a dictionary of order books for the pair
        special case for binance which API fails if # of parmaeters > 2
    '''
    nobinance= {key:value for key, value in ex.items() if key != 'binance'and  key != 'bitfinex'}
    order_books = {key: value.fetch_order_book(pair,limit=2000 if key!='bithumb' else 50,
                        params={'full':1,'level':3,'limit_bids':0,'limit_asks':0,'type':'both'})
                        for key,value in nobinance.items() }
    if 'binance' in ex:
        order_books['binance'] =  ex['binance'].fetch_order_book(pair,limit=1000)
    if 'bitfinex' in ex:
        order_books['bitfinex'] =  ex['bitfinex'].fetch_order_book(pair,limit=2000)
    return order_books

def aggregate_order_books(dict_of_order_books):
    '''dict_of_order_books is a dict of ccxt like order_books
        retuns a ccxt like dictionary order book sorted by prices 
    '''
    agg_dict_order_book = {}
    bids = []
    for x in dict_of_order_books:
        for bid in dict_of_order_books[x]['bids']:
            bids.append(bid+[x])
    asks = []
    for x in dict_of_order_books:
        for ask in dict_of_order_books[x]['asks']:
            asks.append(ask+[x])
    agg_dict_order_book['bids'] = (pd.DataFrame(bids)).sort_values(by=0,ascending=False).values.tolist()
    agg_dict_order_book['asks'] = (pd.DataFrame(asks)).sort_values(by=0,ascending=True).values.tolist()
    return agg_dict_order_book

def normalize_order_book(order_book, cutoff=.1, step=.001):
    '''order_book is a dictionary with keys bids asks timestamp datetime ...
    where bids is a list of list [[bid,bid_size]] and 
    asks is a list of list [[ask,ask_size]]
    this is returned by ccxt.'exchange'.fetch_order_book()
    returns a dataframe with columns [ask, ask_size, ask_size_$, cum_ask_size_$, bid_, bid_size, bid_size_$, cum_bid_size_$]
    and an index of shape np.linspace(1 - cutoff,1 + cutoff ,step =.001 ~ 10 bps)
    '''
    try:
        rounding = int(np.ceil(-np.log(step)/np.log(10)))
        agg = True
    except:
        agg = False
    bid_side = pd.DataFrame(order_book['bids'],columns=['bid','bid_size','exc'])
    bid_side['cum_bid_size'] = bid_side['bid_size'].cumsum()
    ask_side = pd.DataFrame(order_book['asks'],columns=['ask','ask_size','exc'])
    ask_side['cum_ask_size'] = ask_side['ask_size'].cumsum()

    ref = (bid_side['bid'][0]+ask_side['ask'][0])/2
    bid_side['bid%'] = round(bid_side['bid']/ref,rounding) if agg else bid_side['bid']/ref
    ask_side['ask%'] = round(ask_side['ask']/ref,rounding) if agg else ask_side['ask']/ref

    bid_side = bid_side[bid_side['bid%']>=1-cutoff]
    ask_side = ask_side[ask_side['ask%']<=1+cutoff]
    
    bid_side['bid_size_$'] = bid_side['bid_size']*bid_side['bid']
    bid_side['cum_bid_size_$'] = bid_side['bid_size_$'].cumsum()
    ask_side['ask_size_$'] = ask_side['ask_size']*ask_side['ask']
    ask_side['cum_ask_size_$'] = ask_side['ask_size_$'].cumsum()

    normalized_bids = pd.DataFrame(bid_side.groupby('bid%',sort=False).mean()['bid'])
    normalized_bids.columns = ['bid']
    normalized_bids['bid_size'] = bid_side.groupby('bid%',sort=False).sum()['bid_size']
    normalized_bids['cum_bid_size'] = normalized_bids['bid_size'].cumsum()
    normalized_bids['bid_size_$'] = bid_side.groupby('bid%',sort=False).sum()['bid_size_$']
    normalized_bids['cum_bid_size_$'] = normalized_bids['bid_size_$'].cumsum()
    normalized_bids['average_bid_fill'] = normalized_bids['cum_bid_size_$']/normalized_bids['cum_bid_size']
    normalized_bids['bids_exc']=bid_side.groupby('bid%',sort=False).apply(lambda x: x['exc'].loc[x['bid_size'].idxmax()])
    normalized_asks = pd.DataFrame(ask_side.groupby('ask%',sort=False).mean()['ask'])
    normalized_asks.columns = ['ask']
    normalized_asks['ask_size'] = ask_side.groupby('ask%',sort=False).sum()['ask_size']
    normalized_asks['cum_ask_size'] = normalized_asks['ask_size'].cumsum()
    normalized_asks['ask_size_$'] = ask_side.groupby('ask%',sort=False).sum()['ask_size_$']
    normalized_asks['cum_ask_size_$'] = normalized_asks['ask_size_$'].cumsum()
    normalized_asks['average_ask_fill']=normalized_asks['cum_ask_size_$']/normalized_asks['cum_ask_size']
    normalized_asks['asks_exc']=ask_side.groupby('ask%',sort=False).apply(lambda x: x['exc'].loc[x['ask_size'].idxmax()])
    book=pd.concat([normalized_asks,normalized_bids],sort=False)
    return book

def build_book(order_books, pair, exchanges, cutoff=.1, step=0.001):
    ''' gets order books aggreagtes them then normalizes
        returns a dataframe
    '''
    return normalize_order_book(aggregate_order_books({key:order_books[key] for key in exchanges}), cutoff, step)

def plot_book(order_books,pair, exc, relative=True, currency=True, cutoff=.1):
    ''' plots the order book as a v shape chart '''
    order_book = build_book(order_books,pair,exc,cutoff)
    best_bid = round(order_book['bid'].max(),4)
    best_ask = round(order_book['ask'].min(),4)
    if currency:
        col_to_chart = '_$'
    else:
        col_to_chart = ''
    if relative:
        trace_asks=go.Scatter(x=order_book.index,y=order_book['cum_ask_size'+col_to_chart],
                        name='asks',marker=dict(color='rgba(203,24,40,0.6)'),fill='tozeroy',fillcolor='rgba(203,24,40,0.2)')
        trace_bids=go.Scatter(x=order_book.index,y=order_book['cum_bid_size'+col_to_chart],
                        name='bids',marker=dict(color='rgba(0,0,255,0.6)'),fill='tozeroy',fillcolor='rgba(0,0,255,0.2)')     
    else:
        trace_asks=go.Scatter(x=order_book['ask'].fillna(0)+order_book['bid'].fillna(0),y=order_book['cum_ask_size'+col_to_chart],
                        name='asks',marker=dict(color='rgba(203,24,40,0.6)'),fill='tozeroy',fillcolor='rgba(203,24,40,0.2)')
        trace_bids=go.Scatter(x=order_book['ask'].fillna(0)+order_book['bid'].fillna(0),y=order_book['cum_bid_size'+col_to_chart],
                        name='bids',marker=dict(color='rgba(0,0,255,0.6)'),fill='tozeroy',fillcolor='rgba(0,0,255,0.2)')
        
    layout = go.Layout(title = ', '.join(exc),
                         xaxis = dict(title= pair +'  ' + str(best_bid)+' - '+ str(best_ask)),
                         showlegend=False, margin = {'t':25,'r': 10,'l': 35})
    data=[trace_asks,trace_bids]
    figure = go.Figure(data=data,layout=layout)
    return figure

def plot_depth(order_books,pair, exc, relative=True, currency=True, cutoff=.1):
    if currency:
        col_to_chart = '_$'
    else:
        col_to_chart = ''
    order_book = build_book(order_books,pair,exc,cutoff)
    mid = (order_book['bid'].max()+order_book['ask'].min())/2 if relative else 1
    trace_asks = go.Scatter(x=order_book['cum_ask_size'+col_to_chart],y=order_book['average_ask_fill']/mid,
                        name='ask depth',marker=dict(color='rgba(203,24,40,0.6)'),fill='tozerox',fillcolor='rgba(203,24,40,0.2)')
    trace_bids = go.Scatter(x=-order_book['cum_bid_size'+col_to_chart],y=order_book['average_bid_fill']/mid,
                        name='bid depth',marker=dict(color='rgba(0,0,255,0.6)'),fill='tozerox',fillcolor='rgba(0,0,255,0.2)')
    data = [trace_asks,trace_bids]
    figure = go.Figure(data=data, layout={'title': 'Market Depth','showlegend':False,'margin' : {'t':25,'r': 10,'l': 25}})
    return figure

def order_fill(order_book_df, order_sizes,in_ccy=True):
    '''takes in an order book dataframe and an np.array of order sizes
        with size in currecncy by default else in coin
        returns an np.array of the purchase costs or the sale proceeds of an order
    '''
    average_fills = np.zeros(order_sizes.shape)
    mid=(order_book_df['ask'].min()+order_book_df['bid'].max())/2
    if in_ccy:
        order_sizes=order_sizes/mid
    for i , order_size in enumerate(order_sizes):
        if order_size > 0:
            try:
                last_line = order_book_df[order_book_df['cum_ask_size']>order_size].iloc[0]
                ccy_fill = last_line['cum_ask_size_$']+(order_size-last_line['cum_ask_size'])*last_line['ask']
                average_fill=ccy_fill/order_size
            except:
                average_fill=np.nan
        elif order_size <= 0:
            try:
                last_line = order_book_df[order_book_df['cum_bid_size'] > -order_size].iloc[0]
                ccy_fill=last_line['cum_bid_size_$']+(-order_size-last_line['cum_bid_size'])*last_line['bid']
                average_fill = -ccy_fill/order_size
            except:
                average_fill = np.nan
        average_fills[i] = average_fill
    return average_fills/mid
 
def get_liq_params(normalized,pair,step):
    #coin stats
    coinmar = ccxt.coinmarketcap()
    coindata=coinmar.load_markets()
    total_coins=float(coindata[pair]['info']['available_supply'])  #number of coins floating
    order_span = (1,10,20,30,40)
    clip = total_coins/(100*1000)                                      #my standard order size 
    ordersizes=np.array([clip* i for i in order_span]+[-clip* i for i in order_span]).astype(int)
    slippage = ((order_fill(normalized,ordersizes,False)-1)*100).round(2)
    #order book
    best_bid = normalized['bid'].max()
    best_ask = normalized['ask'].min()
    mid= (best_bid + best_ask)/2
    spread = best_ask-best_bid
    spread_pct = spread/mid*100
    cross = min(0,spread)
    cross_pct = min(0,spread_pct)
    #arb with while loop!
    arb_ask = normalized.copy()[[c for c in normalized.columns if 'ask' in c]].dropna()
    arb_bid = normalized.copy()[[c for c in normalized.columns if 'bid' in c]].dropna()
    arb = spread < 0
    arb_dollar = 0
    arb_size = 0
    while arb:
        #clean one line of the order book
        bid_size = arb_bid.iloc[0]['bid_size']
        first_bid = arb_bid.iloc[0]['bid']
        ask_size = arb_ask.iloc[0]['ask_size']
        first_ask = arb_ask.iloc[0]['ask']
        sell_first = bid_size < ask_size
        if sell_first:
            arb_size += bid_size
            arb_dollar += bid_size*(first_bid-first_ask)
            arb_bid = arb_bid.iloc[1:]
            arb_ask['ask_size'].iloc[0]-=bid_size
            first_bid = arb_bid.iloc[0]['bid']
        if not sell_first:
            arb_size += ask_size
            arb_dollar += ask_size*(best_bid-best_ask)
            arb_ask = arb_ask.iloc[1:] 
            arb_bid['bid_size'].iloc[0]-=ask_size
            first_ask = arb_ask.iloc[0]['ask']

        arb = first_bid > first_ask

    rounding = [int(np.ceil(-np.log(mid*step)/np.log(10)))+1] if step !=0 else [8]
    result1 = pd.DataFrame([best_bid,best_ask,mid,spread,spread_pct/100,cross,cross_pct/100,arb_dollar,arb_size,arb_dollar/(arb_size*mid) if arb_size!=0 else 0],
    index=['bid','ask','mid','spread','spread%','cross','cross%','arb $','size arb','arb%']).T
    decimals = pd.Series(rounding * 4 +[6] +rounding +[6] + [0]*2 + [6],index=result1.columns)
    result1=result1.round(decimals)
    result2 = pd.DataFrame(index=[str(o) for o in ordersizes])
    ordersizes_dollar=ordersizes*mid
    result2 [0]=(ordersizes_dollar/1e6).round(1)
    result2[1] = slippage
    result2=result2.T
    info = coindata[pair]['info']
    select_info=['symbol','rank','24h_volume_usd','market_cap_usd',
                'available_supply','percent_change_1h','percent_change_24h','percent_change_7d']
    selected_info={key:value for key,value in info.items() if key in select_info}
    result3 = pd.DataFrame(pd.Series(selected_info)).T
    result3.columns=['Coin','Rank','24H % Volume','USD Market Cap M$','Coins Supply M','% 1h','% 24h','% 7d']
    result3['24H % Volume']= round(float(result3['24H % Volume'])/float(result3['USD Market Cap M$'])*100,1)
    result3['USD Market Cap M$'] = round(float(result3['USD Market Cap M$'])/(1000*1000),0)
    result3['Coins Supply M'] = round(float(result3['Coins Supply M'])/(1000*1000),1)
    result3 = result3.astype({'24H % Volume':float,'% 1h':float,'% 24h':float,'% 7d':float})
    result3[['24H % Volume','% 1h','% 24h','% 7d']] = result3[['24H % Volume','% 1h','% 24h','% 7d']].div(100, axis = 0)
    return [result1,result2,result3]
        
title = 'Spot'

layout = html.Div(style={'marginLeft':35,'marginRight':35},
                    children=[ 
                        html.Div(className='row',children=[
                                        html.Div(className='three columns',
                                        children =[
                                            html.H6('Choose Base'),
                                            dcc.RadioItems(id='spot-choose-base',
                                                        options = [{'label':base,'value':base} for base in ['BTC','ETH','ALT']],
                                                        value = 'BTC',
                                                        labelStyle={'display':'inline-block','margin-top':'10px','margin-bottom':'10px'}),
                                            html.Hr(style={'border-color':'#cb1828'}),
                                            html.H6('Choose Pair'),
                                                    dcc.Dropdown(id='spot-pairs',style={'border-color':'#cb1828'}),
                                                    #html.H6('Book Params:'),
                                                    html.Hr(style={'border-color':'#cb1828'}),
                                                    html.Div(className='row',children=[
                                                    html.Div(className='two columns',children = [html.Label( 'X :')]),
                                                    html.Div(className='four columns',children=[
                                                                dcc.RadioItems(id='spot-x-scale',
                                                                options=[{'label':scale,'value':scale} for scale in ['Rel','Abs']], 
                                                                value ='Rel',
                                                                labelStyle={'display':'inline-block'})]),
                                                    html.Div(className='two columns',children = [html.Label('Y :')]),
                                                    html.Div(className='four columns',children=[
                                                                dcc.RadioItems(id='spot-y-scale',
                                                                options=[{'label':j,'value':i} for i,j in {'Ccy':'Quote','Coin':'Base'}.items()],
                                                                value ='Ccy',
                                                                labelStyle={'display':'inline-block'})])
                                                    ]),
                                                    html.Div(className='row',children=[ html.Br(),
                                                    html.Div(className='three columns',children =[html.Label('Cutoff % :')]),
                                                    html.Div(className='three columns',style={'width' :'50%','align':'right'},children =[
                                                        dcc.Slider(id='spot-cutoff',
                                                                min=.05,max=.3,step=.05,value=.1,
                                                                marks={round(j,2): str(round(j,2)) for j in list(np.arange(.05,.35,.05))})]),
                                                    ]),
                                                    html.Div(className='row',children=[ html.Br(),
                                                    html.Div(className='three columns',children =[html.Label('Price Agg (bps):')]),
                                                    html.Div(className='three columns',style={'width' :'50%','align':'right'},children =[
                                                        dcc.Slider(id='spot-agg-level',
                                                                    marks = {i:str(10**(i-2) if i != 0 else 0) for i in range(0,5)},
                                                                    max = 4,
                                                                    value = 3,
                                                                    step = 1)]),
                                                    ]),
                                                    html.Hr(style={'border-color':'#cb1828'}),        
                                                    #html.H6(' Book Charts'),
                                                    dcc.Graph(id='spot-order-book-chart'),
                                                    html.Hr(style={'border-color':'#cb1828'}),
                                                    dcc.Graph(id='spot-market-depth'),
                                                    html.H6(id='spot-time')

                                        ]),
                                html.Div(className='five columns',
                                    children =[html.H6('Choose Exchange'),
                                                dcc.Dropdown(id='spot-exchanges',multi=True,style={'border-color':'#cb1828'},
                                                value = list(get_exchanges_for_pair('BTC/USD').keys())),
                                                #html.H6('Order Book'),
                                                html.Hr(style={'border-color':'#cb1828'}),
                                                #html.Div(id='spot-order-table'),
                                            html.Div(children=[dash_table.DataTable(id='spot-order-table',
                                                    style_table={'border': '1px solid lightgrey','border-collapse':'collapse'},
                                                    style_cell={'textAlign':'center','width':'12%'},
                                                    style_data_conditional= [{'if': {'filter':  'side eq "bid"' }, 'color':'blue' } ] +
                                                                            [{'if': {'filter': 'side eq "ask"' }, 'color':'rgb(203,24,40)' }] +
                                                                            [{'if': {'row_index':'odd'}, 'backgroundColor':'rgb(242,242,242)'} ] +
                                                                            [{'if': {'column_id':'price'}, 'fontWeight':'bold', 'border': 'thin lightgrey solid'}] +
                                                                            [{'if': {'column_id':'from_mid'}, 'fontWeight':'bold'} ] +
                                                                            [{'if': {'column_id': c }, 'textAlign': 'right', 'padding-right' : '2%' } for c in ['size','cum_size','size_$','cum_size_$']],
                                                    style_header={'textAlign':'center','backgroundColor':'#DCDCDC','fontWeight':'bold'},
                                                    #style_as_list_view=True
                                                )
                                                ]),
                                                html.Div( className = 'row', children = [
                                                    html.P(id = 'spot-ob-last-timestamp', style = {'display':'inline-block','font-size':'1.2rem'}),
                                                    html.P(children=' / ', style = {'display':'inline-block', 'margin':'0px 5px','font-size':'1.2rem'}),
                                                    html.P(id = 'spot-ob-new-timestamp', children = dt.datetime.now().strftime('%X'), style = {'display':'inline-block','font-size':'1.2rem'}),
                                                ]),
                                                html.Hr(style={'border-color':'#cb1828'}),
                                                html.H6('Liquidity Metrics'),
                                                html.P(id='spot-liquidity-table'),
                                                html.H6('Slippage %'),
                                                html.P(id='spot-depth-table'),
                                                html.H6('Coin Stats'),
                                                html.P(id='spot-stat-table')]),
                                html.Div(className='four columns',
                                    children =[
                                    html.H6('Manage Orders'),
                                    html.Br(),
                                    html.Hr(style={'border-color':'#cb1828'}),
                                    html.Label('Order'),
                                    html.Hr(style={'border-color':'#cb1828'}),
                                    html.P(id = 'spot-diplay-tick', style={'font-size':'1.2rem', 'font-weight':'bold'}),
                                    html.Div(style={'height':'100px'},children = [dash_table.DataTable(
                                        id ='spot-order-to-send',
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
                                        html.Div(id='spot-active'),
                                        html.Button(id='spot-send-order',style={'display':'none'}),
                                        dcc.ConfirmDialog(id='confirm',message = 'Submit Order ? '),
                                        html.Div(id='spot-output-confirm',style={'margin-top':'30px'},children=['Here']),
                                        html.Div([dcc.Tabs(id='spot-orders',className = 'custom-tabs-container',parent_className='custom-tabs',
                                         children=[
                                            dcc.Tab(label='Open Orders',className='custom-tab', selected_className='custom-tab--selected',
                                            style = {'overflow':'hidden'},
                                            children = [
                                                html.Hr(style={'border-color':'#cb1828'}),
                                                html.Div(style = {'overflow':'hidden'},
                                                children = [dash_table.DataTable(
                                                id ='spot-open-orders', 
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
                                                html.Div(id='spot-cancel-confirm',style={'margin-top':'30px'},children=['Here'])
                                            ]),
                                            dcc.Tab(label='Closed Orders',className='custom-tab', selected_className='custom-tab--selected',
                                            children = [
                                                # dcc.DatePickerSingle(
                                                #     id='spot-go-back-date',
                                                #     max_date_allowed=dt.datetime(pd.to_datetime('today').year, pd.to_datetime('today').month, pd.to_datetime('today').day),
                                                #     date=dt.datetime(pd.to_datetime('today').year, pd.to_datetime('today').month, pd.to_datetime('today').day),
                                                #     display_format='D-M-Y',
                                                #     style = {'margin-top':'10px'},
                                                # ),
                                                html.Hr(style={'border-color':'#cb1828', 'margin-top':'10px'}),
                                                html.Div(style = {'overflow':'hidden'},
                                                children = [dash_table.DataTable(
                                                id ='spot-closed-orders',
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
                                        html.Div(id='spot-balances'),
                                        html.Hr(style={'border-color':'#cb1828'}),
                                        html.H6('Price Precision (decimals)'),
                                        dash_table.DataTable(id = 'spot-tickers-table',
                                                            style_table={'border': '0.5px solid','border-color':'#cb1828','margin' :'2px', 'border-collapse':'collapse'},
                                                            style_header={'fontWeight':'bold'},
                                                            style_cell={'textAlign':'center','width':'12%'},
                                                            style_as_list_view=True,
                                                            style_data_conditional=[{ 'if': {'row_index':'odd'}, 'backgroundColor':'rgb(242,242,242)'}]),
                                    ]),
                                html.Div(id='the-ob-data',style={'display':'none'}),
                                dcc.Interval(
                                    id='ob-interval-component',
                                    interval = refresh_rate * 1000, # in milliseconds= 7 or 10 seconds
                                    n_intervals=0
                                    ),
                                dcc.Interval(
                                    id='second',
                                    interval = 1000,
                                    n_intervals=0
                                )     
                        ]),
                        ])

@app.callback(
    [Output('spot-pairs', 'options'),Output('spot-pairs', 'value')],
    [Input('spot-choose-base', 'value')]
)
def update_pairs_list(base):
    base_list = base if base!='ALT' else Xpto_alt
    ins_pairs = [pair for pair in pairs if pair[:3] in base_list]
    options = [{'label':ins,'value':ins} for ins in ins_pairs]
    value = base+'/USD' if base!='ALT' else 'ADA/USD'
    return options, value

@app.callback(Output('spot-time','children'),
            [Input('second','n_intervals'),Input('the-ob-data','children')])
def update_time(n,order_books):
    time_snap = json.loads(order_books)[1]
    return (dt.datetime.now()-dt.datetime.strptime(time_snap,"%Y-%m-%d  %H:%M:%S")).seconds

@app.callback([Output('spot-exchanges','options'),Output('spot-exchanges','value')],
            [Input('spot-pairs','value')])
def update_exchanges_options(pair):
    return [{'label':exch,'value':exch} for exch in get_exchanges_for_pair(pair).keys()] ,list(get_exchanges_for_pair(pair).keys())

@app.callback(Output('the-ob-data','children'),
            [Input('spot-pairs','value'),Input('spot-exchanges','value'),Input('ob-interval-component','n_intervals')])
def update_data(pair,ex,n):
    now = dt.datetime.now()
    ex = {x:exch_dict[x] for x in ex}
    order_books = get_order_books(pair,ex)
    save_this = (order_books,now.strftime("%Y-%m-%d  %H:%M:%S"))
    return json.dumps(save_this)

@app.callback([Output('spot-order-book-chart','figure'),Output('spot-market-depth','figure'),
            Output('spot-order-table','data'), Output('spot-order-table','columns'),
            Output('spot-liquidity-table','children'),
            Output('spot-depth-table','children'),Output('spot-stat-table','children'),
            Output('spot-ob-last-timestamp', 'children'), Output('spot-ob-new-timestamp', 'children'),
            Output('spot-tickers-table','data'), Output('spot-tickers-table','columns')],
            [Input('the-ob-data','children'),
            Input('spot-pairs','value'),Input('spot-exchanges','value'),
            Input('spot-x-scale','value'),Input('spot-y-scale','value'),
            Input('spot-cutoff','value'),Input('spot-agg-level','value')],
            [State('spot-ob-new-timestamp', 'children')])
def update_page(order_books,pair,exchanges,x_scale,y_scale,cutoff,step,last_update):

    # Get max price precision
    precision = [ticks[ex][pair] for ex in exchanges]
    max_price_precision = max(precision)

    # Display precision for exchanges (to be removed once Spot finalized)
    tickers_data = pd.DataFrame(data = precision, index = exchanges).drop(0, axis=1).rename_axis('exchange')
    tickers_data = tickers_data.T.to_dict('rows')
    tickers_data_columns = [{'id':i,'name':i} for i in exchanges]

    # Load Data
    step = 10**(step-2)/10000 if step !=0 else step          
    relative = x_scale == 'Rel'
    currency = y_scale == 'Currency'
    order_books = json.loads(order_books)[0]
    order_books = {key:order_books[key] for key in order_books if key in exchanges}

    # 1. Plot Book and Depth
    book_plot = plot_book(order_books, pair, exchanges, relative, currency, cutoff)
    depth_plot = plot_depth(order_books, pair, exchanges, relative, currency, cutoff)

    # 2. Order Table
    df =  build_book(order_books, pair, exchanges, cutoff, step)

    df_bids = df[[i for i in df.columns if 'bid' in i]].dropna().iloc[:13]
    df_bids['side'] = 'bid'
    df_bids.columns = ['price','size','cum_size','size_$','cum_size_$','average_fill','exc','side']

    df_asks = df[[i for i in df.columns if 'ask' in i]].dropna().iloc[:13]
    df_asks['side'] = 'ask'
    df_asks.columns = ['price','size','cum_size','size_$','cum_size_$','average_fill','exc','side']

    df_all = pd.concat([df_asks.sort_values(by='price',ascending=False),df_bids]).rename_axis('from_mid')
    df_all=df_all.reset_index()
    df_all['from_mid'] = (df_all['from_mid']-1)

    data_ob = df_all.to_dict('rows')
    ob_new_time = dt.datetime.now().strftime('%X')

    # Get rounidng digits for the table
    mid=(df_bids['price'].max() + df_asks['price'].min())/2
    rounding = max(min(int(np.ceil(-np.log(mid*step)/np.log(10))), max_price_precision[1]),0) if step !=0 else max_price_precision[1]  
    r = int(np.ceil(-np.log(step)/np.log(10)))-2 if step !=0 else int(np.ceil(-np.log(10**-max_price_precision[1]/mid)/np.log(10))-2)

    # Symbols for columns
    base_sym, quote_sym = Xpto_sym[pair.split('/')[0]], Xpto_sym[pair.split('/')[1]]

    columns_ob=[{'id':'from_mid','name':'From Mid','type':'numeric','format':FormatTemplate.percentage(r).sign(Sign.positive)},
            {'id':'price','name':'Price','type':'numeric','format':Format(precision=rounding,scheme=Scheme.fixed)},
            {'id':'size','name':'Size ({})'.format(base_sym),'type':'numeric','format':Format(precision=2,scheme=Scheme.fixed)},
            {'id':'cum_size','name': 'Total ({})'.format(base_sym),'type':'numeric','format':Format(precision=2,scheme=Scheme.fixed)},
            {'id':'size_$','name':'Size ({})'.format(quote_sym),'type':'numeric','format':FormatTemplate.money(0)},
            {'id':'cum_size_$','name':'Total ({})'.format(quote_sym),'type':'numeric','format':FormatTemplate.money(0)},
            {'id':'average_fill','name':'Averge Fill','type':'numeric','format':Format(precision=rounding,scheme=Scheme.fixed)},
            {'id':'exc','name':'Exchange'},
            {'id':'side','name':'side','hidden':True}]
    try:
        liq_dfs = [df.round(10) for df in get_liq_params(df,pair,step)]
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
                style_header={'backgroundColor':'lightgrey','fontWeight':'bold'},
                style_cell={'textAlign':'center','width':'10%'},
                style_table={'border': 'thin lightgrey solid'},
                style_as_list_view=True) for liq_df in liq_dfs]
    except:
        liq_tables=[0]*3
 
    return (book_plot,depth_plot,data_ob,columns_ob) + tuple(liq_tables) + (last_update, ob_new_time, tickers_data, tickers_data_columns)

@app.callback(Output('spot-order-to-send','data'),
            [Input('spot-order-table','active_cell'), Input('spot-pairs','value'), Input('spot-agg-level','value')],
            [State('spot-order-table','data')])
def update_order(active_cell,pair,step,data):
    data_df=pd.DataFrame(data)
    row = data_df.iloc[active_cell[0]]
    columns = ['B/S','Ins','Qty','Limit price','exc']
    order_df=pd.DataFrame(columns=columns)
    size = round(row['cum_size'], 2) if active_cell[1] == 3 else round(row['size'], 2)
    order_df.loc[0]=['B' if row['side']=='bid' else 'S',pair,size,row['price'],row['exc']]
    step = 10**(step-3)/10000
    r = min(int(np.ceil(-np.log(step)/np.log(10)))-2, ticks[row['exc']][pair][1])
    order_df['Limit price'] = round(order_df['Limit price'], r)
    return order_df.to_dict('row')

@app.callback([Output('spot-send-order','children'),Output('spot-send-order','style')],
                [Input('spot-order-to-send','data')])
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