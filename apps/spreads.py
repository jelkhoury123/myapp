import dash
import dash_core_components as dcc 
import dash_html_components as html 
import dash_table 
from dash_table.Format import Format, Scheme, Sign, Symbol
import dash_table.FormatTemplate as FormatTemplate
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go 
import plotly.figure_factory as ffimport 
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import itertools
import json
import datetime as dt 

import sys
sys.path.append('..') # add parent directory to the path to import app

from app import app  # app is the main app which will be run on the server in index.py

# If websocket use diginex.ccxt library and reduce update freq frm 7 to 5 secs

ENABLE_WEBSOCKET_SUPPORT = False
refresh_rate = 5 if ENABLE_WEBSOCKET_SUPPORT else 7
if ENABLE_WEBSOCKET_SUPPORT:
    import diginex.ccxt.websocket_support as ccxt
else:
    import ccxt
    
# Define my world
deriv_exchanges =['deribit','bitmex']

exchanges =  deriv_exchanges 

api_keys = {'deribit':'4v58Wk2hhiG9B','bitmex':'DM55KFt84AfdjJiyNjDBk_km'}
api_secrets = {'deribit':'QLXLOZCOHAEQ6XV247KEXAKPX43GZLT4','bitmex':'jPDWiZcKuXJtVpajawENQiBzCKO2U885i3TWU9WIihBBUZgc'}

exch_dict={}
for x in exchanges:
    exec('exch_dict[x]=ccxt.{}({{"apiKey": "{}", "secret": "{}"}})'.format(x, api_keys[x], api_secrets[x]))
for x,xccxt in exch_dict.items():
    xccxt.load_markets()

deribit = exch_dict['deribit']
bitmex = exch_dict['bitmex']

Xpto_base = ['BTC','ETH']

trade_types = ['Outright','Spread']

def get_instruments():
    '''
    returns 2 dictionaries for deribit and bitmex
    key (base) : value list containing Futures and Swaps traded on base 
    and not expired
    '''
    deribit_all_ins={}
    for base in Xpto_base:
        base_list=[]
        for ins in deribit.markets:
            market = deribit.markets[ins]
            if market['type']=='future' and market['base']==base:
                base_list.append(ins)
        deribit_all_ins[base]=base_list
    bitmex_all_ins={}
    for base in Xpto_base:
        base_list=[]
        for ins in bitmex.markets:
            market = bitmex.markets[ins]
            if market['type'] in ('future','swap') and market['base']==base and market['info']['hasLiquidity']:
                expiry = market['info']['expiry']
                if expiry is None:
                    base_list.append(ins)
                else:
                    dt_expiry = dt.datetime.strptime(expiry,"%Y-%m-%dT%H:%M:%S.%fZ")
                    if dt_expiry > dt.datetime.now():
                        base_list.append(ins)
        bitmex_all_ins[base]=base_list
    return deribit_all_ins,bitmex_all_ins

deribit_all_ins , bitmex_all_ins = get_instruments()
deribit_ins = deribit_all_ins['BTC']
bitmex_ins= bitmex_all_ins['BTC']
instruments = deribit_ins + bitmex_ins


def get_exchanges_for_ins(ins):
    '''input: an instrument 
    output: a dictionary of ccxt exchange objects of the exchanges listing the instrument
    '''
    return {x:exch_dict[x] for x in exch_dict if ins in list(exch_dict[x].load_markets().keys())}

def get_ins_for_exchange(ex):
    '''input: an exchange
    output: a list of instruments '''
    d={}
    exec('d[ex]=ccxt.{}()'.format(ex))
    d[ex].load_markets()
    return d[ex].symbols

def get_order_books(ins,ex):
    '''ins is the instrument string ,'BTC-PERPETUAL'...
        returns a dictionary of order books for the instrument
        We call order book on every exchange the ins is trading and 
        Return a dictionary {ex:order_book}
        Deribit needs a 10 multiplier 
    '''
    order_books = {key: value.fetch_order_book(ins,limit=2000,
                        params={'full':1,'level':3,'limit_bids':0,'limit_asks':0,'type':'both'}) for key,value in ex.items() }
    if 'deribit' in order_books:
        bids_df = pd.DataFrame(order_books['deribit']['bids'])
        asks_df = pd.DataFrame(order_books['deribit']['asks'])
        bids_df[1]=bids_df[1]*10
        asks_df[1]= asks_df[1]*10
        order_books['deribit']['bids']=bids_df.values.tolist()
        order_books['deribit']['asks']=asks_df.values.tolist()
    return order_books

def aggregate_order_books(dict_of_order_books):
    '''dict_of_order_books is a dict of ccxt like order_books returned by order_books with ex name added
        retuns a ccxt like dictionary order book sorted by prices (add exc name on every bid and ask)
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

def normalize_order_book(order_book,cutoff=.1,step=.001):
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
    bid_side = pd.DataFrame(order_book['bids'],columns=['bid','bid_size_$','exc'])
    bid_side['cum_bid_size_$'] = bid_side['bid_size_$'].cumsum()
    ask_side = pd.DataFrame(order_book['asks'],columns=['ask','ask_size_$','exc'])
    ask_side['cum_ask_size_$'] = ask_side['ask_size_$'].cumsum()
    ref = (bid_side['bid'][0]+ask_side['ask'][0])/2
    bid_side['bid%'] = round(bid_side['bid']/ref,rounding) if agg else bid_side['bid']/ref
    ask_side['ask%'] = round(ask_side['ask']/ref,rounding) if agg else ask_side['ask']/ref
    bid_side = bid_side[bid_side['bid%']>=1-cutoff]
    ask_side = ask_side[ask_side['ask%']<=1+cutoff]
    bid_side['bid_size'] = bid_side['bid_size_$']/bid_side['bid']
    bid_side['cum_bid_size'] = bid_side['bid_size'].cumsum()
    ask_side['ask_size'] = ask_side['ask_size_$']/ask_side['ask']
    ask_side['cum_ask_size'] = ask_side['ask_size'].cumsum()
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

def build_book(order_books,ins,exchanges,cutoff=.1,step=0.001):
    ''' gets order books aggreagtes them then normalizes
        returns a dataframe
    '''
    return normalize_order_book(aggregate_order_books({key:order_books[key] for key in exchanges}),cutoff,step)

def plot_book(order_books,ins, exc, relative=True, currency=True, cutoff=.1):
    ''' plots the order book as a v shape chart '''
    order_book = build_book(order_books,ins,exc,cutoff)
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
        
    layout = go.Layout(title = ' - '.join(exc), xaxis = dict(title= ins +'  ' + str(best_bid)+' - '+ str(best_ask)))
    data=[trace_asks,trace_bids]
    figure = go.Figure(data=data,layout=layout)
    return figure

def plot_depth(order_books,ins, exc, relative=True, currency=True, cutoff=.1):
    if currency:
        col_to_chart = '_$'
    else:
        col_to_chart = ''
    order_book = build_book(order_books,ins,exc,cutoff)
    mid = (order_book['bid'].max()+order_book['ask'].min())/2 if relative else 1
    trace_asks = go.Scatter(x=order_book['cum_ask_size'+col_to_chart],y=order_book['average_ask_fill']/mid,
                        name='ask depth',marker=dict(color='rgba(203,24,40,0.6)'),fill='tozerox',fillcolor='rgba(203,24,40,0.2)')
    trace_bids = go.Scatter(x=-order_book['cum_bid_size'+col_to_chart],y=order_book['average_bid_fill']/mid,
                        name='bid depth',marker=dict(color='rgba(0,0,255,0.6)'),fill='tozerox',fillcolor='rgba(0,0,255,0.2)')
    data = [trace_asks,trace_bids]
    figure = go.Figure(data=data, layout={'title': 'Market Depth'})
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
 
def get_liq_params(normalized,ins,step):
    #coin stats
    coinmar = ccxt.coinmarketcap()
    coindata=coinmar.load_markets()
    total_coins=float(coindata[ins]['info']['available_supply'])  #number of coins floating
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

    rounding = [int(np.ceil(-np.log(mid*step)/np.log(10)))+1]
    result1 = pd.DataFrame([best_bid,best_ask,mid,spread,spread_pct,cross,cross_pct,arb_dollar,arb_size,100*arb_dollar/(arb_size*mid) if arb_size!=0 else 0],
    index=['bid','ask','mid','spread','spread%','cross','cross%','arb $','size arb','arb%']).T
    decimals = pd.Series(rounding * 4 +[2] +rounding +[2] + [0]*2 + [2],index=result1.columns)
    result1=result1.round(decimals)
    result2 = pd.DataFrame(index=[str(o) for o in ordersizes])
    ordersizes_dollar=ordersizes*mid
    result2 [0]=(ordersizes_dollar/1e6).round(1)
    result2[1] = slippage
    result2=result2.T
    info = coindata[ins]['info']
    select_info=['symbol','rank','24h_volume_usd','market_cap_usd',
                'available_supply','percent_change_1h','percent_change_24h','percent_change_7d']
    selected_info={key:value for key,value in info.items() if key in select_info}
    result3 = pd.DataFrame(pd.Series(selected_info)).T
    result3.columns=['Coin','Rank','24H % Volume','Market Cap M$','Coins Supply M','% 1h','% 24h','% 7d']
    result3['24H % Volume']= round(float(result3['24H % Volume'])/float(result3['Market Cap M$'])*100,1)
    result3['Market Cap M$'] = round(float(result3['Market Cap M$'])/(1000*1000),0)
    result3['Coins Supply M'] = round(float(result3['Coins Supply M'])/(1000*1000),1)   
    return [result1,result2,result3]

def get_open_orders():
    deribit_open_orders = []
    for coin in deribit_all_ins:
        for symbol in deribit_all_ins[coin]:
            orders = deribit.fetch_open_orders(symbol)
            deribit_open_orders+= orders
    for order in deribit_open_orders:
        order['ex']='deribit'   

    bitmex_open_orders=bitmex.fetch_open_orders()
    for order in bitmex_open_orders:
        order['ex']='bitmex' 

    open_orders = deribit_open_orders + bitmex_open_orders

    for order in open_orders:
        order.pop('info')
    if len(open_orders) > 0:
        open_orders_df=pd.DataFrame(open_orders)
        columns = ['id','type','symbol','side','amount','price','status','filled','ex']
        oo = open_orders_df[columns].copy()
    else: 
        columns = ['id','type','symbol','side','amount','price','status','filled','ex']
        oo= pd.DataFrame(columns=columns)
    return oo

def get_closed_orders(start):
    deribit_closed_orders = []
    for coin in deribit_all_ins:
        for symbol in deribit_all_ins[coin]:
            orders = deribit.fetch_closed_orders(symbol)
            deribit_closed_orders+= orders
    for order in deribit_closed_orders:
        order['ex']='deribit'   
    bitmex_closed_orders=bitmex.fetch_closed_orders()
    for order in bitmex_closed_orders:
        order['ex']='bitmex'

    closed_orders = deribit_closed_orders + bitmex_closed_orders

    for order in closed_orders:
        order.pop('info')

    if len(closed_orders) > 0:
        closed_orders = pd.DataFrame(closed_orders).sort_values(by=['timestamp'],ascending=False)
        closed_orders = closed_orders[closed_orders['timestamp']>start]
        columns = ['id','type','symbol','side','amount','price','average','filled','ex']
        co=closed_orders[columns].copy()
    else: 
        columns = ['id','type','symbol','side','price','price','average','filled','ex']
        co= pd.DataFrame(columns=columns)
    return co

def get_balances():
    deribit_balance=deribit.fetch_balance()
    bitmex_balance=bitmex.fetch_balance()
    d_balance={key:deribit_balance[key] for key in ['free','used','total']}
    d_balance['exc']='deribit'
    b_balance={key:bitmex_balance[key] for key in ['free','used','total']}
    b_balance['exc']='bitmex'
    b=pd.concat([pd.DataFrame(b_balance),pd.DataFrame(d_balance)])
    b = b[['exc','used','free','total']]
    return b


title = 'Spreads'

layout = html.Div(style={'marginLeft':35,'marginRight':35},
                    children=[ 
                        html.Div(className='row',style={'margin-top':'2px'},children=[
                                        html.Div(className='four columns',
                                            children = [
                                                html.Div(className = 'row',children = [
                                                    html.H6('Base'),
                                                    dcc.RadioItems(id='spr-choose-base',
                                                            options = [{'label':base,'value':base} for base in Xpto_base],
                                                            value = Xpto_base[0],
                                                            labelStyle={'display':'inline-block'})]),
                                                html.Hr(style={'border-color':'#cb1828'}),
                                                html.Div(className='row', children = [
                                                html.Div(className = 'five columns', children = [
                                                html.H6('+ Ins'),
                                                dcc.Dropdown(id='spr-ins',
                                                            style={'border-color':'#cb1828'})]),
                                                html.Div(className = 'five columns', children = [
                                                html.H6('- Ins'),
                                                dcc.Dropdown(id='spr-short-ins',
                                                            style={'border-color':'#cb1828'})]),
                                                html.Hr(style={'border-color':'#cb1828'}),
                                                ]),
                                                html.Div(className='row', children = [
                                                html.H6('Book Params:'),
                                                html.Hr(style={'border-color':'#cb1828'}),
                                                html.Div(className='row',children=[
                                                html.Div(className='two columns',children = [html.H6( 'X :')]),
                                                html.Div(className='four columns',children=[dcc.RadioItems(id='spr-x-scale',
                                                            options=[{'label':scale,'value':scale} for scale in ['Relative','Absolute']], 
                                                            value ='Relative',
                                                            labelStyle={'display':'inline-block'})]),
                                                html.Div(className='two columns',children = [html.H6('Y :')]),
                                                html.Div(className='four columns',children=[dcc.RadioItems(id='spr-y-scale',
                                                            options=[{'label':scale,'value':scale} for scale in ['Currency','Coin']], 
                                                            value ='Currency',
                                                            labelStyle={'display':'inline-block'})])
                                                ]),
                                                html.Div(className='row',children=[
                                                html.Div(className='three columns',children =[html.H6('Cutoff % :')]),
                                                html.Div(className='three columns',style={'width' :'50%','align':'right'},children =[
                                                dcc.Slider(id='spr-cutoff',
                                                        min=.05,max=.3,step=.05,value=.1,
                                                        marks={round(j,2): str(round(j,2)) for j in list(np.arange(.05,.35,.05))})]),
                                                ]),
                                                html.Div(className='row',children=[
                                                html.Div(className='three columns',children =[html.H6('Price Agg (bps):')]),
                                                html.Div(className='three columns',style={'width' :'50%','align':'right'},children =[dcc.Slider(id='spr-agg-level',
                                                                                    marks = {i:10**(i-2) for i in range(0,5)},
                                                                                    max = 4,
                                                                                    value = 2,
                                                                                    step = 1)]),
                                                ]),
                                                html.Hr(style={'border-color':'#cb1828'})]),       
                                                html.H6('Book Charts'),
                                                dcc.Graph(id='spr-order-book-chart'),
                                                html.Hr(style={'border-color':'#cb1828'}),
                                                dcc.Graph(id='spr-market-depth'),
                                                html.H6(id='spr-time')
                                             ]),

                                        html.Div(className='four columns',
                                            children =[html.H6('Choose Exchange'),
                                                dcc.Dropdown(id='spr-exchanges',multi=True,style={'border-color':'#cb1828'},
                                                value = list(get_exchanges_for_ins('BTC/USD').keys()),
                                                options = [{'label':exch,'value':exch} for exch in get_exchanges_for_ins('BTC/USD').keys()]),
                                                html.H6('Order Book'),
                                                html.Hr(style={'border-color':'#cb1828'}),
                                                html.Div(children=[dash_table.DataTable(id='spr-order-table',
                                                columns=[{'id':'from_mid','name':'From Mid','type':'numeric','format':FormatTemplate.percentage(4).sign(Sign.positive)},
                                                {'id':'price','name':'Price'},{'id':'size','name':'Size'},{'id':'cum_size','name': 'Size Total'},
                                                {'id':'size_$','name':'Size $','type':'numeric','format':FormatTemplate.money(0)},
                                                {'id':'cum_size_$','name':'Size Total $','type':'numeric','format':FormatTemplate.money(0)},
                                                {'id':'average_fill','name':'Averge Fill'},{'id':'exc','name':'Exchange'},
                                                {'id':'side','name':'side','hidden':True}],
                                                    style_table={'border': '1px solid lightgrey','border-collapse':'collapse'},
                                                    style_header={'backgroundColor':'#DCDCDC','fontWeight':'bold'},
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
                                                    ],
                                                    style_as_list_view=True
                                                )
                                                ]),
                                                html.Hr(style={'border-color':'#cb1828'}),
                                                html.H6('Liquidity Metrics'),
                                                html.P(id='spr-liquidity-table'),
                                                html.H6('Slippage %'),
                                                html.P(id='spr-depth-table'),
                                                html.H6('Coin Stats'),
                                                html.P(id='spr-stat-table')]),

                                html.Div(className='four columns',
                                    children =[html.H6('Order Management'),
                                    html.Hr(style={'border-color':'#cb1828'}),
                                    html.H6('Order'),
                                    html.Hr(style={'border-color':'#cb1828'}),
                                    html.Div(style={'height':'100px'},children = [dash_table.DataTable(
                                        id ='spr-order-to-send',
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
                                        html.Div(id='spr-active'),
                                        html.Button(id='spr-send-order',style={'display':'none'}),
                                        dcc.ConfirmDialog(id='spr-confirm',message = 'Submit Order ? '),
                                        html.Div(id='spr-output-confirm',style={'margin-top':'30px'},children=['Here']),
                                        html.Div([dcc.Tabs(id='spr-orders',className = 'custom-tabs-container',parent_className='custom-tabs',
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
                                                html.Hr(style={'border-color':'#cb1828'}),
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
                                        html.Div(id='spr-balances')
                                    ]),
                                html.Div(id='the-spr-data',style={'display':'none'}),
                                dcc.Interval(
                                    id='spr-interval-component',
                                    interval = refresh_rate * 1000, # in milliseconds= 7 or 10 seconds
                                    n_intervals=0
                                    ),
                                dcc.Interval(
                                    id='spr-second',
                                    interval = 1000,
                                    n_intervals=0
                                )     
                        ]),
                        ])

@app.callback([Output('spr-ins','options'),Output('spr-ins','value')],
            [Input('spr-choose-base','value')])
def update_ins(radio_button_value):
    instruments = deribit_all_ins[radio_button_value]+bitmex_all_ins[radio_button_value]
    options=[{'label':ins,'value':ins} for ins in instruments]
    value=instruments[0]
    return options, value

@app.callback([Output('spr-exchanges','options'),Output('spr-exchanges','value')],
            [Input('spr-ins','value')])
def update_exchanges_options(ins):
    return [{'label':exch,'value':exch} for exch in get_exchanges_for_ins(ins).keys()] ,list(get_exchanges_for_ins(ins).keys())

@app.callback(Output('the-spr-data','children'),
            [Input('spr-ins','value'),Input('spr-exchanges','value'),Input('spr-interval-component','n_intervals')])
def update_data(ins,ex,n):
    now = dt.datetime.now()
    ex = {x:exch_dict[x] for x in ex}
    order_books = get_order_books(ins,ex)
    save_this = (order_books,now.strftime("%Y-%m-%d  %H:%M:%S"))
    return json.dumps(save_this)

@app.callback(Output('spr-time','children'),
            [Input('spr-second','n_intervals'),Input('the-spr-data','children')])
def update_time(n,order_books):
    time_snap = json.loads(order_books)[1]
    return (dt.datetime.now()-dt.datetime.strptime(time_snap,"%Y-%m-%d  %H:%M:%S")).seconds

@app.callback([Output('spr-order-book-chart','figure'),Output('spr-market-depth','figure'),
            Output('spr-order-table','data'),#Output('spr-order-table','columns'),
            Output('spr-liquidity-table','children'),
            Output('spr-depth-table','children'),Output('spr-stat-table','children')],
            [Input('the-spr-data','children'),Input('spr-choose-base','value'),
            Input('spr-ins','value'),Input('spr-exchanges','value'),
            Input('spr-x-scale','value'),Input('spr-y-scale','value'),
            Input('spr-cutoff','value'),Input('spr-agg-level','value')])
def update_page(order_books,base,ins,exchanges,x_scale,y_scale,cutoff,step):
    #load data
    step = 10**(step-2)/10000
    relative = x_scale == 'Relative'
    currency = y_scale == 'Currency'
    order_books = json.loads(order_books)[0]
    order_books= {key:order_books[key] for key in order_books if key in exchanges}
    #plot book
    book_plot = plot_book(order_books,ins,exchanges,relative,currency,cutoff)
    #plot depth
    depth_plot = plot_depth(order_books,ins,exchanges,relative,currency,cutoff)
    # order table columns
    r = int(np.ceil(-np.log(step)/np.log(10)))-2
    columns_ob=[{'id':'from_mid','name':'From Mid','type':'numeric','format':FormatTemplate.percentage(4).sign(Sign.positive)},
                {'id':'price','name':'Price'},{'id':'size','name':'Size'},{'id':'cum_size','name': 'Size Total'},
                {'id':'size_$','name':'Size $','type':'numeric','format':FormatTemplate.money(0)},
                {'id':'cum_size_$','name':'Size Total $','type':'numeric','format':FormatTemplate.money(0)},
                {'id':'average_fill','name':'Averge Fill'},{'id':'exc','name':'Exchange'},
                {'id':'side','name':'side','hidden':True}],
    # order table data
    df =  build_book(order_books,ins,exchanges,cutoff,step)
    df_bids = df[[i for i in df.columns if 'bid' in i]].dropna().iloc[:13]
    df_bids['side'] = 'bid'
    df_bids.columns=['price','size','cum_size','size_$','cum_size_$','average_fill','exc','side']
    df_asks = df[[i for i in df.columns if 'ask' in i]].dropna().iloc[:13]
    df_asks['side'] = 'ask'
    df_asks.columns = ['price','size','cum_size','size_$','cum_size_$','average_fill','exc','side']
    mid=(df_bids['price'].max() + df_asks['price'].min())/2
    df_all=pd.concat([df_asks.sort_values(by='price',ascending=False),df_bids]).rename_axis('from_mid')
    rounding = [int(np.ceil(-np.log(mid*step)/np.log(10)))+1]
    decimals=pd.Series(rounding+[1]*4+rounding+[2]*2,index=df_all.columns)
    df_all=df_all.round(decimals).reset_index()
    df_all['from_mid'] = (df_all['from_mid']-1)
    data_ob = df_all.to_dict('rows')
    try:
        pair = base +'/USD'
        liq_dfs = [df.round(4) for df in get_liq_params(df,pair,step)]
        liq_tables = [dash_table.DataTable(
                data=liq_df.to_dict('rows'),
                columns=[{'id': c,'name':c} for c in liq_df.columns],
                style_header={'backgroundColor':'#DCDCDC','fontWeight':'bold'},
                style_cell={'textAlign':'center','width':'10%'},
                style_table={'border': '1px solid lightgrey','border-collapse':'collapse'},
                ) for liq_df in liq_dfs]
    except:
        liq_tables=[0]*3
 
    return (book_plot,depth_plot,data_ob) + tuple(liq_tables)

@app.callback(Output('spr-order-to-send','data'),
            [Input('spr-order-table','active_cell'),Input('spr-ins','value')],
            [State('spr-order-table','data')])
def update_order(active_cell,ins,data):
    data_df=pd.DataFrame(data)
    row = data_df.iloc[active_cell[0]]
    columns = ['B/S','Ins','Qty','Limit price','exc']
    order_df=pd.DataFrame(columns=columns)
    size = row['cum_size'] if active_cell[1] == 3 else row['size']
    order_df.loc[0]=['B' if row['side']=='bid' else 'S',ins,size,row['price'],row['exc']]
    return order_df.to_dict('row')

@app.callback([Output('apr-send-order','children'),Output('spr-send-order','style')],
                [Input('spr-order-to-send','data')])
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


@app.callback(Output('spr-confirm','displayed'),
            [Input('spr-send-order','n_clicks')])
def display_confirm(n_clicks):
    return True

@app.callback(Output('spr-output-confirm','children'),
[Input('confirm','submit_n_clicks')],
[State('spr-order-to-send','data')])
def update_output(submit_n_clicks,order):
    if submit_n_clicks:
        order = pd.DataFrame(order).iloc[0]
        side ='buy' if order['B/S'] =='B' else 'sell'
        exc = order['exc']
        exch_dict[exc].create_limit_order(order['Ins'],side, str(order['Qty']), str(order['Limit price']))
        return 'Order sent'

@app.callback([Output('spr-open-orders','data'),Output('spr-open-orders','columns')],
            [Input('spr-interval-component','n_intervals')])
def update_open(interval):
    oo = get_open_orders()
    oo['Cancel']='X'
    data = oo.to_dict('rows')
    names=['Id','Type','Ins','B/S','Qty','Price','State','Filled','Exc','Cancel']
    columns =[{'id':id,'name':name} for id,name in zip(oo.columns,names)]
    columns[0]['hidden'] = True
    return data,columns

@app.callback(Output('spr-cancel-confirm','children'),
            [Input('spr-open-orders','active_cell')],
            [State ('spr-open-orders','data')])
def cancel_order(active_cell,open_orders):
    oo=pd.DataFrame(open_orders)
    if active_cell[1] == 8:
        exc = oo.iloc[active_cell[0]]['ex']
        order_id = oo.iloc[active_cell[0]]['id']
        exch_dict[exc].cancel_order(order_id)
        return 'Canceled {}  {}'.format(order_id ,exc)
    else:
        return active_cell

@app.callback([Output('spr-closed-orders','data'),Output('spr-closed-orders','columns')],
            [Input('spr-interval-component','n_intervals')])
def update_closed(interval):
    year = pd.to_datetime('today').year
    month = pd.to_datetime('today').month
    day = pd.to_datetime('today').day
    midnight = pd.to_datetime(str(year)+'-'+str(month)+'-'+str(day)).timestamp()*10**3
    co = get_closed_orders(midnight)
    data = co.to_dict('rows')
    names=['Id','Type','Ins','B/S','Qty','Price','Average','Filled','Exc']
    columns =[{'id':id,'name':name} for id,name in zip(co.columns,names)]
    columns[0]['hidden'] = True
    return data,columns

@app.callback(Output('spr-balances','children'),
            [Input('spr-interval-component','n_intervals')])
def update_balance(interval):
    b = get_balances()
    return dash_table.DataTable(
                data=b.to_dict('rows'),
                columns=[{'id': c,'name':c} for c in b.columns],
                style_header={'backgroundColor':'#DCDCDC','fontWeight':'bold'},
                style_cell={'textAlign':'center','width':'10%'},
                style_table={'border': '1px solid lightgrey','border-collapse':'collapse'},
                )