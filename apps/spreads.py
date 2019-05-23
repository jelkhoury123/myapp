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
sys.path.append('../dashapp') # add parent directory to the path to import app
import deribit_api3 as my_deribit
from requests.auth import HTTPBasicAuth

from app import app

# If websocket use diginex.ccxt library and reduce update interval from 7 to 5 secs

ENABLE_WEBSOCKET_SUPPORT = False
refresh_rate = 5 if ENABLE_WEBSOCKET_SUPPORT else 7
if ENABLE_WEBSOCKET_SUPPORT:
    import diginex.ccxt.websocket_support as ccxt
else:
    import ccxt

# Define my world
deriv_exchanges =['bitmex','deribit']

exchanges =  deriv_exchanges

api_keys = {'deribit':'4v58Wk2hhiG9B','bitmex':'DM55KFt84AfdjJiyNjDBk_km'}
api_secrets = {'deribit':'W654NMZDNCQLNXC2UC3VHQEZ4P3M5C55','bitmex':'jPDWiZcKuXJtVpajawENQiBzCKO2U885i3TWU9WIihBBUZgc'}

exch_dict={}
for x in exchanges:
    exec('exch_dict[x]=ccxt.{}({{"apiKey": "{}", "secret": "{}"}})'.format(x, api_keys[x], api_secrets[x]))
    #exec('exch_dict[x]=ccxt.{}()'.format(x))
for x,xccxt in exch_dict.items():
    xccxt.load_markets()

deribit = exch_dict['deribit']
bitmex = exch_dict['bitmex']

Xpto_main = ['BTC','ETH']
Xpto_alt = ['ADA','BCH','EOS','LTC','TRX','XRP']
Xpto_sym = {'BTC':'฿','ETH':'⧫','ADA':'ADA','BCH':'BCH','EOS':'EOS','LTC':'LTC','TRX':'TRX','XRP':'XRP'}

def get_d1_instruments():
    '''
    returns a dictionary with keys deribit and bitmex
    each of which is in turn a dictionary with keys: BTC ETH and values: list containing Futures and Swaps traded on base 
    and not expired
    '''
    all_ins={}
    inversed={}
    tick_sizes={}
    for exc,exc_obj in exch_dict.items():
        all_ins[exc]={}
        for base in Xpto_main+Xpto_alt:
            base_list=[]
            for ins in getattr(exc_obj,'markets'):
                market = getattr(exc_obj,'markets')[ins]
                if market['type'] in ('future','swap') and market['base']==base and not ins.startswith('.') and '_' not in ins:
                    if exc == 'bitmex':
                        expiry = market['info']['expiry']
                        if expiry is None:
                            base_list.append(ins)
                            inversed[ins]=True if market['info']['positionCurrency']=='USD' or market['info']['positionCurrency']=='' else False 
                            tick_sizes[ins]=market['info']['tickSize']
                        else:
                            dt_expiry = dt.datetime.strptime(expiry,"%Y-%m-%dT%H:%M:%S.%fZ")
                            if dt_expiry > dt.datetime.now():
                                base_list.append(ins)
                                inversed[ins]=True if market['info']['positionCurrency']=='USD' or market['info']['positionCurrency']=='' else False
                                tick_sizes[ins]=market['info']['tickSize']
                    else:
                        base_list.append(ins)
                        inversed[ins]=True if 'positionCurrency' in market['info'].keys() and len(base_list)!=0 else True
                        tick_sizes[ins]=market['info']['tickSize']
            if len(base_list)!=0 and base in Xpto_main:
                all_ins[exc][base] = base_list
            elif len(base_list)!=0 and base in Xpto_alt and 'ALT' in all_ins[exc].keys():
                all_ins[exc]['ALT']+=base_list
            elif len(base_list)!=0 and base in Xpto_alt:
                all_ins[exc].update({'ALT':base_list})
    return all_ins, inversed, tick_sizes
            
instruments, inversed, ticks = get_d1_instruments()
deribit_d1_ins , bitmex_d1_ins = instruments['deribit'], instruments['bitmex']

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
    order_books = {key: value.fetch_order_book(ins,limit=250,
                        params={'full':1,'level':3,'limit_bids':0,'limit_asks':0,'type':'both'}) for key,value in ex.items() }
    try:
        order_books['deribit']=my_deribit.get_order_book(ins,250)
    except:
        pass

    if 'deribit' in order_books and 'BTC' in ins:
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

def normalize_order_book(ins,order_book, cutoff = 0.1, step = 0.001):

    '''
    ins is needed to determine if inverse
    order_book is a dictionary with keys bids asks timestamp datetime ...
    where bids is a list of list [[bid,bid_size,exchange]] and 
    asks is a list of list [[ask,ask_size,exchange]]
    this is returned by aggregate_order_books
    returns a dataframe with columns [ask, ask_size, ask_size_$, cum_ask_size_$, bid_, bid_size, bid_size_$, cum_bid_size_$]
    horizontal stack
    and an index of shape np.linspace(1 - cutoff,1 + cutoff ,step =.001 ~ 10 bps)
    cutoff % depth around mid
    step : aggregation level of prices

    '''
    try:
        rounding = int(np.ceil(-np.log(step)/np.log(10)))
        agg = True
    except:
        agg = False
    if inversed[ins]:
        bid_side = pd.DataFrame(order_book['bids'], columns = ['bid', 'bid_size_$', 'exc'])
        bid_side['cum_bid_size_$'] = bid_side['bid_size_$'].cumsum()
        ask_side = pd.DataFrame(order_book['asks'], columns = ['ask', 'ask_size_$', 'exc'])
        ask_side['cum_ask_size_$'] = ask_side['ask_size_$'].cumsum()
        ref = (bid_side['bid'][0] + ask_side['ask'][0])/2
        bid_side['bid%'] = round(bid_side['bid']/ref, rounding) if agg else bid_side['bid']/ref
        ask_side['ask%'] = round(ask_side['ask']/ref, rounding) if agg else ask_side['ask']/ref
        bid_side = bid_side[bid_side['bid%']>=1-cutoff]
        ask_side = ask_side[ask_side['ask%']<=1+cutoff]
        bid_side['bid_size'] = bid_side['bid_size_$']/bid_side['bid']
        bid_side['cum_bid_size'] = bid_side['bid_size'].cumsum()
        ask_side['ask_size'] = ask_side['ask_size_$']/ask_side['ask']
        ask_side['cum_ask_size'] = ask_side['ask_size'].cumsum()
    else:
        bid_side = pd.DataFrame(order_book['bids'], columns = ['bid', 'bid_size', 'exc'])
        bid_side['cum_bid_size'] = bid_side['bid_size'].cumsum()
        ask_side = pd.DataFrame(order_book['asks'], columns = ['ask', 'ask_size', 'exc'])
        ask_side['cum_ask_size'] = ask_side['ask_size'].cumsum()
        ref = (bid_side['bid'][0] + ask_side['ask'][0])/2
        bid_side['bid%'] = round(bid_side['bid']/ref, rounding) if agg else bid_side['bid']/ref
        ask_side['ask%'] = round(ask_side['ask']/ref, rounding) if agg else ask_side['ask']/ref
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

def build_book(order_books,ins,exchanges,cutoff=.1,step=0.001):
    ''' gets order books aggreagtes them then normalizes
        returns a dataframe
    '''
    return normalize_order_book(ins,aggregate_order_books({key:order_books[key] for key in exchanges}),cutoff,step)

def process_ob_for_dashtable(base, ins, df, step):
    # order table data
    
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

    # order table columns
    precision = len(str(ticks[ins]).split('.')[1]) if '.' in str(ticks[ins]) else int(str(ticks[ins]).split('e-')[1])
    mid=(df_bids['price'].max() + df_asks['price'].min())/2
    rounding = max(min(int(np.ceil(-np.log(mid*step)/np.log(10))), precision),0) if step !=0 else precision
    r = int(np.ceil(-np.log(step)/np.log(10)))-2 if step !=0 else int(np.ceil(-np.log(10**-precision/mid)/np.log(10))-2)

    base_sym = Xpto_sym[base] if base!='ALT' else Xpto_sym[ins[:3]]
    quote_sym = '$' if inversed[ins] else '฿'

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

def build_spread_book(nob1,nob2):
    bid_columns = [col for col in nob1.columns if 'bid' in col]
    ask_columns = [col for col in nob1.columns if 'ask' in col]
    nob2_asks = nob2[ask_columns].dropna()
    nob1_bids = nob1 [bid_columns].dropna()
    nob1_bids.columns = [c.replace('bid','') for c in nob1_bids.columns]
    nob2_asks.columns = [c.replace('ask','') for c in nob2_asks.columns]
    spread_book_asks = pd.DataFrame(columns=nob2_asks.columns,index = range((len(nob2_asks)+len(nob1_bids))+1))
    nob2_bids = nob2[bid_columns].dropna()
    nob1_asks = nob1 [ask_columns].dropna()
    nob1_asks.columns = [c.replace('ask','') for c in nob1_asks.columns]
    nob2_bids.columns = [c.replace('bid','') for c in nob2_bids.columns]
    spread_book_bids = pd.DataFrame(columns=nob2_bids.columns,index= range((len(nob2_bids)+len(nob1_asks))+1))
    #build ask side
    i = 0
    while len(nob1_bids)>1 and len(nob2_asks) >1:
        trade_buy = (nob2_asks.iloc[0,:-1]-nob1_bids.iloc[0,:-1])
        #trade_buy=trade_buy.append(pd.Series([nob2_asks.iloc[0,-1]+'/'+nob1_bids.iloc[0,-1]],index = ['s_exc']))
        ask_emptied = True if trade_buy[3]<0 else False
        trade_buy [1:-1] = nob2_asks.iloc[0,1:-2] if ask_emptied else nob1_bids.iloc[0,1:-2]
        spread_book_asks.iloc[i]=trade_buy
        if ask_emptied :
            nob2_asks = nob2_asks.drop(nob2_asks.index[0])
            nob1_bids.iloc[0,1:-2] =  nob1_bids.iloc[0,1:-2] - trade_buy[1:-1]
        else:
            nob1_bids = nob1_bids.drop(nob1_bids.index[0])
            nob2_asks.iloc[0,1:-2] = nob2_asks.iloc[0,1:-2] - trade_buy[1:-1]
        i+=1
    #build bid side
    i = 0
    while len(nob1_asks)*len(nob2_bids) !=0:
        trade_sell = (nob2_bids.iloc[0,:-1]-nob1_asks.iloc[0,:-1])
        #trade_sell=trade_sell.append(pd.Series([nob2_bids.iloc[0,-1]+'/'+nob1_asks.iloc[0,-1]],index = ['s_exc']))
        bid_emptied = True if trade_sell[3]<0 else False
        trade_sell [1:-1] = nob2_bids.iloc[0,1:-2].astype(float) if bid_emptied else nob1_asks.iloc[0,1:-2].astype(float)
        spread_book_bids.iloc[i]=trade_sell
        if bid_emptied :
            nob2_bids = nob2_bids.drop(nob2_bids.index[0])
            nob1_asks.iloc[0,1:-2] =  nob1_asks.iloc[0,1:-1]-trade_sell[1:-1]
        else:
            nob1_asks = nob1_asks.drop(nob1_asks.index[0])
            nob2_bids.iloc[0,1:-2] =  nob2_bids.iloc[0,1:-1]-trade_sell[1:-1]
        i+=1
    spread_book_asks.columns = ask_columns
    spread_book_bids.columns = bid_columns
    spread_book_asks['cum_ask_size'] = spread_book_asks['ask_size'].cumsum()
    spread_book_bids['cum_bid_size'] = spread_book_bids['bid_size'].cumsum()
    spread_book_asks['cum_ask_size_$'] = spread_book_asks['ask_size_$'].cumsum()
    spread_book_bids['cum_bid_size_$'] = spread_book_bids['bid_size_$'].cumsum()
    spread_book = pd.concat([spread_book_asks,spread_book_bids],sort=False).astype(float)
    spread_book['asks_exc'] = nob2_asks.iloc[0,-1]+'/'+nob1_bids.iloc[0,-1]
    spread_book['bids_exc'] = nob2_asks.iloc[0,-1]+'/'+nob1_bids.iloc[0,-1]
    
    return spread_book

title = 'Spreads'

layout =  html.Div( className = 'row', style = {'marginLeft':35,'marginRight':35,'margin-top':'20px'}, children = [
    html.Div( className = 'row', style = {'border-bottom':'1px solid #cb1828','padding-bottom':'30px','margin-bottom':'30px'}, children = [
        html.Div( className = 'two columns', children = [
            html.Label('Choose Base', style = {'margin-bottom':'10px','font-size':'1.4rem','font-weight':'bold'}),
            dcc.RadioItems( id = 'spread-base',
                            options = [{'label':base,'value':base} for base in Xpto_main+['ALT']],
                            value = Xpto_main[0],
                            labelStyle = {'display':'inline-block','margin-right':'10px','font-size':'1.2rem'}),]),
        html.Div( className = 'four columns', children = [
            html.Label('Available Exchanges', style = {'margin-bottom':'10px','font-size':'1.4rem','font-weight':'bold'}),
            dcc.Dropdown(   id = 'spread-sup-exchanges',
                            options = [{'label':i,'value':i} for i in instruments.keys()],
                            value = [i for i in instruments.keys()],
                            style = {'border-color':'#cb1828'},
                            multi = True),]),
        html.Div( className = 'three columns', children = [
            html.Label('Price Agg (bps):', style = {'margin-bottom':'10px','font-size':'1.4rem','font-weight':'bold'}),
            html.Div( className = 'row', style = {'width':'75%','font-size':'1.2rem'}, children = [
                dcc.Slider( id='spread-agg-level',
                            max = 4, step = 1, value = 3,
                            marks = {i:str(10**(i-2) if i != 0 else 0) for i in range(0,5)})]),]),
        html.Div( className = 'three columns', children = [
            html.Label('Cutoff % :', style = {'margin-bottom':'10px','font-size':'1.4rem','font-weight':'bold'}),
            html.Div( className = 'row', style = {'width':'75%','font-size':'1.2rem'}, children = [
                dcc.Slider( id = 'spread-cutoff',
                            min = .05, max = .3, step = .05, value = .1,
                            marks = {round(j,2): str(round(j,2)) for j in list(np.arange(.05,.35,.05))})]),]),]),
    html.Div(className = 'row', children = [
        html.Div(className = 'four columns', children = [
            html.Div( className = 'row', children = [
                html.Label('Instrument [+]', style = {'margin-bottom':'10px','font-size':'1.4rem'}),
                html.Div( className = 'row', children = [
                    html.Div( style = {'width':'30%', 'display':'inline-block'}, children = [
                        dcc.Dropdown(   id = 'spread-ins2-ex', style = {'border-color':'#cb1828'},
                                #options = [{'label':i,'value':i} for i in instruments.keys()],
                                #value = list(instruments.keys())[0],
                                )]),
                    html.Div( style = {'width':'70%', 'display':'inline-block'}, children = [
                        dcc.Dropdown(   id = 'spread-ins2', style = {'border-color':'#cb1828'},
                                #options = [{'label':i,'value':i} for i in instruments['deribit']['BTC']],
                                #value = instruments['deribit']['BTC'][1],
                                )]),]),]),
            html.Div( children=[
                html.Br(),
                dash_table.DataTable(   id = 'spread-order-table2',
                                        style_table = {'border': '1px solid lightgrey','border-collapse':'collapse'},
                                        style_cell = {'textAlign':'center','width':'12%'},
                                        style_header={'textAlign':'center','backgroundColor':'#DCDCDC','fontWeight':'bold'},
                                        style_data_conditional =    [{'if' : {'filter':  'side eq "bid"' }, 'color':'blue' }] +
                                                                    [{'if' : {'filter': 'side eq "ask"' }, 'color':'rgb(203,24,40)' }] +
                                                                    [{'if': {'row_index':'odd'}, 'backgroundColor':'rgb(242,242,242)'}] +
                                                                    [{'if':{'column_id':'price'}, 'fontWeight':'bold','border': 'thin lightgrey solid'}] +
                                                                    [{'if':{'column_id':'from_mid'}, 'fontWeight':'bold'}] +
                                                                    [{'if' : {'column_id': c }, 'textAlign': 'right','padding-right' : '2%'} for c in ['size','cum_size','size_$','cum_size_$'] ],),
                html.P(id = 'spread-ob2-last-timestamp', style = {'display':'inline-block','font-size':'1.2rem'}),
                html.P(children=' / ', style = {'display':'inline-block', 'margin':'0px 5px','font-size':'1.2rem'}),
                html.P(id = 'spread-ob2-new-timestamp', children = dt.datetime.now().strftime('%X'), style = {'display':'inline-block','font-size':'1.2rem'}),]),]),
        html.Div(className = 'four columns', children = [
            html.Div( className = 'row', children = [
                html.Label('Instrument [-]', style = {'margin-bottom':'10px','font-size':'1.4rem'}),
                html.Div( className = 'row', children = [
                    html.Div( style = {'width':'30%', 'display':'inline-block'}, children = [
                        dcc.Dropdown(   id = 'spread-ins1-ex', style = {'border-color':'#cb1828'},
                                #options = [{'label':i,'value':i} for i in instruments.keys()],
                                #value = list(instruments.keys())[0],
                                )]),
                    html.Div( style = {'width':'70%', 'display':'inline-block'}, children = [
                        dcc.Dropdown(   id = 'spread-ins1', style = {'border-color':'#cb1828'},
                                #options = [{'label':i,'value':i} for i in instruments['deribit']['BTC']],
                                #value = instruments['deribit']['BTC'][0],
                                )]),]),]),
            html.Div( children=[
                html.Br(),
                dash_table.DataTable(   id = 'spread-order-table1',
                                        style_table = {'border': '1px solid lightgrey','border-collapse':'collapse'},
                                        style_cell = {'textAlign':'center','width':'12%'},
                                        style_header={'textAlign':'center','backgroundColor':'#DCDCDC','fontWeight':'bold'},
                                        style_data_conditional =    [{'if' : {'filter':  'side eq "bid"' }, 'color':'blue' }] +
                                                                    [{'if' : {'filter': 'side eq "ask"' }, 'color':'rgb(203,24,40)' }] +
                                                                    [{'if': {'row_index':'odd'}, 'backgroundColor':'rgb(242,242,242)'}] +
                                                                    [{'if':{'column_id':'price'}, 'fontWeight':'bold','border': 'thin lightgrey solid'}] +
                                                                    [{'if':{'column_id':'from_mid'}, 'fontWeight':'bold'}] +
                                                                    [{'if' : {'column_id': c }, 'textAlign': 'right','padding-right' : '2%'} for c in ['size','cum_size','size_$','cum_size_$'] ],),
                html.P(id = 'spread-ob1-last-timestamp', style = {'display':'inline-block','font-size':'1.2rem'}),
                html.P(children=' / ', style = {'display':'inline-block', 'margin':'0px 5px','font-size':'1.2rem'}),
                html.P(id = 'spread-ob1-new-timestamp', children = dt.datetime.now().strftime('%X'), style = {'display':'inline-block','font-size':'1.2rem'}),]),]),
        html.Div(className = 'four columns', children = [
            html.Div( className = 'row', children = [
                html.Label('Spread', style = {'margin-bottom':'5px','font-size':'1.4rem'}),
                html.H6( id = 'spread-spread-pair1'),
                html.H6( id = 'spread-spread-pair2')]),
            html.Div( children=[
                html.Br(),
                dash_table.DataTable(   id = 'spread-book',
                                        style_table = {'border': '1px solid lightgrey','border-collapse':'collapse'},
                                        style_cell = {'textAlign':'center','width':'12%'},
                                        style_header={'textAlign':'center','backgroundColor':'#DCDCDC','fontWeight':'bold'},
                                        style_data_conditional =    [{'if' : {'filter':  'side eq "bid"' }, 'color':'blue' }] +
                                                                    [{'if' : {'filter': 'side eq "ask"' }, 'color':'rgb(203,24,40)' }] +
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
@app.callback(
    [Output('spread-sup-exchanges','options'),Output('spread-sup-exchanges','value')],
    [Input('spread-base','value')]
    )
def update_spread_sup_exchs(radio_button_value):
    sup_exchanges = [i for i in instruments if radio_button_value in instruments[i].keys()]
    options_sup_exchanges = [{'label':exch,'value':exch} for exch in sup_exchanges]
    value_sup_exchanges = sup_exchanges
    return options_sup_exchanges, value_sup_exchanges

@app.callback(
    [Output('spread-ins1-ex','options'), Output('spread-ins1-ex','value'),
     Output('spread-ins2-ex','options'), Output('spread-ins2-ex','value')],
    [Input('spread-sup-exchanges','value')]
)
def update_spread_ins_exchs(exs):
    options_ex = [{'label':i,'value':i} for i in exs]
    values_ex = exs[0]
    return options_ex, values_ex, options_ex, values_ex

@app.callback(
    [Output('spread-ins1','options'), Output('spread-ins1','value'),
     Output('spread-ins2','options'), Output('spread-ins2','value')],
    [Input('spread-ins1-ex','value'), Input('spread-ins2-ex','value'), Input('spread-base','value')]
)
def update_ins(ins1_ex, ins2_ex, base):
    options1 = [{'label':i,'value':i} for i in instruments[ins1_ex][base]]
    values1 = instruments[ins1_ex][base][0]
    options2 = [{'label':i,'value':i} for i in instruments[ins2_ex][base]]
    values2 = instruments[ins2_ex][base][1]
    return options1, values1, options2, values2

@app.callback(
    [Output('spread-spread-pair1','children'), Output('spread-spread-pair1','style'),
     Output('spread-spread-pair2','children'), Output('spread-spread-pair2','style')],
    [Input('spread-ins1-ex','value'), Input('spread-ins1','value'),
     Input('spread-ins2-ex','value'), Input('spread-ins2','value')],
)
def update_spread_pair_label(ins1_ex,ins1,ins2_ex,ins2):
    style1 = {'color': 'red', 'width':'50%', 'display':'inline-block'} if inversed[ins1] else {'color': 'green', 'width':'50%', 'display':'inline-block'}
    style2 = {'color': 'red', 'width':'50%', 'display':'inline-block', 'text-align':'right'} if inversed[ins2] else {'color': 'green', 'width':'50%', 'display':'inline-block', 'text-align':'right'}
    return '[+] {} ({})'.format(ins2, ins2_ex), style1, '[-] {} ({})'.format( ins1, ins1_ex), style2

@app.callback(Output('spread-data','children'),
            [Input('spread-ins1','value'),Input('spread-ins2','value'),
             Input('spread-ins1-ex','value'),Input('spread-ins2-ex','value'),Input('spread-interval-component','n_intervals')])
def update_data(ins1,ins2,ex1,ex2,n):
    now = dt.datetime.now()
    ob1, ob2 = get_order_books(ins1,{ex1:exch_dict[ex1]}), get_order_books(ins2,{ex2:exch_dict[ex2]})
    save_this = (ob1, ob2, now.strftime("%Y-%m-%d  %H:%M:%S"))
    return json.dumps(save_this)

@app.callback(
    [Output('spread-order-table1','data'), Output('spread-order-table1','columns'),
     Output('spread-order-table2','data'), Output('spread-order-table2','columns'),
     Output('spread-book','data'), Output('spread-book','columns'),
     Output('spread-ob1-last-timestamp', 'children'), Output('spread-ob2-last-timestamp', 'children'), Output('spread-sb-last-timestamp', 'children'),
     Output('spread-ob1-new-timestamp', 'children'), Output('spread-ob2-new-timestamp', 'children'), Output('spread-sb-new-timestamp', 'children')],
    [Input('spread-data','children'), Input('spread-base','value'),
     Input('spread-ins1','value'), Input('spread-ins1-ex','value'),
     Input('spread-ins2','value'),Input('spread-ins2-ex','value'),
     Input('spread-cutoff','value'), Input('spread-agg-level','value'), ],
    [State('spread-ob1-new-timestamp', 'children'), State('spread-ob2-new-timestamp', 'children'), State('spread-sb-new-timestamp', 'children')]
)
def update_tables(order_books, base, ins1, ex1, ins2, ex2, cutoff, step, ob1_last_update, ob2_last_update, sb_last_update):
    
    step = 10**(step-2)/10000 if step !=0 else step
    order_books = json.loads(order_books)
    ob1 = {key:order_books[0][key] for key in order_books[0] if key in ex1}
    ob2 = {key:order_books[1][key] for key in order_books[1] if key in ex2}
    
    nob1 = build_book(ob1,ins1,[ex1],cutoff,step)
    nob2 = build_book(ob2,ins2,[ex2],cutoff,step)
    # Orderbooks Formatting
    data1, columns1 = process_ob_for_dashtable(base, ins1, nob1, step)
    ob1_new_time = dt.datetime.now().strftime('%X')
    data2, columns2 = process_ob_for_dashtable(base, ins2, nob2, step)
    ob2_new_time = dt.datetime.now().strftime('%X')

    # Spreadbook Formatting
    df = build_spread_book(nob1,nob2)
    df_bids = df[[i for i in df.columns if 'bid' in i ]].dropna().iloc[:13]
    df_bids['side'] = 'bid'
    df_bids.columns=['price','size','cum_size','size_$','cum_size_$','average_fill','exc','side']

    df_asks = df[[i for i in df.columns if 'ask' in i ]].dropna().iloc[:13]
    df_asks['side'] = 'ask'
    df_asks.columns = ['price','size','cum_size','size_$','cum_size_$','average_fill','exc','side']
    mid = (df_bids['price'].max() + df_asks['price'].min())/2
    df_bids['from_mid'] = df_bids['price']/mid
    df_asks['from_mid'] = df_asks['price']/mid
    df_all = pd.concat([df_asks.sort_values(by='price',ascending=False), df_bids])
    df_all['from_mid'] = (df_all['from_mid']-1)
    data_ob = df_all.to_dict('rows')
    sb_new_time = dt.datetime.now().strftime('%X')

    columns_ob=[{'id':'from_mid','name':'From Mid','type':'numeric','format':FormatTemplate.percentage(2).sign(Sign.positive)},
                {'id':'price','name':'Price','type':'numeric','format':Format(precision=2,scheme=Scheme.fixed)},
                {'id':'size','name':'Size','type':'numeric','format':Format(precision=2,scheme=Scheme.fixed)},
                {'id':'cum_size','name': 'Size Total','type':'numeric','format':Format(precision=2,scheme=Scheme.fixed)},
                {'id':'size_$','name':'Size $','type':'numeric','format':FormatTemplate.money(0)},
                {'id':'cum_size_$','name':'Size Total $','type':'numeric','format':FormatTemplate.money(0)},
                {'id':'average_fill','name':'Averge Fill','type':'numeric','format':Format(precision=2,scheme=Scheme.fixed)},
                {'id':'exc','name':'Exchange', 'hidden': True},
                {'id':'side','name':'side','hidden':True}]
    return data1, columns1, data2, columns2, data_ob, columns_ob, ob1_last_update, ob2_last_update, sb_last_update, ob1_new_time, ob2_new_time, sb_new_time