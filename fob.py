import pandas as pd
import numpy as np
import itertools
import json
import datetime as dt 
import os
import sys
import plotly.graph_objs as go 
import plotly.figure_factory as ff
import pandas as pd
import dash_table 
from dash_table.Format import Format, Scheme, Sign, Symbol
import dash_table.FormatTemplate as FormatTemplate
sys.path.append('..') # add parent directory 
import deribit_api3 as my_deribit
from requests.auth import HTTPBasicAuth

ENABLE_WEBSOCKET_SUPPORT = False
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
#Alt coins
Xpto_alt = ['ADA','BCH','EOS','LTC','TRX','XRP']
#Symbols
Xpto_sym = {i:i for i in Xpto_alt}
Xpto_sym.update({'BTC':'฿','ETH':'⧫'})

Fiat = ['USD','EUR','GBP','CHF','HKD','JPY','CNH']

def get_d1_instruments():
    '''
    returns a tuple of 
    1. a dictionary with keys deribit and bitmex ...
        each of which is in turn a dictionary with keys: base BTC ETH ALT
        and values: a list containing Futures and Swaps traded on base and not expired
    2. a dictionary {ins:True if inverse instrument}
    3. a dictionary {ins:tick size}
    '''
    all_ins,inversed,tick_sizes={},{},{}
    for exc,exc_obj in exch_dict.items():
        all_ins[exc]={}
        for base in Xpto_main + Xpto_alt:
            base_list=[]
            for ins in getattr(exc_obj,'markets'):
                market = getattr(exc_obj,'markets')[ins]
                if market['type'] in ('future','swap') and market['base'] == base and not ins.startswith('.') and '_' not in ins:
                    if exc == 'bitmex':
                        expiry = market['info']['expiry']
                        if expiry is None:
                            base_list.append(ins)
                            inversed[ins] = True if market['info']['positionCurrency'] in ('USD','') else False 
                            tick_sizes[ins]=market['info']['tickSize']
                        else:
                            dt_expiry = dt.datetime.strptime(expiry,"%Y-%m-%dT%H:%M:%S.%fZ")
                            if dt_expiry > dt.datetime.now():
                                base_list.append(ins)
                                inversed[ins]=True if market['info']['positionCurrency'] in ('USD','') else False
                                tick_sizes[ins]=market['info']['tickSize']
                    else:
                        base_list.append(ins)
                        inversed[ins] = True 
                        tick_sizes[ins]=market['info']['tickSize']
            if len(base_list)!=0 and base in Xpto_main:
                all_ins[exc][base] = base_list
            elif len(base_list)!=0 and base in Xpto_alt and 'ALT' in all_ins[exc].keys():
                all_ins[exc]['ALT']+=base_list
            elif len(base_list)!=0 and base in Xpto_alt:
                all_ins[exc].update({'ALT':base_list})
    return all_ins, inversed, tick_sizes


instruments,inversed, ticks = get_d1_instruments()
deribit_d1_ins , bitmex_d1_ins = instruments['deribit'], instruments['bitmex']


def get_exchanges_for_ins(ins):
    '''input: an instrument 
        output: a dictionary of ccxt exchange objects were the instrument is listed
    '''
    return {x:exch_dict[x] for x in exch_dict if ins in list(exch_dict[x].load_markets().keys())}

def get_ins_for_exchange(ex):
    '''input: an exchange
    output: a list of instruments traded on the exchange'''
    d={}
    exec('d[ex]=ccxt.{}()'.format(ex))
    d[ex].load_markets()
    return d[ex].symbols

def get_order_books(ins,ex,size=100):
    '''ins is the instrument string example 'BTC-PERPETUAL',
        ex is a dictionary of ccxt objects where ins is traded
        returns a dictionary of order books for the instrument
        We call order book on every exchange where the ins is trading and 
        Return a dictionary {exchange:order_book} where order_book is a dictionary :
        {'bids': [[price,quantity]],'asks':[[price,quantity]]}
        Deribit needs a 10 multiplier 
    '''
    if not 'deribit' in ex.keys():
        order_books = {key: value.fetch_order_book(ins,limit=size,
        params={'full':1,'level':3,'limit_bids':0,'limit_asks':0,'type':'both'}) for key,value in ex.items() }
    else:
        order_books={}
        order_books['deribit']=my_deribit.get_order_book(ins,size*2)
    
    return order_books

def aggregate_order_books(dict_of_order_books):
    '''dict_of_order_books is a dict of ccxt like order_books returned by get_order_books with ex name added
        retuns a ccxt like dictionary order book sorted by prices (add exc name on every bid and ask):
        {'bids':[[price,quantity,exchange]],'asks':[[price,quantity,exchange]]}
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

    if ins in inversed:
        if  inversed[ins]:
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
    #rounding = [int(np.ceil(-np.log(mid*step)/np.log(10)))+1]
    df_all=df_all.reset_index()
    df_all['from_mid'] = (df_all['from_mid']-1)

    return df_all,mid

def process_ob_for_dashtable(base, ins, normalized_order_book, step=.001):
    # order table data
    
    df_all, mid  = build_stack_book(normalized_order_book,step=step)
    data_ob = df_all.to_dict('rows')

    # order table columns
    precision = len(str(ticks[ins]).split('.')[1]) if '.' in str(ticks[ins]) else int(str(ticks[ins]).split('e-')[1])
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
        
    layout = go.Layout(title = ' - '.join(exc),
                         xaxis = dict(title= ins +'  ' + str(best_bid)+' - '+ str(best_ask)),
                         showlegend=False, margin = {'t':25,'r': 10,'l': 25})
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
    clip = total_coins/(100*1000)   #10^-5 of number of coins available is my standard order size 
    ordersizes=np.array([clip * i for i in order_span]+[-clip* i for i in order_span]).astype(int)
    slippage = ((order_fill(normalized,ordersizes,False)-1)*100).round(2)
    #order book params
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
    ordersizes_dollar = ordersizes*mid
    result2 [0]=(ordersizes_dollar/1e6).round(1) if pair[-3:] in Fiat else (ordersizes_dollar).round(1)
    result2[1] = slippage
    result2=result2.T
    info = coindata[pair]['info']
    select_info=['symbol','rank','24h_volume_usd','market_cap_usd',
                'available_supply','percent_change_1h','percent_change_24h','percent_change_7d']
    selected_info={key:value for key,value in info.items() if key in select_info}
    result3 = pd.DataFrame(pd.Series(selected_info)).T
    result3.columns=['Coin','Rank','24H % Volume','Market Cap M$','Coins Supply M','% 1h','% 24h','% 7d']
    result3['24H % Volume']= round(float(result3['24H % Volume'])/float(result3['Market Cap M$'])*100,1)
    result3['Market Cap M$'] = round(float(result3['Market Cap M$'])/(1000*1000),0)
    result3['Coins Supply M'] = round(float(result3['Coins Supply M'])/(1000*1000),1)
    result3 = result3.astype({'24H % Volume':float,'% 1h':float,'% 24h':float,'% 7d':float})
    result3[['24H % Volume','% 1h','% 24h','% 7d']] = result3[['24H % Volume','% 1h','% 24h','% 7d']].div(100, axis = 0)
    return [result1,result2,result3]

def get_open_orders():
    deribit_open_orders = []
    for coin in deribit_d1_ins:
        for symbol in deribit_d1_ins[coin]:
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
       
    d_eth_closed = pd.DataFrame(my_deribit.get_order_history_by_currency(auth,'ETH',include_old='true').json()['result'])
    d_btc_closed = pd.DataFrame(my_deribit.get_order_history_by_currency(auth,'BTC',include_old='true').json()['result'])
    d_closed = pd.concat([d_btc_closed,d_eth_closed],sort=True)
    dcolumns = {'id':'order_id','type':'order_type','symbol':'instrument_name','side':'direction',
          'amount':'amount','price':'price','average':'average_price','filled':'filled_amount','timestamp':'creation_timestamp'}
    d_closed=d_closed[dcolumns.values()]
    d_closed.columns=dcolumns.keys()
    d_closed['ex']='deribit'

    bitmex_closed_orders=bitmex.fetch_closed_orders()
    for order in bitmex_closed_orders:
        order['ex']='bitmex'

    closed_orders =  bitmex_closed_orders

    for order in closed_orders:
        order.pop('info')

    if len(closed_orders) > 0:
        closed_orders = pd.DataFrame(closed_orders).sort_values(by=['timestamp'],ascending=False)
        closed_orders = closed_orders[closed_orders['timestamp']>start]
        columns = ['id','type','symbol','side','amount','price','average','filled','timestamp','ex']
        co=closed_orders[columns].copy()
    else: 
        columns = ['id','type','symbol','side','amount','price','average','filled','timestamp','ex']
        co= pd.DataFrame(columns=columns)

    closed_df = pd.concat([co,d_closed],sort = False )
    closed_df = closed_df.sort_values(by='timestamp',ascending = False)
    return closed_df[closed_df['timestamp']>start]

def get_balances():
    deribit_balance=deribit.fetch_balance()
    deribit_balance['total']['ETH'] = my_deribit.get_account_summary(auth,'ETH','true').json()['result']['equity']
    deribit_balance['free']['ETH'] = my_deribit.get_account_summary(auth,'ETH','true').json()['result']['available_funds']
    deribit_balance['used']['ETH'] = round(deribit_balance['total']['ETH'] - deribit_balance['free']['ETH'],4)
    d_balance={key:deribit_balance[key] for key in ['free','used','total']}
    d_balance['exc']='deribit'
    bitmex_balance=bitmex.fetch_balance()
    b_balance={key:bitmex_balance[key] for key in ['free','used','total']}
    b_balance['exc']='bitmex'
    b=pd.concat([pd.DataFrame(b_balance),pd.DataFrame(d_balance)])
    b = b[['exc','used','free','total']]
    return b

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