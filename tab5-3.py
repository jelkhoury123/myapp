import dash
import dash_core_components as dcc 
import dash_html_components as html 
import dash_table  
from dash.dependencies import Input, Output
import plotly.graph_objs as go 
import plotly.figure_factory as ffimport 
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import itertools
import ccxt
import json
import datetime as dt 

# Define my world
spot_exchanges = ['bitfinex','bitstamp','coinbasepro','kraken','liquid','gemini','binance',
                    'bitbank','huobipro','poloniex','bithumb','bittrex','kucoin2']

exch_dict={}
for x in spot_exchanges:
    exec('exch_dict[x]=ccxt.{}()'.format(x))

Xpto= ['BTC','ETH','XRP','XMR','BCH','EOS','USDT','USDC','TRX','XLM','BSV','XBT','CSP','DAI']
Fiat=['USD','EUR','GBP','CHF','HKD','JPY','CNH','KRW']
xpto_fiat = [xpto+'/'+ fiat for xpto in Xpto for fiat in Fiat]
xpto_xpto = [p[0]+'/'+p[1] for p in itertools.permutations(Xpto,2)]

all_pairs = set(sum(itertools.chain([*exch_dict[x].load_markets()] for x in exch_dict),[])) 
pairs = list(set(xpto_fiat + xpto_xpto) & set(all_pairs))
pairs.sort()

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

def build_book(order_books,pair,exchanges,cutoff=.1,step=0.001):
    ''' gets order books aggreagtes them then normalizes
        returns a dataframe
    '''
    return normalize_order_book(aggregate_order_books({key:order_books[key] for key in exchanges}),cutoff,step)

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
                        name='asks',marker=dict(color='rgba(255,0,0,0.6)'),fill='tozeroy',fillcolor='rgba(255,0,0,0.2)')
        trace_bids=go.Scatter(x=order_book.index,y=order_book['cum_bid_size'+col_to_chart],
                        name='asks',marker=dict(color='rgba(0,0,255,0.6)'),fill='tozeroy',fillcolor='rgba(0,0,255,0.2)')     
    else:
        trace_asks=go.Scatter(x=order_book['ask'].fillna(0)+order_book['bid'].fillna(0),y=order_book['cum_ask_size'+col_to_chart],
                        name='asks',marker=dict(color='rgba(255,0,0,0.6)'),fill='tozeroy',fillcolor='rgba(255,0,0,0.15)')
        trace_bids=go.Scatter(x=order_book['ask'].fillna(0)+order_book['bid'].fillna(0),y=order_book['cum_bid_size'+col_to_chart],
                        name='asks',marker=dict(color='rgba(0,0,255,0.6)'),fill='tozeroy',fillcolor='rgba(0,0,255,0.15)')
        
    layout = go.Layout(title = ' - '.join(exc), xaxis = dict(title= pair +'  ' + str(best_bid)+' - '+ str(best_ask)))
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
                        name='ask depth',marker=dict(color='rgba(255,0,0,0.6)'),fill='tozerox',fillcolor='rgba(255,0,0,0.15)')
    trace_bids = go.Scatter(x=-order_book['cum_bid_size'+col_to_chart],y=order_book['average_bid_fill']/mid,
                        name='bid depth',marker=dict(color='rgba(0,0,255,0.6)'),fill='tozerox',fillcolor='rgba(0,0,255,0.15)')
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
        elif order_size < 0:
            try:
                last_line = order_book_df[order_book_df['cum_bid_size'] > -order_size].iloc[0]
                ccy_fill=last_line['cum_bid_size_$']+(-order_size-last_line['cum_bid_size'])*last_line['bid']
                average_fill = -ccy_fill/order_size
            except:
                average_fill = np.nan
        average_fills[i] = average_fill
    return average_fills/mid
 
def get_liq_params(normalized,pair):
    coinmar = ccxt.coinmarketcap()
    coindata=coinmar.load_markets()
    total_coins=float(coindata[pair]['info']['available_supply'])  #number of coins floating
    order_span = (1,10,20,30,40)
    clip = total_coins/100000                                      #my standard order size 
    ordersizes=np.array([clip* i for i in order_span]+[-clip* i for i in order_span]).astype(int)
    slippage = ((order_fill(normalized,ordersizes,False)-1)*100).round(2)
    best_bid = normalized['bid'].max()
    best_ask = normalized['ask'].min()
    mid= (best_bid + best_ask)/2
    spread = best_ask-best_bid
    spread_pct = spread/mid*100
    cross = min(0,spread)
    cross_pct = min(0,spread_pct)
    result1 = pd.DataFrame([best_bid,best_ask,mid,spread,spread_pct,cross,cross_pct],
    index=['bid','ask','mid','spread','spread%','cross','cross%']).T
    result2 = pd.DataFrame(slippage,index=[str(o) for o in ordersizes]).T
    info = coindata[pair]['info']
    select_info=['symbol','rank','24h_volume_usd','market_cap_usd',
                'available_supply','percent_change_1h','percent_change_24h','percent_change_7d']
    selected_info={key:value for key,value in info.items() if key in select_info}
    result3 = pd.DataFrame(pd.Series(selected_info)).T
    result3.columns=['Coin','Rank','24H % Volume','USD Market Cap M$','Coins Supply M','% 1h','% 24h','% 7d']
    result3['24H % Volume']= round(float(result3['24H % Volume'])/float(result3['USD Market Cap M$'])*100,1)
    result3['USD Market Cap M$'] = round(float(result3['USD Market Cap M$'])/(1000*1000),0)
    result3['Coins Supply M'] = round(float(result3['Coins Supply M'])/(1000*1000),1)
    return [result1,result2,result3]
    

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
app.title = 'Order Books'
app.layout = html.Div(style={'marginLeft':25,'marginRight':25},
                    children=[ html.Div(className='row',children=[
                                        html.Div(className='six columns',
                                        children =[html.H6('Choose Pair'),
                                                    dcc.Dropdown(id='pairs',
                                                                options=[{'label':pair,'value':pair} for pair in pairs],
                                                                value='BTC/USD'),
                                                    html.Div(className='row',children=[
                                                    html.Div(className='three columns',children = [html.H6('\n \n X scale :')]),
                                                    html.Div(className='three columns',children=[dcc.RadioItems(id='x-scale',
                                                                options=[{'label':scale,'value':scale} for scale in ['Relative','Absolute']], 
                                                                value ='Relative',
                                                                labelStyle={'display':'inline-block'})])
                                                    ]),
                                                    html.Div(className='row',children=[
                                                    html.Div(className='three columns',children = [html.H6('\n \n Y scale :')]),
                                                    html.Div(className='three columns',children=[dcc.RadioItems(id='y-scale',
                                                                options=[{'label':scale,'value':scale} for scale in ['Currency','Coin']], 
                                                                value ='Currency',
                                                                labelStyle={'display':'inline-block'})])
                                                    ]),
                                                    html.H6('\n \n Cutoff'),
                                                    html.Div(style={'width' :'50%','align':'right'},children =[
                                                    dcc.Slider(id='cutoff',
                                                            min=.05,max=.3,step=.05,value=.1,
                                                            marks={round(j,2): str(round(j,2)) for j in list(np.arange(.05,.35,.05))})]),
                                                    html.H6(' ... '),        
                                                    html.H6(' Book Chart'),
                                                    dcc.Graph(id='order-book-chart'),
                                                    dcc.Graph(id='market-depth'),
                                                    html.H6(id='time')

                                        ]),
                                html.Div(className='six columns',
                                    children =[html.H6('Choose Exchange'),
                                                dcc.Dropdown(id='exchanges',multi=True),
                                                html.Div(children = [
                                                html.Div(className='three columns',children = [html.H6('Order Book')]),
                                                html.Div(className='three columns',children = [html.H6('Bps Aggregation Level:')]),
                                                html.Div(className = 'three columns', children =
                                                                                        [dcc.RadioItems(id='agg-level',
                                                                                    options=[{'label':i,'value':i/10000} for i in [0,.1,1,10,100,1000,10000]],value=10/10000,
                                                                                    labelStyle={'display':'inline-block'})])]),
                                                html.Div(id='order-table'),
                                                html.H6('Liquidity Metrics'),
                                                html.H6('----------------------'),
                                                html.P(id='liquidity-table'),
                                                html.H6('Slippage %'),
                                                html.H6('--------------'),
                                                html.P(id='depth-table'),
                                                html.H6('Coin Stats'),
                                                html.H6('--------------'),
                                                html.P(id='stat-table')]),
                                html.Div(id='the-data',style={'display':'none'}),
                                dcc.Interval(
                                    id='interval-component',
                                    interval=10*1000, # in milliseconds= 10 seconds
                                    n_intervals=0
                                    ),
                                dcc.Interval(
                                    id='second',
                                    interval = 1000,
                                    n_intervals=0
                                )     
                        ])
                        ])

@app.callback(Output('time','children'),
            [Input('second','n_intervals'),Input('the-data','children')])
def update_time(n,order_books):
    #return   20* ' '  +  '{}'.format(dt.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
    time_snap = json.loads(order_books)[1]
    return (dt.datetime.now()-dt.datetime.strptime(time_snap,"%Y-%m-%d  %H:%M:%S")).seconds

@app.callback(Output('exchanges','options'),
            [Input('pairs','value')])
def update_exchanges_options(pair):
    return [{'label':exch,'value':exch} for exch in get_exchanges_for_pair(pair).keys()]

@app.callback(Output('exchanges','value'),
            [Input('pairs','value')])
def update_exchanges_values(pair):
    return list(get_exchanges_for_pair(pair).keys())

@app.callback(Output('the-data','children'),
            [Input('pairs','value'),Input('exchanges','value'),Input('interval-component','n_intervals')])
def update_data(pair,ex,n):
    now = dt.datetime.now()
    ex = {x:exch_dict[x] for x in ex}
    order_books = get_order_books(pair,ex)
    save_this = (order_books,now.strftime("%Y-%m-%d  %H:%M:%S"))
    return json.dumps(save_this)

@app.callback(Output('order-book-chart','figure'),
            [Input('the-data','children'),
            Input('pairs','value'),Input('exchanges','value'),
            Input('x-scale','value'),Input('y-scale','value'),
            Input('cutoff','value'),Input('agg-level','value')])
def update_graph(order_books,pair,exchanges,x_scale,y_scale,cutoff,step):
    relative = x_scale == 'Relative'
    currency = y_scale == 'Currency'
    order_books = json.loads(order_books)[0]
    order_books= {key:order_books[key] for key in order_books if key in exchanges}
    return plot_book(order_books,pair,exchanges,relative,currency,cutoff)

@app.callback(Output('market-depth','figure'),
            [Input('the-data','children'),
            Input('pairs','value'),Input('exchanges','value'),
            Input('x-scale','value'),Input('y-scale','value'),
            Input('cutoff','value'),Input('agg-level','value')])
def update_depth(order_books,pair,exchanges,x_scale,y_scale,cutoff,step):
    relative = x_scale == 'Relative'
    currency = y_scale == 'Currency'
    order_books = json.loads(order_books)[0]
    order_books= {key:order_books[key] for key in order_books if key in exchanges}
    return plot_depth(order_books,pair,exchanges,relative,currency,cutoff)

@app.callback(Output('order-table','children'),
            [Input('the-data','children'),
            Input('pairs','value'),Input('exchanges','value'),
            Input('x-scale','value'),Input('y-scale','value'),
            Input('cutoff','value'),Input('agg-level','value')])
def update_table(order_books,pair,exchanges,x_scale,y_scale,cutoff,step):
    order_books = json.loads(order_books)[0]
    order_books= {key:order_books[key] for key in order_books if key in exchanges}
    df =  build_book(order_books,pair,exchanges,cutoff,step)
    df_bids = df[[i for i in df.columns if 'bid' in i]].dropna().iloc[:12]
    df_bids['side'] = 'bid'
    df_bids.columns=['price','size','cum_size','size_$','cum_size_$','average_fill','exc','side']
    df_asks = df[[i for i in df.columns if 'ask' in i]].dropna().iloc[:12]
    df_asks['side'] = 'ask'
    df_asks.columns = ['price','size','cum_size','size_$','cum_size_$','average_fill','exc','side']
    mid=(df_bids['price'].max() + df_asks['price'].min())/2
    df_all=pd.concat([df_asks.sort_values(by='price',ascending=False),df_bids]).rename_axis('from_mid')
    step2 = step if step !=0 else .000001
    rounding = [int(np.ceil(-np.log(mid*step2)/np.log(10)))+1] 
    decimals=pd.Series(rounding+[2]*4+rounding+[2]*2,index=df_all.columns)
    df_all=df_all.round(decimals).reset_index()
    return dash_table.DataTable(
        data=df_all.to_dict('rows'),
        columns=[{'id': c,'name':c} for c in df_all.columns[:-1]] + [{'id':'side','name':'side','hidden':True}],
        style_table={'border': 'thin lightgrey solid'},
        style_header={'backgroundColor':'lightgrey','fontWeight':'bold'},
        style_cell={'textAlign':'center','width':'12%'},
        style_data_conditional=[{
            'if' : {'filter':  'side eq "bid"' },
            'color':'blue'
                    }
            ]+[
            {
            'if' : {'filter': 'side eq "ask"' },
            'color':'red'
        }]+[
            { 'if': {'row_index':'odd'},
            'backgroundColor':'rgb(242,242,242)'}
        ]+[
            {'if':{'column_id':'price'},
            'fontWeight':'bold',
            'border': 'thin lightgrey solid'}
        ],
        style_as_list_view=True
    )

@app.callback(Output('liquidity-table','children'),
            [Input('the-data','children'),
            Input('pairs','value'),Input('exchanges','value'),
            Input('x-scale','value'),Input('y-scale','value'),
            Input('cutoff','value'),Input('agg-level','value')])
def update_liquidity_table(order_books,pair,exchanges,x_scale,y_scale,cutoff,step):
    order_books = json.loads(order_books)[0]
    order_books= {key:order_books[key] for key in order_books if key in exchanges}
    df =  build_book(order_books,pair,exchanges,cutoff,step)
    liq_df = get_liq_params(df,pair)[0].round(4)
    return dash_table.DataTable(
            data=liq_df.to_dict('rows'),
            columns=[{'id': c,'name':c} for c in liq_df.columns],
            style_header={'backgroundColor':'lightgrey','fontWeight':'bold'},
            style_cell={'textAlign':'center','width':'10%'},
            style_table={'border': 'thin lightgrey solid'},
            style_as_list_view=True)

@app.callback(Output('depth-table','children'),
            [Input('the-data','children'),
            Input('pairs','value'),Input('exchanges','value'),
            Input('x-scale','value'),Input('y-scale','value'),
            Input('cutoff','value'),Input('agg-level','value')])
def update_depth_table(order_books,pair,exchanges,x_scale,y_scale,cutoff,step):
    order_books = json.loads(order_books)[0]
    order_books= {key:order_books[key] for key in order_books if key in exchanges}
    df =  build_book(order_books,pair,exchanges,cutoff,step)
    liq_df = get_liq_params(df,pair)[1].round(4)
    return dash_table.DataTable(
            data=liq_df.to_dict('rows'),
            columns=[{'id': c,'name':c} for c in liq_df.columns],
            style_header={'backgroundColor':'lightgrey','fontWeight':'bold'},
            style_cell={'textAlign':'center','width':'10%'},
            style_table={'border': 'thin lightgrey solid'},
            style_as_list_view=True)

@app.callback(Output('stat-table','children'),
            [Input('the-data','children'),
            Input('pairs','value'),Input('exchanges','value'),
            Input('x-scale','value'),Input('y-scale','value'),
            Input('cutoff','value'),Input('agg-level','value')])
def update_stat_table(order_books,pair,exchanges,x_scale,y_scale,cutoff,step):
    order_books = json.loads(order_books)[0]
    order_books= {key:order_books[key] for key in order_books if key in exchanges}
    df =  build_book(order_books,pair,exchanges,cutoff,step)
    liq_df = get_liq_params(df,pair)[2].round(4)
    return dash_table.DataTable(
            data=liq_df.to_dict('rows'),
            columns=[{'id': c,'name':c} for c in liq_df.columns],
            style_header={'backgroundColor':'lightgrey','fontWeight':'bold'},
            style_cell={'textAlign':'center','width':'10%'},
            style_table={'border': 'thin lightgrey solid'},
            style_as_list_view=True)

if __name__ == '__main__':
    app.run_server(debug=True,port = 8055)