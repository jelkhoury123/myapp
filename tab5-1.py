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

# Define my world
spot_exchanges = ['bitstamp','coinbasepro','kraken','liquid','bitfinex','gemini','binance',
                'bittrex','bitbank','huobi','poloniex']
exch_dict={}
for x in spot_exchanges:
    exec('exch_dict[x]=ccxt.{}()'.format(x))

Xpto= ['BTC','ETH','XRP','XMR','BCH','EOS','USDT','USDC','TRX','XLM','BSV','XBT']
Fiat=['USD','EUR','GBP','CHF','HKD','JPY']
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

def get_order_books(pair):
    '''pair is the pair string ,'BTC/USD'...
        returns a dictionary of order books for the pair
        special case for binance which API fails if # of parmaeters > 2
    '''
    ex = get_exchanges_for_pair(pair)
    nobinance= {key:value for key, value in ex.items() if key != 'binance'}
    order_books = {key: value.fetch_order_book(pair,limit=2000,
                        params={'full':1,'level':3,'limit_bids':0,'limit_asks':0,'type':'both'})
                        for key,value in nobinance.items() }
    if 'binance' in ex:
        order_books['binance'] =  ex['binance'].fetch_order_book(pair,limit=1000)
    return order_books

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
    bid_side = pd.DataFrame(order_book['bids'],columns=['bid','bid_size'])
    bid_side['cum_bid_size'] = bid_side['bid_size'].cumsum()
    ask_side = pd.DataFrame(order_book['asks'],columns=['ask','ask_size'])
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
    normalized_bids['average_bid_fill']=normalized_bids['cum_bid_size_$']/normalized_bids['cum_bid_size']
    normalized_asks = pd.DataFrame(ask_side.groupby('ask%',sort=False).mean()['ask'])
    normalized_asks.columns = ['ask']
    normalized_asks['ask_size'] = ask_side.groupby('ask%',sort=False).sum()['ask_size']
    normalized_asks['cum_ask_size'] = normalized_asks['ask_size'].cumsum()
    normalized_asks['ask_size_$'] = ask_side.groupby('ask%',sort=False).sum()['ask_size_$']
    normalized_asks['cum_ask_size_$'] = normalized_asks['ask_size_$'].cumsum()
    normalized_asks['average_ask_fill']=normalized_asks['cum_ask_size_$']/normalized_asks['cum_ask_size']
    book=pd.concat([normalized_asks,normalized_bids],sort=False)
    return book

def aggregate_order_books(dict_of_order_books):
    '''dict_of_order_books is a dict of ccxt like order_books
        retuns a ccxt like dictionary order book sorted by prices 
    '''
    agg_dict_order_book = {}
    agg_dict_order_book['bids'] = (pd.DataFrame(list(itertools.chain(*[order_book['bids'] 
                                    for order_book in dict_of_order_books.values()])))).sort_values(by=0,ascending=False).values.tolist()
    agg_dict_order_book['asks'] = (pd.DataFrame(list(itertools.chain(*[order_book['asks'] 
                                    for order_book in dict_of_order_books.values()])))).sort_values(by=0,ascending=True).values.tolist()
    return agg_dict_order_book

def build_book(pair,exchanges,cutoff=.1,step=0.001):
    ''' gets order books aggreagtes them then normalizes
        returns a dataframe
    '''
    order_books = get_order_books(pair)
    return normalize_order_book(aggregate_order_books({key:order_books[key] for key in exchanges}),cutoff,step)

def plot_book(pair, exc, relative=True, currency=True, cutoff=.1):
    ''' plots the order book as a v shape chart '''
    order_book = build_book(pair,exc,cutoff)
    best_bid = round(order_book['bid'].max(),3)
    best_ask = round(order_book['ask'].min(),3)
    if currency:
        col_to_chart = '_$'
    else:
        col_to_chart = ''
    if relative:
        figure = go.Figure(data=[{'x':order_book.index,'y':order_book['cum_ask_size'+col_to_chart],'name':'asks'}
                      ,{'x':order_book.index,'y':order_book['cum_bid_size'+col_to_chart],'name':'bids'}],
     layout={'title':' / '.join(exc), 'xaxis':{'title':pair +'  ' + str(best_bid)+' / '+ str(best_ask)}})
    else:
        figure= go.Figure(data=[{'x':order_book['ask'].fillna(0)+order_book['bid'].fillna(0),'y':order_book['cum_ask_size'+col_to_chart],'name':'asks'}
                      ,{'x':order_book['ask'].fillna(0)+order_book['bid'].fillna(0),'y':order_book['cum_bid_size'+col_to_chart],'name':'bids'}],
     layout={'title':' / '.join(exc), 'xaxis':{'title':pair +'  ' + str(best_bid)+' / '+ str(best_ask)}})
    return figure

def plot_depth(pair, exc, relative=True, currency=True, cutoff=.1):
    if currency:
        col_to_chart = '_$'
    else:
        col_to_chart = ''
    order_book = build_book(pair,exc,cutoff)
    mid = (order_book['bid'].max()+order_book['ask'].min())/2 if relative else 1
    figure = go.Figure(data=[{'x':order_book['cum_ask_size'+col_to_chart],'y':order_book['average_ask_fill']/mid,'name':'ask depth'},
                            {'x':- order_book['cum_bid_size'+col_to_chart],'y':order_book['average_bid_fill']/mid,'name':'bid depth'}],
    layout={'title': 'Market Depth'})

    return figure

def order_fill(order_book_df, order_sizes,in_ccy=True):
    '''takes in an order book dataframe and an np.array of order sizes
        with size in currecncy by default else in coin
        returns an np.array of the purchase costs or the sale proceeds of an order
    '''
    average_fills = np.zeros(order_sizes.shape)
    mid=(order_book_df['ask'].min()+order_book_df['bid'].max())/2
    print(mid)
    if in_ccy:
        order_sizes=order_sizes/mid
    for i , order_size in enumerate(order_sizes):
        if order_size > 0:
            last_line = order_book_df[order_book_df['cum_ask_size']>order_size].iloc[0]
            ccy_fill = last_line['cum_ask_size_$']+(order_size-last_line['cum_ask_size'])*last_line['ask']
            average_fill=ccy_fill/order_size
        elif order_size < 0:
            last_line=order_book_df[order_book_df['cum_bid_size'] > -order_size].iloc[0]
            ccy_fill=last_line['cum_bid_size_$']+(-order_size-last_line['cum_bid_size'])*last_line['bid']
            average_fill = -ccy_fill/order_size
        average_fills[i]=average_fill
    return average_fills
 

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
                                                    dcc.Graph(id='market-depth')
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
                                                html.Div(id='order-table')])
                        ])
                        ])

@app.callback(Output('exchanges','options'),
            [Input('pairs','value')])
def update_exchanges_options(pair):
    return [{'label':exch,'value':exch} for exch in get_exchanges_for_pair(pair).keys()]

@app.callback(Output('exchanges','value'),
            [Input('pairs','value')])
def update_exchanges_values(pair):
    return list(get_exchanges_for_pair(pair).keys())

@app.callback(Output('order-book-chart','figure'),
            [Input('pairs','value'),Input('exchanges','value'),
            Input('x-scale','value'),Input('y-scale','value'),
            Input('cutoff','value')])
def update_graph(pair,exchanges,x_scale,y_scale,cutoff):
    relative = x_scale == 'Relative'
    currency = y_scale == 'Currency'
    return plot_book(pair,exchanges,relative,currency,cutoff)

@app.callback(Output('market-depth','figure'),
            [Input('pairs','value'),Input('exchanges','value'),
            Input('x-scale','value'),Input('y-scale','value'),
            Input('cutoff','value')])
def update_depth(pair,exchanges,x_scale,y_scale,cutoff):
    relative = x_scale == 'Relative'
    currency = y_scale == 'Currency'
    return plot_depth(pair,exchanges,relative,currency,cutoff)

@app.callback(Output('order-table','children'),
            [Input('pairs','value'),Input('exchanges','value'),
            Input('x-scale','value'),Input('cutoff','value'),Input('agg-level','value')])
def update_table(pair,exchanges,x_scale,cutoff,step):
    df =  build_book(pair,exchanges,cutoff,step).round(2)
    df_bids = df[[i for i in df.columns if 'bid' in i]].dropna().iloc[:10]
    df_bids['side'] = 'bid'
    df_bids.columns=['price','size','cum_size','size_$','cum_size_$','average_fill','side']
    df_asks = df[[i for i in df.columns if 'ask' in i]].dropna().iloc[:10]
    df_asks['side'] = 'ask'
    df_asks.columns = ['price','size','cum_size','size_$','cum_size_$','average_fill','side']
    df_all=pd.concat([df_asks.sort_values(by='price',ascending=False),df_bids]).rename_axis('from_mid').reset_index().round(6)
    return dash_table.DataTable(
        data=df_all.to_dict('rows'),
        columns=[{'id': c,'name':c} for c in df_all.columns[:-1]] + [{'id':'side','name':'side','hidden':True}],
        style_table={'maxHeight': '700','border': 'thin lightgrey solid'},
        style_header={'backgroundColor':'lightgrey','fontWeight':'bold'},
        style_cell={'textAlign':'center','width':'15%'},
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
            'backgroundColor':'rgb(242,242,242'}
        ],
        style_as_list_view=True
    )

if __name__ == '__main__':
    app.run_server(debug=True,port = 8055)