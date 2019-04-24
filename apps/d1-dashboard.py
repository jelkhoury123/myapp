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

def get_d1_instruments():
    '''
    returns a dictionary with keys deribit and bitmex
    each of whic is in turn a dictionary with keys: BTC ETH and values: list containing Futures and Swaps traded on base 
    and not expired
    '''
    all_ins={}
    for exc,exc_obj in exch_dict.items():
        all_ins[exc]={}
        for base in Xpto_base:
            base_list=[]
            for ins in getattr(exc_obj,'markets'):
                market = getattr(exc_obj,'markets')[ins]
                if market['type'] in ('future','swap') and market['base']==base and not ins.startswith('.') and '_' not in ins:
                    if exc == 'bitmex':
                        expiry = market['info']['expiry']
                        if expiry is None:
                            base_list.append(ins)
                        else:
                            dt_expiry = dt.datetime.strptime(expiry,"%Y-%m-%dT%H:%M:%S.%fZ")
                            if dt_expiry > dt.datetime.now():
                                base_list.append(ins)
                    else:
                        base_list.append(ins)
            all_ins[exc][base]=base_list
    
    return all_ins

instruments = get_d1_instruments()
deribit_d1_ins , bitmex_d1_ins = instruments['deribit'], instruments['bitmex']

def get_tickers(base):
    tickers_dict={}
    for key in instruments:
        ins_list = instruments[key][base]
        tickers_dict[key]={i:exch_dict[key].fetch_ticker(i)['last'] for i in ins_list}
    return tickers_dict

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = app = dash.Dash(__name__,external_stylesheets=external_stylesheets)

app.layout= html.Div([
    dcc.Graph(id = 'live',animate = True),
    dcc.Store(id='memory'),
    dcc.Interval(id='update',interval = 5 * 1000,n_intervals = 0)
])

@app.callback(Output('memory','data'),
    [Input('update','n_intervals')],
    [State('memory' , 'data')])
def update_data(n_intervals,data):
    new_tick = get_tickers('BTC')['deribit']['BTC-PERPETUAL']
    data = json.loads(data)
    data.append(new_tick)
    return json.dumps(data)

@app.callback(Output('live','figure'),
[Input('update','n_intervals')],
[State('memory','data')])
def update_graph(n_intervals,data):
    data = json.loads(data)
    return go.Figure(data=data)

if __name__ =='__main__':
    app.run_server(debug=True,port = 5088 )


