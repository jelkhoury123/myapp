#----------------------------------------------------------------------------------------------#
# IMPORTS
#----------------------------------------------------------------------------------------------#
# Dash Imports
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
from dash_table.Format import Format, Scheme, Sign, Symbol
import dash_table.FormatTemplate as FormatTemplate

# Plotly Imports
import plotly
import plotly.graph_objs as go

# Other Imports
import ccxt
import time
import pandas as pd
import numpy as np
import datetime as dt
import json
import sys
import pymongo
sys.path.append('..')
import deribit_api3 as my_deribit
from app import app  # app is the main app which will be run on the server in index.py


#----------------------------------------------------------------------------------------------#
# ENVIRONMENT & FUNCTIONS
#----------------------------------------------------------------------------------------------# 
title = 'Delta One'
#app = dash.Dash(__name__, external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'])

refresh_rate = 5

sections = ['BTCUSD', 'ETHUSD', 'ETHBTC', 'ALTBTC']
section_precision = {'BTCUSD':2, 'ETHUSD':2, 'ETHBTC':5, 'ALTBTC':8}
sym_ref = { 'BTCUSD':['BTC','XBT'],
            'ETHUSD':['ETH'],
            'ETHBTC':['ETHXBT'],
            'ALTBTC':['ADAXBT','BCHXBT','EOSXBT','LTCXBT','TRXXBT','XRPXBT']}

bmx = ccxt.bitmex()
client = pymongo.MongoClient("mongodb+srv://noman:1234@cluster0-insmh.mongodb.net/test?retryWrites=true&w=majority")
db = client.d1
doc = db.d1

# Load database data
def load_db():
    t = (dt.datetime.utcnow() - dt.timedelta(hours = 6)).strftime('%Y-%m-%d %H:%M:%S')
    doc.create_index([('timestamp',pymongo.ASCENDING)])
    cursor = doc.find({'timestamp':{'$gte':t}})
    data_db = pd.DataFrame(cursor)
    data = []
    for name,group in data_db.groupby('symbol', sort=False):
        data.append(group[['symbol','unsym','mid','prem','pprem','anprem','timestamp']].set_index('timestamp'))
    return json.dumps([i.to_dict() for i in data])

# Get new data
def get_data():
    bmx_ticks = pd.DataFrame([ins['info'] for ins in bmx.fetch_tickers().values()])
    bmx_ticks = pd.DataFrame(bmx_ticks[(bmx_ticks['symbol'].str.contains('30M') == False) & (bmx_ticks['symbol'].str.contains('_') == False)
                                    & ((bmx_ticks['quoteCurrency'] == 'USD') | (bmx_ticks['quoteCurrency'] == 'XBT'))
                                    & (bmx_ticks['symbol'] != '.XBT')])
    bmx_ticks['mid'] = bmx_ticks['midPrice'].fillna(bmx_ticks['lastPrice'])
    bmx_ticks['exp'] = bmx_ticks['expiry'].apply(lambda x: round(((dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ') - dt.datetime.now()).days*24 + (dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ') - dt.datetime.now()).seconds/3600), 0) if x != None else 0)
    bmx_ticks = bmx_ticks.sort_values(['underlyingSymbol','symbol'])[['symbol','mid','underlyingSymbol','exp']]
    bmx_ticks['unsym'] = bmx_ticks['underlyingSymbol'].str.strip('=')
    bmx_ticks['symbol'].replace(['XBTUSD','ETHUSD'],['BTC/USD','ETH/USD'], inplace=True)
    bmx_ticks['ex'] = 'bmx'
    bmx_ticks['symbol'] = bmx_ticks['symbol']+'_bmx'

    dbt_btc = my_deribit.get_summary_by_currency('BTC', 'future')
    dbt_eth = my_deribit.get_summary_by_currency('ETH', 'future')
    dbt_ticks = pd.concat([dbt_btc, dbt_eth], ignore_index=True)[['instrument_name', 'ask_price', 'bid_price', 'estimated_delivery_price', 'base_currency']]
    dbt_ticks['mid'] = (dbt_ticks['ask_price'] + dbt_ticks['ask_price'])/2
    dbt_ticks['exp'] = dbt_ticks['instrument_name'].apply(lambda x: x.split('-')[1]+' 08:00:00' if x.split('-')[1]!='PERPETUAL' else np.nan)
    dbt_ticks['exp'] = dbt_ticks['exp'].apply(lambda x: round(((dt.datetime.strptime(str(x), '%d%b%y %H:%M:%S') - dt.datetime.now()).days*24 + (dt.datetime.strptime(str(x), '%d%b%y %H:%M:%S') - dt.datetime.now()).seconds/3600), 0) if x is not np.nan else 0)
    dbt_ticks = dbt_ticks.append({'instrument_name':'.BTC', 'base_currency':'BTC','exp': 0, 'mid':dbt_ticks['estimated_delivery_price'][dbt_ticks['base_currency']=='BTC'].unique()[0]}, ignore_index=True)
    dbt_ticks = dbt_ticks.append({'instrument_name':'.ETH', 'base_currency':'ETH','exp': 0, 'mid':dbt_ticks['estimated_delivery_price'][dbt_ticks['base_currency']=='ETH'].unique()[0]}, ignore_index=True)
    dbt_ticks = dbt_ticks.rename(columns = {'base_currency':'unsym', 'instrument_name':'symbol'})
    dbt_ticks = dbt_ticks[['symbol','mid','unsym','exp']].sort_values(['unsym','symbol'])
    dbt_ticks['ex'] = 'dbt'
    dbt_ticks['symbol'] = dbt_ticks['symbol']+'_dbt'

    ticks = pd.concat([bmx_ticks,dbt_ticks],sort=False)
    ticks['prem'] = ticks['mid'].groupby([ticks['ex'],ticks['unsym']]).apply(lambda x: x.apply(lambda y: y-x.iloc[0]))
    ticks['pprem'] = ticks['mid'].groupby([ticks['ex'],ticks['unsym']]).apply(lambda x: x.apply(lambda y: (y-x.iloc[0])/x.iloc[0]))
    ticks['anprem'] = ticks['pprem'].div(ticks['exp'])*8760
    ticks['anprem'] = ticks['anprem'].replace([np.inf, -np.inf, np.nan], 0)
    ticks.sort_values(['ex','unsym','exp'],inplace = True)
    ticks = ticks[['symbol', 'unsym', 'mid', 'prem', 'pprem', 'anprem']]

    ticks['timestamp'] = dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    #ticks[['pprem','anprem']].replace(0, np.nan, inplace=True)
    ticks.set_index('timestamp', inplace=True)

    return ticks


#----------------------------------------------------------------------------------------------#
# Layout
#----------------------------------------------------------------------------------------------# 

layout = html.Div(style={'paddingLeft':35,'paddingRight':35},
    #className = 'twelve columns',
    children = [
        # Hidden Data Storage
        html.Div( id = 'd1-hidden_storage', style = {'display': 'none'}, children = load_db() ),
        # Refresh Rate
        dcc.Interval( id = 'd1-refresh_rate', interval = refresh_rate*1000, n_intervals = 0, ),

        html.Div( children = [ 
            html.Div( className = 'row', style = {'border-bottom': '1px solid #CB1828', 'margin-bottom': '20px', 'margin-top': '10px'} if i!='ALTBTC' else {'boder-bottom': 'none'}, children = [
                # Table
                html.Div( className = 'one-third column', children = [
                    html.H6( style = { 'font-weight': 'bold', 'font-size': '1.2rem' }, children = [ i ], ),
                    dash_table.DataTable(
                        id = 'd1-'+i+'_table',
                        columns = [ { 'name': 'Instr', 'id': 'symbol', },
                                    { 'name': 'Mid', 'id': 'mid', 'type': 'numeric', 'format': Format( precision = section_precision[i], scheme=Scheme.fixed ) },
                                    { 'name': 'Prem', 'id': 'prem', 'type': 'numeric', 'format': Format( precision = section_precision[i], scheme=Scheme.fixed, sign=Sign.positive ) },
                                    { 'name': '% Prem', 'id': 'pprem', 'type': 'numeric', 'format': FormatTemplate.percentage(2).sign(Sign.positive) },
                                    { 'name': 'An Prem', 'id': 'anprem', 'type': 'numeric', 'format': FormatTemplate.percentage(2).sign(Sign.positive) },
                                    { 'name': 'F Cycles', 'id': 'fcycles', 'type': 'numeric', 'format': Format( precision = 2, scheme=Scheme.fixed ) } ],
                        #style_header={'backgroundColor':'#373737', 'font-size':'bold'}, #17191b
                        style_header={'font-size':'bold'},
                        #style_cell = { 'padding-left': '10px', 'padding-right': '10px', 'backgroundColor':'#2c2c2c', 'color':'white', 'border':'1px solid #565656'}, #272c30
                        style_cell = { 'padding-left': '10px', 'padding-right': '10px'},
                        style_data_conditional = [ { 'if': { 'filter': '{pprem} > 0' }, "color": "#008000", "fontWeight": "bold" }, { 'if': { 'filter': '{pprem} < 0' }, "color": "#B22222", "fontWeight": "bold" },]
                        ),
                    html.P(id = 'd1-'+i+'-last-timestamp', style = {'display':'inline-block','font-size':'1.2rem'}),
                    html.P(children=' / ', style = {'display':'inline-block', 'margin':'0px 5px','font-size':'1.2rem'}),
                    html.P(id = 'd1-'+i+'-new-timestamp', style = {'display':'inline-block','font-size':'1.2rem'}, children = dt.datetime.now().strftime('%X')),
                    ], ),
                # Chart - Last Price
                html.Div( className = 'one-third column', style = {'margin-top': '10px', 'margin-right': '0px', 'margin-left': '50px'}, children = [
                    dcc.Graph( id = 'd1-'+i+'_last', figure = {}, ),
                    ], ),
                # Chart - Premium
                html.Div( className = 'one-third column', style = {'margin-top': '10px', 'margin-right': '0px', 'margin-left': '0px'}, children = [
                    dcc.Graph( id ='d1-'+ i+'_prem', figure = {},),
                    ], ),
                ],)
        for i in sections],)
    ],)


#----------------------------------------------------------------------------------------------#
# Callbacks
#----------------------------------------------------------------------------------------------#
@app.callback(
    Output('d1-hidden_storage', 'children'),
    [Input('d1-refresh_rate', 'n_intervals')],
    [State('d1-hidden_storage', 'children')],
)
def update_hidden_div(n_intervals,data):
    data = [pd.DataFrame(i) for i in json.loads(data)]
    ticks = get_data()
    data = [data[i].append(ticks.iloc[i]) for i in range(len(ticks))]
    return json.dumps([i.to_dict() for i in data])


@app.callback(
    [Output('d1-BTCUSD_table', 'data'), Output('d1-ETHUSD_table', 'data'), Output('d1-ETHBTC_table', 'data'), Output('d1-ALTBTC_table', 'data'),
     Output('d1-BTCUSD-last-timestamp', 'children'), Output('d1-ETHUSD-last-timestamp', 'children'), Output('d1-ETHBTC-last-timestamp', 'children'), Output('d1-ALTBTC-last-timestamp', 'children'),
     Output('d1-BTCUSD-new-timestamp', 'children'), Output('d1-ETHUSD-new-timestamp', 'children'), Output('d1-ETHBTC-new-timestamp', 'children'), Output('d1-ALTBTC-new-timestamp', 'children'),],
    [Input('d1-refresh_rate', 'n_intervals')],
    [State('d1-hidden_storage','children'), State('d1-BTCUSD-new-timestamp', 'children'), State('d1-ETHUSD-new-timestamp', 'children'), State('d1-ETHBTC-new-timestamp', 'children'), State('d1-ALTBTC-new-timestamp', 'children')]
)
def update_table(n_intervals, data, BTCUSD_last_timestamp, ETHUSD_last_timestamp, ETHBTC_last_timestamp, ALTBTC_last_timestamp):
    data = [pd.DataFrame(i) for i in json.loads(data)]
    tab_data = []
    new_timestamps = []
    last_timestamps = [BTCUSD_last_timestamp, ETHUSD_last_timestamp, ETHBTC_last_timestamp, ALTBTC_last_timestamp]

    for pair in sym_ref:
        pair_data = pd.DataFrame([data[i][['symbol','mid','prem','pprem','anprem']].iloc[-1] for i in range(len(data)) if data[i]['unsym'].iloc[-1] in sym_ref[pair]])
        pair_data['symbol'] = pair_data['symbol'].str.split('_').str[0]
        pair_data['fcycles'] = pair_data['pprem']/pair_data['anprem'] * 8760/8
        pair_data.replace(0, np.nan, inplace = True)
        tab_data.append(pair_data.to_dict('rows'))
        new_timestamps.append(dt.datetime.now().strftime('%X'))

    return tab_data[0],tab_data[1],tab_data[2],tab_data[3],last_timestamps[0],last_timestamps[1],last_timestamps[2],last_timestamps[3],new_timestamps[0],new_timestamps[1],new_timestamps[2],new_timestamps[3]


# BTCUSD
@app.callback(
    [Output('d1-BTCUSD_last', 'figure'), Output('d1-BTCUSD_prem', 'figure'),
     Output('d1-ETHUSD_last', 'figure'), Output('d1-ETHUSD_prem', 'figure'),
     Output('d1-ETHBTC_last', 'figure'), Output('d1-ETHBTC_prem', 'figure'),
     Output('d1-ALTBTC_last', 'figure'), Output('d1-ALTBTC_prem', 'figure') ],
    [Input('d1-BTCUSD_table', 'active_cell'), Input('d1-ETHUSD_table', 'active_cell'), Input('d1-ETHBTC_table', 'active_cell'), Input('d1-ALTBTC_table', 'active_cell'),
     Input('d1-BTCUSD_table', 'data'), Input('d1-refresh_rate', 'n_intervals'), ],
    [State('d1-hidden_storage','children')],
)
def get_btcusd(btcusd_cell, ethusd_cell, ethbtc_cell, altbtc_cell, tab_data, n_intervals, data):
    data = [pd.DataFrame(i) for i in json.loads(data)]
    time = data[0].index
    title = {'prem': 'Premium', 'pprem': 'Percentage Premium', 'anprem': 'Anualized Premium'}
    chart_data, instruments, ins, ex = [], [], [], []
    
    btcusd_cell = [0,'prem'] if btcusd_cell==None else ([btcusd_cell['row'],'prem'] if btcusd_cell['column'] in (0,1,2,5) else [btcusd_cell['row'],btcusd_cell['column_id']])
    ethusd_cell = [0,'prem'] if ethusd_cell==None else ([ethusd_cell['row'],'prem'] if ethusd_cell['column'] in (0,1,2,5) else [ethusd_cell['row'],ethusd_cell['column_id']])
    ethbtc_cell = [0,'prem'] if ethbtc_cell==None else ([ethbtc_cell['row'],'prem'] if ethbtc_cell['column'] in (0,1,2,5) else [ethbtc_cell['row'],ethbtc_cell['column_id']])
    altbtc_cell = [0,'prem'] if altbtc_cell==None else ([altbtc_cell['row'],'prem'] if altbtc_cell['column'] in (0,1,2,5) else [altbtc_cell['row'],altbtc_cell['column_id']])

    for pair in sym_ref:
        pair_data = [data[i] for i in range(len(data)) if data[i]['unsym'].iloc[-1] in sym_ref[pair]]
        pair_instruments = [pair_data[i]['symbol'][0] for i in range(len(pair_data))]
        pair_ins, pair_ex = [i.split('_')[0] for i in pair_instruments],[i.split('_')[1] for i in pair_instruments]
        instruments.append(pair_instruments)
        ins.append(pair_ins)
        ex.append(pair_ex)
        chart_data.append(pair_data)

    btcusd_show = [ex[0][btcusd_cell[0]]] if btcusd_cell[0]!=0 else list(set(ex[0]))
    ethusd_show = [ex[1][ethusd_cell[0]]] if ethusd_cell[0]!=0 else list(set(ex[1]))
    ethbtc_show = [ex[2][ethbtc_cell[0]]] if ethbtc_cell[0]!=0 else list(set(ex[2]))
    sym = ins[3][altbtc_cell[0]].strip('.B')[:3] if not ins[3][altbtc_cell[0]][:3] in ('.BB','BCH') else 'BCH'
    altbtc_show = [i for i in ins[3] if sym in i]

    btcusd_idx = [dict(x = time, y = chart_data[0][i]['mid'], mode = 'lines', name = instruments[0][i].split('_')[0], line = {'width':1}) for i in range(len(chart_data[0])) if instruments[0][i].startswith('.')]
    ethusd_idx = [dict(x = time, y = chart_data[1][i]['mid'], mode = 'lines', name = instruments[1][i].split('_')[0], line = {'width':1}) for i in range(len(chart_data[1])) if instruments[1][i].startswith('.')]
    ethbtc_idx = [dict(x = time, y = chart_data[2][i]['mid'], mode = 'lines', name = instruments[2][i].split('_')[0], line = {'width':1}) for i in range(len(chart_data[2])) if instruments[2][i].startswith('.')]
    altbtc_idx = [dict(x = time, y = chart_data[3][i]['mid'], mode = 'lines', name = instruments[3][i].split('_')[0], line = {'width':1}) for i in range(len(chart_data[3])) if instruments[3][i].startswith('.') and any(instruments[3][i].split('_')[j] in altbtc_show for j in range(2))]

    btcusd_mid = [dict(x = time, y = chart_data[0][i][btcusd_cell[1]] if [btcusd_cell[1]]==['prem'] else chart_data[0][i][btcusd_cell[1]]*100, mode = 'lines', name = instruments[0][i].split('_')[0], line = {'width':1}) for i in range(len(chart_data[0])) if not instruments[0][i].startswith('.') and not (chart_data[0][i][btcusd_cell[1]] == 0).all() and any(instruments[0][i].split('_')[j] in btcusd_show for j in range(2))] #and not chart_data[0][i][btcusd_cell[1]].isnull().any() 
    ethusd_mid = [dict(x = time, y = chart_data[1][i][ethusd_cell[1]] if [btcusd_cell[1]]==['prem'] else chart_data[1][i][ethusd_cell[1]]*100, mode = 'lines', name = instruments[1][i].split('_')[0], line = {'width':1}) for i in range(len(chart_data[1])) if not instruments[1][i].startswith('.') and not (chart_data[1][i][ethusd_cell[1]] == 0).all() and any(instruments[1][i].split('_')[j] in ethusd_show for j in range(2))] #and not chart_data[1][i][ethusd_cell[1]].isnull().any()
    ethbtc_mid = [dict(x = time, y = chart_data[2][i][ethbtc_cell[1]] if [btcusd_cell[1]]==['prem'] else chart_data[2][i][ethbtc_cell[1]]*100, mode = 'lines', name = instruments[2][i].split('_')[0], line = {'width':1}) for i in range(len(chart_data[2])) if not instruments[2][i].startswith('.') and not (chart_data[2][i][ethbtc_cell[1]] == 0).all() and any(instruments[2][i].split('_')[j] in ethbtc_show for j in range(2))] #and not chart_data[2][i][ethbtc_cell[1]].isnull().any() 
    altbtc_mid = [dict(x = time, y = chart_data[3][i][altbtc_cell[1]] if [btcusd_cell[1]]==['prem'] else chart_data[3][i][altbtc_cell[1]]*100, mode = 'lines', name = instruments[3][i].split('_')[0], line = {'width':1}) for i in range(len(chart_data[3])) if not instruments[3][i].startswith('.') and not (chart_data[3][i][altbtc_cell[1]] == 0).all() and any(instruments[3][i].split('_')[j] in altbtc_show for j in range(2))] #and not chart_data[3][i][altbtc_cell[1]].isnull().any()

    return ({'data': btcusd_idx,
             'layout': { 'margin': {'t': 30, 'b': 50, 'l': 70, 'r': 30, }, 'height': 300, 'width': 600, 'title': 'Last Price', 'legend': dict(orientation='h'), 'showlegend': True, 'xaxis': dict(showline = True), 'yaxis': dict(showline = True) ,'uirevision':str(btcusd_cell)} }, #'font': {"color": "#ffffff"}, 'paper_bgcolor':"#2c2c2c", 'plot_bgcolor':"#2c2c2c"
            {'data': btcusd_mid,
             'layout': { 'margin': {'t': 30, 'b': 50, 'l': 50, 'r': 30, }, 'height': 300, 'width': 600, 'title': title[btcusd_cell[1]], 'legend': dict(orientation='h'), 'showlegend': True, 'xaxis': dict(showline = True), 'yaxis': dict(showline = True) ,'uirevision':str(btcusd_cell)} },
            {'data': ethusd_idx,
             'layout': { 'margin': {'t': 30, 'b': 50, 'l': 70, 'r': 30, }, 'height': 300, 'width': 600, 'title': 'Last Price', 'legend': dict(orientation='h'), 'showlegend': True, 'xaxis': dict(showline = True), 'yaxis': dict(showline = True) ,'uirevision':str(ethusd_cell)} },
            {'data': ethusd_mid,
             'layout': { 'margin': {'t': 30, 'b': 50, 'l': 50, 'r': 30, }, 'height': 300, 'width': 600, 'title': title[ethusd_cell[1]], 'legend': dict(orientation='h'), 'showlegend': True, 'xaxis': dict(showline = True), 'yaxis': dict(showline = True) ,'uirevision':str(ethusd_cell)} },
            {'data': ethbtc_idx,
             'layout': { 'margin': {'t': 30, 'b': 50, 'l': 70, 'r': 30, }, 'height': 300, 'width': 600, 'title': 'Last Price', 'legend': dict(orientation='h'), 'showlegend': True, 'xaxis': dict(showline = True), 'yaxis': dict(showline = True) ,'uirevision':str(ethbtc_cell)} },
            {'data': ethbtc_mid,
             'layout': { 'margin': {'t': 30, 'b': 50, 'l': 50, 'r': 30, }, 'height': 300, 'width': 600, 'title': title[ethbtc_cell[1]], 'legend': dict(orientation='h'), 'showlegend': True, 'xaxis': dict(showline = True), 'yaxis': dict(showline = True) ,'uirevision':str(ethbtc_cell)} },
            {'data': altbtc_idx,
             'layout': { 'margin': {'t': 30, 'b': 50, 'l': 70, 'r': 30, }, 'height': 300, 'width': 600, 'title': 'Last Price', 'legend': dict(orientation='h'), 'showlegend': True, 'xaxis': dict(showline = True), 'yaxis': dict(showline = True) ,'uirevision':str(altbtc_cell)} },
            {'data': altbtc_mid,
             'layout': { 'margin': {'t': 30, 'b': 50, 'l': 50, 'r': 30, }, 'height': 300, 'width': 600, 'title': title[altbtc_cell[1]], 'legend': dict(orientation='h'), 'showlegend': True, 'xaxis': dict(showline = True), 'yaxis': dict(showline = True) ,'uirevision':str(altbtc_cell)} },)



#----------------------------------------------------------------------------------------------#
# Run
#----------------------------------------------------------------------------------------------#
# if __name__ == '__main__':
#     app.run_server(debug = True, port = 8060)