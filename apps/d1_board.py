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
from datetime import datetime as dt
import json
import sys
sys.path.append('..')
import deribit_api3 as dbapi
from app import app  # app is the main app which will be run on the server in index.py

#----------------------------------------------------------------------------------------------#
# APP
#----------------------------------------------------------------------------------------------# 
title = 'Delta One'
#app = dash.Dash(__name__, external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'])

bmx = ccxt.bitmex()
sections = ['BTCUSD', 'ETHUSD', 'ETHBTC', 'ALTBTC']
section_precision = {'BTCUSD':2, 'ETHUSD':2, 'ETHBTC':5, 'ALTBTC':8}

def get_df():

    markets_bmx = pd.DataFrame.from_dict(bmx.load_markets(), orient = 'index')
    ind = markets_bmx[ (markets_bmx['symbol'].str.startswith('.')) & ((markets_bmx['quoteId'] == 'USD') | (markets_bmx['quoteId'] == 'XBT')) & (markets_bmx['symbol'].str.contains('30M') == False) & (markets_bmx['symbol'] != '.XBT')]
    ins = markets_bmx[ (markets_bmx['active'] == True) & (markets_bmx['symbol'].str.contains('_') == False) ]
    bmx_ins = pd.concat([ind,ins], sort = True)
    bmx_ins['exp'] = bmx_ins['info'].apply(lambda x: round(((dt.strptime(x.get('expiry'), '%Y-%m-%dT%H:%M:%S.%fZ') - dt.now()).days*24 + (dt.strptime(x.get('expiry'), '%Y-%m-%dT%H:%M:%S.%fZ') - dt.now()).seconds/3600), 0) if x.get('expiry') != None else 0)
    bmx_ins['pair'] = bmx_ins['base'] + bmx_ins['quote']
    bmx_ins['exchange'] = 'bitmex'
    bmx_ins = bmx_ins[['pair','exp', 'exchange']]

    dbt_btc_ins = pd.DataFrame(dbapi.get_instruments('BTC', 'future')[['instrument_name', 'expiration_timestamp', 'base_currency', 'quote_currency']]).set_index('instrument_name')
    dbt_eth_ins = pd.DataFrame(dbapi.get_instruments('ETH', 'future')[['instrument_name', 'expiration_timestamp', 'base_currency', 'quote_currency']]).set_index('instrument_name')
    dbt_ins = pd.concat([dbt_btc_ins, dbt_eth_ins])
    dbt_ins['pair'] = dbt_ins['base_currency'] + dbt_ins['quote_currency']
    dbt_ins['exchange'] = 'deribit'
    dbt_ins['exp'] = dbt_ins['expiration_timestamp'].apply(lambda x: dt.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S.%f'))
    dbt_ins['exp'] = dbt_ins['exp'].apply(lambda x: 0 if dt.strptime(x, '%Y-%m-%d %H:%M:%S.%f').year == 3000 else round((dt.strptime(x, '%Y-%m-%d %H:%M:%S.%f')-dt.now()).days*24 + (dt.strptime(x, '%Y-%m-%d %H:%M:%S.%f')-dt.now()).seconds/3600, 0))
    dbt_ins = dbt_ins[['pair', 'exp', 'exchange']]
    dbt_ins.loc['.BTC',:] = ['BTCUSD', 0, 'deribit']
    dbt_ins.loc['.ETH',:] = ['ETHUSD', 0, 'deribit']
    dbt_ins.sort_index(inplace = True)

    history = pd.concat([bmx_ins, dbt_ins]).to_json(orient = 'split', date_format='iso')

    return history


#----------------------------------------------------------------------------------------------#
# Layout
#----------------------------------------------------------------------------------------------# 

layout = html.Div(style={'marginLeft':35,'marginRight':35},
    className = 'twelve columns',
    children = [
        # Hidden Data Storage
        html.Div( id = 'd1-hidden_storage', style = {'display': 'none'}, children = get_df() ),
        # Refresh Rate
        dcc.Interval( id = 'd1-refresh_rate', interval = 10*1000, n_intervals = 0, ),

        html.Div( children = [ 
            html.Div( className = 'row', style = {'border-bottom': '1px solid #CB1828', 'margin-bottom': '20px', 'margin-top': '10px'} if i!='ALTBTC' else {'boder-bottom': 'none'}, children = [
                # Table
                html.Div( className = 'one-third column', children = [
                    html.H6( style = { 'font-weight': 'bold', 'font-size': '1.2rem' }, children = [ i ], ),
                    dash_table.DataTable(
                        id = 'd1-'+i+'_table',
                        columns = [ { 'name': 'Instr.', 'id': 'Instr.', },
                                    { 'name': 'Last', 'id': 'Last', 'type': 'numeric', 'format': Format( precision = section_precision[i], scheme=Scheme.fixed ) },
                                    { 'name': 'Prem.', 'id': 'Prem.', 'type': 'numeric', 'format': Format( precision = section_precision[i], scheme=Scheme.fixed, sign=Sign.positive ) },
                                    { 'name': '% Prem.', 'id': '% Prem.', 'type': 'numeric', 'format': FormatTemplate.percentage(2).sign(Sign.positive) },
                                    { 'name': 'An. Prem.', 'id': 'An. Prem.', 'type': 'numeric', 'format': FormatTemplate.percentage(2).sign(Sign.positive) },
                                    { 'name': 'F. Cycles', 'id': 'F. Cycles', 'type': 'numeric', 'format': Format( precision = 2, scheme=Scheme.fixed ) }, 
                                    { 'name': 'Exchange', 'id': 'Exchange', 'hidden': 'True' }, ],
                        style_cell = { 'padding-left': '10px', 'padding-right': '10px', },
                        style_data_conditional = [ { 'if': { 'filter': '"% Prem." > num(0)' }, "color": "#008000", "fontWeight": "bold" }, { 'if': { 'filter': '"% Prem." < num(0)' }, "color": "#B22222", "fontWeight": "bold" }, ], ),
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
# Update Data
@app.callback(
    Output('d1-hidden_storage', 'children'),
    [Input('d1-refresh_rate', 'n_intervals')],
    [State('d1-hidden_storage', 'children')],
)
def process_data(n_intervals, history):

    history = pd.read_json(history, orient = 'split')

    bmx_tickers = bmx.fetch_tickers()
    bmx_last = pd.DataFrame(bmx_tickers)[history[history['exchange'] == 'bitmex'].index].loc['last']
    dbt_btc = dbapi.get_summary_by_currency('BTC', 'future')
    dbt_eth = dbapi.get_summary_by_currency('ETH', 'future')
    dbt_tickers = pd.concat([dbt_btc, dbt_eth])[['instrument_name', 'last', 'estimated_delivery_price']].set_index('instrument_name')
    dbt_tickers.loc['.BTC',:] = dbt_tickers.loc['BTC-PERPETUAL', 'estimated_delivery_price']
    dbt_tickers.loc['.ETH',:] = dbt_tickers.loc['ETH-PERPETUAL', 'estimated_delivery_price']
    last = pd.DataFrame(pd.concat([bmx_last, dbt_tickers.sort_index()['last']]))
    history[dt.now()] = last

    #print(dt.now())

    dump_history = history.to_json(date_format = 'iso', orient = 'split')

    return dump_history


# Get Tables
@app.callback(
    [Output('d1-BTCUSD_table', 'data'), Output('d1-ETHUSD_table', 'data'), Output('d1-ETHBTC_table', 'data'), Output('d1-ALTBTC_table', 'data')],
    [Input('d1-refresh_rate', 'n_intervals')],
    [State('d1-hidden_storage','children')]
)
def show_table(n_intervals, history):

    history = pd.read_json(history, orient = 'split')
    data = []
    alt_data = pd.DataFrame([])
    
    for pair in ['BTCUSD', 'ETHUSD', 'ETHBTC', 'BCHBTC', 'LTCBTC', 'XRPBTC', 'TRXBTC', 'EOSBTC', 'ADABTC']:

        table = pd.DataFrame(history[history['pair'] == pair].iloc[:,[1,2,len(history.columns)-1]])
        table.sort_index(inplace = True)
        exp = table['exp']
        last = table.filter(like='.', axis=0).iloc[:,1:]
        table['Prem.'] = table.apply(lambda row: (row[-1]-last[last['exchange']=='bitmex'].iloc[0,1]) if row[1] == 'bitmex' else row[-1]-last[last['exchange']=='deribit'].iloc[0,1], axis = 1)
        table['Prem.'] = table['Prem.'].apply(lambda x: None if x==0 else x)
        table['% Prem.'] = table.apply(lambda row: (row['Prem.']/last[last['exchange']=='bitmex'].iloc[0,1]) if row[1] == 'bitmex' else (row['Prem.']/last[last['exchange']=='bitmex'].iloc[0,1]), axis = 1)
        table['An. Prem.'] = table.loc[:,'% Prem.'] * (8760/exp)
        table['F. Cycles'] = table['exp'].apply(lambda x: x/8 if x!=0 else None)
        table['ex'] = table['exchange']
        table.sort_values(by=['exchange', 'exp'], inplace = True, ascending = [True,True])
        table = table.iloc[:,2:]
        if pair in ['BTCUSD', 'ETHUSD', 'ETHBTC']:
            table = table.reset_index()
            table.columns = ['Instr.', 'Last', 'Prem.', '% Prem.', 'An. Prem.', 'F. Cycles', 'Exchange']
            data.append(table.to_dict('rows'))
        else:
            table = pd.DataFrame(table)
            alt_data = pd.concat([alt_data, table])
    alt_data = alt_data.reset_index()
    alt_data.columns = ['Instr.', 'Last', 'Prem.', '% Prem.', 'An. Prem.', 'F. Cycles', 'Exchange']
    data.append(alt_data.to_dict('rows'))

    return data


# BTCUSD
@app.callback(
    [Output('d1-BTCUSD_last', 'figure'), Output('d1-BTCUSD_prem', 'figure') ],
    [Input('d1-BTCUSD_table', 'active_cell'), Input('d1-BTCUSD_table', 'data'), Input('d1-refresh_rate', 'n_intervals'), ],
    [State('d1-hidden_storage','children')],
)
def get_btcusd(cell, tab_data, n_intervals, history):
    history = pd.read_json(history, orient = 'split')
    data = history[history['pair']=='BTCUSD']
    chart = get_charts(data, cell, 'BTCUSD', tab_data)
    return chart


# ETHUSD
@app.callback(
    [Output('d1-ETHUSD_last', 'figure'), Output('d1-ETHUSD_prem', 'figure'),],
    [Input('d1-ETHUSD_table', 'active_cell'), Input('d1-ETHUSD_table', 'data'), Input('d1-refresh_rate', 'n_intervals'), ],
    [State('d1-hidden_storage','children'),],
)
def get_ethusd(cell, tab_data, n_intervals, history):
    history = pd.read_json(history, orient = 'split')
    data = history[history['pair']=='ETHUSD']
    chart = get_charts(data, cell, 'ETHUSD', tab_data)
    return chart


# ETHXBT
@app.callback(
    [Output('d1-ETHBTC_last', 'figure'), Output('d1-ETHBTC_prem', 'figure'),],
    [Input('d1-ETHBTC_table', 'active_cell'), Input('d1-ETHBTC_table', 'data'), Input('d1-refresh_rate', 'n_intervals'), ],
    [State('d1-hidden_storage','children'),],
)
def get_ethxbt(cell, tab_data, n_intervals, history):
    history = pd.read_json(history, orient = 'split')
    data = history[history['pair']=='ETHBTC']
    chart = get_charts(data, cell, 'ETHBTC', tab_data)
    return chart


# ALTXXX
@app.callback(
    [Output('d1-ALTBTC_last', 'figure'), Output('d1-ALTBTC_prem', 'figure'),],
    [Input('d1-ALTBTC_table', 'active_cell'), Input('d1-ALTBTC_table', 'data'), Input('d1-refresh_rate', 'n_intervals'), ],
    [State('d1-hidden_storage','children'),],
)
def get_altxxx(cell, tab_data, n_intervals, history):
    history = pd.read_json(history, orient = 'split')
    cell_row = 0 if cell==None else cell[0]
    pairs = ['BCHBTC', 'BCHBTC', 'LTCBTC', 'LTCBTC', 'XRPBTC', 'XRPBTC', 'TRXBTC', 'TRXBTC', 'EOSBTC', 'EOSBTC', 'ADABTC', 'ADABTC']
    data = history[history['pair']==pairs[cell_row]]
    chart = get_charts(data, cell, pairs[cell_row], tab_data)
    return chart


def get_charts(data, cell, pair, tab_data = None):
    cell_col = 0 if cell==None else cell[1]
    cell_row = None if cell==None else cell[0]
    tab_data = pd.DataFrame(tab_data) if tab_data != None else tab_data
    row_data = ['bitmex', 'deribit'] if cell_row == None else [tab_data['Exchange'][cell_row]]
    row_data = ['bitmex', 'deribit'] if cell_row == 0 else row_data
    title = ('Bitmex - '+pair+' - Premium' if row_data==['bitmex'] else ( 'Deribit - '+pair+' - Premium' if row_data==['deribit'] else 'Bitmex, Deribit - '+pair+' - Premium')) if pair == 'BTCUSD' or pair == 'ETHUSD' else pair+' - Premium'
    prem_traces = []
    last_traces = []

    groups = data.groupby('exchange')

    for name, group in groups:

        data = group.T
        exp = data.loc['exp',:]
        exp = exp.apply(lambda x: None if x==0 else x)
        data = data.iloc[3:,:].tail(500) if len(data.index[3:]) > 100 else data.iloc[3:,:]
        time = list(data.index)
        last = list(data.loc[:, data.columns[0]])
        last_traces.append( go.Scatter( x = time, y = last, name = data.columns[0], line = dict(width = 1), mode = 'lines', ) )

        prem_ins = [i for i in data.columns if ('.' not in i and '/' not in i and 'PERPETUAL' not in i) ] if cell_col==4 else [i for i in data.columns if '.' not in i ]

        if name in row_data:

            for i in prem_ins:
                if cell_col==4:
                    prem = list(((data[i] - data.iloc[:,0]) / data.iloc[:,0]) * 100 * (8760/exp[i]))
                    title = ('Bitmex - '+pair+' - Annualised % Premium' if row_data==['bitmex'] else ( 'Deribit - '+pair+' - Annualised % Premium' if row_data==['deribit'] else 'Bitmex, Deribit - '+pair+' - Annualised % Premium')) if pair == 'BTCUSD' or pair == 'ETHUSD' else pair+' - Annualised % Premium'
                    prem_traces.append( go.Scatter( x = time, y = prem, name = i, line = dict(width = 1), mode = 'lines',) )
                elif cell_col== 3:
                    prem = list(((data[i] - data.iloc[:,0]) / data.iloc[:,0]) * 100)
                    title = ('Bitmex - '+pair+' - % Premium' if row_data==['bitmex'] else ( 'Deribit - '+pair+' - % Premium' if row_data==['deribit'] else 'Bitmex, Deribit - '+pair+' - % Premium')) if pair == 'BTCUSD' or pair == 'ETHUSD' else pair+' - % Premium'
                    prem_traces.append( go.Scatter( x = time, y = prem, name = i, line = dict(width = 1), mode = 'lines',) )
                else:
                    prem = list(data[i] - data.iloc[:,0])
                    prem_traces.append( go.Scatter( x = time, y = prem, name = i, line = dict(width = 1), mode = 'lines',) )

    return ({'data': last_traces,
             'layout': { 'margin': {'t': 30, 'b': 50, 'l': 70, 'r': 30, }, 'height': 300, 'width': 600, 'title': pair+' - Last Price', 'legend': dict(orientation='h'), 'xaxis': dict(showline = True), 'yaxis': dict(showline = True) ,'uirevision':str(cell)} },
            {'data': prem_traces,
             'layout': { 'margin': {'t': 30, 'b': 50, 'l': 50, 'r': 30, }, 'height': 300, 'width': 600, 'title': title, 'legend': dict(orientation='h'), 'xaxis': dict(showline = True), 'yaxis': dict(showline = True) ,'uirevision':str(cell)} }, )


#----------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    app.run_server(debug = True, port = 5060)