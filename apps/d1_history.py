#----------------------------------------------------------------------------------------------#
# IMPORTS & One Timers
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
import time
import pandas as pd
import numpy as np
import datetime as dt
import json
import pymongo
from app import app  # app is the main app which will be run on the server in index.py

# Database
client = pymongo.MongoClient("mongodb+srv://noman:1234@cluster0-insmh.mongodb.net/test?retryWrites=true&w=majority")
db = client.d1
doc = db.d1


#----------------------------------------------------------------------------------------------#
# Get Indices History From the Database
#----------------------------------------------------------------------------------------------# 
sym_ref = { 'BTCUSD':['BTC','XBT'],
            'ETHUSD':['ETH'],
            'ETHBTC':['ETHXBT'],
            'ALTBTC':['ADAXBT','BCHXBT','EOSXBT','LTCXBT','TRXXBT','XRPXBT']}


#----------------------------------------------------------------------------------------------#
# Functions
#----------------------------------------------------------------------------------------------#
def load_db(base):
    doc.create_index([('unsym',pymongo.ASCENDING),('timestamp',pymongo.ASCENDING)])
    cursor = doc.find({'unsym':{'$in':sym_ref[base]}, 'timestamp':{'$gte':'2019-06-03 09:04:57'}})
    data_db = pd.DataFrame(cursor)
    data_db.replace(0, np.nan, inplace=True)
    data = []
    for _,group in data_db.groupby('symbol', sort=False):
        data.append(group[['timestamp','symbol','mid','prem','pprem','anprem']].set_index('timestamp'))
    return data


#----------------------------------------------------------------------------------------------#
# Layout
#----------------------------------------------------------------------------------------------#
section_precision = {'BTCUSD':2, 'ETHUSD':2, 'ETHBTC':5, 'ALTBTC':8}

title = 'D1 History'

layout = html.Div( style={'marginLeft':35,'marginRight':35},
    children = [
        html.Div( className = 'row', style = {'margin-top': '35px'}, children = [
            html.Div( className = 'one-third column', children = [
                dcc.RadioItems(
                    id = 'd1h-base',
                    options = [{'label':i, 'value':i} for i in sym_ref],
                    value = 'BTCUSD',
                    labelStyle = {'display': 'inline-block'},
                    style = {'font-size': '1.4rem'}
                ),
                html.H6( id = 'd1h-table-timestamp', style = {'margin-top': '20px'}),
                dash_table.DataTable(
                    id = 'd1h-table',
                    style_cell = { 'padding-left': '10px', 'padding-right': '10px', },
                    style_data_conditional = [ { 'if': { 'filter': '{pprem} > 0' }, "color": "#008000", "fontWeight": "bold" }, { 'if': { 'filter': '{pprem} < 0' }, "color": "#B22222", "fontWeight": "bold" },]
                )
            ]),
            html.Div( className = 'two-thirds column', style = {'margin-top':'30px'}, children = [
                dcc.Graph(
                    id = 'd1h-index-main',
                ),
            ]),
        ]),
        html.Div( className = 'row', children = [
            html.Div( className = 'one-half column', children = [
                dcc.Graph(id = 'd1h-graph-ins')
            ]),
            html.Div( className = 'one-half column', children = [
                dcc.Graph(id = 'd1h-graph-prem')
            ]),
        ]),
        html.Div( className = 'row', children = [
            html.Div( className = 'one-half column', children = [
                dcc.Graph(id = 'd1h-graph-pprem')
            ]),
            html.Div( className = 'one-half column', children = [
                dcc.Graph(id = 'd1h-graph-anprem')
            ]),
        ]),
        html.Div( id = 'd1h-timestamp-history', style = {'display':'none'}),
        html.Div( id = 'd1h-hidden', style = {'display':'none'}),

        html.Div(id = 'd1h-test')
    ]
)


#----------------------------------------------------------------------------------------------#
# Callbacks
#----------------------------------------------------------------------------------------------#

# 1. Load Database for base
@app.callback(
    [Output('d1h-hidden', 'children'),Output('d1h-table', 'columns'), Output('d1h-table', 'active_cell')],
    [Input('d1h-base', 'value')]
)
def get_indices(base):
    data = load_db(base) # add timestamp in future to load data from a certain date
    columns = [ { 'name': 'Instr.', 'id': 'instr', },
                { 'name': 'Mid', 'id': 'mid', 'type': 'numeric', 'format': Format( precision = section_precision[base], scheme=Scheme.fixed ) },
                { 'name': 'Prem.', 'id': 'prem', 'type': 'numeric', 'format': Format( precision = section_precision[base], scheme=Scheme.fixed, sign=Sign.positive ) },
                { 'name': '% Prem.', 'id': 'pprem', 'type': 'numeric', 'format': FormatTemplate.percentage(2).sign(Sign.positive) },
                { 'name': 'An. Prem.', 'id': 'anprem', 'type': 'numeric', 'format': FormatTemplate.percentage(2).sign(Sign.positive) },
                { 'name': 'Exchange', 'id': 'ex', 'hidden': 'True' }]
    active_cell = None

    return json.dumps([i.to_dict() for i in data]), columns, active_cell


# 2. Update Table
@app.callback(
    [Output('d1h-table','data'),Output('d1h-timestamp-history','children'),Output('d1h-table-timestamp','children')],
    [Input('d1h-hidden', 'children'),Input('d1h-index-main', 'clickData'),Input('d1h-graph-ins', 'clickData'),
     Input('d1h-graph-prem', 'clickData'),Input('d1h-graph-pprem', 'clickData'),Input('d1h-graph-anprem', 'clickData')],
    [State('d1h-timestamp-history','children'),State('d1h-table-timestamp','children')]
)
def update_table(data,clickdata_main,clickdata_ins,clickdata_prem,clickdata_pprem,clickdata_anprem,timestamp_history,table_timestamp):
    data = [pd.DataFrame(i) for i in json.loads(data)]
    clickdata = [i['points'][0]['x'] for i in (clickdata_main,clickdata_ins,clickdata_prem,clickdata_pprem,clickdata_anprem) if i!=None]
    clickdata = [data[0].index[-1]] if clickdata==[] else clickdata

    table_timestamp = clickdata if table_timestamp==None else [table_timestamp]
    timestamp_history = clickdata if timestamp_history==None else json.loads(timestamp_history)
    t = [i for i in clickdata if i not in timestamp_history] if clickdata!=timestamp_history else table_timestamp
    t = pd.to_datetime(t).strftime('%Y-%m-%d %H:%M:%S')

    instruments = [data[i].loc[t,'symbol'].str.split('_').str[0] for i in range(len(data))]
    tab_data = []
    tab_data.append(instruments)
    tab_data.append([data[i].loc[t[-1],'mid'] for i in range(len(data)) if t[-1] in data[i].index])
    tab_data.append([data[i].loc[t[-1],'prem'] for i in range(len(data)) if t[-1] in data[i].index])
    tab_data.append([data[i].loc[t[-1],'pprem'] for i in range(len(data)) if t[-1] in data[i].index])
    tab_data.append([data[i].loc[t[-1],'anprem'] for i in range(len(data)) if t[-1] in data[i].index])
    table_data = pd.DataFrame(tab_data, index = ['instr','mid','prem','pprem','anprem']).T
    table_data = table_data.to_dict('rows')


    return table_data,json.dumps(clickdata), t[-1]


# 3. Update Index Graph
@app.callback(
    Output('d1h-index-main','figure'),
    [Input('d1h-hidden','children'),Input('d1h-table', 'active_cell')]
)
def main_chart(data,cell):
    data = [pd.DataFrame(i) for i in json.loads(data)]
    time = data[0].index

    instruments = [data[i]['symbol'][0] for i in range(len(data))]
    ins,ex = [i.split('_')[0] for i in instruments],[i.split('_')[1] for i in instruments]

    if cell==None:
        cell = 0
    else:
        if cell['row']<=len(ins):
            cell = cell['row']
        else:
            cell = 0

    print('---------------------------------------')
    print('main')
    print(cell)
    print('---------------------------------------')

    if any(i in ['.B'+sym for sym in sym_ref['ALTBTC']] for i in ins):
        sym = ins[cell].strip('.B')[:3] if not ins[cell][:3] in ('.BB','BCH') else 'BCH'
        show = [i for i in ins if sym in i]
    else:
        show = list(set(ex))

    traces_idx = [dict(x = time, y = data[i]['mid'], mode = 'lines', name = instruments[i].split('_')[0], line = {'width':1}) for i in range(len(data)) if instruments[i].startswith('.') and any(instruments[i].split('_')[j] in show for j in range(2))]
    return ({'data': traces_idx,
             'layout': { 'margin': {'t': 30, 'b': 50, 'l': 70, 'r': 30, }, 'height': 350, 'title': 'Index - Mid', 'xaxis': dict(showline = True), 'yaxis': dict(showline = True), 'showlegend':True} })


# 4. Update Other Graphs
@app.callback(
    [Output('d1h-graph-ins', 'figure'),Output('d1h-graph-prem', 'figure'),
     Output('d1h-graph-pprem', 'figure'),Output('d1h-graph-anprem', 'figure')],
    [Input('d1h-hidden', 'children'),Input('d1h-index-main', 'relayoutData'),Input('d1h-table', 'active_cell')],
)
def other_charts(data, selectedData, cell):

    data = [pd.DataFrame(i) for i in json.loads(data)]
    time = data[0].index

    xaxis_range = [selectedData['xaxis.range[0]'], selectedData['xaxis.range[1]']] if 'xaxis.range[1]' in selectedData else [time[0], time[-1]]

    instruments = [data[i]['symbol'][0] for i in range(len(data))]
    ins,ex = [i.split('_')[0] for i in instruments],[i.split('_')[1] for i in instruments]

    if cell==None:
        cell = 0
    else:
        if cell['row']<=len(ins):
            cell = cell['row']
        else:
            cell = 0

    if any(i in ['.B'+sym for sym in sym_ref['ALTBTC']] for i in ins):
        sym = ins[cell].strip('.B')[:3] if not ins[cell][:3] in ('.BB','BCH') else 'BCH'
        show = [i for i in ins if sym in i]
    else:
        show = ex[cell] if cell!=0 else list(set(ex))

    traces_ins = [dict(x = time, y = data[i]['mid'], mode = 'lines', name = instruments[i].split('_')[0], line = {'width':1}) for i in range(len(data)) if not instruments[i].startswith('.') and any(instruments[i].split('_')[j] in show for j in range(2))]
    traces_prem = [dict(x = time, y = data[i]['prem'], mode = 'lines', name = instruments[i].split('_')[0], line = {'width':1}) for i in range(len(data)) if not instruments[i].startswith('.') and any(instruments[i].split('_')[j] in show for j in range(2))]
    traces_pprem = [dict(x = time, y = data[i]['pprem'], mode = 'lines', name = instruments[i].split('_')[0], line = {'width':1}) for i in range(len(data)) if not instruments[i].startswith('.') and any(instruments[i].split('_')[j] in show for j in range(2))]
    traces_anprem = [dict(x = time, y = data[i]['anprem'], mode = 'lines', name = instruments[i].split('_')[0], line = {'width':1}) for i in range(len(data)) if not instruments[i].startswith('.') and any(instruments[i].split('_')[j] in show for j in range(2))]

    return ({'data': traces_ins, 'layout':{'xaxis':{'range': xaxis_range, 'showline':True},'yaxis':{'showline':True}, 'title': 'Instruments - Mid', 'showlegend':True}},
            {'data': traces_prem, 'layout':{'xaxis':{'range': xaxis_range, 'showline':True},'yaxis':{'showline':True}, 'title': 'Premium', 'showlegend':True}},
            {'data': traces_pprem, 'layout':{'xaxis':{'range': xaxis_range, 'showline':True},'yaxis':{'showline':True}, 'title': '% Premium', 'showlegend':True}},
            {'data': traces_anprem, 'layout':{'xaxis':{'range': xaxis_range, 'showline':True},'yaxis':{'showline':True}, 'title': 'Annualized Premium', 'showlegend':True}})
