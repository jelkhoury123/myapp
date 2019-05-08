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
import json
import datetime as dt
import pytz

import sys
sys.path.append('..') # add parent directory to the path to import app

from app import app  # app is the main app which will be run on the server in index.py

REFRESH_RATE = 5 
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

def get_bitmex():
    ref=pd.DataFrame(bitmex.load_markets()).T
    indexes=[ref.loc[ins]['info']['referenceSymbol'] for ins in ref.index ]
    indexes=list(set([i for i in indexes if len(i)>4 and (i.endswith('XBT') or i.endswith('ETH'))]))
    ins=list(ref[ref['active']==True].index)
    ins=[ i for i in ins if not '_' in i]
    bitmex_ins=indexes+ins
    expiries = pd.DataFrame([(ins,ref.loc[ins]['info']['expiry']) for ins in ref.index ]).set_index(0).T[bitmex_ins].T
    bitmex_set=pd.DataFrame(bitmex.load_markets())[bitmex_ins].T
    bitmex_set['pair']=bitmex_set.apply(lambda x : x['base']+'/'+x['quote'],axis=1)
    bitmex_set['type']=bitmex_set.apply(lambda x: 'index' if x['id'].startswith('.') else 'swap' if x['swap'] else 'future',axis=1)
    bitmex_set['T']=pd.to_datetime(expiries[1],utc=True).apply(lambda x:(x-dt.datetime.now(tz=pytz.utc))/dt.timedelta(days=365.25))
    bitmex_set = bitmex_set[['pair','type','T']]
    return bitmex_set

def add_last(data):
    last=pd.DataFrame(bitmex.fetch_tickers())[data.T.columns].T['last']
    time = dt.datetime.now()
    data[time]=last
    return data

pairs = ['BTC/USD','ETH/USD','ETH/BTC']
alt_pairs= ['BCH/BTC','LTC/BTC','XRP/BTC','TRX/BTC','EOS/BTC','ADA/BTC']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)

precision = {'BTC/USD':2,'ETH/USD':2,'ETH/BTC':5}

app.layout= html.Div([
        html.Div(className= 'four columns',children = [
            html.Div(style={'height':'380px'},children=[
            html.Br(),  
            html.Hr(style={'border-color':'#cb1828'}),
            html.B(id='d1-text{}'.format(i),children=pairs[i]),
            dash_table.DataTable(id='d1-table{}'.format(i),
                                columns=[{'id':'Instrument','name':'Instrument'},
                                        {'id':'Last','name':'Last','type':'numeric','format':Format(precision=list(precision.values())[i],scheme=Scheme.fixed,)},
                                        {'id':'premium','name':'Premium','type':'numeric','format':Format(precision=list(precision.values())[i],scheme=Scheme.fixed,)},
                                        {'id':'premium%','name':'Premium%','type':'numeric','format':FormatTemplate.percentage(2).sign(Sign.positive)},
                                        {'id':'Annpremium%','name':'Annpremium%','type':'numeric','format':FormatTemplate.percentage(2).sign(Sign.positive)},
                                        ],
                                style_header={'backgroundColor':'#DCDCDC','fontWeight':'bold'},
                                style_cell={'textAlign':'center','width':'10%'},
                                style_table={'border': '1px solid lightgrey','border-collapse':'collapse'},
                                style_data_conditional=[{
                                                        'if' : {'filter':  'premium > num(0)' },
                                                        'color':'green'
                                                                }
                                                        ]+[
                                                        {
                                                        'if' : {'filter':  'premium < num(0)' },
                                                        'color':'#cb1828'
                                                        }
                                                    ]
                                ) 
                            ]) for i in range(len(pairs)) 
        ]+[html.Div(style={'height':'380px'},
            children=[html.Br(),
                    html.Hr(style={'border-color':'#cb1828'}),
                    html.B(id='d1-text-altcoins',children='Alt Coins'),
                    dash_table.DataTable(id='d1-table-altcoins',
                                        columns=[{'id':'Instrument','name':'Instrument'},
                                                {'id':'Last','name':'Last','type':'numeric','format':Format(precision=8,scheme=Scheme.fixed,)},
                                                {'id':'premium','name':'Premium','type':'numeric','format':Format(precision=8,scheme=Scheme.fixed,)},
                                                {'id':'premium%','name':'Premium%','type':'numeric','format':FormatTemplate.percentage(2).sign(Sign.positive)},
                                                {'id':'Annpremium%','name':'Annpremium%','type':'numeric','format':FormatTemplate.percentage(2).sign(Sign.positive)},
                                                ],
                                            style_header={'backgroundColor':'#DCDCDC','fontWeight':'bold'},
                                            style_cell={'textAlign':'center','width':'10%'},
                                            style_table={'border': '1px solid lightgrey','border-collapse':'collapse'},
                                            style_data_conditional=[{
                                                        'if' : {'filter':  'premium > num(0)' },
                                                        'color':'green'
                                                                }
                                                        ]+[
                                                        {
                                                        'if' : {'filter':  'premium < num(0)' },
                                                        'color':'#cb1828'
                                                        }
                                                    ]
                                            
                                        )
                        ]
                    )]
        ),
        html.Div(className = 'four columns',children=[
            html.Div(style={'height':'380px'},children= [
            html.Br(),
            html.Hr(style={'border-color':'#cb1828'}),
            dcc.Graph(id='d1-spot-graph{}'.format(i),style={'height':'90%'})]) for i in range(4)
        ]),
        html.Div(className = 'four columns',children=[
            html.Div(style={'height':'380px'},children=[
            html.Br(),
            html.Hr(style={'border-color':'#cb1828'}),
            dcc.Graph(id='d1-premium-graph{}'.format(i),style={'height':'90%'})]) for i in range(4)
        ]),
        dcc.Interval(id='d1-interval',interval = REFRESH_RATE * 1000,n_intervals = 0),
        html.Div(id='d1-data',style={'display':'none'},children=get_bitmex().to_json(date_format='iso',orient='split'))
])

@app.callback(Output('d1-data','children'),
    [Input('d1-interval','n_intervals')],
    [State('d1-data','children')])
def update(n,data):
    data = pd.read_json(data,orient='split')
    return add_last(data).to_json(date_format='iso',orient='split')

@app.callback([Output('d1-table{}'.format(i),'data') for i in range(len(pairs))],
    [Input('d1-data','children')])
def update_main_tables(data):
    tables=[]
    data = pd.read_json(data,orient='split')
    for pair in pairs:
        data_pair=data[data['pair']==pair]
        data_pair = data_pair[['pair','type','T',data_pair.columns[-1]]]
        data_pair.columns=['pair','type','T','Last']
        index_last = data_pair[data_pair['type']=='index']['Last'][0]
        data_pair.loc[:,'premium']=data_pair.loc[:,'Last']-index_last
        data_pair.loc[:,'premium%']=data_pair.loc[:,'premium']/index_last
        data_pair.loc[:,'Annpremium%']=data_pair.loc[:,'premium%'] /data_pair.loc[:,'T']
        data_pair = data_pair.drop(['pair','type','T'],axis=1).reset_index()
        data_pair.columns=['Instrument',*data_pair.columns[1:]]
        table_data = data_pair.to_dict('rows')
        tables.append(table_data)
    return tables

@app.callback(Output('d1-table-altcoins','data'),
    [Input('d1-data','children')])
def update_alt_table(data):
    data = pd.read_json(data,orient='split')
    data_pair=data[data['pair'].apply(lambda x: True if x in alt_pairs else False)]
    data_pair = data_pair[['pair','type','T',data_pair.columns[-1]]]
    data_pair.columns=['pair','type','T','Last']
    alt_df =pd.DataFrame(columns=['Last','premium','premium%','Annpremium%'])
    for pdf in data_pair.groupby('pair'):
        pdf=pdf[-1]
        index_last = pdf[pdf['type']=='index']['Last'][0]
        pdf.loc[:,'premium']=pdf.loc[:,'Last']-index_last
        pdf.loc[:,'premium%']=pdf.loc[:,'premium']/index_last
        pdf.loc[:,'Annpremium%']=pdf.loc[:,'premium%'] /pdf.loc[:,'T']
        pdf = pdf.drop(['pair','type','T'],axis=1).reset_index()
        pdf.columns=['Instrument',*pdf.columns[1:]]
        alt_df=pd.concat([alt_df,pdf],sort = True)
    table = alt_df.to_dict('rows')      
    return table

@app.callback([Output('d1-spot-graph{}'.format(i),'figure') for i in range(3)],
    [Input('d1-data','children')])
def update_spot_graphs(data):
    data = pd.read_json(data,orient='split')
    figures = []
    for pair in pairs:
        data_pair = data[data['pair']==pair]
        data_pair = data_pair.drop(['pair','type','T'],axis=1)
        figure = go.Figure(data= [{'x':data_pair.columns , 'y':data_pair.iloc[0],'name':data_pair.index[0]} ],
        layout = {'title':pair})
        figures.append(figure)
    return figures

@app.callback([Output('d1-premium-graph{}'.format(i),'figure') for i in range(3)],
    [Input('d1-data','children')])
def update_prem_graphs(data):
    data = pd.read_json(data,orient='split')
    figures = []
    for pair in pairs:
        data_pair = data[data['pair']==pair]
        index_column = data_pair[data_pair['type']=='index'].index[0]
        data_pair=data_pair.T
        data_pair.drop(['pair','type','T'],inplace=True)
        for col in data_pair.columns[1:]:
            data_pair[col]=data_pair[col]/data_pair[index_column]-1
        data_pair[index_column]=0
        data_pair=data_pair.T
        figure = go.Figure(data= [{'x':data_pair.columns , 'y':data_pair.loc[i]*100,'name':i} for i in data_pair.index],
        layout = {'title':pair})
        figures.append(figure)
    return figures

@app.callback(Output('d1-spot-graph3','figure'),
    [Input('d1-data','children'),Input('d1-table-altcoins','active_cell')],
    [State('d1-table-altcoins','data')])
def update_alt__spot_graphs(data,active_cell,table_data):
    data = pd.read_json(data,orient='split')
    row = active_cell[0]
    table_data=pd.DataFrame(table_data)
    instrument = table_data['Instrument'].iloc[row]
    pair = data.loc[instrument,'pair']
    data_pair = data[data['pair']==pair]
    data_pair = data_pair.drop(['pair','type'],axis=1)
    figure = go.Figure(data= [{'x':data_pair.columns , 'y':data_pair.iloc[0],'name':data_pair.index[0]}],
                        layout = {'title':pair})

    return figure

@app.callback(Output('d1-premium-graph3','figure'),
    [Input('d1-data','children'),Input('d1-table-altcoins','active_cell')],
    [State('d1-table-altcoins','data')])
def update_alt_prem_graphs(data,active_cell,table_data):
    data = pd.read_json(data,orient='split')
    row ,column = active_cell
    table_data=pd.DataFrame(table_data)
    instrument = table_data['Instrument'].iloc[row]
    pair = data.loc[instrument,'pair']
    data_pair = data[data['pair']==pair]
    index_column = data_pair[data_pair['type']=='index'].index[0]
    data_pair=data_pair.T
    data_pair.drop(['pair','type'],inplace=True)
    for col in data_pair.columns[1:]:
        data_pair[col]=data_pair[col]/data_pair[index_column]-1
    data_pair[index_column]=0
    data_pair=data_pair.T
    figure = go.Figure(data= [{'x':data_pair.columns , 'y':data_pair.loc[i],'name':i} for i in data_pair.index],
    layout = {'title':pair})
    return figure


if __name__ =='__main__':
    app.run_server(debug=True,port = 5089 )


