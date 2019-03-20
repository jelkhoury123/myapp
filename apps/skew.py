import dash
import dash_core_components as dcc 
import dash_html_components as html 
import dash_table
from dash_table.Format import Format, Scheme, Sign, Symbol
import dash_table.FormatTemplate as FormatTemplate
from dash.dependencies import Input, Output

import plotly.graph_objs as go 
import plotly.figure_factory as ff
from plotly.offline import plot, iplot

import datetime as dt
import json

import numpy as np
import pandas as pd

import sys
sys.path.append('..')

from deribit_api2 import get_data

from app import app

title='Skew Plots'

fit_columns = ['Expiry','Sigma','Skew','Kurt','Alpha','Ref','vol_q','price_q','VolSpread']
fit_table_columns = [{'id':c,'name':c} for c in fit_columns ]
for dic in fit_table_columns:
    if dic['id'] in ['Sigma','Skew','Kurt','VolSpread','vol_q']:
        dic['type'] = 'numeric'
        dic['format']= FormatTemplate.percentage(2)

layout_block = [ html.Div(className='six columns',children=[
                    html.Div(
                    dcc.RadioItems(id='skew-rb{}'.format(i), options = [
                                                    {'label':'Graph','value':'Graph'},
                                                    {'label':'Table','value':'Table'}
                                                    ],
                                            value='Graph',
                                            labelStyle={'display':'inline-block'})
                    ),
                    html.Div(dcc.Dropdown(id='skew-dd{}'.format(i))),
                    html.Div(id='skew-disp{}'.format(i))])
                 for i in range (1,5)]

nav_menu = html.Div(children=[
                    html.A('HVG', href='/apps/hvg',style={'backgroung-color':'red','color':'black','padding':'10px 15px',
                                        'text-align':'center','display':'inline-block','text-decoration':'None'}),
                    html.A('Skew', href='/apps/skew',style={'backgroung-color':'#c1bfbf','color':'black','padding':'10px 15px',
                                        'text-align':'center','display':'inline-block','text-decoration':'None'}),
                    html.A('HIV', href='/apps/hiv',style={'backgroung-color':'#c1bfbf','color':'black','padding':'10px 15px',
                                        'text-align':'center','display':'inline-block','text-decoration':'None'}),
                    html.A('Pricer', href='/apps/pricer',style={'backgroung-color':'#c1bfbf','color':'black','padding':'10px 15px',
                                        'text-align':'center','display':'inline-block','text-decoration':'None'}),
                    html.A('Order Book', href='/apps/order_book',style={'backgroung-color':'#c1bfbf','color':'black','padding':'10px 15px',
                                        'text-align':'center','display':'inline-block','text-decoration':'None'}),
                    html.A('Futures', href='/apps/futures',style={'backgroung-color':'#c1bfbf','color':'black','padding':'10px 15px',
                                        'text-align':'center','display':'inline-block','text-decoration':'None'}),
                    ])

layout = html.Div(id='skew-page',children=[
                        html.Div(className='row',children=[nav_menu]),
                        html.Hr(),
                        html.Div(id='top',className='row',children = layout_block[:2]),
                        html.Div(id='middle',className='row',children=layout_block[-2:]),
                        html.Div(id='bottom',className='row',children=[
                                            html.Div(className='six columns',children=[
                                                    html.H6(id = 'last-skew-update'),
                                                    html.Div(dash_table.DataTable(id ='skew-fit-table', columns = fit_table_columns,
                                                        style_header = {'backgroundColor':'lightgrey','fontWeight':'bold'},
                                                        style_cell = {'textAlign':'center'},
                                                        style_table = {'vertical-align':'middle','horizontal-align':'middle'},
                                                        )
                                                    ),]
                                                    ),
                                                    html.Div(id='skew-TS-charts',className='six columns',children=[
                                                            dcc.Graph(id='skew-TS')
                                                ])
                                        
                                    ]),
                        html.Div(id='the-skew-data',style={'display':'none'}),
                        dcc.Interval(
                            id='skew-interval-component',
                            interval=1200*1000, # in milliseconds= 2 minutes
                            n_intervals=0
                            ), 
                        ]
    )
    
def generate_table(data,exp,max_rows=25):
    '''This is the option table at the chosen maturity'''
    data = [pd.read_json(json_data,orient='split') for json_data in json.loads(data)]
    fitparams = data[-2]
    optmats = data[:-2]
    mat = fitparams.index.get_loc(exp)
    df = optmats[mat].reset_index().tail(max_rows)
    df = df[['Strike','Ins','uPx','Bid$','Ask$','TV','MidVol','Fit','Vega']]
    decimals = pd.Series([0]*2+[2]*4+[4]*2+[2],index=df.columns)
    df = df.round(decimals)
    option_table_columns = [{'id': c,'name':[exp,c]}for c in df.columns]
    for dic in option_table_columns:
        if dic['id'] in ['MidVol','Fit']:
            dic['type'] = 'numeric'
            dic['format']= FormatTemplate.percentage(2)
    return dash_table.DataTable(
        data=df.to_dict('rows'),
        columns =option_table_columns,
        n_fixed_rows=2,
        style_table={
            'maxHeight': '500',
            'overflowY': 'scroll'
        },
        style_cell={'maxWidth':30,'textAlign':'center'},
        style_header= {'backgroundColor':'lightgrey','fontWeight':'bold'},
        merge_duplicate_headers=True
    )

def skewplot(data,exp):
    '''This is the skew figure at the chosen maturity'''
    data = [pd.read_json(json_data,orient='split') for json_data in json.loads(data)]
    fitparams = data[-2].round(4)
    optmats = data[:-2]
    mat = fitparams.index.get_loc(exp)
    optchart = optmats[mat].copy()
    title = str(fitparams.iloc[mat])
    chart_title=" - ".join([" ".join(i.split()) for i in title.split('\n')[:-4]])
    bid = go.Scatter(x=optchart["Strike"],y=optchart["BidVol"],mode='markers',name='Bid vol')
    ask = go.Scatter(x=optchart["Strike"],y=optchart["AskVol"],mode='markers',name = 'Ask Vol')
    mid = go.Scatter(x=optchart["Strike"],y=optchart["MidVol"],mode = 'lines',name = 'Mid Vol')
    fit = go.Scatter(x=optchart["Strike"],y=optchart["Fit"],mode = 'lines',name = 'Fit Vol')
    layout = go.Layout(title=chart_title,titlefont=dict(
                family='Courier New, monospace',
                size=15,
                color='#7f7f7f'),
                yaxis=dict(
                    title='<b>' + ' '.join(title.split()).split(' ')[-4] +'</b>',
                    titlefont=dict(
                        family='Courier New, monospace',
                        size=18,
                        color='#7f7f7f'),
                        tickformat ='.1%'
                )
        )
    return dcc.Graph(figure=go.Figure(data=[bid,mid,fit,ask],layout=layout))

@app.callback([Output('the-skew-data','children'),Output('last-skew-update','children')],
            [Input('skew-interval-component','n_intervals')])
def update_data(n):
    print('updating',dt.datetime.now())
    data =  get_data()
    results = json.dumps([df.to_json(date_format='iso',orient='split') for df in data])
    print('finished updating', dt.datetime.now())
    return results , 'Last update:' + 20* ' '  +  '{}'.format(dt.datetime.now().strftime("%Y-%m-%d  %H:%M"))

output_ids =['skew-dd{}'.format(i) for i in range(1,5)]
output_properties=['options','value']
output_elements = [(i,j) for i in output_ids for j in output_properties]

def create_callback(output):
    def callback(dfs):
        fitparams = pd.read_json(json.loads(dfs)[-2],orient='split')
        if output[1] == 'options':
            return  [{'label':expiry,'value':expiry} for expiry in fitparams.index]
        elif output[1] == 'value':
            ix = int(output[0][-1]) - 5
            return fitparams.index[ix]
    return callback

for output_element in output_elements:
    callback = create_callback(output_element)
    app.callback(Output(*output_element),[Input('the-skew-data','children')])(callback)

def show(data,display,exp):
    if display == 'Graph':
        return skewplot(data,exp)
    elif display =='Table':
        return generate_table(data,exp)

for i in range(1,5):
    app.callback(
        Output('skew-disp{}'.format(i),'children'),
        [Input('the-skew-data','children'),Input('skew-rb{}'.format(i),'value'),Input('skew-dd{}'.format(i),'value')])(show)

@app.callback(
    Output('skew-fit-table','data'),
    [Input('the-skew-data','children')])
def ts_table(data):
    data = [pd.read_json(json_data,orient='split') for json_data in json.loads(data)]
    fitparams = data[-2].round(4)
    table_data = fitparams.reset_index().rename(columns={'index':'Expiry'}).to_dict('rows')
    return table_data   

@app.callback(
    Output('skew-TS','figure'),
    [Input('the-skew-data','children'),Input('skew-fit-table','active_cell')])
def TS(data,choice):
    data= [pd.read_json(json_data,orient='split') for json_data in json.loads(data)]
    fitparams=data[-2]
    palette=dict(zip(fitparams.columns,['#50AEEC','#FF6347','#228B22','#DAA520','#708090','#C0C0C0','#F4A460','#D2691E']))
    choice = fitparams.columns[choice[-1]-1]
    return go.Figure(data= [go.Scatter(x=fitparams.index,y=fitparams[choice],
                            name =choice,line=dict(color=palette[choice]))],
                    layout=go.Layout(dict(title = choice + ' Term Structure',
                            yaxis=dict(tickformat ='.1%' if choice in ['Sigma','Skew','Kurt','vol_q','VolSpread'] else '.2f'))))

