import dash
import dash_core_components as dcc 
import dash_html_components as html 
import dash_table  
from dash.dependencies import Input, Output

import plotly.graph_objs as go 
import plotly.figure_factory as ff
from plotly.offline import plot, iplot

import datetime as dt
import json

import numpy as np
import pandas as pd

from deribit_api2 import get_data


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,external_stylesheets=external_stylesheets)

pricer_columns=['Qty','Expiry','T','Forward','Strike','Ins','Vol','Price','Delta','Gamma','Vega']

app.title='Pricer'

app.layout = html.Div(id='page',children=[

        html.Div(id='top',className='row',children=[

            html.Div(className='six columns',children=
                    [dcc.Graph(id='disp{}'.format(i)) for i in range(1,4)]),
            html.Div(className='six columns',children=
                    [dcc.Graph(id='disp{}'.format(i)) for i in range(4,7)]),
        ]),
        html.H6(id = 'last-update'),
        html.Div(id='the-data',style={'display':'none'}),
        dcc.Interval(
            id='interval-component',
            interval=120*1000, # in milliseconds= 2 minutes
            n_intervals=0
            ),
        html.Pre(id = 'click-data') ,
        dash_table.DataTable(id='pricer',
            columns =[{'id': c,'name':c} for c in pricer_columns],
            style_table={
                'maxHeight': '500',
                'overflowY': 'scroll'
            },
            style_cell={'maxWidth':30,'textAlign':'center'},
            style_header= {'backgroundColor':'lightgrey','fontWeight':'bold'},
            merge_duplicate_headers=True
            )
    ])

def flatplot(data,exp):
    
    data= [pd.read_json(json_data,orient='split') for json_data in json.loads(data)]
    fitparams=data[-2].round(4)
    optmats=data[:-2]
    mat=fitparams.index.get_loc(exp)
    optchart = optmats[mat].copy()
    title = str(fitparams.iloc[mat])
    chart_title=" - ".join([" ".join(i.split()) for i in title.split('\n')[:-4]])
    try:
        bid=go.Scatter(x=optchart["Strike"],y=optchart['BidVol']-optchart["Fit"],
        mode='markers',customdata=optchart['Ins'],text=optchart['T'],name='Bid Vol')
        ask=go.Scatter(x=optchart["Strike"],y=optchart["AskVol"]-optchart['Fit'],
        mode='markers',customdata=optchart['Ins'],text=optchart['T'],name ='Ask Vol')
    except:
        bid=go.Scatter(x=optchart["Strike"],y=optchart['BidVol']-optchart["MidVol"],
        mode='markers',customdata=optchart['Ins'],text=optchart['T'],name='Bid Vol')
        ask=go.Scatter(x=optchart["Strike"],y=optchart["AskVol"]-optchart['MidVol'],
        mode='markers',customdata=optchart['Ins'],text=optchart['T'],name='Ask Vol')

    layout=go.Layout(title=chart_title,titlefont=dict(
                family='Courier New, monospace',
                size=15,
                color='#7f7f7f'),
                xaxis=dict(title='<b>' + ' '.join(title.split()).split(' ')[-4] +'</b>'),
                yaxis=dict(showticklabels=False),
                height=200,
                showlegend=False
                )
    return go.Figure(data=[bid,ask],layout=layout)

def update_pricer(pricer,clickdata):
    line_to_add=dict(Qty = 1,
    Expiry = None,
    T = clickdata[0]['text'],
    Forward = None,
    Strike = clickdata[0]['x'],
    Ins = clickdata[0]['customdata'],
    Vol = None,
    Price = None,
    Delta = None,
    Gamma = None,
    Vega = None)
    pricer.append(line_to_add)
    return pricer 


@app.callback(Output('the-data','children'),
            [Input('interval-component','n_intervals')])
def update_data(n):
    print('updating',dt.datetime.now())
    data =  get_data()
    results = json.dumps([df.to_json(date_format='iso',orient='split') for df in data])
    print('finished updating', dt.datetime.now())
    return results

@app.callback(
    Output ('disp1','figure'),
    [Input('the-data','children')])
def plot1(data):
    try:
        d = [pd.read_json(json_data,orient='split') for json_data in json.loads(data)]
        fitparams=d[-2].round(4)
        mat=fitparams.index[0]
        return flatplot(data,mat)
    except:
        layout=go.Layout(height=200,yaxis=dict(showticklabels=False))
        return go.Figure(data=[],layout=layout)

@app.callback(
    Output ('disp2','figure'),
    [Input('the-data','children')])
def plot2(data):
    try:
        d = [pd.read_json(json_data,orient='split') for json_data in json.loads(data)]
        fitparams=d[-2].round(4)
        mat=fitparams.index[1]
        return flatplot(data,mat)
    except:
        layout=go.Layout(height=200,yaxis=dict(showticklabels=False))
        return go.Figure(data=[],layout=layout)

@app.callback(
    Output ('disp3','figure'),
    [Input('the-data','children')])
def plot3(data):
    try:
        d = [pd.read_json(json_data,orient='split') for json_data in json.loads(data)]
        fitparams=d[-2].round(4)
        mat=fitparams.index[2]
        return flatplot(data,mat)
    except:
        layout=go.Layout(height=200,yaxis=dict(showticklabels=False))
        return go.Figure(data=[],layout=layout)

@app.callback(
    Output ('disp4','figure'),
    [Input('the-data','children')])
def plot4(data):
    try:
        d = [pd.read_json(json_data,orient='split') for json_data in json.loads(data)]
        fitparams=d[-2].round(4)
        mat=fitparams.index[3]
        return flatplot(data,mat)
    except:
        layout=go.Layout(height=200,yaxis=dict(showticklabels=False))
        return go.Figure(data=[],layout=layout)

@app.callback(
    Output ('disp5','figure'),
    [Input('the-data','children')])
def plot5(data):
    try:
        d = [pd.read_json(json_data,orient='split') for json_data in json.loads(data)]
        fitparams=d[-2].round(4)
        mat=fitparams.index[4]
        return flatplot(data,mat)
    except:
        layout=go.Layout(height=200,yaxis=dict(showticklabels=False))
        return go.Figure(data=[],layout=layout)

@app.callback(
    Output ('disp6','figure'),
    [Input('the-data','children')])
def plot6(data):
    try:
        d = [pd.read_json(json_data,orient='split') for json_data in json.loads(data)]
        fitparams=d[-2].round(4)
        mat=fitparams.index[5]
        return flatplot(data,mat)
    except:
        layout=go.Layout(height=200,yaxis=dict(showticklabels=False))
        return go.Figure(data=[],layout=layout)

@app.callback(
    Output('click-data','children'),
    [Input('disp1','clickData')])
def display_clickdata(clickdata):
    return json.dumps(clickdata)


if __name__ == '__main__':
    app.run_server(debug=True,port=8057)