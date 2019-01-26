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

app.title='Skew Plots'

layout_block = [ html.Div(className='six columns',children=[
                    html.Div(
                    dcc.RadioItems(id='rb{}'.format(i), options = [
                                                    {'label':'Graph','value':'Graph'},
                                                    {'label':'Table','value':'Table'}
                                                    ],
                                            value='Graph',
                                            labelStyle={'display':'inline-block'})
                    ),
                    html.Div(dcc.Dropdown(id='dd{}'.format(i))),
                    html.Div(id='disp{}'.format(i))])
                 for i in range (1,5)]

app.layout = html.Div(id='page',children=[
                                        html.Div(id='top',className='row',children = layout_block[:2]),
                                        html.Div(id='middle',className='row',children=layout_block[-2:]),
                                        html.Div(id='bottom',className='row',style= {'backgroundColor':'#FFFFFF'},children=[
                                                            html.Div(className='six columns',children=[
                                                            html.H5(id = 'last-update'),
                                                            html.Div(dash_table.DataTable(id='fit-table',
                                                                        columns =[{'id': c,'name':c} for c in ['Expiry','Sigma','Skew','Kurt','Alpha','Ref','vol_q','price_q','VolSpread']],
                                                                        style_header= {'backgroundColor':'lightgrey','fontWeight':'bold'},
                                                                        style_cell = {'textAlign':'center'},
                                                                        style_table = {'vertical-align':'middle','horizontal-align':'middle'}
                                                                        )
                                                            ),
                                                            ]),
                                                            html.Div(id='TS-charts',className='six columns',children=[
                                                                    dcc.Graph(id='TS')
                                                        ])
                                                        
                                                    ]),
                                        html.Div(id='the-data',style={'display':'none'}),
                                        dcc.Interval(
                                            id='interval-component',
                                            interval=1200*1000, # in milliseconds= 2 minutes
                                            n_intervals=0
                                            )      
    ])
    
def generate_table(data,exp,max_rows=25):
    data= [pd.read_json(json_data,orient='split') for json_data in json.loads(data)]
    fitparams=data[-2]
    optmats=data[:-2]
    mat=fitparams.index.get_loc(exp)
    df=optmats[mat].reset_index().tail(max_rows).round(2)
    df=df[['Strike','Ins','uPx','Bid$','Ask$','TV','MidVol','Fit','Vega']]
    return dash_table.DataTable(
        data=df.to_dict('rows'),
        columns =[{'id': c,'name':[exp,c]}for c in df.columns],
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
    data= [pd.read_json(json_data,orient='split') for json_data in json.loads(data)]
    fitparams=data[-2].round(4)
    optmats=data[:-2]
    mat=fitparams.index.get_loc(exp)
    optchart = optmats[mat].copy()
    title = str(fitparams.iloc[mat])
    chart_title=" - ".join([" ".join(i.split()) for i in title.split('\n')[:-4]])
    bid=go.Scatter(x=optchart["Strike"],y=optchart["BidVol"],mode='markers',name='Bid vol')
    ask=go.Scatter(x=optchart["Strike"],y=optchart["AskVol"],mode='markers',name = 'Ask Vol')
    mid=go.Scatter(x=optchart["Strike"],y=optchart["MidVol"],mode = 'lines',name = 'Mid Vol')
    fit=go.Scatter(x=optchart["Strike"],y=optchart["Fit"],mode = 'lines',name = 'Fit Vol')
    layout=go.Layout(title=chart_title,titlefont=dict(
                family='Courier New, monospace',
                size=15,
                color='#7f7f7f'),
                yaxis=dict(
                    title='<b>' + ' '.join(title.split()).split(' ')[-4] +'</b>',
                    titlefont=dict(
                        family='Courier New, monospace',
                        size=18,
                        color='#7f7f7f')
                )
        )
    return dcc.Graph(figure=go.Figure(data=[bid,mid,fit,ask],layout=layout))

@app.callback(Output('the-data','children'),
            [Input('interval-component','n_intervals')])
def update_data(n):
    print('updating',dt.datetime.now())
    data =  get_data()
    results = json.dumps([df.to_json(date_format='iso',orient='split') for df in data])
    print('finished updating', dt.datetime.now())
    return results

@app.callback(Output('last-update','children'),[Input('interval-component','n_intervals')])
def last_update(n):
    return  'Last update:' + 20* ' '  +  '{}'.format(dt.datetime.now().strftime("%Y-%m-%d  %H:%M"))

output_ids =['dd{}'.format(i) for i in range(1,5)]
output_properties=['options','value']
output_elements = [(i,j) for i in output_ids for j in output_properties]

def create_callback(output):
    def callback(dfs):
        fitparams = pd.read_json(json.loads(dfs)[-2],orient='split')
        if output[1]=='options':
            return  [{'label':expiry,'value':expiry} for expiry in fitparams.index]
        elif output[1]=='value':
            ix = int(output[0][-1]) - 5
            return fitparams.index[ix]
    return callback

for output_element in output_elements:
    callback=create_callback(output_element)
    app.callback(Output(*output_element),[Input('the-data','children')])(callback)

def show(data,display,exp):
    if display == 'Graph':
        return skewplot(data,exp)
    elif display =='Table':
        return generate_table(data,exp)

for i in range(1,5):
    app.callback(
        Output('disp{}'.format(i),'children'),
        [Input('the-data','children'),Input('rb{}'.format(i),'value'),Input('dd{}'.format(i),'value')])(show)

@app.callback(
    Output('fit-table','data'),
    [Input('the-data','children')])
def ts_table(data):
    data= [pd.read_json(json_data,orient='split') for json_data in json.loads(data)]
    fitparams=data[-2].round(4)
    return fitparams.reset_index().rename(columns={'index':'Expiry'}).to_dict('rows')           

@app.callback(
    Output('TS','figure'),
    [Input('the-data','children'),Input('fit-table','active_cell')])
def TS(data,choice):
    data= [pd.read_json(json_data,orient='split') for json_data in json.loads(data)]
    fitparams=data[-2]
    palette=dict(zip(fitparams.columns,['#50AEEC','#FF6347','#228B22','#DAA520','#708090','#C0C0C0','#F4A460','#D2691E']))
    choice=fitparams.columns[choice[-1]-1]
    return go.Figure(data= [go.Scatter(x=fitparams.index,y=fitparams[choice],
                            name =choice,line=dict(color=palette[choice]))],
                    layout=go.Layout(title=choice + ' Term Structure'))

if __name__ == '__main__':
    app.run_server(debug=True,port=8052)
