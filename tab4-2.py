import dash
import dash_core_components as dcc 
import dash_html_components as html 
import dash_table  
from dash.dependencies import Input, Output, State
from plotly import tools

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

strategy_columns = ['Qty','Expiry','Strike','Ins'] 
tv_columns= ['T','BidVol','Fit','AskVol','Bid$','TV','Ask$']
greek_columns= ['Delta%','Delta$','Gamma$','Vega','Vega$']
pricer_columns = strategy_columns + tv_columns + greek_columns

app.title='Pricer'

app.layout = html.Div(id='page',children=[
        html.H3('Options'),
        html.Div(id='top',children=[
            html.Div(children=
                    [dcc.Graph(id='disp')],
        )]),
        html.H6(id = 'last-update'),
        html.Div(id='the-data',style={'display':'none'}),
        dcc.Interval(
            id='interval-component',
            interval=120*1000, # in milliseconds= 2 minutes
            n_intervals=0
            ),
        html.Pre(id = 'click-data') ,
        html.Div(children=[
            dash_table.DataTable(id='select-option',
                columns =[{'id': c,'name':c} for c in strategy_columns],
                data=[],
                style_cell={'maxWidth':30,'textAlign':'center'},
                style_header= {'backgroundColor':'lightgrey','fontWeight':'bold'},
                merge_duplicate_headers=True
                )],style={'display':'none'}),
        
        html.Div(className='four columns',children=[
            html.H3('Strategy'),
            dash_table.DataTable(id='strategy',
                columns =[{'id': c,'name':c} for c in strategy_columns],
                data=[],
                style_cell={'maxWidth':30,'textAlign':'center'},
                style_header= {'backgroundColor':'lightgrey','fontWeight':'bold'},
                merge_duplicate_headers=True,
                row_deletable = True,
                editable = True
                ),
                html.H3('Pricer')]
                ),
        html.Div(className='three columns',children =[
            html.Button(id='button',children='Price Strategy')]),
        html.Div(className='row',children=[
            dash_table.DataTable(id='pricer',
                columns =[{'id': c,'name':c} for c in pricer_columns],
                data=[],
                style_cell={'maxWidth':30,'textAlign':'center'},
                style_header= {'backgroundColor':'lightgrey','fontWeight':'bold'},
                merge_duplicate_headers=True,
                row_deletable = True,
                editable=True)
        ])
             
])

def flatplot(data):
    
    data= [pd.read_json(json_data,orient='split') for json_data in json.loads(data)]
    fitparams=data[-2].round(4)
    optmats=data[:-2]
    chart_titles= [str(fitparams.iloc[mat]) for mat in range(len(fitparams.index))]
    subplot_titles=tuple(" - ".join([" ".join(i.split()) for i in title.split('\n')[:-4]]) for title in chart_titles)
    fig = tools.make_subplots(rows = 3 , cols = 2,subplot_titles=subplot_titles)
    fig['layout'].update(height=600)
    for mat in range(len(fitparams.index)):
        domain_ref= .225*(4 - 2*(mat//2))
        position=(mat//2+1,mat%2+1)
        optchart = optmats[mat].copy()
        try:
            bid=go.Scatter(x=optchart["Strike"],y=optchart['BidVol']-optchart["Fit"],
            mode='markers',customdata=optchart['Ins'],text=optchart['Expiry'],name='Bid Vol',line=dict(color='blue'))
            ask=go.Scatter(x=optchart["Strike"],y=optchart["AskVol"]-optchart['Fit'],
            mode='markers',customdata=optchart['Ins'],text=optchart['Expiry'],name ='Ask Vol',line=dict(color='red'))
        except:
            bid=go.Scatter(x=optchart["Strike"],y=optchart['BidVol']-optchart["MidVol"],
            mode='markers',customdata=optchart['Ins'],text=optchart['Expiry'],name='Bid Vol',line=dict(color='blue'))
            ask=go.Scatter(x=optchart["Strike"],y=optchart["AskVol"]-optchart['MidVol'],
            mode='markers',customdata=optchart['Ins'],text=optchart['Expiry'],name='Ask Vol',line=dict(color='red'))
        fig.append_trace(bid,*position)
        fig.append_trace(ask,*position)
        fig['layout']['xaxis{}'.format('' if mat ==0 else mat+1)].update(title='<b>' + str(fitparams.index[mat]).split(' ')[0]+'</b>')
        fig['layout']['yaxis{}'.format('' if mat ==0 else mat+1)].update(showticklabels=False,domain=[domain_ref,domain_ref+0.1])
        fig['layout'].update(showlegend=False)
        fig['layout'].update(clickmode='event+select')

    return fig


@app.callback(Output('the-data','children'),
            [Input('interval-component','n_intervals')])
def update_data(n):
    print('updating',dt.datetime.now())
    data =  get_data()
    results = json.dumps([df.to_json(date_format='iso',orient='split') for df in data])
    print('finished updating', dt.datetime.now())
    return results

@app.callback(Output('click-data','children'),
[Input('disp','clickData')])
def clickdata(clickdata):
    return json.dumps(clickdata)

@app.callback(
    Output ('disp','figure'),
    [Input('the-data','children')])
def plotax(data):
    return flatplot(data)

@app.callback(
    Output('select-option','data'),
    [Input('disp','clickData')])
def update_selection(clickdata):
    line_to_add=dict(Qty = -1 if clickdata['points'][0]['curveNumber']%2 == 0 else 1,
    Expiry = clickdata['points'][0]['text'], 
    Strike = clickdata['points'][0]['x'],
    Ins = clickdata['points'][0]['customdata'],
    )
    return [line_to_add]

@app.callback(
    Output('strategy','data'),
    [Input('select-option','data')],
    [State('strategy','data')]
)
def update_startegy(option_to_add,strategy):
    strategy.append(option_to_add[0])
    return strategy
@app.callback(
    Output('pricer','data'),
    [Input('button','n_clicks')],
    [State('strategy','data'),State('pricer','data'),State('the-data','children')]
)
def update_pricer(n_clicks,strategy,pricerdata,alldata):
    data= [pd.read_json(json_data,orient='split') for json_data in json.loads(alldata)]
    fitparams = data[-2].round(4)
    st=[]
    for line in strategy:
        pricer_line=dict(Qty=line['Qty'],Expiry=line['Expiry'],Strike=line['Strike'],Ins=line['Ins'])
        expix=list(fitparams.index).index(pd.to_datetime(line['Expiry']))
        opts=data[expix]
        my_option=opts[(opts['Strike']==line['Strike']) & (opts['Ins']==line['Ins'])]
        print(my_option)
        for col in tv_columns:
            pricer_line[col]=round(my_option[col].iloc[0],4)
            print(pricer_line[col])
        st.append(pricer_line)
    st.append(dict(Qty='Total', TV = sum([line['TV'] * line['Qty'] for line in st])))
    pricerdata+=st
    return pricerdata

if __name__ == '__main__':
    app.run_server(debug=True,port=8054)