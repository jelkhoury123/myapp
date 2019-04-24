import dash
import dash_core_components as dcc 
import dash_html_components as html 
import dash_table 
from dash_table.Format import Format, Scheme, Sign, Symbol
import dash_table.FormatTemplate as FormatTemplate
from dash.dependencies import Input, Output
import plotly.graph_objs as go 
import plotly.figure_factory as ff
import pandas as pd 
import pandas_datareader.data as web
import datetime as dt
import numpy as np
import json
from scipy.stats import norm

import sys
sys.path.append('..')

from app import app

xpto= ['BTC-USD','ETH-USD','XRP-USD','ETH-BTC']#,'BCH-USD','EOS-USD','USDT-USD','XLM-USD']
        
title = 'Historical Volatility'

pair_options = [{'label':pair,'value':pair} for pair in xpto]

layout = html.Div(style={'marginLeft':25,'marginRight':25},children = [
                html.Div(className='row',children=[
                    html.Div(className='four columns',children=
                            dcc.Dropdown(id='pairs-chosen',options = pair_options,value=[xpto[0]],
                            style={'width':'90%','display':'inline-block','border-color':'#cb1828'},
                            multi=True
                            )
                        ),
                    html.Div(className='two columns',children =html.H6('Start date:'),style={'width':'20%','display':'inline=block','textAlign':'center'}),

                    html.Div(className='two columns',children=   
                        dcc.DatePickerSingle(id='starting-date',
                                        min_date_allowed=dt.datetime(2012,1,1),
                                        max_date_allowed=dt.datetime.now(),
                                        initial_visible_month=dt.datetime(2017,1,1),
                                        date=dt.datetime(2018,12,1),
                                        placeholder='Starting Date:',
                                        day_size=35,style={'border-color':'#cb1828'}),
                        style={'textAlign':'left'}
                        ),
                    html.Div(className='three columns',children=
                        html.Div(id='last-update',style={'width': '40%','display':'inline-block','textAlign':'right'})
                        )
                    ]
                ),
                html.Div(className='row',children=
                    [
                    dcc.Graph(id='price-chart')                       
                    ]
                ),
                html.Div(className= 'row',                  
                        children=[dcc.Graph(id='realized-vol')]                 
                ),
                html.Div(className='row',children=[                 
                    html.Div(id='table',className='six columns'),                
                    html.Div(className='six columns',children=
                    [  
                        dcc.Graph(id='return-dist')
                    ])
                ]),
                html.Div(id='the-hvg-data',style={'display':'none'})  ,
                dcc.Interval(
                    id='hvg-interval-component',
                    interval=300*1000, # in milliseconds= 5 minutes
                    n_intervals=0
                    ),       
            ])

@app.callback([Output('the-hvg-data','children'),Output('last-update','children')],
            [Input('hvg-interval-component','n_intervals')])
def get_data(n):
    ''' Get the data and log the time of data update'''
    json_data={}
    now = dt.datetime.now().strftime("%Y-%m-%d")
    start_date=dt.datetime(2012,1,1)
    for pair in xpto:
        table = web.DataReader(pair,'yahoo',start_date,now).drop('Adj Close',axis=1)
        table['Rtn %'] = ((np.log(table["Close"]/table["Close"].shift(1))))
        table['RVol30'] = (np.sqrt(table['Rtn %'].apply(np.square).rolling(30).sum()*(365/30)))
        table['RVol10'] = (np.sqrt(table['Rtn %'].apply(np.square).rolling(10).sum()*(365/10)))
        table['1DVol'] = (table['Rtn %'].abs()*np.sqrt(365))
        table['High'] = table.apply(lambda x: max(x['High'],x['Open']),axis=1)
        table['Low'] = table.apply(lambda x: min(x['Low'],x['Open']),axis=1)
        table['pair'] = pair
        json_data[pair] = table.to_json(date_format='iso',orient='split')
    return json.dumps(json_data),'   Updated     {}'.format(dt.datetime.now().strftime("%Y-%m-%d   %H:%M"))

@app.callback([Output('price-chart','figure'),Output('realized-vol','figure'),Output('return-dist','figure'),Output('table','children')],
            [Input('the-hvg-data','children'),Input('pairs-chosen','value'),Input('starting-date','date')])
def update_page(json_data,pairs,start_date):
    # load data
    start_date=start_date.split(' ')[0]
    json_data=json.loads(json_data)
    data={key:pd.read_json(json_data[key],orient='split') for key in json_data}
    #This is the first pair chosen
    df=data[pairs[0]]
    df=df[df.index.get_loc(start_date):]
    name = df['pair'][0]
    # price chart
    if len(pairs) == 1:
        trace1 = go.Ohlc(x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],close=df['Close'],name='Price')
        df['EWMA']=df.Close.ewm(span=20).mean()
        df['Upper']=df.EWMA +2*(df.Close.rolling(20).std())
        df['Lower']=df.EWMA -2*(df.Close.rolling(20).std())
        trace2 = go.Scatter(x=df.index,y=df.EWMA,name='EWMA')
        trace3 = go.Scatter(x=df.index,y=df.Upper,name='Upper')
        trace4 = go.Scatter(x=df.index,y=df.Lower,name='Lower')
        layout = go.Layout(title=name,height=500,xaxis =dict(rangeslider=dict(visible=False)))
        price_figure = go.Figure(data=[trace1,trace2,trace3,trace4],layout=layout)
        df.drop(['EWMA','Upper','Lower'],axis=1,inplace=True)
    elif len(pairs)>1:
        cum_data={}
        for pair in data:
            df_pair=data[pair][data[pair].index.get_loc(start_date):]
            df_pair['Cum']=df_pair['Close']/df_pair['Close'][0]
            cum_data[pair]=df_pair
        cum=pd.concat((cum_data[k]['Cum'] for k in cum_data),axis=1)
        cum.columns = data.keys()
        cum2 = cum[pairs]
        traces = [go.Scatter(x=cum2.index,y=cum2[pair],name=pair) for pair in cum2.columns]
        price_figure = go.Figure(data=traces)
    # realized vol of the first pair chosen
    trace1 = go.Scatter(x=df.index,y=df['RVol10'],name = 'RVol10')
    trace2 = go.Scatter(x=df.index,y=df['RVol30'],name = 'RVol30')
    layout = go.Layout(title=name,height=500,yaxis={'tickformat':'.0%'})
    rvol_figure = go.Figure(data=[trace1,trace2],layout=layout)
    # distribution of returns
    hist_data=[data[pair][data[pair].index.get_loc(start_date):]['Rtn %'] for pair in pairs]
    ret_dist_fig = ff.create_distplot(hist_data,pairs,show_hist=False)  
    if len(pairs)==1:
        std = hist_data[0].std()
        x = np.linspace(hist_data[0].min(),hist_data[0].max(),100)
        y = norm.pdf(x,scale=std)
        ret_dist_fig.add_scatter(x=x,y=y,name='Normal',line=dict(width=.5))
    # historical table of the first pair chosen
    df.drop(['pair'],axis=1,inplace=True)
    df.index.name=name
    decimals = pd.Series([2]*4+[0]+[4]*4,index=df.columns)
    df = df.round(decimals)
    df=df.reset_index().tail(40).sort_values(pairs[0],ascending=False)
    hist_table = dash_table.DataTable(
        data=df.to_dict('rows'),
        columns =[{'id': c,'name':c} for c in df.columns[:-4]]+ [{'id':c,'name':c,'type':'numeric',
                                            'format':FormatTemplate.percentage(2)}for c in df.columns[-4:]],
        style_table={'overflowY': 'scroll','maxHeight':'500 px'},
        style_header= {'backgroundColor':'lightgrey','fontWeight':'bold'},
        style_cell = {'textAlign':'center'},
        n_fixed_rows=1
    )
    return price_figure,rvol_figure,ret_dist_fig,hist_table
