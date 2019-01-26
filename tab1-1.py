import dash
import dash_core_components as dcc 
import dash_html_components as html 
import dash_table  
from dash.dependencies import Input, Output
import plotly.graph_objs as go 
import plotly.figure_factory as ff
import pandas as pd 
import pandas_datareader.data as web
import datetime as dt
import numpy as np
import json
from scipy.stats import norm


xpto= ['BTC-USD','ETH-USD','XRP-USD','XMR-USD','BCH-USD','EOS-USD','USDT-USD','XLM-USD']
        
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
app.title = 'Historical Volatility'

pair_options = [{'label':pair,'value':pair} for pair in xpto]

app.layout = html.Div(className='row',style= {'backgroundColor':'#FFFFFF'},children=
            [
                html.Div(className='row',children=[
                    html.Div(className='three columns',children=
                            dcc.Dropdown(id='pairs-chosen',options = pair_options,value=[xpto[0]],style={'width':'90%','display':'inline-block'},
                            multi=True
                            )
                        ),
                    html.Div(className='two columns',children ='Start date:',style={'width':'25%','display':'inline=block','textAlign':'right'}),

                    html.Div(className='two columns',children=   
                        dcc.DatePickerSingle(id='starting-date',
                                        min_date_allowed=dt.datetime(2012,1,1),
                                        max_date_allowed=dt.datetime.now(),
                                        initial_visible_month=dt.datetime(2017,1,1),
                                        date=dt.datetime(2018,1,1),
                                        placeholder='Starting Date:',
                                        day_size=35)
                        ),
                    html.Div(className='three columns',children=
                        html.Div(id='last-update',style={'width': '100%','display':'inline-block','textAlign':'right'})
                        )
                    ]
                ),
                html.Div(
                    [
                    dcc.Graph(id='price-chart')                       
                    ]
                ),
                html.Div(                   
                        children=[dcc.Graph(id='realized-vol')]                 
                ),
                html.Div(className='row',children=[                 
                    html.Div(id='table',className='six columns'),                
                    html.Div(className='six columns',children=
                    [  
                        dcc.Graph(id='return-dist')
                    ])
                ]),
                html.Div(id='the-data',style={'display':'none'})  ,
                dcc.Interval(
                    id='interval-component',
                    interval=300*1000, # in milliseconds= 5 minutes
                    n_intervals=0
                    )             
            ])

@app.callback(Output('the-data','children'),
            [Input('interval-component','n_intervals')])
def get_data(n):
    json_data={}
    now = dt.datetime.now().strftime("%Y-%m-%d")
    start_date=dt.datetime(2012,1,1)
    for pair in xpto:
        table = web.DataReader(pair,'yahoo',start_date,now).drop('Adj Close',axis=1).round(2)
        table['Rtn %'] = ((np.log(table["Close"]/table["Close"].shift(1)))*100).round(2)
        table['RVol30 %'] = (np.sqrt(table['Rtn %'].apply(np.square).rolling(30).sum()*(365/30))).round(2)
        table['RVol10 %'] = (np.sqrt(table['Rtn %'].apply(np.square).rolling(10).sum()*(365/10))).round(2)
        table['1DVol %'] = (table['Rtn %'].abs()*np.sqrt(365)).round(2)
        table['High'] = table.apply(lambda x: max(x['High'],x['Open']),axis=1)
        table['Low'] = table.apply(lambda x: min(x['Low'],x['Open']),axis=1)
        table['pair'] = pair
        json_data[pair] = table.to_json(date_format='iso',orient='split')
    return json.dumps(json_data)

@app.callback(Output('last-update','children'),[Input('interval-component','n_intervals')])
def last_update(n):
    return '   Updated     {}'.format(dt.datetime.now().strftime("%Y-%m-%d   %H:%M"))

@app.callback(Output('return-dist','figure'),
            [Input('the-data','children'),Input('pairs-chosen','value'),Input('starting-date','date')])
def dist_returns(json_data,pairs,start_date):
    start_date=start_date.split(' ')[0]
    json_data=json.loads(json_data)
    data={key:pd.read_json(json_data[key],orient='split') for key in json_data}
    hist_data=[data[pair][data[pair].index.get_loc(start_date):]['Rtn %'] for pair in pairs]
    fig = ff.create_distplot(hist_data,pairs,show_hist=False)  
    if len(pairs)==1:
        std=hist_data[0].std()
        x=np.linspace(hist_data[0].min(),hist_data[0].max(),100)
        y=norm.pdf(x,scale=std)
        fig.add_scatter(x=x,y=y,name='Normal',line=dict(width=.5))
    
    return fig

@app.callback(Output('realized-vol','figure'),
            [Input('the-data','children'),Input('pairs-chosen','value'),Input('starting-date','date')])
def plot_rvol(json_data,pairs,start_date):
    start_date=start_date.split(' ')[0]
    json_data=json.loads(json_data)
    data={key:pd.read_json(json_data[key],orient='split') for key in json_data}
    df=data[pairs[0]]
    df=df[df.index.get_loc(start_date):]
    name = df['pair'][0]
    trace1 = go.Scatter(x=df.index,y=df['RVol10 %'],name = 'RVol10 %')
    trace2 = go.Scatter(x=df.index,y=df['RVol30 %'],name = 'RVol30 %')
    layout = go.Layout(title=name,height=500)
    return go.Figure(data=[trace1,trace2],layout=layout)

@app.callback(Output('price-chart','figure'),
            [Input('the-data','children'),Input('pairs-chosen','value'),Input('starting-date','date')])
def prices_chart(json_data,pairs,start_date):
    start_date=start_date.split(' ')[0]
    json_data=json.loads(json_data)
    data={key:pd.read_json(json_data[key],orient='split') for key in json_data}
    if len(pairs)==1:
        df=data[pairs[0]]
        df=df[df.index.get_loc(start_date):]
        name = df['pair'][0]
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
        return go.Figure(data=[trace1,trace2,trace3,trace4],layout=layout)
    elif len(pairs)>1:
        cum_data={}
        for pair in data:
            df=data[pair][data[pair].index.get_loc(start_date):]
            df['Cum']=df['Close']/df['Close'][0]
            cum_data[pair]=df
        cum=pd.concat((cum_data[k]['Cum'] for k in cum_data),axis=1)
        cum.columns = data.keys()
        cum2 = cum[pairs]
        traces = [go.Scatter(x=cum2.index,y=cum2[pair],name=pair) for pair in cum2.columns]
        return go.Figure(data=traces)
          

@app.callback(Output('table','children'),
            [Input('the-data','children'),Input('pairs-chosen','value')])
def generate_table(json_data,pairs,max_rows =20):
    json_data=json.loads(json_data)
    data={key:pd.read_json(json_data[key],orient='split') for key in json_data}
    df=data[pairs[0]].round(2)
    name = df['pair'][0]
    df.drop('pair',axis=1,inplace=True)
    df.index.name=name
    df=df.reset_index().tail(max_rows).sort_values(pairs[0],ascending=False)
    return dash_table.DataTable(
        data=df.to_dict('rows'),
        columns =[{'id': c,'name':c} for c in df.columns],
        style_table={
            'maxHeight': '500',
            'overflowY': 'scroll'
        },
        style_header= {'backgroundColor':'lightgrey','fontWeight':'bold'},
        style_cell = {'textAlign':'center'},
        n_fixed_rows=1
    )
if __name__ == '__main__':
    app.run_server(debug=True,port = 8051)