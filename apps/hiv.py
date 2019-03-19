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
import pymongo
import os

import sys
sys.path.append('..')

from app import app

os.system('mongorestore --drop -d awsfits --gzip fits')

client=pymongo.MongoClient()
db=client.awsfits
vvv_fits=db.vvv_fits
load =[i for i in db.vvv_fits.find()]

fits= []
futs= []
opts= []
tstamps=[]
for i in range(len(load)):
    name=load[i]['Time']
    tstamps.append(name)
    fit = pd.DataFrame(json.loads(str(load[i]['Surface'])))
    fit.reset_index(inplace=True)
    fit['index'] = fit['index'].apply(lambda x: x.split(' ')[0])
    fit.set_index('index',drop = True,inplace=True)
    fit.index.name=name
    fits.append(fit)
    futures=pd.DataFrame(json.loads(str(load[i]['Futures'])))
    futures['Expiry']=futures['Expiry'].apply(lambda x:x.split(' ')[0])
    futures.index.name=name
    futs.append(futures)
    
    options = pd.DataFrame(json.loads(str(load[i]['Options'])))
    options['Expiry']=options['Expiry'].apply(lambda x:x.split(' ')[0])
    options['VolSpread']= options['AskVol']-options['BidVol']
    options.index.name=name
    opts.append(options)
    
    fits[i]['VolSpread']=opts[i].groupby('Expiry').mean()['VolSpread']

palette=dict(zip(fits[-1].columns,['#50AEEC','#FF6347','#228B22','#DAA520','#708090','#C0C0C0','#F4A460','#D2691E']))

title='Historical Implied Vols'

ix = [(pd.to_datetime(fit.index.name,utc=True),fit['Ref'][0]) for fit in fits]

layout = html.Div(children=[
    dcc.Graph(id = 'base',
        figure = go.Figure(
            data=[go.Scatter(x=pd.DataFrame(ix)[0],y=pd.DataFrame(ix)[1])],
            layout= go.Layout(title='BTC USD',hovermode='closest'))
            ),
    html.Div(className='row',children=[
            html.Div(className='six columns',children=[
                html.H5('Fit Table'),
                html.H5(' . '),
                html.Div(dash_table.DataTable(id='hiv-fit-table',
                        style_header= {'backgroundColor':'lightgrey','fontWeight':'bold'},
                        style_cell = {'textAlign':'center'}
    )),
                html.H5(' . '),
                html.H5('Last Fit:  '+tstamps[-1]),
                html.H5('Number of fits:     '+str(len(load)))
            ]),
            html.Div(className='six columns',id='hiv-TS-chart',children=[dcc.Graph(id='hiv-TS')])]),
    html.Div(className='row',children=[
            html.Div(id='skew-plot',className='six columns'),
            html.Div(id='hist-plot',className='six columns'),
            ]),
    dcc.Link('hvg',href='/apps/hvg'),
    dcc.Link('skew',href='/apps/skew'),
    dcc.Link('pricer',href='/app/pricer'),
    dcc.Link('order_book',href='/app/order_book'),
    dcc.Link('futures',href='/app/futures'), 
])

def skewplot(timeindex,expindex):
    
    optmats=opts[timeindex]
    fitparams=fits[timeindex]
    mat=fitparams.index[expindex]
    optchart = optmats[optmats['Expiry']==mat].sort_values('Strike')
    title = str(fitparams.loc[mat])
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
                    title='<b>' + mat+'</b>',
                    titlefont=dict(
                        family='Courier New, monospace',
                        size=18,
                        color='#7f7f7f')
                )
        )
    return dcc.Graph(figure=go.Figure(data=[bid,mid,fit,ask],layout=layout))


@app.callback(Output('hiv-fit-table','columns'),
            [Input('base','clickData')])
def update_fit_table(clickdata):
    timestamp = int(clickdata['points'][0]['pointIndex'])
    table = fits[timestamp].reset_index().round(4)
    table_columns = [{'id': c,'name':c} for c in table.columns]
    for dic in table_columns:
        if dic['id'] in ['Sigma','Skew','Kurt','vol_q','VolSpread']:
            dic['type'] = 'numeric'
            dic['format']= FormatTemplate.percentage(2)
    return table_columns 

@app.callback(Output('hiv-fit-table','data'),
            [Input('base','clickData')])
def fit_table_data(clickdata):
    timestamp = int(clickdata['points'][0]['pointIndex'])
    table = fits[timestamp].reset_index().round(4)
    return table.to_dict('rows')

@app.callback([Output('hiv-TS','figure'),Output('skew-plot','children'),Output('hist-plot','children')],
        [Input('base','clickData'),Input('hiv-fit-table','active_cell')])
def TS_chart(clickdata,active_cell):
    timeindex = int(clickdata['points'][0]['pointIndex'])
    fitparams = fits[timeindex].round(4)
    ts_parameter=fitparams.columns[active_cell[-1]-1]
    ts_plot = go.Figure(data= [go.Scatter(x=fitparams.index,y=fitparams[ts_parameter],
                    name =ts_parameter,line=dict(color=palette[ts_parameter]))],
                    layout=go.Layout(title='<b>'+ ts_parameter + '</b>' + '     Term Structure'+'      '+ str(clickdata['points'][0]['x'])))
    expindex = active_cell[0]
    skew_plot = skewplot(timeindex,expindex)
    param = fitparams.columns[active_cell[1]-1]
    mat = fitparams.index[active_cell[0]]
    hist=[]
    for fit in fits:
        try:
            hist.append((pd.to_datetime(fit.index.name,utc=True),fit[param][mat]))
        except:
            pass
    scatter=go.Scatter(x=pd.DataFrame(hist)[0],y=pd.DataFrame(hist)[1],line=dict(color=palette[param],width=.85))
    layout=go.Layout(title=mat +'     '+param)
    hist_graph = dcc.Graph(figure=go.Figure(data=[scatter],layout=layout))
    return ts_plot, skew_plot, hist_graph

