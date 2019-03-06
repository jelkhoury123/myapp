import numpy as np
import pandas as pd

from models import BSprice, BSgreeks, BSiv, otm, vvv, vvv_fitter
from deribit_api import RestClient

def get_data():
    '''Get and process and fit Deribit market data'''

    client = RestClient()
    alld = pd.DataFrame(client.getinstruments())
    pullderibit = pd.DataFrame(client.getsummary('all'))
    allderibit = alld.merge(pullderibit,on='instrumentName')
    columns = ['instrumentName','baseCurrency','optionType','expiration','strike','uPx','uIx','iR','bidPrice','askPrice','midPrice','openInterest','volume']
    newcolumns = ['Name','Und','Ins','Expiry','Strike','uPx','uIx','iR','Bid','Ask','Mid','OI','Volume']
    prices = allderibit[columns]
    prices.columns = newcolumns
    prices = prices.sort_values(['Expiry','Strike','Ins']).reset_index(drop=True)[:-1]
    now = pd.datetime.now().strftime('%Y-%m-%d %H:%M')
    prices['T'] = prices.apply(lambda x: round((pd.to_datetime(x['Expiry'])-pd.to_datetime('now')).total_seconds()/(365*24*60*60),4) ,axis=1)
    prices['Expiry'] = prices["Expiry"].apply(lambda x: x.split(' ')[0])
    prices['Volume'].replace('',0,inplace=True)
    prices.replace('',np.nan,inplace=True)
    prices['Ins'] = prices['Ins'].apply(lambda x : 'C' if x =='call' else ('P' if x =='put' else 'F'))
    prices['OTM'] = prices.apply(lambda x :otm(x['uPx'],x['Strike'],x['Ins'],.001),axis=1)

    sheets = prices.copy()

    futures = sheets[sheets['Ins']=='F'].drop(['Strike','uIx','uPx','iR','OTM'],axis=1)
    futures.reset_index(drop=True,inplace=True)
    options = sheets[sheets['OTM']==True].dropna().drop('OTM',axis=1)
    options.reset_index(inplace=True,drop=True)

    options['BidVol'] = options.apply(lambda x : BSiv(x['uPx'], x['Strike'], x['T'], x['iR'], x['Bid']*x['uPx'], option = x['Ins']) ,axis=1)
    options['MidVol'] = options.apply(lambda x : BSiv(x['uPx'], x['Strike'], x['T'], x['iR'], x['Mid']*x['uPx'], option = x['Ins']) ,axis=1)
    options['AskVol'] = options.apply(lambda x : BSiv(x['uPx'], x['Strike'], x['T'], x['iR'], x['Ask']*x['uPx'], option = x['Ins']) ,axis=1)
    options['VolSpread']= options['AskVol']-options['BidVol']
    #options = options[options['AskVol']-options['BidVol']<.3]
    options['Bid$'] = options['Bid']*options['uPx']
    options['Mid$'] = options['Mid']*options['uPx']
    options['Ask$'] = options['Ask']*options['uPx']
    options['Vega'] = options.apply(lambda x : BSgreeks(x['uPx'], x['Strike'], x['T'], 0, x['MidVol'], option = 'C')[2]*.01,axis=1)

    optmats = []
    T = []
    Expiries = []
    Ref = []
    fitparams = []
    volfitquality = []
    pricefitquality = []

    for i in options['T'].unique():
        mat = options[options['T']==i].copy()
        optmats.append(mat)
        t = mat.iloc[0]['T']
        Expiry = mat.iloc[0]['Expiry']
        Expiries.append(Expiry)
        T.append(t)
        ref = mat.iloc[0]['uPx']
        Ref.append(ref)
        try:
            fit=[round(i,4) for i in vvv_fitter(mat['Strike'][:],mat['MidVol'][:],mat['Vega'][:],t,ref)]
            fitparams.append(fit)
            mat['Fit'] = mat['Strike'].apply(lambda x : vvv(x,t,ref,*fit))
            volfitquality.append(round(np.abs(mat['Fit']-mat['MidVol']).mean(),4))
            mat['TV'] = mat.apply(lambda x: BSprice (ref, x['Strike'], t, x['iR'], x['Fit'], option = x['Ins']),axis=1)
            pricefitquality.append(round(np.abs(mat['TV']-mat['Mid$']).mean(),4))
        except:
            fit=['nofit']
            fitparams.append(fit)
            volfitquality.append(None)
            pricefitquality.append(None)
            continue
            
    fitparams = pd.DataFrame(fitparams,columns=['Sigma','Skew','Kurt','Alpha'],index=Expiries)
    fitparams['Ref']=Ref
    fitparams.columns.name = now
    fitparams['vol_q'] = volfitquality
    fitparams['price_q'] = pricefitquality
    fitparams['VolSpread'] = options.groupby('Expiry').mean()['VolSpread']
    data = optmats + [fitparams] + [futures]
    return data