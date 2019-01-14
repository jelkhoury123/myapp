import numpy as np
import pandas as pd
import sched,time, datetime

from models import BSprice, BSgreeks, BSiv, otm
from models import vvv, vvv_fitter

import ccxt

s=sched.scheduler(time.time,time.sleep)

def save_fits(sc):
    save_fits.counter +=1
    ex = ccxt.deribit()
    ctcs=list(ex.load_markets().keys())

    O = ['C','P']
    prices=pd.DataFrame(ctcs,columns=['Name']).sort_values('Name').reset_index(drop=True)
    prices['Und']=prices['Name'].apply(lambda x : x.split('-')[0])
    prices['Ins']=prices['Name'].apply(lambda x: x.split('-')[-1] if len(x.split('-'))>3 else 'F')
    prices['Expiry']=prices['Name'].apply(lambda x: pd.to_datetime(x.split('-')[1]).strftime("%Y-%m-%d")
                     if not x.split('-')[1].startswith('P') else np.NaN)
    prices['T']=prices.apply(lambda x: round((pd.to_datetime(x['Expiry'])-pd.to_datetime('today')).total_seconds()/(365*24*60*60),4) ,axis=1)
    prices['Strike']=prices.apply(lambda x: int(x['Name'].split('-')[-2]) if x['Ins'] in O else np.nan,axis=1)
    prices=prices.sort_values(['T','Strike']).reset_index(drop=True)
    prices=prices[prices['T']>1/365]

    prices['all']=prices['Name'].apply(lambda x :ex.fetch_ticker(x))
    now=pd.datetime.now().strftime('%Y-%m-%d %H:%M')

    sheets=prices.copy()
    sheets['info']=sheets['all'].apply(lambda x:x['info'])
    sheets['uPx']=sheets.apply(lambda x: x['info']['uPx'] if x['Ins'] in O else np.nan ,axis=1)
    sheets['uIx']=sheets.apply(lambda x: x['info']['uIx'] if x['Ins'] in O else np.nan ,axis=1)
    sheets['iR']=sheets.apply(lambda x: x['info']['iR'] if x['Ins'] in O else np.nan ,axis=1)
    sheets['Bid']=sheets['info'].apply(lambda x: x['bidPrice'])
    sheets['Ask']=sheets['info'].apply(lambda x: x['askPrice'])
    sheets['Mid']=sheets['info'].apply(lambda x: x['midPrice'])
    sheets['OI']=sheets['info'].apply(lambda x: x['openInterest'])
    sheets['Volume']=sheets['info'].apply(lambda x: x['volume'])
    sheets['OTM']=sheets.apply(lambda x: otm(x['uPx'],x['Strike'],x['Ins'],.001),axis=1)
    sheets['Volume'].replace('',0,inplace=True)
    sheets.replace('', np.nan, inplace=True)
    sheets.drop(['all','info'],axis=1,inplace=True)

    futures=sheets[sheets['Ins']=='F'].drop(['Strike','uIx','uPx','iR','OTM'],axis=1)
    futures.reset_index(drop=True,inplace=True)
    options=sheets[sheets['OTM']==True].dropna().drop('OTM',axis=1)
    options.reset_index(inplace=True,drop=True)

    options['BidVol']=options.apply(lambda x : BSiv(x['uPx'], x['Strike'], x['T'], x['iR'], x['Bid']*x['uPx'], option = x['Ins']) ,axis=1)
    options['MidVol']=options.apply(lambda x : BSiv(x['uPx'], x['Strike'], x['T'], x['iR'], x['Mid']*x['uPx'], option = x['Ins']) ,axis=1)
    options['AskVol']=options.apply(lambda x : BSiv(x['uPx'], x['Strike'], x['T'], x['iR'], x['Ask']*x['uPx'], option = x['Ins']) ,axis=1)
    options=options[(options['AskVol']-options['BidVol'])<.3]
    options['Bid$']=options['Bid']*options['uPx']
    options['Mid$']=options['Mid']*options['uPx']
    options['Ask$']=options['Ask']*options['uPx']
    options['Vega']=options.apply(lambda x : BSgreeks(x['uPx'], x['Strike'], x['T'], 0, x['MidVol'], option = 'C')[2]*.01,axis=1)

    optmats=[]
    T=[]
    Expiries=[]
    Ref=[]
    fitparams=[]
    vol_q=[]
    price_q=[]

    for i in options['T'].unique():
        mat=options[options['T']==i].copy()
        optmats.append(mat)
        t=mat.iloc[0]['T']
        Expiry=mat.iloc[0]["Expiry"]
        Expiries.append(Expiry)
        T.append(t)
        ref=mat.iloc[0]['uPx']
        Ref.append(ref)
        try:
            fit=[round(i,4) for i in vvv_fitter(mat['Strike'],mat['MidVol'],mat['Vega'],t,ref)]
            fitparams.append(fit)
            fitmat= vvv(mat['Strike'][:],t,ref,*fit)
            mat['Fit']=fitmat
            vol_q.append(round(np.abs(mat['Fit']-mat['MidVol']).mean(),4))
            mat['TV']=mat.apply(lambda x: BSprice (ref, x['Strike'], t, x['iR'], x['Fit'], option = x['Ins']),axis=1)
            price_q.append(round(np.abs(mat['TV']-mat['Mid$']).mean(),4))
        except:
            fit=['nofit']
            fitparams.append(fit)
            vol_q.append(None)
            price_q.append(None)
            continue

    fitparams=pd.DataFrame(fitparams,columns=['Sigma','Skew','Kurt','Alpha'],index=Expiries)
    fitparams['Ref']=Ref
    fitparams.columns.name=now
    fitparams['vol_q']=vol_q
    fitparams['price_q']=price_q

    opt=optmats[0]
    for i in range(len(optmats)):
        opt=opt.merge(optmats[i],'outer')
    opt.columns.name=now

    import pymongo
    import subprocess

    client=pymongo.MongoClient()
    db=client.fits
    vvv_fits=db.vvv_fits

    vvv_fits.insert_one({"Time":now,"Surface":fitparams.to_json(),"Options":opt.to_json(),"Futures":futures.to_json()})
    print ("One save made at:", now)
    if save_fits.counter % 12 == 0 :
        subprocess.call(["mongodump","--db","fits","--collection","vvv_fits"])
    s.enter(300,1,save_fits,(sc,))
save_fits.counter=0
s.enter(300,1,save_fits,(s,))
while True:
    try:
        s.run()
    except:
        continue
