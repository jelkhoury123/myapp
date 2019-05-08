import numpy as np
import pandas as pd

from models import BSprice, BSgreeks, BSiv, otm, vvv, vvv_fitter
import requests
import datetime


'''Deribit Rest API'''


'''I - Market data Public API'''

def get_summary_by_currency(ccy,kind=None):
    payload={'currency':ccy,'kind':kind}
    r = requests.get('https://deribit.com/api/v2/public/get_book_summary_by_currency',params=payload)
    return pd.DataFrame((r.json())['result'])

def get_summary_by_instrument(ins):
    payload={'instrument_name':ins}
    r = requests.get('https://deribit.com/api/v2/public/get_book_summary_by_instrument',params=payload)
    return pd.DataFrame((r.json())['result'])

def get_contract_size(ins):
    payload={'instrument_name':ins}
    r = requests.get('https://deribit.com/api/v2/public/get_contract_size',params=payload)
    return pd.DataFrame([(r.json())['result']],index=[ins])

def get_currencies():
    r = requests.get('https://deribit.com/api/v2/public/get_currencies')
    return pd.DataFrame(r.json()['result'])

def get_funding_chart_data(ins,length):
    payload={'instrument_name':ins}
    r = requests.get('https://www.deribit.com/api/v2/public/get_funding_chart_data',params=payload)
    return pd.DataFrame([(r.json())['result']],index=[ins])

def get_historical_volatility(ccy):
    payload={'currency':ccy}
    r = requests.get('https://www.deribit.com/api/v2/public/get_historical_volatility',params=payload)
    return pd.DataFrame(r.json()['result'])

def get_index(ccy):
    payload={'currency':ccy}
    r = requests.get('https://www.deribit.com/api/v2/public/get_index',params=payload)
    return pd.DataFrame([r.json()['result']])

def get_instruments(currency,kind=None,expired='false'):
    payload={'currency':currency,'kind':kind,'expired':expired}
    r = requests.get('https://deribit.com/api/v2/public/get_instruments',params=payload)
    return pd.DataFrame((r.json())['result'])

def get_order_book(ins,depth=100):
    payload={'instrument_name':ins,'depth':depth}
    r = requests.get('https://www.deribit.com/api/v2/public/get_order_book',params=payload)
    bids = r.json()['result']['bids']
    asks = r.json()['result']['asks']
    return {'bids':bids,'asks':asks}

def get_trade_volumes():
    r = requests.get('https://www.deribit.com/api/v2/public/get_trade_volumes')
    return pd.DataFrame(r.json()['result'])

def ticker(ins):
    payload={'instrument_name':ins}
    r = requests.get('https://www.deribit.com/api/v2/public/ticker',params=payload)
    return pd.DataFrame([r.json()['result']],index=[ins])

'''II -Trade Private API'''

def buy(auth,ins,amount,price,params={}):
    payload = dict(instrument_name=ins,amount=amount,price=price,**params)
    r = requests.get('https://www.deribit.com/api/v2/private/buy',params=payload,auth=auth)
    return r

def sell(auth,ins,amount,price,params={}):
    payload = dict(instrument_name=ins,amount=amount,price=price,**params)
    r = requests.get('https://www.deribit.com/api/v2/private/buy',params=payload,auth=auth)
    return r

def edit(auth,order_id,amount,price,params={}):
    payload = dict(order_id=order_id,amount=amount,price=price,**params)
    r = requests.get('https://www.deribit.com/api/v2/private/edit',params=payload,auth=auth)
    return r

def cancel(auth,order_id):
    payload = dict(order_id=order_id)
    r = requests.get('https://www.deribit.com/api/v2/private/cancel',params=payload,auth=auth)
    return r

def cancel_all(auth):
    r = requests.get('https://www.deribit.com/api/v2/private/cancel_all',auth=auth)
    return r 

def cancel_all_by_currency(auth,ccy,kind=None,type=None):
    payload= dict(currency=ccy,kind=kind,type=type)
    r = requests.get('https://www.deribit.com/api/v2/private/cancel_all_by_currency',params=payload,auth=auth)
    return r 

def cancel_all_by_instrument(auth,ins,type=None):
    payload= dict(instrument_name=ins,type=type)
    r = requests.get('https://www.deribit.com/api/v2/private/cancel_all_by_instrument',params=payload,auth=auth)
    return r 

def close_position(auth,ins,price,type='limit'):
    payload= dict(instrument_name=ins,type=type,price=price)
    r = requests.get('https://www.deribit.com/api/v2/private/close_position',params=payload,auth=auth)
    return r

def get_open_orders_by_currency(auth,ccy,kind=None,type=None):
    payload= dict(currency=ccy,kind=kind,type=type)
    r = requests.get('https://www.deribit.com/api/v2/private/get_open_orders_by_currency',params=payload,auth=auth)
    return r 

def get_open_orders_by_instrument(auth,ins,type=None):
    payload= dict(instrument_name=ins,type=type)
    r = requests.get('https://www.deribit.com/api/v2/private/get_open_orders_by_instrument',params=payload,auth=auth)
    return r
    
def get_order_history_by_currency(auth,ccy,kind=None,count= 20,offset=0,include_old='false',include_unfilled='true'):
    payload= dict(currency=ccy,kind=kind,count= count,offset=offset,include_old=include_old,include_unfilled=include_unfilled)
    r = requests.get('https://www.deribit.com/api/v2/private/get_order_history_by_currency',params=payload,auth=auth)
    return r 

def get_order_history_by_instrument(auth,ins,count= 20,offset=0,include_old='false',include_unfilled='true'):
    payload= dict(instrument_name=ins,count= count,offset=offset,include_old=include_old,include_unfilled=include_unfilled)
    r = requests.get('https://www.deribit.com/api/v2/private/get_order_history_by_instrument',params=payload,auth=auth)
    return r 

def get_account_summary(auth,ccy,extended=None):
    payload = dict(currency=ccy,extended=extended)
    r= requests.get('https://www.deribit.com/api/v2/private/get_account_summary',params = payload, auth =auth)
    return r 

'''III -Get my option prices sheet'''

def get_prices(currency,kind=None,expired='false'):
    instruments=get_instruments(currency,kind,expired)
    instruments.set_index('instrument_name',inplace=True)
    summary_ins=get_summary_by_currency(currency,kind)
    summary_ins.set_index('instrument_name',inplace=True)
    allins=instruments.join(summary_ins,how ='outer',lsuffix='1')
    columns = ['base_currency','option_type','expiration_timestamp','strike','underlying_price',
            'underlying_index','interest_rate','bid_price',
            'ask_price','mid_price','open_interest','volume']
    prices=allins[columns].reset_index()
    return prices


def get_data(currency,kind=None,expired='false'):
    '''Get and process and fit Deribit market data
    for currency BTC or ETH'''

    newcolumns = ['Name','Und','Ins','Expiry','Strike','uPx','uIx','iR','Bid','Ask','Mid','OI','Volume']
    prices = get_prices(currency,kind,expired)
    prices.columns = newcolumns
    prices = prices.sort_values(['Expiry','Strike','Ins']).reset_index(drop=True)[:-1]
    prices['Expiry'] = prices["Expiry"].apply(lambda x: datetime.datetime.utcfromtimestamp(x/1000).strftime('%Y-%m-%d'))
    now = pd.datetime.utcnow().strftime('%Y-%m-%d %H:%M')
    prices['T'] = prices.apply(lambda x: round((pd.to_datetime(x['Expiry'])-pd.to_datetime(now)).total_seconds()/(365*24*60*60),4) ,axis=1)
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
    options['Bid$'] = options['Bid']*options['uPx']
    options['Mid$'] = options['Mid']*options['uPx']
    options['Ask$'] = options['Ask']*options['uPx']
    options['Vega'] = options.apply(lambda x : BSgreeks(x['uPx'], x['Strike'], x['T'], 0, x['MidVol'], option = 'C')[-1]*.01,axis=1)

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
            fit=[np.nan]
            fitparams.append(fit)
            volfitquality.append(None)
            pricefitquality.append(None)
            mat['Fit'] = mat['MidVol']
            mat['TV'] = mat.apply(lambda x: BSprice (ref, x['Strike'], t, x['iR'], x['Fit'], option = x['Ins']),axis=1)
            continue
            
    fitparams = pd.DataFrame(fitparams,columns=['Sigma','Skew','Kurt','Alpha'],index=Expiries)
    fitparams['Ref']=Ref
    fitparams.columns.name = now
    fitparams['vol_q'] = volfitquality
    fitparams['price_q'] = pricefitquality
    fitparams['VolSpread'] = options.groupby('Expiry').mean()['VolSpread']
    data = optmats + [fitparams] + [futures]
    return data