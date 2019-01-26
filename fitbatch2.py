import numpy as np
import pandas as pd
import sched, time, datetime, os

from models import BSprice, BSgreeks, BSiv, otm
from models import vvv, vvv_fitter

from deribit_api2 import get_data
import pymongo
#import subprocess


s=sched.scheduler(time.time,time.sleep)

def save_fits(sc):
    save_fits.counter +=1
    data = get_data()
    futures= data[-1]
    fitparams = data[-2]
    optmats = data[:-2]
  
    opt=pd.concat(optmats)
    now = fitparams.columns.name
    opt.columns.name=now

    client=pymongo.MongoClient()
    db=client.fits
    vvv_fits=db.vvv_fits

    vvv_fits.insert_one({"Time":now,"Surface":fitparams.to_json(),"Options":opt.to_json(),"Futures":futures.to_json()})
    print ("One save made at:", now)
    if save_fits.counter % 12 == 0 :
        #subprocess.call(["mongodump","--db","fits","--collection","vvv_fits"])
        try:
            os.system('mongodump --db fits --collection vvv_fits --gzip ')
            print('dumped 12 additional fits')
        except:
            print('dump failed')
            pass
    s.enter(300,1,save_fits,(sc,))
save_fits.counter=0
s.enter(300,1,save_fits,(s,))
while True:
    try:
        s.run()
    except:
        continue
