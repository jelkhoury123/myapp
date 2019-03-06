#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.stats as si
from scipy.interpolate import interp1d
import quandl
from datetime import datetime , timedelta

N = si.norm.cdf
Np = lambda x:np.exp(-x**2/2)/np.sqrt(2*np.pi)

def BSprice (F, K, T, r, sigma, option = 'C'):
    
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = (np.log(F / K) - 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    
    if option == 'C':
        price = np.exp(-r * T) * (F * N(d1) - K *  N(d2))
    if option == 'P':
        price = np.exp(-r * T) *(K *  N(-d2) - F * N(-d1))

    return price

def BSgreeks(F, K, T, r, sigma, option = 'C'):
    
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    
    if option == 'C':
        delta = N(d1)
    if option == 'P':
        delta = -N(-d1)
    gamma = np.exp(r * T)*Np(d1)/(F*sigma*np.sqrt(T))
    vega = F * np.exp(-r * T) * Np(d1)*np.sqrt(T)
    theta = (-1/2*gamma*sigma**2*(F*np.exp(-r*T))**2)/365    
    return delta,gamma,theta,vega

def BSiv(F, K, T, r, premium, option = 'C'):
    
    intrinsic_value = max(F-K,0) if option == 'C' else max(K-F,0)
    time_value = premium - intrinsic_value
    if time_value <=0:
        return 0
    else:
        sigma = 2.
        price = BSprice(F,K,T,r,sigma,option)
        dist = (premium-price)
        while np.abs(dist) >= 1e-7:
            vega = BSgreeks(F,K,T,r,sigma,option)[-1]
            adj = dist/vega
            sigma += adj
            price = BSprice(F,K,T,r,sigma,option)
            dist = (premium-price)

        return sigma


def otm(ref,strike,option,cutoff):
    
    if (strike>(1-cutoff)*ref and option == 'C'):
        return True
    elif(strike<(1+cutoff)*ref and option == 'P'):
        return True
    else:
        return False


def vvv (K,T,F,sigma,skew,kurt,alpha):
    
    # K is an iterable of strikes or a float/int
    # returns a list of vols of the same length than K or a float
    
    def vvvscalar(K,T,F,sigma,skew,kurt,alpha):
        
            sigbar= sigma*(1+1/4*kurt*T*sigma**alpha)
            m = K/F

            def f(x) :

                t1 = x**(4-alpha)
                t2 = kurt*T/4*x**4
                t3 = x**(3-alpha)*(sigbar + skew /np.sqrt(T)*(m-1))
                t4 = kurt/T* np.log(m)**2
                return t1+t2-t3-t4

            def fprime(x):

                t1 = (4-alpha)*x**(3-alpha)
                t2 = kurt*T*x**3
                t3 = (3-alpha)*x**(2-alpha)*(sigbar+skew/np.sqrt(T)*(m-1))
                return t1+t2-t3

            from scipy.optimize import newton as nt
            return nt(f,5,fprime)
    
    try:
        vols=[]
        for k in K:
            vols.append(vvvscalar(k,T,F,sigma,skew,kurt,alpha))
        return vols
    except:
        return vvvscalar(K,T,F,sigma,skew,kurt,alpha)



def vvv_fitter(K,vol,vega,t,ref):

    from scipy.optimize import curve_fit
    init_values=[.69,-.05,.03,0]
    bounds = ([.1,-.4,0.001,0], [4.,.5,2,4])
    vvv_mat = lambda K, sigma, skew, kurt, alpha : vvv(K,t,ref,sigma,skew,kurt,alpha)
    return curve_fit(vvv_mat,K,vol,sigma =1/vega,absolute_sigma=True,bounds=bounds,p0 = init_values)[0]

def get_usd_curve():
    yesterday = datetime.now() - timedelta(days=1)
    usd_zrc=quandl.get("USTREASURY/YIELD",start_date=yesterday).iloc[-1]
    usd_curve=usd_zrc.reset_index()
    usd_curve['T']=usd_curve['index'].apply(lambda x: int(x.split(' ')[0])*30/365 if x.split(' ')[1]=='MO' else int(x.split(' ')[0]))
    usd_curve.drop('index', axis=1 , inplace=True)
    usd_curve.columns=['rate','T']
    usd_curve['rate']/=100
    rate_function=interp1d(usd_curve['T'],usd_curve['rate'],fill_value='extrapolate')
    return rate_function


