from __future__ import division
import pandas as pd
from lifelines.estimation import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from myfisher import *
from numpy import *

__all__ = ['estTECI',
            'estTEPH']

def estTECI(df, treatment_col='treated', event_col='disease2'):
    """Estimates treatment efficacy using cumulative incidence/attack rate.

    Ignores all censoring information. 
    
    TE = 1 - RR
    
    RR = (c1/N1) / (c0/N0)
    
    Parameters
    ----------
    df : pandas.DataFrame

    treatment_col : string
        Column in df indicating treatment.
    event_col : string
        Column in df indicating events (censored data are 0)
    covars : list
        List of other columns to include in Cox model as covariates.
    
    Returns
    -------
    est : float
        Estimate of vaccine efficacy
    ci : vector, length 2
        95% confidence interval, [LL, UL]
    pvalue : float
        P-value for H0: TE=0 from Fisher's Exact test"""
    
    a = ((df[treatment_col]==1) & (df[event_col]==1)).sum()
    b = ((df[treatment_col]==1) & (df[event_col]==0)).sum()
    c = ((df[treatment_col]==0) & (df[event_col]==1)).sum()
    d = ((df[treatment_col]==0) & (df[event_col]==0)).sum()
    rr = (a/(a+b)) / (c/(c+d))

    te = 1 - rr
    se = sqrt(b/(a*(a+b))+ d/(c*(c+d)))
    ci = 1 - exp(array([log(rr)+se*1.96, log(rr)-se*1.96]))
    """Use the two-sided p-value from a Fisher's Exact test for now, although I think there are better ways.
    Consider a Binomial Test"""
    pvalue = fisherTest([[a,b],[c,d]])[1]
    
    return te,ci,pvalue

def estTEPH(df, treatment_col='treated', duration_col='dx2', event_col='disease2',covars=[]):
    """Estimates treatment efficacy using proportional hazards (Cox model).
    
    Parameters
    ----------
    df : pandas.DataFrame
    
    treatment_col : string
        Column in df indicating treatment.
    duration_col : string
        Column in df indicating survival times.
    event_col : string
        Column in df indicating events (censored data are 0)
    covars : list
        List of other columns to include in Cox model as covariates.
    
    Returns
    -------
    est : float
        Estimate of vaccine efficacy
    ci : vector, length 2
        95% confidence interval, [LL, UL]
    pvalue : float
        P-value for H0: VE=0"""
    
    coxphf = CoxPHFitter()
    
    coxphf.fit(df[[treatment_col, duration_col, event_col]+covars],duration_col = duration_col,event_col = event_col)
    
    te = 1-exp(coxphf.hazards_.loc['coef',treatment_col])
    ci = 1-exp(coxphf.confidence_intervals_[treatment_col].loc[['upper-bound','lower-bound']])
    pvalue = coxphf._compute_p_values()[0]
    return te,ci,pvalue

def scoreci(x, n, conf_level=0.95):
    """Wilson’s confidence interval for a single proportion.
    Score CI based on inverting the asymptotic normal test
    using the null standard error
    
    Wilson, E.B. (1927) Probable inference, the law of succession, and statistical inference J. Amer.
    Stat. Assoc 22, 209–212

    Parameters
    ----------
    x : int
        Number of events
    n : int
        Number of trials/subjects
    conf_level : float
        Specifies coverage of the confidence interval (1 - alpha)

    Returns
    -------
    ci : array
        Confidence interval array [LL, UL]"""

    zalpha = abs(stats.norm.ppf((1-conf_level)/2))
    phat = x/n
    bound = (zalpha*((phat*(1-phat)+(zalpha**2)/(4*n))/n)**(1/2))/(1+(zalpha**2)/n)
    midpnt = (phat+(zalpha**2)/(2*n))/(1+(zalpha**2)/n)

    uplim = midpnt + bound
    lowlim = midpnt - bound

    return array([lowlim, uplim])

def myriskscoreci(x1,n1,x2,n2,conf_level=0.95):
    """Compute CI for the ratio of two binomial rates.
    
    Parameters
    ----------
    xi : int
        Number of events in group i
    ni : int
        Number of trials/subjects in group i
    conf_level : float
        Specifies coverage of the confidence interval (1 - alpha)

    Returns
    -------
    ci : array
        Confidence interval array [LL, UL]"""

    z =  abs(stats.norm.ppf((1-conf_level)/2))
    if x2==0 and x1 == 0:
        ul = inf
        ll = 0
    else:
        p1hat = x1/n1
        p2hat = x2/n2

        r = p1hat/p2hat

        gdot = array([1/p1hat, 1/p2hat])

        covmat = array([[p1hat*(1-p1hat), 0],
                        [0, p2hat*(1-p2hat)]])
        
        varg = (1/n) * dot(dot(gdot,covmat),gdot)

        ul = r + z*sqrt(varg)
        ll = r - z*sqrt(varg)

    return exp(array([ll, ul]))