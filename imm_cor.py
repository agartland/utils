from __future__ import division
import pandas as pd
import numpy as np
from lifelines.estimation import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from myfisher import *

__all__ = ['estTECI',
            'estTEPH']

def estTECI(df, followup, treatment_col='treated', duration_col='dx', event_col='disease', alpha=0.05):
    """Estimates treatment efficacy using cumulative incidence (CI) Nelson-Aalen (NA) estimators.

    TE = 1 - RR
    
    RR = CI_NA1 / CI_NA2

    Confidence 
    
    Parameters
    ----------
    df : pandas.DataFrame
        Each row is a participant.
    followup : float
        Follow-up time tau, at which CI will be estimated.
    treatment_col : string
        Column in df indicating treatment (values 1 or 0).
    dx_col : string
        Column in df indicating time of the event or censoring.
    event_col : string
        Column in df indicating events (values 1 or 0 with censored data as 0).
    
    Returns
    -------
    est : float
        Estimate of treatment efficacy
    ci : vector, length 2
        (1-alpha)%  confidence interval, [LL, UL]
    pvalue : float
        P-value for H0: TE=0 from Wald statistic"""
    
    ind1 = df[treatment_col] == 0
    naf1 = NelsonAalenFitter()
    naf1.fit(durations=df.loc[ind1, duration_col], event_observed=df.loc[ind1, event_col])

    ind2 = df[treatment_col] == 1
    naf2 = NelsonAalenFitter()
    naf2.fit(durations=df.loc[ind2, duration_col], event_observed=df.loc[ind2, event_col])

    te = 1 - naf1._cumulative_hazard / naf2._cumulative_hazard
    ci = np.array([np.nan, np.nan])
    pvalue = np.nan
    
    return te, ci, pvalue

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
    """Wilson's confidence interval for a single proportion.
    Score CI based on inverting the asymptotic normal test
    using the null standard error

    Wilson, E.B. (1927) Probable inference, the law of succession, and statistical inference
    J. Amer. Stat. Assoc 22, 209-212

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

def unconditionalVE(nv,Nv, np, Np, alpha=0.025):
    """VE point-estimate, CI and p-value, without conditioning on the total number of events"""
    rr = (nv/(Nv)) / (np/(Np))

    ve = 1 - rr
    se = sqrt((Nv-nv)/(nv*Nv) + (Np-np)/(np*Np))

    z = stats.norm.ppf(1 - alpha)

    ci = 1 - exp(array([log(rr) + se*z, log(rr) - se*z]))
    
    """Wald CI"""
    pvalue = stats.norm.cdf(log(rr)/se)

    return  pd.Series([ve, ci[0], ci[1], pvalue], index=['VE','LL', 'UL','p'])

def AgrestiScoreVE(nv,Nv, np, Np, alpha=0.025):
    """Conditional test based on a fixed number of events,
    n = nv + np
    phat = nv/n"""
    def scoreCIbinProp(pHat, n, alpha):
        """Score confidence interval for binomial proportion following Agresti and Coull (Am Statistician, 1998)"""
        z = stats.norm.ppf(1-alpha)
        return ((pHat + z**2/(2*n) + z*sqrt((pHat*(1-pHat)+z**2/(4*n))/n))/(1+z**2/n),
                (pHat + z**2/(2*n) - z*sqrt((pHat*(1-pHat)+z**2/(4*n))/n))/(1+z**2/n))

    veFunc = lambda pvhat, Nv, Np: 1 - (Np/Nv)*(pvhat/(1-pvhat))
    rr = (nv/Nv)/(np/Np)
    ve = 1 - rr
    n = nv + np
    pvhat = nv/n

    ll,ul = scoreCIbinProp(pvhat, n, alpha=alpha)
    
    
    ve = veFunc(pvhat, Nv, Np)
    ci = veFunc(ll, Nv, Np), veFunc(ul, Nv, Np)
    p = stats.binom.cdf(nv,n,Nv/(Nv+Np))
    
    return  pd.Series([ve, ci[0], ci[1], p], index=['VE','LL', 'UL','p'])